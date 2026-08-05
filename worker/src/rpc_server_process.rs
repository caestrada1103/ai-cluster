//! `ggml-rpc-server` process supervision (`distributed_role = "rpc_server"`).
//!
//! Spawns and supervises the `ggml-rpc-server` child(ren) that lend this
//! node's GPU(s) to a `llama-server --rpc` lead running elsewhere. Mirrors
//! `llamaserver_process.rs`'s spawn/health/kill shape; health here is a raw
//! TCP connect probe since ggml-RPC has no HTTP surface. See docs/configuration.md.
//!
//! Feature-gated behind `llamacpp-rpc` — see `worker/Cargo.toml`.

use std::time::Duration;

use tokio::net::TcpStream;
use tokio::process::{Child, Command};
use tokio::time::{sleep, Instant};
use tracing::{debug, info, warn};

use crate::error::WorkerError;

/// Build the exact `ggml-rpc-server` argv (excluding the binary itself).
/// `device` is the backend-specific `-d` value (e.g. `"CUDA0"`, `"Vulkan0"`);
/// omitted when `None`, letting `ggml-rpc-server` pick its own default device.
pub fn build_rpc_server_args(bind_host: &str, port: u16, device: Option<&str>) -> Vec<String> {
    let mut args = vec![
        "-H".to_string(),
        bind_host.to_string(),
        "-p".to_string(),
        port.to_string(),
    ];
    if let Some(device) = device {
        args.push("-d".to_string());
        args.push(device.to_string());
    }
    args
}

/// A supervised `ggml-rpc-server` child process. Spawning does not block on
/// readiness — call [`Self::wait_until_healthy`] afterwards. `kill_on_drop`
/// is a backstop so the child never outlives the worker.
pub struct RpcServerProcess {
    /// Registry key of the model this process lends GPU memory to (for logs/errors).
    model_name: String,
    /// Interface `ggml-rpc-server` was told to bind (`-H`).
    bind_host: String,
    /// Port `ggml-rpc-server` was told to listen on (`-p`).
    port: u16,
    /// The child handle. `try_wait`/`start_kill` need `&mut`, so callers reach
    /// this through the outer `tokio::sync::Mutex` the loader stores it behind.
    child: Child,
}

impl RpcServerProcess {
    /// Spawn `binary args...` WITHOUT waiting for the port to accept connections.
    ///
    /// `binary` is resolved via `PATH` unless it is an absolute path.
    pub fn spawn(
        model_name: &str,
        bind_host: &str,
        port: u16,
        binary: &str,
        args: &[String],
    ) -> Result<Self, WorkerError> {
        info!(
            "spawning ggml-rpc-server for '{}': {} {}",
            model_name,
            binary,
            args.join(" ")
        );
        let child = Command::new(binary)
            .args(args)
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| {
                WorkerError::ModelLoad(format!(
                    "failed to spawn ggml-rpc-server binary '{binary}' for model '{model_name}': \
                     {e} (install llama.cpp's ggml-rpc-server on PATH or set RPC_SERVER_BINARY_PATH)"
                ))
            })?;
        Ok(Self {
            model_name: model_name.to_string(),
            bind_host: bind_host.to_string(),
            port,
            child,
        })
    }

    /// `true` while the child is still running. Reaps the child on exit
    /// (non-blocking), so a self-exited process is detected here and never
    /// lingers as a zombie.
    pub fn is_running(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    /// Poll a raw TCP connect to `bind_host:port` until it succeeds, the
    /// child exits, or `timeout` elapses. ggml-RPC has no HTTP surface (and
    /// no handshake worth speaking here), so a bare connect is the health
    /// signal — mirrors `LlamaServerProcess::wait_until_healthy`'s shape.
    pub async fn wait_until_healthy(&mut self, timeout: Duration) -> Result<(), WorkerError> {
        let deadline = Instant::now() + timeout;
        loop {
            // Fail fast: if the child already exited it will never be healthy.
            if let Ok(Some(status)) = self.child.try_wait() {
                return Err(WorkerError::ModelLoad(format!(
                    "ggml-rpc-server for '{}' exited during startup ({status})",
                    self.model_name
                )));
            }
            if TcpStream::connect((self.bind_host.as_str(), self.port))
                .await
                .is_ok()
            {
                info!(
                    "ggml-rpc-server for '{}' healthy on {}:{}",
                    self.model_name, self.bind_host, self.port
                );
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(WorkerError::ModelLoad(format!(
                    "ggml-rpc-server for '{}' did not become healthy within {:?} ({}:{})",
                    self.model_name, timeout, self.bind_host, self.port
                )));
            }
            sleep(Duration::from_millis(250)).await;
        }
    }

    /// Send the terminate signal without awaiting. Safe to call on an
    /// already-exited child.
    pub fn start_kill(&mut self) -> Result<(), WorkerError> {
        self.child.start_kill().map_err(WorkerError::Io)
    }

    /// Terminate the child and await its reaping. Used by graceful unload and by
    /// tests that need a deterministic "it is gone now" point.
    pub async fn shutdown(&mut self) -> Result<(), WorkerError> {
        if let Err(e) = self.child.start_kill() {
            // Already exited / already killed — nothing to do.
            debug!("start_kill for '{}': {}", self.model_name, e);
        }
        if let Err(e) = self.child.wait().await {
            warn!("wait after kill for '{}': {}", self.model_name, e);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;

    // --- argv building -------------------------------------------------

    #[test]
    fn build_args_with_device() {
        let args = build_rpc_server_args("10.100.88.2", 50052, Some("CUDA0"));
        assert_eq!(
            args,
            vec!["-H", "10.100.88.2", "-p", "50052", "-d", "CUDA0"]
        );
    }

    #[test]
    fn build_args_without_device_omits_flag() {
        let args = build_rpc_server_args("10.100.88.2", 50052, None);
        assert_eq!(args, vec!["-H", "10.100.88.2", "-p", "50052"]);
    }

    // --- supervision (no real ggml-rpc-server binary / GPU) -------------

    #[tokio::test]
    async fn spawn_reports_running_then_kill_on_unload_stops_it() {
        // A harmless long-running stand-in for ggml-rpc-server.
        let mut proc = RpcServerProcess::spawn(
            "test-model",
            "127.0.0.1",
            23456,
            "sleep",
            &["30".to_string()],
        )
        .expect("`sleep` should spawn on Linux");
        assert!(
            proc.is_running(),
            "child should be running right after spawn"
        );

        proc.shutdown().await.unwrap();
        assert!(!proc.is_running(), "child must be gone after shutdown()");
    }

    #[tokio::test]
    async fn exited_child_is_detected() {
        // `true` exits 0 immediately — stands in for a ggml-rpc-server that
        // dies on its own (bad flags, missing device, port clash).
        let mut proc = RpcServerProcess::spawn("test-model", "127.0.0.1", 23457, "true", &[])
            .expect("`true` should spawn on Linux");
        let mut gone = false;
        for _ in 0..100 {
            if !proc.is_running() {
                gone = true;
                break;
            }
            sleep(Duration::from_millis(20)).await;
        }
        assert!(gone, "a self-exited child must report not-running");
    }

    #[tokio::test]
    async fn wait_until_healthy_fails_fast_when_child_exits_during_startup() {
        let mut proc = RpcServerProcess::spawn("test-model", "127.0.0.1", 23458, "true", &[])
            .expect("`true` should spawn on Linux");
        // Give `true` a moment to actually exit before polling.
        sleep(Duration::from_millis(50)).await;
        let err = proc
            .wait_until_healthy(Duration::from_secs(2))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("exited during startup"));
    }

    #[tokio::test]
    async fn wait_until_healthy_times_out_when_nothing_listens() {
        // "sleep" never opens the port, and nothing else is bound to it, so
        // the connect probe must exhaust the timeout and fail cleanly.
        let mut proc = RpcServerProcess::spawn(
            "test-model",
            "127.0.0.1",
            23459,
            "sleep",
            &["30".to_string()],
        )
        .expect("`sleep` should spawn on Linux");
        let err = proc
            .wait_until_healthy(Duration::from_millis(300))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("did not become healthy"));
        proc.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn wait_until_healthy_succeeds_once_the_port_accepts_connections() {
        // The health check is a bare TCP connect, so any listener on the
        // target port satisfies it — a real ggml-rpc-server isn't needed to
        // exercise the polling logic itself.
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        // Keep the listener alive for the duration of the probe.
        let _accept_task = tokio::spawn(async move {
            let _ = listener.accept().await;
        });

        let mut proc = RpcServerProcess::spawn(
            "test-model",
            "127.0.0.1",
            port,
            "sleep",
            &["30".to_string()],
        )
        .expect("`sleep` should spawn on Linux");
        proc.wait_until_healthy(Duration::from_secs(2))
            .await
            .expect("connect probe should succeed against the bound listener");
        proc.shutdown().await.unwrap();
    }
}
