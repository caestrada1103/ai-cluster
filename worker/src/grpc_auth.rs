//! Shared-secret gRPC authentication (C1).
//!
//! The worker's gRPC service has no mTLS (out of scope per the security
//! audit — a shared secret plus a loopback-by-default bind is the agreed
//! deliverable). When [`crate::config::WorkerConfig::grpc_auth_token`] /
//! `WORKER_GRPC_AUTH_TOKEN` is configured, every RPC (except the standard
//! gRPC health-check service, which orchestrators/load balancers need to
//! reach without a credential) must present a matching `x-worker-token`
//! metadata value. When no token is configured the server stays OPEN —
//! existing single-host deployments where `grpc_bind_host` stays loopback
//! keep working unchanged; this is the "gated on config" requirement.

use tonic::metadata::MetadataMap;
use tonic::service::Interceptor;
use tonic::{Request, Status};

/// gRPC metadata key clients must set to the shared secret.
pub const TOKEN_METADATA_KEY: &str = "x-worker-token";

/// Constant-time byte comparison — avoids leaking the secret's length-prefix
/// timing signal a naive `==` would give an attacker probing over the
/// network. Still short-circuits on length mismatch (the length itself is
/// not the secret) but never on content, so equal-length guesses all take
/// the same time regardless of how many leading bytes match.
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Tonic server interceptor enforcing the shared-secret token.
///
/// Cheap to clone (one `Option<String>`) — tonic clones the interceptor per
/// connection/request in some configurations.
#[derive(Clone)]
pub struct TokenInterceptor {
    /// `None` disables auth entirely (open server — only safe on a loopback
    /// bind or another trust boundary in front of it).
    token: Option<String>,
}

impl TokenInterceptor {
    /// Build an interceptor. `token` is the effective, already-resolved
    /// secret (env `WORKER_GRPC_AUTH_TOKEN` beats `worker.toml`'s
    /// `grpc_auth_token` — resolved by the caller, mirroring `HF_TOKEN`).
    pub fn new(token: Option<String>) -> Self {
        Self {
            token: token.filter(|t| !t.is_empty()),
        }
    }

    // `tonic::Status` is the standard error type for every gRPC handler in
    // this crate (see worker.rs) — boxing it here alone would be
    // inconsistent, not smaller in practice.
    #[allow(clippy::result_large_err)]
    fn check(&self, metadata: &MetadataMap) -> Result<(), Status> {
        let Some(expected) = &self.token else {
            return Ok(());
        };
        let presented = metadata
            .get(TOKEN_METADATA_KEY)
            .and_then(|v| v.to_str().ok());
        match presented {
            Some(presented) if constant_time_eq(presented.as_bytes(), expected.as_bytes()) => {
                Ok(())
            }
            _ => Err(Status::unauthenticated(format!(
                "missing or invalid '{TOKEN_METADATA_KEY}' metadata"
            ))),
        }
    }
}

impl Interceptor for TokenInterceptor {
    fn call(&mut self, request: Request<()>) -> Result<Request<()>, Status> {
        self.check(request.metadata())?;
        Ok(request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_time_eq_matches_and_mismatches() {
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secret", b"secre1"));
        assert!(!constant_time_eq(b"secret", b"shorter"));
        assert!(!constant_time_eq(b"", b"x"));
        assert!(constant_time_eq(b"", b""));
    }

    fn req_with(header: Option<&str>) -> Request<()> {
        let mut req = Request::new(());
        if let Some(v) = header {
            req.metadata_mut().insert(
                TOKEN_METADATA_KEY,
                tonic::metadata::MetadataValue::try_from(v).unwrap(),
            );
        }
        req
    }

    #[test]
    fn no_configured_token_allows_everything() {
        let mut interceptor = TokenInterceptor::new(None);
        assert!(interceptor.call(req_with(None)).is_ok());
        assert!(interceptor.call(req_with(Some("anything"))).is_ok());
    }

    #[test]
    fn empty_configured_token_is_treated_as_disabled() {
        let mut interceptor = TokenInterceptor::new(Some(String::new()));
        assert!(interceptor.call(req_with(None)).is_ok());
    }

    #[test]
    fn configured_token_requires_exact_match() {
        let mut interceptor = TokenInterceptor::new(Some("s3cret".to_string()));
        assert!(interceptor.call(req_with(Some("s3cret"))).is_ok());
        assert!(interceptor.call(req_with(Some("wrong"))).is_err());
        assert!(interceptor.call(req_with(None)).is_err());
    }
}
