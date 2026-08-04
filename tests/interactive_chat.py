"""Interactive chat straight against a worker's gRPC endpoint (no coordinator).

Unlike `cluster_chat.py` (which POSTs to the coordinator's HTTP API), this
script IS the coordinator for one model: nothing else fills in the
`LoadModelRequest` for it. So it mirrors `ClusterCoordinator._load_model_on_worker`
— registry lookup, `ModelConfig.metadata` engine-routing keys, GPU ids — because
without that metadata the worker cannot route a GGUF model to its llama.cpp
engine and falls through to the Burn/safetensors path instead.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import grpc

REPO_ROOT = Path(__file__).resolve().parents[1]
# `python tests/interactive_chat.py` (the invocation CONTRIBUTING.md documents)
# puts tests/ on sys.path, not the repo root, so `import coordinator` needs help.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import coordinator.proto.cluster_pb2 as cluster_pb2  # noqa: E402
import coordinator.proto.cluster_pb2_grpc as cluster_pb2_grpc  # noqa: E402
from coordinator.config import Settings  # noqa: E402
from coordinator.models import ModelConfig, ModelRegistry  # noqa: E402

# Smallest GGUF in the registry (~0.4 GB, engine = "llamacpp"): the cheapest
# way to prove the worker's llama.cpp path end to end before spending a 22 GB
# download on the DGX Spark tier. Same engine, so a pass here means the only
# variable left for `--model qwen3.6-35b-a3b-gguf` is size.
DEFAULT_MODEL = "qwen2.5-0.5b-gguf"

# Prompt templates. The in-process llama.cpp engine does NOT apply the GGUF's
# chat template — the prompt is tokenized verbatim — so the client must wrap it
# or the model free-associates instead of answering.
TEMPLATES = {
    "chatml": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
    "tinyllama": "<|user|>\n{prompt}</s>\n<|assistant|>\n",
    "mistral": "[INST] {prompt} [/INST]",
    "raw": "{prompt}",
}


def pick_template(model_name: str, config: Optional[ModelConfig]) -> str:
    """Guess a chat template from the model name/family."""
    haystack = f"{model_name} {config.family.value if config else ''}".lower()
    if "tinyllama" in haystack:
        return "tinyllama"
    if "qwen" in haystack or "deepseek" in haystack:
        return "chatml"
    if "mistral" in haystack or "devstral" in haystack:
        return "mistral"
    return "raw"


def format_stats(
    sent_at: float,
    first_token_at: Optional[float],
    last_token_at: float,
    tokens: int,
    server_ms: int,
) -> str:
    """One-line throughput summary for a completed response.

    Two rates, because they answer different questions:

    * `decode` — tokens after the first, over the first-to-last token window.
      This is the steady-state generation speed, prompt eval excluded.
    * `end-to-end` — every token over the full round trip, so prefill, gRPC,
      and this client's own print overhead all count against it.

    `tokens` is the worker's count of streamed chunks (one per sampled token).
    It undercounts slightly on multi-byte output — the UTF-8 decoder holds back
    partial sequences, and empty pieces are never sent (llamacpp_engine.rs).
    """
    total_s = last_token_at - sent_at
    ttft_ms = (first_token_at - sent_at) * 1000 if first_token_at else 0.0
    decode_s = last_token_at - first_token_at if first_token_at else 0.0

    # The first token's cost is prefill, not decode — exclude both it and its
    # latency, or short responses read as artificially slow.
    decode_tps = (tokens - 1) / decode_s if decode_s > 0 and tokens > 1 else 0.0
    e2e_tps = tokens / total_s if total_s > 0 else 0.0

    return (
        f"  [{tokens} tokens | {decode_tps:.1f} tok/s decode | "
        f"{e2e_tps:.1f} tok/s end-to-end | TTFT {ttft_ms:.0f} ms | "
        f"worker {server_ms} ms]"
    )


def load_registry(models_config: Optional[str]) -> Settings:
    """Populate ModelRegistry from models.toml, returning the Settings used.

    `Settings.models_config` is repo-root-relative, so it only resolves when the
    script is run from the repo root; fall back to a path derived from __file__
    so `python tests/interactive_chat.py` works from anywhere.
    """
    if models_config:
        settings = Settings(models_config=Path(models_config))
    else:
        settings = Settings()
        if not settings.models_config.exists():
            settings = Settings(models_config=REPO_ROOT / "config" / "models.toml")

    config_dict = settings.load_models_config()
    if config_dict:
        ModelRegistry.load_from_dict(config_dict)
    else:
        print(f"Warning: no models config at {settings.models_config}")
    return settings


def build_load_request(
    model_name: str,
    config: Optional[ModelConfig],
    worker_gpus: list,
    quantization: int,
) -> cluster_pb2.LoadModelRequest:
    """Mirror of ClusterCoordinator._load_model_on_worker's request assembly."""
    if config:
        config_pb = cluster_pb2.ModelConfig(
            architecture=config.family.value,
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_kv_heads or 0,
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_seq_len,
            intermediate_size=config.intermediate_size,
            # The engine-routing keys ("engine", "gguf_repo_id", "gguf_file",
            # "n_ctx", "cache_type_k/v", ...). Empty for Burn models.
            metadata=config.grpc_metadata(),
        )
        if config.local_gpu_ids is not None:
            gpu_ids = list(config.local_gpu_ids)
        else:
            gpu_ids = [g.id for g in worker_gpus[: config.recommended_gpus]]
            if not gpu_ids and worker_gpus:
                gpu_ids = [worker_gpus[0].id]
        model_path = config.hf_repo_id or ""
    else:
        # Unknown model: treat model_name as a HF repo id, let the worker pull
        # config.json and overwrite these placeholders.
        config_pb = cluster_pb2.ModelConfig(architecture="llama")
        gpu_ids = [g.id for g in worker_gpus]
        model_path = ""

    return cluster_pb2.LoadModelRequest(
        model_name=model_name,
        model_path=model_path,
        config=config_pb,
        gpu_ids=gpu_ids,
        quantization=quantization,
        parallelism=cluster_pb2.ParallelismStrategy.AUTO,
    )


def chat() -> int:
    parser = argparse.ArgumentParser(description="Interactive chat with AI Worker")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature (default: 0.7)")
    parser.add_argument("--host", type=str, default="localhost:50051", help="Worker gRPC host")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Registry key (default: {DEFAULT_MODEL}; DGX Spark tier: qwen3.6-35b-a3b-gguf)",
    )
    parser.add_argument("--models-config", type=str, default=None, help="Path to models.toml")
    parser.add_argument(
        "--quant",
        type=str,
        default="none",
        choices=["none", "fp16", "int8", "int4"],
        help="Quantization. Only 'none' is implemented — the worker rejects the "
        "others outright, and a GGUF carries its own quantization internally",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="auto",
        choices=["auto", *TEMPLATES],
        help="Chat template to wrap prompts in (default: auto, from the model name)",
    )
    parser.add_argument(
        "--load-timeout",
        type=int,
        default=None,
        help="LoadModel deadline in seconds. Covers the worker's GGUF download "
        "(a 22 GB model over a 10 MB/s link needs ~37 min). Defaults to the "
        "coordinator's COORDINATOR_MODEL_LOAD_TIMEOUT.",
    )
    parser.add_argument("--skip-load", action="store_true", help="Assume the model is loaded")
    parser.add_argument("--no-stats", action="store_true", help="Hide the per-response tok/s line")
    args = parser.parse_args()

    quant_map = {
        "none": cluster_pb2.Quantization.NONE,
        "fp16": cluster_pb2.Quantization.FP16,
        "int8": cluster_pb2.Quantization.INT8,
        "int4": cluster_pb2.Quantization.INT4,
    }
    quantization = quant_map[args.quant]

    settings = load_registry(args.models_config)
    load_timeout = args.load_timeout or settings.model_load_timeout

    model_config = ModelRegistry.get_model(args.model)
    if model_config is None:
        print(f"'{args.model}' is not in the registry — treating it as a HuggingFace repo id.")
        print(f"Known models: {', '.join(sorted(ModelRegistry.list_models()))}")

    template = args.template if args.template != "auto" else pick_template(args.model, model_config)
    if template == "raw":
        print(f"Warning: no chat template matched '{args.model}'; sending prompts verbatim.")

    channel = grpc.insecure_channel(args.host)
    stub = cluster_pb2_grpc.WorkerStub(channel)

    print(f"Connecting to worker at {args.host}...")
    try:
        response = stub.HealthCheck(cluster_pb2.Empty(), timeout=10)
        if response.status != cluster_pb2.HealthCheckResponse.SERVING:
            print("Worker is not serving!")
            return 1
        status = stub.GetStatus(cluster_pb2.Empty(), timeout=10)
    except grpc.RpcError as e:
        print(f"Could not connect to worker: {e.details()}")
        return 1

    print(f"Worker {status.worker_id} is ready — {len(status.gpus)} GPU(s):")
    for gpu in status.gpus:
        print(
            f" - [{gpu.id}] {gpu.name}: "
            f"{gpu.available_memory / 1e9:.1f} / {gpu.total_memory / 1e9:.1f} GB free"
        )

    if args.skip_load:
        print(f"Skipping load; assuming {args.model} is already loaded.")
    else:
        engine = model_config.engine if model_config else "unknown"
        print(f"Loading {args.model} (engine={engine}, quantization={args.quant})...")
        print(f"First load downloads the weights — deadline is {load_timeout}s.")
        request = build_load_request(args.model, model_config, list(status.gpus), quantization)
        started = time.monotonic()
        try:
            load_response = stub.LoadModel(request, timeout=load_timeout)
        except grpc.RpcError as e:
            # Abort: continuing would make every Infer below fail identically.
            print(f"Load failed after {time.monotonic() - started:.0f}s: {e.details()}")
            return 1
        if not load_response.success:
            print(f"Load failed: {load_response.message}")
            return 1
        print(
            f"Loaded in {time.monotonic() - started:.0f}s, "
            f"{load_response.memory_used / 1e9:.2f} GB on GPU(s) "
            f"{list(load_response.loaded_on_gpus)}"
        )

    print("\n" + "=" * 50)
    print(f"Chatting with {args.model}")
    print(f"Params: max_tokens={args.max_tokens}, temp={args.temp}, template={template}")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 50 + "\n")

    while True:
        try:
            prompt = input("You: ")
            if prompt.lower() in ["quit", "exit"]:
                break

            if not prompt.strip():
                continue

            print("Model: ", end="", flush=True)

            full_prompt = TEMPLATES[template].format(prompt=prompt)

            request = cluster_pb2.InferenceRequest(
                model_name=args.model,
                prompt=full_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temp,
                stream=True,
            )

            sent_at = time.monotonic()
            first_token_at = None
            last_token_at = sent_at
            tokens = 0
            server_ms = 0

            for response in stub.Infer(request):
                if response.text:
                    now = time.monotonic()
                    if first_token_at is None:
                        first_token_at = now
                    last_token_at = now
                    print(response.text, end="", flush=True)
                # Monotonically increasing; the final (finished) message carries
                # the total, so just keep the latest.
                tokens = response.tokens_generated or tokens
                server_ms = response.processing_time_ms or server_ms

            print()
            if not args.no_stats and tokens:
                print(format_stats(sent_at, first_token_at, last_token_at, tokens, server_ms))
            print()

        except KeyboardInterrupt:
            break
        except grpc.RpcError as e:
            print(f"\nRPC Error: {e.details()}")

    return 0


if __name__ == "__main__":
    sys.exit(chat())
