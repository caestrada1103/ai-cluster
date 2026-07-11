"""Model registry and configuration.

AICluster's goal is running LLM inference on consumer GPUs (gaming-PC cards,
~8-16 GB VRAM each, including idle ones) across both NVIDIA and AMD hardware.
The llama.cpp/GGUF engine (``engine="llamacpp"``) is the recommended, primary
path for that: it natively quantizes models (Q4_K_M/Q5_K_M/Q8_0/...) so they
fit in limited consumer VRAM, and runs on NVIDIA (CUDA/Vulkan) and AMD
(ROCm/Vulkan). The Burn engine (``engine="burn"``, the dataclass default) is
the experimental/reference path: it loads weights as FP32 only (no
quantization) and runs a model on a single GPU per worker.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Allowed ggml KV-cache quantization type names for `gguf_cache_type_k` /
# `gguf_cache_type_v` (llama.cpp's `--cache-type-k`/`--cache-type-v` flags).
# `f16` is llama.cpp's own default (today's behavior); the rest trade VRAM for
# precision — see Plan 11 Task 2b / Appendix D.
_ALLOWED_KV_CACHE_TYPES = frozenset({"f16", "q8_0", "q4_0", "q5_0", "q5_1", "q4_1"})


class ModelFamily(str, Enum):
    """Supported model families."""

    DEEPSEEK = "deepseek"
    LLAMA = "llama"
    MISTRAL = "mistral"
    PHI = "phi"
    GEMMA = "gemma"
    QWEN = "qwen"


class Quantization(str, Enum):
    """Supported quantization types."""

    NONE = "none"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    FP8 = "fp8"


class ParallelismStrategy(str, Enum):
    """Supported parallelism strategies."""

    AUTO = "auto"
    SINGLE = "single"
    PIPELINE = "pipeline"
    TENSOR = "tensor"
    DATA = "data"
    EXPERT = "expert"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    # Basic info
    name: str
    family: ModelFamily
    parameters: str  # e.g., "7B", "67B", "8B"

    # Resource requirements
    min_memory_gb: float  # Minimum VRAM required per GPU
    recommended_gpus: int  # Recommended number of GPUs
    max_gpus: int  # Maximum GPUs that can be used

    # Model architecture
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    vocab_size: int
    max_seq_len: int
    intermediate_size: int
    num_kv_heads: Optional[int] = None  # For GQA/MQA

    # Features
    supports_quantization: List[Quantization] = field(default_factory=list)
    supports_parallelism: List[ParallelismStrategy] = field(default_factory=list)
    is_moe: bool = False  # Mixture of Experts
    num_experts: Optional[int] = None  # For MoE models

    # File paths
    config_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    weights_path: Optional[str] = None

    # HuggingFace repo id used by workers to download weights (LoadModelRequest.model_path)
    hf_repo_id: Optional[str] = None

    # Metadata
    description: str = ""
    paper_url: Optional[str] = None
    model_url: Optional[str] = None

    # Inference engine selection ("burn" = default Burn/safetensors path,
    # "llamacpp" = GGUF served by the worker's llama.cpp engine)
    engine: str = "burn"
    gguf_repo_id: Optional[str] = None  # HF repo containing the GGUF file
    gguf_file: Optional[str] = None  # exact .gguf filename inside the repo
    gguf_n_gpu_layers: Optional[int] = None  # -1 = all layers on GPU
    gguf_n_ctx: Optional[int] = None  # context window override

    # KV-cache quantization (Plan 11 Task 2b / Appendix D): halves (q8_0) or
    # quarters (q4_0) the KV-cache VRAM footprint vs today's fp16 KV, which is
    # the biggest lever for fitting large `n_ctx` in limited VRAM. Both None
    # (the default) is today's behavior — llama.cpp's own fp16 KV cache.
    gguf_cache_type_k: Optional[str] = None
    gguf_cache_type_v: Optional[str] = None

    # llama-server engine (Plan 13 — agentic serving). The worker supervises a
    # `llama-server` child process per model; the coordinator proxies raw
    # OpenAI/Anthropic JSON+SSE straight to it (see coordinator/proxy.py), so
    # `tools`, streaming `tool_calls` and Anthropic `/v1/messages` flow through
    # unmodified. These models reuse the SAME gguf.* source keys as the
    # in-process `llamacpp` engine (the child still loads a GGUF) plus the
    # llamaserver.* runtime knobs below.
    llamaserver_port: Optional[int] = None  # REQUIRED for engine=="llamaserver"; unique per model
    llamaserver_parallel: Optional[int] = None  # llama-server --parallel (default 4 when emitted)
    llamaserver_extra_args: Optional[str] = None  # extra CLI args, whitespace-split by the worker

    # Level 1 — local multi-GPU split (llama.cpp splitting a model across a
    # single node's OWN same-vendor GPUs, in-process, no network). Both None
    # (the default) is today's behavior: the coordinator picks
    # `worker.gpus[:recommended_gpus]` and no split metadata is sent.
    local_gpu_ids: Optional[List[int]] = None
    local_tensor_split: Optional[List[float]] = None  # weights, same order/len as local_gpu_ids

    # Level 2 — distributed (cross-node ggml-RPC) registry schema. A model
    # splits pipeline layers across a "lead" node (owns the real llama.cpp
    # context) and one or more "rpc_server" peer nodes that lend local GPUs.
    # `distributed=False` (the default) is today's single-node behavior.
    distributed: bool = False
    distributed_lead: Optional[str] = None  # worker_id of the lead node
    distributed_peers: List[str] = field(default_factory=list)  # worker_ids of rpc_server peers
    distributed_split: Optional[List[float]] = None  # None = auto-derive at load time (Task 6)
    distributed_rpc_port: int = 50151  # base ggml-RPC port each peer binds from
    distributed_gpu_ids: Dict[str, List[int]] = field(
        default_factory=dict
    )  # worker_id -> local GPU ids that node contributes

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.is_moe and not self.num_experts:
            raise ValueError("MoE models must specify num_experts")

        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_attention_heads

        if self.engine not in ("burn", "llamacpp", "llamaserver"):
            raise ValueError(
                f"Unknown engine '{self.engine}' " "(expected 'burn', 'llamacpp', or 'llamaserver')"
            )

        # Both llama.cpp transports load a GGUF: the in-process 'llamacpp' engine
        # and the 'llamaserver' child process the coordinator proxies to. Both
        # therefore require the gguf.* source keys.
        if self.engine in ("llamacpp", "llamaserver") and not (
            self.gguf_repo_id and self.gguf_file
        ):
            raise ValueError(f"{self.engine} engine models must set gguf.repo_id and gguf.file")

        # llamaserver_port is coordinator-assigned and REQUIRED so the proxy
        # knows where the worker-local llama-server listens (Plan 13 contract).
        # Registry-wide uniqueness is enforced separately by
        # ModelRegistry.validate_llamaserver_ports() (a single config can't see
        # its peers).
        if self.engine == "llamaserver" and self.llamaserver_port is None:
            raise ValueError("llamaserver engine models must set llamaserver_port")

        if self.gguf_cache_type_k is not None and self.gguf_cache_type_k not in (
            _ALLOWED_KV_CACHE_TYPES
        ):
            raise ValueError(
                f"gguf_cache_type_k must be one of {sorted(_ALLOWED_KV_CACHE_TYPES)} "
                f"(got {self.gguf_cache_type_k!r})"
            )
        if self.gguf_cache_type_v is not None and self.gguf_cache_type_v not in (
            _ALLOWED_KV_CACHE_TYPES
        ):
            raise ValueError(
                f"gguf_cache_type_v must be one of {sorted(_ALLOWED_KV_CACHE_TYPES)} "
                f"(got {self.gguf_cache_type_v!r})"
            )

        if self.local_tensor_split is not None:
            if self.local_gpu_ids is None or len(self.local_tensor_split) != len(
                self.local_gpu_ids
            ):
                raise ValueError(
                    "local_tensor_split length must equal local_gpu_ids length "
                    f"(got {len(self.local_tensor_split)} weights vs "
                    f"{len(self.local_gpu_ids) if self.local_gpu_ids is not None else 0} gpu ids)"
                )
            if any(weight <= 0 for weight in self.local_tensor_split):
                raise ValueError("local_tensor_split values must all be > 0")

        if self.distributed:
            if self.engine != "llamacpp":
                raise ValueError("distributed models must use engine='llamacpp'")
            if not self.distributed_lead:
                raise ValueError("distributed models must set distributed_lead")
            if not self.distributed_peers:
                raise ValueError("distributed models must set at least one distributed peer")

    def _gguf_metadata(self) -> Dict[str, str]:
        """Base engine + GGUF-source metadata shared by every GGUF transport
        helper (in-process single-node, distributed lead, distributed
        rpc_server, and the llama-server child all need the same model source
        keys). ``engine`` carries ``self.engine`` verbatim so the worker can
        dispatch ``"llamacpp"`` (in-process) vs ``"llamaserver"`` (child
        process) off the same map."""
        metadata: Dict[str, str] = {
            "engine": self.engine,
            "gguf_repo_id": self.gguf_repo_id or "",
            "gguf_file": self.gguf_file or "",
        }
        if self.gguf_n_gpu_layers is not None:
            metadata["n_gpu_layers"] = str(self.gguf_n_gpu_layers)
        if self.gguf_n_ctx is not None:
            metadata["n_ctx"] = str(self.gguf_n_ctx)
        if self.gguf_cache_type_k is not None:
            metadata["cache_type_k"] = self.gguf_cache_type_k
        if self.gguf_cache_type_v is not None:
            metadata["cache_type_v"] = self.gguf_cache_type_v
        return metadata

    def _llamaserver_metadata(self) -> Dict[str, str]:
        """Metadata for a llamaserver-engine model (Plan 13 Task 2).

        Rides the shared engine/gguf.* source keys from `_gguf_metadata()` (the
        worker's llama-server child still loads a GGUF) plus the EXACT
        cross-language contract keys the worker parses to spawn/configure the
        process: `llamaserver.port`, `llamaserver.parallel`, and — only when set
        — `llamaserver.extra_args`. Parallel defaults to 4 when unset (maps to
        llama-server `--parallel`).
        """
        metadata = self._gguf_metadata()
        metadata["llamaserver.port"] = str(self.llamaserver_port)
        metadata["llamaserver.parallel"] = str(
            self.llamaserver_parallel if self.llamaserver_parallel is not None else 4
        )
        if self.llamaserver_extra_args:
            metadata["llamaserver.extra_args"] = self.llamaserver_extra_args
        return metadata

    def grpc_metadata(self) -> Dict[str, str]:
        """Engine-routing metadata carried in the gRPC ModelConfig.metadata map.

        The worker's model loader reads these string keys to route the model.
        Burn models send an empty map (zero proto change, fully backwards
        compatible). `llamacpp` models send the gguf.* source (plus a
        `tensor_split` key for Level-1 local multi-GPU splits). `llamaserver`
        models send the gguf.* source plus the `llamaserver.*` keys the worker
        uses to spawn the llama-server child (see `_llamaserver_metadata`).
        """
        if self.engine == "llamaserver":
            return self._llamaserver_metadata()
        if self.engine != "llamacpp":
            return {}
        metadata = self._gguf_metadata()
        if self.local_tensor_split is not None:
            metadata["tensor_split"] = ",".join(str(w) for w in self.local_tensor_split)
        return metadata

    def grpc_metadata_lead(
        self, peer_endpoints: List[str], tensor_split: Optional[List[float]]
    ) -> Dict[str, str]:
        """Metadata for the LEAD node of a Level-2 distributed load.

        Rides the same base engine/gguf keys as `grpc_metadata()` plus the
        distributed-role keys the worker parses: `distributed_role="lead"`,
        `rpc_peers` (ordered, comma-joined "host:port" — SAME order as the
        peer-portion of `tensor_split`), and `tensor_split` (the combined
        [lead gpu_ids..., peer_1 lent gpus..., ...] weights) when given.
        """
        metadata = self._gguf_metadata()
        metadata["distributed_role"] = "lead"
        metadata["rpc_peers"] = ",".join(peer_endpoints)
        if tensor_split is not None:
            metadata["tensor_split"] = ",".join(str(w) for w in tensor_split)
        return metadata

    def grpc_metadata_rpc_server(self, base_port: int) -> Dict[str, str]:
        """Metadata for an RPC_SERVER peer node of a Level-2 distributed load.

        Carries the same base engine/gguf keys (the peer needs the identical
        GGUF source to fetch/serve the same model) plus
        `distributed_role="rpc_server"` and `rpc_bind_port` — the base port; a
        node lending k GPUs binds `rpc_bind_port..+k-1`.
        """
        metadata = self._gguf_metadata()
        metadata["distributed_role"] = "rpc_server"
        metadata["rpc_bind_port"] = str(base_port)
        return metadata


class ModelRegistry:
    """Registry of all available models.

    Featured/recommended: the ``*-gguf`` entries below (``engine="llamacpp"``)
    are the primary path for AICluster's consumer-GPU goal — quantized models
    that fit in 8-16 GB of VRAM, on NVIDIA or AMD. The remaining entries use
    the Burn engine (``engine="burn"``, the dataclass default), an
    experimental/reference path that loads FP32 weights on a single GPU.
    """

    # Predefined model configurations
    MODELS: Dict[str, ModelConfig] = {}

    @classmethod
    def initialize(cls) -> None:
        """Initialize the model registry with default models."""
        # Note: In a production environment, this could be empty and
        # only populated via load_from_config.

        # GGUF / llama.cpp models (PRIMARY — quantized, consumer NVIDIA+AMD)
        # ====================================================================
        # Recommended path for AICluster's actual goal: running a quantized
        # model on consumer GPUs (~8-16 GB VRAM gaming cards, including idle
        # ones) on NVIDIA (CUDA/Vulkan) or AMD (ROCm/Vulkan). Served by the
        # worker's llama.cpp engine (worker built with --features llamacpp).
        # These entries mirror config/models.toml so the two sources never
        # diverge; architecture fields are unused for this engine (real
        # values come from GGUF metadata at load time), so they are zeroed.
        cls.MODELS["qwen2.5-0.5b-gguf"] = ModelConfig(
            name="qwen2.5-0.5b-gguf",
            family=ModelFamily.QWEN,
            parameters="0.5B",
            min_memory_gb=1,
            recommended_gpus=1,
            max_gpus=1,
            num_layers=0,
            hidden_size=0,
            num_attention_heads=0,
            vocab_size=0,
            max_seq_len=4096,
            intermediate_size=0,
            supports_quantization=[Quantization.NONE],
            supports_parallelism=[ParallelismStrategy.SINGLE],
            description=(
                "Qwen2.5 0.5B Instruct — Q4_K_M GGUF served by the llama.cpp engine "
                "(smoke-test model)"
            ),
            model_url="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            engine="llamacpp",
            gguf_repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            gguf_file="qwen2.5-0.5b-instruct-q4_k_m.gguf",
            gguf_n_gpu_layers=-1,
            gguf_n_ctx=4096,
        )

        cls.MODELS["qwen2.5-7b-instruct-gguf"] = ModelConfig(
            name="qwen2.5-7b-instruct-gguf",
            family=ModelFamily.QWEN,
            parameters="7B",
            min_memory_gb=6,
            recommended_gpus=1,
            max_gpus=1,
            num_layers=0,
            hidden_size=0,
            num_attention_heads=0,
            vocab_size=0,
            max_seq_len=8192,
            intermediate_size=0,
            supports_quantization=[Quantization.NONE],
            supports_parallelism=[ParallelismStrategy.SINGLE],
            description="Qwen2.5 7B Instruct — Q4_K_M GGUF, fits an 8 GB consumer GPU",
            model_url="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF",
            engine="llamacpp",
            gguf_repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
            gguf_file="qwen2.5-7b-instruct-q4_k_m.gguf",
            gguf_n_gpu_layers=-1,
            gguf_n_ctx=8192,
        )

        cls.MODELS["qwen2.5-coder-7b-gguf"] = ModelConfig(
            name="qwen2.5-coder-7b-gguf",
            family=ModelFamily.QWEN,
            parameters="7B",
            min_memory_gb=6,
            recommended_gpus=1,
            max_gpus=1,
            num_layers=0,
            hidden_size=0,
            num_attention_heads=0,
            vocab_size=0,
            max_seq_len=8192,
            intermediate_size=0,
            supports_quantization=[Quantization.NONE],
            supports_parallelism=[ParallelismStrategy.SINGLE],
            description="Qwen2.5-Coder 7B Instruct — Q4_K_M GGUF, fits an 8 GB consumer GPU",
            model_url="https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            engine="llamacpp",
            gguf_repo_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            gguf_file="qwen2.5-coder-7b-instruct-q4_k_m.gguf",
            gguf_n_gpu_layers=-1,
            gguf_n_ctx=8192,
        )

        cls.MODELS["llama3.1-8b-instruct-gguf"] = ModelConfig(
            name="llama3.1-8b-instruct-gguf",
            family=ModelFamily.LLAMA,
            parameters="8B",
            min_memory_gb=7,
            recommended_gpus=1,
            max_gpus=1,
            num_layers=0,
            hidden_size=0,
            num_attention_heads=0,
            vocab_size=0,
            max_seq_len=8192,
            intermediate_size=0,
            supports_quantization=[Quantization.NONE],
            supports_parallelism=[ParallelismStrategy.SINGLE],
            description="Meta Llama 3.1 8B Instruct — Q4_K_M GGUF, fits an 8 GB consumer GPU",
            model_url="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            engine="llamacpp",
            gguf_repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            gguf_file="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            gguf_n_gpu_layers=-1,
            gguf_n_ctx=8192,
        )

        cls.MODELS["qwen2.5-14b-instruct-gguf"] = ModelConfig(
            name="qwen2.5-14b-instruct-gguf",
            family=ModelFamily.QWEN,
            parameters="14B",
            min_memory_gb=12,
            recommended_gpus=1,
            max_gpus=1,
            num_layers=0,
            hidden_size=0,
            num_attention_heads=0,
            vocab_size=0,
            max_seq_len=8192,
            intermediate_size=0,
            supports_quantization=[Quantization.NONE],
            supports_parallelism=[ParallelismStrategy.SINGLE],
            description="Qwen2.5 14B Instruct — Q4_K_M GGUF, fits a 12-16 GB consumer GPU",
            model_url="https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF",
            engine="llamacpp",
            gguf_repo_id="Qwen/Qwen2.5-14B-Instruct-GGUF",
            gguf_file="qwen2.5-14b-instruct-q4_k_m.gguf",
            gguf_n_gpu_layers=-1,
            gguf_n_ctx=8192,
        )

        # Multi-GPU split via llama.cpp is the upcoming worker feature; this is
        # the "quantized model split across two consumer GPUs" showcase entry.
        cls.MODELS["qwen2.5-coder-32b-gguf"] = ModelConfig(
            name="qwen2.5-coder-32b-gguf",
            family=ModelFamily.QWEN,
            parameters="32B",
            min_memory_gb=20,
            recommended_gpus=2,
            max_gpus=4,
            num_layers=0,
            hidden_size=0,
            num_attention_heads=0,
            vocab_size=0,
            max_seq_len=8192,
            intermediate_size=0,
            supports_quantization=[Quantization.NONE],
            supports_parallelism=[ParallelismStrategy.SINGLE],
            description="Qwen2.5-Coder 32B Instruct — Q4_K_M GGUF, split across 2 consumer GPUs",
            model_url="https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
            engine="llamacpp",
            gguf_repo_id="Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
            gguf_file="qwen2.5-coder-32b-instruct-q4_k_m.gguf",
            gguf_n_gpu_layers=-1,
            gguf_n_ctx=8192,
        )

        # Burn models (experimental — FP32 reference engine, single GPU)
        # ====================================================================

        # DeepSeek models
        cls.MODELS["deepseek-7b"] = ModelConfig(
            name="deepseek-7b",
            family=ModelFamily.DEEPSEEK,
            parameters="7B",
            min_memory_gb=16,
            recommended_gpus=1,
            max_gpus=2,
            num_layers=30,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=32,
            vocab_size=102400,
            max_seq_len=4096,
            intermediate_size=11008,
            supports_quantization=[Quantization.NONE],
            supports_parallelism=[ParallelismStrategy.SINGLE, ParallelismStrategy.PIPELINE],
            description="DeepSeek LLM 7B Base (dense, Llama-style)",
            model_url="https://huggingface.co/deepseek-ai/deepseek-llm-7b-base",
            hf_repo_id="deepseek-ai/deepseek-llm-7b-base",
        )

        cls.MODELS["deepseek-67b"] = ModelConfig(
            name="deepseek-67b",
            family=ModelFamily.DEEPSEEK,
            parameters="67B",
            min_memory_gb=140,
            recommended_gpus=4,
            max_gpus=8,
            num_layers=95,
            hidden_size=8192,
            num_attention_heads=64,
            num_kv_heads=8,
            vocab_size=102400,
            max_seq_len=4096,
            intermediate_size=22016,
            supports_quantization=[Quantization.NONE],
            supports_parallelism=[
                ParallelismStrategy.PIPELINE,
                ParallelismStrategy.TENSOR,
            ],
            description="DeepSeek LLM 67B Base (dense, GQA)",
            model_url="https://huggingface.co/deepseek-ai/deepseek-llm-67b-base",
            hf_repo_id="deepseek-ai/deepseek-llm-67b-base",
        )

        # Llama 3 models
        cls.MODELS["llama3-8b"] = ModelConfig(
            name="llama3-8b",
            family=ModelFamily.LLAMA,
            parameters="8B",
            min_memory_gb=16,
            recommended_gpus=1,
            max_gpus=2,
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=8,  # GQA
            vocab_size=128256,
            max_seq_len=8192,
            intermediate_size=14336,
            supports_quantization=[Quantization.NONE],
            supports_parallelism=[
                ParallelismStrategy.SINGLE,
                ParallelismStrategy.PIPELINE,
                ParallelismStrategy.TENSOR,
            ],
            description="Meta Llama 3 8B Instruct",
            model_url="https://huggingface.co/meta-llama/Meta-Llama-3-8B",
            hf_repo_id="meta-llama/Meta-Llama-3-8B",
        )

        cls.MODELS["llama3-70b"] = ModelConfig(
            name="llama3-70b",
            family=ModelFamily.LLAMA,
            parameters="70B",
            min_memory_gb=140,
            recommended_gpus=4,
            max_gpus=8,
            num_layers=80,
            hidden_size=8192,
            num_attention_heads=64,
            num_kv_heads=8,  # GQA
            vocab_size=128256,
            max_seq_len=8192,
            intermediate_size=28672,
            supports_quantization=[Quantization.NONE],
            supports_parallelism=[
                ParallelismStrategy.PIPELINE,
                ParallelismStrategy.TENSOR,
            ],
            description="Meta Llama 3 70B Instruct",
            model_url="https://huggingface.co/meta-llama/Meta-Llama-3-70B",
            hf_repo_id="meta-llama/Meta-Llama-3-70B",
        )

        # Mistral models
        cls.MODELS["mistral-7b"] = ModelConfig(
            name="mistral-7b",
            family=ModelFamily.MISTRAL,
            parameters="7B",
            min_memory_gb=14,
            recommended_gpus=1,
            max_gpus=2,
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=8,  # GQA
            vocab_size=32000,
            max_seq_len=32768,  # Sliding window attention
            intermediate_size=14336,
            supports_quantization=[Quantization.NONE],
            supports_parallelism=[
                ParallelismStrategy.SINGLE,
                ParallelismStrategy.PIPELINE,
            ],
            description="Mistral 7B v0.1 (planned — worker cannot load Mistral yet)",
            model_url="https://huggingface.co/mistralai/Mistral-7B-v0.1",
            hf_repo_id="mistralai/Mistral-7B-v0.1",
        )

        # Phi models
        cls.MODELS["phi-2"] = ModelConfig(
            name="phi-2",
            family=ModelFamily.PHI,
            parameters="2.7B",
            min_memory_gb=6,
            recommended_gpus=1,
            max_gpus=1,
            num_layers=32,
            hidden_size=2560,
            num_attention_heads=32,
            vocab_size=51200,
            max_seq_len=2048,
            intermediate_size=10240,
            supports_quantization=[Quantization.NONE],
            supports_parallelism=[ParallelismStrategy.SINGLE],
            description="Microsoft Phi-2 (2.7B) (planned — worker cannot load Phi yet)",
            model_url="https://huggingface.co/microsoft/phi-2",
            hf_repo_id="microsoft/phi-2",
        )

        # Qwen3-Coder-32B
        cls.MODELS["qwen3-coder-32b"] = ModelConfig(
            name="qwen3-coder-32b",
            family=ModelFamily.QWEN,
            parameters="32B",
            min_memory_gb=65,
            recommended_gpus=2,
            max_gpus=4,
            num_layers=64,
            hidden_size=5120,
            num_attention_heads=40,
            num_kv_heads=8,
            vocab_size=152064,
            max_seq_len=32768,
            intermediate_size=27648,
            supports_quantization=[Quantization.NONE],
            supports_parallelism=[ParallelismStrategy.SINGLE, ParallelismStrategy.TENSOR],
            description="Qwen2.5-Coder 32B Instruct — strong open-source coding model",
            model_url="https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct",
            hf_repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        )

        # DeepSeek V3
        cls.MODELS["deepseek-v3"] = ModelConfig(
            name="deepseek-v3",
            family=ModelFamily.DEEPSEEK,
            parameters="671B",
            min_memory_gb=600,
            recommended_gpus=8,
            max_gpus=16,
            num_layers=61,
            hidden_size=7168,
            num_attention_heads=128,
            num_kv_heads=128,
            vocab_size=129280,
            max_seq_len=163840,
            intermediate_size=18432,
            is_moe=True,
            num_experts=256,
            supports_quantization=[Quantization.NONE],
            supports_parallelism=[
                ParallelismStrategy.SINGLE,
                ParallelismStrategy.TENSOR,
                ParallelismStrategy.PIPELINE,
            ],
            description="DeepSeek V3 — 671B MoE (37B active), Claude Opus tier (SWE-Bench 73.1%)",
            model_url="https://huggingface.co/deepseek-ai/DeepSeek-V3",
            hf_repo_id="deepseek-ai/DeepSeek-V3",
        )

        logger.info(f"Initialized model registry with {len(cls.MODELS)} models")

    @classmethod
    def load_from_dict(cls, config_dict: Dict[str, Any]) -> None:
        """Load models from a configuration dictionary (e.g. from models.toml)."""
        if "models" not in config_dict:
            return

        defaults = config_dict.get("defaults", {})

        def get_dict(data: Dict[str, Any], key: str) -> Dict[str, Any]:
            val = data.get(key, {})
            if isinstance(val, dict):
                return val
            return {"default": val}  # Fallback for flat values from defaults

        for name, data in config_dict["models"].items():
            try:
                # Merge with defaults
                # We need to be careful with nested dicts like 'architecture'
                # For now, we just do a shallow copy of defaults and update with model data
                final_data = defaults.copy()
                final_data.update(data)

                # Extract components with robustness for string vs dict
                arch = get_dict(final_data, "architecture")
                quants = get_dict(final_data, "quantization")
                parallel = get_dict(final_data, "parallelism")
                paths = get_dict(final_data, "paths")
                hf = get_dict(final_data, "hf")
                gguf = get_dict(final_data, "gguf")
                # llama-server knobs: accept either a nested `[models.X.llamaserver]`
                # table (port/parallel/extra_args, mirrors the dotted metadata
                # keys) or flat `llamaserver_port`/`llamaserver_parallel`/
                # `llamaserver_extra_args` keys under the model.
                llamaserver = get_dict(final_data, "llamaserver")
                ls_port = llamaserver.get("port", final_data.get("llamaserver_port"))
                ls_parallel = llamaserver.get("parallel", final_data.get("llamaserver_parallel"))
                ls_extra = llamaserver.get("extra_args", final_data.get("llamaserver_extra_args"))
                distributed_cfg = get_dict(final_data, "distributed")
                distributed_gpu_ids_raw = get_dict(distributed_cfg, "gpu_ids")

                # Determine supported quantizations
                supported_quants = quants.get("supported")
                if not supported_quants:
                    default_q = quants.get("default", "none")
                    supported_quants = [default_q]

                # Determine supported parallelism
                supported_parallel = parallel.get("supported")
                if not supported_parallel:
                    default_p = parallel.get("default", "auto")
                    supported_parallel = [default_p]

                model_name = final_data.get("name", name)
                raw_family = final_data.get("family", "llama")
                try:
                    model_family = ModelFamily(raw_family)
                except ValueError:
                    logger.warning(
                        f"Unknown model family '{raw_family}' for model {name}, defaulting to llama"
                    )
                    model_family = ModelFamily.LLAMA

                model = ModelConfig(
                    name=model_name,
                    family=model_family,
                    parameters=final_data.get("parameters", "Unknown"),
                    min_memory_gb=float(final_data.get("min_memory_gb", 8.0)),
                    recommended_gpus=int(final_data.get("recommended_gpus", 1)),
                    max_gpus=int(final_data.get("max_gpus", 1)),
                    num_layers=int(arch.get("num_layers", 0)),
                    hidden_size=int(arch.get("hidden_size", 0)),
                    num_attention_heads=int(arch.get("num_attention_heads", 0)),
                    vocab_size=int(arch.get("vocab_size", 32000)),
                    max_seq_len=int(arch.get("max_seq_len", 2048)),
                    intermediate_size=int(arch.get("intermediate_size", 0)),
                    num_kv_heads=arch.get("num_kv_heads"),
                    supports_quantization=[Quantization(q) for q in supported_quants],
                    supports_parallelism=[ParallelismStrategy(p) for p in supported_parallel],
                    is_moe=arch.get("is_moe", False),
                    num_experts=arch.get("num_experts"),
                    config_path=paths.get("config"),
                    tokenizer_path=paths.get("tokenizer"),
                    weights_path=paths.get("weights"),
                    hf_repo_id=hf.get("repo_id"),
                    description=final_data.get("description", ""),
                    engine=str(final_data.get("engine", "burn")),
                    gguf_repo_id=gguf.get("repo_id"),
                    gguf_file=gguf.get("file"),
                    gguf_n_gpu_layers=(
                        int(gguf["n_gpu_layers"]) if "n_gpu_layers" in gguf else None
                    ),
                    gguf_n_ctx=int(gguf["n_ctx"]) if "n_ctx" in gguf else None,
                    gguf_cache_type_k=gguf.get("cache_type_k"),
                    gguf_cache_type_v=gguf.get("cache_type_v"),
                    llamaserver_port=(int(ls_port) if ls_port is not None else None),
                    llamaserver_parallel=(int(ls_parallel) if ls_parallel is not None else None),
                    llamaserver_extra_args=(str(ls_extra) if ls_extra is not None else None),
                    local_gpu_ids=(
                        [int(x) for x in final_data["local_gpu_ids"]]
                        if "local_gpu_ids" in final_data
                        else None
                    ),
                    local_tensor_split=(
                        [float(x) for x in final_data["local_tensor_split"]]
                        if "local_tensor_split" in final_data
                        else None
                    ),
                    distributed=bool(distributed_cfg.get("enabled", False)),
                    distributed_lead=distributed_cfg.get("lead"),
                    distributed_peers=[str(p) for p in distributed_cfg.get("peers", [])],
                    distributed_split=(
                        [float(x) for x in distributed_cfg["split"]]
                        if isinstance(distributed_cfg.get("split"), list)
                        else None
                    ),
                    distributed_rpc_port=int(distributed_cfg.get("rpc_port", 50151)),
                    distributed_gpu_ids={
                        str(worker_id): [int(x) for x in gpu_ids]
                        for worker_id, gpu_ids in distributed_gpu_ids_raw.items()
                    },
                )
                cls.MODELS[model_name] = model
            except Exception as e:
                logger.error(f"Failed to load model {name} from config: {e}")

        # Fail loudly on a mis-configured registry: two llamaserver models
        # sharing a port would make the proxy route to the wrong process.
        cls.validate_llamaserver_ports()

        logger.info(f"Updated model registry. Total models: {len(cls.MODELS)}")

    @classmethod
    def validate_llamaserver_ports(cls, models: Optional[Dict[str, ModelConfig]] = None) -> None:
        """Assert every llamaserver-engine model has a unique `llamaserver_port`.

        A single `ModelConfig.__post_init__` can only check that its own port is
        set; registry-wide uniqueness (Plan 13 contract) has to be checked
        across all entries. Called at the end of `load_from_dict` so a
        mis-configured `config/models.toml` fails loudly at load time. Raises
        `ValueError` on the first duplicate. Operates on ``cls.MODELS`` unless an
        explicit ``models`` mapping is supplied (used by tests for isolation).
        """
        registry = cls.MODELS if models is None else models
        seen: Dict[int, str] = {}
        for cfg in registry.values():
            if cfg.engine != "llamaserver" or cfg.llamaserver_port is None:
                continue
            owner = seen.get(cfg.llamaserver_port)
            if owner is not None and owner != cfg.name:
                raise ValueError(
                    f"llamaserver_port {cfg.llamaserver_port} is assigned to both "
                    f"'{owner}' and '{cfg.name}' — ports must be unique per model"
                )
            seen[cfg.llamaserver_port] = cfg.name

    @classmethod
    def get_model(cls, name: str) -> Optional[ModelConfig]:
        """Get model configuration by name."""
        return cls.MODELS.get(name)

    @classmethod
    def list_models(cls) -> List[str]:
        """List all available model names."""
        return list(cls.MODELS.keys())

    @classmethod
    def find_models_by_family(cls, family: ModelFamily) -> List[ModelConfig]:
        """Find all models of a given family."""
        return [m for m in cls.MODELS.values() if m.family == family]

    @classmethod
    def validate_requirements(
        cls,
        model_name: str,
        available_memory: float,
        num_gpus: int,
        quantization: Quantization = Quantization.FP16,
    ) -> tuple[bool, str]:
        """Validate if model can run on available hardware."""
        model = cls.get_model(model_name)
        if not model:
            return False, f"Unknown model: {model_name}"

        # Adjust memory for quantization
        memory_multiplier = {
            Quantization.NONE: 1.0,
            Quantization.FP16: 0.5,
            Quantization.INT8: 0.25,
            Quantization.INT4: 0.125,
            Quantization.FP8: 0.25,
        }

        required_memory = model.min_memory_gb * memory_multiplier.get(quantization, 1.0)

        if available_memory < required_memory:
            return False, (
                f"Insufficient memory: need {required_memory:.1f}GB, "
                f"have {available_memory:.1f}GB"
            )

        if num_gpus < model.recommended_gpus:
            return False, (
                f"Insufficient GPUs: recommend {model.recommended_gpus}, " f"have {num_gpus}"
            )

        return True, "Requirements satisfied"


# Initialize registry on import
ModelRegistry.initialize()
