"""Model registry and configuration."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


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
    
    # Metadata
    description: str = ""
    paper_url: Optional[str] = None
    model_url: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if self.is_moe and not self.num_experts:
            raise ValueError("MoE models must specify num_experts")
        
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_attention_heads


class ModelRegistry:
    """Registry of all available models."""
    
    # Predefined model configurations
    MODELS: Dict[str, ModelConfig] = {}
    
    @classmethod
    def initialize(cls):
        """Initialize the model registry with default models."""
        # Note: In a production environment, this could be empty and 
        # only populated via load_from_config.
        
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
            vocab_size=32256,
            max_seq_len=4096,
            intermediate_size=11008,
            supports_quantization=[Quantization.FP16, Quantization.INT8],
            supports_parallelism=[ParallelismStrategy.SINGLE, ParallelismStrategy.PIPELINE],
            is_moe=True,
            num_experts=64,
            description="DeepSeek 7B Base Model with MoE architecture",
            model_url="https://huggingface.co/deepseek-ai/deepseek-llm-7b-base",
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
            num_kv_heads=64,
            vocab_size=32256,
            max_seq_len=4096,
            intermediate_size=22016,
            supports_quantization=[Quantization.INT8, Quantization.INT4],
            supports_parallelism=[
                ParallelismStrategy.PIPELINE,
                ParallelismStrategy.TENSOR,
                ParallelismStrategy.EXPERT,
            ],
            is_moe=True,
            num_experts=128,
            description="DeepSeek 67B Model with extensive MoE",
            model_url="https://huggingface.co/deepseek-ai/deepseek-llm-67b-base",
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
            supports_quantization=[Quantization.FP16, Quantization.INT8],
            supports_parallelism=[
                ParallelismStrategy.SINGLE,
                ParallelismStrategy.PIPELINE,
                ParallelismStrategy.TENSOR,
            ],
            description="Meta Llama 3 8B Instruct",
            model_url="https://huggingface.co/meta-llama/Meta-Llama-3-8B",
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
            supports_quantization=[Quantization.INT8, Quantization.INT4],
            supports_parallelism=[
                ParallelismStrategy.PIPELINE,
                ParallelismStrategy.TENSOR,
            ],
            description="Meta Llama 3 70B Instruct",
            model_url="https://huggingface.co/meta-llama/Meta-Llama-3-70B",
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
            supports_quantization=[Quantization.FP16, Quantization.INT8],
            supports_parallelism=[
                ParallelismStrategy.SINGLE,
                ParallelismStrategy.PIPELINE,
            ],
            description="Mistral 7B v0.2",
            model_url="https://huggingface.co/mistralai/Mistral-7B-v0.1",
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
            supports_quantization=[Quantization.FP16],
            supports_parallelism=[ParallelismStrategy.SINGLE],
            description="Microsoft Phi-2 (2.7B)",
            model_url="https://huggingface.co/microsoft/phi-2",
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
            vocab_size=151936,
            max_seq_len=131072,
            intermediate_size=27648,
            supports_quantization=[Quantization.FP16, Quantization.INT8, Quantization.INT4],
            supports_parallelism=[ParallelismStrategy.SINGLE, ParallelismStrategy.TENSOR],
            description="Qwen3-Coder 32B — top open-source coding model (SWE-Bench 70.6%), Claude Sonnet tier",
            model_url="https://huggingface.co/Qwen/Qwen3-Coder-32B",
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
            supports_quantization=[Quantization.FP16, Quantization.INT8, Quantization.INT4],
            supports_parallelism=[
                ParallelismStrategy.SINGLE,
                ParallelismStrategy.TENSOR,
                ParallelismStrategy.PIPELINE,
            ],
            description="DeepSeek V3 — 671B MoE (37B active params), Claude Opus tier (SWE-Bench 73.1%)",
            model_url="https://huggingface.co/deepseek-ai/DeepSeek-V3",
        )

        logger.info(f"Initialized model registry with {len(cls.MODELS)} models")

    @classmethod
    def load_from_dict(cls, config_dict: Dict[str, Any]):
        """Load models from a configuration dictionary (e.g. from models.toml)."""
        if "models" not in config_dict:
            return

        defaults = config_dict.get("defaults", {})

        for name, data in config_dict["models"].items():
            try:
                # Merge with defaults
                # We need to be careful with nested dicts like 'architecture'
                # For now, we just do a shallow copy of defaults and update with model data
                final_data = defaults.copy()
                final_data.update(data)
                
                # Extract components with robustness for string vs dict
                def get_dict(key):
                    val = final_data.get(key, {})
                    if isinstance(val, dict):
                        return val
                    return {"default": val} # Fallback for flat values from defaults

                arch = get_dict("architecture")
                quants = get_dict("quantization")
                parallel = get_dict("parallelism")
                paths = get_dict("paths")
                
                # Determine supported quantizations
                supported_quants = quants.get("supported")
                if not supported_quants:
                    default_q = quants.get("default", "fp16")
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
                    logger.warning(f"Unknown model family '{raw_family}' for model {name}, defaulting to llama")
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
                    description=final_data.get("description", ""),
                )
                cls.MODELS[model_name] = model
            except Exception as e:
                logger.error(f"Failed to load model {name} from config: {e}")

        logger.info(f"Updated model registry. Total models: {len(cls.MODELS)}")


    
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
                f"Insufficient GPUs: recommend {model.recommended_gpus}, "
                f"have {num_gpus}"
            )
        
        return True, "Requirements satisfied"


# Initialize registry on import
ModelRegistry.initialize()