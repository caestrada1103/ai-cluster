"""Tests for coordinator.models — ModelRegistry, ModelConfig, and enums."""

from pathlib import Path

import pytest
import toml

from coordinator.models import (
    ModelConfig,
    ModelFamily,
    ModelRegistry,
    ParallelismStrategy,
    Quantization,
)

# ---------------------------------------------------------------------------
# Enum sanity checks
# ---------------------------------------------------------------------------


def test_quantization_enum_values() -> None:
    assert Quantization.FP16.value == "fp16"
    assert Quantization.INT8.value == "int8"
    assert Quantization.INT4.value == "int4"
    assert Quantization.FP8.value == "fp8"
    assert Quantization.NONE.value == "none"


def test_parallelism_strategy_enum_values() -> None:
    assert ParallelismStrategy.SINGLE.value == "single"
    assert ParallelismStrategy.PIPELINE.value == "pipeline"
    assert ParallelismStrategy.TENSOR.value == "tensor"
    assert ParallelismStrategy.EXPERT.value == "expert"


# ---------------------------------------------------------------------------
# ModelConfig validation
# ---------------------------------------------------------------------------


def test_moe_requires_num_experts() -> None:
    with pytest.raises(ValueError, match="num_experts"):
        ModelConfig(
            name="bad-moe",
            family=ModelFamily.DEEPSEEK,
            parameters="7B",
            min_memory_gb=16,
            recommended_gpus=1,
            max_gpus=2,
            num_layers=30,
            hidden_size=4096,
            num_attention_heads=32,
            vocab_size=32000,
            max_seq_len=4096,
            intermediate_size=11008,
            is_moe=True,
            num_experts=None,  # should raise
        )


def test_gqa_defaults_num_kv_heads() -> None:
    cfg = ModelConfig(
        name="test-gqa",
        family=ModelFamily.LLAMA,
        parameters="8B",
        min_memory_gb=16,
        recommended_gpus=1,
        max_gpus=2,
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        vocab_size=32000,
        max_seq_len=4096,
        intermediate_size=14336,
        num_kv_heads=None,  # should default to num_attention_heads
    )
    assert cfg.num_kv_heads == 32


# ---------------------------------------------------------------------------
# ModelRegistry — default models
# ---------------------------------------------------------------------------


def test_registry_initialized() -> None:
    models = ModelRegistry.list_models()
    assert len(models) >= 5


def test_get_known_model_deepseek() -> None:
    cfg = ModelRegistry.get_model("deepseek-7b")
    assert cfg is not None
    assert cfg.name == "deepseek-7b"
    assert cfg.family == ModelFamily.DEEPSEEK
    assert cfg.num_layers == 30


def test_get_known_model_llama() -> None:
    cfg = ModelRegistry.get_model("llama3-8b")
    assert cfg is not None
    assert cfg.family == ModelFamily.LLAMA
    assert cfg.num_kv_heads == 8  # GQA


def test_get_unknown_model_returns_none() -> None:
    assert ModelRegistry.get_model("totally-nonexistent-model-xyz") is None


def test_find_models_by_family_llama() -> None:
    llama_models = ModelRegistry.find_models_by_family(ModelFamily.LLAMA)
    names = [m.name for m in llama_models]
    assert "llama3-8b" in names
    assert "llama3-70b" in names


def test_find_models_by_family_returns_only_matching() -> None:
    deepseek_models = ModelRegistry.find_models_by_family(ModelFamily.DEEPSEEK)
    for m in deepseek_models:
        assert m.family == ModelFamily.DEEPSEEK


# ---------------------------------------------------------------------------
# ModelRegistry.validate_requirements
# ---------------------------------------------------------------------------


def test_validate_requirements_pass() -> None:
    # phi-2: min_memory_gb=6, FP16 multiplier=0.5 → need 3GB → fits in 6GB
    ok, msg = ModelRegistry.validate_requirements("phi-2", available_memory=6.0, num_gpus=1)
    assert ok is True
    assert "satisfied" in msg


def test_validate_requirements_fail_memory() -> None:
    # phi-2 requires >0 GB; 0.1 GB will always fail
    ok, msg = ModelRegistry.validate_requirements("phi-2", available_memory=0.1, num_gpus=1)
    assert ok is False
    assert "memory" in msg.lower() or "insufficient" in msg.lower()


def test_validate_requirements_unknown_model() -> None:
    ok, msg = ModelRegistry.validate_requirements("ghost-model", available_memory=100.0, num_gpus=8)
    assert ok is False
    assert "Unknown" in msg or "ghost-model" in msg


# ---------------------------------------------------------------------------
# ModelRegistry.load_from_dict
# ---------------------------------------------------------------------------


def test_load_from_dict_adds_model() -> None:
    config_dict = {
        "models": {
            "test-tiny": {
                "name": "test-tiny",
                "family": "llama",
                "parameters": "1B",
                "min_memory_gb": 2,
                "recommended_gpus": 1,
                "max_gpus": 1,
                "architecture": {
                    "num_layers": 8,
                    "hidden_size": 512,
                    "num_attention_heads": 8,
                    "vocab_size": 32000,
                    "max_seq_len": 512,
                    "intermediate_size": 2048,
                },
                "quantization": {"supported": ["none"]},
                "parallelism": {"supported": ["single"]},
            }
        }
    }
    ModelRegistry.load_from_dict(config_dict)
    cfg = ModelRegistry.get_model("test-tiny")
    assert cfg is not None
    assert cfg.parameters == "1B"
    assert cfg.num_layers == 8


# ---------------------------------------------------------------------------
# Task 19 — quantization honesty
# ---------------------------------------------------------------------------


def test_registry_models_advertise_only_none_quantization() -> None:
    """Workers reject non-NONE quantization until real quantized inference lands."""
    for name in ModelRegistry.list_models():
        cfg = ModelRegistry.get_model(name)
        assert cfg is not None
        assert cfg.supports_quantization == [Quantization.NONE], name


# ---------------------------------------------------------------------------
# Task 20 — registry-name -> HF repo resolution
# ---------------------------------------------------------------------------


def test_hf_repo_id_loaded_from_toml() -> None:
    ModelRegistry.load_from_dict(
        {
            "models": {
                "friendly-name": {
                    "family": "llama",
                    "architecture": {"num_layers": 2, "hidden_size": 8},
                    "hf": {"repo_id": "org/real-repo"},
                }
            }
        }
    )
    cfg = ModelRegistry.get_model("friendly-name")
    assert cfg is not None
    assert cfg.hf_repo_id == "org/real-repo"


def test_builtin_models_carry_hf_repo_id() -> None:
    cfg = ModelRegistry.get_model("llama3-8b")
    assert cfg is not None
    assert cfg.hf_repo_id == "meta-llama/Meta-Llama-3-8B"


# ---------------------------------------------------------------------------
# Task 21 — models.toml + coordinator/models.py data corrections
# ---------------------------------------------------------------------------


def test_deepseek_7b_is_dense() -> None:
    cfg = ModelRegistry.get_model("deepseek-7b")
    assert cfg is not None
    assert cfg.is_moe is False
    assert cfg.num_experts is None
    assert cfg.vocab_size == 102400


def test_qwen_coder_entry_matches_real_repo() -> None:
    cfg = ModelRegistry.get_model("qwen3-coder-32b")
    assert cfg is not None
    assert cfg.hf_repo_id == "Qwen/Qwen2.5-Coder-32B-Instruct"
    assert cfg.vocab_size == 152064
    assert cfg.max_seq_len == 32768


# ---------------------------------------------------------------------------
# llama.cpp engine / GGUF registry fields
# ---------------------------------------------------------------------------


def test_engine_defaults_to_burn() -> None:
    cfg = ModelRegistry.get_model("llama3-8b")
    assert cfg is not None
    assert cfg.engine == "burn"
    assert cfg.grpc_metadata() == {}


def test_llamacpp_engine_requires_gguf_source() -> None:
    with pytest.raises(ValueError, match="gguf"):
        ModelConfig(
            name="bad-gguf",
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
            engine="llamacpp",  # no gguf_repo_id / gguf_file -> must raise
        )


def test_unknown_engine_rejected() -> None:
    with pytest.raises(ValueError, match="engine"):
        ModelConfig(
            name="bad-engine",
            family=ModelFamily.LLAMA,
            parameters="7B",
            min_memory_gb=8,
            recommended_gpus=1,
            max_gpus=1,
            num_layers=0,
            hidden_size=0,
            num_attention_heads=0,
            vocab_size=0,
            max_seq_len=2048,
            intermediate_size=0,
            engine="vllm",
        )


def test_load_from_dict_parses_gguf_section() -> None:
    ModelRegistry.load_from_dict(
        {
            "models": {
                "registry-test-gguf": {
                    "family": "qwen",
                    "parameters": "0.5B",
                    "min_memory_gb": 1,
                    "engine": "llamacpp",
                    "gguf": {
                        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                        "file": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
                        "n_gpu_layers": -1,
                        "n_ctx": 4096,
                    },
                }
            }
        }
    )
    cfg = ModelRegistry.get_model("registry-test-gguf")
    assert cfg is not None
    assert cfg.engine == "llamacpp"
    assert cfg.gguf_repo_id == "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    assert cfg.gguf_file == "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    assert cfg.gguf_n_gpu_layers == -1
    assert cfg.gguf_n_ctx == 4096


def test_grpc_metadata_for_llamacpp_model() -> None:
    ModelRegistry.load_from_dict(
        {
            "models": {
                "metadata-test-gguf": {
                    "family": "qwen",
                    "parameters": "0.5B",
                    "min_memory_gb": 1,
                    "engine": "llamacpp",
                    "gguf": {
                        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                        "file": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
                        "n_gpu_layers": 20,
                        "n_ctx": 2048,
                    },
                }
            }
        }
    )
    cfg = ModelRegistry.get_model("metadata-test-gguf")
    assert cfg is not None
    assert cfg.grpc_metadata() == {
        "engine": "llamacpp",
        "gguf_repo_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "gguf_file": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "n_gpu_layers": "20",
        "n_ctx": "2048",
    }


def test_real_models_toml_gguf_entry_loads() -> None:
    models_toml = Path(__file__).resolve().parents[2] / "config" / "models.toml"
    ModelRegistry.load_from_dict(toml.load(models_toml))
    cfg = ModelRegistry.get_model("qwen2.5-0.5b-gguf")
    assert cfg is not None
    assert cfg.engine == "llamacpp"
    assert cfg.gguf_repo_id == "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    assert cfg.gguf_file == "qwen2.5-0.5b-instruct-q4_k_m.gguf"
