"""Tests for coordinator.models — ModelRegistry, ModelConfig, and enums."""

import pytest

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

def test_quantization_enum_values():
    assert Quantization.FP16.value == "fp16"
    assert Quantization.INT8.value == "int8"
    assert Quantization.INT4.value == "int4"
    assert Quantization.FP8.value == "fp8"
    assert Quantization.NONE.value == "none"


def test_parallelism_strategy_enum_values():
    assert ParallelismStrategy.SINGLE.value == "single"
    assert ParallelismStrategy.PIPELINE.value == "pipeline"
    assert ParallelismStrategy.TENSOR.value == "tensor"
    assert ParallelismStrategy.EXPERT.value == "expert"


# ---------------------------------------------------------------------------
# ModelConfig validation
# ---------------------------------------------------------------------------

def test_moe_requires_num_experts():
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


def test_gqa_defaults_num_kv_heads():
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

def test_registry_initialized():
    models = ModelRegistry.list_models()
    assert len(models) >= 5


def test_get_known_model_deepseek():
    cfg = ModelRegistry.get_model("deepseek-7b")
    assert cfg is not None
    assert cfg.name == "deepseek-7b"
    assert cfg.family == ModelFamily.DEEPSEEK
    assert cfg.num_layers == 30


def test_get_known_model_llama():
    cfg = ModelRegistry.get_model("llama3-8b")
    assert cfg is not None
    assert cfg.family == ModelFamily.LLAMA
    assert cfg.num_kv_heads == 8  # GQA


def test_get_unknown_model_returns_none():
    assert ModelRegistry.get_model("totally-nonexistent-model-xyz") is None


def test_find_models_by_family_llama():
    llama_models = ModelRegistry.find_models_by_family(ModelFamily.LLAMA)
    names = [m.name for m in llama_models]
    assert "llama3-8b" in names
    assert "llama3-70b" in names


def test_find_models_by_family_returns_only_matching():
    deepseek_models = ModelRegistry.find_models_by_family(ModelFamily.DEEPSEEK)
    for m in deepseek_models:
        assert m.family == ModelFamily.DEEPSEEK


# ---------------------------------------------------------------------------
# ModelRegistry.validate_requirements
# ---------------------------------------------------------------------------

def test_validate_requirements_pass():
    # phi-2: min_memory_gb=6, FP16 multiplier=0.5 → need 3GB → fits in 6GB
    ok, msg = ModelRegistry.validate_requirements("phi-2", available_memory=6.0, num_gpus=1)
    assert ok is True
    assert "satisfied" in msg


def test_validate_requirements_fail_memory():
    # phi-2 requires >0 GB; 0.1 GB will always fail
    ok, msg = ModelRegistry.validate_requirements("phi-2", available_memory=0.1, num_gpus=1)
    assert ok is False
    assert "memory" in msg.lower() or "insufficient" in msg.lower()


def test_validate_requirements_unknown_model():
    ok, msg = ModelRegistry.validate_requirements("ghost-model", available_memory=100.0, num_gpus=8)
    assert ok is False
    assert "Unknown" in msg or "ghost-model" in msg


# ---------------------------------------------------------------------------
# ModelRegistry.load_from_dict
# ---------------------------------------------------------------------------

def test_load_from_dict_adds_model():
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
                "quantization": {"supported": ["fp16"]},
                "parallelism": {"supported": ["single"]},
            }
        }
    }
    ModelRegistry.load_from_dict(config_dict)
    cfg = ModelRegistry.get_model("test-tiny")
    assert cfg is not None
    assert cfg.parameters == "1B"
    assert cfg.num_layers == 8
