"""Tests for coordinator.models — ModelRegistry, ModelConfig, and enums."""

from pathlib import Path
from typing import Any, Dict

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


# ---------------------------------------------------------------------------
# GGUF/llama.cpp is the featured, primary engine — hardcoded registry coverage
# ---------------------------------------------------------------------------

# model_name -> (gguf_repo_id, gguf_file, gguf_n_ctx)
_FEATURED_GGUF_MODELS = {
    "qwen2.5-0.5b-gguf": (
        "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        4096,
    ),
    "qwen2.5-7b-instruct-gguf": (
        "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "qwen2.5-7b-instruct-q4_k_m.gguf",
        8192,
    ),
    "qwen2.5-coder-7b-gguf": (
        "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        8192,
    ),
    "llama3.1-8b-instruct-gguf": (
        "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        8192,
    ),
    "qwen2.5-14b-instruct-gguf": (
        "Qwen/Qwen2.5-14B-Instruct-GGUF",
        "qwen2.5-14b-instruct-q4_k_m.gguf",
        8192,
    ),
    "qwen2.5-coder-32b-gguf": (
        "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
        "qwen2.5-coder-32b-instruct-q4_k_m.gguf",
        8192,
    ),
}


@pytest.mark.parametrize("model_name", sorted(_FEATURED_GGUF_MODELS))
def test_featured_gguf_model_registered_out_of_the_box(model_name: str) -> None:
    """The hardcoded registry (no models.toml load needed) already ships the
    featured GGUF/llama.cpp models — the primary engine for consumer GPUs."""
    repo_id, file, n_ctx = _FEATURED_GGUF_MODELS[model_name]
    cfg = ModelRegistry.get_model(model_name)
    assert cfg is not None, model_name
    assert cfg.engine == "llamacpp"
    assert cfg.gguf_repo_id == repo_id
    assert cfg.gguf_file == file
    assert cfg.gguf_n_gpu_layers == -1
    assert cfg.gguf_n_ctx == n_ctx
    assert cfg.supports_quantization == [Quantization.NONE]


def test_qwen_coder_32b_gguf_is_the_multi_gpu_showcase() -> None:
    cfg = ModelRegistry.get_model("qwen2.5-coder-32b-gguf")
    assert cfg is not None
    assert cfg.recommended_gpus == 2
    assert cfg.max_gpus == 4


def test_hardcoded_gguf_models_match_models_toml() -> None:
    """coordinator/models.py and config/models.toml must never diverge for the
    featured GGUF models — both are read as sources of truth in different
    deployment modes (hardcoded default vs. config-file override)."""
    models_toml = Path(__file__).resolve().parents[2] / "config" / "models.toml"
    toml_data = toml.load(models_toml)["models"]
    for model_name in _FEATURED_GGUF_MODELS:
        hardcoded = ModelRegistry.get_model(model_name)
        assert hardcoded is not None, model_name
        toml_entry = toml_data[model_name]
        assert toml_entry["engine"] == "llamacpp"
        assert hardcoded.gguf_repo_id == toml_entry["gguf"]["repo_id"]
        assert hardcoded.gguf_file == toml_entry["gguf"]["file"]
        assert hardcoded.gguf_n_gpu_layers == toml_entry["gguf"]["n_gpu_layers"]
        assert hardcoded.gguf_n_ctx == toml_entry["gguf"]["n_ctx"]
        assert hardcoded.min_memory_gb == pytest.approx(float(toml_entry["min_memory_gb"]))
        assert hardcoded.recommended_gpus == toml_entry["recommended_gpus"]
        assert hardcoded.max_gpus == toml_entry["max_gpus"]


# ---------------------------------------------------------------------------
# Plan 11 Task 1 — Level 1: local multi-GPU split (local_gpu_ids / local_tensor_split)
# ---------------------------------------------------------------------------


def _llamacpp_kwargs(**overrides: Any) -> Dict[str, Any]:
    """Minimal required ModelConfig kwargs for an engine='llamacpp' model."""
    base: Dict[str, Any] = dict(
        name="local-split-test",
        family=ModelFamily.QWEN,
        parameters="32B",
        min_memory_gb=20,
        recommended_gpus=1,
        max_gpus=4,
        num_layers=0,
        hidden_size=0,
        num_attention_heads=0,
        vocab_size=0,
        max_seq_len=8192,
        intermediate_size=0,
        engine="llamacpp",
        gguf_repo_id="Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
        gguf_file="qwen2.5-coder-32b-instruct-q4_k_m.gguf",
    )
    base.update(overrides)
    return base


def test_local_gpu_ids_and_tensor_split_default_to_none() -> None:
    cfg = ModelConfig(**_llamacpp_kwargs())
    assert cfg.local_gpu_ids is None
    assert cfg.local_tensor_split is None
    assert cfg.grpc_metadata() == {
        "engine": "llamacpp",
        "gguf_repo_id": "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
        "gguf_file": "qwen2.5-coder-32b-instruct-q4_k_m.gguf",
    }


def test_local_tensor_split_length_must_match_local_gpu_ids() -> None:
    with pytest.raises(ValueError, match="length"):
        ModelConfig(**_llamacpp_kwargs(local_gpu_ids=[0, 1], local_tensor_split=[0.5]))


def test_local_tensor_split_requires_local_gpu_ids() -> None:
    with pytest.raises(ValueError, match="length"):
        ModelConfig(**_llamacpp_kwargs(local_tensor_split=[0.5, 0.5]))


def test_local_tensor_split_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="must all be"):
        ModelConfig(**_llamacpp_kwargs(local_gpu_ids=[0, 1], local_tensor_split=[0.5, 0.0]))


def test_local_gpu_ids_without_tensor_split_is_valid() -> None:
    # Equal-weight split across the pinned GPUs is a valid Level-1 config.
    cfg = ModelConfig(**_llamacpp_kwargs(local_gpu_ids=[0, 1]))
    assert cfg.local_gpu_ids == [0, 1]
    assert cfg.local_tensor_split is None
    assert "tensor_split" not in cfg.grpc_metadata()


def test_grpc_metadata_includes_tensor_split_when_local_split_set() -> None:
    cfg = ModelConfig(**_llamacpp_kwargs(local_gpu_ids=[0, 1], local_tensor_split=[0.6, 0.4]))
    assert cfg.grpc_metadata() == {
        "engine": "llamacpp",
        "gguf_repo_id": "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
        "gguf_file": "qwen2.5-coder-32b-instruct-q4_k_m.gguf",
        "tensor_split": "0.6,0.4",
    }


def test_load_from_dict_parses_local_multi_gpu_keys() -> None:
    ModelRegistry.load_from_dict(
        {
            "models": {
                "local-multi-gpu-gguf": {
                    "family": "qwen",
                    "parameters": "32B",
                    "min_memory_gb": 20,
                    "engine": "llamacpp",
                    "gguf": {
                        "repo_id": "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
                        "file": "qwen2.5-coder-32b-instruct-q4_k_m.gguf",
                    },
                    "local_gpu_ids": [1, 2],
                    "local_tensor_split": [0.7, 0.3],
                }
            }
        }
    )
    cfg = ModelRegistry.get_model("local-multi-gpu-gguf")
    assert cfg is not None
    assert cfg.local_gpu_ids == [1, 2]
    assert cfg.local_tensor_split == [0.7, 0.3]


def test_load_from_dict_local_multi_gpu_keys_absent_by_default() -> None:
    ModelRegistry.load_from_dict(
        {
            "models": {
                "no-local-split-gguf": {
                    "family": "qwen",
                    "parameters": "7B",
                    "min_memory_gb": 6,
                    "engine": "llamacpp",
                    "gguf": {
                        "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
                        "file": "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
                    },
                }
            }
        }
    )
    cfg = ModelRegistry.get_model("no-local-split-gguf")
    assert cfg is not None
    assert cfg.local_gpu_ids is None
    assert cfg.local_tensor_split is None


# ---------------------------------------------------------------------------
# Plan 11 Task 4 — Level 2: distributed registry schema (ggml-RPC)
# ---------------------------------------------------------------------------


def test_distributed_fields_default_off() -> None:
    cfg = ModelConfig(**_llamacpp_kwargs())
    assert cfg.distributed is False
    assert cfg.distributed_lead is None
    assert cfg.distributed_peers == []
    assert cfg.distributed_split is None
    assert cfg.distributed_rpc_port == 50151
    assert cfg.distributed_gpu_ids == {}


def test_distributed_requires_llamacpp_engine() -> None:
    with pytest.raises(ValueError, match="engine"):
        ModelConfig(
            name="bad-distributed-engine",
            family=ModelFamily.QWEN,
            parameters="32B",
            min_memory_gb=20,
            recommended_gpus=1,
            max_gpus=4,
            num_layers=0,
            hidden_size=0,
            num_attention_heads=0,
            vocab_size=0,
            max_seq_len=8192,
            intermediate_size=0,
            engine="burn",  # distributed requires llamacpp
            distributed=True,
            distributed_lead="node-1",
            distributed_peers=["node-2"],
        )


def test_distributed_requires_lead() -> None:
    with pytest.raises(ValueError, match="distributed_lead"):
        ModelConfig(**_llamacpp_kwargs(distributed=True, distributed_peers=["node-2"]))


def test_distributed_requires_at_least_one_peer() -> None:
    with pytest.raises(ValueError, match="peer"):
        ModelConfig(**_llamacpp_kwargs(distributed=True, distributed_lead="node-1"))


def test_distributed_valid_config() -> None:
    cfg = ModelConfig(
        **_llamacpp_kwargs(
            distributed=True,
            distributed_lead="amd-node-1",
            distributed_peers=["amd-node-2", "amd-node-3"],
            distributed_rpc_port=50151,
            distributed_gpu_ids={"amd-node-1": [0], "amd-node-2": [0]},
        )
    )
    assert cfg.distributed is True
    assert cfg.distributed_lead == "amd-node-1"
    assert cfg.distributed_peers == ["amd-node-2", "amd-node-3"]
    assert cfg.distributed_gpu_ids == {"amd-node-1": [0], "amd-node-2": [0]}


def test_grpc_metadata_lead_shape() -> None:
    cfg = ModelConfig(
        **_llamacpp_kwargs(
            distributed=True,
            distributed_lead="amd-node-1",
            distributed_peers=["amd-node-2", "amd-node-3"],
        )
    )
    metadata = cfg.grpc_metadata_lead(
        peer_endpoints=["10.0.0.2:50151", "10.0.0.3:50151"],
        tensor_split=[0.4, 0.3, 0.3],
    )
    assert metadata == {
        "engine": "llamacpp",
        "gguf_repo_id": "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
        "gguf_file": "qwen2.5-coder-32b-instruct-q4_k_m.gguf",
        "distributed_role": "lead",
        "rpc_peers": "10.0.0.2:50151,10.0.0.3:50151",
        "tensor_split": "0.4,0.3,0.3",
    }


def test_grpc_metadata_lead_omits_tensor_split_when_not_given() -> None:
    cfg = ModelConfig(
        **_llamacpp_kwargs(
            distributed=True,
            distributed_lead="amd-node-1",
            distributed_peers=["amd-node-2"],
        )
    )
    metadata = cfg.grpc_metadata_lead(peer_endpoints=["10.0.0.2:50151"], tensor_split=None)
    assert metadata == {
        "engine": "llamacpp",
        "gguf_repo_id": "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
        "gguf_file": "qwen2.5-coder-32b-instruct-q4_k_m.gguf",
        "distributed_role": "lead",
        "rpc_peers": "10.0.0.2:50151",
    }


def test_grpc_metadata_rpc_server_shape() -> None:
    cfg = ModelConfig(
        **_llamacpp_kwargs(
            distributed=True,
            distributed_lead="amd-node-1",
            distributed_peers=["amd-node-2"],
        )
    )
    metadata = cfg.grpc_metadata_rpc_server(base_port=50151)
    assert metadata == {
        "engine": "llamacpp",
        "gguf_repo_id": "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
        "gguf_file": "qwen2.5-coder-32b-instruct-q4_k_m.gguf",
        "distributed_role": "rpc_server",
        "rpc_bind_port": "50151",
    }


def test_grpc_metadata_stays_unchanged_for_distributed_models() -> None:
    """grpc_metadata() (the non-distributed transport helper) must never grow
    distributed_role/rpc_peers/rpc_bind_port keys — those live exclusively on
    grpc_metadata_lead / grpc_metadata_rpc_server."""
    cfg = ModelConfig(
        **_llamacpp_kwargs(
            distributed=True,
            distributed_lead="amd-node-1",
            distributed_peers=["amd-node-2", "amd-node-3"],
        )
    )
    assert cfg.grpc_metadata() == {
        "engine": "llamacpp",
        "gguf_repo_id": "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
        "gguf_file": "qwen2.5-coder-32b-instruct-q4_k_m.gguf",
    }


def test_load_from_dict_parses_distributed_section() -> None:
    ModelRegistry.load_from_dict(
        {
            "models": {
                "distributed-test-gguf": {
                    "family": "qwen",
                    "parameters": "32B",
                    "min_memory_gb": 20,
                    "engine": "llamacpp",
                    "gguf": {
                        "repo_id": "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
                        "file": "qwen2.5-coder-32b-instruct-q4_k_m.gguf",
                    },
                    "distributed": {
                        "enabled": True,
                        "lead": "amd-node-1",
                        "peers": ["amd-node-2", "amd-node-3"],
                        "split": "auto",
                        "rpc_port": 50151,
                        "gpu_ids": {
                            "amd-node-1": [0],
                            "amd-node-2": [0],
                            "amd-node-3": [0],
                        },
                    },
                }
            }
        }
    )
    cfg = ModelRegistry.get_model("distributed-test-gguf")
    assert cfg is not None
    assert cfg.distributed is True
    assert cfg.distributed_lead == "amd-node-1"
    assert cfg.distributed_peers == ["amd-node-2", "amd-node-3"]
    assert cfg.distributed_split is None  # "auto" is not a fixed weight list
    assert cfg.distributed_rpc_port == 50151
    assert cfg.distributed_gpu_ids == {
        "amd-node-1": [0],
        "amd-node-2": [0],
        "amd-node-3": [0],
    }


def test_load_from_dict_parses_distributed_explicit_split() -> None:
    ModelRegistry.load_from_dict(
        {
            "models": {
                "distributed-explicit-split-gguf": {
                    "family": "qwen",
                    "parameters": "32B",
                    "min_memory_gb": 20,
                    "engine": "llamacpp",
                    "gguf": {
                        "repo_id": "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
                        "file": "qwen2.5-coder-32b-instruct-q4_k_m.gguf",
                    },
                    "distributed": {
                        "enabled": True,
                        "lead": "amd-node-1",
                        "peers": ["amd-node-2"],
                        "split": [0.6, 0.4],
                    },
                }
            }
        }
    )
    cfg = ModelRegistry.get_model("distributed-explicit-split-gguf")
    assert cfg is not None
    assert cfg.distributed_split == [0.6, 0.4]


def test_load_from_dict_distributed_absent_by_default() -> None:
    ModelRegistry.load_from_dict(
        {
            "models": {
                "no-distributed-gguf": {
                    "family": "qwen",
                    "parameters": "7B",
                    "min_memory_gb": 6,
                    "engine": "llamacpp",
                    "gguf": {
                        "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
                        "file": "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
                    },
                }
            }
        }
    )
    cfg = ModelRegistry.get_model("no-distributed-gguf")
    assert cfg is not None
    assert cfg.distributed is False
    assert cfg.distributed_lead is None
    assert cfg.distributed_peers == []
    assert cfg.distributed_gpu_ids == {}


def test_real_models_toml_distributed_reference_entry_loads() -> None:
    """config/models.toml's qwen2.5-coder-32b-gguf.distributed reference block
    (Plan 11 Task 4) parses into a valid, fully populated distributed schema."""
    models_toml = Path(__file__).resolve().parents[2] / "config" / "models.toml"
    ModelRegistry.load_from_dict(toml.load(models_toml))
    cfg = ModelRegistry.get_model("qwen2.5-coder-32b-gguf")
    assert cfg is not None
    assert cfg.distributed is True
    assert cfg.distributed_lead == "amd-node-1"
    assert cfg.distributed_peers == ["amd-node-2", "amd-node-3"]
    assert cfg.distributed_split is None  # "auto"
    assert cfg.distributed_rpc_port == 50151
    assert cfg.distributed_gpu_ids == {
        "amd-node-1": [0],
        "amd-node-2": [0],
        "amd-node-3": [0],
    }


# ---------------------------------------------------------------------------
# Plan 11 Task 2b — KV-cache quantization (cache_type_k / cache_type_v)
# ---------------------------------------------------------------------------


def test_gguf_cache_type_defaults_to_none() -> None:
    cfg = ModelConfig(**_llamacpp_kwargs())
    assert cfg.gguf_cache_type_k is None
    assert cfg.gguf_cache_type_v is None
    assert "cache_type_k" not in cfg.grpc_metadata()
    assert "cache_type_v" not in cfg.grpc_metadata()


@pytest.mark.parametrize("cache_type", ["f16", "q8_0", "q4_0", "q5_0", "q5_1", "q4_1"])
def test_gguf_cache_type_accepts_allowed_values(cache_type: str) -> None:
    cfg = ModelConfig(
        **_llamacpp_kwargs(gguf_cache_type_k=cache_type, gguf_cache_type_v=cache_type)
    )
    assert cfg.gguf_cache_type_k == cache_type
    assert cfg.gguf_cache_type_v == cache_type


def test_gguf_cache_type_k_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="gguf_cache_type_k"):
        ModelConfig(**_llamacpp_kwargs(gguf_cache_type_k="q2_k"))


def test_gguf_cache_type_v_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="gguf_cache_type_v"):
        ModelConfig(**_llamacpp_kwargs(gguf_cache_type_v="fp16"))  # ggml name is "f16", not "fp16"


def test_grpc_metadata_includes_cache_type_when_set() -> None:
    cfg = ModelConfig(**_llamacpp_kwargs(gguf_cache_type_k="q8_0", gguf_cache_type_v="q4_0"))
    metadata = cfg.grpc_metadata()
    assert metadata["cache_type_k"] == "q8_0"
    assert metadata["cache_type_v"] == "q4_0"


def test_grpc_metadata_lead_includes_cache_type_when_set() -> None:
    cfg = ModelConfig(
        **_llamacpp_kwargs(
            distributed=True,
            distributed_lead="amd-node-1",
            distributed_peers=["amd-node-2"],
            gguf_cache_type_k="q8_0",
            gguf_cache_type_v="q8_0",
        )
    )
    metadata = cfg.grpc_metadata_lead(peer_endpoints=["10.0.0.2:50151"], tensor_split=None)
    assert metadata["cache_type_k"] == "q8_0"
    assert metadata["cache_type_v"] == "q8_0"


def test_grpc_metadata_rpc_server_includes_cache_type_when_set() -> None:
    cfg = ModelConfig(
        **_llamacpp_kwargs(
            distributed=True,
            distributed_lead="amd-node-1",
            distributed_peers=["amd-node-2"],
            gguf_cache_type_k="q4_0",
            gguf_cache_type_v="q4_0",
        )
    )
    metadata = cfg.grpc_metadata_rpc_server(base_port=50151)
    assert metadata["cache_type_k"] == "q4_0"
    assert metadata["cache_type_v"] == "q4_0"


def test_metadata_producers_omit_cache_type_when_unset() -> None:
    """All three llama.cpp metadata helpers must stay byte-for-byte unchanged
    for models that don't set cache_type_k/v (back-compat)."""
    cfg = ModelConfig(
        **_llamacpp_kwargs(
            distributed=True,
            distributed_lead="amd-node-1",
            distributed_peers=["amd-node-2"],
        )
    )
    assert "cache_type_k" not in cfg.grpc_metadata()
    assert "cache_type_v" not in cfg.grpc_metadata()

    lead_metadata = cfg.grpc_metadata_lead(peer_endpoints=["10.0.0.2:50151"], tensor_split=None)
    assert "cache_type_k" not in lead_metadata
    assert "cache_type_v" not in lead_metadata

    rpc_server_metadata = cfg.grpc_metadata_rpc_server(base_port=50151)
    assert "cache_type_k" not in rpc_server_metadata
    assert "cache_type_v" not in rpc_server_metadata


def test_load_from_dict_parses_cache_type_keys() -> None:
    ModelRegistry.load_from_dict(
        {
            "models": {
                "cache-type-test-gguf": {
                    "family": "qwen",
                    "parameters": "32B",
                    "min_memory_gb": 20,
                    "engine": "llamacpp",
                    "gguf": {
                        "repo_id": "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
                        "file": "qwen2.5-coder-32b-instruct-q4_k_m.gguf",
                        "cache_type_k": "q8_0",
                        "cache_type_v": "q8_0",
                    },
                }
            }
        }
    )
    cfg = ModelRegistry.get_model("cache-type-test-gguf")
    assert cfg is not None
    assert cfg.gguf_cache_type_k == "q8_0"
    assert cfg.gguf_cache_type_v == "q8_0"


def test_load_from_dict_cache_type_absent_by_default() -> None:
    ModelRegistry.load_from_dict(
        {
            "models": {
                "no-cache-type-gguf": {
                    "family": "qwen",
                    "parameters": "7B",
                    "min_memory_gb": 6,
                    "engine": "llamacpp",
                    "gguf": {
                        "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
                        "file": "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
                    },
                }
            }
        }
    )
    cfg = ModelRegistry.get_model("no-cache-type-gguf")
    assert cfg is not None
    assert cfg.gguf_cache_type_k is None
    assert cfg.gguf_cache_type_v is None


def test_real_models_toml_cache_type_example_loads() -> None:
    """config/models.toml's qwen2.5-coder-32b-gguf.gguf cache_type_k/v example
    (Plan 11 Task 2b) parses and flows into grpc_metadata()."""
    models_toml = Path(__file__).resolve().parents[2] / "config" / "models.toml"
    ModelRegistry.load_from_dict(toml.load(models_toml))
    cfg = ModelRegistry.get_model("qwen2.5-coder-32b-gguf")
    assert cfg is not None
    assert cfg.gguf_cache_type_k == "q8_0"
    assert cfg.gguf_cache_type_v == "q8_0"
    metadata = cfg.grpc_metadata()
    assert metadata["cache_type_k"] == "q8_0"
    assert metadata["cache_type_v"] == "q8_0"
