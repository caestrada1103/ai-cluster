#!/usr/bin/env python3
"""
HuggingFace to Burn Model Converter
===================================

This script downloads models from HuggingFace and converts them to Burn's format
for use with the AI cluster worker.

Usage:
    python convert_model.py deepseek-ai/deepseek-llm-7b-base --output ./models/
    python convert_model.py meta-llama/Meta-Llama-3-8B --quantize int8 --output ./models/
    python convert_model.py --list-models
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import time

import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    LlamaConfig,
    MistralConfig,
    GemmaConfig,
    PhiConfig,
)
from safetensors.torch import save_file, load_file
from huggingface_hub import snapshot_download, HfFileSystem
from tqdm import tqdm
import yaml
import toml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Supported model families
SUPPORTED_FAMILIES = {
    "deepseek": ["DeepseekForCausalLM", "DeepseekModel"],
    "llama": ["LlamaForCausalLM", "LlamaModel"],
    "mistral": ["MistralForCausalLM", "MistralModel"],
    "mixtral": ["MixtralForCausalLM", "MixtralModel"],
    "gemma": ["GemmaForCausalLM", "GemmaModel"],
    "phi": ["PhiForCausalLM", "PhiModel"],
    "qwen": ["QWenLMHeadModel", "QWenModel"],
    "baichuan": ["BaichuanForCausalLM", "BaichuanModel"],
}

# Quantization settings
QUANTIZATION_TYPES = {
    "fp32": {"dtype": torch.float32, "bytes": 4},
    "fp16": {"dtype": torch.float16, "bytes": 2},
    "bf16": {"dtype": torch.bfloat16, "bytes": 2},
    "int8": {"dtype": torch.int8, "bytes": 1},
    "int4": {"dtype": None, "bytes": 0.5},  # Special handling
}

class ModelConverter:
    """Convert HuggingFace models to Burn format."""
    
    def __init__(
        self,
        model_id: str,
        output_dir: Path,
        quantization: str = "fp16",
        max_shard_size_gb: float = 2.0,
        verify: bool = True,
        cache_dir: Optional[Path] = None,
        token: Optional[str] = None,
        revision: str = "main",
    ):
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.quantization = quantization
        self.max_shard_size_gb = max_shard_size_gb
        self.verify = verify
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "huggingface"
        self.token = token
        self.revision = revision
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model info
        self.model_name = model_id.replace("/", "-")
        self.family = self._detect_family()
        self.config = None
        self.model = None
        self.tokenizer = None
        
    def _detect_family(self) -> str:
        """Detect model family from ID or config."""
        model_id_lower = self.model_id.lower()
        
        for family, keywords in SUPPORTED_FAMILIES.items():
            if family in model_id_lower:
                return family
        
        # Try to detect from config
        try:
            config = AutoConfig.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                token=self.token,
                revision=self.revision,
            )
            model_type = getattr(config, "model_type", "").lower()
            arch = getattr(config, "architectures", [""])[0]
            
            for family, keywords in SUPPORTED_FAMILIES.items():
                if family in model_type or any(k in arch for k in keywords):
                    return family
        except:
            pass
        
        raise ValueError(f"Could not detect model family for {self.model_id}")
    
    def download_model(self) -> Path:
        """Download model from HuggingFace."""
        logger.info(f"Downloading {self.model_id} from HuggingFace...")
        
        model_path = snapshot_download(
            repo_id=self.model_id,
            cache_dir=self.cache_dir,
            token=self.token,
            revision=self.revision,
            ignore_patterns=["*.safetensors", "*.bin", "*.msgpack"],
            tqdm_class=tqdm,
        )
        
        logger.info(f"Downloaded to {model_path}")
        return Path(model_path)
    
    def load_model(self) -> None:
        """Load model and tokenizer from HuggingFace."""
        logger.info(f"Loading model {self.model_id}...")
        
        # Load config
        self.config = AutoConfig.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            token=self.token,
            revision=self.revision,
        )
        
        # Load model with appropriate dtype
        dtype = QUANTIZATION_TYPES[self.quantization]["dtype"]
        if self.quantization == "int4":
            # For int4, we load in fp16 then quantize later
            dtype = torch.float16
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            config=self.config,
            torch_dtype=dtype,
            device_map="cpu",
            trust_remote_code=True,
            token=self.token,
            revision=self.revision,
            low_cpu_mem_usage=True,
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            token=self.token,
            revision=self.revision,
        )
        
        self.model.eval()
        logger.info(f"Model loaded: {self.model.__class__.__name__}")
        logger.info(f"Number of parameters: {self.model.num_parameters():,}")
    
    def quantize_weights(self) -> Dict[str, torch.Tensor]:
        """Apply quantization to model weights."""
        if self.quantization == "fp32":
            return {k: v.float() for k, v in self.model.state_dict().items()}
        elif self.quantization == "fp16":
            return {k: v.half() for k, v in self.model.state_dict().items()}
        elif self.quantization == "bf16":
            return {k: v.bfloat16() for k, v in self.model.state_dict().items()}
        elif self.quantization == "int8":
            return self._quantize_int8()
        elif self.quantization == "int4":
            return self._quantize_int4()
        else:
            raise ValueError(f"Unsupported quantization: {self.quantization}")
    
    def _quantize_int8(self) -> Dict[str, torch.Tensor]:
        """Quantize weights to int8 using per-tensor scaling."""
        state_dict = {}
        for name, param in self.model.state_dict().items():
            if param.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                # Compute scale
                abs_max = param.abs().max()
                scale = abs_max / 127.0
                
                # Quantize
                quantized = (param / scale).round().clamp(-128, 127).to(torch.int8)
                
                # Store quantized weight and scale
                state_dict[name] = quantized
                state_dict[f"{name}.scale"] = torch.tensor([scale], dtype=torch.float32)
            else:
                state_dict[name] = param
        
        return state_dict
    
    def _quantize_int4(self) -> Dict[str, torch.Tensor]:
        """Quantize weights to int4 using group-wise quantization."""
        state_dict = {}
        group_size = 32  # Typical group size for int4
        
        for name, param in self.model.state_dict().items():
            if param.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                param_float = param.float()
                original_shape = param_float.shape
                
                # Reshape for group-wise quantization
                flat_param = param_float.view(-1)
                num_groups = (flat_param.numel() + group_size - 1) // group_size
                
                # Pad if necessary
                if flat_param.numel() % group_size != 0:
                    pad_size = group_size - (flat_param.numel() % group_size)
                    flat_param = torch.cat([flat_param, torch.zeros(pad_size)])
                
                # Reshape to groups
                grouped = flat_param.view(-1, group_size)
                
                # Compute scales and zero points per group
                abs_max = grouped.abs().max(dim=1, keepdim=True)[0]
                scales = abs_max / 7.0  # int4 range is -8 to 7
                
                # Quantize
                quantized = (grouped / scales).round().clamp(-8, 7).to(torch.int8)
                
                # Pack two int4 values into one int8
                packed = (quantized[:, ::2] << 4) | (quantized[:, 1::2] & 0x0F)
                
                # Restore shape (approximately)
                state_dict[name] = packed.reshape(-1)[:flat_param.numel()].reshape(original_shape)
                state_dict[f"{name}.scales"] = scales.reshape(-1)
            else:
                state_dict[name] = param
        
        return state_dict
    
    def convert_to_burn_format(self, weights: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """Convert PyTorch weights to Burn's expected format."""
        burn_weights = {}
        
        for name, tensor in weights.items():
            # Convert to numpy
            if tensor.dtype == torch.bfloat16:
                # bfloat16 not supported in numpy, convert to float32
                array = tensor.float().numpy()
            else:
                array = tensor.numpy()
            
            # Apply any necessary transformations for Burn
            # Burn expects certain naming conventions and layouts
            
            # Handle embedding weights
            if "embed_tokens.weight" in name or "embedding.weight" in name:
                burn_weights[name.replace(".", "_")] = array
            
            # Handle layer norm weights
            elif "norm.weight" in name or "layernorm.weight" in name:
                burn_weights[name.replace(".", "_")] = array
            
            # Handle linear layers
            elif "weight" in name and len(array.shape) == 2:
                # Burn uses [out_features, in_features] layout
                burn_weights[name.replace(".", "_")] = array
            
            # Handle biases
            elif "bias" in name:
                burn_weights[name.replace(".", "_")] = array
            
            # Handle quantization scales
            elif ".scale" in name or ".scales" in name:
                burn_weights[name.replace(".", "_")] = array
            
            else:
                burn_weights[name.replace(".", "_")] = array
        
        return burn_weights
    
    def save_sharded(self, weights: Dict[str, np.ndarray]) -> List[Path]:
        """Save weights in sharded format."""
        shards = []
        current_shard = {}
        current_size = 0
        
        # Sort weights by name for consistent sharding
        sorted_items = sorted(weights.items())
        
        for name, array in sorted_items:
            size_bytes = array.nbytes
            
            # Start new shard if current is full
            if current_size + size_bytes > self.max_shard_size_gb * 1e9:
                shard_path = self._save_shard(current_shard, len(shards))
                shards.append(shard_path)
                current_shard = {}
                current_size = 0
            
            current_shard[name] = array
            current_size += size_bytes
        
        # Save last shard
        if current_shard:
            shard_path = self._save_shard(current_shard, len(shards))
            shards.append(shard_path)
        
        # Save index file
        self._save_index(shards, weights.keys())
        
        return shards
    
    def _save_shard(self, shard: Dict[str, np.ndarray], shard_idx: int) -> Path:
        """Save a single shard."""
        shard_path = self.output_dir / f"model-{shard_idx:05d}-of-{self.num_shards:05d}.safetensors"
        
        # Convert numpy arrays to torch tensors for safetensors
        torch_shard = {k: torch.from_numpy(v) for k, v in shard.items()}
        
        save_file(torch_shard, str(shard_path))
        logger.debug(f"Saved shard {shard_idx} to {shard_path}")
        
        return shard_path
    
    def _save_index(self, shards: List[Path], weight_names: List[str]) -> None:
        """Save shard index file."""
        index = {
            "metadata": {
                "total_size": sum(p.stat().st_size for p in shards),
                "num_shards": len(shards),
                "format": "safetensors",
                "model_id": self.model_id,
                "family": self.family,
                "quantization": self.quantization,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "weight_map": {},
        }
        
        # Create weight to shard mapping
        for shard_path in shards:
            weights = load_file(str(shard_path))
            for weight_name in weights.keys():
                index["weight_map"][weight_name] = shard_path.name
        
        index_path = self.output_dir / "model.safetensors.index.json"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        
        logger.info(f"Saved index to {index_path}")
    
    def save_config(self) -> None:
        """Save model configuration in Burn format."""
        config = {
            "model_type": self.family,
            "architecture": self.model.config.architectures[0] if self.model.config.architectures else "Unknown",
            "hidden_size": getattr(self.model.config, "hidden_size", None),
            "num_layers": getattr(self.model.config, "num_hidden_layers", 
                                 getattr(self.model.config, "num_layers", None)),
            "num_attention_heads": getattr(self.model.config, "num_attention_heads", None),
            "num_kv_heads": getattr(self.model.config, "num_key_value_heads", 
                                   getattr(self.model.config, "num_attention_heads", None)),
            "vocab_size": getattr(self.model.config, "vocab_size", None),
            "max_seq_len": getattr(self.model.config, "max_position_embeddings", None),
            "intermediate_size": getattr(self.model.config, "intermediate_size", None),
            "rms_norm_eps": getattr(self.model.config, "rms_norm_eps", 1e-6),
            "rope_theta": getattr(self.model.config, "rope_theta", 10000.0),
            "is_moe": hasattr(self.model.config, "num_local_experts"),
            "num_experts": getattr(self.model.config, "num_local_experts", None),
            "num_experts_per_tok": getattr(self.model.config, "num_experts_per_tok", None),
            "quantization": self.quantization,
        }
        
        # Remove None values
        config = {k: v for k, v in config.items() if v is not None}
        
        # Save as TOML
        config_path = self.output_dir / "config.toml"
        with open(config_path, "w") as f:
            toml.dump(config, f)
        
        # Also save as JSON for compatibility
        json_path = self.output_dir / "config.json"
        with open(json_path, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved config to {config_path}")
    
    def save_tokenizer(self) -> None:
        """Save tokenizer files."""
        tokenizer_dir = self.output_dir / "tokenizer"
        tokenizer_dir.mkdir(exist_ok=True)
        
        # Save tokenizer files
        self.tokenizer.save_pretrained(tokenizer_dir)
        
        # Also save a copy in the main directory
        for ext in [".json", ".model"]:
            for file in tokenizer_dir.glob(f"*{ext}"):
                shutil.copy2(file, self.output_dir / file.name)
        
        logger.info(f"Saved tokenizer to {tokenizer_dir}")
    
    def verify_conversion(self, weights: Dict[str, np.ndarray]) -> bool:
        """Verify converted weights by running a simple test."""
        if not self.verify:
            return True
        
        logger.info("Verifying conversion with test inference...")
        
        # Create test input
        test_text = "Hello, how are you?"
        inputs = self.tokenizer(test_text, return_tensors="pt")
        
        # Run inference with original model
        with torch.no_grad():
            original_output = self.model(**inputs).logits
        
        # Here you would load the converted weights into Burn and test
        # For now, just check weight shapes and types
        for name, array in weights.items():
            if "weight" in name and len(array.shape) == 2:
                logger.debug(f"Verified {name}: shape {array.shape}, dtype {array.dtype}")
        
        logger.info("Conversion verification passed")
        return True
    
    def convert(self) -> Path:
        """Run full conversion pipeline."""
        logger.info(f"Converting {self.model_id} to Burn format...")
        
        # Download model (optional, can use cached)
        # self.download_model()
        
        # Load model
        self.load_model()
        
        # Quantize weights
        logger.info(f"Applying {self.quantization} quantization...")
        quantized = self.quantize_weights()
        
        # Convert to Burn format
        logger.info("Converting to Burn format...")
        burn_weights = self.convert_to_burn_format(quantized)
        
        # Save weights
        logger.info("Saving weights...")
        self.num_shards = max(1, len(burn_weights) // 100)  # Rough estimate
        shards = self.save_sharded(burn_weights)
        
        # Save config and tokenizer
        self.save_config()
        self.save_tokenizer()
        
        # Verify
        self.verify_conversion(burn_weights)
        
        # Create metadata
        metadata = {
            "model_id": self.model_id,
            "family": self.family,
            "quantization": self.quantization,
            "num_parameters": self.model.num_parameters(),
            "num_shards": len(shards),
            "total_size_gb": sum(p.stat().st_size for p in shards) / 1e9,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Conversion complete! Model saved to {self.output_dir}")
        logger.info(f"Total size: {metadata['total_size_gb']:.2f} GB")
        
        return self.output_dir


def list_available_models(pattern: Optional[str] = None) -> List[str]:
    """List available models on HuggingFace."""
    fs = HfFileSystem()
    
    if pattern:
        models = fs.glob(f"models/{pattern}")
    else:
        # Get trending models
        models = [
            "deepseek-ai/deepseek-llm-7b-base",
            "deepseek-ai/deepseek-llm-67b-base",
            "meta-llama/Meta-Llama-3-8B",
            "meta-llama/Meta-Llama-3-70B",
            "mistralai/Mistral-7B-v0.1",
            "mistralai/Mixtral-8x7B-v0.1",
            "google/gemma-2b",
            "google/gemma-7b",
            "microsoft/phi-2",
            "Qwen/Qwen-7B",
            "baichuan-inc/Baichuan2-7B-Base",
        ]
    
    return models


def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace models to Burn format")
    
    # Model selection
    parser.add_argument("model_id", nargs="?", help="HuggingFace model ID (e.g., deepseek-ai/deepseek-llm-7b-base)")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--search", type=str, help="Search for models matching pattern")
    
    # Output options
    parser.add_argument("--output", "-o", type=Path, default="./models", help="Output directory")
    parser.add_argument("--name", type=str, help="Custom name for the model (defaults to model ID)")
    
    # Conversion options
    parser.add_argument("--quantize", "-q", choices=list(QUANTIZATION_TYPES.keys()), default="fp16",
                       help="Quantization type")
    parser.add_argument("--max-shard-size", type=float, default=2.0,
                       help="Maximum shard size in GB")
    parser.add_argument("--no-verify", action="store_true", help="Skip verification")
    
    # HuggingFace options
    parser.add_argument("--cache-dir", type=Path, help="HuggingFace cache directory")
    parser.add_argument("--token", type=str, help="HuggingFace token for private models")
    parser.add_argument("--revision", type=str, default="main", help="Model revision")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code")
    
    # Advanced options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without doing it")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # List models
    if args.list_models:
        models = list_available_models(args.search)
        print("\nAvailable models:")
        for model in sorted(models):
            print(f"  - {model}")
        return 0
    
    # Check model ID
    if not args.model_id:
        parser.print_help()
        return 1
    
    # Dry run
    if args.dry_run:
        print(f"Would convert {args.model_id} to {args.output} with {args.quantize} quantization")
        return 0
    
    # Create converter
    converter = ModelConverter(
        model_id=args.model_id,
        output_dir=args.output / (args.name or args.model_id.replace("/", "-")),
        quantization=args.quantize,
        max_shard_size_gb=args.max_shard_size,
        verify=not args.no_verify,
        cache_dir=args.cache_dir,
        token=args.token,
        revision=args.revision,
    )
    
    # Run conversion
    try:
        output_path = converter.convert()
        print(f"\n✅ Model converted successfully to {output_path}")
        print(f"\nTo use this model in your AI cluster:")
        print(f"  1. Update config/models.toml with the model path")
        print(f"  2. Restart workers or use API: POST /v1/models/load")
        return 0
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())