#!/usr/bin/env bash
# Downloads diffusion model sets into $SD_MODELS_DIR. See docs/diffusion.md.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO_ROOT/.env" ] && set -a && . "$REPO_ROOT/.env" && set +a
DEST="${SD_MODELS_DIR:-$REPO_ROOT/models/diffusion}"

HF="https://huggingface.co"
WAN21="$HF/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files"

# set name -> "subdir|repo path|...";  one entry per file.
declare -A SETS=(
  [realvis]="\
.|SG161222/RealVisXL_V5.0/resolve/main/RealVisXL_V5.0_fp16.safetensors"
  [juggernaut]="\
.|RunDiffusion/Juggernaut-XL-v9/resolve/main/juggernautXL_v9Rundiffusionphoto2.safetensors"
  [wan22-i2v]="\
diffusion_models|QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/main/HighNoise/Wan2.2-I2V-A14B-HighNoise-Q8_0.gguf
diffusion_models|QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/main/LowNoise/Wan2.2-I2V-A14B-LowNoise-Q8_0.gguf
text_encoders|city96/umt5-xxl-encoder-gguf/resolve/main/umt5-xxl-encoder-Q8_0.gguf
vae|__WAN21__/vae/wan_2.1_vae.safetensors"
  [wan22-lora]="\
loras|lightx2v/Wan2.2-Distill-Loras/resolve/main/wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors
loras|lightx2v/Wan2.2-Distill-Loras/resolve/main/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors"
)

usage() {
  echo "Usage: $(basename "$0") <set>...   sets: ${!SETS[*]}"
  echo "       $(basename "$0") --list"
}

[ $# -eq 0 ] && { usage >&2; exit 1; }
[ "$1" = "--list" ] && { for k in "${!SETS[@]}"; do echo "$k"; done | sort; exit 0; }

for set_name in "$@"; do
  [ -n "${SETS[$set_name]:-}" ] || { echo "unknown set: $set_name" >&2; usage >&2; exit 1; }
  while IFS='|' read -r subdir path; do
    [ -n "$path" ] || continue
    url="${path/__WAN21__/$WAN21}"
    [ "$url" = "$path" ] && url="$HF/$path"
    out="$DEST/$subdir/$(basename "$path")"
    mkdir -p "$(dirname "$out")"
    if [ -f "$out" ]; then
      echo "have $(basename "$out")"
      continue
    fi
    echo "get  $(basename "$out")"
    curl -fL --retry 3 --progress-bar -o "$out.part" "$url"
    mv "$out.part" "$out"
  done <<< "${SETS[$set_name]}"
done

echo "done -> $DEST"
