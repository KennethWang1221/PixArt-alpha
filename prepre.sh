#!/usr/bin/env bash
set -euo pipefail

# One-shot project bootstrap for Way1 + Way2.
# - Creates .venv with uv (if missing)
# - Installs compatible packages for train_pixart_lora_hf.py
# - Clones local datasets from Hugging Face git repos
# - Prepares Flickr imagefolder data with metadata.jsonl

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

log() { printf "[prepre] %s\n" "$*"; }
warn() { printf "[prepre][warn] %s\n" "$*"; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

require_cmd git
require_cmd uv

if [[ ! -d ".venv" ]]; then
  log "Creating uv virtual environment at .venv"
  uv venv .venv
fi

PY_BIN=".venv/bin/python"

log "Installing compatible core dependencies into .venv"
uv pip install --python "$PY_BIN" -U \
  "torch==2.1.1" \
  "torchvision==0.16.1" \
  "torchaudio==2.1.1" \
  "transformers==4.36.2" \
  "huggingface-hub==0.24.6" \
  "diffusers==0.25.0" \
  "accelerate==0.25.0" \
  "datasets==2.16.1" \
  "peft==0.6.2" \
  "sentencepiece==0.1.99" \
  "tensorboard" \
  "ftfy" \
  "beautifulsoup4"

if [[ ! -d "pokemon-blip-captions/.git" ]]; then
  log "Cloning Way1 dataset: lambdalabs/pokemon-blip-captions"
  git clone "git@hf.co:datasets/lambdalabs/pokemon-blip-captions" "pokemon-blip-captions"
else
  log "Dataset already exists: pokemon-blip-captions"
fi

if [[ ! -d "flickr30k/.git" ]]; then
  log "Cloning Way2 dataset: nlphuji/flickr30k"
  git clone "git@hf.co:datasets/nlphuji/flickr30k" "flickr30k"
else
  log "Dataset already exists: flickr30k"
fi

if command -v git-lfs >/dev/null 2>&1; then
  log "Ensuring LFS files are available for datasets"
  git lfs install --skip-smudge >/dev/null 2>&1 || true
  git -C "pokemon-blip-captions" lfs pull || warn "git lfs pull failed for pokemon-blip-captions"
  git -C "flickr30k" lfs pull || warn "git lfs pull failed for flickr30k"
else
  warn "git-lfs not found. Install git-lfs if dataset zip/large files are missing."
fi

log "Preparing local Flickr imagefolder dataset at data/flickr1k_local"
"$PY_BIN" "./tools/prepare_flickr1k_imagefolder.py" \
  --flickr_root "./flickr30k" \
  --out_dir "./data/flickr1k_local" \
  --split "train" \
  --count 1000

log "Bootstrap complete."
log "Way1 run:"
echo "  ./.venv/bin/python ./train_scripts/train_pixart_lora_hf.py --dataset_name ./pokemon-blip-captions"
log "Way2 run:"
echo "  ./.venv/bin/python ./train_scripts/train_pixart_lora_hf.py --dataset_name imagefolder --train_data_dir ./data/flickr1k_local --image_column image --caption_column text"
