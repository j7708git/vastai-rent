#!/bin/bash
# vast.ai 開機自動佈建：下載 Qwen-Image-2.1 Turbo (int8) 模型組 + 更新 ComfyUI
#
# 由 vast.ai 模板環境變數 PROVISIONING_SCRIPT 指向此檔的 raw URL，於容器啟動時執行。
# 模型來源：https://huggingface.co/Comfy-Org/Qwen-Image-2.1（公開 repo，免 token）
#
# 下載清單（Turbo int8 最小集，約 17.3 GB）：
#   diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors  (7.26 GB)
#   text_encoders/qwen3vl_8b_int8_convrot.safetensors               (9.35 GB)
#   vae/qwen_image_2.1_vae_bf16.safetensors                         (0.68 GB)
#
# 查詢日期 2026-10-10；檔名以 HF API 核對過。
set -o pipefail

log() { echo "[provision] $*"; }

HF_BASE="https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main"

# ------------------------------------------------------- locate ComfyUI root
COMFY_DIR=""
for d in /workspace/ComfyUI /opt/ComfyUI; do
  if [ -d "$d" ]; then COMFY_DIR="$d"; break; fi
done
if [ -z "$COMFY_DIR" ] && [ -d /workspace/storage/stable_diffusion ]; then
  COMFY_DIR="/opt/ComfyUI"
fi
if [ -z "$COMFY_DIR" ]; then
  log "error: ComfyUI directory not found (/workspace/ComfyUI or /opt/ComfyUI)"
  exit 1
fi
log "ComfyUI root: $COMFY_DIR"

MODELS="$COMFY_DIR/models"
mkdir -p "$MODELS/diffusion_models" "$MODELS/text_encoders" "$MODELS/vae"

# ------------------------------------------------------------------ downloads
# wget -c 支援斷點續傳；下載中會有 .part 之類暫存，完成後檔案大小才正確。
download() {
  local dest_dir="$1"
  local filename="$2"
  local url="$HF_BASE/$1/$filename"
  if [ -s "$dest_dir/$filename" ]; then
    log "already present, skip: $filename"
    return 0
  fi
  log "downloading $filename"
  wget -c -q --show-progress -P "$dest_dir" "$url"
  local rc=$?
  if [ $rc -ne 0 ]; then
    log "error: download failed (rc=$rc): $filename"
    return 1
  fi
  log "downloaded: $dest_dir/$filename"
}

FAILED=0
download diffusion_models qwen_image_2.1_turbo_int8_convrot.safetensors || FAILED=1
download text_encoders    qwen3vl_8b_int8_convrot.safetensors              || FAILED=1
download vae              qwen_image_2.1_vae_bf16.safetensors              || FAILED=1

# ------------------------------------------------------- update ComfyUI code
# Qwen-Image-2.1 節點（TextEncodeQwenImage21 / QwenImage21Cache）需要
# ComfyUI >= v0.37.0（2026-09-21）。模板內建版本可能較舊，這裡拉到最新。
cd "$COMFY_DIR" || exit 1
if [ -d .git ]; then
  log "updating ComfyUI (need >= v0.37.0 for Qwen-Image-2.1 nodes)"
  if git pull --ff-only 2>/dev/null; then
    log "ComfyUI updated: $(git rev-parse --short HEAD 2>/dev/null)"
  else
    log "warning: git pull failed (local changes?), keeping current version"
  fi
  # 更新後重裝依賴（節點可能需要新套件）
  if [ -f requirements.txt ]; then
    if command -v python3 >/dev/null 2>&1; then
      python3 -m pip install --no-cache-dir -r requirements.txt \
        2>/dev/null || log "warning: pip install failed (non-fatal)"
    fi
  fi
  # 讓 ComfyUI 以新版程式碼重啟（supervisor 會自動拉起；找不到就交給下次重啟）
  if command -v supervisorctl >/dev/null 2>&1; then
    supervisorctl restart comfyui 2>/dev/null \
      || pkill -f "ComfyUI/main.py" 2>/dev/null \
      || true
    log "ComfyUI restart requested"
  else
    pkill -f "ComfyUI/main.py" 2>/dev/null || true
  fi
else
  log "warning: $COMFY_DIR is not a git repo, cannot update ComfyUI"
fi

if [ "$FAILED" -ne 0 ]; then
  log "provisioning finished WITH ERRORS (some downloads failed)"
  exit 1
fi

log "provisioning complete"
