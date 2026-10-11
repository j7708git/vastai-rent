#!/bin/bash
# vast.ai 開機自動佈建：下載 Qwen-Image-2.1 Turbo (int8) 模型組 + 更新 ComfyUI
#
# 由 vast.ai 模板環境變數 PROVISIONING_SCRIPT 指向此檔的 raw URL，於容器啟動時執行。
# 模型來源：https://huggingface.co/Comfy-Org/Qwen-Image-2.1（公開 repo，免 token）
#
# 下載方式：優先 huggingface_hub（Xet 高速傳輸、續傳、自動驗 sha），
#           映像缺套件時自動補裝，裝不起來退回 wget -c。
#
# 下載清單（Turbo int8 最小集，約 17.3 GB）：
#   diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors  (7.26 GB)
#   text_encoders/qwen3vl_8b_int8_convrot.safetensors               (9.35 GB)
#   vae/qwen_image_2.1_vae_bf16.safetensors                         (0.68 GB)
# 另加自訂 LoRA：HF_LORA_REPO 的 loras/qwen2.1/NSFW Qwen by TheseAlpacas V2.safetensors
#   -> models/loras/qwen 2.1/（LoRA 失敗僅警告，不擋佈建）
#
# 查詢日期 2026-10-10；檔名以 HF API 核對過。
# HF 下載改版 2026-10-11：已在 vastai/comfy 映像實測（0.68GB ≈ 2 Gb/s、etag 續傳/跳過）。
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
# 優先用 huggingface_hub（Xet 高速傳輸、斷點續傳、下載後自動驗 sha），
# 映像沒有時自動補裝；裝不起來才退回 wget -c。
# HF_TOKEN／HF_HOME 由 vast.ai 模板環境變數帶入容器，hub 會自動讀取（本 repo 公開，免 token 也能抓）。
HF_REPO_ID="Comfy-Org/Qwen-Image-2.1"
# 自訂人物 LoRA（私有 repo；容器內的 HF_TOKEN 由 vast.ai 模板環境變數帶入）
HF_LORA_REPO="dennygod/comfyui-models"

# wget -c 支援斷點續傳。dest 為 "$MODELS/<類別>"（絕對路徑），不要用相對路徑。
download_wget() {
  local category="$1"
  local filename="$2"
  local dest_dir="$MODELS/$category"
  local url="$HF_BASE/$category/$filename"
  if [ -s "$dest_dir/$filename" ]; then
    log "already present, skip: $filename"
    return 0
  fi
  log "downloading $filename -> $dest_dir"
  wget -c -q --show-progress -P "$dest_dir" "$url"
  local rc=$?
  if [ $rc -ne 0 ]; then
    log "error: download failed (rc=$rc): $filename"
    return 1
  fi
  log "downloaded: $dest_dir/$filename"
}

# 偵測／補裝 huggingface_hub；Python 3.12 externally-managed 環境退一步用
# --break-system-packages（pip 舊版不認得此旗標會失敗，由呼叫端走 wget fallback）。
ensure_hf_py() {
  command -v python3 >/dev/null 2>&1 || return 1
  python3 -c "import huggingface_hub" 2>/dev/null && return 0
  log "huggingface_hub 未預裝，補裝中（huggingface_hub + hf_xet）..."
  python3 -m pip install --quiet --no-cache-dir huggingface_hub hf_xet 2>/dev/null \
    || python3 -m pip install --quiet --no-cache-dir --break-system-packages \
         huggingface_hub hf_xet 2>/dev/null
  python3 -c "import huggingface_hub" 2>/dev/null
}

FAILED=0
if ensure_hf_py; then
  # Xet 高效能傳輸（hub 2.x 起取代 hf_transfer）；無 hf_xet 時僅是多餘變數，無害
  export HF_XET_HIGH_PERFORMANCE=1
  log "downloading via huggingface_hub (Xet high-performance)..."
  python3 - "$MODELS" "$HF_REPO_ID" "$HF_LORA_REPO" <<'PY' || FAILED=1
import os
import shutil
import sys
import tempfile

from huggingface_hub import hf_hub_download

models_dir, qwen_repo, lora_repo = sys.argv[1], sys.argv[2], sys.argv[3]
# (repo_id, HF 內路徑, models 下相對目的路徑)
# 目的路徑含空白是有意的：ComfyUI 選單會顯示成 qwen 2.1/<檔名>
entries = [
    (qwen_repo, "diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors",
     "diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors"),
    (qwen_repo, "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
     "text_encoders/qwen3vl_8b_int8_convrot.safetensors"),
    (qwen_repo, "vae/qwen_image_2.1_vae_bf16.safetensors",
     "vae/qwen_image_2.1_vae_bf16.safetensors"),
    # 自訂人物 LoRA（私有 repo，需容器內有可用的 HF_TOKEN）
    (lora_repo, "loras/qwen2.1/NSFW Qwen by TheseAlpacas V2.safetensors",
     "loras/qwen 2.1/NSFW Qwen by TheseAlpacas V2.safetensors"),
]

failed = False
for repo_id, hf_path, dest_rel in entries:
    dest = os.path.join(models_dir, dest_rel)
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        print(f"[provision] already present, skip: {dest_rel}")
        continue
    print(f"[provision] downloading (hf): {hf_path}", flush=True)
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if hf_path == dest_rel:
            # 目的與 HF 路徑相同：直接落到 models 目錄（享受 local_dir 續傳）
            hf_hub_download(repo_id=repo_id, filename=hf_path, local_dir=models_dir)
        else:
            # 目的不同（如 LoRA）：先抓到暫存目錄再搬到定位
            tmp_dir = tempfile.mkdtemp(dir=models_dir, prefix=".hf-dl-")
            try:
                got = hf_hub_download(repo_id=repo_id, filename=hf_path,
                                      local_dir=tmp_dir)
                shutil.move(got, dest)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"[provision] downloaded: {dest}", flush=True)
    except Exception as exc:  # noqa: BLE001 - 佈建腳本要的是明確錯誤訊息
        # 主模型三檔失敗要讓佈建整體失敗；LoRA 屬加購品，失敗只警告不擋開機
        if hf_path.startswith("loras/"):
            print(f"[provision] warning: LoRA 下載失敗（不擋佈建）{dest_rel}: {exc}",
                  flush=True)
        else:
            print(f"[provision] error: hf download failed {hf_path}: {exc}",
                  flush=True)
            failed = True

sys.exit(1 if failed else 0)
PY
else
  log "huggingface_hub 無法使用，退回 wget -c"
  download_wget diffusion_models qwen_image_2.1_turbo_int8_convrot.safetensors || FAILED=1
  download_wget text_encoders    qwen3vl_8b_int8_convrot.safetensors              || FAILED=1
  download_wget vae              qwen_image_2.1_vae_bf16.safetensors              || FAILED=1
fi

# ------------------------------------------------------- update ComfyUI code
# Qwen-Image-2.1 節點（TextEncodeQwenImage21 / QwenImage21Cache）需要
# ComfyUI >= v0.37.0（2026-09-21）。映像內建可能為 v0.30.0（舊 commit 不在上游
# 歷史線上，git pull --ff-only 會失敗），這裡先 ff-only 再 hard reset 對齊。
cd "$COMFY_DIR" || exit 1
if [ -d .git ]; then
  log "updating ComfyUI (need >= v0.37.0 for Qwen-Image-2.1 nodes)"
  if git pull --ff-only >/dev/null 2>&1; then
    log "ComfyUI updated (pull): $(git rev-parse --short HEAD 2>/dev/null)"
  else
    # 映像 commit 不在 origin/master 歷史上時，ff-only 必失敗；
    # 佈建機器是全新容器，hard reset 對齊上游是安全作法
    if git fetch origin master >/dev/null 2>&1 && git reset --hard origin/master >/dev/null 2>&1; then
      log "ComfyUI updated (reset): $(git rev-parse --short HEAD 2>/dev/null)"
    else
      log "warning: cannot update ComfyUI (fetch/reset failed), keeping current version"
    fi
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
