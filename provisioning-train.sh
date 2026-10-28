#!/bin/bash
# vast.ai 開機自動佈建（訓練模式）：安裝 ostris/ai-toolkit、下載 Qwen-Image-2.1
# base int8 訓練模型與官方設定檔、生成起手 YAML、建立資料集目錄、啟動訓練 UI。
#
# 由 vast.ai 模板環境變數 PROVISIONING_SCRIPT 指向此檔的 raw URL，於容器開機時執行
# （vastai/base-image 家族：Supervisor 啟動後的 Phase 9）。對應租機入口：
#   python3 scripts/rent_gpu.py --mode train
#
# 模型來源（查詢日期 2026-10-11；檔名以 HF 官方 repo 核對過，與研究筆記一致）：
#   Comfy-Org/Qwen-Image-2.1（權重，int8 量化，訓練用 base 非 turbo）：
#     https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/diffusion_models/qwen_image_2.1_int8_convrot.safetensors  (約 7.26 GB)
#     https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/qwen3vl_8b_int8_convrot.safetensors      (約 9.35 GB)
#     https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors                (約 0.68 GB)
#   Qwen/Qwen-Image-2.1（僅設定檔／processor，Diffusers 全精度權重不必下載）：
#     processor/*  */config.json  text_encoder/generation_config.json
#     scheduler/scheduler_config.json  model_index.json
#
# 可用環境變數覆寫（皆有預設）：
#   AI_TOOLKIT_DIR  AI_TOOLKIT_REF  AI_TOOLKIT_REPO  HF_ROOT  DATASET_DIR
#   TRAIN_NAME  TRIGGER_WORD  TRAIN_RESOLUTION  TRAIN_STEPS  TRAIN_RANK  TRAIN_LR
#   TRAIN_GRAD_CP  TRAIN_LAYER_OFFLOAD  TORCH_INDEX_URL  AI_TOOLKIT_UI_PORT
#   TRAIN_DISABLE_SAMPLING  TRAIN_PERF_LOG_EVERY  TRAIN_LOG_EVERY
#   AITK_REGISTER_UI_JOB  AITK_REGISTER_UI_JOB_URL  AITK_REGISTER_WAIT  REGISTER_TOOL
#
# 註：本腳本由 vast.ai 模板以 raw URL 取得，執行時才知道 repo 內容；因此腳本會
# 用「同一個 repo 的 raw URL」再取得 UI 註冊工具（scripts/register_ui_job.py）。
# 兩者都必須已 push 到 GitHub，佈建才拿得到最新版。
set -o pipefail

log()  { echo "[provision-train] $*"; }
fail() { log "error: $*"; exit 1; }

AI_TOOLKIT_DIR="${AI_TOOLKIT_DIR:-/workspace/ai-toolkit}"
AI_TOOLKIT_REPO="${AI_TOOLKIT_REPO:-https://github.com/ostris/ai-toolkit.git}"
AI_TOOLKIT_REF="${AI_TOOLKIT_REF:-}"          # 空=main；可填 commit/tag 固定版本
HF_ROOT="${HF_ROOT:-/workspace/hf}"
DATASET_DIR="${DATASET_DIR:-/workspace/dataset}"
AI_TOOLKIT_UI_PORT="${AI_TOOLKIT_UI_PORT:-8675}"

# 起手 YAML 參數（研究筆記 §D 的第三方 12GB 實測設定為底）
TRAIN_NAME="${TRAIN_NAME:-qwen21_character_lora}"
TRIGGER_WORD="${TRIGGER_WORD:-mychar}"
TRAIN_RESOLUTION="${TRAIN_RESOLUTION:-1024}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"
TRAIN_RANK="${TRAIN_RANK:-16}"
TRAIN_LR="${TRAIN_LR:-0.0001}"
# 記憶體策略開關（見 YAML 內註解；兩者同時開啟是 issue #1054 的高風險組合）。
TRAIN_GRAD_CP="${TRAIN_GRAD_CP:-true}"             # → train.gradient_checkpointing
TRAIN_LAYER_OFFLOAD="${TRAIN_LAYER_OFFLOAD:-true}"  # → model.layer_offloading（含兩個 *_percent）
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
TORCH_PKGS="${TORCH_PKGS:-torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0}"

# 網頁可見性相關（見 docs/qwen21-lora-training.md「在網頁上跑訓練」）：
TRAIN_DISABLE_SAMPLING="${TRAIN_DISABLE_SAMPLING:-false}"    # true=完全不產生預覽圖（16GB 卡可開）
TRAIN_PERF_LOG_EVERY="${TRAIN_PERF_LOG_EVERY:-1}"            # >0 才會寫 speed_string（網頁速度欄）
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-1}"                      # logging.log_every（loss 圖取樣間隔）

# 佈建完成後把起手 YAML 註冊成 UI 任務（讓使用者可在網頁按 ▶ 開始並監看）：
AITK_REGISTER_UI_JOB="${AITK_REGISTER_UI_JOB:-1}"            # 設 0 可完全關閉
AITK_REGISTER_UI_JOB_URL="${AITK_REGISTER_UI_JOB_URL:-https://raw.githubusercontent.com/j7708git/vastai-rent/main/scripts/register_ui_job.py}"
AITK_REGISTER_WAIT="${AITK_REGISTER_WAIT:-300}"              # 等 UI HTTP 就緒的上限秒數
REGISTER_TOOL="${REGISTER_TOOL:-/workspace/register_ui_job.py}"

YAML_PATH="$AI_TOOLKIT_DIR/config/train_qwen21_character_lora.yaml"
TRAIN_RUNNER="$AI_TOOLKIT_DIR/train.sh"

# huggingface 快取放在 workspace（開機後不隨容器重來；房量也較大）
export HF_HOME="$HF_ROOT"
export HF_HUB_CACHE="$HF_ROOT/hub"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$HF_ROOT" "$DATASET_DIR" /workspace
log "ai-toolkit: $AI_TOOLKIT_DIR"
log "HF cache  : $HF_ROOT"
log "dataset   : $DATASET_DIR"

# ------------------------------------------------------- node / npm（UI 需要）
# 官方 vastai/pytorch 映像「沒有」內建 Node（2026-10-11 實測）。
# 策略：先找系統／NVM 的 npm，找不到就從 nodejs.org 抓 LTS tarball 自裝。
ensure_npm() {
  [ -x /opt/node/bin/npm ] && { export PATH="/opt/node/bin:$PATH"; return 0; }
  command -v npm >/dev/null 2>&1 && return 0
  local d
  for d in "${NVM_DIR:-}" /root/.nvm /usr/local/nvm "$HOME/.nvm"; do
    if [ -n "$d" ] && [ -s "$d/nvm.sh" ]; then
      # shellcheck disable=SC1090
      . "$d/nvm.sh"
      nvm use --lts >/dev/null 2>&1 || nvm use node >/dev/null 2>&1 || true
      command -v npm >/dev/null 2>&1 && return 0
    fi
  done
  return 1
}
install_node_tarball() {
  # 固定版本求穩定；用 .tar.gz（gzip 必有，xz 不一定有）
  local ver="v22.14.0" arch pkg url
  case "$(uname -m)" in
    x86_64) arch=x64 ;;
    aarch64|arm64) arch=arm64 ;;
    *) log "error: 未知的 CPU 架構 $(uname -m)，無法安裝 Node"; return 1 ;;
  esac
  pkg="node-$ver-linux-$arch"
  url="https://nodejs.org/dist/$ver/$pkg.tar.gz"
  log "下載 Node.js $ver tarball → /opt/node ..."
  mkdir -p /opt/node || return 1
  wget -qO "/tmp/$pkg.tar.gz" "$url" || { log "error: Node tarball 下載失敗 $url"; return 1; }
  tar -xzf "/tmp/$pkg.tar.gz" -C /opt/node --strip-components=1 || return 1
  rm -f "/tmp/$pkg.tar.gz"
  export PATH="/opt/node/bin:$PATH"
  command -v npm >/dev/null 2>&1
}
if ! ensure_npm; then
  install_node_tarball || fail "找不到 npm/Node 且 tarball 安裝失敗（訓練 UI 需要 Node > 20）"
fi
command -v git >/dev/null 2>&1 || {
  log "缺 git，嘗試 apt 安裝..."
  (apt-get update -qq && apt-get install -y -qq git) >/dev/null 2>&1 \
    || fail "缺 git 且 apt 安裝失敗（clone ai-toolkit 需要）"
}
log "node: $(node --version 2>/dev/null) / npm: $(npm --version 2>/dev/null) / git: $(git --version 2>/dev/null)"

# ------------------------------------------------------- clone ai-toolkit
if [ -d "$AI_TOOLKIT_DIR/.git" ]; then
  log "ai-toolkit 已存在，更新中..."
  git -C "$AI_TOOLKIT_DIR" fetch --all --quiet || log "warning: git fetch 失敗（沿用現有版本）"
  if git -C "$AI_TOOLKIT_DIR" pull --ff-only --quiet 2>/dev/null; then
    log "ai-toolkit updated (pull): $(git -C "$AI_TOOLKIT_DIR" rev-parse --short HEAD)"
  else
    log "warning: git pull 失敗（可能有本機修改），沿用現有版本"
  fi
else
  log "clone ai-toolkit..."
  git clone --quiet "$AI_TOOLKIT_REPO" "$AI_TOOLKIT_DIR" \
    || fail "git clone 失敗（b 網路或 repo 不可達）"
fi
if [ -n "$AI_TOOLKIT_REF" ]; then
  log "checkout AI_TOOLKIT_REF=$AI_TOOLKIT_REF"
  git -C "$AI_TOOLKIT_DIR" checkout --quiet "$AI_TOOLKIT_REF" \
    || fail "git checkout $AI_TOOLKIT_REF 失敗"
fi
[ -f "$AI_TOOLKIT_DIR/requirements.txt" ] || fail "ai-toolkit 目錄不完整（缺 requirements.txt）"

# ------------------------------------------------------- venv + torch + deps
VENV="$AI_TOOLKIT_DIR/venv"
if [ ! -x "$VENV/bin/python" ]; then
  log "建立 venv..."
  python3 -m venv "$VENV" || fail "python3 -m venv 失敗（需要 python >=3.10）"
fi
# shellcheck disable=SC1091
. "$VENV/bin/activate"

# 已有 torch 就跳過（重跑佈建時省時間）
if ! python -c "import torch" >/dev/null 2>&1; then
  log "安裝 PyTorch（$TORCH_PKGS，index=$TORCH_INDEX_URL）..."
  pip install --no-cache-dir $TORCH_PKGS --index-url "$TORCH_INDEX_URL" \
    || fail "PyTorch 安裝失敗（檢查 GPU 驅動與 index-url 是否相符）"
else
  log "PyTorch 已存在，跳過：$(python -c 'import torch; print(torch.__version__)')"
fi

log "安裝 ai-toolkit requirements（可能數分鐘）..."
pip install --no-cache-dir -r "$AI_TOOLKIT_DIR/requirements.txt" \
  || fail "requirements.txt 安裝失敗"

# ------------------------------------------------------- 下載訓練模型（HF cache）
# 用 venv 的 hf CLI（huggingface_hub）；無 hf 時退回 huggingface-cli。
HFBIN="$VENV/bin/hf"
[ -x "$HFBIN" ] || HFBIN="$VENV/bin/huggingface-cli"
[ -x "$HFBIN" ] || fail "找不到 hf / huggingface-cli（huggingface_hub 未安裝完成）"

COMFY_REPO="Comfy-Org/Qwen-Image-2.1"
COMFY_BASE="https://huggingface.co/$COMFY_REPO/resolve/main"
COMFY_FILES=(
  "diffusion_models/qwen_image_2.1_int8_convrot.safetensors"
  "text_encoders/qwen3vl_8b_int8_convrot.safetensors"
  "vae/qwen_image_2.1_vae_bf16.safetensors"
)
log "下載訓練權重（int8 三檔，約 17.3GB）..."
for f in "${COMFY_FILES[@]}"; do
  log "  weight: $COMFY_BASE/$f"
  "$HFBIN" download "$COMFY_REPO" "$f" \
    || fail "權重下載失敗: $f（檢查網路／repo 是否仍存在此檔）"
done

BASE_REPO="Qwen/Qwen-Image-2.1"
log "下載官方設定檔／processor（$BASE_REPO）..."
cfg_args=(download "$BASE_REPO")
for pat in "processor/*" "*/config.json" "text_encoder/generation_config.json" \
           "scheduler/scheduler_config.json" "model_index.json"; do
  cfg_args+=(--include "$pat")
done
"$HFBIN" "${cfg_args[@]}" \
  || fail "官方設定檔下載失敗（ai-toolkit 讀取 config/processor 需要）"

# ------------------------------------------------------- 生成起手 YAML
mkdir -p "$AI_TOOLKIT_DIR/config"
log "寫入起手 YAML: $YAML_PATH"

# model.layer_offloading 與兩個 *_percent 一併受 TRAIN_LAYER_OFFLOAD 控制；
# 關閉時不寫 percent，避免留下「看似有開」的誤導值。
if [ "$TRAIN_LAYER_OFFLOAD" = "true" ]; then
  LAYER_OFFLOAD_BLOCK="        layer_offloading: true
        # 0 是關閉、1.0 是 100% 的層（ai-toolkit ModelConfig）。
        layer_offloading_transformer_percent: 1.0
        layer_offloading_text_encoder_percent: 1.0"
else
  LAYER_OFFLOAD_BLOCK="        layer_offloading: false"
fi

cat > "$YAML_PATH" <<YAML
# Qwen-Image-2.1 人物 LoRA 起手設定（ai-toolkit sd_trainer）
# 由 vast.ai 佈建腳本自動生成；數值以第三方 RTX 4070 12GB 實測設定為底，請視訓練集調整。
# 用法：cd ai-toolkit && ./train.sh   （等同 python run.py config/train_qwen21_character_lora.yaml）
# 訓練完務必用本專案的 scripts/check_lora.py 檢查產出的 LoRA，確認 lora_B 非零（不是空包彈）。
job: extension
config:
  name: ${TRAIN_NAME}
  process:
    - type: sd_trainer
      training_folder: output
      device: cuda:0
      # 讓網頁看得到「速度」與「loss 曲線」（ai-toolkit 預設兩者都不寫）：
      #   performance_log_every > 0 → 週期性寫入 speed_string（否則網頁速度欄恆為 '?'）
      #   logging.use_ui_logger = true → 寫出 <job>/loss_log.db（網頁 loss 圖的來源）
      performance_log_every: ${TRAIN_PERF_LOG_EVERY}
      logging:
        use_ui_logger: true
        log_every: ${TRAIN_LOG_EVERY}
      # trigger_word：caption 內寫 [trigger] 會被替換成這個詞；若 caption 尚未含觸發詞，
      # 也會自動補上（已含則不重複加）。
      # 註：現行 ai-toolkit（0.13.25 / commit f7a1fb9）在寫入 text-embedding 快取「之前」就已
      # 注入觸發詞，所以即使開了 cache_text_embeddings 仍會生效（舊版警語已過時）。
      # 保留本欄位屬保險做法；若你的 caption 已直接寫入觸發詞，把這行註解掉可避免重複注入。
      trigger_word: "${TRIGGER_WORD}"
      network:
        type: lora
        linear: ${TRAIN_RANK}
        linear_alpha: ${TRAIN_RANK}
      save:
        dtype: float16
        save_every: 250
        max_step_saves_to_keep: 4
      datasets:
        # folder_path 指向資料集目錄：放 jpg/png/jpeg + 同名 .txt（caption）。
        # 不支援 webp。不需自行裁切／縮放，loader 會自動依 aspect ratio 分桶。
        - folder_path: "${DATASET_DIR}"
          caption_ext: txt
          caption_dropout_rate: 0.0
          shuffle_tokens: false
          flip_x: false
          buckets: true
          # square_crop 保持預設 false（故不寫此欄）：true 會跳過 aspect-ratio 分桶、
          # 把每張圖中央裁成 resolution × resolution 的正方形。官方範例與預設皆 false。
          resolution: [${TRAIN_RESOLUTION}]
          cache_latents_to_disk: true
          cache_text_embeddings: true
          num_workers: 0
          cache_latents_num_workers: 1
      train:
        batch_size: 1
        steps: ${TRAIN_STEPS}
        gradient_accumulation: 1
        train_unet: true
        train_text_encoder: false
        # gradient_checkpointing 與 layer_offloading 同時開啟，是 issue #1054 回報會產生
        # 「全零 lora_B 空包彈 LoRA」的組合；但本專案已於實機（RTX 3090 24G）實測「未重現」。
        # 若日後遇到「LoRA 存得出來卻完全沒效果」，先用本專案 scripts/check_lora.py 檢查，
        # 再考慮把 TRAIN_GRAD_CP 設為 false 關掉 checkpointing（VRAM 需求上升）。
        gradient_checkpointing: ${TRAIN_GRAD_CP}
        cache_text_embeddings: true
        noise_scheduler: flowmatch
        timestep_type: shift
        optimizer: adamw8bit
        lr: ${TRAIN_LR}
        dtype: bf16
        skip_first_sample: true
        # disable_sampling: true 會完全不產生預覽圖（網頁就看不到圖）。
        # 預設 false（開啟取樣，每 sample_every 步出一張預覽圖）；16GB 卡想省時間／VRAM
        # 可設 TRAIN_DISABLE_SAMPLING=true 關掉。
        disable_sampling: ${TRAIN_DISABLE_SAMPLING}
      model:
        # 權重來源為 Comfy-Org 的 int8 量化 repack；設定檔／processor 來自官方 repo。
        name_or_path: ${COMFY_REPO}
        extras_name_or_path: ${BASE_REPO}
        arch: qwen_image_2
        quantize: true
        qtype: convrot8
        quantize_te: true
        qtype_te: convrot8
        low_vram: true
        # layer_offloading 與 gradient_checkpointing 同時開啟，是 issue #1054 回報會產生
        # 「全零 lora_B 空包彈 LoRA」的組合；本專案實機（RTX 3090 24G）已實測「未重現」。
        # 用 TRAIN_LAYER_OFFLOAD=false 可一併關閉 layer_offloading 與兩個 *_percent
        # （16GB 卡建議保留 true；24GB 卡可視情況關閉）。
$LAYER_OFFLOAD_BLOCK
      sample:
        sampler: flowmatch
        sample_every: 250
        width: ${TRAIN_RESOLUTION}
        height: ${TRAIN_RESOLUTION}
        prompts:
          - ${TRIGGER_WORD}, portrait, plain background
        neg: ""
        seed: 42
        walk_seed: false
        # guidance_scale 用 1.0：Qwen-Image-2.1 是 CFG-distilled，官方推論用 CFG=1；
        # ai-toolkit 的 sample pipeline 在 >1 時會真的多跑一次 CFG forward。
        guidance_scale: 1.0
        sample_steps: 40
meta:
  name: "[name]"
  version: "1.0"
YAML
[ -s "$YAML_PATH" ] || fail "YAML 生成失敗（檔案為空）"

# ------------------------------------------------------- 訓練啟動腳本
log "寫入訓練啟動腳本: $TRAIN_RUNNER"
cat > "$TRAIN_RUNNER" <<RUNNER
#!/bin/bash
# 由佈建腳本生成：固定 HF cache 與 CUDA alloc 設定後啟動訓練。
# 用法：./train.sh [config 檔路徑]（預設 config/train_qwen21_character_lora.yaml）
set -euo pipefail
cd "\$(cd "\$(dirname "\$0")" && pwd)"
export HF_HOME="$HF_ROOT"
export HF_HUB_CACHE="$HF_ROOT/hub"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source venv/bin/activate
CONFIG="\${1:-config/train_qwen21_character_lora.yaml}"
exec python run.py "\$CONFIG"
RUNNER
chmod +x "$TRAIN_RUNNER"

# ------------------------------------------------------- UI build（Supervisor 常駐）
log "建置訓練 UI（npm install + build）..."
UI_DIR="$AI_TOOLKIT_DIR/ui"
[ -d "$UI_DIR" ] || fail "找不到 ui 目錄：$UI_DIR"
(
  cd "$UI_DIR" || exit 1
  npm run install_deps || fail "UI npm install 失敗"
  npm run update_db   || fail "UI prisma db 生成失敗"
  npm run build       || fail "UI build 失敗"
) || fail "訓練 UI 建置失敗"

# 把 UI 以 Supervisor 常駐，並保留 nvm 路徑寫進 wrapper
NVM_DIR_FOUND=""
for d in "${NVM_DIR:-}" /root/.nvm /usr/local/nvm "$HOME/.nvm"; do
  if [ -n "$d" ] && [ -s "$d/nvm.sh" ]; then NVM_DIR_FOUND="$d"; break; fi
done

UI_WRAPPER="/opt/supervisor-scripts/ai-toolkit-ui.sh"
if [ -d /opt/supervisor-scripts ]; then
  log "註冊 Supervisor 服務 ai-toolkit-ui"
  cat > "$UI_WRAPPER" <<WRAP
#!/bin/bash
# supervisor 環境沒有佈建 shell 的 PATH；Node 裝在 /opt/node（tarball），必須明確帶入
export PATH="/opt/node/bin:\$PATH"
export HF_HOME="$HF_ROOT"
export HF_HUB_CACHE="$HF_ROOT/hub"
cd "$UI_DIR" || exit 1
exec npm run start
WRAP
  chmod +x "$UI_WRAPPER"
  cat > /etc/supervisor/conf.d/ai-toolkit-ui.conf <<CONF
[program:ai-toolkit-ui]
environment=PROC_NAME="%(program_name)s"
command=$UI_WRAPPER
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
redirect_stderr=true
CONF
  supervisorctl reread >/dev/null 2>&1 || true
  supervisorctl update >/dev/null 2>&1 || true
  supervisorctl restart ai-toolkit-ui >/dev/null 2>&1 \
    || supervisorctl start ai-toolkit-ui >/dev/null 2>&1 \
    || log "warning: supervisorctl 啟動失敗，稍後由 Supervisor 自動拉起"
else
  log "warning: 找不到 /opt/supervisor-scripts，改用 nohup 啟動 UI"
  ( cd "$UI_DIR" && HF_HOME="$HF_ROOT" HF_HUB_CACHE="$HF_ROOT/hub" \
      nohup npm run start >/var/log/ai-toolkit-ui.log 2>&1 & ) || true
fi

# ------------------------------------------------------- 註冊成 UI 任務（防禦性）
# 讓起手 YAML 以「ai-toolkit UI 原生任務」的形式出現在網頁 Jobs 清單：使用者點進任務
# 詳細頁按 ▶ 即開始訓練，並可在網頁看狀態／步數／速度／loss／預覽圖。
# 全程防禦性：UI 沒起來、下載不到工具、或註冊失敗，都只記 warning，**不影響佈建結果**
# （訓練仍可用 ./train.sh）。可用 AITK_REGISTER_UI_JOB=0 完全關閉。
if [ "$AITK_REGISTER_UI_JOB" = "0" ]; then
  log "略過 UI 任務註冊（AITK_REGISTER_UI_JOB=0）"
else
  # 取得註冊工具：與本腳本相同的 raw URL 模式（repo 需已 push）
  if [ ! -s "$REGISTER_TOOL" ]; then
    log "下載 UI 註冊工具：$AITK_REGISTER_UI_JOB_URL"
    wget -qO "$REGISTER_TOOL" "$AITK_REGISTER_UI_JOB_URL" || rm -f "$REGISTER_TOOL"
    # 404 頁面等非腳本內容會有 HTML；確認看起來是 Python 再留著
    if [ -s "$REGISTER_TOOL" ] && ! head -1 "$REGISTER_TOOL" | grep -q '^#!.*python'; then
      log "warning: 下載到的內容不像 Python 腳本，丟棄"
      rm -f "$REGISTER_TOOL"
    fi
  fi
  if [ ! -s "$REGISTER_TOOL" ]; then
    log "warning: 取不到 $REGISTER_TOOL，略過 UI 任務註冊（訓練仍可用 ./train.sh）"
  else
    log "等 UI ($AI_TOOLKIT_UI_PORT) 就緒後註冊任務（上限 ${AITK_REGISTER_WAIT}s）..."
    ui_up=0
    waited=0
    while [ "$waited" -lt "$AITK_REGISTER_WAIT" ]; do
      code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 \
             "http://127.0.0.1:$AI_TOOLKIT_UI_PORT/api/jobs" 2>/dev/null || true)
      [ "$code" = "200" ] && { ui_up=1; break; }
      sleep 5
      waited=$((waited + 5))
    done
    if [ "$ui_up" != "1" ]; then
      log "warning: UI 在 ${AITK_REGISTER_WAIT}s 內未就緒，略過 UI 任務註冊（訓練仍可用 ./train.sh）"
    elif "$VENV/bin/python" "$REGISTER_TOOL" --yaml "$YAML_PATH" --name "$TRAIN_NAME" \
           --api-url "http://127.0.0.1:$AI_TOOLKIT_UI_PORT"; then
      log "已註冊 UI 任務：$TRAIN_NAME（網頁 Jobs 可見；點進詳細頁按 ▶ 開始訓練）"
    else
      log "warning: UI 任務註冊失敗（退出碼 $?），訓練仍可用 ./train.sh"
    fi
  fi
fi

log "佈建完成。UI 應在 http://127.0.0.1:$AI_TOOLKIT_UI_PORT（經 Instance Portal / SSH 埠轉發對外）"
log "訓練：cd $AI_TOOLKIT_DIR && ./train.sh"
log "資料集：$DATASET_DIR（放 jpg/png + 同名 .txt）"
log "provisioning complete"
