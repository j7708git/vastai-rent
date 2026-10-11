#!/usr/bin/env python3
"""
vast.ai 一鍵租機腳本（Qwen-Image-2.1）

兩種工作負載，用 `--mode` 切換（預設 inference，行為不變）：
  * inference：租 ComfyUI 機跑 Qwen-Image-2.1-Turbo 推論（現有流程）。
  * train    ：租 PyTorch 機裝 ai-toolkit，訓練 Qwen-Image-2.1 人物 LoRA。

流程: 掃市場 → 依條件篩選 → 用對應模板租下（關閉對外驗證）→ 等開機 →
      等佈建完成（inference：模型檔＋ComfyUI 節點；train：ai-toolkit＋模型檔＋
      訓練 UI 埠）→ 抓對外 URL → 回報

用法:
  python3 scripts/rent_gpu.py                          # 推理模式，租最便宜且符合條件的機器
  python3 scripts/rent_gpu.py --mode train             # 訓練模式（16G+ 消費級卡、磁碟 80G）
  python3 scripts/rent_gpu.py --gpu RTX_3090           # 指定 GPU 型號
  python3 scripts/rent_gpu.py --min-vram 24000         # 最低 VRAM (MiB)
  python3 scripts/rent_gpu.py --dry-run                # 只掃描不租
  python3 scripts/rent_gpu.py --destroy 49693360       # 刪除實例(停止計費)

外部連線（SillyTavern 直連 / 訓練 GUI）:
  建立實例時會設定環境變數關閉模板的登入驗證：
    WEB_ENABLE_AUTH=false  ENABLE_AUTH=false  WEB_ENABLE_HTTPS=false
  （兩套命名都設，涵蓋 ai-dock 與 vastai/base-image 兩種 image 家族）
"""
import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

MODE_INFERENCE = "inference"
MODE_TRAIN = "train"

# ---- 推理模式（--mode inference，預設；行為不變）----
# 官方 vastai/comfy 互動式模板（2026-09-24 版，用 `vastai search templates` 查得；
# 內建 portal + Cloudflare tunnel，PROVISIONING_SCRIPT/COMFYUI_ARGS 可用環境變數覆寫）。
# 可用 --template 覆寫。
DEFAULT_TEMPLATE_HASH = "9e4091bea3fc83cd55070f37cc1b5bed"
# 佈建腳本（本 repo 的 provisioning.sh，需已 push 到 GitHub 才會生效）
PROVISIONING_URL = ("https://raw.githubusercontent.com/j7708git/vastai-rent/"
                    "main/provisioning.sh")
# ComfyUI 對外埠（模板慣例 18188）
COMFY_PORT = 18188
# Qwen-Image-2.1 支援自 ComfyUI v0.37.0，/object_info 應該要能看到這個節點
QWEN21_NODE = "TextEncodeQwenImage21"
DEFAULT_DISK = 60  # GB；模型 17.3GB + ComfyUI 環境，開機後不可再調整
DEFAULT_MIN_VRAM = 12000  # MiB；Qwen-2.1-Turbo 12G 實測可跑

# ---- 訓練模式（--mode train）----
# 官方 PyTorch (Vast) 模板（2026-01-23 版，`vastai search templates` 查得；
# 映像 vastai/pytorch，屬 vastai/base-image 家族，支援 PROVISIONING_SCRIPT 與
# Instance Portal 的 Cloudflare tunnel）。可用 --template 覆寫。
TRAIN_TEMPLATE_HASH = "b84ca276fa572e949cd7ff43ae5fe855"
# 訓練版佈建腳本（本 repo 的 provisioning-train.sh，需已 push 才會生效）
TRAIN_PROVISIONING_URL = ("https://raw.githubusercontent.com/j7708git/vastai-rent/"
                          "main/provisioning-train.sh")
# ai-toolkit 訓練 UI 埠（官方預設）；訓練腳本在此路徑裝機
AI_TOOLKIT_PORT = 8675
TRAIN_DISK = 80       # GB；模型 ~17.3GB + latent/TE cache + optimizer state，開機後不可調
TRAIN_MIN_VRAM = 16000  # MiB；使用者指定 16G+ 消費級卡（4080/4080S/5070Ti/5080）
# 訓練機上的固定路徑（與 provisioning-train.sh 一致，勿任意改）
TRAIN_AI_TOOLKIT_DIR = "/workspace/ai-toolkit"
TRAIN_HF_ROOT = "/workspace/hf"
TRAIN_DATASET_DIR = "/workspace/dataset"
# Instance Portal 連結（PyTorch (Vast) 模板預設 + AI Toolkit UI）。
# 格式 hostname:external_port:internal_port:path:name；external 由 -p 開放。
TRAIN_PORTAL_CONFIG = ("localhost:1111:11111:/:Instance Portal|"
                       "localhost:8080:18080:/:Jupyter|"
                       "localhost:8080:8080:/terminals/1:Jupyter Terminal|"
                       "localhost:8384:18384:/:Syncthing|"
                       "localhost:6006:16006:/:Tensorboard|"
                       f"localhost:{AI_TOOLKIT_PORT}:{AI_TOOLKIT_PORT}:/:AI Toolkit")

LOG = print


def _default_cli():
    """vastai CLI：環境變數 VASTAI_CLI 覆寫 > PATH 搜尋"""
    return os.environ.get("VASTAI_CLI") or shutil.which("vastai") or "vastai"


def _default_ssh_key():
    """SSH key：環境變數 VAST_SSH_KEY 覆寫 > ~/.ssh/id_ed25519"""
    env = os.environ.get("VAST_SSH_KEY")
    if env:
        return env
    key = Path.home() / ".ssh" / "id_ed25519"
    return key.as_posix()


def _load_api_key():
    """環境沒有 VAST_API_KEY 時，讀取腳本旁 git-ignored 的 vast-api-key.env（SDK 備用）"""
    if os.environ.get("VAST_API_KEY"):
        return
    envf = Path(__file__).resolve().parent.parent / "vast-api-key.env"
    try:
        for line in envf.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line.startswith("VAST_API_KEY="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                if value:
                    os.environ["VAST_API_KEY"] = value
                return
    except OSError:
        pass


VASTAI = _default_cli()
SSH_KEY = _default_ssh_key()


def run_vastai(args_list, timeout=120):
    """執行 vastai CLI，回傳 (exit_code, stdout, stderr)"""
    out = subprocess.run([VASTAI] + args_list,
                         capture_output=True, text=True, timeout=timeout)
    return out.returncode, out.stdout, out.stderr


def search_offers(extra_query=""):
    """抓全量可租+verified 的 offers。

    注意：vastai CLI 的 search offers 預設只回前 64 筆，務必加 --limit。
    """
    q = f"rentable=true verified=true {extra_query}".strip()
    rc, out, err = run_vastai(["search", "offers", q, "--raw", "-o", "dph_total",
                               "--limit", "2000"], timeout=180)
    if rc != 0:
        LOG(f"⚠️ search offers 失敗: {err}")
        return []
    return json.loads(out or "[]")


def ptb(x):
    """$/GB -> $/TB"""
    return (x or 0) * 1000


def filter_offers(offers, max_cost, min_down, min_vram, gpu_names=None,
                  server_gpu=False, min_disk_free=0, min_pcie=0):
    """篩選條件：VRAM、算力、下載頻寬、流量費、PCIe 頻寬、主機可用磁碟"""
    hits = []
    for o in offers:
        if gpu_names and o.get("gpu_name") not in gpu_names:
            continue
        if not server_gpu and not (o.get("gpu_name") or "").startswith("RTX "):
            continue  # 預設只要消費級 RTX 卡
        if o.get("gpu_ram", 0) < min_vram:       # VRAM (MiB)
            continue
        flops = o.get("total_flops", 0) or 0
        dlperf = o.get("dlperf", 0) or 0
        if flops < 25 and dlperf < 25:           # 算力
            continue
        if o.get("inet_down", 0) < min_down:     # 下載頻寬
            continue
        if ptb(o.get("inet_down_cost")) > max_cost:  # 流量費
            continue
        if ptb(o.get("inet_up_cost")) > max_cost:
            continue
        if min_pcie and o.get("pcie_bw", 0) < min_pcie:  # PCIe 沒閹割
            continue
        # 主機可用磁碟：不足時 vast 會「默默縮水」實際配置（2026-10-11 實測
        # 要求 80GB 只給 18.5GB，下載模型時 No space left on device）
        if min_disk_free and o.get("disk_space", 0) < min_disk_free:
            continue
        hits.append(o)
    # 排序：純粹便宜優先（Qwen-Image-2.1-Turbo 12G VRAM 實測可跑，不強制 24G）
    hits.sort(key=lambda o: o.get("dph_total", 0))
    return hits


def build_env_args(mode=MODE_INFERENCE):
    """組出 --env 參數：關閉對外驗證 + 指定佈建腳本。

    vast.ai 的 --env 格式：`-e KEY=value`（環境變數）、`-p 宿主:容器`（埠對應）。
    值內含空白需用單引號包住（vastai 端會做 shell 式切分）。

    inference：套用現有 ComfyUI 模板的參數（行為不變）。
    train：改用 PyTorch (Vast) 模板；除了關閉驗證，另開放 8675 並把 AI Toolkit UI
           加入 Instance Portal，讓 portal 產生對外 tunnel 連結。
    """
    if mode == MODE_TRAIN:
        pairs = [
            ("WEB_ENABLE_AUTH", "false"),      # ai-dock 系
            ("ENABLE_AUTH", "false"),          # vastai/base-image 系
            ("WEB_ENABLE_HTTPS", "false"),
            ("ENABLE_HTTPS", "false"),
            # AI Toolkit UI 埠排除在驗證之外（關閉登入）
            ("AUTH_EXCLUDE", str(AI_TOOLKIT_PORT)),
            ("PORTAL_CONFIG", "'" + TRAIN_PORTAL_CONFIG + "'"),
            ("PROVISIONING_SCRIPT", TRAIN_PROVISIONING_URL),
        ]
        # 開放 AI Toolkit UI 對外埠（portal 需要 external port 已被 -p 開放）
        parts = [f"-p {AI_TOOLKIT_PORT}:{AI_TOOLKIT_PORT}"]
        parts += [f"-e {k}={v}" for k, v in pairs]
        return " ".join(parts)

    pairs = [
        ("WEB_ENABLE_AUTH", "false"),      # ai-dock 系
        ("ENABLE_AUTH", "false"),          # vastai/base-image 系
        ("WEB_ENABLE_HTTPS", "false"),
        # 模板 portal 對外埠 8188 -> 內部 18188，排除在驗證之外（SillyTavern 直連）
        ("AUTH_EXCLUDE", "8188"),
        ("COMFYUI_ARGS",
         # --use-pytorch-cross-attention：Blackwell（算力 12.0）等新卡上，映像內建
         # 的 xformers kernel 不支援（memory_efficient_attention_forward 噴
         # "capability too new"），強制改用 PyTorch 內建 attention。舊卡也可用。
         f"'--disable-auto-launch --port {COMFY_PORT} --enable-cors-header "
         f"--use-pytorch-cross-attention'"),
        ("PROVISIONING_SCRIPT", PROVISIONING_URL),
    ]
    return " ".join(f"-e {k}={v}" for k, v in pairs)


def create_instance(offer_id, disk, template_hash, mode=MODE_INFERENCE):
    """用對應模板建立實例（含關驗證環境變數與佈建腳本）"""
    env_args = build_env_args(mode)
    LOG(f"🚀 建立實例: offer={offer_id} disk={disk}GB template={template_hash[:8]}...")
    if mode == MODE_TRAIN:
        LOG("   環境變數: WEB_ENABLE_AUTH=false ENABLE_AUTH=false ENABLE_HTTPS=false "
            f"AUTH_EXCLUDE={AI_TOOLKIT_PORT} PROVISIONING_SCRIPT=<本 repo>")
        LOG(f"   對外埠: -p {AI_TOOLKIT_PORT}:{AI_TOOLKIT_PORT}（AI Toolkit UI）")
    else:
        LOG("   環境變數: WEB_ENABLE_AUTH=false ENABLE_AUTH=false "
            f"COMFYUI_ARGS=... --port {COMFY_PORT} PROVISIONING_SCRIPT=<本 repo>")
    rc, out, err = run_vastai(
        ["create", "instance", str(offer_id),
         "--template_hash", template_hash,
         "--env", env_args,
         "--disk", str(disk)], timeout=120)
    if rc != 0:
        LOG(f"❌ 建立失敗 (rc={rc}): {err.strip()[:300]}")
        return None
    if not out.strip():
        # rc=0 但沒有回應：多半是伺服端拒絕（訊息在 stderr），列出來判讀
        LOG(f"❌ 建立無回應，stderr: {err.strip()[:300]!r}")
        return None
    LOG(f"  原始回應: {out.strip()[:200]}")
    m = re.search(r"new_contract['\"]?\s*[:=]\s*['\"]?(\d+)", out)
    if m:
        return int(m.group(1))
    try:
        return json.loads(out).get("new_contract")
    except Exception:
        return None


def wait_running(instance_id, timeout=600):
    """等待實例 running"""
    LOG("⏳ 等待實例啟動...")
    start = time.time()
    while time.time() - start < timeout:
        rc, out, err = run_vastai(["show", "instance", str(instance_id), "--raw"],
                                  timeout=60)
        if rc == 0:
            try:
                d = json.loads(out)
                if isinstance(d, list):
                    d = d[0] if d else {}
                status = d.get("actual_status") or d.get("cur_state") or ""
                gpu = d.get("gpu_name", "?")
                dph = d.get("dph_total", 0)
                if status == "running":
                    LOG(f"✅ running! GPU={gpu} ${dph:.3f}/hr")
                    return d
                if status in ("exited", "offline", "unknown", "failed"):
                    LOG(f"❌ 狀態異常: {status}")
                    return None
                LOG(f"  [{int(time.time()-start)}s] status={status}")
            except Exception:
                pass
        time.sleep(15)
    LOG("❌ 等待超時")
    return None


def ssh_run(ssh_cmd, remote_cmd, timeout=40):
    """SSH 執行遠端命令；失敗回傳 (stdout, stderr)，永不拋例外"""
    argv = shlex.split(ssh_cmd) + [remote_cmd]
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return r.stdout or "", r.stderr or ""
    except subprocess.TimeoutExpired:
        return "", "(timeout)"
    except Exception as e:
        return "", str(e)


def make_ssh_cmd(ssh_host, ssh_port):
    return (f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "
            f"-o BatchMode=yes "
            f"-i {SSH_KEY} -p {ssh_port} root@{ssh_host}")


def wait_models_ready(instance_id, info, timeout=1800):
    """SSH 進去等模型下載完成 + ComfyUI 就緒（含 Qwen-2.1 節點檢查），
    回傳 (ssh_cmd, tunnel_url)

    就緒判據：
    - 模型：三個檔案都存在（provisioning log 有 "provisioning complete"）
    - ComfyUI API：本機 /system_stats 可回應
    - Qwen-2.1 支援：/object_info 含 TextEncodeQwenImage21（版本 >= 0.37.0）
    - tunnel：從本機 curl 探測 trycloudflare URL
    """
    ssh_host = info.get("ssh_host") or info.get("public_ipaddr")
    ssh_port = info.get("ssh_port")
    if not ssh_host or not ssh_port:
        rc, out, err = run_vastai(["ssh-url", str(instance_id)], timeout=30)
        m = re.search(r"ssh://root@([^:]+):(\d+)", out or "")
        if not m:
            LOG(f"⚠️ 無法取得 SSH 資訊 (ssh-url 也失敗: {err.strip()})")
            return None, None
        ssh_host, ssh_port = m.group(1), int(m.group(2))

    ssh_cmd = make_ssh_cmd(ssh_host, ssh_port)

    LOG("🔑 附加上本機 SSH key")
    run_vastai(["attach", "ssh", str(instance_id), f"{SSH_KEY}.pub"], timeout=60)
    time.sleep(3)

    LOG(f"⏳ 等模型下載 (~17.3GB) + ComfyUI 啟動 + tunnel 就緒 (可達 30 分鐘)...")
    start = time.time()
    ssh_ok = False
    model_ok = False
    api_ok = False
    qwen_ok = False
    tunnel_url = None
    ssh_fail_n = 0
    ssh_fallback_done = False

    while time.time() - start < timeout:
        t = int(time.time() - start)
        if not ssh_ok:
            out, err = ssh_run(ssh_cmd, "echo ssh_ok")
            if "ssh_ok" in out:
                ssh_ok = True
                LOG(f"  [{t}s] ✅ SSH 就緒")
            else:
                ssh_fail_n += 1
                LOG(f"  [{t}s] SSH 未就緒: {err.strip()[:80]}")
                if ssh_fail_n >= 3 and not ssh_fallback_done:
                    ssh_fallback_done = True
                    rc2, out2, _ = run_vastai(["ssh-url", str(instance_id)],
                                              timeout=30)
                    m2 = re.search(r"ssh://root@([^:]+):(\d+)", out2 or "")
                    if m2 and (m2.group(1), int(m2.group(2))) != (ssh_host, ssh_port):
                        ssh_host, ssh_port = m2.group(1), int(m2.group(2))
                        ssh_cmd = make_ssh_cmd(ssh_host, ssh_port)
                        LOG(f"  ⚠️ proxy 連不上，切換 ssh-url 直連: "
                            f"root@{ssh_host}:{ssh_port}")

        if ssh_ok and not model_ok:
            marker, _ = ssh_run(
                ssh_cmd,
                'for f in /var/log/portal/provisioning.log '
                '/var/log/provisioning.log; do '
                'if [ -f "$f" ]; then grep -c "provisioning complete" "$f"; '
                'exit 0; fi; done; echo missing')
            m = marker.strip()
            prov_done = (m == "missing") or (m.isdigit() and int(m) >= 1)
            sizes, _ = ssh_run(
                ssh_cmd,
                'ls -l '
                '/workspace/ComfyUI/models/diffusion_models/ '
                '/workspace/ComfyUI/models/text_encoders/ '
                '/workspace/ComfyUI/models/vae/ 2>/dev/null | '
                'grep -cE "qwen_image_2\\.1_turbo_int8|qwen3vl_8b_int8|'
                'qwen_image_2\\.1_vae" || true')
            try:
                n_files = int(sizes.strip() or "0")
            except ValueError:
                n_files = 0
            if prov_done and n_files >= 3:
                model_ok = True
                LOG(f"  [{t}s] ✅ 模型下載完成 ({n_files}/3 檔)")
            else:
                LOG(f"  [{t}s] 佈建中... 模型檔={n_files}/3 "
                    f"prov={'done' if prov_done else 'running'}")

        if ssh_ok and not api_ok:
            stats, _ = ssh_run(
                ssh_cmd,
                f'curl -s --max-time 8 http://127.0.0.1:{COMFY_PORT}/system_stats '
                '|| curl -s --max-time 8 http://127.0.0.1:8188/system_stats || true')
            if "ram_total" in stats:
                api_ok = True
                LOG(f"  [{t}s] ✅ ComfyUI API 就緒")
            else:
                LOG(f"  [{t}s] ComfyUI 啟動中...")

        if api_ok and not qwen_ok:
            obj, _ = ssh_run(
                ssh_cmd,
                f'curl -s --max-time 15 http://127.0.0.1:{COMFY_PORT}/object_info '
                '|| curl -s --max-time 15 http://127.0.0.1:8188/object_info || true')
            if QWEN21_NODE in obj:
                qwen_ok = True
                LOG(f"  [{t}s] ✅ Qwen-Image-2.1 節點可用 ({QWEN21_NODE})")
            else:
                LOG(f"  [{t}s] ⚠️ 尚無 {QWEN21_NODE} 節點（ComfyUI 版本可能 < 0.37.0，"
                    f"佈建腳本會嘗試更新後重啟）")
                # 沒有節點就別等了，持續輪詢；佈建腳本會更新重啟

        if ssh_ok and not tunnel_url:
            out, _ = ssh_run(
                ssh_cmd,
                'grep -ohE "https://[a-zA-Z0-9-]+\\.trycloudflare\\.com" '
                '/var/log/portal/tunnel_manager.log 2>/dev/null | sort -u')
            candidates = sorted(set(out.split()))
            if candidates:
                LOG(f"  [{t}s] tunnel 候選 {len(candidates)} 條，探測中...")
            for url in candidates:
                r2 = subprocess.run(
                    ["curl", "-s", "--max-time", "12", f"{url}/system_stats"],
                    capture_output=True, text=True, timeout=25)
                if "comfyui" in r2.stdout.lower() or "ram_total" in r2.stdout:
                    tunnel_url = url
                    LOG(f"  [{t}s] ✅ ComfyUI tunnel: {url}")
                    break

        if ssh_ok and model_ok and api_ok and qwen_ok and tunnel_url:
            break
        time.sleep(15)

    if not ssh_ok:
        LOG("❌ SSH 一直連不上，無法檢查模型狀態")
        return ssh_cmd, None
    if not tunnel_url:
        LOG("⚠️ 找不到 tunnel URL！手動查詢:")
        LOG(f"    vastai ssh-url {instance_id}")
        LOG("    ssh root@<HOST> -p <PORT> "
            "'grep trycloudflare /var/log/portal/tunnel_manager.log'")

    return ssh_cmd, tunnel_url


def make_user_ssh(ssh_host, ssh_port):
    """給使用者看的乾淨 SSH 指令（不含 BatchMode），可搭配 -L 做埠轉發"""
    return (f"ssh -o StrictHostKeyChecking=no -i {SSH_KEY} "
            f"-p {ssh_port} root@{ssh_host}")


def wait_train_ready(instance_id, info, timeout=3600):
    """SSH 進去等訓練機佈建完成，回傳 (ssh_cmd, ui_url, ssh_host, ssh_port)

    就緒判據（不單靠 marker，「marker 不可靠」的教訓）：
    - ai-toolkit：venv 與 run.py 存在（安裝完成）
    - 模型：HF cache 內三個 base int8 檔都在
    - 訓練 UI：本機 127.0.0.1:8675 有 HTTP 回應
    - 對外 URL：UI 就緒後，從 Instance Portal 的 tunnel log 找 trycloudflare URL
      並探測（找不到時 ui_url=None，summary 會改用 SSH 埠轉發）
    """
    ssh_host = info.get("ssh_host") or info.get("public_ipaddr")
    ssh_port = info.get("ssh_port")
    if not ssh_host or not ssh_port:
        rc, out, err = run_vastai(["ssh-url", str(instance_id)], timeout=30)
        m = re.search(r"ssh://root@([^:]+):(\d+)", out or "")
        if not m:
            LOG(f"⚠️ 無法取得 SSH 資訊 (ssh-url 也失敗: {err.strip()})")
            return None, None, None, None
        ssh_host, ssh_port = m.group(1), int(m.group(2))

    ssh_cmd = make_ssh_cmd(ssh_host, ssh_port)
    user_ssh = make_user_ssh(ssh_host, ssh_port)

    LOG("🔑 附加上本機 SSH key")
    run_vastai(["attach", "ssh", str(instance_id), f"{SSH_KEY}.pub"], timeout=60)
    time.sleep(3)

    LOG("⏳ 等 ai-toolkit 安裝 + 模型下載 (~17.3GB) + 訓練 UI 就緒 (可達 60 分鐘)...")
    start = time.time()
    ssh_ok = False
    toolkit_ok = False
    model_ok = False
    ui_ok = False
    ui_url = None
    ssh_fail_n = 0
    ssh_fallback_done = False

    models_cmd = (
        'n=0; for f in qwen_image_2.1_int8_convrot.safetensors '
        'qwen3vl_8b_int8_convrot.safetensors '
        'qwen_image_2.1_vae_bf16.safetensors; do '
        f'find {TRAIN_HF_ROOT} -name "$f" 2>/dev/null | head -1 | grep -q . '
        '&& n=$((n+1)); done; echo $n')

    while time.time() - start < timeout:
        t = int(time.time() - start)
        if not ssh_ok:
            out, err = ssh_run(ssh_cmd, "echo ssh_ok")
            if "ssh_ok" in out:
                ssh_ok = True
                LOG(f"  [{t}s] ✅ SSH 就緒")
            else:
                ssh_fail_n += 1
                LOG(f"  [{t}s] SSH 未就緒: {err.strip()[:80]}")
                if ssh_fail_n >= 3 and not ssh_fallback_done:
                    ssh_fallback_done = True
                    rc2, out2, _ = run_vastai(["ssh-url", str(instance_id)],
                                              timeout=30)
                    m2 = re.search(r"ssh://root@([^:]+):(\d+)", out2 or "")
                    if m2 and (m2.group(1), int(m2.group(2))) != (ssh_host, ssh_port):
                        ssh_host, ssh_port = m2.group(1), int(m2.group(2))
                        ssh_cmd = make_ssh_cmd(ssh_host, ssh_port)
                        user_ssh = make_user_ssh(ssh_host, ssh_port)
                        LOG(f"  ⚠️ proxy 連不上，切換 ssh-url 直連: "
                            f"root@{ssh_host}:{ssh_port}")

        if ssh_ok and not toolkit_ok:
            out, _ = ssh_run(
                ssh_cmd,
                f'test -d {TRAIN_AI_TOOLKIT_DIR}/venv '
                f'&& test -f {TRAIN_AI_TOOLKIT_DIR}/run.py '
                '&& echo toolkit_ok || echo toolkit_no')
            if "toolkit_ok" in out:
                toolkit_ok = True
                LOG(f"  [{t}s] ✅ ai-toolkit 安裝完成")
            else:
                LOG(f"  [{t}s] 安裝 ai-toolkit 中...")

        if ssh_ok and not model_ok:
            sizes, _ = ssh_run(ssh_cmd, models_cmd)
            try:
                n_files = int(sizes.strip() or "0")
            except ValueError:
                n_files = 0
            if n_files >= 3:
                model_ok = True
                LOG(f"  [{t}s] ✅ 訓練模型下載完成 ({n_files}/3 檔)")
            else:
                LOG(f"  [{t}s] 下載訓練模型中... 模型檔={n_files}/3")

        if ssh_ok and not ui_ok:
            code, _ = ssh_run(
                ssh_cmd,
                f'curl -s -o /dev/null -w "%{{http_code}}" --max-time 8 '
                f'http://127.0.0.1:{AI_TOOLKIT_PORT}/ || true')
            if code.strip() in ("200", "301", "302", "307", "308"):
                ui_ok = True
                LOG(f"  [{t}s] ✅ AI Toolkit UI 就緒 (127.0.0.1:{AI_TOOLKIT_PORT})")
            else:
                LOG(f"  [{t}s] 訓練 UI 啟動中... (port {AI_TOOLKIT_PORT})")

        if ssh_ok and ui_ok and not ui_url:
            out, _ = ssh_run(
                ssh_cmd,
                'grep -ohE "https://[a-zA-Z0-9-]+\\.trycloudflare\\.com" '
                '/var/log/portal/tunnel_manager.log 2>/dev/null | sort -u')
            candidates = sorted(set(out.split()))
            for url in candidates:
                r2 = subprocess.run(
                    ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                     "--max-time", "12", url],
                    capture_output=True, text=True, timeout=25)
                if r2.stdout.strip() in ("200", "301", "302", "307", "308"):
                    ui_url = url
                    LOG(f"  [{t}s] ✅ portal tunnel 候選: {url}")
                    break

        if ssh_ok and toolkit_ok and model_ok and ui_ok:
            break
        time.sleep(15)

    if not ssh_ok:
        LOG("❌ SSH 一直連不上，無法檢查佈建狀態")
        return user_ssh, None, ssh_host, ssh_port
    if not (toolkit_ok and model_ok and ui_ok):
        LOG("⚠️ 佈建尚未全部就緒，請看實例上的 /var/log/portal/provisioning*.log")
        LOG(f"    ssh {user_ssh} 'tail -n 100 /var/log/portal/*.log'")
    return user_ssh, ui_url, ssh_host, ssh_port


def show_train_summary(instance_id, info, ssh_cmd, ui_url,
                       ssh_host=None, ssh_port=None):
    gpu = info.get("gpu_name", "?")
    dph = info.get("dph_total", 0)
    yaml_name = "train_qwen21_character_lora.yaml"
    host = ssh_host or "<HOST>"
    port = ssh_port or "<PORT>"
    print("\n" + "=" * 56)
    print("🎉 訓練機租機完成！")
    print("=" * 56)
    print(f"  實例 ID   : {instance_id}")
    print(f"  GPU       : {gpu}")
    print(f"  費率      : ${dph:.3f}/hr")
    print(f"  SSH       : {ssh_cmd}")
    print("  ── 訓練 GUI（AI Toolkit UI）──")
    print(f"   1) SSH 埠轉發（可靠）: {ssh_cmd} -N -L {AI_TOOLKIT_PORT}:127.0.0.1:{AI_TOOLKIT_PORT}")
    print(f"      然後開 http://localhost:{AI_TOOLKIT_PORT}")
    if ui_url:
        print(f"   2) portal tunnel: {ui_url}（可能導向 Instance Portal，非 UI 本身）")
    print("  ── 資料集與訓練 ──")
    print(f"   起手 YAML : {TRAIN_AI_TOOLKIT_DIR}/config/{yaml_name}")
    print(f"   資料集目錄: {TRAIN_DATASET_DIR}（先上傳 jpg/png + 同名 .txt）")
    print("   上傳訓練集（本機執行，範例）:")
    print(f"     scp -P {port} -r <你的訓練集>/* root@{host}:{TRAIN_DATASET_DIR}/")
    print("   開始訓練:")
    print(f"     {ssh_cmd} 'cd {TRAIN_AI_TOOLKIT_DIR} && ./train.sh'")
    print("   下載訓好的 LoRA:")
    print(f"     scp -P {port} root@{host}:{TRAIN_AI_TOOLKIT_DIR}/output/"
          "<訓練名稱>/<訓練名稱>.safetensors .")
    print(f"  用完刪除  : {VASTAI} destroy instance {instance_id}")
    print("               （或 python3 scripts/rent_gpu.py --destroy "
          f"{instance_id}）")
    print("=" * 56)


def show_summary(instance_id, info, ssh_cmd, tunnel_url):
    gpu = info.get("gpu_name", "?")
    dph = info.get("dph_total", 0)
    print("\n" + "=" * 56)
    print("🎉 租機完成！")
    print("=" * 56)
    print(f"  實例 ID    : {instance_id}")
    print(f"  GPU        : {gpu}")
    print(f"  費率       : ${dph:.3f}/hr")
    print(f"  SSH        : {ssh_cmd}")
    if tunnel_url:
        print(f"  ComfyUI URL: {tunnel_url}")
        print("    → SillyTavern 填這個 URL（Source=ComfyUI, Standard Server）")
        print(f"    → Turbo 設定：steps=8, cfg=1, euler/simple")
    print(f"  用完刪除   : {VASTAI} destroy instance {instance_id}")
    print("               （或 python3 scripts/rent_gpu.py --destroy "
          f"{instance_id}）")
    print("=" * 56)


def main():
    _load_api_key()
    ap = argparse.ArgumentParser(
        description="vast.ai 一鍵租機（Qwen-Image-2.1 推理 / LoRA 訓練）")
    ap.add_argument("--mode", choices=[MODE_INFERENCE, MODE_TRAIN],
                    default=MODE_INFERENCE,
                    help="inference=ComfyUI 推論(預設)；train=ai-toolkit LoRA 訓練")
    ap.add_argument("--gpu", default=None, help="指定 GPU 型號, 逗號分隔 (ex: RTX_3090,RTX_4090)")
    ap.add_argument("--offer-id", type=int, default=None,
                    help="指定 offer ID 直接租（跳過篩選，鎖定特定機器）")
    ap.add_argument("--min-vram", type=int, default=None,
                    help=f"最低 VRAM MiB (inference 預設 {DEFAULT_MIN_VRAM}；train 預設 {TRAIN_MIN_VRAM})")
    ap.add_argument("--max-cost", type=float, default=1.0, help="流量費上限 $/TB (預設 1.0)")
    ap.add_argument("--min-down", type=int, default=800, help="最低下載 Mb/s (預設 800)")
    ap.add_argument("--min-pcie", type=float, default=0,
                    help="最低 PCIe 頻寬 GB/s (預設 0=不限；設 16 可避開被閹割的卡)")
    ap.add_argument("--disk", type=int, default=None,
                    help=f"磁碟 GB (inference 預設 {DEFAULT_DISK}；train 預設 {TRAIN_DISK}；開機後不可再調整)")
    ap.add_argument("--template", default=None, help="vast.ai 模板 hash（不給則依 --mode 選預設）")
    ap.add_argument("--server-gpu", action="store_true", help="允許伺服器卡(A10/A100/PRO等)")
    ap.add_argument("--dry-run", action="store_true", help="只掃描不租")
    ap.add_argument("--destroy", metavar="INSTANCE_ID", help="刪除指定實例(停止計費)")
    args = ap.parse_args()

    is_train = args.mode == MODE_TRAIN
    # 依模式解析預設值（推理模式維持原行為）
    min_vram = args.min_vram if args.min_vram is not None else (
        TRAIN_MIN_VRAM if is_train else DEFAULT_MIN_VRAM)
    disk = args.disk if args.disk is not None else (
        TRAIN_DISK if is_train else DEFAULT_DISK)
    template = args.template or (TRAIN_TEMPLATE_HASH if is_train
                                 else DEFAULT_TEMPLATE_HASH)
    prov_url = TRAIN_PROVISIONING_URL if is_train else PROVISIONING_URL

    if args.destroy:
        rc, out, err = run_vastai(["destroy", "instance", str(args.destroy), "-y"],
                                  timeout=120)
        if rc == 0:
            print(f"✅ 實例 {args.destroy} 已刪除，停止計費")
            if out.strip():
                print(out.strip()[:300])
        else:
            print(f"❌ 刪除失敗: {err}")
        return

    # 佈建腳本必須已 push 才會被 worker 抓到
    try:
        r = subprocess.run(["curl", "-sI", "--max-time", "10", prov_url],
                           capture_output=True, text=True, timeout=15)
        lines = r.stdout.splitlines()
        if lines and " 200 " not in lines[0]:
            if args.dry_run:
                # dry-run 不會租機，僅提示；實際租機前務必先 push
                LOG(f"⚠️ 佈建腳本 URL 目前不可用: {prov_url}")
                LOG("   （dry-run 不租機，繼續；實際租機前請先 push 到 GitHub）")
            else:
                LOG(f"❌ 佈建腳本 URL 不可用: {prov_url}")
                LOG("   請先把 repo push 到 GitHub（佈建腳本要能被 worker 下載）")
                return
        else:
            LOG(f"✅ 佈建腳本 URL 可用: {prov_url}")
    except Exception as e:
        LOG(f"⚠️ 無法檢查佈建腳本 URL: {e}（繼續，但開機佈建可能失敗）")

    gpu_names = None
    gpu_q = ""
    if args.gpu:
        gpu_names = [g.replace("_", " ") for g in args.gpu.split(",")]
        gpu_q = "gpu_name in [" + ",".join(gpu_names) + "] "

    if is_train:
        LOG(f"🔍 掃描市場 (mode=train, GPU={gpu_names or '全部'}, "
            f"VRAM>={min_vram}MiB (16G 消費級可), "
            f"主機磁碟>={disk}G, 流量費<=${args.max_cost}/TB)...")
    else:
        LOG(f"🔍 掃描市場 (GPU={gpu_names or '全部'}, VRAM>={min_vram//1024}G, "
            f"流量費<=${args.max_cost}/TB)...")
    offers = search_offers(gpu_q)
    if not offers:
        LOG("❌ 查無可租機器")
        return

    hits = filter_offers(offers, args.max_cost, args.min_down, min_vram,
                         gpu_names, args.server_gpu, min_disk_free=disk,
                         min_pcie=args.min_pcie)
    if args.offer_id:
        # 指定 offer：直接鎖定該台，跳過門檻篩選（仍須存在於市場清單）
        picked = [o for o in offers if o.get("id") == args.offer_id]
        if not picked:
            LOG(f"❌ 指定的 offer id={args.offer_id} 不在目前可租清單（已被租走？）")
            return
        LOG(f"🎯 指定 offer id={args.offer_id}，鎖定該台（略過篩選）")
        hits = picked
    if not hits:
        LOG("❌ 目前沒有符合全部條件的機器！")
        LOG("   可試: --gpu 指定型號 / --max-cost 2 放寬流量費 / "
            "--server-gpu 接受伺服器卡")
        near = sorted(offers, key=lambda x: x.get("dph_total", 0))[:5]
        LOG("\n   最便宜的未過濾機器:")
        for o in near:
            LOG(f"     {o.get('gpu_name','?'):<12} ${o.get('dph_total',0):.3f}/hr "
                f"VRAM={o.get('gpu_ram',0)/1024:.0f}G "
                f"pcie={o.get('pcie_bw',0):.1f} down={o.get('inet_down',0):.0f} "
                f"流量${ptb(o.get('inet_down_cost')):.2f}/TB id={o['id']}")
        return

    LOG(f"\n📋 符合條件的候選 ({len(hits)} 台):")
    for i, o in enumerate(hits[:8]):
        LOG(f"  [{i+1}] {o.get('gpu_name','?'):<12} ${o.get('dph_total',0):.3f}/hr "
            f"VRAM={o.get('gpu_ram',0)/1024:.0f}G "
            f"TFLOPS={o.get('total_flops',0):.0f} pcie={o.get('pcie_bw',0):.1f} "
            f"down={o.get('inet_down',0):.0f}Mb/s "
            f"流量${ptb(o.get('inet_down_cost')):.2f}/TB id={o['id']}")

    if args.dry_run:
        LOG("\n(dry-run 模式，不建立實例)")
        return

    created = None
    for o in hits[:5]:
        created = create_instance(o["id"], disk, template, args.mode)
        if created:
            LOG(f"✅ 實例建立成功! ID={created}")
            break
        LOG("  該機已被租走，試下一台...")

    if not created:
        LOG("❌ 全部候選都被搶走了，稍後再試")
        return

    info = wait_running(created)
    if not info:
        LOG("❌ 實例未能 running")
        return

    if is_train:
        ssh_cmd, ui_url, ssh_host, ssh_port = wait_train_ready(created, info)
        show_train_summary(created, info, ssh_cmd, ui_url, ssh_host, ssh_port)
    else:
        ssh_cmd, tunnel_url = wait_models_ready(created, info)
        show_summary(created, info, ssh_cmd, tunnel_url)


if __name__ == "__main__":
    main()
