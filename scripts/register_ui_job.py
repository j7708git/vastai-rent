#!/usr/bin/env python3
"""把一份 ai-toolkit 訓練 YAML 註冊成 ai-toolkit 網頁 UI 的原生訓練任務。

做了什麼：
  1. 讀取 ai-toolkit 的訓練 YAML（`job: extension` / `config: {...}` 格式）。
  2. 轉成 UI 的 job_config（外層 `{job, config, meta}`，`process[0].type` 改成
     `diffusion_trainer`，補上 `sqlite_db_path`），這樣訓練端才會回報狀態給網頁。
  3. 以 `POST /api/jobs` 在 UI 的資料庫建立任務列。**預設只註冊、狀態留在
     `stopped`**，讓使用者自己在網頁的任務詳細頁按 ▶ 啟動。

為什麼要轉這三個東西（UI 判定「原生任務」的條件，ai-toolkit `f7a1fb9` 原始碼）：
  * `config.process[0].type = diffusion_trainer`（`sd_trainer` 不會回報 UI）。
  * config 內有 `sqlite_db_path`（UI 啟動時會覆寫成 `<TOOLKIT_ROOT>/aitk_db.db`）。
  * 環境變數 `AITK_JOB_ID`（由 UI 的 worker 啟動時自動帶入，本工具不需處理）。

另外會把 `sample.prompts`（CLI 風格）補成 `sample.samples`（web 表單風格）：
  * UI 的 **Samples 分頁**讀 `sample.samples.length`；CLI 風格的 YAML 沒有這個欄位時，
    該分頁會前端崩潰（`SampleImages.tsx:92`，2026-10-11 實測 TypeError）。
  * 訓練端兩者都吃（`toolkit/config_modules.py` 的 `SampleConfig`：給了 `samples`
    會優先採用，沒給才用 legacy `prompts`），故補上等價內容不影響訓練結果。

啟動路徑的關鍵差異（交棒說明必寫）：
  * 任務**詳細頁**（`/jobs/<jobID>`）的 ▶ 會同時做
    `GET /api/jobs/<id>/start` ＋ `GET /api/queue/<gpu_ids>/start` → 一鍵就會開始訓練。
  * **Jobs 清單頁**的 ▶ 只會排入佇列，還要另外啟動佇列才會跑。

用法（在實例上跑，最常用；YAML 路徑可寫成位置參數或 --yaml）：
    /workspace/ai-toolkit/venv/bin/python scripts/register_ui_job.py \
        /workspace/ai-toolkit/config/train_qwen21_character_lora.yaml
    # 等同：... --yaml /workspace/ai-toolkit/config/train_qwen21_character_lora.yaml

用法（在本機對既有實例跑，透過 SSH；工具會自動上傳自己到 /tmp 再執行）：
    python3 scripts/register_ui_job.py --ssh root@<instance-host> \
        --ssh-args "-i ~/.ssh/id_ed25519 -p <port>" \
        /workspace/ai-toolkit/config/train_qwen21_character_lora.yaml

參數：
  YAML / --yaml PATH
                  ai-toolkit 訓練 YAML（必填，位置參數或 --yaml 二選一；
                  SSH 模式下為遠端路徑）
  --name NAME     任務名稱（預設取 YAML 的 config.name）。同時會覆寫
                  config.name，確保輸出／預覽圖目錄 <training_folder>/<name>/ 一致。
  --gpu-ids IDS   要綁定的 GPU 群組字串（預設 "0"）
  --api-url URL   UI 位址（預設 http://127.0.0.1:8675）
  --dry-run       只印出將送出的 job_config JSON，不發任何請求
  --start         註冊後立刻 start ＋ 啟動佇列（自動啟動情境用）
  --update        同名任務已存在時就地更新設定（預設為明確報錯）
  --wait SEC      使用 --start 時等待狀態變成 queued/running 的秒數（預設 60）
  --timeout SEC   單一 HTTP 請求逾時秒數（預設 20）
  --ssh HOST      改以 SSH 在遠端執行（搭配 --ssh-args）
  --ssh-args ARGS 附加給 ssh 的參數，例如 "-i ~/.ssh/id_ed25519 -p <port>"

退出碼：
  0  成功
  1  失敗（UI 連不上、YAML 有問題、名稱重複未加 --update、建立後查不到…）
  2  部分完成／警告（任務已註冊，但 --start 後未在時限內變成 queued/running）
"""

import argparse
import copy
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_API_URL = "http://127.0.0.1:8675"
DEFAULT_SQLITE_DB_PATH = "./aitk_db.db"
JOB_TYPE_TRAIN = "train"

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_PARTIAL = 2


class ToolError(Exception):
    """可預期的使用者錯誤（訊息即可讀，直接印出）。"""


class UiUnreachable(ToolError):
    pass


# --------------------------------------------------------------------------- HTTP


def _http_json(method, url, payload=None, timeout=20):
    """以標準函式庫發 HTTP，回傳 (status, parsed_json_or_text)。"""
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
            status = resp.status
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        status = e.code
    except urllib.error.URLError as e:
        raise UiUnreachable(f"連不上 UI（{url}）：{e.reason}") from e
    except OSError as e:
        raise UiUnreachable(f"連不上 UI（{url}）：{e}") from e

    parsed = None
    if body.strip():
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = body
    return status, parsed


def _short(value, n=400):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= n else text[:n] + "…"


# --------------------------------------------------------------------------- YAML


def load_yaml(path):
    """讀 YAML；缺 PyYAML 時給明確訊息。"""
    try:
        import yaml  # noqa: WPS433 (延後匯入，SSH 模式的執行端才需要)
    except ImportError as e:
        raise ToolError(
            "找不到 PyYAML（yaml）模組。請改用安裝好相依套件的解譯器，例如：\n"
            "  /workspace/ai-toolkit/venv/bin/python scripts/register_ui_job.py ...\n"
            "或在該環境安裝：pip install pyyaml"
        ) from e

    p = Path(path)
    if not p.is_file():
        raise ToolError(f"找不到 YAML 檔：{path}")
    try:
        with p.open("r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
    except yaml.YAMLError as e:
        raise ToolError(f"YAML 解析失敗：{path}\n{e}") from e
    return doc


def _ensure_ui_sample_shape(process_config):
    """把 CLI 風格的 sample.prompts 補成 UI 表單風格的 sample.samples。

    UI 的 Samples 分頁讀 `sample.samples.length`（`SampleImages.tsx:92`），只有
    `prompts` 時該分頁會前端崩潰；訓練端兩者都支援（`SampleConfig` 以 samples
    優先），所以補上等價內容即可，不改變訓練行為。
    """
    sample = process_config.get("sample")
    if not isinstance(sample, dict):
        return
    existing = sample.get("samples")
    if isinstance(existing, list) and existing:
        return
    prompts = sample.get("prompts")
    if isinstance(prompts, list) and prompts:
        sample["samples"] = [
            {"prompt": p} for p in prompts
        ]


def build_job_config(doc, name):
    """把 ai-toolkit YAML 文件轉成 UI 的 job_config。"""
    if not isinstance(doc, dict):
        raise ToolError("YAML 根節點必須是 mapping（要有 job / config / meta）")
    config = doc.get("config")
    if not isinstance(config, dict):
        raise ToolError("YAML 缺少 config 區塊（ai-toolkit 訓練設定應放在 config:）")

    cfg = copy.deepcopy(config)
    cfg["name"] = name  # 輸出／預覽圖目錄 = <training_folder>/<name>/，必須與任務名一致

    process = cfg.get("process")
    if not isinstance(process, list) or not process:
        raise ToolError("config.process 必須是非空清單")
    first = process[0]
    if not isinstance(first, dict):
        raise ToolError("config.process[0] 必須是 mapping")

    # UI 原生任務的兩個必要標記
    first["type"] = "diffusion_trainer"
    first.setdefault("sqlite_db_path", DEFAULT_SQLITE_DB_PATH)
    # UI 前端（Samples 分頁）需要的欄位形狀
    _ensure_ui_sample_shape(first)

    meta = doc.get("meta")
    meta = dict(meta) if isinstance(meta, dict) else {}
    meta["name"] = name
    meta.setdefault("version", "1.0")

    return {"job": doc.get("job", "extension"), "config": cfg, "meta": meta}


def validate_name(name):
    if not name or not name.strip():
        raise ToolError("任務名稱不可為空")
    if name != name.strip():
        raise ToolError(f"任務名稱前後不可有空白：{name!r}")
    if "/" in name or "\\" in name or "\0" in name:
        raise ToolError(f"任務名稱不可含路徑分隔字元：{name!r}")
    return name


# --------------------------------------------------------------------------- UI API


class UiClient:
    def __init__(self, api_url, timeout=20):
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout

    def _url(self, path):
        return f"{self.api_url}{path}"

    def ping(self):
        status, body = _http_json("GET", self._url("/api/jobs"), timeout=self.timeout)
        if status != 200 or not isinstance(body, dict):
            raise ToolError(f"UI /api/jobs 回應異常（HTTP {status}）：{_short(body)}")
        return body

    def list_jobs(self):
        body = self.ping()
        jobs = body.get("jobs")
        if not isinstance(jobs, list):
            raise ToolError(f"UI /api/jobs 格式非預期：{_short(body)}")
        return jobs

    def find_by_name(self, name):
        for job in self.list_jobs():
            if job.get("name") == name:
                return job
        return None

    def get_job(self, job_id):
        status, body = _http_json(
            "GET", self._url(f"/api/jobs?id={job_id}"), timeout=self.timeout
        )
        if status == 200 and isinstance(body, dict) and body.get("id"):
            return body
        return None

    def create_job(self, name, gpu_ids, job_config):
        status, body = _http_json(
            "POST",
            self._url("/api/jobs"),
            payload={"name": name, "gpu_ids": gpu_ids, "job_config": job_config},
            timeout=self.timeout,
        )
        if status == 409:
            raise ToolError(
                "UI 回報名稱重複（HTTP 409）：同名任務已存在。加 --update 可就地更新設定。"
            )
        if status != 200 or not isinstance(body, dict) or not body.get("id"):
            raise ToolError(f"建立任務失敗（HTTP {status}）：{_short(body)}")
        return body

    def update_job(self, job_id, name, gpu_ids, job_config):
        status, body = _http_json(
            "POST",
            self._url("/api/jobs"),
            payload={
                "id": job_id,
                "name": name,
                "gpu_ids": gpu_ids,
                "job_config": job_config,
            },
            timeout=self.timeout,
        )
        if status != 200 or not isinstance(body, dict) or not body.get("id"):
            raise ToolError(f"更新任務失敗（HTTP {status}）：{_short(body)}")
        return body

    def start_job(self, job_id):
        status, body = _http_json(
            "GET", self._url(f"/api/jobs/{job_id}/start"), timeout=self.timeout
        )
        if status not in (200, 201):
            raise ToolError(f"任務排入佇列失敗（HTTP {status}）：{_short(body)}")
        return body

    def start_queue(self, gpu_ids):
        status, body = _http_json(
            "GET", self._url(f"/api/queue/{gpu_ids}/start"), timeout=self.timeout
        )
        if status not in (200, 201):
            raise ToolError(f"啟動佇列失敗（HTTP {status}）：{_short(body)}")
        return body

    def wait_for_status(self, job_id, wanted, timeout=60.0, interval=2.0):
        """輪詢直到狀態落在 wanted（set）內；回傳最終 job，逾時回傳 None。"""
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            last = self.get_job(job_id)
            if last and last.get("status") in wanted:
                return last
            time.sleep(interval)
        return None

    def get_job_status(self, job_id):
        job = self.get_job(job_id)
        return job.get("status") if job else None


# --------------------------------------------------------------------------- SSH


def _remote_args(args, name):
    remote = ["--yaml", args.yaml, "--gpu-ids", args.gpu_ids,
              "--api-url", args.api_url, "--timeout", str(args.timeout),
              "--wait", str(args.wait)]
    remote += ["--name", name] if name else []
    if args.dry_run:
        remote.append("--dry-run")
    if args.start:
        remote.append("--start")
    if args.update:
        remote.append("--update")
    return remote


def run_over_ssh(args, name):
    """把本工具上傳到遠端 /tmp 執行，再把結果透傳回來。"""
    ssh_prefix = ["ssh"] + shlex.split(args.ssh_args) + [args.ssh]
    remote_script = f"/tmp/register_ui_job_{os.getpid()}.py"
    src = Path(__file__).read_text(encoding="utf-8")

    print(f"[ssh] 上傳工具到 {args.ssh}:{remote_script}")
    up = subprocess.run(ssh_prefix + [f"cat > {shlex.quote(remote_script)}"],
                        input=src, text=True)
    if up.returncode != 0:
        print(f"❌ SSH 上傳失敗（ssh 退出碼 {up.returncode}）", file=sys.stderr)
        return EXIT_FAIL

    py = ('PY=/workspace/ai-toolkit/venv/bin/python; '
          '[ -x "$PY" ] || PY=python3; ')
    cmd = py + '"$PY" ' + shlex.quote(remote_script) + " " + \
        " ".join(shlex.quote(a) for a in _remote_args(args, name))
    try:
        proc = subprocess.run(ssh_prefix + [cmd])
        rc = proc.returncode
    finally:
        subprocess.run(ssh_prefix + [f"rm -f {shlex.quote(remote_script)}"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return rc


# --------------------------------------------------------------------------- main


def _print_result(action, name, job_id, status, api_url, gpu_ids, extra=None):
    job_url = f"{api_url.rstrip('/')}/jobs/{job_id}"
    print("")
    print("✅ UI 任務已就緒" if action != "updated" else "✅ UI 任務已更新")
    print(f"   任務名稱  : {name}")
    print(f"   job id    : {job_id}")
    print(f"   狀態      : {status}")
    print(f"   gpu_ids   : {gpu_ids}")
    print(f"   任務詳細頁: {job_url}")
    print("   ▶ 啟動    : 點進上面的任務詳細頁，按 ▶ 就會開始訓練")
    print("               （Jobs 清單頁按 ▶ 只會排入佇列，還要另外啟動佇列）")
    print("JSON: " + json.dumps({
        "ok": True,
        "action": action,
        "name": name,
        "job_id": job_id,
        "status": status,
        "gpu_ids": gpu_ids,
        "api_url": api_url,
        "job_url": job_url,
        **(extra or {}),
    }, ensure_ascii=False))


def run(args):
    name_hint = args.name
    doc = load_yaml(args.yaml)
    if not isinstance(doc, dict):
        raise ToolError("YAML 根節點必須是 mapping")
    cfg_name = doc.get("config", {}).get("name") if isinstance(doc.get("config"), dict) else None
    name = validate_name(name_hint or cfg_name or "")
    job_config = build_job_config(doc, name)

    if args.dry_run:
        print(f"# dry-run：以下為將送至 {args.api_url}/api/jobs 的 job_config")
        print(json.dumps(job_config, ensure_ascii=False, indent=2))
        first = job_config["config"]["process"][0]
        print("", file=sys.stderr)
        print(f"# 檢查：process[0].type={first.get('type')!r} "
              f"sqlite_db_path={first.get('sqlite_db_path')!r} "
              f"config.name={job_config['config'].get('name')!r}", file=sys.stderr)
        return EXIT_OK

    client = UiClient(args.api_url, timeout=args.timeout)
    client.ping()  # 連不上時直接 UiUnreachable

    existing = client.find_by_name(name)
    if existing and not args.update:
        raise ToolError(
            f"同名任務已存在：{name}（job id {existing.get('id')}，狀態 "
            f"{existing.get('status')}）。要就地更新請加 --update；"
            f"要用新名稱請改 --name。"
        )

    extra = {}
    if existing:
        if existing.get("status") in ("running", "queued", "stopping"):
            raise ToolError(
                f"同名任務正在執行中（狀態 {existing.get('status')}），"
                f"不建議就地更新設定。請先在網頁停止，或改 --name 另建一個。"
            )
        job = client.update_job(existing["id"], name, args.gpu_ids, job_config)
        action = "updated"
    else:
        job = client.create_job(name, args.gpu_ids, job_config)
        action = "created"

    job_id = job["id"]

    # 驗證：建立後一定查得到
    verified = client.get_job(job_id)
    if not verified:
        raise ToolError(f"送出後查不到任務（job id {job_id}），UI 可能未正確寫入資料庫")
    status = verified.get("status")
    if verified.get("name") != name:
        raise ToolError(
            f"任務名稱不符：預期 {name!r}，UI 回報 {verified.get('name')!r}"
        )
    extra["verified_status"] = status

    if not args.start:
        _print_result(action, name, job_id, status, args.api_url, args.gpu_ids, extra)
        if status != "stopped":
            print(f"⚠️ 預期預設狀態為 stopped，實際為 {status}", file=sys.stderr)
            return EXIT_PARTIAL
        return EXIT_OK

    # --start：start ＋ queue start，等同任務詳細頁按 ▶
    client.start_job(job_id)
    client.start_queue(args.gpu_ids)
    final = client.wait_for_status(job_id, {"queued", "running", "completed"},
                                   timeout=args.wait, interval=2.0)
    if final is None:
        got = client.get_job_status(job_id)
        print(f"⚠️ 已送出 start，但 {args.wait}s 內狀態未變成 queued/running"
              f"（目前 {got}）", file=sys.stderr)
        _print_result(action, name, job_id, got, args.api_url, args.gpu_ids,
                      {**extra, "started": False})
        return EXIT_PARTIAL
    extra["started"] = True
    _print_result(action, name, job_id, final.get("status"), args.api_url,
                  args.gpu_ids, extra)
    if final.get("status") == "completed":
        print("⚠️ 任務在等待期間已完成（步驟極少時可能發生）", file=sys.stderr)
        return EXIT_PARTIAL
    return EXIT_OK


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="把 ai-toolkit 訓練 YAML 註冊成 ai-toolkit 網頁 UI 的原生訓練任務"
                    "（預設只註冊不啟動）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("yaml_pos", nargs="?", default=None, metavar="YAML",
                    help="ai-toolkit 訓練 YAML 路徑（等同 --yaml；位置參數寫法）")
    ap.add_argument("--yaml", default=None,
                    help="ai-toolkit 訓練 YAML 路徑（必填；或用位置參數 YAML）")
    ap.add_argument("--name", default=None,
                    help="任務名稱（預設取 YAML 的 config.name；會同時覆寫 config.name）")
    ap.add_argument("--gpu-ids", default="0", help="GPU 群組字串（預設 0）")
    ap.add_argument("--api-url", default=DEFAULT_API_URL,
                    help=f"UI 位址（預設 {DEFAULT_API_URL}）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只印出將送出的 job_config，不發請求")
    ap.add_argument("--start", action="store_true",
                    help="註冊後立刻 start ＋ 啟動佇列（自動啟動情境）")
    ap.add_argument("--update", action="store_true",
                    help="同名任務已存在時就地更新設定（預設報錯）")
    ap.add_argument("--wait", type=float, default=60.0,
                    help="使用 --start 時等待 queued/running 的秒數（預設 60）")
    ap.add_argument("--timeout", type=float, default=20.0,
                    help="單一 HTTP 請求逾時秒數（預設 20）")
    ap.add_argument("--ssh", default=None,
                    help="改以 SSH 在遠端執行（值是 ssh 目標，如 root@host）")
    ap.add_argument("--ssh-args", default="-o StrictHostKeyChecking=no",
                    help="附加給 ssh 的參數（如 '-i ~/.ssh/id_ed25519 -p <port>'）")
    args = ap.parse_args(argv)

    if args.yaml and args.yaml_pos and args.yaml != args.yaml_pos:
        print("❌ 同時給了位置參數與 --yaml，且兩者不同；請只留一個。", file=sys.stderr)
        return EXIT_FAIL
    args.yaml = args.yaml or args.yaml_pos

    try:
        if not args.yaml:
            raise ToolError("缺少 YAML 路徑（請用位置參數或 --yaml 指定）")
        if args.ssh:
            return run_over_ssh(args, args.name)
        return run(args)
    except ToolError as e:
        print(f"❌ {e}", file=sys.stderr)
        return EXIT_FAIL
    except KeyboardInterrupt:
        print("已中斷", file=sys.stderr)
        return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
