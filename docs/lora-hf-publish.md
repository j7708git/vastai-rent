# 把自訓 LoRA 上傳到 Hugging Face（私有收藏）

訓練完成的 LoRA（目前有 Qwen-Image-2.1 與 Krea-2-Turbo）可以放上 Hugging Face 的
**私有** model repo 保存，之後換機器或重灌時直接下載，不必重訓。以下是可重複的標準流程。

> 前置：已安裝 `hf` CLI（`pip install -U huggingface_hub`）並登入（`hf auth login`）。
> 檢查登入狀態：`hf auth whoami`。

## 1. 建立私有收藏 repo（每個底模一個）

**一個底模一個 repo，不要混。** repo 名稱要能看出是自己訓練的、以及底模版本。
目前使用：

| 底模 | repo 名稱 |
| --- | --- |
| Qwen-Image-2.1 | `my-qwen-image-2.1-loras` |
| Krea-2-Turbo | `my-krea-2-loras` |

```bash
hf repos create <你的帳號>/my-krea-2-loras --type model --private
```

**務必加 `--private`**：人物 LoRA 一律私有，不公開。

**改名**：任何時候都能改，內容不會遺失（本專案實測過）：

```bash
hf repos move <你的帳號>/<舊名稱> <你的帳號>/<新名稱> --type model
```

建立或改名後都要驗證可見度：

```bash
curl -s -H "Authorization: Bearer $(cat ~/.cache/huggingface/token)" \
  "https://huggingface.co/api/models/<你的帳號>/my-qwen-image-2.1-loras" \
  | python3 -c "import sys,json; print('private:', json.load(sys.stdin).get('private'))"
```

應輸出 `private: True`。

## 2. 準備要上傳的資料夾

一支 LoRA 一個子資料夾，名稱沿用訓練時的 `config.name`（方便對回訓練紀錄）：

```
<本機暫存>/hf-upload/
├── README.md                      # 收藏索引（底模、清單、共同訓練條件、授權）
└── <lora-name>/
    ├── <lora-name>.safetensors    # 主檔（建議的最終步數）
    ├── <lora-name>_000001250.safetensors   # 備選步數（可省略）
    ├── training-config.yaml       # 訓練當時的設定（可重現）
    └── README.md                  # 這支 LoRA 的說明卡
```

Krea-2-Turbo 的收藏則是一個訓練 job 一個資料夾，`.safetensors` 檔名沿用訓練輸出、
不改名（方便對回訓練紀錄）：

```
my-krea-2-loras/
├── README.md
├── akaonikou_v2/
├── akaonikou_v2_1/
├── LeeDahye_lora_v1/
├── lily_lora_v1/
├── lily_lora_v2/
└── mibin1215_lora_v1/
```

**不要上傳**：訓練集影像、預覽圖（含真人影像）、任何憑證、本機絕對路徑。
預覽圖留在本機或訓練機即可。

模型卡（README）建議欄位：`base_model`（Qwen 用 `Qwen/Qwen-Image-2.1`，Krea 用
`krea/Krea-2-Turbo`）、`license: other`（Qwen 另加 `license_name:
qwen-research-license`）、tags（`lora`、`text-to-image`、底模系列、`ai-toolkit`）、
觸發詞、rank、步數、解析度、最佳化設定、訓練工具與 commit、使用方式（ComfyUI 節點與
參數）、已知限制。

## 3. 上傳

```bash
hf upload <你的帳號>/my-qwen-image-2.1-loras <本機暫存>/hf-upload \
  --type model \
  --commit-message "新增 Qwen-Image-2.1 LoRA：<lora-name>"
```

### 實務注意（本專案實測踩過的點）

- **檔案不會「邊傳邊出現」**：`hf upload` 是把整批檔案傳完後才建立一個 commit，
  所以在傳完之前，網頁上的 repo 會看起來是空的（或只有舊檔案）。這是正常現象，不是卡住。
  想看進度就用 `2>&1 | tee <log>` 看它的 `Uploading... x MB transferred` 行。
- **一次傳一個大檔**比「一次傳整個資料夾」好：先傳小檔（README、設定）讓 repo 有內容，
  再逐個傳大檔，這樣每傳完一個就有一次 commit，中斷也不至於全盤重來。
- **不要用 `pkill -f "hf upload"` 之類的寫法停掉上傳**：`pkill -f` 會比對到**你自己這行指令**
  （字串裡就有 `hf upload`），連自己的 shell 一起殺掉，也會殺掉正在傳的檔案。
  要停就用不會自我匹配的樣式（例如 `pkill -f "[h]f-cli/venv/bin/python.*upload"`）。
- **速度**：一支 rank 16（約 76 MB）的 LoRA，從一般家用網路上傳約 5–15 分鐘（本專案實測
  約 0.2–0.3 MB/s）。**從訓練機（機房頻寬）上傳同一支檔案只要約 6 秒**——差兩個數量級。
  建議做法就是「在訓練機上直接上傳」：訓練完成時檔案已在該機器上，直接在該機器的 venv 內用
  `hf upload` 上傳即可（該機器需有 HF 寫入權杖，通常由 vast.ai 模板的環境變數提供）。
  若權杖不想放在租來的機器上，就退回本機上傳，忍受較慢的速度。
- 上傳完成後再跑第 1 節的驗證指令，確認檔案清單與大小（`?blobs=true` 會回報 size）。

### 從舊 repo 合併（server-side copy）

同一個帳號、同一個 storage region 的 repo 之間，可以用 `hf repos cp` 直接複製，
不必下載再上傳（本專案實測：218 MB 的檔約 4 秒，sha256 不變）：

```bash
hf repos cp \
  hf://<帳號>/<舊repo>/<舊路徑>.safetensors \
  hf://<帳號>/my-krea-2-loras/<新資料夾>/<檔名>.safetensors
```

複製後用 `?blobs=true` 回報的 `lfs.sha256` 逐一比對；全部一致才處理舊 repo。

## 4. 之後要新增或退役 LoRA

**新增**：把新的 job 子資料夾放進同一個暫存目錄，更新索引 `README.md`，再重跑第 3 節
即可（同一 repo 會多一個 commit）。

**退役舊 repo**：先確認新 repo 的 sha256 全部對上，再改名保留 30 天：

```bash
hf repos move <帳號>/<舊repo> <帳號>/archive-<舊repo> --type model
```

在 archive repo 的 README 最上面標註合併日與預定刪除日；30 天後再 `hf repos delete`。
改名後舊網址會失效，新 repo 才是正式來源。

## 注意事項

- **授權**：Qwen-Image-2.1 為 Qwen Research License（非商用）；以它訓練的 LoRA 屬衍生
  作品，散布時須標示「Built with Qwen」，且不得以「Qwen」作為主名稱。
  Krea-2-Turbo 為 gated／客製授權（`license: other`，非商業），以它訓練的 LoRA 同受約束。
- **肖像權**：素材同意已取得，放私有 repo 不需要額外處理；只有改為公開或商用時才需要
  回頭看底模的商用限制。
- **命名**：repo 名稱固定含「自己訓練」與底模版本（`my-qwen-image-2.1-loras`、
  `my-krea-2-loras`）；新增其他底模時另開對應名稱的 repo，不要混在同一處。
