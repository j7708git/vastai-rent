# 在 vast.ai 訓練 Qwen-Image-2.1 人物 LoRA（訓練模式）

本專案的訓練模式會一鍵租一台 GPU 機器，自動裝好
[ostris/ai-toolkit](https://github.com/ostris/ai-toolkit)、下載 Qwen-Image-2.1
base（int8 量化）訓練模型與官方設定檔、生成起手訓練設定，並把訓練 UI 開好等你去連。
適合訓練 **Qwen-Image-2.1 人物 LoRA**。

> 相關的推理（ComfyUI 生圖）流程見 [README](../README.md)；本篇只講訓練。

## 快速開始

```bash
# 1) 租訓練機（掃描 → 篩選 → 租 → 等佈建完成 → 回報連線資訊）
python3 scripts/rent_gpu.py --mode train

# 只掃描不租，先看有哪些候選機
python3 scripts/rent_gpu.py --mode train --dry-run

# 2) 上傳訓練集（見下方「資料集格式」與「上傳方式」）
# 3) 連上訓練 UI，點進任務詳細頁按 ▶ 開始訓練（見「在網頁上跑訓練」）
#    （命令列替代方案：SSH 進去 ./train.sh）
# 4) 下載訓好的 LoRA，放回 ComfyUI 使用
# 5) 用完刪除實例，停止計費
python3 scripts/rent_gpu.py --destroy <實例ID>
```

佈建完成後，起手任務**已經註冊好、狀態是 `stopped`**，你在網頁按 ▶ 就會開始——
細節（怎麼看狀態／步數／速度／loss／預覽圖、怎麼 Stop）見下方
[在網頁上跑訓練](#在網頁上跑訓練ui-原生任務)。

訓練模式的預設篩選條件與推理模式不同：

| 條件 | 訓練模式預設 | 說明 |
| --- | --- | --- |
| VRAM | ≥ 16000 MiB | 16GB 消費級卡（4080 / 4080S / 5070 Ti / 5080）也可 |
| 磁碟 | 80 GB | 模型 ~17.3GB＋latent/TE 快取＋optimizer state；**開機後不可再調整** |
| GPU 類型 | 消費級 RTX | 伺服器卡可加 `--server-gpu` |
| 模板 | PyTorch (Vast) | 乾淨的 PyTorch 環境，與推理用的 ComfyUI 環境分開 |

其餘參數（`--gpu`、`--min-vram`、`--disk`、`--max-cost` 等）與推理模式共用。

## 資料集格式

ai-toolkit 的資料集是「一個資料夾，裡面放圖片 + 同名文字檔」：

- 圖片格式：**jpg / jpeg / png**。**不支援 webp。**
- 每張圖搭配**同名 `.txt`**，內容是 caption，例如 `image001.png` 配 `image001.txt`。
- **不需自行裁切或縮放**：loader 會自動依 aspect ratio 分桶（bucket，`buckets: true`）並縮放；
  起手設定**不開** `square_crop`（維持預設 `false`）。若把它設成 `true`，
  loader 會**跳過分桶**、把每張圖中央裁成 `resolution × resolution` 的正方形，
  建議保持關閉（官方範例與 ai-toolkit 預設皆為 `false`）。
- 建議 **15–30 張**，涵蓋不同角度、表情、姿勢、服裝、背景；
  只放你希望推論時看到的視角。
- caption 建議描述「會變動的屬性」（服裝、場景、動作），
  固定不變的辨識特徵（例如標誌性髮型）不必寫進 caption。

**觸發詞（trigger word）**：用罕見 token（如 `ohwx`、`sks` 或自造字）把「這個人」
綁到該 token。ai-toolkit 的行為是：

- caption 內寫 `[trigger]` 佔位符 → 執行時替換成設定裡的 `trigger_word`。
- 若設定了 `trigger_word` 而 caption 沒提到它 → 會自動補進 caption（已含則不重複加）。
- 這個注入發生在**寫入 text-embedding 快取之前**，所以即使同時開了
  `cache_text_embeddings` 仍會生效（舊版「開快取時觸發詞失效」的警語已過時）。

所以兩種寫法都可用：直接在 caption 寫觸發詞，或在 caption 寫 `[trigger]` 並靠設定替換。
起手設定已帶一個 `trigger_word` 佔位；若 caption 已直接寫入觸發詞，把該行註解掉可避免重複注入。

### 目錄結構

佈建完成的機器上，資料集目錄是 `/workspace/dataset/`：

```
/workspace/dataset/
  image001.png
  image001.txt     # e.g. "mychar, portrait, cafe, blue shirt"
  image002.png
  image002.txt
  ...
```

## 上傳方式（scp）

租機完成後，腳本會回報 SSH 指令與主機／埠，例如：

```bash
# 把本機訓練集資料夾內的所有檔案上傳到機器的資料集目錄
scp -P <PORT> -r <你的訓練集>/* root@<HOST>:/workspace/dataset/
```

（`<HOST>` 與 `<PORT>` 由租機腳本的 summary 提供；也可用 `vastai ssh-url <實例ID>`
查詢。機器已附上你的 SSH 公鑰。）

資料集較大時也可用 `rsync -avz -e "ssh -p <PORT>"`，或先壓縮再傳。

## 訓練 UI 與連線

租機腳本會回報兩種連線方式：

1. **SSH 埠轉發（建議，穩定）**：

   ```bash
   ssh -o StrictHostKeyChecking=no -i <你的私鑰> -p <PORT> \
       -N -L 8675:127.0.0.1:8675 root@<HOST>
   ```

   然後在本機瀏覽器開 <http://localhost:8675>。

   這台訓練機的常用內部埠（本機埠可自行更換）：

   | 服務 | 容器內埠 | 備註 |
   | --- | --- | --- |
   | Instance Portal | 11111 | Portal 後端，無驗證；1111 是 Caddy 入口，需 basic auth／token |
   | AI Toolkit UI | 8675 | 訓練介面本體，免登入 |
   | Jupyter | 8080 | 只提供 HTTPS，需帶 `?token=` |
   | TensorBoard | 16006 | |
   | Syncthing | 18384 | |

   > 本機 8080 常已被其他服務佔用（例如 Caddy），`-L 8080:...` 會噴
   > `Address already in use`。改用 18080、18675 之類的埠即可，例如：
   > `-L 18080:127.0.0.1:11111 -L 18675:127.0.0.1:8675`。

2. **Instance Portal tunnel**：vast.ai 的 Instance Portal 會替 UI 建立
   Cloudflare tunnel 連結，可從實例卡片的 Open／tunnel 資訊取得
   （回報的 URL 有時導向 Portal 本身，非 UI 本體；以 SSH 埠轉發為準）。

UI 本身不需一直開著，訓練啟動後關掉瀏覽器不影響訓練；它只用來啟動／停止／監看 job。

## 在網頁上跑訓練（UI 原生任務）

佈建完成後，起手 YAML 會被自動註冊成 ai-toolkit UI 的**原生訓練任務**，而且**預設只註冊、
不啟動**——任務會出現在網頁 Jobs 清單、狀態停在 `stopped`，由你自己按 ▶ 開始。

> 註冊是**防禦性**的：UI 還沒起來、或註冊失敗，佈建仍會成功（只記 warning），訓練照樣可以
> 用命令列 `./train.sh` 跑。可用環境變數 `AITK_REGISTER_UI_JOB=0` 關閉自動註冊。

### 流程

1. 連上 UI（`http://localhost:8675`，或實例的 `<tunnel-url>`）。
2. **Jobs 清單**看得到任務，名稱＝`TRAIN_NAME`（預設 `qwen21_character_lora`），狀態 `stopped`。
3. **點進任務詳細頁**（`/jobs/<jobID>`），按 ▶ → 開始訓練。
   * **關鍵**：只有**任務詳細頁**的 ▶ 會「排入佇列 ＋ 啟動佇列」，也就是一鍵開始；
     **Jobs 清單頁**的 ▶ 只會排入佇列，還要另外啟動佇列才會跑。**請點進詳細頁按 ▶。**
4. 訓練中看得到 **狀態、步數（`step`／`total_steps`）、速度、loss 曲線**；跑到
   `sample_every` 的倍數步時，**Samples** 會出現預覽圖
   （讀 `<training_folder>/<job.name>/samples/`，即機器上的
   `/workspace/ai-toolkit/output/<任務名>/samples/`）。
5. 網頁按鈕：**Stop Job**（停止，程序會結束）、**Save Next Step**（下個存檔點立刻存）、
   **Sample Next Step**（下個取樣點立刻出圖）；另有 **Mark as Stopped**（程序已不在時把狀態改回
   停止，避免卡在 running）。

> ⚠️ 網頁的 **Delete** 會**遞迴刪除該任務的輸出目錄**（`output/<任務名>/`，含訓好的 LoRA）。
> 要保留成果就別按 Delete，改用下載／上傳到 Hugging Face 的方式保存（見
> [lora-hf-publish.md](lora-hf-publish.md)）。

### 讓「速度」與「loss 圖」有資料

ai-toolkit 預設**不寫**這兩樣，網頁速度欄會顯示 `?`、loss 圖會是空的。佈建產生的起手 YAML
已經帶好以下欄位：

| YAML 欄位 | 作用 | 起手值 |
| --- | --- | --- |
| `process[0].performance_log_every` | `> 0` 才會週期性寫入 `speed_string`（速度欄） | `1` |
| `process[0].logging.use_ui_logger` | `true` 才會寫出 `<job>/loss_log.db`（loss 圖來源） | `true` |
| `process[0].logging.log_every` | loss 取樣間隔（步） | `1` |
| `sample.sample_every` | 每幾步出一張預覽圖（需 `train.disable_sampling: false`） | `250` |

（`train.disable_sampling` 預設已改為 `false`；想省 VRAM／時間可設 `TRAIN_DISABLE_SAMPLING=true`
關掉取樣，但這樣網頁就不會有預覽圖。）

### 把「既有 YAML」註冊成 UI 任務（`scripts/register_ui_job.py`）

佈建已自動註冊一次；要另外註冊別的 YAML（例如你自己改過的 `config/xxx.yaml`）時：

```bash
# 方式一：在實例上執行（YAML 與 UI 都在同一台）
ssh -i <你的私鑰> -p <PORT> root@<HOST> \
  '/workspace/ai-toolkit/venv/bin/python /workspace/register_ui_job.py \
     --yaml /workspace/ai-toolkit/config/xxx.yaml'
```

```bash
# 方式二：在本機透過 SSH 執行（工具會自動上傳自己到實例的 /tmp 再跑；
#         --yaml 是實例上的路徑）
python3 scripts/register_ui_job.py --ssh root@<HOST> \
  --ssh-args "-i ~/.ssh/id_ed25519 -p <PORT>" \
  --yaml /workspace/ai-toolkit/config/xxx.yaml
```

常用參數：

| 參數 | 說明 |
| --- | --- |
| `--yaml PATH` | 必填。ai-toolkit 訓練 YAML（SSH 模式下為實例上的路徑） |
| `--name NAME` | 任務名稱（預設取 YAML 的 `config.name`；會**一併覆寫 `config.name`**，確保輸出與預覽圖目錄 `<training_folder>/<name>/` 一致） |
| `--gpu-ids IDS` | 綁定的 GPU 群組字串（預設 `0`） |
| `--api-url URL` | UI 位址（預設 `http://127.0.0.1:8675`） |
| `--dry-run` | 只印出將送出的 `job_config` JSON，不發任何請求 |
| `--start` | 註冊後立刻排入佇列並啟動佇列（自動啟動情境） |
| `--update` | 同名任務已存在時就地更新設定（預設會明確報錯，並附既有 job id 與提示） |

退出碼：**0** 成功、**1** 失敗（UI 連不上、YAML 有問題、名稱重複未加 `--update`…）、
**2** 部分完成／警告（例如 `--start` 後未在時限內變成 `queued`／`running`）。

工具做的事：把 YAML 轉成 UI 的 `job_config`（外層 `{job, config, meta}`，
`process[0].type` 改成 **`diffusion_trainer`**、補上 `sqlite_db_path`），再 `POST /api/jobs`
建任務列。`diffusion_trainer` ＋ `sqlite_db_path` ＋ UI 啟動時注入的 `AITK_JOB_ID`
三者齊備，訓練端才會把狀態／步數／速度／loss 回報到網頁——這是「用 `sd_trainer` 從命令列跑」
時網頁看不到任務的原因。

## 起手訓練設定（YAML）

佈建腳本會生成起手設定檔，放在機器上：

```
/workspace/ai-toolkit/config/train_qwen21_character_lora.yaml
```

可直接在 UI 載入，或用命令列跑：

```bash
ssh -i <你的私鑰> -p <PORT> root@<HOST> \
    'cd /workspace/ai-toolkit && ./train.sh'
```

（`./train.sh` 是佈建生成的包裝腳本，會帶好 HF 快取與 CUDA 記憶體設定後執行
`python run.py config/train_qwen21_character_lora.yaml`。）

主要欄位：

| 欄位 | 起手值 | 說明 |
| --- | --- | --- |
| `config.name` | `qwen21_character_lora` | 輸出資料夾／檔名；改它可避免不同實驗互相覆蓋或續訓 |
| `trigger_word` | 觸發詞佔位 | 搭配 caption 的 `[trigger]`；caption 已含觸發詞時請註解掉 |
| `network.linear` / `linear_alpha` | 16 / 16 | LoRA rank；更高 rank 需更多 VRAM（可試 32） |
| `datasets[].folder_path` | `/workspace/dataset` | 資料集路徑 |
| `datasets[].resolution` | `[1024]` | 訓練解析度；影像邊長須為 **32 的倍數** |
| `save.save_every` / `max_step_saves_to_keep` | 250 / 4 | 每 250 步存一次，保留 4 個中繼點 |
| `train.steps` | 2000 | 總步數。**先跑 200 步冒煙測試**（loss 有動、能存檔、不 OOM），主力結果看 **1250–1750 步**，**每 250 步比較** checkpoint（見「訓後自檢與步數」） |
| `train.optimizer` | `adamw8bit` | 8-bit AdamW，省 VRAM |
| `train.lr` | `0.0001` | 學習率 |
| `train.noise_scheduler` / `timestep_type` | `flowmatch` / `shift` | 2.1 的 flow-matching 設定 |
| `train.gradient_checkpointing` | `true` | 省 VRAM，略慢。由 `TRAIN_GRAD_CP` 環境變數控制（預設 `true`）；與 `layer_offloading` 併用是 issue #1054 的組合，見「訓後自檢與步數」 |
| `train.disable_sampling` | `false` | 訓練中**會**出樣本圖（網頁預覽圖來源）。設 `true` 可省 VRAM／時間，但網頁就沒有預覽圖；由 `TRAIN_DISABLE_SAMPLING` 環境變數控制 |
| `train.skip_first_sample` | `true` | 跳過訓練前 baseline 樣本（順帶避開 issue #1059） |
| `process[0].performance_log_every` | `1` | `>0` 才會寫速度（`speed_string`）；`0` 會讓網頁速度欄恆為 `?` |
| `process[0].logging.use_ui_logger` / `log_every` | `true` / `1` | 寫出 `<job>/loss_log.db`（網頁 loss 圖來源）；`false` 時 loss 圖永遠是空的 |
| `model.arch` | `qwen_image_2` | 2.1 專用架構；別與初代 Qwen-Image 混用 |
| `model.name_or_path` | `Comfy-Org/Qwen-Image-2.1` | 權重來源（int8 量化 repack） |
| `model.extras_name_or_path` | `Qwen/Qwen-Image-2.1` | 設定檔／processor 來源 |
| `model.quantize` / `qtype` | `true` / `convrot8` | 直接以 int8 量化權重訓練，省 VRAM |
| `model.low_vram` / `layer_offloading` | `true` / `true` | 低 VRAM 模式＋CPU 層卸載。`layer_offloading`（含兩個 `*_percent`）由 `TRAIN_LAYER_OFFLOAD` 環境變數一起控制（預設 `true`） |
| `sample.guidance_scale` | `1.0` | Qwen-Image-2.1 是 **CFG-distilled**：官方推論用 **CFG=1**。ai-toolkit 的 sample pipeline 在 `>1` 時會真的多跑一次 CFG forward，會偏離真實推論，故用 `1.0` |

**12–16GB 卡的建議**：保留 `low_vram`、`layer_offloading`（含兩個 `*_percent: 1.0`）、
`optimizer: adamw8bit`、`disable_sampling: true`，並可把 `resolution` 降到 512–768。

**兩個記憶體策略環境變數**（在租機前 export，或寫進 vast.ai 模板環境變數；未設時與預設行為相同）：

| 環境變數 | 預設 | 對應 YAML 欄位 | 說明 |
| --- | --- | --- | --- |
| `TRAIN_GRAD_CP` | `true` | `train.gradient_checkpointing` | `false` 可關掉 checkpointing（省時間但 **VRAM 需求上升**） |
| `TRAIN_LAYER_OFFLOAD` | `true` | `model.layer_offloading`（含兩個 `*_percent`） | `false` 可整組關掉 CPU 層卸載（VRAM 需求上升） |
| `TRAIN_DISABLE_SAMPLING` | `false` | `train.disable_sampling` | `true` 完全不產生預覽圖（省 VRAM／時間，但網頁看不到圖） |
| `TRAIN_PERF_LOG_EVERY` | `1` | `process[0].performance_log_every` | 速度（`speed_string`）更新間隔；`0` 關閉 |
| `TRAIN_LOG_EVERY` | `1` | `process[0].logging.log_every` | loss 圖取樣間隔 |
| `AITK_REGISTER_UI_JOB` | `1` | （佈建行為） | 設 `0` 可關閉「佈建完成後自動把起手 YAML 註冊成 UI 任務」 |

「速度」與「loss 圖」的欄位細節與網頁操作，見上方
[在網頁上跑訓練](#在網頁上跑訓練ui-原生任務)。

改完 YAML 後可另存不同檔名（放在 `config/` 下），再用
`./train.sh config/<你的檔名>.yaml` 執行指定的設定；不帶參數時預設跑
`config/train_qwen21_character_lora.yaml`。

## 訓後自檢與步數（確認 LoRA 不是「空包彈」）

### 為什麼要自檢：issue #1054 / #1075

ai-toolkit 有兩條回報，症狀都是「訓練 loss 看起來正常，但存出的 LoRA 在 ComfyUI
載入後**完全沒效果**」，原因是整批 `lora_B` 權重被存成 0（`lora_B` 是零初始化，
若梯度自始為 0 就會永遠為 0）：

- **#1054（closed 未修）**：`gradient_checkpointing: true` ＋ `layer_offloading`
  同時開啟時，回報 **192 個 `lora_B` 全為 0**。**我們的起手設定正好是這個組合。**
- **#1075（open）**：另一個變體，即使關掉 checkpointing／offload，仍有部分
  `img_mlp` 的 `lora_B` 全零。

**本專案已於實機（RTX 3090 24G、ai-toolkit `f7a1fb9`、起手設定組合）實測**：10 步產出的
LoRA 中 **192 個 `lora_B` 全數非零**，**#1054 與 #1075 皆未重現**，因此**不必**為此放棄
base int8 路線或預設關掉 checkpointing。但仍建議正式訓練後跑一次自檢（成本近乎零）。

### 自檢指令（`scripts/check_lora.py`）

把訓好的 LoRA 下載回本機後：

```bash
python3 scripts/check_lora.py <下載的 LoRA.safetensors>
```

它會統計 `lora_B.weight` 張量總數與全為 0 的數量，並列出前幾個全零的 key。
退出碼語意：**0＝全部非零（正常）**、**1＝有全零的 `lora_B`（疑似空包彈）**、
**2＝檔案讀不到／格式不符**（例如不是 safetensors、或檔案內沒有 `lora_B`）。

若自檢回報疑似空包彈，且訓練時 checkpointing 與 offload 同時開啟，可先用
`TRAIN_GRAD_CP=false` 關掉 `gradient_checkpointing` 重訓（VRAM 需求上升）；
必要時再一併 `TRAIN_LAYER_OFFLOAD=false`。

### 步數建議

- **先 200 步冒煙測試**：確認 loss 有在動、能存檔、不會 OOM（也可把 `TRAIN_STEPS` 改小）。
- **主力結果看 1250–1750 步**：社群在 2.1 上的實測常見最佳點落在 1500 附近，
  1750 之後有機會出現皮膚顆粒化等過擬合現象。
- **每 250 步比較**：起手設定每 250 步存一次（`save_every: 250`）。比較時用**同一組
  prompt、seed、解析度、CFG**（2.1 固定 CFG=1），另外跑一個「無關 prompt」檢查有沒有外洩。
- 要填的 prompt 可直接複製 [Sample Prompts 範本](lora-sample-prompts.md)：含身分驗證、
  角度壓力測試、泳裝身材檢查、風格彈性與無觸發詞對照組，每組都是獨立可複製的區塊。
  該範本另附 **MOSS-Image 的取樣參數差異表**；換模型時請看那張表，不要照抄本頁的
  `guidance_scale` 與解析度倍數。

## 訓完：把 LoRA 放回 ComfyUI 使用

訓練輸出在機器上：`/workspace/ai-toolkit/output/<config.name>/`，
LoRA 是標準 safetensors：

```bash
# 下載主要權重檔（存到本機）
scp -P <PORT> root@<HOST>:/workspace/ai-toolkit/output/<config.name>/<config.name>.safetensors .
```

把下載的 `.safetensors` 放進 ComfyUI 的 `ComfyUI/models/loras/`，然後：

- 用**訓練用的同一個 base**（`qwen_image_2.1_int8_convrot`）載入，**不要**套在 Turbo 版上
  ——基底不同，效果會跑掉。
- 工作流：`UNETLoader(qwen_image_2.1_int8_convrot)` → `LoraLoaderModelOnly`／`LoraLoader`
  （套上你的 LoRA）→ 既有 Qwen-Image-2.1 節點鏈。
- prompt 帶上觸發詞（例如 `mychar, portrait, ...`）。

節點鏈的細節與常見錯誤對照，見 [ComfyUI Qwen-Image-2.1 節點說明](comfyui-qwen21-turbo.md)。

## 注意事項

- **訓練用 base，不是 Turbo**：Turbo 是 8 步蒸餾 checkpoint，適合推論；LoRA 訓在 base
  才能保有完整能力。本專案的佈建腳本下載的是 base（`qwen_image_2.1_int8_convrot`）。
- **文字編碼器是 Qwen3-VL-8B**，與 prompt 擴寫用的另一個模型檔名相近，勿搞混。
- **影像邊長須為 32 的倍數**（VAE 16× 空間壓縮 + RoPE）。
- **vast container 磁碟開機後不可調整**：模型＋快取＋optimizer state 要一次租夠
  （訓練模式預設 80GB）。
- **授權**：Qwen-Image-2.1 為 Qwen Research License（非商用）。如需商用請另行確認授權。
- **成本**：以 24GB 級卡（~US$0.3/hr）估，一輪 2–4 小時＋模型下載與安裝，
  單輪約 **US$1–3**，建議抓 **US$3–5** buffer（含試錯、重跑）。實際費率以租機當下為準。
- **Blackwell 世代（5070 Ti / 5080）在訓練情境的相容性未經本專案實測**；
  首次試訓建議先跑短步數（例如把 `steps` 改成 10）冒煙測試。
- **訓練完務必做一次空包彈自檢**：`python3 scripts/check_lora.py <LoRA.safetensors>`，
  確認 `lora_B` 非零（見「訓後自檢與步數」）。
