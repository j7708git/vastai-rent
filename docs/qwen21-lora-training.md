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
# 3) SSH 進去開始訓練
# 4) 下載訓好的 LoRA，放回 ComfyUI 使用
# 5) 用完刪除實例，停止計費
python3 scripts/rent_gpu.py --destroy <實例ID>
```

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
- **不需自行裁切或縮放**：loader 會自動依 aspect ratio 分桶（bucket）並縮放。
- 建議 **15–30 張**，涵蓋不同角度、表情、姿勢、服裝、背景；
  只放你希望推論時看到的視角。
- caption 建議描述「會變動的屬性」（服裝、場景、動作），
  固定不變的辨識特徵（例如標誌性髮型）不必寫進 caption。

**觸發詞（trigger word）**：用罕見 token（如 `ohwx`、`sks` 或自造字）把「這個人」
綁到該 token。ai-toolkit 的行為是：

- caption 內寫 `[trigger]` 佔位符 → 執行時替換成設定裡的 `trigger_word`。
- 若設定了 `trigger_word` 而 caption 沒提到它 → 會自動補進 caption（已含則不重複加）。

所以兩種寫法都可用：直接在 caption 寫觸發詞，或在 caption 寫 `[trigger]` 並靠設定替換。
起手設定已帶一個 `trigger_word` 佔位；不想要設定檔動 caption 時，把該行註解掉即可。

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

2. **Instance Portal tunnel**：vast.ai 的 Instance Portal 會替 UI 建立
   Cloudflare tunnel 連結，可從實例卡片的 Open／tunnel 資訊取得
   （回報的 URL 有時導向 Portal 本身，非 UI 本體；以 SSH 埠轉發為準）。

UI 本身不需一直開著，訓練啟動後關掉瀏覽器不影響訓練；它只用來啟動／停止／監看 job。

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
| `train.steps` | 2000 | 總步數；建議 1500–3000，每 250 步比較 |
| `train.optimizer` | `adamw8bit` | 8-bit AdamW，省 VRAM |
| `train.lr` | `0.0001` | 學習率 |
| `train.noise_scheduler` / `timestep_type` | `flowmatch` / `shift` | 2.1 的 flow-matching 設定 |
| `train.gradient_checkpointing` | `true` | 省 VRAM，略慢 |
| `train.disable_sampling` | `true` | 訓練中不出樣本圖（省 VRAM／時間） |
| `model.arch` | `qwen_image_2` | 2.1 專用架構；別與初代 Qwen-Image 混用 |
| `model.name_or_path` | `Comfy-Org/Qwen-Image-2.1` | 權重來源（int8 量化 repack） |
| `model.extras_name_or_path` | `Qwen/Qwen-Image-2.1` | 設定檔／processor 來源 |
| `model.quantize` / `qtype` | `true` / `convrot8` | 直接以 int8 量化權重訓練，省 VRAM |
| `model.low_vram` / `layer_offloading` | `true` / `true` | 低 VRAM 模式＋CPU 層卸載 |

**12–16GB 卡的建議**：保留 `low_vram`、`layer_offloading`（含兩個 `*_percent: 1.0`）、
`optimizer: adamw8bit`、`disable_sampling: true`，並可把 `resolution` 降到 512–768。

改完 YAML 後可另存不同檔名（放在 `config/` 下），再用
`./train.sh config/<你的檔名>.yaml` 執行指定的設定；不帶參數時預設跑
`config/train_qwen21_character_lora.yaml`。

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
