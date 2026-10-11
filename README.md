# vastai-rent

在 [Vast.ai](https://vast.ai) 自動租用 GPU 主機的工具集。目標是把「掃描市場 →
篩選機器 → 租用 → 等開機 → 確認環境就緒」整段流程自動化，並提供刪除實例停止計費的入口。

> 專案剛成立，以下為規劃中的功能，實作進行中會陸續補上。

## 規劃中的功能

- **一鍵租機**：依條件（VRAM、算力、下載頻寬、流量費、PCIe 頻寬）篩選市面上的
  可租機器，自動建立實例、等待開機、確認環境就緒，最後回報 SSH 與服務連線資訊。
- **市場掃描**：只掃描不租用的輕量查詢，快速查看現有機型與價格。
- **釋放實例**：用完一鍵刪除，停止計費。

## 篩選條件（預設值，皆可用參數調整）

| 條件 | 預設門檻 |
| --- | --- |
| VRAM | ≥ 16 GB |
| 算力 | > 25 TFLOPS |
| 下載頻寬 | ≥ 800 Mb/s |
| 流量費 | ≤ $1/TB（上傳與下載） |
| PCIe 頻寬 | ≥ 16 GB/s（避免被閹割的卡） |
| GPU 類型 | 預設只選消費級 RTX 卡 |

## 使用前準備

1. 安裝 vast.ai CLI：`pip install vastai`
2. 設定 API key：`vastai set api-key <你的-key>`
   （key 可在 Vast.ai 帳號設定頁取得；也可以用 `VAST_API_KEY` 環境變數提供）
3. 綁定 SSH key 以便登入租到的機器：`vastai set ssh-key <你的公鑰檔>`

## 使用方式

```bash
# 一鍵租機（自動掃描、篩選、租下、等模型下載完、回報連線資訊）
python3 scripts/rent_gpu.py

# 常用選項
python3 scripts/rent_gpu.py --gpu RTX_3090,RTX_4090   # 指定 GPU 型號
python3 scripts/rent_gpu.py --min-vram 12000          # 最低 VRAM (MiB)
python3 scripts/rent_gpu.py --min-down 1000           # 最低下載 Mb/s（預設 800）
python3 scripts/rent_gpu.py --min-pcie 16             # 最低 PCIe 頻寬 GB/s（預設 0=不限）
python3 scripts/rent_gpu.py --offer-id 12345678       # 鎖定特定 offer（跳過篩選）
python3 scripts/rent_gpu.py --dry-run                 # 只掃描不租

# 用完刪除實例（停止計費）
python3 scripts/rent_gpu.py --destroy <實例ID>
```

租機時會自動設定 vast.ai 模板環境變數，關閉對外登入驗證
（`WEB_ENABLE_AUTH=false`、`ENABLE_AUTH=false`，兩套 image 家族都涵蓋），
並透過 `PROVISIONING_SCRIPT` 在開機時自動下載模型。
外部工具（如 SillyTavern）直接連回報的 tunnel URL 即可。

## 佈建內容（provisioning.sh）

開機自動下載 Qwen-Image-2.1 Turbo 的 int8 模型組（來源為
[Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1)，約 17.3 GB）：

| 類別 | 檔案 | 大小 |
| --- | --- | --- |
| 主模型 | `qwen_image_2.1_turbo_int8_convrot.safetensors` | 7.26 GB |
| 文字編碼器 | `qwen3vl_8b_int8_convrot.safetensors` | 9.35 GB |
| VAE | `qwen_image_2.1_vae_bf16.safetensors` | 0.68 GB |

並將 ComfyUI 更新至 ≥ v0.37.0（Qwen-Image-2.1 節點所需的最低版本）。

**下載方式**：模型一律走 `huggingface_hub`（Xet 高速傳輸、斷點續傳、下載後自動
驗 sha）；映像沒預裝時會自動補裝，補裝失敗才退回 `wget -c`。容器內的
`HF_TOKEN`／`HF_HOME` 環境變數會被自動讀取（本 repo 公開，無 token 也可下載）。
已在 `vastai/comfy` 映像實測：0.68GB 約 2 Gb/s、已存在檔案秒級跳過。

**Turbo 生圖參數**：steps=8、cfg=1、sampler=euler、scheduler=simple。

## 訓練模式（Qwen-Image-2.1 人物 LoRA）

除了推論，也能一鍵租一台裝好 [ai-toolkit](https://github.com/ostris/ai-toolkit)、
下載好 Qwen-Image-2.1 base（int8）訓練模型、訓練 GUI 可連的機器：

```bash
python3 scripts/rent_gpu.py --mode train            # 租訓練機
python3 scripts/rent_gpu.py --mode train --dry-run  # 只掃描
```

訓練模式改用 PyTorch (Vast) 模板、磁碟下限 80GB、VRAM 預設 ≥16GB 的消費級 RTX，
佈建腳本換成 `provisioning-train.sh`（裝 ai-toolkit、下載 base int8 三檔與官方設定檔、
生成起手 YAML、建立 `/workspace/dataset/`、啟動訓練 UI）。

完整用法、資料集格式、上傳方式、起手 YAML 欄位與訓完的 LoRA 如何放回 ComfyUI，
見 **[docs/qwen21-lora-training.md](docs/qwen21-lora-training.md)**。

## 注意：CLI 分頁上限

`vastai search offers` 預設只回傳前 64 筆（會漏掉大部分市場），
本腳本已加 `--limit 2000` 抓全量；自行使用 CLI 時記得加這個參數。

## 授權

MIT
