# ComfyUI Qwen-Image-2.1 Turbo：節點需求與設定

查證日期：2026-10-10。依據：ComfyUI v0.39.0 內建範本
`image_qwen_image_2_1_t2i.json`（comfyui-workflow-templates 0.11.78）、
官方模型卡 [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1)。
已在 RTX PRO 4000 Blackwell 24G 上實測出圖成功。

## 結論：需要專用節點，但 KSampler 是一般的

Qwen-Image-2.1 不能用 SD1.5／SDXL 那套標準工作流，但有兩個常見誤解：

1. **KSampler 不用換**——一般的 `KSampler` 就可以（官方範本也用它）。
   會噴錯通常是下面這些地方沒接對，而不是 KSampler 本身。
2. **文字編碼必須用專用節點** `TextEncodeQwenImage21`，不能用
   `CLIPTextEncode`。它吃 `prompt`／`negative_prompt` 兩個字串
   （不是單一 `text`），並且直接輸出 positive／negative 兩條 conditioning。

## 必要節點鏈

```
UNETLoader ──→ QwenImage21Cache ──→ KSampler ──→ VAEDecode ──→ SaveImage
CLIPLoader(type=qwen_image) ──→ TextEncodeQwenImage21 ──┬──→ KSampler(positive/negative)
EmptyLatentImage ──────────────────────────────────────→ KSampler(latent_image)
VAELoader ──→ VAEDecode
```

| 節點 | 關鍵設定 | 說明 |
| --- | --- | --- |
| `UNETLoader` | `qwen_image_2.1_turbo_int8_convrot.safetensors` | Qwen 是 diffusion model，用 UNETLoader，**不是** CheckpointLoaderSimple |
| `QwenImage21Cache` | device=`auto`、dtype=`default` | Qwen 2.1 的 KV 快取管理節點；官方範本放在模型和 KSampler 之間 |
| `CLIPLoader` | type=**`qwen_image`** | 文字編碼器是 `qwen3vl_8b_int8_convrot.safetensors`；type 選錯 KSampler 會噴錯 |
| `TextEncodeQwenImage21` | resolution=1024 | 專用節點；reference 圖是選填（image edit 用） |
| `EmptyLatentImage` | 1024×1024×1 | 一般的空 latent 即可，latent 不用從編碼節點來（純文字生圖） |
| `KSampler` | steps=8、cfg=1、euler、simple、denoise=1 | 官方明載 Turbo 版用 **8 steps** |
| `VAELoader` | `qwen_image_2.1_vae_bf16.safetensors` | Qwen 2.1 專用 VAE，不能拿別的模型 VAE 代用 |

## 模型檔案位置

```
ComfyUI/models/diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors
ComfyUI/models/text_encoders/qwen3vl_8b_int8_convrot.safetensors
ComfyUI/models/vae/qwen_image_2.1_vae_bf16.safetensors
```

（另有非 Turbo 的 `qwen_image_2.1_bf16`／`_int8_convrot` 與 Turbo bf16 版，
Turbo 也有 LoRA 版可用 base model + LoRA 跑。）

## 常見錯誤對照

| 症狀 | 原因 | 解法 |
| --- | --- | --- |
| `memory_efficient_attention_forward` 找不到 operator、
`fa3F/fa2F ... capability (12,0) (too new)` | 新架構 GPU（Blackwell，算力 12.0）跑不動
映像內建的 xformers kernel | 啟動參數加 `--use-pytorch-cross-attention`
（強制改用 PyTorch 內建 attention） |
| KSampler 報 conditioning／shape 錯誤 | 用了 `CLIPTextEncode` 或 CLIPLoader 的 type 不是 `qwen_image` | 改用 `TextEncodeQwenImage21` + type=`qwen_image` |
| CheckpointLoaderSimple 找不到模型 | Qwen 是 diffusion model 不是 checkpoint | 改用 `UNETLoader` |

## SillyTavern 對接

用 ComfyUI 的 **API 格式**工作流（Dev Mode → Save (API Format)），貼進
SillyTavern 的 ComfyUI Workflow Editor，或直接放進
`data/default-user/user/workflows/`。

現成的檔案：[`examples/qwen21-turbo-sillytavern-workflow.json`](../examples/qwen21-turbo-sillytavern-workflow.json)。

SillyTavern 端建議設定：**steps=8、cfg=1**（工作流用 `%steps%`／`%scale%`
佔位符，會吃 SillyTavern 這兩個欄位的值）；sampler/scheduler 已寫死
euler／simple。佔位符數值在 JSON 裡要用雙引號包起來
（`"%seed%"`、`"%width%"`），這是 SillyTavern 替換系統的要求。

## 參考

- 官方範本（含 UI 版子圖）：
  [image_qwen_image_2_1_t2i.json](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_t2i.json)
- SillyTavern ComfyUI 設定文件：
  [docs.sillytavern.app/extensions/stable-diffusion](https://docs.sillytavern.app/extensions/stable-diffusion/)
- 官方模型說明：
  [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1)
