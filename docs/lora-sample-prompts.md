# 人物 LoRA 訓練用的 Sample Prompts 範本

訓練人物 LoRA 時，ai-toolkit 會在訓練途中定期用固定的 prompt 生成預覽圖。
這份範本的用途是**驗證「像不像」與「有沒有過擬合」**，不是生好看的圖——
所以刻意做成壓力測試，而不是每個人都會拍的那種照片。

範例用觸發詞 `Lily`；換成你自己的觸發詞即可（注意事項見文末）。
prompt 內容本身與模型無關，但**取樣參數必須跟著模型改**，見下一節。

## 先確認你的模型：取樣參數不要照抄

同樣的 prompt 丟進不同模型，能比對的前提是取樣設定正確。兩個常見模型的差異：

| 項目 | MOSS-Image | Qwen-Image-2.1 |
| --- | --- | --- |
| `guidance_scale` | **3.0**（建議 2.5–6；prompt 越短越需要調高） | **1.0**（CFG-distilled，>1 會偏離真實推論） |
| `sample_steps` | **30–50**（官方推論建議至少 50） | 40 |
| sampler | flowmatch（官方推論建議 Euler） | flowmatch |
| shift | 5.0 | — |
| 邊長倍數 | **16** 的倍數 | **32** 的倍數 |
| 建議總像素 | 約 1M／1.5M（支援 0.25M–4M） | 依訓練解析度 |
| 特別注意 | 訓練與取樣都**排除 block 22**（SLG 用的那一層）；預覽若沒用同一設定，看到的圖與實際推論不一致 | — |

> 上表 MOSS-Image 一欄依 ai-toolkit 的 `moss_image` 實作與 OpenMOSS 官方推論建議整理
> （2026-10 版本）。換版本或換模型時請重新確認，不要沿用。

## 在 AI Toolkit 哪裡填

Job 詳細頁 → 編輯 → **Samples** 區塊的 **Sample Prompts**，一行一組。

| 設定 | 建議值 | 說明 |
| --- | --- | --- |
| `sample_every` | 與 `save_every` 相同（`250`） | 每個 checkpoint 都配一張對照圖，方便逐輪比較 |
| 一次放幾組 | **4–6 組** | 取樣會拖慢訓練；其餘留到訓完用 ComfyUI 補測 |

全身照建議另外用**直式**跑（MOSS-Image 用 `832 × 1216`，約 1.01M 像素；
Qwen-Image-2.1 用 `1024 × 1536`），人物佔比才夠，正方形裡全身會太小。

## 撰寫原則

| 原則 | 原因 |
| --- | --- |
| 觸發詞放最前面 | 讓模型先鎖定主體，後面的描述才不會被當成主體 |
| **不要寫五官細節**（臉型、眼睛大小、鼻子高低） | 那些應該由 LoRA 學。寫進 prompt 等於用文字硬拉，會掩蓋真實效果 |
| 一組只變一個維度 | 同時換場景又換角度，出問題時分不清是哪個因素 |
| 保留至少一組**沒有觸發詞** | 檢查訓練集的概念有沒有外洩到一般生成 |
| MOSS-Image 用完整句子更穩 | 它走 Qwen3.5-9B 的 chat template，長 prompt 在 CFG 3 下運作良好；tag 串也吃，但句子較不易被斷章取義 |

## A. 身分驗證（正面、乾淨光源）

```text
photo of Lily, a woman, close-up portrait, natural window light, looking at camera, neutral expression, 85mm lens, shallow depth of field, sharp focus on eyes
```

```text
Lily, candid photo, laughing, outdoors in a park, soft afternoon sunlight, 50mm lens, slight film grain
```

```text
Lily, full body photo, standing, wearing a white t-shirt and jeans, plain gray background, studio lighting, sharp focus
```

```text
Lily, three-quarter view portrait, side lighting, black background, dramatic shadows, detailed skin texture
```

## B. 角度與距離壓力測試

最容易看出「有沒有真的學到臉」，而不是只學到一個正面模板。

```text
Lily, profile view, looking to the side, soft overcast daylight, 135mm lens
```

```text
Lily, extreme close-up of the face, no makeup, visible skin texture, harsh direct flash
```

```text
Lily, wide shot, walking on a city street at night, neon reflections, blurred background, 35mm lens
```

## C. 情境與服裝（測是否只會複製訓練集）

```text
Lily, wearing a black business suit, office interior, sitting at a desk, corporate headshot style
```

```text
Lily, wearing a red summer dress, beach at sunset, backlit, warm tones
```

```text
Lily, winter coat and scarf, snowy street, cold blue tones, documentary photography
```

## D. 泳裝／身材檢查

衣服遮蔽少，肩寬、腰線、髖寬、腿長比例可以直接對照本人照片。
這組刻意固定取景距離與姿勢，**方便逐項比對，不是為了好看**。

### D-1 正面全身（基準圖）

```text
Lily, full body photo, standing straight facing the camera, arms relaxed at sides, wearing a simple black one-piece swimsuit, plain light gray background, even soft studio lighting, 50mm lens, sharp focus, entire body in frame
```

每一輪 checkpoint 都先看這張，對照肩寬、髖寬、腰的位置、腿長佔比。

### D-2 側面全身（比例最難蒙混的角度）

```text
Lily, full body photo, side profile standing, wearing a dark two-piece swimsuit, plain light gray background, soft studio lighting, 50mm lens, entire body in frame
```

胸部、臀部、腹部曲線與腿部前後比例，靠 prompt 幾乎無法造假。

### D-3 背面全身

```text
Lily, full body photo from behind, standing, wearing a one-piece swimsuit, plain background, even lighting, 50mm lens, entire body in frame
```

### D-4 實景（測背景與膚色是否互相干擾）

```text
Lily, full body photo, wearing a blue one-piece swimsuit, standing on a beach, overcast daylight, no harsh shadows, 35mm lens, entire body in frame
```

```text
Lily, full body photo, three-quarter angle, wearing a white bikini, standing beside a swimming pool, bright daylight, 50mm lens
```

### D-5 坐姿（壓力測試，比例一定變形）

```text
Lily, full body photo, sitting on the edge of a pool with legs in the water, wearing a one-piece swimsuit, daylight, 35mm lens
```

> 這張不是用來比對比例，而是看模型遇到遮擋與關節彎折時會不會崩壞（多手多腳、關節錯位）。

## E. 風格彈性（可選，1–2 組）

```text
Lily, black and white portrait, high contrast, studio, editorial fashion photography
```

## F. 對照組（不放觸發詞）

```text
photo of a woman, close-up portrait, natural window light, 85mm lens
```

用來確認「沒有觸發詞時不會硬套那張臉」。若這張也長得像本人，代表訓練集的概念外洩，
通常是 caption 沒帶觸發詞，或訓練步數過多。

## G. 文字渲染檢查（MOSS-Image 專屬，選配）

MOSS-Image 的強項是畫面內文字；加 LoRA 之後文字若開始糊掉或拼錯，代表訓練影響到
不該動的地方。

```text
Lily, standing in a cafe, holding a white paper cup with the words "LILY" printed in black bold letters, natural daylight, 50mm lens, sharp focus on the cup
```

## 判讀重點

| 現象 | 代表什麼 | 處置 |
| --- | --- | --- |
| 每一張都長一樣、皮膚像蠟 | likeness 飽和／過擬合 | 提早停，或改用更早的 checkpoint |
| 臉像，但身材被拉成「平均值」（腰變細、腿變長） | 全身照樣本不足 | 補全身／不同距離的照片進訓練集 |
| 換背景就崩、人物邊緣糊掉 | 訓練集背景太單一 | 補不同場景的照片，或降背景權重 |
| 無觸發詞也出現那張臉 | 概念外洩（單人資料集的常見現象） | 檢查 caption 是否漏帶觸發詞、步數是否過多；必要時混入少量非本人照片 |
| 膚色被拉平、失去膚色不均的細節 | 膚色樣本少或學習率問題 | 補不同光源下的照片；先確認不是過擬合 |
| 泳裝邊緣融進皮膚、接縫糊成一團 | 模型沒學到衣物與皮膚的分界 | 通常同時意味著全身照不足 |
| 每張都比本人「好看」一截 | **可能不是 LoRA，是模型本身的寫實人像傾向** | 用同 prompt 跑未套 LoRA 的 base 對照，扣掉這個 baseline 再判斷 |

比較時**固定同一組 prompt、seed、解析度、CFG**，只換 checkpoint，否則差異分不清是步數
還是參數造成的。

## 換成自己的設定

- **觸發詞別用常見字。** `Lily` 同時是常見英文名與花名，模型本身已有既有概念，
  容易互相干擾。若相似度一直上不去，可換成不常見的 token（例如 `lilyperson`、`ohwx lily`）重訓。
- **性別與攝影語彙**：範例以 `a woman` 與寫實攝影為主；若主體不是女性，或想測動畫／插畫風格，
  整組的 gender 與攝影詞需要一併替換（只改一處會造成 prompt 內部矛盾）。
- **觸發詞要與訓練集的 caption 一致**，否則取樣圖測到的不是 LoRA 的實際行為。
- **角色 LoRA 的 caption 不要描述外貌**（髮色、瞳色、臉型），那是要讓 LoRA 學的東西；
  寫進 caption 會削弱學習效果。

## 授權提醒

真人素材的同意已取得（是本人），訓練與取樣都不必再確認，也不需要標示來源。
只有要公開或商用時才需要看底模的商用限制。
