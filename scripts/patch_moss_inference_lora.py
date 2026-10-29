#!/usr/bin/env python3
"""在 ai-toolkit 的 MOSS-Image 模型加上「只在取樣時啟用」的額外 LoRA。

背景：ai-toolkit 的 `model.inference_lora_path` 只在部分架構有實作
（flux、qwen_image_2、krea2…），MOSS 沒有。直接設定的話
`BaseModel.generate_images()` 會存取 `self.assistant_lora.is_active`，
而它是 None → 第一次取樣就 AttributeError。

用途：訓練時疊一顆 4-step 蒸餾 LoRA 當預覽（4 步、CFG 1），
讓 sample 圖反映實際推論設定，而不是 base 的 30 步 CFG 3。

為什麼不用 ai-toolkit 的 LoRASpecialNetwork：
  MOSS 的 qkv 是**融合**的單一 Linear（`proj.qkv`，out=9216），
  而外部 LoRA（如 akhaliq 的 4-step）是 q/k/v **分開**、且各投影的
  down 不共用 → 融合後 rank 會變成 3r，跟訓練 LoRA 的 rank 不同。
  LoRASpecialNetwork 每個 network 只有一個 rank，硬套會形狀不符。
  這裡改成獨立的小型 forward hook：逐模組把 delta 加上去，
  不受 rank／命名慣例限制，也不影響訓練中的 LoRA。

做法：
  1. `load_model()` 載入 transformer 後，若 `inference_lora_path` 有值
     就呼叫 `load_inference_lora()`。
  2. 該方法把外部 LoRA 依 `layers.<n>.proj.<q|k|v|o|ffn_up_gate|ffn_down>`
     分組，q/k/v 併成 block-diag（對應融合的 `proj.qkv`），
     在對應的 Linear 上註冊 forward hook，預設不啟用。
  3. 覆寫 `generate_images()`：取樣期間打開旗標，結束關閉。
  4. 載入後把 `model_config.inference_lora_path` 清成 None，
     避免 `BaseModel.generate_images()` 去碰 `assistant_lora`。

精度：權重保持 fp32 放 GPU，delta 以 fp32 計算後才轉回 bf16。
      官方模型卡提醒「merge 進 bf16 會先毀掉小字」，不要 merge。

用法：
    python3 patch_moss_inference_lora.py /workspace/ai-toolkit
"""

import argparse
import pathlib
import sys

MARKER = "def load_inference_lora"

CALL_OLD = """        transformer.aitk_post_load(**self.component_load_kwargs("transformer"))
        flush()
"""
CALL_NEW = """        transformer.aitk_post_load(**self.component_load_kwargs("transformer"))
        flush()

        if self.model_config.inference_lora_path is not None:
            self.load_inference_lora(transformer)
"""

METHOD_OLD = """    # ------------------------------------------------------------------
    # Generation (training previews)
    # ------------------------------------------------------------------
    def get_generation_pipeline(self):
"""

METHOD_NEW = '''    # ------------------------------------------------------------------
    # Sampling-only LoRA (e.g. a 4-step distill)
    # ------------------------------------------------------------------
    def load_inference_lora(self, transformer: MossImageTransformer):
        """把外部 LoRA 掛成取樣時才啟用的 forward hook。

        不 merge、也不建立 LoRASpecialNetwork：MOSS 的 qkv 是融合的單一
        Linear，而外部 LoRA 多為 q/k/v 分開且 down 不共用，rank 會是 3r，
        與訓練 LoRA 的 rank 不同，硬套會形狀不符。
        """
        import collections
        import re

        from huggingface_hub import hf_hub_download

        lora_path = self.model_config.inference_lora_path
        if not os.path.exists(lora_path):
            repo_id, _, filename = lora_path.rpartition("/")
            if not repo_id:
                raise ValueError(f"inference_lora_path not found: {lora_path}")
            lora_path = hf_hub_download(repo_id=repo_id, filename=filename)
        self._inference_lora_path = lora_path

        state_dict = load_file(lora_path)
        pattern = re.compile(
            r"diffusion_model\\.layers\\.(\\d+)\\.proj\\.([a-z_]+)\\.(.+)$"
        )
        parts = collections.defaultdict(dict)
        for key, value in state_dict.items():
            match = pattern.match(key)
            if match is None:
                continue
            layer, name, tail = match.groups()
            parts[(int(layer), name)][tail] = value
        if not parts:
            raise ValueError(
                f"inference LoRA has no diffusion_model.layers.*.proj.* keys: {lora_path}"
            )

        # (layer, target) -> list of (A, B) 已含 alpha/rank 縮放，順序 q,k,v
        grouped = collections.defaultdict(list)
        for (layer, name), entry in sorted(parts.items()):
            down = entry.get("lora_down.weight", entry.get("lora_A.weight"))
            up = entry.get("lora_up.weight", entry.get("lora_B.weight"))
            if down is None or up is None:
                continue
            rank = down.shape[0]
            alpha = entry.get("alpha", None)
            scale = float(alpha) / float(rank) if alpha is not None else 1.0
            target = "qkv" if name in ("q", "k", "v") else name
            grouped[(layer, target)].append((name, down.float(), up.float() * scale))

        hooks = []
        for (layer, target), entries in grouped.items():
            block = transformer.layers[layer]
            module = block.proj[target]
            if len(entries) == 1:
                a_mat = entries[0][1]
                b_mat = entries[0][2]
            else:
                order = {"q": 0, "k": 1, "v": 2}
                entries.sort(key=lambda e: order.get(e[0], 99))
                a_mat = torch.cat([e[1] for e in entries], dim=0)
                b_mat = torch.block_diag(*[e[2] for e in entries])
            a_mat = a_mat.to(self.device_torch)
            b_mat = b_mat.to(self.device_torch)
            hooks.append(
                module.register_forward_hook(
                    self._make_inference_lora_hook(self, a_mat, b_mat)
                )
            )

        self._inference_lora_hooks = hooks
        self._inference_lora_active = False
        # 已自行處理，避免 BaseModel.generate_images() 去碰 assistant_lora
        self.model_config.inference_lora_path = None
        self.print_and_status_update(
            f"Loaded inference LoRA ({len(hooks)} modules) from {lora_path}"
        )

    @staticmethod
    def _make_inference_lora_hook(owner, a_mat, b_mat):
        def hook(module, inputs, output):
            if not owner._inference_lora_active:
                return output
            x = inputs[0]
            delta = torch.matmul(torch.matmul(x.float(), a_mat.t()), b_mat.t())
            return output + delta.to(output.dtype)

        return hook

    def generate_images(self, *args, **kwargs):
        hooks = getattr(self, "_inference_lora_hooks", None)
        if not hooks:
            return super().generate_images(*args, **kwargs)
        self._inference_lora_active = True
        try:
            return super().generate_images(*args, **kwargs)
        finally:
            self._inference_lora_active = False

    # ------------------------------------------------------------------
    # Generation (training previews)
    # ------------------------------------------------------------------
    def get_generation_pipeline(self):
'''


def patch(path: pathlib.Path) -> int:
    src = path.read_text(encoding="utf-8")
    if MARKER in src:
        print(f"already patched: {path}")
        return 0

    if CALL_OLD not in src:
        print(f"anchor not found (load_model): {path}", file=sys.stderr)
        return 1
    if METHOD_OLD not in src:
        print(f"anchor not found (get_generation_pipeline): {path}", file=sys.stderr)
        return 1

    src = src.replace(CALL_OLD, CALL_NEW, 1)
    src = src.replace(METHOD_OLD, METHOD_NEW, 1)
    path.write_text(src, encoding="utf-8")
    print(f"patched: {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "toolkit_root",
        nargs="?",
        default="/workspace/ai-toolkit",
        help="ai-toolkit 根目錄（預設 /workspace/ai-toolkit）",
    )
    args = ap.parse_args()
    target = (
        pathlib.Path(args.toolkit_root)
        / "extensions_built_in"
        / "diffusion_models"
        / "moss_image"
        / "moss_image.py"
    )
    if not target.is_file():
        print(f"not found: {target}", file=sys.stderr)
        return 1
    return patch(target)


if __name__ == "__main__":
    raise SystemExit(main())
