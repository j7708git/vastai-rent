#!/usr/bin/env python3
"""把 ai-toolkit 訓練的 MOSS-Image LoRA 併進 bf16 transformer（單檔模型）。

產出的 safetensors 與 Comfy-Org 的 `moss_image_sft_bf16.safetensors` 同一種鍵格式
（`layers.N.proj.*`，沒有 `diffusion_model.` 前綴），可直接用 ComfyUI 的
`Load Diffusion Model` 載入，不需要 LoRA 節點。

ai-toolkit 存檔時會把融合的 `proj.qkv` 拆成 q/k/v 三個 LoRA（共用同一個
`lora_A`）。這裡把它們併回 `proj.qkv`：`W[q|k|v] += cat(B_q,B_k,B_v) @ A`。

用法：
    python merge_moss_lora.py \
        --base models/diffusion_models/moss_image_sft_bf16.safetensors \
        --lora output/mibin1215_lora_v1.1/mibin1215_lora_v1.1.safetensors \
        --out models/merged/moss_image_sft_mibin1215_v1.1_bf16.safetensors
"""

import argparse
import collections
import os

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

QKV = ("q", "k", "v")


def build_deltas(lora_sd):
    """回傳 {(layer, target): (delta_fp32, rank_info)}。"""
    parts = collections.defaultdict(dict)
    for key, value in lora_sd.items():
        if not key.startswith("diffusion_model."):
            continue
        bits = key.split(".")
        # diffusion_model.layers.N.proj.<name>.(lora_A|lora_B).weight
        if len(bits) != 7 or bits[1] != "layers" or bits[3] != "proj":
            continue
        layer, name, kind = int(bits[2]), bits[4], bits[5]
        parts[(layer, name)][kind] = value.float()

    deltas = {}
    for (layer, name), entry in parts.items():
        a = entry.get("lora_A")
        b = entry.get("lora_B")
        if a is None or b is None:
            raise ValueError(f"layer {layer} {name}: 缺 lora_A 或 lora_B")
        target = "qkv" if name in QKV else name
        deltas.setdefault((layer, target), []).append((name, a, b))

    merged = {}
    for (layer, target), items in deltas.items():
        if len(items) == 1:
            _, a, b = items[0]
            merged[(layer, target)] = b @ a
        else:
            order = {name: i for i, name in enumerate(QKV)}
            items.sort(key=lambda x: order.get(x[0], 99))
            base_a = items[0][1]
            if all(torch.equal(base_a, a) for _, a, _ in items[1:]):
                # 訓練時 qkv 是融合的、共用同一個 A
                a = base_a
                b = torch.cat([b for _, _, b in items], dim=0)
            else:
                a = torch.cat([a for _, a, _ in items], dim=0)
                b = torch.block_diag(*[b for _, _, b in items])
            merged[(layer, target)] = b @ a
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"base: {args.base} ({os.path.getsize(args.base) / 1e9:.2f} GB)")
    state = load_file(args.base)

    lora_sd = load_file(args.lora)
    deltas = build_deltas(lora_sd)
    print(f"LoRA 模組數: {len(deltas)}")

    # 留兩個樣本做併完後的驗證
    probes = {}
    for (layer, target), delta in list(deltas.items()):
        if target in ("o", "ffn_down") and len(probes) < 2:
            key = f"layers.{layer}.proj.{target}.weight"
            probes[key] = (state[key].clone(), delta)

    for (layer, target), delta in deltas.items():
        key = f"layers.{layer}.proj.{target}.weight"
        if key not in state:
            raise KeyError(f"base 找不到 {key}")
        weight = state[key]
        if weight.shape != delta.shape:
            raise ValueError(f"{key}: base {tuple(weight.shape)} vs delta {tuple(delta.shape)}")
        weight.add_(delta.to(weight.dtype))

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    save_file(state, args.out)
    print(f"saved: {args.out} ({os.path.getsize(args.out) / 1e9:.2f} GB)")

    # 驗證：重新讀檔，抽兩個模組確認差值等於預期
    with safe_open(args.out, "pt") as f:
        for key, (before, delta) in probes.items():
            after = f.get_tensor(key).float()
            got = (after - before.float()).abs().max().item()
            want = delta.abs().max().item()
            print(f"check {key}: delta_max={want:.6f} 讀回誤差={got:.6f}")


if __name__ == "__main__":
    main()
