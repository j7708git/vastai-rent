#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""檢查 ai-toolkit 訓練出的 LoRA 是否為「空包彈」。

讀入一個 LoRA 的 `.safetensors`，統計其中 `lora_B.weight` 張量的總數，以及
「全為 0」的數量，並列出前幾個全零的 key。

為什麼要檢查：ai-toolkit 的 issue #1054（closed 未修）與 #1075（open）回報，
在特定設定下訓練 loss 看起來正常，但存出的 LoRA 有整批 `lora_B` 全為 0，
在 ComfyUI 載入後完全沒效果。`lora_B` 是零初始化，若梯度自始為 0 就會永遠為 0，
所以「是否有非零的 `lora_B`」是判斷空包彈最直接的方法。

退出碼：
  0  所有 lora_B.weight 都有非零值
  1  存在全為 0 的 lora_B.weight（疑似空包彈）
  2  檔案讀不到或格式不符（例如檔案不存在、不是 safetensors、
     或檔案內沒有任何 lora_B.weight）

用法：
  python3 scripts/check_lora.py <LoRA.safetensors>
"""

from __future__ import annotations

import argparse
import sys

EXIT_OK = 0
EXIT_EMPTY = 1
EXIT_ERROR = 2

B_SUFFIX = "lora_B.weight"
A_SUFFIX = "lora_A.weight"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="check_lora.py",
        description=(
            "檢查 LoRA safetensors 是否為空包彈：統計 lora_B.weight 張量總數與"
            "全為 0 的數量。退出碼 0=全部非零、1=有全零、2=讀不到或格式不符。"
        ),
        epilog="例：python3 scripts/check_lora.py output/mychar/mychar.safetensors",
    )
    parser.add_argument(
        "path",
        nargs="?",
        metavar="LORA.safetensors",
        help="要檢查的 LoRA safetensors 檔案路徑",
    )
    parser.add_argument(
        "--max-list",
        type=int,
        default=8,
        metavar="N",
        help="最多列出幾個全零的 key（預設 8）",
    )
    return parser.parse_args(argv)


def _open_tensors(path):
    """開啟 safetensors，回傳 (handle, framework)。

    safetensors 的 import 刻意放在這裡（不在檔案頂端），讓 `--help` 在沒有安裝
    safetensors 的環境也能執行。優先使用 numpy 後端以省去載入 torch。
    """
    from safetensors import safe_open

    last_err = None
    for framework in ("np", "pt"):
        try:
            return safe_open(path, framework=framework), framework
        except Exception as exc:  # noqa: BLE001 - 後端不可用時改試下一個
            last_err = exc
    raise RuntimeError(
        "無法開啟 safetensors（np / pt 後端皆失敗）：{}".format(last_err)
    )


def _max_abs(tensor):
    """回傳張量絕對值最大值（同時支援 numpy array 與 torch tensor）。"""
    return float(abs(tensor).max().item())


def analyze(path, max_list=8):
    """回傳 (exit_code, 報告字串)。"""
    try:
        handle, framework = _open_tensors(path)
    except ImportError:
        return EXIT_ERROR, (
            "錯誤：找不到 safetensors 套件，無法讀取檔案。\n"
            "      請先安裝（例如 pip install safetensors，或改用訓練環境的 venv）後再執行。"
        )
    except Exception as exc:  # noqa: BLE001
        return EXIT_ERROR, "錯誤：無法讀取檔案 {}：{}".format(path, exc)

    # safe_open 的 handle 只在 with 區塊內有效，所有張量讀取都要在這裡完成。
    with handle as f:
        keys = list(f.keys())
        b_keys = sorted(k for k in keys if k.endswith(B_SUFFIX))
        a_keys = sorted(k for k in keys if k.endswith(A_SUFFIX))

        if not b_keys:
            return EXIT_ERROR, (
                "錯誤：檔案 {} 內找不到任何 lora_B.weight 張量（共 {} 個張量）。\n"
                "      這不像 ai-toolkit 產出的 LoRA safetensors；請確認路徑是否正確。".format(
                    path, len(keys)
                )
            )

        zero_keys = [k for k in b_keys if _max_abs(f.get_tensor(k)) == 0.0]
        a_zeros = sum(1 for k in a_keys if _max_abs(f.get_tensor(k)) == 0.0)

    lines = []
    lines.append("LoRA 自檢報告")
    lines.append("  檔案            : {}".format(path))
    lines.append("  後端            : safetensors/{}".format(framework))
    lines.append("  張量總數        : {}".format(len(keys)))
    lines.append("  lora_A.weight   : {} 個（其中全零 {} 個，僅供參考）".format(
        len(a_keys), a_zeros,
    ))
    lines.append("  lora_B.weight   : {} 個".format(len(b_keys)))
    lines.append("  其中全為 0 的   : {} 個".format(len(zero_keys)))

    if zero_keys:
        lines.append("")
        lines.append("疑似空包彈：以下 lora_B 全為 0（最多列 {} 個）：".format(max_list))
        for k in zero_keys[:max_list]:
            lines.append("  - {}".format(k))
        if len(zero_keys) > max_list:
            lines.append("  ...（其餘 {} 個略）".format(len(zero_keys) - max_list))
        lines.append("")
        lines.append(
            "建議：此 LoRA 在推論端很可能完全無效。若訓練時 gradient_checkpointing "
            "與 layer_offloading 同時開啟，可試著關掉 gradient_checkpointing "
            "（環境變數 TRAIN_GRAD_CP=false）後重新訓練；詳見 docs/qwen21-lora-training.md。"
        )
        return EXIT_EMPTY, "\n".join(lines)

    lines.append("")
    lines.append("通過：所有 lora_B 都有非零值（不是空包彈）。")
    return EXIT_OK, "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    if not args.path:
        print("錯誤：請提供要檢查的 LoRA safetensors 路徑（或加 --help 看用法）。",
              file=sys.stderr)
        return EXIT_ERROR
    code, report = analyze(args.path, max_list=args.max_list)
    stream = sys.stdout if code != EXIT_ERROR else sys.stderr
    print(report, file=stream)
    return code


if __name__ == "__main__":
    sys.exit(main())
