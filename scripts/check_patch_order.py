# -*- coding: utf-8 -*-
"""Reproduce the two typical patchify mistakes, for the report's bug analysis.

The correct implementation separates the two spatial grid axes before
flattening. The first wrong variant below only reshapes, so the flattening
order mixes rows of different patches. The second wrong variant forgets the
transpose step entirely. Both keep the array shape [B,49,3072], so only the
numerical check can catch them.

Run from the project root:
    python scripts/check_patch_order.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime import load_images, load_weights  # noqa: E402
from image_encoder import ImageEncoder  # noqa: E402


def main():
    images = load_images()
    encoder = ImageEncoder(load_weights())
    with np.load(ROOT / 'assets/reference_outputs.npz', allow_pickle=False) as ref:
        reference = ref['patches']

    batch = images.shape[0]
    correct = encoder.patchify(images)

    # Wrong variant 1: one reshape over C,H,W without separating the 7x7 grid.
    wrong_flat = images.reshape(batch, 49, 3072)

    # Wrong variant 2: reshape + transpose of the whole axis list (a common
    # first attempt); it returns [B,49,3072] but the patch order is wrong.
    wrong_transpose = images.reshape(batch, 3, 7, 32, 7, 32).transpose(
        0, 3, 5, 2, 4, 1).reshape(batch, 49, 3072)

    result = {
        'correct_max_abs': float(np.abs(correct - reference).max()),
        'wrong_flat_vs_reference_max_abs': float(np.abs(wrong_flat - reference).max()),
        'wrong_flat_vs_reference_mean_abs': float(np.abs(wrong_flat - reference).mean()),
        'wrong_transpose_vs_reference_max_abs': float(
            np.abs(wrong_transpose - reference).max()),
        'wrong_flat_projected_token_max_abs': float(
            np.abs(wrong_flat @ encoder.patch_weight.T - correct @ encoder.patch_weight.T).max()),
    }
    print(json.dumps(result, indent=2))
    (ROOT / 'artifacts' / 'patch_order_check.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
