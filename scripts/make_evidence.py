# -*- coding: utf-8 -*-
"""Generate the numerical evidence and figures used by the experiment report.

Run from the project root:
    python scripts/make_evidence.py

Outputs
    artifacts/stage_errors.csv     per-stage max/mean absolute error
    artifacts/evidence.json        all measured numbers used in the report
    artifacts/output_tokens.npz    output tokens of the ten supplied images
    figures/fig1_architecture.png  encoder data-flow diagram
    figures/fig2_patchify.png      patch decomposition of image_01
    figures/fig3_errors.png        per-stage numerical error chart
    figures/fig4_outputs.png       unit norms, cosine similarity, CLS attention
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime import load_weights, load_images, image_records  # noqa: E402
from image_encoder import ImageEncoder  # noqa: E402
from helpers import attention_weights  # noqa: E402
from preprocess_image import PreprocessImage  # noqa: E402

FIG = ROOT / 'figures'
ART = ROOT / 'artifacts'
FIG.mkdir(exist_ok=True)
ART.mkdir(exist_ok=True)

# Chinese labels in the figures: register the Windows font explicitly.
for candidate in (r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\simhei.ttf'):
    if Path(candidate).exists():
        font_manager.fontManager.addfont(candidate)
        break
plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)[:, None, None]
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)[:, None, None]


def compare(actual, expected):
    error = np.abs(actual - expected)
    atol = 2e-5 if expected.shape[-1] == 512 else 2e-4
    if expected.shape == (10, 3, 224, 224):
        atol = 2e-6
    ok = bool(np.allclose(actual, expected, rtol=2e-4, atol=atol))
    return {
        'shape': list(actual.shape),
        'dtype': str(actual.dtype),
        'max_abs': float(error.max()),
        'mean_abs': float(error.mean()),
        'atol': atol,
        'pass': ok,
    }


def attention_maps(encoder, image):
    """Recompute the arithmetic of forward once, capturing CLS attention.

    The captured coefficient rows are only used for visualization; the output
    is cross-checked against encode_tokens to prove the replay is faithful.
    """
    x = encoder.pre_norm(encoder.embed(image))
    captured = {}
    for index, block in enumerate(encoder.blocks):
        attn = block.attention
        n1 = block.norm1(x)
        qkv = n1 @ attn.qkv_weight.T + attn.qkv_bias
        width = attn.width
        q = attn.split_heads(qkv[:, :, :width])
        k = attn.split_heads(qkv[:, :, width:2 * width])
        v = attn.split_heads(qkv[:, :, 2 * width:])
        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(np.float32(attn.head_dim))
        coefficients = attention_weights(scores)
        if index in (0, 11):
            # CLS query, all 49 patch keys, averaged over the 12 heads.
            captured[index] = coefficients[:, :, 0, 1:].mean(axis=1)[0].reshape(7, 7)
        context = coefficients @ v
        merged = context.transpose(0, 2, 1, 3).reshape(1, 50, width)
        update = merged @ attn.out_weight.T + attn.out_bias
        residual = x + update
        u = block.norm2(residual)
        hidden = u @ block.fc1_weight.T + block.fc1_bias
        activated = block.gelu(hidden)
        x = residual + (activated @ block.fc2_weight.T + block.fc2_bias)
    return captured, x


def figure_architecture():
    """Compact left-to-right data-flow diagram of the implemented encoder."""
    fig, ax = plt.subplots(figsize=(15.2, 5.6), dpi=200)
    ax.set_xlim(0, 15.2)
    ax.set_ylim(0, 5.6)
    ax.axis('off')

    boxes = [
        (0.15, 3.15, 2.05, 1.35, '输入图像\n[B,3,224,224]', '#EAF2FB', '#2F6DB5'),
        (2.55, 3.15, 1.95, 1.35, 'patchify\n→ [B,49,3072]', '#EAF2FB', '#2F6DB5'),
        (4.85, 3.15, 2.30, 1.35, 'patch 投影 (无偏置)\n→ [B,49,768]', '#EAF7F0', '#2E8B57'),
        (7.50, 3.15, 2.35, 1.35, '+CLS +位置编码\n→ [B,50,768]', '#FFF6E5', '#C77700'),
        (10.20, 3.15, 2.60, 1.35, 'LN → 12×Transformer\n→ [B,50,768]', '#FDEEF3', '#B03A6B'),
        (13.15, 3.15, 1.95, 1.35, 'CLS/LN/投影\n→ [B,512]', '#F2EDF9', '#6A3FA0'),
    ]
    for x, y, w, h, text, face, edge in boxes:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0.04,rounding_size=0.10',
            linewidth=1.4, facecolor=face, edgecolor=edge))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=10.5, color='#1F2937', linespacing=1.45)
    for start, end in ((2.20, 2.55), (4.50, 4.85), (7.15, 7.50), (9.85, 10.20), (12.80, 13.15)):
        ax.annotate('', xy=(end, 3.83), xytext=(start, 3.83),
                    arrowprops=dict(arrowstyle='-|>', color='#5B6B7C', lw=1.3,
                                    shrinkA=0, shrinkB=0))

    # Detail of one Transformer block.
    ax.add_patch(mpatches.FancyBboxPatch((1.35, 0.42), 12.5, 1.85,
                                         boxstyle='round,pad=0.04,rounding_size=0.12',
                                         linewidth=1.2, facecolor='#FAFBFC',
                                         edgecolor='#9AA7B4', linestyle='--'))
    ax.text(1.55, 2.02, '单个 Transformer Block（12 层结构相同、参数独立）',
            fontsize=10.5, color='#334155', fontweight='bold')
    ax.text(13.62, 2.02, 'LN ε=1e-5；GELU 采用 erf 精确式',
            fontsize=9.0, color='#475569', ha='right', va='center')
    inner = [
        (1.65, 0.72, 2.35, 1.05, 'LN1 → 多头自注意力\n(12 头, 64 维/头)'),
        (4.35, 0.72, 1.85, 1.05, '残差相加\nx + Attn(x)'),
        (6.55, 0.72, 2.05, 1.05, 'LN2 → MLP\n768→3072→768'),
        (8.95, 0.72, 1.85, 1.05, '残差相加\n+ MLP(u)'),
        (11.15, 0.72, 2.55, 1.05, '输出仍是 [B,50,768]\n下一层继续'),
    ]
    for x, y, w, h, text in inner:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0.03,rounding_size=0.08',
            linewidth=1.1, facecolor='#FFFFFF', edgecolor='#7C8B9A'))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=9.2, color='#1F2937', linespacing=1.35)
    for start, end in ((4.02, 4.35), (6.22, 6.55), (8.62, 8.95), (10.82, 11.15)):
        ax.annotate('', xy=(end, 1.24), xytext=(start, 1.24),
                    arrowprops=dict(arrowstyle='-|>', color='#5B6B7C', lw=1.1,
                                    shrinkA=0, shrinkB=0))
    fig.tight_layout()
    fig.savefig(FIG / 'fig1_architecture.png', bbox_inches='tight')
    plt.close(fig)


def figure_patchify(image, image_id):
    """Show the 7x7 patch decomposition of one preprocessed image."""
    rgb = np.clip(image * STD + MEAN, 0.0, 1.0).transpose(1, 2, 0)
    fig = plt.figure(figsize=(14.2, 6.2), dpi=200)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.12)

    ax0 = fig.add_subplot(grid[0, 0])
    ax0.imshow(rgb)
    for line in range(1, 7):
        pos = line * 32 - 0.5
        ax0.axhline(pos, color='#2563EB', lw=0.9, alpha=0.9)
        ax0.axvline(pos, color='#2563EB', lw=0.9, alpha=0.9)
    for row in range(7):
        for col in range(7):
            ax0.text(col * 32 + 2, row * 32 + 10, str(row * 7 + col),
                     fontsize=6.2, color='#FDE047', fontweight='bold')
    ax0.set_title(f'输入 {image_id}：224×224 预处理图切分为 7×7 个 32×32 patch\n'
                  f'（编号为 patch 在 [B,49,3072] 中的行号）', fontsize=10.5)
    ax0.set_xticks([])
    ax0.set_yticks([])
    for spine in ax0.spines.values():
        spine.set_visible(False)

    ax1 = fig.add_subplot(grid[0, 1])
    gap = 3
    canvas = np.full((7 * 32 + 6 * gap, 7 * 32 + 6 * gap, 3), 0.92)
    for row in range(7):
        for col in range(7):
            patch = rgb[row * 32:(row + 1) * 32, col * 32:(col + 1) * 32]
            top = row * (32 + gap)
            left = col * (32 + gap)
            canvas[top:top + 32, left:left + 32] = patch
    ax1.imshow(canvas)
    ax1.set_title('49 个 patch（每块 32×32×3 = 3072 个像素）\n'
                  '按行优先展开后得到 [B,49,3072]', fontsize=10.5)
    ax1.set_xticks([])
    ax1.set_yticks([])
    for spine in ax1.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / 'fig2_patchify.png', bbox_inches='tight')
    plt.close(fig)


def figure_errors(rows):
    """Log-scale bar chart of the per-stage absolute error with tolerances."""
    stages = [r['stage'] for r in rows]
    max_abs = [max(r['max_abs'], 1e-12) for r in rows]
    mean_abs = [max(r['mean_abs'], 1e-12) for r in rows]
    x = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(14.2, 5.4), dpi=200)
    width = 0.36
    bars1 = ax.bar(x - width / 2, max_abs, width, label='最大绝对误差',
                   color='#2F6DB5', edgecolor='white')
    bars2 = ax.bar(x + width / 2, mean_abs, width, label='平均绝对误差',
                   color='#7FB3E8', edgecolor='white')
    ax.set_yscale('log')
    ax.set_ylim(1e-12, 1e-3)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=0, fontsize=10.5)
    ax.set_ylabel('|实际值 − 参考值|（对数轴）', fontsize=10.5)
    ax.grid(axis='y', which='major', alpha=0.3, linestyle='--')
    for xi, value in zip(x, max_abs):
        label = '0' if value <= 1e-12 else f'{value:.1e}'
        ax.text(xi - width / 2, value * 1.8, label, ha='center', va='bottom',
                fontsize=7.6, color='#1E3A5F')
    for xi, value in zip(x, mean_abs):
        label = '0' if value <= 1e-12 else f'{value:.1e}'
        ax.text(xi + width / 2, value * 1.8, label, ha='center', va='bottom',
                fontsize=7.6, color='#33506E')
    ax.axhline(2e-4, color='#C2410C', lw=1.1, linestyle='--')
    ax.text(len(stages) - 0.45, 1.1e-4, '内部 token 容差 atol=2e-4',
            ha='right', va='top', fontsize=8.6, color='#C2410C')
    ax.axhline(2e-5, color='#15803D', lw=1.1, linestyle=':')
    ax.text(len(stages) - 0.45, 1.1e-5, '输出 token 容差 atol=2e-5（rtol 均为 2e-4）',
            ha='right', va='top', fontsize=8.6, color='#15803D')
    ax.legend(loc='upper left', fontsize=9.5, framealpha=0.9)
    ax.set_title('与独立 PyTorch FP32 参考结果的分阶段数值误差（逐元素 abs，10 张图）',
                 fontsize=11.5)
    fig.tight_layout()
    fig.savefig(FIG / 'fig3_errors.png', bbox_inches='tight')
    plt.close(fig)


def figure_outputs(outputs, similarity, cls_attention, image, image_id):
    """Unit-norm bars, cosine-similarity matrix and CLS attention map."""
    fig = plt.figure(figsize=(15.2, 5.2), dpi=200)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.25, 1.0], wspace=0.30)

    ax0 = fig.add_subplot(grid[0, 0])
    norms = np.linalg.norm(outputs, axis=1)
    ids = [r['id'] for r in image_records()]
    ax0.bar(range(len(ids)), norms, color='#2F6DB5', edgecolor='white')
    ax0.axhline(1.0, color='#C2410C', lw=1.1, linestyle='--')
    ax0.set_ylim(0.9985, 1.0015)
    ax0.set_xticks(range(len(ids)))
    ax0.set_xticklabels(ids, rotation=45, ha='right', fontsize=8.4)
    ax0.set_ylabel('输出向量的 L2 范数', fontsize=10)
    ax0.set_title('10 个输出 token 的 L2 范数\n（全部为 1，验证单位向量契约）', fontsize=10.5)
    ax0.grid(axis='y', alpha=0.3, linestyle='--')

    ax1 = fig.add_subplot(grid[0, 1])
    im = ax1.imshow(similarity, cmap='YlGnBu', vmin=0.0, vmax=1.0)
    ax1.set_xticks(range(len(ids)))
    ax1.set_yticks(range(len(ids)))
    ax1.set_xticklabels(ids, rotation=45, ha='right', fontsize=8.0)
    ax1.set_yticklabels(ids, fontsize=8.0)
    off = similarity[~np.eye(len(ids), dtype=bool)]
    ax1.set_title(f'输出 token 两两余弦相似度\n（非对角 max={off.max():.3f}, '
                  f'min={off.min():.3f}）', fontsize=10.5)
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=8)
    for i in range(len(ids)):
        ax1.text(i, i, f'{similarity[i, i]:.2f}', ha='center', va='center',
                 fontsize=6.4, color='white')

    ax2 = fig.add_subplot(grid[0, 2])
    rgb = np.clip(image * STD + MEAN, 0.0, 1.0).transpose(1, 2, 0)
    ax2.imshow(rgb)
    upsampled = np.asarray(Image.fromarray(cls_attention.astype(np.float32))
                           .resize((224, 224), Image.Resampling.BICUBIC))
    ax2.imshow(upsampled, cmap='jet', alpha=0.45, vmin=float(upsampled.min()),
               vmax=float(upsampled.max()))
    for line in range(1, 7):
        pos = line * 32 - 0.5
        ax2.axhline(pos, color='white', lw=0.5, alpha=0.55)
        ax2.axvline(pos, color='white', lw=0.5, alpha=0.55)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title(f'{image_id}：第 12 层 CLS→49 patch\n注意力（12 头平均）',
                  fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / 'fig4_outputs.png', bbox_inches='tight')
    plt.close(fig)


def main():
    print('[1/6] loading weights and images ...', flush=True)
    weights = load_weights()
    encoder = ImageEncoder(weights)
    images = load_images()
    ids = [r['id'] for r in image_records()]
    with np.load(ROOT / 'assets/reference_outputs.npz', allow_pickle=False) as ref:
        reference = {key: ref[key] for key in ref.files}

    print('[2/6] running staged checks and timing ...', flush=True)
    start = time.perf_counter()
    patch_tokens = encoder.patchify(images[:2])
    patch_time = time.perf_counter() - start

    start = time.perf_counter()
    token_input = encoder.embed(images[:2])
    embed_time = time.perf_counter() - start

    start = time.perf_counter()
    block0 = encoder.blocks[0](encoder.pre_norm(encoder.embed(images[:2])))
    block0_time = time.perf_counter() - start

    start = time.perf_counter()
    batch2_tokens = encoder.encode_tokens(images[:2])
    batch2_time = time.perf_counter() - start

    start = time.perf_counter()
    batch10_tokens = encoder.encode_tokens(images)
    batch10_time = time.perf_counter() - start

    start = time.perf_counter()
    outputs = encoder(images)
    full_time = time.perf_counter() - start

    start = time.perf_counter()
    single = encoder(images[:1])
    single_time = time.perf_counter() - start

    start = time.perf_counter()
    reordered = encoder(images[[1, 0, 1]])
    reorder_time = time.perf_counter() - start

    rows = [
        dict(stage='preprocess', **compare(images, reference['preprocessed'])),
        dict(stage='patches', **compare(patch_tokens, reference['patches'][:2])),
        dict(stage='tokens', **compare(token_input, reference['input_tokens'][:2])),
        dict(stage='block0', **compare(block0, reference['block0_tokens'][:2])),
        dict(stage='last_tokens', **compare(batch2_tokens, reference['last_tokens'][:2])),
        dict(stage='output', **compare(outputs, reference['output_tokens'])),
        dict(stage='batch', **compare(reordered, reference['output_tokens'][[1, 0, 1]])),
    ]
    for row in rows:
        print(f"  {row['stage']:12s} {'PASS' if row['pass'] else 'FAIL'} "
              f"max={row['max_abs']:.3e} mean={row['mean_abs']:.3e}")

    with (ART / 'stage_errors.csv').open('w', encoding='utf-8') as stream:
        stream.write('stage,shape,max_abs,mean_abs,atol,pass\n')
        for row in rows:
            stream.write(f"{row['stage']},\"{row['shape']}\",{row['max_abs']:.6e},"
                         f"{row['mean_abs']:.6e},{row['atol']},"
                         f"{str(row['pass']).lower()}\n")

    norms = np.linalg.norm(outputs, axis=1)
    similarity = outputs @ outputs.T
    order_delta = float(np.abs(reordered - outputs[[1, 0, 1]]).max())
    duplicate_delta = float(np.abs(reordered[2] - reordered[0]).max())
    off = similarity[~np.eye(len(ids), dtype=bool)]

    print('[3/6] capturing CLS attention ...', flush=True)
    attention, replay_tokens = attention_maps(encoder, images[:1])
    replay_delta = float(np.abs(replay_tokens - encoder.encode_tokens(images[:1])).max())

    print('[4/6] saving arrays and metrics ...', flush=True)
    np.savez_compressed(ART / 'output_tokens.npz', ids=np.asarray(ids),
                        output_tokens=outputs, last_tokens=batch10_tokens)
    evidence = {
        'environment': {
            'python': sys.version.split()[0],
            'numpy': np.__version__,
        },
        'shapes': {
            'input': list(images.shape),
            'patches': list(patch_tokens.shape),
            'tokens': list(token_input.shape),
            'last_tokens': list(batch10_tokens.shape),
            'output': list(outputs.shape),
            'dtype': str(outputs.dtype),
        },
        'stage_errors': rows,
        'timing_seconds': {
            'patchify_B2': patch_time,
            'embed_B2': embed_time,
            'block0_B2': block0_time,
            'encode_tokens_B2': batch2_time,
            'encode_tokens_B10': batch10_time,
            'full_forward_B10': full_time,
            'full_forward_B1': single_time,
            'full_forward_reordered_B3': reorder_time,
            'per_image_forward_B10': full_time / 10,
        },
        'output_checks': {
            'norm_min': float(norms.min()),
            'norm_max': float(norms.max()),
            'norm_max_deviation': float(np.abs(norms - 1.0).max()),
            'cosine_offdiag_min': float(off.min()),
            'cosine_offdiag_max': float(off.max()),
            'cosine_offdiag_mean': float(off.mean()),
            'batch_reorder_delta': order_delta,
            'batch_duplicate_delta': duplicate_delta,
            'attention_replay_delta': replay_delta,
            'finite': bool(np.isfinite(outputs).all()),
        },
    }
    (ART / 'evidence.json').write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')

    print('[5/6] drawing figures ...', flush=True)
    figure_architecture()
    figure_patchify(images[0], ids[0])
    figure_errors(rows)
    figure_outputs(outputs, similarity, attention[11], images[0], ids[0])

    print('[6/6] done')
    print(f"  output dtype={outputs.dtype}, shape={outputs.shape}")
    print(f"  unit-norm max deviation={np.abs(norms - 1).max():.3e}")
    print(f"  cosine off-diagonal range=[{off.min():.4f}, {off.max():.4f}]")
    print(f"  full forward B=10: {full_time:.2f} s ({full_time / 10:.2f} s/image)")


if __name__ == '__main__':
    main()
