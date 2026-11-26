# 项目一：用 NumPy 实现 OpenCLIP ViT-B/32 图像编码器

> 课程：高级项目（2026 Advanced Projects）· 项目一（小项目）
> 姓名：王旭　学号：2025681094　班级：01 班　学院：人工智能学院

本项目只使用 **NumPy + Pillow**，完整复现 OpenCLIP ViT-B/32 图像塔
（image tower）的前向计算：输入一张图像，输出一个 L2 范数为 1 的
512 维特征向量。代码不依赖 PyTorch / OpenCLIP / GPU / 网络，可作为
最终项目 `szu_python_openclip` 的图像编码器直接复用。

## 1. 功能与输入输出

| 项目 | 说明 |
| --- | --- |
| 输入 | `[B,3,224,224]` float32，已按课程给定方式完成 Resize、中心裁剪与归一化 |
| 中间表示 | patch 序列 `[B,49,3072]` → 图像 token `[B,50,768]`（CLS + 49 patch） |
| 输出 | `[B,512]` float32，逐图像独立做单位化，L2 范数为 1 |
| 参数 | 课程提供的 152 个图像塔张量，`weights/image_encoder.npz`（不随仓库提交） |
| 训练 | 无。本作业只做推理，所有参数固定 |

## 2. 代码结构与实现对应关系

```text
project1-numpy-encoder/
├── image_encoder.py        # 唯一需要补全的实现：6 个 TODO（见下表）
├── helpers.py              # 课程提供：LayerNorm、精确 GELU、softmax、单位化
├── preprocess_image.py     # 课程提供：BICUBIC 缩放 + 中心裁剪 + CHW 归一化
├── runtime.py              # 课程提供：权重/图像/参考数据加载
├── run.py                  # 课程提供：命令行入口，保存 [B,512] 输出
├── check.py                # 课程提供：分阶段数值校验（含 batch 顺序/重复检查）
├── transformer_steps.py    # 课程提供：3 token 的教学版 Transformer 块
├── config.json             # 模型结构常量（224/32/768/12 头/12 层/512 维）
├── requirements.txt        # numpy、Pillow、imageio
├── weights/
│   └── manifest.json       # 152 个张量的名称、形状、dtype、SHA-256
├── assets/
│   ├── images/             # 10 张课程提供的测试图像
│   ├── images.json         # 图像 ID 与 SHA-256
│   └── manifest.json       # 参考输出文件的 SHA-256
├── scripts/
│   ├── make_evidence.py    # 生成报告中的误差、耗时、范数、相似度与全部插图
│   └── check_patch_order.py# 复现 patch 顺序写错时的数值差异（报告 4.2 节）
├── figures/                # 报告插图（由 scripts/make_evidence.py 生成）
├── artifacts/              # check 输出、分阶段误差 CSV、evidence JSON
└── report/                 # 实验报告 docx / pdf
```

六个 TODO 全部位于 [image_encoder.py](image_encoder.py)，实现要点：

| 编号 | 方法 | 实现方法 | 关键形状 / 约定 |
| --- | --- | --- | --- |
| TODO-01 | `ImageEncoder.patchify` | `reshape` 成 6 维 `[B,C,gH,pH,gW,pW]`，`transpose(0,2,4,1,3,5)` 后展平 | `[B,3,224,224] → [B,49,3072]`，patch 行优先、块内 C,H,W 展平 |
| TODO-02 | `MultiHeadAttention.split_heads` | 特征轴切分为 H 个 head_dim 连续切片，再把 head 轴移到 token 轴之前 | `[B,N,768] → [B,12,N,64]` |
| TODO-03 | `MultiHeadAttention.__call__` | `scores = q @ kᵀ / √64`；`context = coeff @ v`；head 合并；输出投影 | `[B,H,N,N]`、`[B,N,768]`；线性层权重为 `[out,in]`，用 `W.T` |
| TODO-04 | `TransformerBlock.__call__` | 先 `x + Attn(LN1(x))`，再 `residual + MLP(LN2(residual))`，两条残差 | `[B,50,768]` 全程不变 |
| TODO-05 | `ImageEncoder.embed` | patch 投影（无偏置）→ 拼接 CLS → 加位置编码 | `[B,49,3072] @ Wᵀ → [B,49,768] + CLS → [B,50,768]` |
| TODO-06 | `ImageEncoder.__call__` | 取 CLS 行 → post-Norm → `visual.proj`（已是 `[in,out]`）→ 单位化 | `[B,50,768] → [B,768] → [B,512]` |

## 3. 运行与复现

环境：Python 3.9+（本次实测 Python 3.12.14）、NumPy 2.3.5、Pillow 12.3.0；
仅需 CPU，无随机性。

```bash
# 1) 安装依赖
python -m pip install -r requirements.txt

# 2) 放入课程附件中的两个大文件（见第 4 节）：
#    weights/image_encoder.npz            约 311 MiB
#    assets/reference_outputs.npz         约 7.3 MiB

# 3) 分阶段数值校验（按顺序执行，先定位第一个不一致的阶段）
python check.py --stage assets
python check.py --stage preprocess
python check.py --stage patches
python check.py --stage tokens
python check.py --stage block0
python check.py --stage last_tokens
python check.py --stage output
python check.py --stage batch
python check.py --stage all          # 8 项检查（assets 校验 + 7 个数值阶段）

# 4) 对 10 张课程图像生成输出 token，保存为 npz
python run.py --output artifacts/output_tokens.npz

# 5) 复现报告中的全部数值与插图（约 40 秒）
python scripts/make_evidence.py

# 6) 复现 patch 顺序写错的数值差异（报告 4.2 节）
python scripts/check_patch_order.py
```

`run.py` 也支持任意图像：

```bash
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz
```

## 4. 数据与参数的获取（重要）

按作业要求，仓库**不重复提交**大参数文件与参考数据，仅保留课程原始清单
（`weights/manifest.json`、`assets/images.json`、`assets/manifest.json`）：

| 文件 | 大小 | 获取方法 | 完整性校验 |
| --- | --- | --- | --- |
| `weights/image_encoder.npz` | 326,388,585 B | 课程附件 `project1-学生版.zip` → `weights/` | `check.py --stage assets` 内部校验 SHA-256 |
| `assets/reference_outputs.npz` | 7,655,354 B | 课程附件 `project1-学生版.zip` → `assets/` | 同上 |

`check.py` 会自动校验权重文件 SHA-256（`8567f805…37e7a40`）、152 个张量的
名称/形状/dtype/逐张量 SHA-256，以及 10 张图像与参考输出的 SHA-256，
因此用到的数据可被独立复核。

## 5. 验证结果（全部 PASS）

`python check.py --stage all` 的实际输出（10 张图，容差
`abs(actual-ref) ≤ atol + rtol*abs(ref)`，rtol=2e-4）：

| 阶段 | 形状 | max_abs | mean_abs | atol | 结果 |
| --- | --- | ---: | ---: | ---: | --- |
| preprocess | `[10,3,224,224]` | 0 | 0 | 2e-6 | PASS |
| patches | `[2,49,3072]` | 0 | 0 | 2e-4 | PASS |
| tokens | `[2,50,768]` | 4.01e-5 | 1.29e-6 | 2e-4 | PASS |
| block0 | `[2,50,768]` | 7.06e-6 | 3.25e-7 | 2e-4 | PASS |
| last_tokens | `[2,50,768]` | 4.77e-5 | 2.70e-6 | 2e-4 | PASS |
| output | `[10,512]` | 5.44e-7 | 8.31e-8 | 2e-5 | PASS |
| batch | `[3,512]` | 4.62e-7 | 7.63e-8 | 2e-5 | PASS |

其他契约检查：

- 输出 dtype 为 float32，全部有限；L2 范数与 1 的最大偏差 `1.19e-7`；
- batch 输入按 `[1,0,1]` 重排：与参考重排结果的 max 差 `5.8e-8`；
- 同一图像重复两次的输出差为 `0`（前向完全确定，无隐式状态）；
- 10 张图输出的两两余弦相似度范围 `0.462–0.674`，说明特征能区分不同图像；
- 纯 NumPy 前向耗时：B=10 为 5.59 s（0.56 s/图），B=1 为 0.48 s。

完整证据见 `artifacts/check_all.txt`、`artifacts/stage_errors.csv`、
`artifacts/evidence.json`；实验报告见 `report/`。

## 6. 实验报告

`report/项目一实验报告-王旭-2025681094.pdf`（正文 5 页以内），内容包括任务与方法、
可复现性、结果与证据、分析与反思、贡献与工具说明。

## 7. 说明与致谢

- 以下内容由课程提供、本仓库原样保留：`helpers.py`、`preprocess_image.py`、
  `runtime.py`、`run.py`、`check.py`、`transformer_steps.py`、`config.json`、
  `weights/manifest.json`、`assets/` 中的图像与清单、`lecture-demos/`、`practice/`。
- `image_encoder.py` 中 6 个 TODO 的实现、`scripts/`、`figures/`、`artifacts/`
  与实验报告由本人完成。
- 方法参考：OpenCLIP（ViT-B/32 image tower）、ViT 论文与课程 PPT 第 39–76 页。
