# DEIMv2: Real-Time Object Detection Meets DINOv3

<p align="center">
    <a href="https://github.com/Intellindust-AI-Lab/DEIMv2/blob/master/LICENSE">
        <img alt="license" src="https://img.shields.io/badge/LICENSE-Apache%202.0-blue">
    </a>
    <a href="https://arxiv.org/abs/2509.20787">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2509.20787-red">
    </a>
   <a href="https://intellindust-ai-lab.github.io/projects/DEIMv2/">
        <img alt="project webpage" src="https://img.shields.io/badge/Webpage-DEIMv2-purple">
    </a>
</p>

<p align="center">
DEIMv2 是 DEIM 框架的进化版本，利用了 DINOv3 的丰富特征。我们的方法设计了多种模型规模，从超轻量级版本到 S、M、L 和 X，以适应各种场景。在这些变体中，DEIMv2 实现了最先进的性能，其中 S 型模型在具有挑战性的 COCO 基准上显著超过了 50 AP。
</p>

---

<div align="center">
  <a href="http://www.shihuahuang.cn">Shihua Huang</a><sup>1*</sup>,&nbsp;&nbsp;
  Yongjie Hou<sup>1,2*</sup>,&nbsp;&nbsp;
  Longfei Liu<sup>1*</sup>,&nbsp;&nbsp;
  <a href="https://xuanlong-yu.github.io/">Xuanlong Yu</a><sup>1</sup>,&nbsp;&nbsp;
  <a href="https://xishen0220.github.io">Xi Shen</a><sup>1†</sup>&nbsp;&nbsp;
</div>

  
<p align="center">
<i>
1. <a href="https://intellindust-ai-lab.github.io"> Intellindust AI Lab</a> &nbsp;&nbsp; 2. Xiamen University &nbsp; <br> 
* Equal Contribution &nbsp;&nbsp; † Corresponding Author
</i>
</p>

---

## 📖 关于本项目

本项目基于 [DEIMv2 官方仓库](https://github.com/Intellindust-AI-Lab/DEIMv2) 进行改进和优化：

### 主要改进

1. **完善的中文教程**
   - 详细的自定义数据集训练指南
   - 从 YOLO 格式到 COCO 格式的转换教程
   - 完整的训练和推理流程说明

2. **代码兼容性修复**
   - 适配 `torch 2.8.0` 和 `torchvision 0.23.0`
   - 修复 `torchvision.transforms.v2` API 变更（`_transform` → `transform`）
   - 修复 UTF-8 编码问题，支持中文路径

3. **工具优化**
   - 提供简洁的 `predict.py` 推理脚本
   - 修复 YOLO 转 COCO 的类别索引问题（0-based）
   - 添加自定义类别名称显示功能

4. **使用体验改进**
   - 清晰的命令行参数说明
   - 详细的错误处理和提示
   - 适配低显存环境的训练配置

---

## 📖 中文教程

本教程将指导你如何使用自己的数据集训练 DEIMv2 目标检测模型。

### 📋 目录
1. [快速开始](#快速开始)
2. [环境准备](#1-环境准备)
3. [数据集准备](#2-数据集准备)
4. [配置文件说明](#3-配置文件说明)
5. [开始训练](#4-开始训练)
6. [模型验证与推理](#5-模型验证与推理)
7. [训练参数说明](#6-训练参数说明)
8. [常见问题](#7-常见问题)
9. [技术修改说明](#-技术修改说明)

---

## 🚀 快速开始

如果你想快速开始训练，只需 3 步：

### 第 1 步：安装环境

```bash
conda create -n deimv2 python=3.11 -y
conda activate deimv2
pip install -r requirements.txt
```

### 第 2 步：下载 Backbone 权重

下载 **S 模型**的 backbone 权重（~80 MB）：
- 链接：https://drive.google.com/file/d/1YMTq_woOLjAcZnHSYNTsNg7f0ahj5LPs/view?usp=sharing
- 保存为：`ckpts/vitt_distill.pt`

### 第 3 步：开始训练

```bash
python train.py \
  -c configs/deimv2/deimv2_dinov3_s_coco.yml \
  --use-amp --seed=0 \
  -t deimv2_dinov3_s_coco.pth \
  --update \
    train_dataloader.total_batch_size=4 \
    epoches=62 \
    flat_epoch=29 \
    no_aug_epoch=12 \
    num_classes=4
```

**就这么简单！** 🎉 训练会自动开始，checkpoint 保存在 `outputs/deimv2_dinov3_s_coco/`

---

## 1. 环境准备

### 1.1 创建虚拟环境
```bash
conda create -n deimv2 python=3.11 -y
conda activate deimv2
```

### 1.2 安装依赖
```bash
pip install -r requirements.txt
```

### 1.3 下载预训练权重

#### ⭐ Backbone 权重（训练必需）

根据使用的模型，需要下载相应的 backbone 预训练权重：

#### 📥 各模型对应的 Backbone 权重

| DEIMv2 模型 | 使用的 Backbone | 权重文件名 | 下载链接 |
|------------|----------------|-----------|---------|
| **S** | ViT-Tiny (蒸馏版) | `vitt_distill.pt` | [Google Drive](https://drive.google.com/file/d/1YMTq_woOLjAcZnHSYNTsNg7f0ahj5LPs/view?usp=sharing) |
| **M** | ViT-Tiny+ (蒸馏版) | `vittplus_distill.pt` | [Google Drive](https://drive.google.com/file/d/1COHfjzq5KfnEaXTluVGEOMdhpuVcG6Jt/view?usp=sharing) |
| **L** | DINOv3 ViT-S/16 | `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` | [百度网盘](https://pan.baidu.com/s/16DbtIXNnXn9swg6mIyG-eA?pwd=k6uy) (k6uy) / [Hugging Face](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) |
| **X** | DINOv3 ViT-S+/16 | `dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth` | [百度网盘](https://pan.baidu.com/s/1MN1NTQh5FB-zlMNDeiw5lg?pwd=1p6j) (1p6j) / [Hugging Face](https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m) |

#### 📝 详细下载步骤

**1. ViT-Tiny / ViT-Tiny+ (S/M 模型) - Google Drive 下载**

```bash
# 步骤：
# 1. 在浏览器打开 Google Drive 链接
# 2. 点击右上角的"下载"按钮（↓图标）
# 3. 下载完成后移动到 ckpts/ 目录
# 
# S 模型: https://drive.google.com/file/d/1YMTq_woOLjAcZnHSYNTsNg7f0ahj5LPs/view?usp=sharing
# M 模型: https://drive.google.com/file/d/1COHfjzq5KfnEaXTluVGEOMdhpuVcG6Jt/view?usp=sharing
```

**2. DINOv3 ViT-S/S+ (L/X 模型) - 多种下载方案**

⚠️ **重要**：Meta 官方链接在中国地区受限，请使用以下替代方案：

**方案 A：百度网盘（推荐，国内高速）**

```bash
# L 模型 (dinov3_vits16_pretrain_lvd1689m-08c60483.pth)
链接: https://pan.baidu.com/s/16DbtIXNnXn9swg6mIyG-eA?pwd=k6uy
提取码: k6uy

# X 模型 (dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth)
链接: https://pan.baidu.com/s/1MN1NTQh5FB-zlMNDeiw5lg?pwd=1p6j
提取码: 1p6j

# 下载后直接放入 ckpts/ 目录即可，无需重命名
```

**方案 B：Hugging Face（备选方案）**

```bash
# 步骤：
# 1. 在浏览器打开 Hugging Face 链接
# 2. 点击 "Files and versions" 标签
# 3. 下载 pytorch_model.bin 或 model.safetensors
# 4. 重命名为配置文件中的文件名
# 5. 移动到 ckpts/ 目录
#
# L 模型: https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
# X 模型: https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m

# 如果 Hugging Face 访问慢，可以使用镜像站：
# https://hf-mirror.com/facebook/dinov3-vits16-pretrain-lvd1689m
# https://hf-mirror.com/facebook/dinov3-vits16plus-pretrain-lvd1689m
```

**3. 最终目录结构**

下载完成后，`ckpts/` 目录应该包含（根据你使用的模型）：

```bash
ckpts/
├── vitt_distill.pt                                    # S 模型（~80 MB）
├── vittplus_distill.pt                                # M 模型（~110 MB）
├── dinov3_vits16_pretrain_lvd1689m-08c60483.pth      # L 模型（~80 MB）
└── dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth  # X 模型（~110 MB）
```

**4. 验证下载**

```bash
# 检查文件是否存在
ls ckpts/

# 或在 PowerShell 中
dir ckpts\
```

**⚠️ 重要提示：**
- ✅ **最小配置**：如果只训练 S 模型，只需下载 `vitt_distill.pt` (~80 MB)
- ⚠️ **无 Backbone**：如果不下载权重，训练将从随机初始化开始，需要 200+ epochs，精度降低 5-10%
- 📦 **全部下载**：如果想尝试所有模型，总大小约 380 MB
- 🚀 **国内用户**：L/X 模型推荐使用百度网盘下载，速度快且无需重命名

#### 完整模型权重（可选）

如果需要使用预训练的完整模型进行推理或微调，请从官方仓库下载：

| 模型 | AP | 下载链接 |
|------|-----|---------|
| **Atto** | 23.8 | [Google Drive](https://drive.google.com/file/d/18sRJXX3FBUigmGJ1y5Oo_DPC5C3JCgYc/view?usp=sharing) / [Quark](https://pan.quark.cn/s/04c997582fca) |
| **Femto** | 31.0 | [Google Drive](https://drive.google.com/file/d/16hh6l9Oln9TJng4V0_HNf_Z7uYb7feds/view?usp=sharing) / [Quark](https://pan.quark.cn/s/169f3cefec1b) |
| **Pico** | 38.5 | [Google Drive](https://drive.google.com/file/d/1PXpUxYSnQO-zJHtzrCPqQZ3KKatZwzFT/view?usp=sharing) / [Quark](https://pan.quark.cn/s/0db5b1dff721) |
| **N** | 43.0 | [Google Drive](https://drive.google.com/file/d/1G_Q80EVO4T7LZVPfHwZ3sT65FX5egp9K/view?usp=sharing) / [Quark](https://pan.quark.cn/s/1f626f191d11) |
| **S** | 50.9 | [Google Drive](https://drive.google.com/file/d/1MDOh8UXD39DNSew6rDzGFp1tAVpSGJdL/view?usp=sharing) / [Quark](https://pan.quark.cn/s/f4d05c349a24) |
| **M** | 53.0 | [Google Drive](https://drive.google.com/file/d/1nPKDHrotusQ748O1cQXJfi5wdShq6bKp/view?usp=sharing) / [Quark](https://pan.quark.cn/s/68a719248756) |
| **L** | 56.0 | [Google Drive](https://drive.google.com/file/d/1dRJfVHr9HtpdvaHlnQP460yPVHynMray/view?usp=sharing) / [Quark](https://pan.quark.cn/s/966b7ef89bdf) |
| **X** | 57.8 | [Google Drive](https://drive.google.com/file/d/1pTiQaBGt8hwtO0mbYlJ8nE-HGztGafS7/view?usp=sharing) / [Quark](https://pan.quark.cn/s/038aa966b283) |

将下载的模型放在项目根目录：

```
DEIMv2/
├── deimv2_dinov3_s_coco.pth
├── deimv2_dinov3_m_coco.pth
└── ...
```

---

## 2. 数据集准备

### 2.1 数据集格式

本项目使用 **COCO 格式**的数据集。数据集应包含以下结构：

```
dataset/
├── train/
│   ├── _annotations.coco.json
│   └── [图像文件]
├── valid/
│   ├── _annotations.coco.json
│   └── [图像文件]
└── test/
    ├── _annotations.coco.json
    └── [图像文件]
```

### 2.2 从 YOLO 格式转换

如果你的数据集是 YOLO 格式，可以使用项目提供的转换脚本：

```bash
python yolo_to_coco.py
```

转换后会自动生成包含 train、valid、test 三个目录的 COCO 格式数据集。

---

## 3. 配置文件说明

### 3.1 修改数据集配置

打开 `configs/dataset/custom_detection.yml`，根据你的数据集修改：

```yaml
num_classes: 4  # 修改为你的类别数量
remap_mscoco_category: False  # 自定义数据集设为 False

train_dataloader:
  dataset:
    img_folder: ./dataset/train
    ann_file: ./dataset/train/_annotations.coco.json

val_dataloader:
  dataset:
    img_folder: ./dataset/valid
    ann_file: ./dataset/valid/_annotations.coco.json
```

### 3.2 修改模型配置

打开 `configs/deimv2/deimv2_dinov3_s_coco.yml`，确认使用自定义数据集配置：

```yaml
__include__: [
  '../dataset/custom_detection.yml',  # 使用自定义数据集
  '../runtime.yml',
  '../base/dataloader.yml',
  '../base/optimizer.yml',
  '../base/deimv2.yml',
]
```

---

## 4. 开始训练

### 单GPU训练（batch=4, epoch=50）

```bash
python train.py -c configs/deimv2/deimv2_dinov3_s_coco.yml --use-amp --seed=0 -t deimv2_dinov3_s_coco.pth --update train_dataloader.total_batch_size=4 epoches=62 flat_epoch=29 no_aug_epoch=12 num_classes=4
```

### 多GPU训练（例如4块GPU）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/deimv2/deimv2_dinov3_s_coco.yml --use-amp --seed=0 -t deimv2_dinov3_s_coco.pth --update train_dataloader.total_batch_size=16 epoches=62 flat_epoch=29 no_aug_epoch=12 num_classes=4
```

---

## 5. 模型验证与推理

### 使用简易推理脚本

项目提供了 `predict.py` 脚本，可以快速对单张图片进行检测：

```bash
# 使用训练好的模型
python predict.py -c configs/deimv2/deimv2_dinov3_s_coco.yml -r outputs/deimv2_dinov3_s_coco/checkpoint0049.pth -i test1.jpg -o result.jpg --classes "类别1,类别2,类别3,类别4"

# 调整置信度阈值
python predict.py -c configs/deimv2/deimv2_dinov3_s_coco.yml -r outputs/checkpoint0049.pth -i test1.jpg -o result.jpg --conf 0.3
```

### 评估模型性能

```bash
python train.py -c configs/deimv2/deimv2_dinov3_s_coco.yml --test-only -r outputs/deimv2_dinov3_s_coco/checkpoint0049.pth --update num_classes=4
```

---

## 6. 训练参数说明

### 重要超参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `epoches` | 总训练轮数（包括EMA） | 实际训练轮数 + 12 |
| `total_batch_size` | 批次大小 | 根据显存调整（4/8/16/32） |
| `num_workers` | 数据加载线程数 | 4-8 |
| `flat_epoch` | 学习率平台期 | 4 + 实际训练轮数 // 2 |

### 显存优化建议

| 显存大小 | 推荐batch_size | 推荐模型 |
|---------|---------------|----------|
| 4GB | 2 | Atto/Femto/Pico |
| 6GB | 4 | N |
| 8GB | 4-8 | S |
| 12GB+ | 8-16 | S/M |

---

## 7. 常见问题

### 7.1 显存不足

1. 减小 `total_batch_size`
2. 使用 `--use-amp`（混合精度）
3. 选择更小的模型

### 7.2 类别ID说明

- 本项目使用 **0-based 索引**
- 4个类别的 `category_id` 应该是：**0, 1, 2, 3**
- 使用 `yolo_to_coco.py` 会自动处理为 0-based 索引

### 7.3 Backbone 权重找不到

**错误示例**：
```
FileNotFoundError: [Errno 2] No such file or directory: './ckpts/vitt_distill.pt'
```

**解决方案**：
1. 确认 `ckpts/` 目录存在：`mkdir ckpts`
2. 确认权重文件已下载并正确命名
3. 检查文件名是否与配置文件中的 `weights_path` 一致

### 7.4 下载慢或无法访问

**解决方案（按推荐顺序）**：
1. **百度网盘**（最推荐）：L/X 模型已提供百度网盘下载链接，国内高速
2. **Hugging Face 镜像站**：将 URL 中的 `huggingface.co` 替换为 `hf-mirror.com`
3. **Google Drive**：S/M 模型使用 Google Drive

### 7.5 DINOv3 权重文件重命名

**百度网盘**：下载的文件名已经正确，**无需重命名**，直接放入 `ckpts/` 目录即可。

**Hugging Face**：下载的文件通常名为 `pytorch_model.bin` 或 `model.safetensors`，需要重命名：

```bash
# L 模型
mv pytorch_model.bin dinov3_vits16_pretrain_lvd1689m-08c60483.pth

# X 模型
mv pytorch_model.bin dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
```

### 7.6 训练时 CUDA 错误：device-side assert triggered

**原因**：类别索引错误，使用了 1-based 索引而非 0-based

**解决方案**：
- 使用本项目提供的 `yolo_to_coco.py` 转换数据集
- 或手动检查 `_annotations.coco.json` 中的 `category_id` 是否从 0 开始

---

## 🔧 技术修改说明

### 修改的文件列表

1. **`engine/data/transforms/_transforms.py`**
   - **修改原因**：适配 `torchvision 0.23.0` API 变更
   - **具体修改**：将 `_transform()` 方法改为 `transform()` 方法
   - **影响类**：`ConvertPILImage`, `ConvertBoxes`

2. **`engine/core/yaml_utils.py`**
   - **修改原因**：支持中文路径和中文配置文件
   - **具体修改**：为所有 `open()` 调用添加 `encoding='utf-8'`

3. **`yolo_to_coco.py`**
   - **修改原因**：修复类别索引错误（导致训练时 CUDA 错误）
   - **具体修改**：使用 0-based 索引代替 1-based 索引
   - **关键变更**：`category_id: class_id` (不是 `class_id + 1`)

4. **新增 `predict.py`**
   - **功能**：简化的推理脚本，支持自定义输出路径和类别名称
   - **特性**：清晰的输出信息、错误处理、灵活的参数配置

### 环境兼容性

| 组件 | 官方版本 | 本项目测试版本 | 状态 |
|------|---------|---------------|------|
| Python | 3.11 | 3.11 | ✅ |
| PyTorch | 2.5.1 | 2.8.0+cu128 | ✅ 已适配 |
| torchvision | 0.20.1 | 0.23.0+cu128 | ✅ 已适配 |

---

## 📖 完整使用流程示例

以下是从零开始训练和推理的完整流程：

```bash
# ========== 1. 环境准备 ==========
conda create -n deimv2 python=3.11 -y
conda activate deimv2
pip install -r requirements.txt

# ========== 2. 准备 Backbone 权重 ==========
mkdir ckpts
# 下载 vitt_distill.pt 到 ckpts/ 目录（S 模型）
# Google Drive: https://drive.google.com/file/d/1YMTq_woOLjAcZnHSYNTsNg7f0ahj5LPs/view?usp=sharing
#
# 如果使用 L 模型，可以从百度网盘下载（国内高速）：
# 百度网盘: https://pan.baidu.com/s/16DbtIXNnXn9swg6mIyG-eA?pwd=k6uy (提取码: k6uy)

# ========== 3. 准备数据集 ==========
# 方案 A：如果已有 COCO 格式数据集
# 确保目录结构：
# dataset/
# ├── train/_annotations.coco.json
# ├── valid/_annotations.coco.json
# └── test/_annotations.coco.json

# 方案 B：如果是 YOLO 格式，先转换
python yolo_to_coco.py

# ========== 4. 修改配置文件 ==========
# 编辑 configs/dataset/custom_detection.yml
# 设置 num_classes: 4（改为你的类别数）

# ========== 5. 开始训练 ==========
python train.py \
  -c configs/deimv2/deimv2_dinov3_s_coco.yml \
  --use-amp --seed=0 \
  -t deimv2_dinov3_s_coco.pth \
  --update \
    train_dataloader.total_batch_size=4 \
    train_dataloader.num_workers=4 \
    val_dataloader.num_workers=4 \
    epoches=62 \
    flat_epoch=29 \
    no_aug_epoch=12 \
    num_classes=4

# ========== 6. 推理检测 ==========
# 使用训练好的模型
python predict.py \
  -c configs/deimv2/deimv2_dinov3_s_coco.yml \
  -r outputs/deimv2_dinov3_s_coco/checkpoint0049.pth \
  -i test1.jpg \
  -o result.jpg \
  --classes "飞机,船,车辆,建筑" \
  --conf 0.45

# ========== 7. 评估模型 ==========
python train.py \
  -c configs/deimv2/deimv2_dinov3_s_coco.yml \
  --test-only \
  -r outputs/deimv2_dinov3_s_coco/checkpoint0049.pth \
  --update num_classes=4
```

---

## 📚 完整文档

更详细的教程请参考原始 [README](README(old).md)

---

## 📜 Citation

If you use DEIMv2 in your work, please cite:

```bibtex
@article{huang2025deimv2,
  title={Real-Time Object Detection Meets DINOv3},
  author={Huang, Shihua and Hou, Yongjie and Liu, Longfei and Yu, Xuanlong and Shen, Xi},
  journal={arXiv},
  year={2025}
}
```

## 🙏 Acknowledgement

本项目基于 [DEIMv2 官方仓库](https://github.com/Intellindust-AI-Lab/DEIMv2) 进行改进和优化，感谢原作者团队的杰出工作！

DEIMv2 的核心代码构建于以下优秀开源项目：
- [D-FINE](https://github.com/Peterande/D-FINE)
- [RT-DETR](https://github.com/lyuwenyu/RT-DETR)
- [DEIM](https://github.com/ShihuaHuang95/DEIM)
- [DINOv3](https://github.com/facebookresearch/dinov3)

感谢所有开源贡献者的付出！

---

**如果你觉得这个项目有帮助，请给我们一个 ⭐！**
