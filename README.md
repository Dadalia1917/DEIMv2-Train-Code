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
1. [环境准备](#1-环境准备)
2. [数据集准备](#2-数据集准备)
3. [配置文件说明](#3-配置文件说明)
4. [开始训练](#4-开始训练)
5. [模型验证与推理](#5-模型验证与推理)
6. [训练参数说明](#6-训练参数说明)
7. [常见问题](#7-常见问题)

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

对于 DINOv3-S 模型，需要下载 ViT-Tiny 的蒸馏权重：

1. 下载 [ViT-Tiny 权重](https://drive.google.com/file/d/1YMTq_woOLjAcZnHSYNTsNg7f0ahj5LPs/view?usp=sharing)
2. 将权重文件放在项目根目录的 `ckpts` 文件夹中

```
ckpts/
└── vitt_distill.pt
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
└── valid/
    ├── _annotations.coco.json
    └── [图像文件]
```

### 2.2 从 YOLO 格式转换

如果你的数据集是 YOLO 格式，可以使用项目提供的转换脚本：

```bash
python yolo_to_coco.py
```

---

## 3. 配置文件说明

打开 `configs/dataset/custom_detection.yml`，根据你的数据集修改：

```yaml
num_classes: 4  # 修改为你的类别数量

train_dataloader:
  dataset:
    img_folder: ./dataset/train
    ann_file: ./dataset/train/_annotations.coco.json

val_dataloader:
  dataset:
    img_folder: ./dataset/valid
    ann_file: ./dataset/valid/_annotations.coco.json
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
