# DEIMv2 自定义数据集训练教程

本教程将指导你如何使用自己的数据集训练 DEIMv2 目标检测模型。

## 📋 目录
1. [环境准备](#1-环境准备)
2. [数据集准备](#2-数据集准备)
3. [配置文件说明](#3-配置文件说明)
4. [开始训练](#4-开始训练)
5. [模型测试](#5-模型测试)
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
2. 将权重文件放在项目根目录的 `ckpts` 文件夹中：

```
ckpts/
└── vitt_distill.pt
```

如果 `ckpts` 文件夹不存在，请手动创建：
```bash
mkdir ckpts
```

---

## 2. 数据集准备

### 2.1 数据集格式

本项目使用 **COCO 格式**的数据集。数据集应包含以下结构：

```
dataset/
├── train/
│   ├── _annotations.coco.json  # 训练集标注文件
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── valid/
│   ├── _annotations.coco.json  # 验证集标注文件
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── test/
    ├── _annotations.coco.json  # 测试集标注文件
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### 2.2 从 YOLO 格式转换

如果你的数据集是 YOLO 格式，可以使用项目提供的转换脚本：

```bash
python yolo_to_coco.py
```

**YOLO 数据集结构（转换前）：**
```
datasets/Data/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

**转换后会自动生成：**
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

### 2.3 COCO 标注文件格式说明

`_annotations.coco.json` 文件包含以下字段：

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],  // 左上角坐标和宽高
      "area": 12800,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "类别1",
      "supercategory": "object"
    }
  ]
}
```

---

## 3. 配置文件说明

### 3.1 修改数据集配置

打开 `configs/dataset/custom_detection.yml`，根据你的数据集修改以下内容：

```yaml
num_classes: 4  # 修改为你的类别数量（示例：4类）

train_dataloader:
  dataset:
    img_folder: ./dataset/train  # 训练集图像文件夹路径
    ann_file: ./dataset/train/_annotations.coco.json  # 训练集标注文件路径

val_dataloader:
  dataset:
    img_folder: ./dataset/valid  # 验证集图像文件夹路径
    ann_file: ./dataset/valid/_annotations.coco.json  # 验证集标注文件路径
```

**注意：** 如果你的数据集路径不同，请相应修改 `img_folder` 和 `ann_file` 的路径。

### 3.2 修改模型配置

打开 `configs/deimv2/deimv2_dinov3_s_coco.yml`（或你选择的其他模型配置文件）：

**重要：修改数据集引用**
```yaml
__include__: [
  '../dataset/custom_detection.yml',  # 将 coco_detection.yml 改为 custom_detection.yml
  '../runtime.yml',
  '../base/dataloader.yml',
  '../base/optimizer.yml',
  '../base/deimv2.yml',
]
```

**可选：修改输出目录**
```yaml
output_dir: ./outputs/deimv2_dinov3_s_coco  # 可以改为你喜欢的名字
```

---

## 4. 开始训练

### 4.1 修改模型配置文件

打开 `configs/deimv2/deimv2_dinov3_s_coco.yml`，修改数据集配置：

```yaml
__include__: [
  '../dataset/custom_detection.yml',  # 将 coco_detection.yml 改为 custom_detection.yml
  '../runtime.yml',
  '../base/dataloader.yml',
  '../base/optimizer.yml',
  '../base/deimv2.yml',
]
```

### 4.2 从预训练权重开始训练（推荐）

**单GPU训练（batch=4, epoch=50）：**
```bash
python train.py -c configs/deimv2/deimv2_dinov3_s_coco.yml --use-amp --seed=0 -t deimv2_dinov3_s_coco.pth --update train_dataloader.total_batch_size=4 epoches=62 flat_epoch=29 no_aug_epoch=12
```

**多GPU训练（例如使用2块GPU）：**
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=7777 --nproc_per_node=2 train.py -c configs/deimv2/deimv2_dinov3_s_coco.yml --use-amp --seed=0 -t deimv2_dinov3_s_coco.pth --update train_dataloader.total_batch_size=4 epoches=62 flat_epoch=29 no_aug_epoch=12
```

**参数说明：**
- `-c`: 配置文件路径
- `--use-amp`: 使用混合精度训练（节省显存，加速训练）
- `--seed=0`: 随机种子，保证结果可复现
- `-t`: 预训练权重路径（tuning模式）
- `--update`: 覆盖配置文件中的参数（格式：key=value）

**训练轮数计算说明：**
- 实际训练50轮，需要设置 `epoches=62`（50 + 12，12为EMA轮数）
- `flat_epoch=29`（4 + 50 // 2）
- `no_aug_epoch=12`（固定值）

### 4.3 从头开始训练（不推荐）

如果想从头训练（不推荐，训练时间长且效果可能不如微调）：

```bash
python train.py -c configs/deimv2/deimv2_dinov3_s_coco.yml --use-amp --seed=0 --update train_dataloader.total_batch_size=4 epoches=62
```

### 4.4 恢复训练

如果训练中断，可以从最后一个检查点恢复：

```bash
python train.py -c configs/deimv2/deimv2_dinov3_s_coco.yml --use-amp --seed=0 --resume outputs/deimv2_dinov3_s_coco/checkpoint0049.pth
```

---

## 5. 模型验证与推理

### 5.1 使用简易推理脚本（推荐）

项目提供了 `predict.py` 脚本，可以快速对单张图片进行检测：

#### 5.1.1 使用预训练模型推理

```bash
# 基础用法
python predict.py -c configs/deimv2/deimv2_dinov3_s_coco.yml -r deimv2_dinov3_s_coco.pth -i test1.jpg -o result.jpg

# 指定类别名称（用逗号分隔）
python predict.py -c configs/deimv2/deimv2_dinov3_s_coco.yml -r deimv2_dinov3_s_coco.pth -i test1.jpg -o result.jpg --classes "飞机,船,车辆,建筑"

# 调整置信度阈值（默认0.45）
python predict.py -c configs/deimv2/deimv2_dinov3_s_coco.yml -r deimv2_dinov3_s_coco.pth -i test1.jpg -o result.jpg --conf 0.3

# 使用CPU推理（如果没有GPU）
python predict.py -c configs/deimv2/deimv2_dinov3_s_coco.yml -r deimv2_dinov3_s_coco.pth -i test1.jpg -o result.jpg --device cpu
```

#### 5.1.2 使用训练好的模型推理

训练完成后，使用保存的checkpoint进行推理：

```bash
# 使用最后一个epoch的checkpoint（假设训练了50轮，epoch 49是最后一个）
python predict.py -c configs/deimv2/deimv2_dinov3_s_coco.yml -r outputs/deimv2_dinov3_s_coco/checkpoint0049.pth -i test1.jpg -o result.jpg

# 指定你的类别名称
python predict.py -c configs/deimv2/deimv2_dinov3_s_coco.yml -r outputs/deimv2_dinov3_s_coco/checkpoint0049.pth -i test1.jpg -o result.jpg --classes "类别1,类别2,类别3,类别4"
```

**参数说明：**
- `-c`: 配置文件路径（与训练时使用的配置文件一致）
- `-r`: 模型权重文件路径
  - 预训练模型：`deimv2_dinov3_s_coco.pth`（项目根目录）
  - 训练后的模型：`outputs/deimv2_dinov3_s_coco/checkpoint00XX.pth`
- `-i`: 输入图像路径
- `-o`: 输出图像路径（可选，默认为 `result.jpg`）
- `--conf`: 置信度阈值，范围 0-1（可选，默认 0.45）
- `--device`: 推理设备（可选，默认 `cuda:0`）
- `--classes`: 类别名称，用逗号分隔（可选）

**输出示例：**
```
============================================================
DEIMv2 目标检测推理
============================================================
配置文件: configs/deimv2/deimv2_dinov3_s_coco.yml
模型权重: deimv2_dinov3_s_coco.pth
输入图像: test1.jpg
输出图像: result.jpg
推理设备: cuda:0
置信度阈值: 0.45
============================================================

[1/5] 加载配置文件...
[2/5] 加载模型权重...
  ✓ 使用EMA权重
  ✓ 模型加载完成
[3/5] 加载图像...
  ✓ 图像尺寸: 1920 x 1080
[4/5] 执行推理...
  ✓ 推理完成，检测到 300 个候选目标

检测到 5 个目标（置信度 > 0.45）:
  [1] 飞机 - 置信度: 0.892 - 位置: (120, 45, 380, 210)
  [2] 船 - 置信度: 0.756 - 位置: (500, 600, 750, 820)
  [3] 车辆 - 置信度: 0.634 - 位置: (890, 340, 1050, 490)
  [4] 建筑 - 置信度: 0.598 - 位置: (1200, 150, 1600, 800)
  [5] 飞机 - 置信度: 0.521 - 位置: (450, 80, 680, 230)

[5/5] 绘制检测结果...

============================================================
✅ 检测完成！结果已保存到: result.jpg
============================================================
```

### 5.2 在验证集上评估模型性能

训练完成后，在验证集上计算AP等指标：

```bash
# 评估训练好的模型
python train.py -c configs/deimv2/deimv2_dinov3_s_coco.yml --test-only -r outputs/deimv2_dinov3_s_coco/checkpoint0049.pth

# 评估预训练模型（在COCO数据集上）
python train.py -c configs/deimv2/deimv2_dinov3_s_coco.yml --test-only -r deimv2_dinov3_s_coco.pth
```

**评估输出示例：**
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all ] = 0.523
Average Precision  (AP) @[ IoU=0.50      | area=   all ] = 0.712
Average Precision  (AP) @[ IoU=0.75      | area=   all ] = 0.568
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small ] = 0.334
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium ] = 0.567
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large ] = 0.689
```

### 5.3 使用原始推理工具（高级）

如果需要更多功能，可以使用项目自带的推理工具：

```bash
# 推理单张图片（输出固定为 torch_results.jpg）
python tools/inference/torch_inf.py -c configs/deimv2/deimv2_dinov3_s_coco.yml -r outputs/deimv2_dinov3_s_coco/checkpoint0049.pth -i your_image.jpg -d cuda:0

# 推理视频（输出固定为 torch_results.mp4）
python tools/inference/torch_inf.py -c configs/deimv2/deimv2_dinov3_s_coco.yml -r outputs/deimv2_dinov3_s_coco/checkpoint0049.pth -i your_video.mp4 -d cuda:0
```

---

## 6. 训练参数说明

### 6.1 重要超参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `epoches` | 总训练轮数（包括EMA） | 实际训练轮数 + 12 |
| `total_batch_size` | 批次大小 | 根据显存调整（4/8/16/32） |
| `lr` | 学习率 | 0.0005（默认） |
| `num_workers` | 数据加载线程数 | 4-8 |
| `flat_epoch` | 学习率平台期 | 4 + 实际训练轮数 // 2 |
| `no_aug_epoch` | 最后不使用数据增强的轮数 | 12 |

### 6.2 显存优化建议

| 显存大小 | 推荐batch_size | 推荐模型 |
|---------|---------------|----------|
| 4GB | 2 | Atto/Femto/Pico |
| 6GB | 4 | N |
| 8GB | 4-8 | S |
| 12GB | 8-16 | S/M |
| 16GB+ | 16-32 | S/M/L |

**如果显存不足，可以：**
1. 减小 `total_batch_size`
2. 使用 `--use-amp`（混合精度）
3. 减小输入图像尺寸（修改 `Resize` 的 `size` 参数）
4. 选择更小的模型（如 Atto/Femto/Pico/N）

### 6.3 调整训练轮数

假设你想训练 **100** 轮，需要修改以下参数：

```yaml
epoches: 112          # 100 + 12
flat_epoch: 54        # 4 + 100 // 2
no_aug_epoch: 12      # 保持不变

train_dataloader:
  dataset:
    transforms:
      policy:
        epoch: [4, 54, 100]   # [start, flat_epoch, 实际训练轮数]
  collate_fn:
    mixup_epochs: [4, 54]
    stop_epoch: 100
    copyblend_epochs: [4, 100]

DEIMCriterion:
  matcher:
    matcher_change_epoch: 90  # 约为实际轮数的90%
```

### 6.4 调整批次大小

如果要将 batch_size 改为 **8**：

```yaml
train_dataloader:
  total_batch_size: 8
```

如果使用多GPU且想要更大的批次（如总batch=16，用2个GPU）：
```yaml
train_dataloader:
  total_batch_size: 16  # 每个GPU会分到 16/2=8 的batch
```

---

## 7. 常见问题

### 7.1 CUDA Out of Memory（显存不足）

**解决方法：**
1. 减小 `total_batch_size`
2. 使用 `--use-amp`
3. 减小图像尺寸
4. 关闭其他占用显存的程序

### 7.2 训练损失不下降

**可能原因：**
1. 学习率过大或过小 → 调整 `lr` 参数
2. 数据集标注错误 → 检查标注文件
3. 类别数设置错误 → 检查 `num_classes`
4. 预训练权重未正确加载 → 检查 `-t` 参数

### 7.3 数据加载慢

**解决方法：**
1. 增加 `num_workers` 数量（如改为8）
2. 使用SSD而不是机械硬盘
3. 减小图像分辨率

### 7.4 如何查看训练日志

训练日志保存在 `output_dir` 指定的目录中：
```
outputs/deimv2_dinov3_s_waste_detection/
├── checkpoint0000.pth  # 模型检查点
├── checkpoint0049.pth
├── config.yml          # 训练配置备份
├── log.txt             # 训练日志
└── summary/            # TensorBoard日志
```

使用 TensorBoard 可视化：
```bash
tensorboard --logdir outputs/deimv2_dinov3_s_waste_detection/summary
```

### 7.5 如何选择模型大小

| 模型 | 参数量 | 速度 | 精度 | 适用场景 |
|------|--------|------|------|----------|
| Atto | 0.5M | 最快 | 低 | 边缘设备、实时性要求极高 |
| Femto | 1.0M | 很快 | 中低 | 移动设备 |
| Pico | 1.5M | 快 | 中 | 嵌入式设备 |
| N | 3.6M | 快 | 中高 | 一般场景 |
| **S** | 9.7M | 中 | 高 | **推荐用于大多数场景** |
| M | 18.1M | 中 | 很高 | 精度要求高的场景 |
| L | 32.2M | 慢 | 极高 | 精度优先 |
| X | 50.3M | 最慢 | 最高 | 学术研究、精度优先 |

### 7.6 类别ID说明

- **本项目使用 0-based 索引**（与标准COCO格式不同）
- 如果你有 4 个类别，`category_id` 应该是：**0, 1, 2, 3**
- 使用项目提供的 `yolo_to_coco.py` 转换脚本会自动处理为 0-based 索引
- 在标注文件的 `categories` 中正确定义每个类别的 id 和 name

**示例（4个类别）：**
```json
{
  "categories": [
    {"id": 0, "name": "飞机", "supercategory": "object"},
    {"id": 1, "name": "船", "supercategory": "object"},
    {"id": 2, "name": "车辆", "supercategory": "object"},
    {"id": 3, "name": "建筑", "supercategory": "object"}
  ]
}
```

**注意：**
- 如果 `category_id` 超出 `[0, num_classes-1]` 范围，训练时会报 CUDA 错误
- 如果从其他来源获得的COCO格式数据集使用1-based索引，需要手动修改为0-based

---

## 8. 命令快速参考

### 8.1 训练命令

#### Windows系统（单GPU，batch=4，epoch=50）
```bash
python train.py -c configs/deimv2/deimv2_dinov3_s_coco.yml --use-amp --seed=0 -t deimv2_dinov3_s_coco.pth --update train_dataloader.total_batch_size=4 epoches=62 flat_epoch=29 no_aug_epoch=12 num_classes=4
```

#### Linux系统（单GPU，batch=4，epoch=50）
```bash
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/deimv2/deimv2_dinov3_s_coco.yml --use-amp --seed=0 -t deimv2_dinov3_s_coco.pth --update train_dataloader.total_batch_size=4 epoches=62 flat_epoch=29 no_aug_epoch=12 num_classes=4
```

#### Linux系统（多GPU，例如4块）
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/deimv2/deimv2_dinov3_s_coco.yml --use-amp --seed=0 -t deimv2_dinov3_s_coco.pth --update train_dataloader.total_batch_size=16 epoches=62 flat_epoch=29 no_aug_epoch=12 num_classes=4
```

#### 通用模板（自定义参数）
```bash
python train.py -c configs/deimv2/deimv2_dinov3_s_coco.yml --use-amp --seed=0 -t deimv2_dinov3_s_coco.pth --update train_dataloader.total_batch_size=<你的batch_size> epoches=<训练轮数+12> flat_epoch=<4+训练轮数//2> no_aug_epoch=12 num_classes=<你的类别数>
```

### 8.2 推理命令

#### 使用预训练模型推理
```bash
# 基础用法
python predict.py -c configs/deimv2/deimv2_dinov3_s_coco.yml -r deimv2_dinov3_s_coco.pth -i test1.jpg -o result.jpg

# 指定类别名称
python predict.py -c configs/deimv2/deimv2_dinov3_s_coco.yml -r deimv2_dinov3_s_coco.pth -i test1.jpg -o result.jpg --classes "飞机,船,车辆,建筑"
```

#### 使用训练好的模型推理
```bash
# 使用checkpoint
python predict.py -c configs/deimv2/deimv2_dinov3_s_coco.yml -r outputs/deimv2_dinov3_s_coco/checkpoint0049.pth -i test1.jpg -o result.jpg --classes "类别1,类别2,类别3,类别4"

# 调整置信度阈值
python predict.py -c configs/deimv2/deimv2_dinov3_s_coco.yml -r outputs/deimv2_dinov3_s_coco/checkpoint0049.pth -i test1.jpg -o result.jpg --conf 0.3
```

### 8.3 模型评估命令

```bash
# 评估训练好的模型
python train.py -c configs/deimv2/deimv2_dinov3_s_coco.yml --test-only -r outputs/deimv2_dinov3_s_coco/checkpoint0049.pth --update num_classes=4
```

---

## 9. 训练指标说明

训练过程中会显示以下指标：

| 指标 | 说明 |
|------|------|
| **AP** | Average Precision，平均精度（主要评价指标） |
| **AP50** | IoU=0.5时的AP |
| **AP75** | IoU=0.75时的AP |
| **AP_small** | 小目标的AP |
| **AP_medium** | 中等目标的AP |
| **AP_large** | 大目标的AP |
| **loss** | 总损失 |
| **loss_vfl** | 分类损失 |
| **loss_bbox** | 边界框回归损失 |
| **loss_giou** | GIoU损失 |

**训练目标：**
- AP 越高越好（范围：0-100）
- loss 应该逐渐下降并趋于稳定

---

## 10. 联系与支持

如果遇到问题，可以：
1. 查看项目 [Issues](https://github.com/Intellindust-AI-Lab/DEIMv2/issues)
2. 阅读原始 [README.md](./README.md)
3. 参考论文：[arXiv:2509.20787](https://arxiv.org/abs/2509.20787)

---

## 11. 快速开始示例

完整的训练+推理流程：

```bash
# ============================================================
# 步骤 1: 环境准备
# ============================================================
conda activate deimv2

# ============================================================
# 步骤 2: 数据集准备
# ============================================================
# 如果是YOLO格式，先转换为COCO格式：
python yolo_to_coco.py

# 检查生成的数据集结构：
# dataset/
# ├── train/
# │   ├── _annotations.coco.json
# │   └── [图像文件]
# └── valid/
#     ├── _annotations.coco.json
#     └── [图像文件]

# ============================================================
# 步骤 3: 修改配置文件
# ============================================================
# 3.1 打开 configs/dataset/custom_detection.yml
#     修改 num_classes 为你的类别数（例如：4）
#     确认 img_folder 和 ann_file 路径正确

# 3.2 打开 configs/deimv2/deimv2_dinov3_s_coco.yml
#     将第2行的 '../dataset/coco_detection.yml' 改为 '../dataset/custom_detection.yml'

# ============================================================
# 步骤 4: 开始训练
# ============================================================
# 单GPU训练（batch=4，50轮，混合精度，使用预训练权重）
python train.py -c configs/deimv2/deimv2_dinov3_s_coco.yml --use-amp --seed=0 -t deimv2_dinov3_s_coco.pth --update train_dataloader.total_batch_size=4 epoches=62 flat_epoch=29 no_aug_epoch=12

# 训练过程中会看到：
# Epoch: [0]  [0/480]  loss: 77.07  ...
# 训练日志保存在：outputs/deimv2_dinov3_s_coco/

# ============================================================
# 步骤 5: 在验证集上评估模型（训练完成后）
# ============================================================
python train.py -c configs/deimv2/deimv2_dinov3_s_coco.yml --test-only -r outputs/deimv2_dinov3_s_coco/checkpoint0049.pth

# 会输出AP、AP50、AP75等评估指标

# ============================================================
# 步骤 6: 使用训练好的模型推理单张图片
# ============================================================
# 使用简易脚本（推荐）
python predict.py -c configs/deimv2/deimv2_dinov3_s_coco.yml -r outputs/deimv2_dinov3_s_coco/checkpoint0049.pth -i test1.jpg -o result.jpg --classes "类别1,类别2,类别3,类别4"

# 或使用预训练模型测试
python predict.py -c configs/deimv2/deimv2_dinov3_s_coco.yml -r deimv2_dinov3_s_coco.pth -i test1.jpg -o result.jpg

# 结果图片会保存为 result.jpg
```

---

## 12. 实战案例：4类遥感目标检测

假设你有一个遥感图像数据集，包含 4 个类别：**飞机、船、车辆、建筑**

### 完整流程

```bash
# 1. 转换数据集（如果是YOLO格式）
python yolo_to_coco.py

# 2. 修改 configs/dataset/custom_detection.yml
#    设置 num_classes: 4

# 3. 修改 configs/deimv2/deimv2_dinov3_s_coco.yml
#    将 '../dataset/coco_detection.yml' 改为 '../dataset/custom_detection.yml'

# 4. 训练（8GB显存，batch=4，50轮）
python train.py -c configs/deimv2/deimv2_dinov3_s_coco.yml --use-amp --seed=0 -t deimv2_dinov3_s_coco.pth --update train_dataloader.total_batch_size=4 train_dataloader.num_workers=4 val_dataloader.num_workers=4 epoches=62 flat_epoch=29 no_aug_epoch=12 num_classes=4

# 5. 评估模型
python train.py -c configs/deimv2/deimv2_dinov3_s_coco.yml --test-only -r outputs/deimv2_dinov3_s_coco/checkpoint0049.pth --update num_classes=4

# 6. 推理测试图片
python predict.py -c configs/deimv2/deimv2_dinov3_s_coco.yml -r outputs/deimv2_dinov3_s_coco/checkpoint0049.pth -i test1.jpg -o result.jpg --classes "飞机,船,车辆,建筑" --conf 0.3
```

**预期训练时间（参考）：**
- 单块 RTX 3060（8GB）：约 5-6 天（50轮）
- 单块 RTX 4090（24GB）：约 1-2 天（50轮，可用更大batch_size）

**训练完成标志：**
- `outputs/deimv2_dinov3_s_coco/` 目录下生成多个 `checkpoint00XX.pth` 文件
- `log.txt` 文件记录了完整的训练日志
- 最后一个checkpoint通常是 `checkpoint0049.pth`（如果训练50轮）

---

## 13. 常见推理场景

### 场景 1：快速测试预训练模型

```bash
python predict.py -c configs/deimv2/deimv2_dinov3_s_coco.yml -r deimv2_dinov3_s_coco.pth -i test1.jpg -o result.jpg
```

### 场景 2：使用训练好的模型，显示自定义类别名称

```bash
python predict.py -c configs/deimv2/deimv2_dinov3_s_coco.yml -r outputs/deimv2_dinov3_s_coco/checkpoint0049.pth -i test1.jpg -o result.jpg --classes "飞机,船,车辆,建筑"
```

### 场景 3：降低置信度阈值，检测更多目标

```bash
python predict.py -c configs/deimv2/deimv2_dinov3_s_coco.yml -r outputs/deimv2_dinov3_s_coco/checkpoint0049.pth -i test1.jpg -o result.jpg --conf 0.2
```

### 场景 4：批量推理（可写简单脚本）

```python
import os
import subprocess

# 待推理的图片列表
images = ['test1.jpg', 'test2.jpg', 'test3.jpg']
class_names = "飞机,船,车辆,建筑"

for img in images:
    output = f"result_{os.path.basename(img)}"
    cmd = f"python predict.py -c configs/deimv2/deimv2_dinov3_s_coco.yml -r outputs/deimv2_dinov3_s_coco/checkpoint0049.pth -i {img} -o {output} --classes {class_names}"
    subprocess.run(cmd, shell=True)
    print(f"✓ {img} -> {output}")
```

**关键提示：**
1. 确保数据集在 `dataset/` 目录下，格式正确
2. 修改 `custom_detection.yml` 中的 `num_classes`
3. 修改模型配置文件，使用 `custom_detection.yml`
4. 训练时通过 `--update num_classes=你的类别数` 确保类别数正确
5. 推理时使用 `--classes` 参数指定类别名称，提高可读性

祝训练顺利！🎉

