import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.dirname(__file__))
from engine.core import YAMLConfig


def draw_detections(image, labels, boxes, scores, class_names=None, threshold=0.45):
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    mask = scores > threshold
    filtered_labels = labels[mask]
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    
    print(f"\n检测到 {len(filtered_boxes)} 个目标（置信度 > {threshold}）:")
    
    for idx, (label, box, score) in enumerate(zip(filtered_labels, filtered_boxes, filtered_scores)):
        x1, y1, x2, y2 = box.tolist()
        label_id = int(label.item())
        confidence = float(score.item())
        
        if class_names and label_id < len(class_names):
            class_name = class_names[label_id]
        else:
            class_name = f"Class_{label_id}"
        
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        
        text = f"{class_name}: {confidence:.2f}"
        bbox = draw.textbbox((x1, y1), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill='red')
        draw.text((x1 + 2, y1 - text_height - 2), text, fill='white', font=font)
        
        print(f"  [{idx+1}] {class_name} - 置信度: {confidence:.3f} - 位置: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
    
    return image


def predict(config_path, model_path, input_image, output_image=None, 
            confidence_threshold=0.45, device='cuda:0', class_names=None):
    
    if not os.path.exists(input_image):
        raise FileNotFoundError(f"输入图像不存在: {input_image}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    if output_image is None:
        output_image = 'result.jpg'
    
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，切换到CPU")
        device = 'cpu'
    
    print(f"{'='*60}")
    print(f"配置: {config_path}")
    print(f"权重: {model_path}")
    print(f"输入: {input_image}")
    print(f"输出: {output_image}")
    print(f"设备: {device} | 阈值: {confidence_threshold}")
    print(f"{'='*60}")
    
    print("\n加载配置...")
    cfg = YAMLConfig(config_path, resume=model_path)
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    
    print("加载模型...")
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
    
    cfg.model.load_state_dict(state)
    
    class InferenceModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
        
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    model = InferenceModel().to(device)
    model.eval()
    
    img_size = cfg.yaml_cfg.get("eval_spatial_size", [640, 640])
    vit_backbone = cfg.yaml_cfg.get('DINOv3STAs', False)
    
    print("处理图像...")
    im_pil = Image.open(input_image).convert('RGB')
    orig_w, orig_h = im_pil.size
    orig_size = torch.tensor([[orig_w, orig_h]]).to(device)
    
    transforms = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
                if vit_backbone else T.Lambda(lambda x: x)
    ])
    
    im_data = transforms(im_pil).unsqueeze(0).to(device)
    
    print("执行推理...")
    with torch.no_grad():
        labels, boxes, scores = model(im_data, orig_size)
    
    labels = labels[0].cpu()
    boxes = boxes[0].cpu()
    scores = scores[0].cpu()
    
    result_image = draw_detections(
        im_pil, labels, boxes, scores, 
        class_names=class_names, 
        threshold=confidence_threshold
    )
    
    result_image.save(output_image)
    print(f"\n{'='*60}")
    print(f"✅ 完成！结果: {output_image}")
    print(f"{'='*60}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='DEIMv2 目标检测推理')
    parser.add_argument('-c', '--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('-r', '--resume', type=str, required=True, help='模型权重路径')
    parser.add_argument('-i', '--input', type=str, required=True, help='输入图像路径')
    parser.add_argument('-o', '--output', type=str, default='result.jpg', help='输出图像路径')
    parser.add_argument('--conf', type=float, default=0.45, help='置信度阈值 (0-1)')
    parser.add_argument('-d', '--device', type=str, default='cuda:0', help='推理设备')
    parser.add_argument('--classes', type=str, default=None, help='类别名称，逗号分隔')
    
    args = parser.parse_args()
    
    class_names = None
    if args.classes:
        class_names = [name.strip() for name in args.classes.split(',')]
    
    try:
        predict(
            config_path=args.config,
            model_path=args.resume,
            input_image=args.input,
            output_image=args.output,
            confidence_threshold=args.conf,
            device=args.device,
            class_names=class_names
        )
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

