"""
Convert YOLO format dataset to COCO format.
This script converts the YOLO dataset structure to COCO format with _annotations.coco.json files.
"""

import json
import os
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import yaml


def load_yolo_config(yaml_path):
    """Load YOLO dataset configuration from yaml file."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def convert_yolo_to_coco_bbox(yolo_bbox, img_width, img_height):
    """
    Convert YOLO bbox format to COCO format.
    
    YOLO format: [x_center, y_center, width, height] (normalized 0-1)
    COCO format: [x_min, y_min, width, height] (absolute pixels)
    """
    x_center, y_center, width, height = yolo_bbox
    
    # Convert from normalized to absolute coordinates
    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    width_abs = width * img_width
    height_abs = height * img_height
    
    # Convert from center format to top-left corner format
    x_min = x_center_abs - width_abs / 2
    y_min = y_center_abs - height_abs / 2
    
    return [x_min, y_min, width_abs, height_abs]


def process_dataset_split(images_dir, labels_dir, output_dir, categories, split_name):
    """
    Process a single dataset split (train/val/test).
    
    Args:
        images_dir: Path to images directory
        labels_dir: Path to labels directory
        output_dir: Output directory for COCO format
        categories: List of category dictionaries
        split_name: Name of the split (for progress bar)
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize COCO format structure
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    # Get all image files
    image_files = sorted(Path(images_dir).glob("*.jpg"))
    if not image_files:
        image_files = sorted(Path(images_dir).glob("*.png"))
    
    annotation_id = 1
    
    print(f"\nProcessing {split_name} split...")
    for img_idx, img_path in enumerate(tqdm(image_files, desc=f"Converting {split_name}")):
        # Read image to get dimensions
        try:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            continue
        
        # Add image info
        image_id = img_idx + 1
        image_info = {
            "id": image_id,
            "file_name": img_path.name,
            "width": img_width,
            "height": img_height
        }
        coco_format["images"].append(image_info)
        
        # Copy image to output directory
        output_img_path = output_dir / img_path.name
        shutil.copy2(img_path, output_img_path)
        
        # Read corresponding label file
        label_path = Path(labels_dir) / (img_path.stem + ".txt")
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Parse YOLO format: class_id x_center y_center width height
                parts = line.split()
                if len(parts) != 5:
                    continue
                
                class_id = int(parts[0])
                yolo_bbox = [float(x) for x in parts[1:5]]
                
                # Convert to COCO format
                coco_bbox = convert_yolo_to_coco_bbox(yolo_bbox, img_width, img_height)
                
                # Calculate area
                area = coco_bbox[2] * coco_bbox[3]
                
                # Add annotation
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,  # Keep original class_id (0-based for this project)
                    "bbox": coco_bbox,
                    "area": area,
                    "iscrowd": 0
                }
                coco_format["annotations"].append(annotation)
                annotation_id += 1
    
    # Save annotations to JSON file
    output_json = output_dir / "_annotations.coco.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(coco_format, f, indent=2)
    
    print(f"✓ {split_name} split completed: {len(coco_format['images'])} images, "
          f"{len(coco_format['annotations'])} annotations")
    print(f"  Output: {output_json}")
    
    return len(coco_format['images']), len(coco_format['annotations'])


def main():
    # Paths (using absolute path to avoid PowerShell encoding issues)
    base_dir = Path(r"D:\U盘备份\毕设\21网络1张金翔 毕业设计\基于RT-DETR的遥感卫星图像目标检测系统\DEIMv2")
    yolo_dataset_dir = base_dir / "datasets" / "Data"
    coco_output_dir = base_dir / "dataset"
    
    # Load YOLO config
    yaml_path = yolo_dataset_dir / "data.yaml"
    print(f"Loading YOLO config from: {yaml_path}")
    config = load_yolo_config(yaml_path)
    
    # Create categories list for COCO format
    categories = []
    for idx, name in enumerate(config['names']):
        categories.append({
            "id": idx,  # Keep 0-based indexing for this project
            "name": name,
            "supercategory": "object"
        })
    
    print(f"\nDataset Info:")
    print(f"  Number of classes: {config['nc']}")
    print(f"  Classes: {config['names']}")
    
    # Process training set
    train_images = yolo_dataset_dir / "train" / "images"
    train_labels = yolo_dataset_dir / "train" / "labels"
    train_output = coco_output_dir / "train"
    
    if train_images.exists() and train_labels.exists():
        train_imgs, train_anns = process_dataset_split(
            train_images, train_labels, train_output, categories, "train"
        )
    else:
        print(f"Warning: Training data not found at {train_images}")
        train_imgs, train_anns = 0, 0
    
    # Process validation set (rename to valid)
    val_images = yolo_dataset_dir / "val" / "images"
    val_labels = yolo_dataset_dir / "val" / "labels"
    valid_output = coco_output_dir / "valid"
    
    if val_images.exists() and val_labels.exists():
        val_imgs, val_anns = process_dataset_split(
            val_images, val_labels, valid_output, categories, "valid"
        )
    else:
        print(f"Warning: Validation data not found at {val_images}")
        val_imgs, val_anns = 0, 0
    
    # Process test set (if exists, otherwise use validation set)
    test_images = yolo_dataset_dir / "test" / "images"
    test_labels = yolo_dataset_dir / "test" / "labels"
    test_output = coco_output_dir / "test"
    
    if test_images.exists() and test_labels.exists():
        test_imgs, test_anns = process_dataset_split(
            test_images, test_labels, test_output, categories, "test"
        )
    else:
        # Use validation set for test set
        print(f"\nTest data not found. Using validation set for test split...")
        if val_images.exists() and val_labels.exists():
            test_imgs, test_anns = process_dataset_split(
                val_images, val_labels, test_output, categories, "test"
            )
        else:
            # Create empty test set if validation also doesn't exist
            print(f"Warning: Validation data also not found. Creating empty test set...")
            test_output.mkdir(parents=True, exist_ok=True)
            empty_coco = {
                "images": [],
                "annotations": [],
                "categories": categories
            }
            with open(test_output / "_annotations.coco.json", 'w', encoding='utf-8') as f:
                json.dump(empty_coco, f, indent=2)
            test_imgs, test_anns = 0, 0
    
    # Summary
    print("\n" + "="*60)
    print("Conversion Summary:")
    print("="*60)
    print(f"Train:  {train_imgs} images, {train_anns} annotations")
    print(f"Valid:  {val_imgs} images, {val_anns} annotations")
    print(f"Test:   {test_imgs} images, {test_anns} annotations")
    print(f"Total:  {train_imgs + val_imgs + test_imgs} images, "
          f"{train_anns + val_anns + test_anns} annotations")
    print("="*60)
    print(f"\nCOCO dataset saved to: {coco_output_dir}")
    print("\nDataset structure:")
    print("dataset/")
    print("├── train/")
    print("│   ├── _annotations.coco.json")
    print("│   └── [image files]")
    print("├── valid/")
    print("│   ├── _annotations.coco.json")
    print("│   └── [image files]")
    print("└── test/")
    print("    ├── _annotations.coco.json")
    print("    └── [image files]")
    print("\n✓ Conversion completed successfully!")


if __name__ == "__main__":
    main()

