import json
import os
from typing import List, Tuple, Dict
from datetime import datetime


def char_boxes_to_word_polygon(char_bboxes: List[Tuple[int, int, int, int]]) -> List[float]:
    """
    Convert character-level bounding boxes to word-level polygon.
    Handles Thai upper/lower vowels by creating convex hull around all characters.

    Args:
        char_bboxes: List of (x1, y1, x2, y2) for each character

    Returns:
        Polygon coordinates as [x1,y1, x2,y2, x3,y3, x4,y4] (clockwise from top-left)
    """
    if not char_bboxes:
        return []

    # Find min/max coordinates across all character boxes
    all_x1 = [bbox[0] for bbox in char_bboxes]
    all_y1 = [bbox[1] for bbox in char_bboxes]
    all_x2 = [bbox[2] for bbox in char_bboxes]
    all_y2 = [bbox[3] for bbox in char_bboxes]

    min_x = min(all_x1)
    min_y = min(all_y1)
    max_x = max(all_x2)
    max_y = max(all_y2)

    # Create polygon (clockwise from top-left)
    polygon = [
        min_x, min_y,  # top-left
        max_x, min_y,  # top-right
        max_x, max_y,  # bottom-right
        min_x, max_y  # bottom-left
    ]

    return polygon


def create_coco_annotation(
        metadata: Dict,
        annotation_id: int,
        category_id: int = 1
) -> Tuple[Dict, List[Dict]]:
    """
    Create COCO format annotations from metadata.
    Returns word-level + character component annotations.

    Categories:
    1: word
    2: base character (consonant)
    3: leading vowel (เ แ โ ใ ไ)
    4: upper_vowel (ิ ี ึ ื ั)
    5: upper_tone (่ ้ ๊ ๋)
    6: upper_diacritic (์ ํ ็)
    7: lower (ุ ู ฺ)
    8: trailing (า from ำ)
    """
    image_info = {
        "id": metadata["image_id"],
        "file_name": metadata["file_name"],
        "width": metadata["width"],
        "height": metadata["height"]
    }

    annotations = []
    current_ann_id = annotation_id

    # 1. Word-level annotation
    word_polygon = char_boxes_to_word_polygon(metadata["char_bboxes"])

    if word_polygon:
        min_x = min(word_polygon[0:: 2])
        min_y = min(word_polygon[1:: 2])
        max_x = max(word_polygon[0:: 2])
        max_y = max(word_polygon[1:: 2])
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y
        area = bbox_width * bbox_height

        annotations.append({
            "id":  current_ann_id,
            "image_id": metadata["image_id"],
            "category_id": 1,
            "segmentation": [word_polygon],
            "bbox": [min_x, min_y, bbox_width, bbox_height],
            "area": area,
            "iscrowd": 0
        })
        current_ann_id += 1

    # 2. Character component annotations
    if "char_positions" in metadata:
        for char_pos in metadata["char_positions"]:

            # Base character
            if char_pos. get("base_bbox"):
                bbox = char_pos["base_bbox"]
                char_polygon = [
                    bbox[0], bbox[1],
                    bbox[2], bbox[1],
                    bbox[2], bbox[3],
                    bbox[0], bbox[3]
                ]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]

                if width > 0 and height > 0:
                    annotations.append({
                        "id": current_ann_id,
                        "image_id": metadata["image_id"],
                        "category_id": 2,
                        "segmentation": [char_polygon],
                        "bbox": [bbox[0], bbox[1], width, height],
                        "area": width * height,
                        "iscrowd": 0
                    })
                    current_ann_id += 1

            # Leading vowel
            if char_pos.get("leading_bbox"):
                bbox = char_pos["leading_bbox"]
                char_polygon = [
                    bbox[0], bbox[1],
                    bbox[2], bbox[1],
                    bbox[2], bbox[3],
                    bbox[0], bbox[3]
                ]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]

                if width > 0 and height > 0:
                    annotations.append({
                        "id": current_ann_id,
                        "image_id": metadata["image_id"],
                        "category_id": 3,
                        "segmentation": [char_polygon],
                        "bbox": [bbox[0], bbox[1], width, height],
                        "area": width * height,
                        "iscrowd": 0
                    })
                    current_ann_id += 1

            # Upper vowel
            if char_pos. get("upper_vowel_bbox"):
                bbox = char_pos["upper_vowel_bbox"]
                char_polygon = [
                    bbox[0], bbox[1],
                    bbox[2], bbox[1],
                    bbox[2], bbox[3],
                    bbox[0], bbox[3]
                ]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]

                if width > 0 and height > 0:
                    annotations.append({
                        "id": current_ann_id,
                        "image_id": metadata["image_id"],
                        "category_id": 4,
                        "segmentation": [char_polygon],
                        "bbox": [bbox[0], bbox[1], width, height],
                        "area": width * height,
                        "iscrowd": 0
                    })
                    current_ann_id += 1

            # Upper tone
            if char_pos.get("upper_tone_bbox"):
                bbox = char_pos["upper_tone_bbox"]
                char_polygon = [
                    bbox[0], bbox[1],
                    bbox[2], bbox[1],
                    bbox[2], bbox[3],
                    bbox[0], bbox[3]
                ]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]

                if width > 0 and height > 0:
                    annotations.append({
                        "id": current_ann_id,
                        "image_id":  metadata["image_id"],
                        "category_id": 5,
                        "segmentation":  [char_polygon],
                        "bbox": [bbox[0], bbox[1], width, height],
                        "area": width * height,
                        "iscrowd": 0
                    })
                    current_ann_id += 1

            # Upper diacritic
            if char_pos.get("upper_diacritic_bbox"):
                bbox = char_pos["upper_diacritic_bbox"]
                char_polygon = [
                    bbox[0], bbox[1],
                    bbox[2], bbox[1],
                    bbox[2], bbox[3],
                    bbox[0], bbox[3]
                ]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]

                if width > 0 and height > 0:
                    annotations.append({
                        "id": current_ann_id,
                        "image_id":  metadata["image_id"],
                        "category_id": 6,
                        "segmentation":  [char_polygon],
                        "bbox": [bbox[0], bbox[1], width, height],
                        "area": width * height,
                        "iscrowd": 0
                    })
                    current_ann_id += 1

            # Lower vowel
            if char_pos.get("lower_bbox"):
                bbox = char_pos["lower_bbox"]
                char_polygon = [
                    bbox[0], bbox[1],
                    bbox[2], bbox[1],
                    bbox[2], bbox[3],
                    bbox[0], bbox[3]
                ]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]

                if width > 0 and height > 0:
                    annotations.append({
                        "id": current_ann_id,
                        "image_id": metadata["image_id"],
                        "category_id": 7,
                        "segmentation": [char_polygon],
                        "bbox": [bbox[0], bbox[1], width, height],
                        "area": width * height,
                        "iscrowd": 0
                    })
                    current_ann_id += 1

            # Trailing
            if char_pos.get("trailing_bbox"):
                bbox = char_pos["trailing_bbox"]
                char_polygon = [
                    bbox[0], bbox[1],
                    bbox[2], bbox[1],
                    bbox[2], bbox[3],
                    bbox[0], bbox[3]
                ]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]

                if width > 0 and height > 0:
                    annotations.append({
                        "id": current_ann_id,
                        "image_id": metadata["image_id"],
                        "category_id": 8,
                        "segmentation": [char_polygon],
                        "bbox": [bbox[0], bbox[1], width, height],
                        "area": width * height,
                        "iscrowd": 0
                    })
                    current_ann_id += 1

    return image_info, annotations


def save_coco_json(
        output_path: str,
        images: List[Dict],
        annotations: List[Dict],
        categories: List[Dict] = None
):
    """Save COCO format JSON file."""
    if categories is None:
        categories = [
            {"id": 1, "name":  "word"},
            {"id":  2, "name": "base"},
            {"id": 3, "name": "leading"},
            {"id": 4, "name": "upper_vowel"},
            {"id": 5, "name": "upper_tone"},
            {"id": 6, "name": "upper_diacritic"},
            {"id": 7, "name": "lower"},
            {"id": 8, "name": "trailing"}
        ]

    coco_format = {
        "info": {
            "description": "Thai Text Detection Dataset",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    os.makedirs(os.path. dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf8") as f:
        json. dump(coco_format, f, ensure_ascii=False, indent=2)

    print(f"Saved COCO annotations to {output_path}")
    print(f"  Images: {len(images)}")
    print(f"  Annotations: {len(annotations)}")


def convert_metadata_to_coco(
        metadata_dir: str,
        output_dir: str,
        train_ratio: float = 0.8
):
    """
    Convert all metadata files to COCO format and split train/val.

    Args:
        metadata_dir: Directory containing *_metadata.json files
        output_dir: Directory to save train. json and val.json
        train_ratio: Ratio of training data (default: 0.8)
    """
    import glob
    import random

    # Find all metadata files
    metadata_files = glob.glob(os.path.join(metadata_dir, "*_metadata.json"))

    if not metadata_files:
        print(f"No metadata files found in {metadata_dir}")
        return

    # Shuffle for random split
    random.shuffle(metadata_files)

    # Split train/val
    split_idx = int(len(metadata_files) * train_ratio)
    train_files = metadata_files[: split_idx]
    val_files = metadata_files[split_idx:]

    print(f"Total files: {len(metadata_files)}")
    print(f"Train:  {len(train_files)}, Val: {len(val_files)}")

    # Process train set
    train_images = []
    train_annotations = []
    annotation_id = 1

    for meta_file in train_files:
        try:
            with open(meta_file, "r", encoding="utf8") as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Skipping corrupted file: {meta_file} - {e}")
            continue

        image_info, annotations = create_coco_annotation(metadata, annotation_id)

        if annotations:
            train_images.append(image_info)
            train_annotations.extend(annotations)
            annotation_id += len(annotations)

    # Process val set
    val_images = []
    val_annotations = []

    for meta_file in val_files:
        try:
            with open(meta_file, "r", encoding="utf8") as f:
                metadata = json. load(f)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Skipping corrupted file: {meta_file} - {e}")
            continue

        image_info, annotations = create_coco_annotation(metadata, annotation_id)

        if annotations:
            val_images.append(image_info)
            val_annotations.extend(annotations)
            annotation_id += len(annotations)

    # Save COCO JSONs
    save_coco_json(
        os.path. join(output_dir, "annotations", "train.json"),
        train_images,
        train_annotations
    )

    save_coco_json(
        os.path.join(output_dir, "annotations", "val.json"),
        val_images,
        val_annotations
    )


if __name__ == "__main__":
    # Example usage
    convert_metadata_to_coco(
        metadata_dir="out/",
        output_dir="dataset/",
        train_ratio=0.8
    )