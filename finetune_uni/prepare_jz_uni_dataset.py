"""Prepare the LabelMe JZ tiles as a COCO objectness dataset for WeDetect-Uni."""

import argparse
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
TILE_RE = re.compile(r"^(.*)_r\d+c\d+$")


def tile_group(stem: str) -> str:
    match = TILE_RE.match(stem)
    return match.group(1) if match else stem


def polygon_to_coco(points):
    flat = [float(v) for point in points for v in point[:2]]
    xs = flat[0::2]
    ys = flat[1::2]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return flat, [x0, y0, x1 - x0, y1 - y0], (x1 - x0) * (y1 - y0)


def make_coco(records, image_dir, ann_path, split_name):
    images = []
    annotations = []
    label_counts = Counter()
    ann_id = 1
    for image_id, (image_path, label_path) in enumerate(records, start=1):
        data = json.loads(label_path.read_text(encoding="utf-8"))
        width = int(data["imageWidth"])
        height = int(data["imageHeight"])
        file_name = image_path.name
        images.append({
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
        })
        for shape in data.get("shapes", []):
            points = shape["points"]
            segmentation, bbox, area = polygon_to_coco(points)
            label = str(shape["label"])
            label_counts[label] += 1
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 0,
                "bbox": bbox,
                "area": area,
                "segmentation": [segmentation],
                "iscrowd": 0,
                "source_label": label,
                "source_shape_type": shape.get("shape_type", "polygon"),
            })
            ann_id += 1

    coco = {
        "info": {
            "description": "JZ LabelMe tiles converted for WeDetect-Uni objectness fine-tuning",
            "version": "1.0",
            "split": split_name,
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 0, "synset": "object", "name": "object", "supercategory": "defect"}],
    }
    ann_path.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
    return label_counts, len(annotations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if dst.exists() and any(dst.iterdir()):
        raise RuntimeError(f"Destination is not empty: {dst}")
    (dst / "train2017").mkdir(parents=True, exist_ok=True)
    (dst / "val2017").mkdir(parents=True, exist_ok=True)
    (dst / "annotations").mkdir(parents=True, exist_ok=True)

    records = []
    for image_path in sorted(src.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = image_path.with_suffix(".json")
        if not label_path.exists():
            raise RuntimeError(f"Missing annotation for {image_path.name}")
        records.append((image_path, label_path))

    groups = defaultdict(list)
    for record in records:
        groups[tile_group(record[0].stem)].append(record)
    group_names = sorted(groups)
    random.Random(args.seed).shuffle(group_names)
    val_group_count = max(1, round(len(group_names) * args.val_ratio))
    val_groups = set(group_names[:val_group_count])
    train_records = [r for r in records if tile_group(r[0].stem) not in val_groups]
    val_records = [r for r in records if tile_group(r[0].stem) in val_groups]

    for split, split_records in (("train", train_records), ("val", val_records)):
        image_dir = dst / ("train2017" if split == "train" else "val2017")
        for image_path, _ in split_records:
            shutil.copy2(image_path, image_dir / image_path.name)
        label_counts, object_count = make_coco(
            split_records,
            image_dir,
            dst / "annotations" / f"instances_{split}2017.json",
            split,
        )
        print(split, "images=", len(split_records), "objects=", object_count, "labels=", dict(label_counts))

    classes = [
        "sulfide_a",
        "globular_oxide_d",
        "alumina_b",
        "silicate_c",
        "titanium_d",
        "titanium_b",
        "spherical_oxide_ds",
    ]
    (dst / "classes.txt").write_text("object\n", encoding="utf-8")
    (dst / "source_label_map.json").write_text(
        json.dumps({"object": classes, "mode": "class_agnostic_objectness"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (dst / "split_summary.json").write_text(
        json.dumps({
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "group_count": len(group_names),
            "train_group_count": len(group_names) - len(val_groups),
            "val_group_count": len(val_groups),
            "train_images": len(train_records),
            "val_images": len(val_records),
            "class_agnostic_category": "object",
            "source_labels": classes,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("groups=", len(group_names), "train_groups=", len(group_names) - len(val_groups), "val_groups=", len(val_groups))
    print("output=", dst)


if __name__ == "__main__":
    main()
