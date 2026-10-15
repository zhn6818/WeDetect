import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

# Allow this script to be launched as ``python finetune_uni/batch_infer_uni.py``.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from generate_proposal import SimpleYOLOWorldDetector
from vis import plot_bounding_boxes


def load_model(checkpoint_path: str, num_proposals: int):
    model = SimpleYOLOWorldDetector(
        backbone_size="base",
        prompt_dim=768,
        num_prompts=256,
        num_proposals=num_proposals,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    for key in list(checkpoint.keys()):
        if "backbone" in key:
            new_key = key.replace("backbone.image_model.model.", "backbone.")
            checkpoint[new_key] = checkpoint.pop(key)

    for key in list(checkpoint.keys()):
        if "bbox_head" in key:
            new_key = key.replace("bbox_head.head_module.", "bbox_head.")
            new_key = new_key.replace("0.2.", "0.6.")
            new_key = new_key.replace("1.2.", "1.6.")
            new_key = new_key.replace("2.2.", "2.6.")
            new_key = new_key.replace("1.bn", "4")
            new_key = new_key.replace("1.conv", "3")
            new_key = new_key.replace("0.bn", "1")
            new_key = new_key.replace("0.conv", "0")
            checkpoint[new_key] = checkpoint.pop(key)

    msg = model.load_state_dict(checkpoint, strict=False)
    print(msg)
    return model.cuda().eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--score-thre", type=float, default=0.1)
    parser.add_argument("--num-proposals", type=int, default=300)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )[: args.max_images]
    if not image_paths:
        raise RuntimeError(f"No images found in {input_dir}")

    print(f"images={len(image_paths)}")
    model = load_model(args.checkpoint, args.num_proposals)
    total_detections = 0

    with torch.no_grad():
        for start in range(0, len(image_paths), args.batch_size):
            batch_paths = image_paths[start : start + args.batch_size]
            outputs = model([str(p) for p in batch_paths])
            for image_path, result in zip(batch_paths, outputs):
                boxes = result["bboxes"].float().cpu()
                scores = result["scores"].float().cpu()
                keep = scores > args.score_thre
                boxes = boxes[keep]
                scores = scores[keep]
                total_detections += len(boxes)

                image = Image.open(image_path).convert("RGB")
                visual = plot_bounding_boxes(image, boxes.tolist())
                visual.save(output_dir / f"{image_path.stem}_uni.jpg", quality=95)
                with open(output_dir / f"{image_path.stem}_uni.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "image": str(image_path),
                            "score_threshold": args.score_thre,
                            "boxes": boxes.tolist(),
                            "scores": scores.tolist(),
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
            print(f"processed={min(start + len(batch_paths), len(image_paths))}/{len(image_paths)}")

    print(f"total_detections={total_detections}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
