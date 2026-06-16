"""Convert labelme polygon annotations into binary cyst masks.

Workflow:
1. Open each brightfield image in labelme (`labelme data/images`).
2. Draw polygons around cystic regions and label them "cyst".
   Optionally draw a polygon labeled "organoid" around the whole organoid
   boundary (recommended) so background outside the organoid is excluded
   from area calculations.
3. Save the .json annotation next to the image (labelme default).
4. Run this script to rasterize the polygons into masks:

   python src/labelme_to_masks.py --images data/images --annotations data/annotations --out data/masks
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def polygons_to_mask(shapes, size, label):
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for shape in shapes:
        if shape["label"] != label:
            continue
        points = [tuple(p) for p in shape["points"]]
        draw.polygon(points, fill=255)
    return np.array(mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="data/images")
    parser.add_argument("--annotations", default="data/annotations",
                         help="Directory containing labelme .json files (defaults to alongside images)")
    parser.add_argument("--out", default="data/masks")
    args = parser.parse_args()

    images_dir = Path(args.images)
    ann_dir = Path(args.annotations)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(ann_dir.glob("*.json")) if ann_dir.exists() else sorted(images_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No labelme .json annotations found in {ann_dir} or {images_dir}")

    for jf in json_files:
        data = json.loads(jf.read_text())
        width, height = data["imageWidth"], data["imageHeight"]
        cyst_mask = polygons_to_mask(data["shapes"], (width, height), "cyst")
        organoid_mask = polygons_to_mask(data["shapes"], (width, height), "organoid")

        if not organoid_mask.any():
            # No organoid boundary drawn: treat whole frame as the organoid.
            organoid_mask = np.full((height, width), 255, dtype=np.uint8)

        # Encode as a single 2-channel mask: channel 0 = organoid extent,
        # channel 1 = cyst region (cyst is always a subset of organoid).
        out = np.stack([organoid_mask, cyst_mask], axis=-1)
        stem = jf.stem
        np.save(out_dir / f"{stem}.npy", out)
        print(f"Wrote {out_dir / f'{stem}.npy'} ({width}x{height})")


if __name__ == "__main__":
    main()
