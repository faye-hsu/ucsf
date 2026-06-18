"""Run the trained model on a folder of brightfield images and report
cystic vs non-cystic area for each organoid, plus a combined CSV summary.

python src/infer_and_quantify.py --images data/new_images --checkpoint checkpoints/best_model.pt --out results
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_fill_holes

from model import build_model


def preprocess(image, image_size):
    h, w = image.shape
    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    resized = clahe.apply(resized)
    norm = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
    tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).float()
    return tensor, (h, w)


def fill_cyst_holes(pred):
    cyst_mask = pred == 2
    filled = binary_fill_holes(cyst_mask)
    new_cyst = filled & (pred != 2)
    pred = pred.copy()
    pred[new_cyst] = 2
    return pred


def circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0
    return 4 * np.pi * area / (perimeter ** 2)


def has_dark_border(image_gray, mask, border_width=6, darkness_threshold=0.85):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_width * 2 + 1, border_width * 2 + 1))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel)
    border_ring = (dilated - mask.astype(np.uint8)).astype(bool)
    if not border_ring.any():
        return False
    inner_mean = image_gray[mask].mean() if mask.any() else 255
    border_mean = image_gray[border_ring].mean()
    return border_mean < inner_mean * darkness_threshold


def clean_prediction(pred, image_gray, image_area):
    cleaned = pred.copy()
    min_organoid_area = image_area * 0.001
    min_cyst_area = image_area * 0.0005
    min_circularity = 0.45

    organoid_mask = (pred > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(organoid_mask, connectivity=8)
    clean_organoid = np.zeros_like(organoid_mask)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_organoid_area:
            clean_organoid[labels == i] = 1

    cleaned[clean_organoid == 0] = 0

    cyst_mask = (cleaned == 2).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cyst_mask, connectivity=8)
    clean_cyst = np.zeros_like(cyst_mask)

    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_cyst_area:
            continue
        blob = (labels == i)
        contours, _ = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        circ = circularity(contours[0])
        if circ < min_circularity:
            continue
        if not has_dark_border(image_gray, blob):
            continue
        clean_cyst[blob] = 1

    result = np.zeros_like(pred)
    result[clean_organoid == 1] = 1
    result[clean_cyst == 1] = 2
    return result


def predict_mask(model, image, image_size, device):
    tensor, (h, w) = preprocess(image, image_size)
    with torch.no_grad():
        logits = model(tensor.to(device))
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    pred = fill_cyst_holes(pred)
    pred = clean_prediction(pred, image, h * w)
    return pred


def overlay_mask(image, pred):
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = color.copy()
    overlay[pred == 1] = (0, 200, 0)
    overlay[pred == 2] = (0, 0, 255)
    return cv2.addWeighted(color, 0.6, overlay, 0.4, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    device = torch.device("cpu")
    model = build_model().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    out_dir = Path(args.out)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)

    rows = []
    image_paths = sorted(p for p in Path(args.images).glob("*")
                          if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    if not image_paths:
        raise SystemExit(f"No images found in {args.images}")

    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        pred = predict_mask(model, image, args.image_size, device)

        organoid_px = int(np.count_nonzero(pred > 0))
        cyst_px = int(np.count_nonzero(pred == 2))
        non_cyst_px = organoid_px - cyst_px
        cyst_frac = cyst_px / organoid_px if organoid_px > 0 else float("nan")

        rows.append({
            "image": path.name,
            "organoid_area_px": organoid_px,
            "non_cystic_area_px": non_cyst_px,
            "cystic_area_px": cyst_px,
            "cystic_fraction": cyst_frac,
        })

        overlay = overlay_mask(image, pred)
        cv2.imwrite(str(out_dir / "overlays" / path.name), overlay)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "area_summary.csv", index=False)
    print(df)
    print(f"\nSaved overlays to {out_dir / 'overlays'} and summary to {out_dir / 'area_summary.csv'}")


if __name__ == "__main__":
    main()
