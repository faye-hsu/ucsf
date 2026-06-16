# PKD1 KO Organoid Cyst Segmentation

CPU-trainable pipeline to segment cystic vs non-cystic area in brightfield
images of kidney organoids after PKD1 knockout, and quantify cystic area
fraction across many images.

## Approach

Per-pixel segmentation (3 classes: background, non-cystic organoid tissue,
cystic region) using a U-Net with a MobileNetV2 encoder
(`segmentation_models_pytorch`), pretrained on ImageNet for transfer
learning so it works well with a small labeled dataset and no GPU.

## 1. Setup

```bash
pip install -r requirements.txt
```

## 2. Annotate training images

1. Put brightfield organoid images in `data/images/`.
2. Launch the labeling tool:
   ```bash
   labelme data/images --output data/images
   ```
3. For each image, draw polygons:
   - Label **`cyst`** around every cystic region.
   - Label **`organoid`** around the full organoid boundary (so background
     outside the organoid is excluded from area math). If omitted, the
     whole frame is treated as organoid.
4. Save (creates a `.json` next to each image).
5. Repeat for a held-out validation set in `data/val_images/`.

Aim for at least ~20-30 well-annotated images to start; augmentation
(`src/dataset.py`) helps stretch a small dataset, but more diverse
annotated examples (different timepoints, organoid sizes) will matter more
for accuracy than anything else.

## 3. Convert annotations to masks

```bash
python src/labelme_to_masks.py --images data/images --out data/masks
python src/labelme_to_masks.py --images data/val_images --out data/val_masks
```

## 4. Train

```bash
python src/train.py \
  --train-images data/images --train-masks data/masks \
  --val-images data/val_images --val-masks data/val_masks \
  --epochs 60 --image-size 384
```

Runs on CPU. Tracks per-class Dice; saves the checkpoint with the best
cystic-class Dice to `checkpoints/best_model.pt`. Increase `--epochs` or
add more annotated images if cystic Dice plateaus low.

## 5. Run on new images and quantify area

```bash
python src/infer_and_quantify.py --images data/new_images --checkpoint checkpoints/best_model.pt --out results
```

Outputs:
- `results/overlays/<image>.png` — visual check (green = non-cystic, red = cystic)
- `results/area_summary.csv` — per-image organoid/non-cystic/cystic pixel
  area and cystic area fraction, ready for downstream stats (e.g. compare
  PKD1 KO vs control across batches).

## Notes on accuracy

- Always inspect overlays for a subset of images before trusting the CSV —
  segmentation errors are usually visible (missed small cysts, organoid
  boundary leaking into background).
- Pixel area is relative to image resolution; if images vary in
  magnification/scale, convert pixel counts to physical area using your
  microscope's pixels-per-micron calibration before comparing across
  experiments.
- If accuracy on cysts is the bottleneck, the highest-leverage fix is
  usually adding more annotated images covering edge cases (small/early
  cysts, out-of-focus regions, debris) rather than tuning hyperparameters.
