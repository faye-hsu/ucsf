"""Train the cyst/non-cyst segmentation model on CPU."""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import OrganoidCystDataset, get_train_transform, get_val_transform
from model import build_model

CLASS_NAMES = ["background", "non_cystic", "cystic"]


class HybridLoss(torch.nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        ce_loss = self.ce(y_pred, y_true)
        num_classes = y_pred.shape[1]
        y_true_onehot = F.one_hot(y_true, num_classes).permute(0, 3, 1, 2).float()
        y_pred_soft = torch.softmax(y_pred, dim=1)
        y_pred_flat = y_pred_soft.view(y_pred.shape[0], num_classes, -1)
        y_true_flat = y_true_onehot.view(y_true_onehot.shape[0], num_classes, -1)
        intersection = (y_pred_flat * y_true_flat).sum(dim=-1)
        cardinality = y_pred_flat.sum(dim=-1) + y_true_flat.sum(dim=-1)
        dice_loss = 1.0 - ((2.0 * intersection + self.smooth) / (cardinality + self.smooth)).mean()
        return ce_loss + dice_loss


def dice_per_class(logits, target, num_classes, eps=1e-6):
    probs = torch.softmax(logits, dim=1)
    target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probs * target_onehot).sum(dims)
    union = probs.sum(dims) + target_onehot.sum(dims)
    return (2 * intersection + eps) / (union + eps)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    dice_sum = torch.zeros(3)
    n_batches = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, masks in tqdm(loader, leave=False):
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            dice_sum += dice_per_class(logits.detach(), masks, num_classes=3).cpu()
            n_batches += 1
    return total_loss / n_batches, dice_sum / n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-images", default="data/images")
    parser.add_argument("--train-masks", default="data/masks")
    parser.add_argument("--val-images", default="data/val_images")
    parser.add_argument("--val-masks", default="data/val_masks")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cpu")
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    train_ds = OrganoidCystDataset(args.train_images, args.train_masks,
                                    transform=get_train_transform(args.image_size))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    has_val = Path(args.val_masks).exists() and any(Path(args.val_masks).glob("*.npy"))
    if has_val:
        val_ds = OrganoidCystDataset(args.val_images, args.val_masks,
                                      transform=get_val_transform(args.image_size))
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model().to(device)
    criterion = HybridLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_dice = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        msg = f"Epoch {epoch}: train_loss={train_loss:.4f} " +               " ".join(f"{n}_dice={d:.3f}" for n, d in zip(CLASS_NAMES, train_dice))

        if has_val:
            val_loss, val_dice = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
            scheduler.step(val_loss)
            msg += f" | val_loss={val_loss:.4f} " +                    " ".join(f"{n}_dice={d:.3f}" for n, d in zip(CLASS_NAMES, val_dice))
            cyst_dice = val_dice[2].item()
        else:
            cyst_dice = train_dice[2].item()

        print(msg)

        if cyst_dice > best_dice:
            best_dice = cyst_dice
            torch.save(model.state_dict(), Path(args.checkpoint_dir) / "best_model.pt")
            print(f"  -> saved new best checkpoint (cyst_dice={cyst_dice:.3f})")

    torch.save(model.state_dict(), Path(args.checkpoint_dir) / "last_model.pt")


if __name__ == "__main__":
    main()
