"""Training and evaluation loops shared by the segmentation models.

Batches come straight from the existing data loaders:
    PH2:   (image, mask, valid)
    DRIVE: (image, vessel_mask, fov_mask)
The third tensor is used as the `valid` mask, so losses and metrics only count
real image pixels (and, for DRIVE, pixels inside the field of view).
"""

from __future__ import annotations

import copy

import torch
from matplotlib.figure import Figure
from torch.utils.data import DataLoader

from metrics import MetricTracker


def unpack_batch(batch, device):
    """Return (images, targets, valid) on `device`."""
    images, targets, *rest = batch
    valid = rest[0].to(device) if rest else None
    return images.to(device), targets.to(device), valid


def eval_loader(dataset, batch_size, num_workers=0):
    """
    Loader over `dataset` with augmentation off and no shuffling. Used to
    evaluate the training split; the dataset is copied so that the training
    loader keeps its random flips.
    """
    dataset = copy.copy(dataset)
    dataset.augment = False
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )


def compute_pos_weight(loader):
    """#background / #foreground pixels (inside the valid mask), for weighted BCE."""
    positives, negatives = 0.0, 0.0
    for batch in loader:
        _, targets, valid = unpack_batch(batch, "cpu")
        if valid is None:
            valid = torch.ones_like(targets)
        positives += (targets * valid).sum().item()
        negatives += ((1 - targets) * valid).sum().item()
    return negatives / positives


def train_one_epoch(model, loader, loss_fn, optimizer, device, max_batches=None):
    """One pass over the training data; returns the mean training loss."""
    model.train()
    total_loss, total_images = 0.0, 0
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        images, targets, valid = unpack_batch(batch, device)

        loss = loss_fn(model(images), targets, valid)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(images)
        total_images += len(images)
    return total_loss / max(total_images, 1)


@torch.no_grad()
def evaluate(model, loader, device, loss_fn=None, max_batches=None):
    """Return (mean loss, metric summary). The loss is None if no loss_fn is given."""
    model.eval()
    tracker = MetricTracker()
    total_loss, total_images = 0.0, 0
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        images, targets, valid = unpack_batch(batch, device)
        logits = model(images)

        tracker.update(logits, targets, valid)
        if loss_fn is not None:
            total_loss += loss_fn(logits, targets, valid).item() * len(images)
            total_images += len(images)

    mean_loss = total_loss / total_images if total_images else None
    return mean_loss, tracker.summary()


def fit(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    device,
    epochs,
    checkpoint_path,
    max_batches=None,
):
    """
    Train for `epochs` epochs. After each epoch the weights are saved to
    `checkpoint_path` if the validation Dice improved. At the end, the best
    weights are loaded back into `model`. Returns (best_epoch, history).
    """
    best_dice, best_epoch, history = -1.0, 0, []
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, max_batches
        )
        val_loss, val_metrics = evaluate(
            model, val_loader, device, loss_fn, max_batches
        )
        val_dice = val_metrics["dice"]["mean"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice": val_dice,
            }
        )

        improved = val_dice > best_dice
        if improved:
            best_dice, best_epoch = val_dice, epoch
            torch.save(model.state_dict(), checkpoint_path)
        print(
            f"epoch {epoch:3d}/{epochs}  train loss {train_loss:.4f}  "
            f"val loss {val_loss:.4f}  val dice {val_dice:.4f}"
            + ("  (best)" if improved else "")
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return best_epoch, history


@torch.no_grad()
def save_prediction_figure(model, loader, device, path, max_images=4):
    """Save image / ground truth / prediction for the first images of `loader`."""
    model.eval()
    images, targets, valid = unpack_batch(next(iter(loader)), device)
    preds = (model(images) > 0).float()
    if valid is not None:
        preds = preds * valid  # pixels outside the FOV are not evaluated

    n = min(max_images, len(images))
    fig = Figure(figsize=(9, 3 * n))
    axes = fig.subplots(n, 3, squeeze=False)
    for row in range(n):
        panels = (
            images[row].permute(1, 2, 0).cpu(),
            targets[row, 0].cpu(),
            preds[row, 0].cpu(),
        )
        for ax, panel, title in zip(
            axes[row], panels, ("image", "ground truth", "prediction")
        ):
            ax.imshow(panel, cmap="gray" if panel.ndim == 2 else None)
            ax.set_title(title)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
