"""Generate simulated point annotations for PH2 weakly supervised segmentation.

Strategies:
  random  : sample positive and negative points uniformly in their regions.
  center  : first positive click is nearest the lesion centroid; extras are random.
  extreme : positive clicks start at top, bottom, left and right lesion extremes;
            extras are random. Negative clicks are random background pixels.

Encoding: 1 = positive, 0 = negative, -1 = unannotated.

The lecture slides introduce random, center and extreme clicks. The exact
sampling implementation here is a practical project choice, not a formula
specified by the slides.
"""
from __future__ import annotations

from typing import Optional
import torch


def _to_2d_mask(mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor")
    if mask.ndim == 3:
        if mask.shape[0] != 1:
            raise ValueError(f"Expected [1,H,W], got {tuple(mask.shape)}")
        mask = mask.squeeze(0)
    if mask.ndim != 2:
        raise ValueError(f"Expected [H,W] or [1,H,W], got {tuple(mask.shape)}")
    return mask > 0.5


def _sample(candidates: torch.Tensor, count: int,
            generator: Optional[torch.Generator] = None) -> torch.Tensor:
    count = min(count, candidates.shape[0])
    if count <= 0:
        return candidates[:0]
    indices = torch.randperm(
        candidates.shape[0], generator=generator, device=candidates.device
    )[:count]
    return candidates[indices]


def _centroid_click(foreground: torch.Tensor) -> torch.Tensor:
    centroid = foreground.float().mean(dim=0)
    distances = ((foreground.float() - centroid) ** 2).sum(dim=1)
    return foreground[torch.argmin(distances)].unsqueeze(0)


def _extreme_clicks(foreground: torch.Tensor) -> torch.Tensor:
    rows, cols = foreground[:, 0], foreground[:, 1]
    candidates = torch.stack([
        foreground[torch.argmin(rows)],  # top
        foreground[torch.argmax(rows)],  # bottom
        foreground[torch.argmin(cols)],  # left
        foreground[torch.argmax(cols)],  # right
    ])
    return torch.unique(candidates, dim=0)


def generate_point_clicks(
    mask: torch.Tensor,
    n_positive: int = 5,
    n_negative: int = 5,
    strategy: str = "random",
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return sparse annotation [1,H,W], positive coords and negative coords.

    Coordinates use (row, column) order. Requested counts are capped by the
    number of available foreground/background pixels.
    """
    if n_positive < 0 or n_negative < 0:
        raise ValueError("Click counts must be non-negative")
    strategy = strategy.lower()
    if strategy not in {"random", "center", "extreme"}:
        raise ValueError("strategy must be 'random', 'center', or 'extreme'")

    binary = _to_2d_mask(mask)
    height, width = binary.shape
    foreground = torch.nonzero(binary, as_tuple=False)
    background = torch.nonzero(~binary, as_tuple=False)

    if n_positive and foreground.numel() == 0:
        raise ValueError("Cannot sample positive clicks from an empty lesion mask")
    if n_negative and background.numel() == 0:
        raise ValueError("Cannot sample negative clicks: no background pixels")

    if strategy == "random":
        positive = _sample(foreground, n_positive, generator)
    elif strategy == "center":
        if n_positive == 0:
            positive = foreground[:0]
        else:
            center = _centroid_click(foreground)
            remaining = foreground[~(foreground == center[0]).all(dim=1)]
            extra = _sample(remaining, n_positive - 1, generator)
            positive = torch.cat([center, extra], dim=0)
    else:  # extreme
        if n_positive == 0:
            positive = foreground[:0]
        else:
            extremes = _extreme_clicks(foreground)
            selected = extremes[:n_positive]
            remaining_count = n_positive - selected.shape[0]
            if remaining_count:
                is_extreme = (foreground[:, None, :] == extremes[None, :, :]).all(dim=2).any(dim=1)
                remaining = foreground[~is_extreme]
                extra = _sample(remaining, remaining_count, generator)
                positive = torch.cat([selected, extra], dim=0)
            else:
                positive = selected

    negative = _sample(background, n_negative, generator)

    annotations = torch.full(
        (1, height, width), -1.0, dtype=torch.float32, device=binary.device
    )
    if positive.numel():
        annotations[0, positive[:, 0], positive[:, 1]] = 1.0
    if negative.numel():
        annotations[0, negative[:, 0], negative[:, 1]] = 0.0

    return annotations, positive, negative


if __name__ == "__main__":
    sample_mask = torch.zeros(1, 256, 256)
    sample_mask[:, 70:190, 85:175] = 1

    for name in ("random", "center", "extreme"):
        clicks, pos, neg = generate_point_clicks(
            sample_mask, 5, 5, name,
            generator=torch.Generator().manual_seed(42),
        )
        assert torch.all(sample_mask[0, pos[:, 0], pos[:, 1]] > 0.5)
        assert torch.all(sample_mask[0, neg[:, 0], neg[:, 1]] <= 0.5)
        assert (clicks == 1).sum().item() == len(pos)
        assert (clicks == 0).sum().item() == len(neg)
        print(f"{name}: positive={len(pos)}, negative={len(neg)}, "
              f"unannotated={(clicks == -1).sum().item()}")
    print("All click-generation checks passed.")
