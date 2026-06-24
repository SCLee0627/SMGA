#!/usr/bin/env python3
"""Minimal SMGA usage demo.

This script creates a tiny CNN and attacks random toy inputs. It is intended
only to verify that SMGA can be imported and executed.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from SMGA import smga


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(8 * 4 * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCNN().to(device).eval()

    # Toy inputs in [0, 1]
    inputs = torch.rand(4, 1, 28, 28, device=device)
    labels = torch.randint(0, 10, (4,), device=device)

    with open(ROOT / "configs" / "smga_params.json", "r") as f:
        params = json.load(f)

    # Keep the demo quick
    params["steps"] = min(int(params.get("steps", 100)), 10)

    adv = smga(model=model, inputs=inputs, labels=labels, **params)

    l0 = (adv - inputs).flatten(1).ne(0).sum(dim=1)
    print("Adversarial batch shape:", tuple(adv.shape))
    print("L0 per sample:", l0.detach().cpu().tolist())
    print("Input range:", float(adv.min()), float(adv.max()))


if __name__ == "__main__":
    main()
