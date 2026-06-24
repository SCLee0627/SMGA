# SMGA: Sparse Multi-Gradient Adversarial Attack

This repository is a **minimal single-file code release** of **SMGA** (Sparse Multi-Gradient Adversarial Attack), a gradient-based $\ell_0$ adversarial attack for generating sparse adversarial examples.

SMGA is implemented as a callable PyTorch function. It takes a model, clean inputs, and labels, then returns adversarial examples with as few perturbed pixels/features as possible.

## Repository Structure

```text
SMGA-github-ready/
├── README.md
├── env.yml
├── SMGA.py                # Self-contained SMGA implementation
├── configs/
│   └── smga_params.json    # Example SMGA hyperparameters
└── examples/
    └── demo_smga.py        # Minimal usage demo
```

## Installation

Create the conda environment:

```bash
conda env create -f env.yml
conda activate smga
```

Or install the main dependencies manually:

```bash
pip install torch torchvision numpy scipy tqdm
```

## Usage

```python
from SMGA import smga

adv = smga(
    model=model,
    inputs=inputs,
    labels=labels,
    steps=100,
    lr=1.0,
    tau=0.04,
    threshold=0.3,
    t=0.01,
    neighbor_samples=4,
    neighbor_radius=0.05,
    neighbor_beta=0.5,
    gamma=50.0,
)
```

You may also use the clearer alias:

```python
from SMGA import smga
adv = smga(model, inputs, labels)
```

Requirements for inputs:

- `model` must be a PyTorch model returning logits.
- `inputs` must be a tensor in `[0, 1]`.
- `labels` must be the ground-truth class labels.
- The attack is untargeted by default.

## Important Parameters

| Parameter | Description |
|---|---|
| `steps` | Number of optimization iterations. |
| `lr` | Adam learning rate. |
| `tau` | Scale parameter of the squared-log L0 surrogate. The default `0.04` matches the setting used in the original experiments. |
| `threshold` | Initial hard-threshold value for pruning small perturbations. |
| `t` | Threshold adaptation rate. |
| `neighbor_samples` | Number of neighborhood samples for multi-gradient estimation. |
| `neighbor_radius` | Radius of local neighborhood perturbations. |
| `neighbor_beta` | Interpolation weight between current gradient and neighborhood gradient. |
| `gamma` | Sharpness of the gated classification loss. |

## Minimal Demo

Run:

```bash
python examples/demo_smga.py
```

The demo builds a tiny CNN and applies SMGA to random toy inputs. It is only meant to verify that the attack function can be imported and executed.

## Citation

If you use this code, please cite your SMGA paper/report. Placeholder:

```bibtex
@misc{smga2026,
  title={SMGA: Sparse Multi-Gradient Adversarial Attack for L0-norm Adversarial Examples},
  author={Li, Sichen},
  year={2026},
  note={Code release}
}
```
