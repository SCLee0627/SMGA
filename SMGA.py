from __future__ import annotations

import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch import Tensor, nn



def _difference_of_logits(logits: Tensor, labels: Tensor) -> Tensor:
    """Return f_y(x) - max_{j != y} f_j(x).

    The value is positive when the model still predicts the true class with a
    larger logit than every other class, and negative once the sample becomes
    adversarial for untargeted attacks.
    """
    labels = labels.to(logits.device)
    true_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    other_logits = logits.clone()
    other_logits.scatter_(1, labels.view(-1, 1), -torch.inf)
    max_other = other_logits.max(dim=1).values
    return true_logits - max_other


def _batch_view(x: Tensor, ndim: int) -> Tensor:
    """Reshape a batch vector so it broadcasts over image dimensions."""
    return x.view(x.shape[0], *([1] * (ndim - 1)))


def _norm_grad(grad: Tensor, p: float) -> Tensor:
    """Normalize gradients sample-wise."""
    flat = grad.flatten(1)
    denom = flat.norm(p=p, dim=1, keepdim=True).clamp_min(1e-12)
    return (flat / denom).view_as(grad)


def _box_clamp_(delta: Tensor, inputs: Tensor) -> None:
    """Clamp delta in-place so inputs + delta stays inside [0, 1]."""
    delta.data.add_(inputs.data).clamp_(0, 1).sub_(inputs.data)


def _update_best(best_delta: Tensor, best_l0: Tensor, delta: Tensor,
                 active: Tensor, not_adv: Tensor) -> None:
    """Store the best adversarial perturbation found for each active sample."""
    l0 = delta.detach().flatten(1).ne(0).sum(dim=1)
    improved = (~not_adv.bool()) & (l0 < best_l0[active])
    best_l0[active] = torch.where(improved, l0, best_l0[active])
    best_delta[active] = torch.where(
        _batch_view(improved, delta.ndim),
        delta.detach().clone(),
        best_delta[active],
    )


def _l0_surrogate_sq(delta: Tensor, tau: float) -> Tensor:
    """Squared-log L0 proxy: 0.5 * log(1 + (delta / tau)^2).

    Its gradient is zero at delta=0, so zero pixels are not directly pulled back
    by the sparsity term. This lets the classification gradient select useful
    pixels first, while still penalizing nonzero perturbations.
    """
    ratio = delta.flatten(1) / tau
    return 0.5 * torch.log1p(ratio * ratio).sum(dim=1)


def _forward_terms(
    model: nn.Module,
    inputs: Tensor,
    delta: Tensor,
    labels: Tensor,
    tau: float,
    gamma: float,
) -> dict[str, Tensor]:
    """Compute SMGA objective terms for the current perturbation."""
    adv = inputs + delta
    logits = model(adv)
    z = _difference_of_logits(logits, labels)


    gate = torch.sigmoid(gamma * z)
    cls_loss = 1.0 + z.clamp(min=0)

    dim = delta.flatten(1).shape[1]
    l0_proxy = _l0_surrogate_sq(delta, tau) / dim

    preds = logits.argmax(dim=1)
    not_adv = (preds == labels).float()

    return {
        "adv": adv,
        "logits": logits,
        "z": z,
        "gate": gate,
        "cls_loss": cls_loss,
        "l0_proxy": l0_proxy,
        "not_adv": not_adv,
        "loss": gate * cls_loss + l0_proxy,
    }


def smga(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    steps: int = 100,
    lr: float = 1.0,
    tau: float = 0.04,
    threshold: float = 0.3,
    t: float = 0.01,
    grad_norm: float = torch.inf,
    epsilon_budget: int | None = None,
    verbose: bool = False,
    neighbor_samples: int = 4,
    neighbor_radius: float = 0.05,
    neighbor_beta: float = 0.5,
    gamma: float = 50.0,
) -> Tensor:
    """Run SMGA and return adversarial samples.

    Parameters
    ----------
    model:
        PyTorch classifier returning logits.
    inputs:
        Input tensor in [0, 1].
    labels:
        Ground-truth labels for untargeted attack.
    steps:
        Number of optimization iterations.
    lr:
        Adam learning rate.
    tau:
        Scale parameter of the squared-log L0 surrogate.
        The default value 0.04 matches the setting used in the original experiments.
    threshold:
        Initial hard-threshold value for pruning small perturbations.
    t:
        Threshold adaptation rate.
    neighbor_samples / neighbor_radius / neighbor_beta:
        Neighborhood gradient estimation parameters.
    gamma:
        Sharpness of the gated classification loss.
    """
    device = inputs.device
    batch_size = inputs.shape[0]
    dim = inputs[0].numel()

    delta = torch.zeros_like(inputs, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([delta], lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr / 10)

    best_delta = delta.detach().clone()
    best_l0 = torch.full((batch_size,), dim, device=device)
    active = torch.ones(batch_size, dtype=torch.bool, device=device)
    below_budget = torch.zeros(batch_size, dtype=torch.bool, device=device)
    threshold_map = torch.full_like(inputs, threshold)

    for step in range(steps):
        optimizer.zero_grad()
        if not active.any():
            break

        a_delta = delta[active].detach().requires_grad_(True)
        a_inputs = inputs[active]
        a_labels = labels[active]

        # Current gradient
        terms = _forward_terms(model, a_inputs, a_delta, a_labels, tau, gamma)
        _update_best(best_delta, best_l0, a_delta, active, terms["not_adv"])
        if epsilon_budget is not None:
            below_budget = best_l0 <= epsilon_budget

        g_curr = torch.autograd.grad(terms["loss"].mean(), a_delta)[0]
        g_curr = _norm_grad(g_curr.detach(), grad_norm)

        # Neighborhood gradient
        if neighbor_samples > 0 and neighbor_radius > 0:
            grads = []
            for _ in range(neighbor_samples):
                noise = torch.empty_like(a_delta).uniform_(-neighbor_radius, neighbor_radius)
                d_nbr = (a_delta.detach() + noise).detach().requires_grad_(True)
                nbr_terms = _forward_terms(model, a_inputs, d_nbr, a_labels, tau, gamma)
                grads.append(torch.autograd.grad(nbr_terms["loss"].mean(), d_nbr)[0].detach())
            g_nbr = torch.stack(grads, dim=0).mean(dim=0)
        else:
            g_nbr = g_curr

        # SMGA gradient mixing
        g_mix = neighbor_beta * g_curr + (1.0 - neighbor_beta) * g_nbr

        if delta.grad is None:
            delta.grad = torch.zeros_like(delta)
        delta.grad.zero_()
        delta.grad[active] = g_mix
        delta.grad.data = _norm_grad(delta.grad.data, grad_norm)

        optimizer.step()
        scheduler.step()
        eta = scheduler.get_last_lr()[0]

        with torch.no_grad():
            _box_clamp_(delta, inputs)

            # Adaptive thresholding: reduce threshold when still not adversarial,
            # increase it once adversarial to encourage sparsity.
            th_active = threshold_map[active]
            not_adv_bool = terms["not_adv"].bool()
            th_active[not_adv_bool] -= t * eta
            th_active[~not_adv_bool] += t * eta
            threshold_map[active] = th_active.clamp(0, 1)
            delta.data[delta.data.abs() < threshold_map] = 0

            active[below_budget] = False

        if verbose and step % 50 == 0:
            median_l0 = delta.detach().flatten(1).ne(0).sum(dim=1).median().item()
            print(
                f"iter {step:3d}  lr={eta:.4f}  tau={tau:.5f}  "
                f"gate={terms['gate'].mean().item():.4f}  "
                f"cls={terms['cls_loss'].mean().item():.4f}  "
                f"l0={terms['l0_proxy'].mean().item():.4f}  "
                f"median_l0={median_l0}"
            )

    return (inputs + best_delta).clamp(0, 1)


