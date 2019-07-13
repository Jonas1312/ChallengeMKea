import torch
import numpy as np


def mixup_criterion(criterion, pred, y_a, y_b, lam, reduction=None):
    if reduction is None:
        reduction = "mean"
    return lam * criterion(pred, y_a, reduction=reduction) + (1 - lam) * criterion(
        pred, y_b, reduction=reduction
    )


def mixup_data(x, y, device, alpha=0.75):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
