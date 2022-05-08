import torch
import torch.nn.functional as F


def criterion(preds, targets):
    targets2 = torch.where(targets >= 0.5, 2*targets -1, 1 - 2*targets)
    preds2 = torch.where(targets >= 0.5, preds, -preds)
    loss = - targets2 * F.logsigmoid(preds2)
    return loss
