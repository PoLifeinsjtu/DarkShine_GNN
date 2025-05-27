import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, weight=None):
        # Caculate BCE without reduction
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        if weight is not None:
            bce = bce * weight  # shape: [N] or [N, C]
        
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    numerator = 2 * (pred * target).sum()
    denominator = pred.sum() + target.sum()
    return 1 - numerator / denominator

def combined_loss(pred, target, weight=None, alpha=0.5):
    # BCE with optional weight
    if weight is not None:
        bce = F.binary_cross_entropy_with_logits(pred, target, weight=weight)
    else:
        bce = F.binary_cross_entropy_with_logits(pred, target)

    # Dice loss
    dice = dice_loss(pred, target)

    return alpha * bce + (1 - alpha) * dice