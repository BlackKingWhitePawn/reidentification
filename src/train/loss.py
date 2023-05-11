from torch import nn
import torch
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Функция потерь Contrastive loss"""

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

# TODO: поменять
    def forward(self, x, y):
        loss_contrastive = torch.mean(
            (1-y) * torch.pow(x, 2) +
            y * torch.pow(torch.clamp(self.margin - x, min=0.0), 2))

        return loss_contrastive
