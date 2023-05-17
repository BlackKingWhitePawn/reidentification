from torch import nn
import torch
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Функция потерь Contrastive loss"""

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, predicted_distance, actual_distance):
        loss_contrastive = torch.mean(
            (1-actual_distance) * torch.pow(predicted_distance, 2) +
            actual_distance * torch.pow(torch.clamp(self.margin - predicted_distance, min=0.0), 2))

        return loss_contrastive


# TODO: разобраться с ипользованием
class TripletLoss(torch.nn.Module):
    """Функция потерь Triplet Loss"""

    def __init__(self, margin=2.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        squarred_distance_1 = (anchor - positive).pow(2).sum(1)
        squarred_distance_2 = (anchor - negative).pow(2).sum(1)
        triplet_loss = F.relu(
            self.margin + squarred_distance_1 - squarred_distance_2).mean()

        return triplet_loss


class QuadrupletLoss(torch.nn.Module):
    """
    Quadruplet loss function.
    Builds on the Triplet Loss and takes 4 data input: one anchor, one positive and two negative examples. The negative examples needs not to be matching the anchor, the positive and each other.
    """

    def __init__(self, margin1=2.0, margin2=1.0):
        super(QuadrupletLoss, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2

    def forward(self, anchor, positive, negative1, negative2):
        squarred_distance_pos = (anchor - positive).pow(2).sum(1)
        squarred_distance_neg = (anchor - negative1).pow(2).sum(1)
        squarred_distance_neg_b = (negative1 - negative2).pow(2).sum(1)
        quadruplet_loss = \
            F.relu(self.margin1 + squarred_distance_pos - squarred_distance_neg) \
            + F.relu(self.margin2 + squarred_distance_pos -
                     squarred_distance_neg_b)

        return quadruplet_loss.mean()
