from torch import nn


def resnet_grad_linear_unlock(model: nn.Module):
    for x in model.parameters():
        x.requires_grad = False
    for x in model.fc.parameters():
        x.requires_grad = True


def resnet_grad_l4_unlock(model: nn.Module):
    for x in model.parameters():
        x.requires_grad = False
    for x in model.layer4.parameters():
        x.requires_grad = True
    for x in model.fc.parameters():
        x.requires_grad = True
