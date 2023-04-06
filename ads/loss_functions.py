import torch

from torch import nn as nn


def dice_score(pred, target, integer=True):
    """Dice score."""
    return generalized_dice(pred, target, torch.tensor([0, 1]), integer)


def gd_score_from_weights(weights):
    return lambda pred, target: generalized_dice(pred, target, weights, True)


def generalized_dice(pred, target, weights, integer=True):
    """ integer: if True, will pick maximum prediction. If False, will use
                 raw probabilities."""
    eps = 1e-7

    # Both shapes should be [N, num_classes, width, height]
    #                    or [N, num_classes, depth, width, height]
    assert pred.shape == target.shape
    num_input_dims = len(pred.shape)
    assert num_input_dims in (4, 5)
    num_classes = pred.shape[1]
    assert weights.shape == torch.Size([num_classes]), f"Expected weights to be a 1d tensor of length {num_classes} but got shape {weights.shape}"
    assert (weights.sum() - 1).abs() < eps, "Weights must sum to 1"

    if len(pred.shape) == 4:  # 2d
        dims = (2, 3)
    else:  # 3d
        dims = (2, 3, 4)

    if integer:
        pred = pred.detach()
        target = target.detach()
        pred = torch.argmax(pred, dim=1, keepdim=True)
        pred = torch.eye(num_classes)[pred]

        if num_input_dims == 4:
            pred = torch.transpose(pred, 2, 4)
            pred = torch.transpose(pred, 3, 4)
        elif num_input_dims == 5:
            pred = torch.transpose(pred, 2, 5)
            pred = torch.transpose(pred, 4, 5)
            pred = torch.transpose(pred, 3, 4)

        pred = torch.squeeze(pred, dim=1)

    # print(pred.shape)
    intersection = torch.sum(
        weights * torch.sum(torch.mul(pred, target), dim=dims),
        dim=1
    )
    union = torch.sum(
        weights * torch.sum(pred + target, dim=dims),
        dim=1
    )
    scores = 2 * (intersection + eps) / (union + eps)
    # print(intersection, union, scores)
    return scores.mean()


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, x, target):
        return 1 - dice_score(x, target, integer=False)


class GeneralizedDiceLoss(nn.Module):
    def __init__(self, w):
        super(GeneralizedDiceLoss, self).__init__()
        self.w = w

    def forward(self, x, target):
        return 1 - generalized_dice(x, target, self.w, integer=False)
