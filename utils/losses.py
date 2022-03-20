"""Loss functions
"""

import torch
import math
from torch.nn import functional as F


def getValid(true, pred, nonvalid=-1):
    """ On CDNEt dataset, the frames are not fully labeled. Only some predefined region of them are labeled.
    This function extracts the labeled part from ground truth and the corresponding part from prediction as  1-D tensors
    Args:
        true (tensor): Ground truth tensor of shape Bx1xHxW
        preds (tensor): Prediction tensor of shape Bx1xHxW
        nonvalid (int): Value used to indicate nonvalid parts of ground truth

    Returns:
        (tensor): 1-D tensor containing the valid pixels of ground truth
        (tensor): 1-D tensor of prediction corresponding the valid ground truth pixels
    """
    # Turn predictions and labels into 1D arrays
    true_valid = true.reshape(-1)
    pred_valid = pred.reshape(-1)

    # Mask of the known parts of the ground truth
    mask = torch.where(true_valid == nonvalid, torch.tensor(0).cuda(), torch.tensor(1).cuda()).type(torch.bool)

    # Discard the unknown parts from the predictions and labels
    return torch.masked_select(true_valid, mask), torch.masked_select(pred_valid, mask)


def binary_cross_entropy_loss(true, pred, smooth=100):
    epsilon = 1e-7
    loss = torch.tensor(epsilon)
    bce = F.binary_cross_entropy_with_logits(pred, true).clamp(0, 1)
    if not math.isnan(bce):
        loss = (bce * smooth) + epsilon
    return loss


def jaccard_loss(true, pred, smooth=100):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true (tensor): 1D ground truth tensor.
        preds (tensor): 1D prediction truth tensor.
        eps (int): Smoothing factor
    Returns:
        jacc_loss: the Jaccard loss.
    """
    epsilon = 1e-7
    loss = torch.tensor(epsilon)
    intersection = torch.sum(true*pred)
    jac = (intersection + smooth) / (torch.sum(true) + torch.sum(pred) - intersection + smooth)
    if not math.isnan(jac):
        loss = (1 - jac) * smooth + epsilon
    return loss


# tversky loss
def tverskyLoss(true, pred, alpha=0.3, beta=0.7, smooth=100):
    epsilon = 1e-7
    loss = torch.tensor(epsilon)

    pred = pred.contiguous()
    true = true.contiguous()

    TP = torch.sum(pred * true)
    FP = torch.sum((1 - true) * pred)
    FN = torch.sum(true * (1 - pred))

    tversky_loss = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    if not math.isnan(tversky_loss):
        loss = (1 - tversky_loss) * smooth + epsilon
    return loss


# calculate overall loss
def tverskyLoss_bce_loss(true, pred, bceWeight=0.5, alpha=0.3, beta=0.7, smooth=100):
    epsilon = 1e-7

    bce = binary_cross_entropy_loss(pred, true, smooth)
    tversky = tverskyLoss(pred, true, alpha, beta, smooth)

    loss = bce * bceWeight + tversky * (1 - bceWeight) + epsilon

    return loss


def focal_tversky_loss(true, pred, alpha=0.3, beta=0.7, smooth=100, gamma=0.75):
    epsilon = 1e-7
    loss = torch.tensor(epsilon)

    tv = tverskyLoss(true, pred, alpha, beta, smooth)
    if not math.isnan(tv):
        loss = torch.pow(tv, gamma) + epsilon

    return loss


def binary_focal_loss(true, pred, alpha=3.0, gamma=2.0, **kwargs):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    :param **kwargs
        balance_index: (int) balance class index, should be specific when alpha is float
    """
    epsilon = 1e-7
    loss = torch.tensor(epsilon)

    pred = torch.clamp(pred, 0, 1.0)

    pos_mask = (true == 1).float()
    neg_mask = (true == 0).float()

    pos_weight = (pos_mask * torch.pow(1 - pred, gamma)).detach()
    pos_loss = -pos_weight * torch.log(pred)  # / (torch.sum(pos_weight) + 1e-4)

    neg_weight = (neg_mask * torch.pow(pred, gamma)).detach()
    neg_loss = -alpha * neg_weight * F.logsigmoid(-pred)  # / (torch.sum(neg_weight) + 1e-4)
    focal_loss = (pos_loss + neg_loss).mean()

    if not math.isnan(focal_loss):
        loss = focal_loss + epsilon

    return loss

def weighted_crossentropy(true, pred, weight_pos=15, weight_neg=1):
    """Weighted cross entropy between ground truth and predictions
    Args:
        true (tensor): 1D ground truth tensor.
        preds (tensor): 1D prediction truth tensor.
    Returns:
        (tensor): Weighted CE.
    """
    bce = (true*pred.log()) + ((1-true)*(1-pred).log())  # Binary cross-entropy

    # Weighting for class imbalance
    weight_vector = true * weight_pos + (1. - true) * weight_neg
    weighted_bce = weight_vector * bce
    return -torch.mean(weighted_bce + 1e-04)

def acc(true, pred):
    """Accuracy between ground truth and predictions
    Args:
        true (tensor): 1D ground truth tensor.
        preds (tensor): 1D prediction truth tensor.
    Returns:
        acc: Accuracy.
    """

    epsilon = 1e-7
    acc = torch.tensor(epsilon)
    if len(true) > 0 and len(pred) > 0:
       acc = torch.mean((true == pred.round()).float()) + epsilon
    return acc

def f_score(true, pred):
    """False Negative Rate between ground truth and predictions
    Args:
        true (tensor): 1D ground truth tensor.
        preds (tensor): 1D prediction truth tensor.
    Returns:
        (tensor): precision
        (tensor): recall
        (tensor): f-score
    """

    epsilon = 1e-7

    fn = torch.sum(true * (1 - pred))
    fp = torch.sum((1 - true) * pred)
    tp = torch.sum(true * pred)
    prec = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    if tp + fn == 0:
        f_score = torch.tensor(1)
    elif tp == 0:
        f_score = torch.tensor(0)
    else:
        f_score = 2 * (prec * recall) / (prec + recall)

    return f_score
