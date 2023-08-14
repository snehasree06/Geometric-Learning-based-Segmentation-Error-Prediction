import torch
from torch.nn import functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
from monai.losses import DiceLoss, DiceCELoss
from torch.nn import functional as F


def dice_loss(score, target):
    target = target.float()
    score = score.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def iou_loss(score, target):
    target = target.float()
    score = score.float()
    smooth = 1e-5
    tp_sum = torch.sum(score * target)
    fp_sum = torch.sum(score * (1-target))
    fn_sum = torch.sum((1-score) * target)
    loss = (tp_sum + smooth) / (tp_sum + fp_sum + fn_sum + smooth)
    loss = 1 - loss
    return loss

def entropy_loss(p,C=2):
    ## p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent

def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div



def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp



def f1(probability, targets):
    probability = probability.flatten()
    targets = targets.flatten()
    
    assert (probability.shape == targets.shape)

    intersection = 2.0 * (probability * targets).sum()
    union = (probability * probability).sum() + (targets * targets).sum()
    dice_score = intersection / union
    return 1.0 - dice_score


def dice_loss_multilabel(preds, targets, num_classes):
    smooth = 1e-5
    for i in range(num_classes):
        pred = preds[:,i,:,:,:]
        target = (targets[:,i,:,:,:]==1)
        target = target.float()
        intersection = torch.sum(pred * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(pred * pred)
        dice_coef = ((2.0 * intersection + smooth) /  (z_sum + y_sum + smooth))
        dice_loss_ = 1-dice_coef
        if i == 0:
            dice_loss = dice_loss_
        else:
            dice_loss = dice_loss + dice_loss_
    dice_loss = dice_loss/num_classes
    return dice_loss


def dice_loss_multilabel(preds, targets, num_classes):
    smooth = 1e-5
    for i in range(num_classes):
        pred = preds[:,i,:,:]>0.5
        pred = pred.float()
        target = (targets[:,i,:,:]==1.)
        target = target.float()
        intersection = torch.sum(pred * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(pred * pred)
        dice_coef = ((2.0 * intersection + smooth) /  (z_sum + y_sum + smooth))
        dice_loss_ = 1-dice_coef
        if i == 0:
            dice_loss = dice_loss_
        else:
            dice_loss = dice_loss + dice_loss_
    dice_loss = dice_loss/num_classes
    return dice_loss

def sdtm_regression_loss(pred_sdt,gt_sdt):
    smooth = 1e-5
    pred_sdt = pred_sdt.float()
    gt_sdt = gt_sdt.float()
    intersection = torch.sum(gt_sdt * pred_sdt)
    y_sum = torch.sum(gt_sdt * gt_sdt)
    z_sum = torch.sum(pred_sdt * pred_sdt)
    loss = ((-1.0*intersection) /  (intersection + z_sum + y_sum))
    return loss


