"""
author: Antoine Spahr

date : 28.09.2020

----------

TO DO :

"""

import torch
import torch.nn as nn

class BinaryDiceLoss(nn.Module):
    """
    Pytorch Module defining the Dice Loss for the Binary Segmentation Case.
    """
    def __init__(self, reduction='mean', p=2):
        """
        Build a Binary Dice Loss Module.
        ----------
        INPUT
            |---- reduction (str) the reduction to apply over the batch. Default is 'mean'. Other options are: 'none' and 'sum'
            |---- p (uint) the power at which to elevate element before the denominator sum. Since the network output a
            |           sigmoid as prediction, a higher power will enforce the network to increase the prediction for
            |           positive pixels.
        OUTPUT
            |---- BinaryDiceLoss (nn.Module) the Loss module.
        """
        super(BinaryDiceLoss, self).__init__()
        assert reduction in ['mean', 'none', 'sum'], f"Reduction mode: '{reduction}' is not supported. Use either 'mean', 'sum' or 'none'."
        self.reduction = reduction
        self.p = 1
        self.eps = 1 # for stability when mask and predictions are empty

    def forward(self, pred, mask):
        """
        Forward pass of the Binary Dice Loss defined as the 1-DiceCoeff.
        ----------
        INPUT
            |---- pred (torch.tensor) the binary prediction with dimension (Batch x Output Dimension).
            |---- mask (torch.tensor) the binary ground truth with dimension (Batch x Output Dimension).
        OUTPUT
            |---- DL (torch.tensor) the Dice Loss with the selected reduction over the Batch.
        """
        assert pred.shape == mask.shape, f'Prediction and Mask should have the same dimensions! Given: Prediction {pred.shape} / Mask {mask.shape}'
        sum_dim = tuple(range(1, pred.ndim)) # dimension of over which to sum : skip the batch dimension
        #compute the Dice Loss
        inter = torch.sum(pred * mask, dim=sum_dim)
        union = torch.sum(pred.pow(self.p), dim=sum_dim) + torch.sum(mask.pow(self.p), dim=sum_dim)
        DL = 1 - (2 * inter + self.eps)/(union + self.eps)
        # Apply the reduction
        if self.reduction == 'mean':
            return DL.mean()
        elif self.reduction == 'sum':
            return DL.sum()
        elif self.reduction == 'none':
            return DL


class ComboLoss(nn.Module):
    """
    Define the ComboLoss described by Asgari et al. It combines a Dice loss and a Binary cross entropy (BCE) loss in which
    the False Posistive (FP) and False Negative (FN) can be weighted.
    """
    def __init__(self, alpha=0.5, beta=0.5, reduction='mean'):
        """
        Build a ComboLoss module for Binary segmentation.
        ----------
        INPUT
            |---- alpha (float) the relative importance of the BCE compared to the Dice loss. Must be between 0 and 1.
            |---- beta (float) the relative importance of FP compared to FN in the BCE Loss. Must be between 0 and 1.
            |---- reduction (str) the reduction to apply over batches. Supported: 'mean', 'sum', 'none'
        OUTPUT
            |---- ComboLoss (nn.Module) a combo loss module.
        """
        assert alpha >= 0 and alpha <= 1, f'ValueError. alpha must in the range [0,1]. {alpha} given'
        assert beta >= 0 and beta <= 1, f'ValueError. beta must in the range [0,1]. {beta} given'
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        self.bin_dice_loss_fn = BinaryDiceLoss(reduction='none', p=1)

    def forward(self, pred, mask):
        """
        Forward pass of the ComboLoss.
        ----------
        INPUT
            |---- pred (torch.tensor) the binary prediction with dimension (Batch x Output Dimension).
            |---- mask (torch.tensor) the binary ground truth with dimension (Batch x Output Dimension).
        OUTPUT
            |---- combo_loss (torch.tensor) the ComboLoss with the selected reduction over the Batch.
        """
        assert pred.shape == mask.shape, f'Prediction and Mask should have the same dimensions! Given: Prediction {pred.shape} / Mask {mask.shape}'
        sum_dim = tuple(range(1, pred.ndim))

        dice_loss = self.bin_dice_loss_fn(pred, mask)
        bce_loss = - torch.sum(self.beta * mask * torch.log(pred + 1e-14) + (1 - self.beta) * (1 - mask) * torch.log(1 - pred + 1e-14), dim=sum_dim)
        combo_loss = self.alpha * bce_loss + (1 - self.alpha) * dice_loss

        # Apply the reduction
        if self.reduction == 'mean':
            return combo_loss.mean()
        elif self.reduction == 'sum':
            return combo_loss.sum()
        elif self.reduction == 'none':
            return combo_loss
