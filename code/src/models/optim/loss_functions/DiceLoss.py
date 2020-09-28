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
    def __init__(self, reduction='mean'):
        """
        Build a Binary Dice Loss Module.
        ----------
        INPUT
            |---- reduction (str) the reduction to apply over the batch. Default is 'mean'. Other options are: 'none' and 'sum'
        OUTPUT
            |---- BinaryDiceLoss (nn.Module) the Loss module.
        """
        super(BinaryDiceLoss, self).__init__()
        assert reduction in ['mean', 'none', 'sum'], f"Reduction mode: '{reduction}' is not supported. Use either 'mean', 'sum' or 'none'."
        self.reduction = reduction
        self.eps = 1e-9 # for stability when mask and predictions are empty

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
        sum_dim = dim=tuple(range(1, pred.ndim)) # dimension of over which to sum
        #compute the Dice Loss
        inter = torch.sum(pred * mask, dim=sum_dim)
        union = torch.sum(pred, dim=sum_dim) + torch.sum(mask, dim=sum_dim)
        DL = 1 - (2 * inter + self.eps)/(union + self.eps)
        # Apply the reduction
        if self.reduction == 'mean':
            return DL.mean()
        elif self.reduction == 'sum':
            return DL.sum()
        elif self.reduction == 'none':
            return DL
