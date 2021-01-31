"""
author: Antoine Spahr

date : 09.10.2020

----------

TO DO :
"""
import torch

def batch_binary_confusion_matrix(pred, target):
    """
    Compute the binary confusion matrix for batch of prediction, target.
    ----------
    INPUT
        |---- pred (torch.tensor) the binary prediction with shape (B x *).
        |---- target (torch.tensor) the binary ground truth with shape (B x *).
    OUTPUT
        |---- tn (torch.tensor) The True Negative for each element of the batch. Shape (B).
        |---- fp (torch.tensor) The False Positive for each element of the batch. Shape (B).
        |---- fn (torch.tensor) The False Negative for each element of the batch. Shape (B).
        |---- tp (torch.tensor) The True Positive for each element of the batch. Shape (B).
    """
    assert pred.shape == target.shape, f'Shapes do not match! {pred.shape} =/= {target.shape}'
    assert pred.ndim > 1, f'The tensor must have more that a single dimension. {pred.ndim} dimension passed.'
    # linearize inputs
    t = target.view(target.shape[0], -1)
    p = pred.view(pred.shape[0], -1)
    # compute TP, TN, FP, FN
    tp = (p*t).sum(dim=1)
    tn = ((1-p)*(1-t)).sum(dim=1)
    fp = (p*(1-t)).sum(dim=1)
    fn = ((1-p)*t).sum(dim=1)

    return tn, fp, fn, tp
