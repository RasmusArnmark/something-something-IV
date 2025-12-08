"""
Metrics for evaluating video action recognition models.
"""

import torch


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    
    Args:
        output: model predictions (batch_size, num_classes)
        target: ground truth labels (batch_size,)
        topk: tuple of k values to compute accuracy for
    
    Returns:
        list of accuracies for each k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def top_k_accuracy(output, target, k=5):
    """
    Computes top-k accuracy.
    
    Args:
        output: model predictions (batch_size, num_classes)
        target: ground truth labels (batch_size,)
        k: k value for top-k accuracy
    
    Returns:
        top-k accuracy as a percentage
    """
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
        return correct_k.mul_(100.0 / batch_size).item()
