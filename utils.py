# UNet/utils.py

import torch
import torch.nn.functional as F

def calculate_metrics(preds, targets, smooth=1e-6):
    """
    Calculates Dice, IoU, Accuracy, Precision, and Recall.
    """
    # Apply sigmoid and threshold to get binary predictions
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    # Flatten tensors
    preds = preds.view(-1)
    targets = targets.view(-1)

    # True Positives, False Positives, False Negatives, True Negatives
    TP = (preds * targets).sum()
    FP = ((1 - targets) * preds).sum()
    FN = (targets * (1 - preds)).sum()
    TN = ((1 - targets) * (1 - preds)).sum()

    # Dice Coefficient
    dice = (2. * TP + smooth) / (2 * TP + FP + FN + smooth)

    # Intersection over Union (IoU)
    iou = (TP + smooth) / (TP + FP + FN + smooth)

    # Accuracy
    accuracy = (TP + TN + smooth) / (TP + TN + FP + FN + smooth)

    # Precision
    precision = (TP + smooth) / (TP + FP + smooth)

    # Recall
    recall = (TP + smooth) / (TP + FN + smooth)

    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
    }

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])