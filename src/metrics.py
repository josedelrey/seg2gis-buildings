import numpy as np


def confusion_from_masks(pred_mask, target_mask):
    pred = pred_mask.astype(bool)
    target = target_mask.astype(bool)

    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, np.logical_not(target)).sum())
    fn = int(np.logical_and(np.logical_not(pred), target).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(target)).sum())

    return tp, fp, fn, tn


def metrics_from_confusion(tp, fp, fn, tn, eps=1e-7):
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "iou_building": float((tp + eps) / (tp + fp + fn + eps)),
        "dice_f1": float((2 * tp + eps) / (2 * tp + fp + fn + eps)),
        "precision": float((tp + eps) / (tp + fp + eps)),
        "recall": float((tp + eps) / (tp + fn + eps)),
        "accuracy": float((tp + tn + eps) / (tp + fp + fn + tn + eps)),
    }
