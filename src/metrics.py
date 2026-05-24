import cv2
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


def mask_to_boundary(mask):
    mask_bool = np.asarray(mask).astype(bool)

    if not mask_bool.any():
        return np.zeros(mask_bool.shape, dtype=bool)

    mask_uint8 = mask_bool.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask_uint8, kernel, iterations=1)
    boundary = np.logical_xor(mask_uint8, eroded)

    return boundary.astype(bool)


def dilate_boundary(boundary, tolerance_px):
    boundary_bool = np.asarray(boundary).astype(bool)

    if tolerance_px <= 0:
        return boundary_bool

    kernel_size = 2 * int(tolerance_px) + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    dilated = cv2.dilate(boundary_bool.astype(np.uint8), kernel, iterations=1)

    return dilated.astype(bool)


def boundary_metrics(pred_mask, target_mask, tolerance_px, eps=1e-7):
    pred_boundary = mask_to_boundary(pred_mask)
    target_boundary = mask_to_boundary(target_mask)

    pred_boundary_count = int(pred_boundary.sum())
    target_boundary_count = int(target_boundary.sum())

    if pred_boundary_count == 0 and target_boundary_count == 0:
        return {
            "boundary_precision": 1.0,
            "boundary_recall": 1.0,
            "boundary_f1": 1.0,
            "boundary_iou": 1.0,
        }

    pred_boundary_dilated = dilate_boundary(pred_boundary, tolerance_px)
    target_boundary_dilated = dilate_boundary(target_boundary, tolerance_px)

    if pred_boundary_count == 0:
        precision = 1.0
    else:
        matched_pred = np.logical_and(pred_boundary, target_boundary_dilated)
        precision = float(matched_pred.sum() / pred_boundary_count)

    if target_boundary_count == 0:
        recall = 1.0
    else:
        matched_target = np.logical_and(target_boundary, pred_boundary_dilated)
        recall = float(matched_target.sum() / target_boundary_count)

    if precision + recall <= eps:
        boundary_f1 = 0.0
    else:
        boundary_f1 = float((2 * precision * recall) / (precision + recall))

    intersection = np.logical_and(
        pred_boundary_dilated,
        target_boundary_dilated,
    )
    union = np.logical_or(
        pred_boundary_dilated,
        target_boundary_dilated,
    )
    if union.sum() == 0:
        boundary_iou = 1.0
    else:
        boundary_iou = float(intersection.sum() / union.sum())

    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": boundary_f1,
        "boundary_iou": boundary_iou,
    }


def boundary_metrics_multi(pred_mask, target_mask, tolerances=(2, 5), eps=1e-7):
    metrics = {}

    for tolerance_px in tolerances:
        tolerance_metrics = boundary_metrics(
            pred_mask,
            target_mask,
            tolerance_px,
            eps=eps,
        )
        suffix = f"{int(tolerance_px)}px"
        metrics[f"boundary_f1_{suffix}"] = tolerance_metrics["boundary_f1"]
        metrics[f"boundary_iou_{suffix}"] = tolerance_metrics["boundary_iou"]
        metrics[f"boundary_precision_{suffix}"] = tolerance_metrics[
            "boundary_precision"
        ]
        metrics[f"boundary_recall_{suffix}"] = tolerance_metrics[
            "boundary_recall"
        ]

    return metrics
