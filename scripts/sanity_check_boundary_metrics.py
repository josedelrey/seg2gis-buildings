import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from metrics import boundary_metrics_multi  # noqa: E402


def make_square_mask(top, left, size, shape=(32, 32)):
    mask = np.zeros(shape, dtype=bool)
    mask[top:top + size, left:left + size] = True
    return mask


def assert_close(actual, expected, name, tol=1e-9):
    if abs(actual - expected) > tol:
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def main():
    target = make_square_mask(8, 8, 12)

    perfect = boundary_metrics_multi(target, target)
    assert_close(perfect["boundary_f1_2px"], 1.0, "perfect boundary_f1_2px")
    assert_close(perfect["boundary_iou_2px"], 1.0, "perfect boundary_iou_2px")

    shifted = make_square_mask(9, 9, 12)
    shifted_metrics = boundary_metrics_multi(shifted, target)
    if shifted_metrics["boundary_f1_2px"] < 0.8:
        raise AssertionError(
            "shifted boundary_f1_2px should be high, "
            f"got {shifted_metrics['boundary_f1_2px']}"
        )
    if shifted_metrics["boundary_f1_5px"] < shifted_metrics["boundary_f1_2px"]:
        raise AssertionError(
            "shifted boundary_f1_5px should be >= boundary_f1_2px, "
            f"got {shifted_metrics['boundary_f1_5px']} < "
            f"{shifted_metrics['boundary_f1_2px']}"
        )

    empty = np.zeros_like(target)
    empty_pred_metrics = boundary_metrics_multi(empty, target)
    assert_close(empty_pred_metrics["boundary_f1_2px"], 0.0, "empty pred f1 2px")
    assert_close(empty_pred_metrics["boundary_f1_5px"], 0.0, "empty pred f1 5px")

    empty_both_metrics = boundary_metrics_multi(empty, empty)
    assert_close(empty_both_metrics["boundary_f1_2px"], 1.0, "empty both f1 2px")
    assert_close(empty_both_metrics["boundary_iou_2px"], 1.0, "empty both iou 2px")
    assert_close(empty_both_metrics["boundary_f1_5px"], 1.0, "empty both f1 5px")
    assert_close(empty_both_metrics["boundary_iou_5px"], 1.0, "empty both iou 5px")

    print("Boundary metric sanity checks passed.")


if __name__ == "__main__":
    main()
