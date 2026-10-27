"""
some basic function need everywhere
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_draw(arrays, label, title, xlabel, ylabel):
    """Plot training and validation curve.

    Args:
        arrays: Values to plot.
        label: Legend label.
        title: Plot title.
        xlabel: Horizontal-axis label.
        ylabel: Vertical-axis label.

    Returns:
        None; display the curve.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(arrays, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()   

def generate_w(shape, seed):
    """Initialize weights randomly.

    Args:
        shape: Shape of the initial weights.
        seed: Random seed.

    Returns:
        An array of the requested shape.
    """
    rng = np.random.default_rng(seed)
    initial_w = 0.02 * rng.standard_normal(shape)
    
    return initial_w

def num_not_nan(data):
    """Count how many nums that are not NaN.

    Args:
        data: Input data array.

    Returns:
        The number of not nan entries.
    """
    return np.sum(~(np.isnan(data)))

def classification_metrics(y_true, scores, threshold=0.0):
    """Compute positive-class metrics (positive: score >= threshold)

    Args:
        y_true: True labels in {-1, +1}.
        scores: Model scores or sigmoid probabilities.
        threshold: Predict +1 when score >= threshold.

    Returns:
        A dictionary contains tp tn fp fn and so on
    """
    y_pred = np.where(scores >= threshold, 1, -1)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == -1) & (y_true == -1))
    fp = np.sum((y_pred == 1) & (y_true == -1))
    fn = np.sum((y_pred == -1) & (y_true == 1))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-15)

    return {
        "accuracy": (tp + tn) / y_true.size,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }

def print_metrics(name, metrics):
    """Print classification metrics directly

    Args:
        name: Label for printed results.
        metrics: Metric dictionary.

    Returns:
        None.
    """
    print(f"{name}: accuracy={metrics['accuracy']:.4f}, balanced_accuracy={metrics['balanced_accuracy']:.4f}, precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}")
    print(f"  TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}")

def find_best_f1_threshold(y, scores):
    """Maximize F1 over score cut points.

    Args:
        y: Labels in {-1, +1}.
        scores: Model scores or sigmoid probabilities.

    Returns:
        A threshold and metrics; ties prefer the higher threshold.
    """

    y, scores = np.asarray(y), np.asarray(scores)

    scores = scores.astype(np.float64)

    if not np.any(y == 1):
        # no positive prediction
        threshold = float(np.nextafter(np.max(scores), np.inf))
        return threshold, classification_metrics(y, scores, threshold)
    
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), y.size - 1]
    tp = np.cumsum(y[order] == 1)[ends]
    predicted_positive = ends + 1
    f1 = 2.0 * tp / (predicted_positive + np.sum(y == 1))
    best_index = int(np.argmax(f1))
    threshold = float(sorted_scores[ends[best_index]])
    return threshold, classification_metrics(y, scores, threshold)

