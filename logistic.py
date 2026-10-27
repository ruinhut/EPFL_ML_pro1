"""
train logistic.
"""

import numpy as np

from implementations import _sigmoid
from basic_fuctions import find_best_f1_threshold

def logistic_objective(binary_y, scores, w, lambda_=0.001, positive_weight=1.0):
    """Compute BCE + L2.

    Args:
        binary_y: Labels in {0, 1}.
        scores: Model scores or sigmoid probabilities.
        w: Linear weights.
        lambda_: Nonnegative squared L2 coefficient.
        positive_weight: Fixed positive-class loss multiplier.

    Returns:
        A scalar regularized objective.
    """
    weights = np.where(binary_y == 1, positive_weight, 1.0)
    losses = np.logaddexp(0.0, scores) - binary_y * scores
    return float(np.sum(weights * losses) / binary_y.size + 0.5 * lambda_ * (w[1:] @ w[1:]))


def logistic_loss_gradient(binary_y, tx, w, lambda_=0.001, positive_weight=1.0):
    """Compute BCE + L2 and gradient.

    Args:
        binary_y: Labels in {0, 1}.
        tx: Input feature matrix.
        w: Linear weights.
        lambda_: Nonnegative squared L2 coefficient.
        positive_weight: Fixed positive-class loss multiplier.

    Returns:
        loss and gradient.
    """
    scores = tx @ w
    weights = np.where(binary_y == 1, positive_weight, 1.0).astype(tx.dtype)
    normalizer = binary_y.size
    loss = logistic_objective(binary_y, scores, w, lambda_, positive_weight)
    errors = (_sigmoid(scores).astype(tx.dtype) - binary_y) * weights
    gradient = tx.T @ errors / normalizer
    gradient[1:] += lambda_ * w[1:]
    return loss, gradient


def predict_scores(tx, w):
    """Apply sigmoid to linear logits.

    Args:
        tx: Input feature matrix.
        w: Linear weights.

    Returns:
        One score per row.
    """
    return _sigmoid(tx @ w)


def train_logistic(x_train, y_train, x_valid, y_valid, lambda_=0.001, epochs=100,
                   batch_size=4096, gamma=0.05, seed=2026, check_every=5,
                   positive_weight=1.0, verbose=True):
    """Train logistic regression with momentum and validation-F1 selection.

    Args:
        x_train: Training feature matrix.
        y_train: Training labels in {-1, +1}.
        x_valid: Validation feature matrix.
        y_valid: Validation labels in {-1, +1}.
        lambda_: Nonnegative squared L2 coefficient.
        epochs: Number of training epochs.
        batch_size: Observations per mini-batch.
        gamma: Gradient step size.
        seed: Random seed.
        check_every: Epochs between validation checks.
        positive_weight: Fixed positive-class loss multiplier.
        verbose: Whether to print metrics.

    Returns:
        Best weights, threshold, epoch, metrics and checkpoint history.
    """
    binary_train = (y_train == 1).astype(x_train.dtype)
    binary_valid = (y_valid == 1).astype(x_valid.dtype)
    
    w = np.zeros(x_train.shape[1], dtype=x_train.dtype)
    prior = float(np.mean(binary_train))
    w[0] = np.log(positive_weight * prior / (1 - prior))
    velocity = np.zeros_like(w)
    rng = np.random.default_rng(seed)
    
    best_f1, best_result, history = -1.0, None, []
    
    for epoch in range(1, epochs + 1):
        permutation = rng.permutation(y_train.size)
        step = gamma / np.sqrt(1 + (epoch - 1) / 20)
        
        for start in range(0, y_train.size, batch_size):
            indices = permutation[start:start + batch_size]
            _, gradient = logistic_loss_gradient(binary_train[indices], x_train[indices], w,
                                                  lambda_, positive_weight)
            velocity = 0.9 * velocity + gradient
            w -= step * velocity

        if epoch == 1 or epoch % check_every == 0 or epoch == epochs:
            train_scores = x_train @ w
            valid_scores = x_valid @ w
            train_loss = logistic_objective(binary_train, train_scores, w, lambda_, positive_weight)
            valid_loss = logistic_objective(binary_valid, valid_scores, w, 0.0, positive_weight)
            threshold, metrics = find_best_f1_threshold(y_valid, _sigmoid(valid_scores))
            
            history.append({"epoch": epoch, "train_objective": train_loss,
                            "valid_bce": valid_loss, "threshold": threshold, **metrics})
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_result = {"w": w.copy(), "threshold": threshold,
                               "best_epoch": epoch, "valid_metrics": metrics.copy()}
            if verbose:
                print(f"epoch {epoch:3d}: train_objective={train_loss:.5f}, valid_bce={valid_loss:.5f}, "
                      f"valid_f1={metrics['f1']:.4f}, threshold={threshold:.6f}", flush=True)
    
    best_result["history"] = history
    return best_result


def train_logistic_full(tx, y, lambda_, epochs, batch_size=4096, gamma=0.05,
                        seed=2026, check_every=5, positive_weight=1.0, verbose=True):
    """Refit LR.

    Args:
        tx: Full input matrix with first-column bias.
        y: All training labels in {-1, +1}.
        lambda_: Selected squared L2 coefficient.
        epochs: Epoch count chosen before refitting.
        batch_size: Observations per mini-batch.
        gamma: Initial gradient step size.
        seed: Random seed.
        check_every: Epochs between training-loss records.
        positive_weight: Positive-class loss multiplier.
        verbose: Whether to print training losses.

    Returns:
        Final weights and training history.
    """
    binary_y = (y == 1).astype(tx.dtype)
    w = np.zeros(tx.shape[1], dtype=tx.dtype)
    prior = float(np.mean(binary_y))
    w[0] = np.log(positive_weight * prior / (1 - prior))
    velocity = np.zeros_like(w)
    rng = np.random.default_rng(seed)
    history = []

    for epoch in range(1, epochs + 1):
        permutation = rng.permutation(y.size)
        step = gamma / np.sqrt(1 + (epoch - 1) / 20)
        for start in range(0, y.size, batch_size):
            indices = permutation[start:start + batch_size]
            _, gradient = logistic_loss_gradient(binary_y[indices], tx[indices], w,
                                                  lambda_, positive_weight)
            velocity = 0.9 * velocity + gradient
            w -= step * velocity
        if epoch == 1 or epoch % check_every == 0 or epoch == epochs:
            logits = tx @ w
            train_bce = logistic_objective(binary_y, logits, w, 0.0, positive_weight)
            objective = logistic_objective(binary_y, logits, w, lambda_, positive_weight)
            history.append({"epoch": epoch, "train_bce": train_bce, "train_objective": objective})
            if verbose:
                print(f"full-data epoch {epoch:3d}: train_bce={train_bce:.5f}, objective={objective:.5f}", flush=True)

    return {"w": w, "epochs": epochs, "history": history}

