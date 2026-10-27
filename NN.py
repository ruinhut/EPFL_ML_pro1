"""
NN train functions.
"""

import numpy as np

from implementations import _sigmoid
from data_processing import expend_data, stratified_holdout_indices
from basic_fuctions import find_best_f1_threshold


def forward_NN(tx, w1, w2):
    """Run the ReLU network with a linear output and final-column biases.

    Args:
        tx: Input feature matrix.
        w1: Input-to-hidden weights, including the bias row.
        w2: Hidden-to-output weights with the final output bias.

    Returns:
        Hidden preactivations, activations, augmented activations and scores.
    """
    z1 = tx @ w1
    h1 = np.maximum(0, z1)
    h1_aug = expend_data(h1)
    s = h1_aug @ w2

    return z1, h1, h1_aug, s


def boost_weight(ty, poly=0.5):
    """Compute inverse-power class weights from the supplied labels.

    Args:
        ty: Labels in {-1, +1}.
        poly: Exponent applied to class counts.

    Returns:
        Positive-class and negative-class weights.
    """

    pos_num, neg_num = np.sum(ty == 1) ** poly, np.sum(ty == -1) ** poly
    pos_boost, neg_boost = pos_num / (pos_num + neg_num), neg_num / (pos_num + neg_num)
    positive_weight = 1 / (2 * pos_boost)
    negative_weight = 1 / (2 * neg_boost)
    return positive_weight, negative_weight


def gradient_NN(tx, ty, w1, w2):
    """Backpropagate the class-weighted MSE.

    Args:
        tx: Input feature matrix.
        ty: Labels in {-1, +1}.
        w1: Input-to-hidden weights, including the bias row.
        w2: Hidden-to-output weights with the final output bias.

    Returns:
        Gradients for w1 and w2.
    """

    N = ty.shape[0]

    # forward
    z1, h1, h1_aug, s = forward_NN(tx, w1, w2)
    positive_weight, negative_weight = boost_weight(ty)
    sample_weights = np.where(ty == 1, positive_weight, negative_weight)

    # backward
    ds = sample_weights * (s - ty) / N
    dw2 = h1_aug.T @ ds
    dh1 = ds[:, None] * w2[:-1][None, :]
    dz1 = dh1 * (z1 > 0)
    dw1 = tx.T @ dz1

    return dw1, dw2


def compute_NN_mse(tx, ty, w1, w2):
    """Compute the class-weighted network MSE.

    Args:
        tx: Input feature matrix.
        ty: Labels in {-1, +1}.
        w1: Input-to-hidden weights, including the bias row.
        w2: Hidden-to-output weights with the final output bias.

    Returns:
        A scalar MSE including the factor 0.5.
    """
    N = ty.shape[0]
    
    # forward
    _, _, _, s = forward_NN(tx, w1, w2)
    
    positive_weight, negative_weight = boost_weight(ty)
    sample_weights = np.where(ty == 1, positive_weight, negative_weight)

    mse = 0.5 * np.sum(sample_weights * (ty - s) ** 2) / N

    return mse


def train_NN_mse(tx, ty, initial_w, epochs, batch_size, gamma, seed):
    """Train MSE network.

    Args:
        tx: Input feature matrix.
        ty: Labels in {-1, +1}.
        initial_w: Initial weight array, or a pair of NN arrays.
        epochs: Number of training epochs.
        batch_size: Observations per mini-batch.
        gamma: Gradient step size.
        seed: Random seed.

    Returns:
        Best w1, best w2, training losses and validation losses.
    """

    train_indeces, valid_indeces = stratified_holdout_indices(ty)
    x_train, y_train = tx[train_indeces], ty[train_indeces]
    x_valid, y_valid = tx[valid_indeces], ty[valid_indeces]

    N = y_train.shape[0]
    rng = np.random.default_rng(seed)

    w1 = initial_w[0].copy()
    w2 = initial_w[1].copy()
    best_w1 = w1.copy()
    best_w2 = w2.copy()
    best_valid_loss = np.inf
    train_losses, valid_losses = [], []

    for _ in range(epochs):
        perm = rng.permutation(N)
        x_shuffled = x_train[perm]                 
        y_shuffled = y_train[perm]                 
        
        for i in range(0, N, batch_size):
            y_batch = y_shuffled[i : min(i + batch_size, N)]
            x_batch = x_shuffled[i : min(i + batch_size, N)]

            dw1, dw2 = gradient_NN(x_batch, y_batch, w1, w2)
            w1 -= gamma * dw1
            w2 -= gamma * dw2
    
        train_loss = compute_NN_mse(x_train, y_train, w1, w2)
        valid_loss = compute_NN_mse(x_valid, y_valid, w1, w2)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_w1 = w1.copy()
            best_w2 = w2.copy()

    return best_w1, best_w2, train_losses, valid_losses


def sigmoid_NN(data):
    """Apply sigmoid .

    Args:
        data: Input data array.

    Returns:
        Scores between zero and one.
    """
    data = np.clip(data, -30, 30)
    return 1 / (1 + np.exp(-data))


def forward_NN_cross_entropy(tx, w1, w2):
    """Run the ReLU network with sigmoid.

    Args:
        tx: Input feature matrix.
        w1: Input-to-hidden weights, including the bias row.
        w2: Hidden-to-output weights with the final output bias.

    Returns:
        Hidden preactivations, activations, augmented activations and probabilities.
    """
    
    z1 = tx @ w1
    h1 = np.maximum(0, z1)
    h1_aug = expend_data(h1)
    logits = h1_aug @ w2
    probabilities = sigmoid_NN(logits)

    return z1, h1, h1_aug, probabilities


def gradient_NN_cross_entropy(tx, ty, w1, w2):
    """gradients of w1 and w2.

    Args:
        tx: Input feature matrix.
        ty: Labels in {-1, +1}.
        w1: Input-to-hidden weights, including the bias row.
        w2: Hidden-to-output weights with the final output bias.

    Returns:
        Gradients for w1 and w2.
    """
        
    N = ty.shape[0]

    # forward
    z1, h1, h1_aug, probabilities = forward_NN_cross_entropy(tx, w1, w2)
    binary_ty = (ty + 1) / 2
    positive_weight, negative_weight = boost_weight(ty)
    sample_weights = np.where(ty == 1, positive_weight, negative_weight)

    # backward
    dlogits = sample_weights * (probabilities - binary_ty) / N
    dw2 = h1_aug.T @ dlogits
    dh1 = dlogits[:, None] * w2[:-1][None, :]
    dz1 = dh1 * (z1 > 0)
    dw1 = tx.T @ dz1

    return dw1, dw2


def compute_NN_cross_entropy(tx, ty, w1, w2):
    """cross binary loss

    Args:
        tx: Input feature matrix.
        ty: Labels in {-1, +1}.
        w1: Input-to-hidden weights, including the bias row.
        w2: Hidden-to-output weights with the final output bias.

    Returns:
        A scalar data loss without L2.
    """

    N = ty.shape[0]

    # forward
    _, _, _, probabilities = forward_NN_cross_entropy(tx, w1, w2)
    probabilities = np.clip(probabilities, 1e-12, 1 - 1e-12)
    binary_ty = (ty + 1) / 2

    positive_weight, negative_weight = boost_weight(ty)
    sample_weights = np.where(ty == 1, positive_weight, negative_weight)

    cross_entropy = -np.sum(sample_weights * (binary_ty * np.log(probabilities) + (1 - binary_ty) * np.log(1 - probabilities))) / N

    return cross_entropy


def train_NN_cross_entropy(tx, ty, initial_w, epochs, batch_size, gamma, seed):
    """Train network.

    Args:
        tx: Input feature matrix.
        ty: Labels in {-1, +1}.
        initial_w: Initial weight array, or a pair of NN arrays.
        epochs: Number of training epochs.
        batch_size: Observations per mini-batch.
        gamma: Gradient step size.
        seed: Random seed.

    Returns:
        Best w1, best w2, training losses and validation losses.
    """

    train_indeces, valid_indeces = stratified_holdout_indices(ty)
    x_train, y_train = tx[train_indeces], ty[train_indeces]
    x_valid, y_valid = tx[valid_indeces], ty[valid_indeces]

    N = y_train.shape[0]
    rng = np.random.default_rng(seed)

    w1 = initial_w[0].copy()
    w2 = initial_w[1].copy()
    best_w1 = w1.copy()
    best_w2 = w2.copy()
    best_valid_loss = np.inf
    train_losses, valid_losses = [], []

    for _ in range(epochs):
        perm = rng.permutation(N)
        x_shuffled = x_train[perm]
        y_shuffled = y_train[perm]

        for i in range(0, N, batch_size):
            y_batch = y_shuffled[i : min(i + batch_size, N)]
            x_batch = x_shuffled[i : min(i + batch_size, N)]

            dw1, dw2 = gradient_NN_cross_entropy(x_batch, y_batch, w1, w2)
            w1 -= gamma * dw1
            w2 -= gamma * dw2

        train_loss = compute_NN_cross_entropy(x_train, y_train, w1, w2)
        valid_loss = compute_NN_cross_entropy(x_valid, y_valid, w1, w2)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_w1 = w1.copy()
            best_w2 = w2.copy()

    return best_w1, best_w2, train_losses, valid_losses


def nn_class_weights(y_train):
    """compute weights of 2 class and give gain.

    Args:
        y_train: Training labels in {-1, +1}.

    Returns:
        gains of 2 classes.
    """
    positive, negative = np.sqrt(np.sum(y_train == 1)), np.sqrt(np.sum(y_train == -1))
    return (float((positive + negative) / (2 * positive)),
            float((positive + negative) / (2 * negative)))


def nn_penalty(w1, w2, lambda_):
    """Compute squared L2 .

    Args:
        w1: Input-to-hidden weights, including the bias row.
        w2: Hidden-to-output weights with the final output bias.
        lambda_: Nonnegative squared L2 coefficient.

    Returns:
        penalty.
    """
    return 0.5 * lambda_ * (np.sum(w1[1:] ** 2, dtype=np.float64)
                            + np.sum(w2[:-1] ** 2, dtype=np.float64))


def nn_bce(y, logits, class_weights):
    """Compute BCE from logits, without L2.

    Args:
        y: Labels in {-1, +1}.
        logits: Unbounded model scores.
        class_weights: Fixed positive-class and negative-class loss weights.

    Returns:
        Mean data loss.
    """
    binary_y = (y == 1).astype(logits.dtype)
    weights = np.where(y == 1, *class_weights).astype(logits.dtype)
    return float(np.mean(weights * (np.logaddexp(0, logits) - binary_y * logits),
                         dtype=np.float64))


def nn_loss_gradient(tx, y, w1, w2, lambda_, class_weights):
    """BP weighted (ReLU hidden layer).

    Args:
        tx: Input feature matrix.
        y: Labels in {-1, +1}.
        w1: Input-to-hidden weights, including the bias row.
        w2: Hidden-to-output weights with the final output bias.
        lambda_: Nonnegative squared L2 coefficient.
        class_weights: Fixed positive-class and negative-class loss weights.

    Returns:
        loss, gradients for w1 and w2.
    """
    z1 = tx @ w1
    h1 = np.maximum(0, z1)
    logits = h1 @ w2[:-1] + w2[-1]
    probabilities = _sigmoid(logits).astype(tx.dtype)
    weights = np.where(y == 1, *class_weights).astype(tx.dtype)
    ds = weights * (probabilities - (y == 1)) / y.size

    dw2 = np.empty_like(w2)
    dw2[:-1] = h1.T @ ds + lambda_ * w2[:-1]
    dw2[-1] = np.sum(ds)
    dz1 = (ds[:, None] * w2[:-1]) * (z1 > 0)
    dw1 = tx.T @ dz1
    dw1[1:] += lambda_ * w1[1:]
    loss = nn_bce(y, logits, class_weights) + nn_penalty(w1, w2, lambda_)
    return loss, dw1, dw2


def nn_logits(tx, w1, w2, batch_size=8192):
    """Compute network logits.

    Args:
        tx: Input feature matrix.
        w1: Input-to-hidden weights, including the bias row.
        w2: Hidden-to-output weights with the final output bias.
        batch_size: Observations per mini-batch.

    Returns:
        logit per row.
    """
    scores = np.empty(tx.shape[0], dtype=w1.dtype)
    for start in range(0, tx.shape[0], batch_size):
        end = min(start + batch_size, tx.shape[0])
        hidden = np.maximum(0, tx[start:end] @ w1)
        scores[start:end] = hidden @ w2[:-1] + w2[-1]
    return scores


def predict_encoded_nn(tx, w1, w2, batch_size=8192):
    """Apply sigmoid 2 network logits.

    Args:
        tx: Input feature matrix.
        w1: Input-to-hidden weights, including the bias row.
        w2: Hidden-to-output weights with the final output bias.
        batch_size: Observations per mini-batch.

    Returns:
        sigmoid score per row.
    """
    return _sigmoid(nn_logits(tx, w1, w2, batch_size))


def train_encoded_nn(x_train, y_train, x_valid, y_valid, hidden_dim=28,
                     epochs=125, batch_size=1024, gamma=0.015, lambda_=0.0001,
                     seed=2026, check_every=5, verbose=True):
    """Train with SGD.

    Args:
        x_train: Training feature matrix.
        y_train: Training labels in {-1, +1}.
        x_valid: Validation feature matrix.
        y_valid: Validation labels in {-1, +1}.
        hidden_dim: Number of ReLU units.
        epochs: Number of training epochs.
        batch_size: Observations per mini-batch.
        gamma: Gradient step size.
        lambda_: Nonnegative squared L2 coefficient.
        seed: Random seed.
        check_every: Epochs between validation checks.
        verbose: Whether to print metrics.

    Returns:
        Weights, threshold, epoch, metrics, class weights and history.
    """
    x_train, x_valid = np.asarray(x_train, dtype=np.float32), np.asarray(x_valid, dtype=np.float32)

    class_weights = nn_class_weights(y_train)
    w1 = (0.02 * np.random.default_rng(seed).standard_normal(
        (x_train.shape[1], hidden_dim))).astype(np.float32)
    w2 = (0.02 * np.random.default_rng(seed + 1).standard_normal(hidden_dim + 1)).astype(np.float32)
    rng = np.random.default_rng(seed)
    best_f1, best_result, history = -1.0, None, []
    for epoch in range(1, epochs + 1):
        permutation = rng.permutation(y_train.size)
        for start in range(0, y_train.size, batch_size):
            indices = permutation[start:start + batch_size]
            _, dw1, dw2 = nn_loss_gradient(x_train[indices], y_train[indices],
                                           w1, w2, lambda_, class_weights)
            w1 -= gamma * dw1
            w2 -= gamma * dw2
        if epoch == 1 or epoch % check_every == 0 or epoch == epochs:
            train_logits = nn_logits(x_train, w1, w2)
            valid_logits = nn_logits(x_valid, w1, w2)
            train_ce = nn_bce(y_train, train_logits, class_weights)
            valid_ce = nn_bce(y_valid, valid_logits, class_weights)
            threshold, metrics = find_best_f1_threshold(y_valid, _sigmoid(valid_logits))
            history.append({"epoch": epoch, "train_ce": train_ce, "valid_ce": valid_ce,
                            "train_objective": train_ce + nn_penalty(w1, w2, lambda_),
                            "threshold": threshold, **metrics})
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_result = {"w1": w1.copy(), "w2": w2.copy(), "threshold": threshold,
                               "best_epoch": epoch, "valid_metrics": metrics.copy()}
            if verbose:
                print(f"epoch {epoch:3d}: train_ce={train_ce:.5f}, valid_ce={valid_ce:.5f}, "
                      f"valid_f1={metrics['f1']:.4f}, threshold={threshold:.6f}", flush=True)
    best_result.update({"class_weights": class_weights, "history": history})
    return best_result


def train_encoded_nn_full(tx, y, lambda_, epochs, hidden_dim=28, batch_size=1024,
                          gamma=0.015, seed=2026, check_every=5, verbose=True):
    """retrain NN on all observations.

    Args:
        tx: Full input matrix with first-column bias.
        y: All training labels in {-1, +1}.
        lambda_: Selected squared L2 coefficient.
        epochs: Epoch count chosen before refitting.
        hidden_dim: Selected number of ReLU units.
        batch_size: Observations per mini-batch.
        gamma: Gradient step size.
        seed: Random seed.
        check_every: Epochs between training-loss records.
        verbose: Whether to print training losses.

    Returns:
        Final weights, full-data class weights and training history.
    """
    tx = np.asarray(tx, dtype=np.float32)
    class_weights = nn_class_weights(y)
    w1 = (0.02 * np.random.default_rng(seed).standard_normal(
        (tx.shape[1], hidden_dim))).astype(np.float32)
    w2 = (0.02 * np.random.default_rng(seed + 1).standard_normal(hidden_dim + 1)).astype(np.float32)
    rng = np.random.default_rng(seed)
    history = []

    for epoch in range(1, epochs + 1):
        permutation = rng.permutation(y.size)
        for start in range(0, y.size, batch_size):
            indices = permutation[start:start + batch_size]
            _, dw1, dw2 = nn_loss_gradient(tx[indices], y[indices], w1, w2, lambda_, class_weights)
            w1 -= gamma * dw1
            w2 -= gamma * dw2
        if epoch == 1 or epoch % check_every == 0 or epoch == epochs:
            logits = nn_logits(tx, w1, w2)
            train_bce = nn_bce(y, logits, class_weights)
            objective = train_bce + nn_penalty(w1, w2, lambda_)
            history.append({"epoch": epoch, "train_bce": train_bce, "train_objective": objective})
            if verbose:
                print(f"full-data epoch {epoch:3d}: train_bce={train_bce:.5f}, objective={objective:.5f}", flush=True)

    return {"w1": w1, "w2": w2, "epochs": epochs, "class_weights": class_weights, "history": history}



