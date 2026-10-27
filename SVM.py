"""
Realize SVM functions
"""


import numpy as np
from implementations import *
from helpers import *
import matplotlib.pyplot as plt

from basic_fuctions import *


def soft_SVM_linear(y, tx, w, lambda_, sample_weights=None, regularize_mask=None):
    """Compute hinge objective and subgradient.

    Args:
        y: Labels in {-1, +1}.
        tx: Input feature matrix.
        w: Linear weights.
        lambda_: Nonnegative squared L2 coefficient.
        sample_weights: Optional per-sample loss weights.
        regularize_mask: Binary mask; zero excludes a coefficient from L2.

    Returns:
        subgradient and hinge loss.
    """
    N = y.shape[0]  # Number of observations.
    if sample_weights is None:
        sample_weights = np.ones(N)
    if regularize_mask is None:
        regularize_mask = np.ones_like(w)

    margin = y * (tx @ w)
    hinge_loss = np.maximum(0, 1 - margin)
    regularized_w = regularize_mask * w
    loss = 0.5 * lambda_ * (regularized_w @ regularized_w) + np.mean(sample_weights * hinge_loss)

    mask = (margin < 1)
    if np.any(mask):
        weighted_y = sample_weights[mask] * y[mask]
        grad_data = (weighted_y[:, None] * tx[mask]).sum(axis=0)
    else:
        grad_data = np.zeros_like(w)
    gradient = lambda_ * regularize_mask * w - (1 / N) * grad_data

    return gradient, loss


def rbf_kernel(X1, X2, gamma):
    """Compute RBF similarities.

    Args:
        X1: First matrix of observations.
        X2: Second matrix of observations.
        gamma: RBF kernel width.

    Returns:
        A matrix with shape (len(X1), len(X2)).
    """
    sq1 = np.sum(X1**2, axis=1).reshape(-1, 1)
    sq2 = np.sum(X2**2, axis=1).reshape(1, -1)
    sqdist = sq1 + sq2 - 2 * X1 @ X2.T
    return np.exp(-gamma * sqdist)


def soft_SVM_rbf(y, X, alpha, lambda_, gamma, K=None):
    """Compute the kernel hinge and coefficient gradient.

    Args:
        y: Labels in {-1, +1}.
        X: Input feature matrix.
        alpha: Kernel coefficients.
        lambda_: Nonnegative squared L2 coefficient.
        gamma: RBF kernel width.
        K: Optional precomputed kernel matrix.

    Returns:
        gradient and hinge loss.
    """
    N = y.shape[0]
    if K is None:
        K = rbf_kernel(X, X, gamma)

    # Evaluate the kernel expansion.
    f = K @ alpha
    margin = y * f
    hinge = np.maximum(0, 1 - margin)

    loss = 0.5 * lambda_ * (alpha @ K @ alpha) + np.mean(hinge)

    mask = margin < 1
    grad_data = K @ (y * mask)
    gradient = lambda_ * (K @ alpha) - (1 / N) * grad_data

    return gradient, loss


def gd_batch_train(y, tx, initial_w, batch_size, max_iters, gamma, seed, epochs=50):
    """Train linear MSE SVM.

    Args:
        y: Labels in {-1, +1}.
        tx: Input feature matrix.
        initial_w: Initial weight array, or a pair of NN arrays.
        batch_size: Observations per mini-batch.
        max_iters: Gradient steps per batch.
        gamma: Gradient step size.
        seed: Random seed.
        epochs: Number of training epochs.

    Returns:
        Final weights and per-epoch MSE values.
    """
    N = y.shape[0]   
    rng = np.random.default_rng(seed)

    temp_w = initial_w.copy()
    losses = []

    for epoch in range(epochs):
        perm = rng.permutation(N)
        x_shuffled = tx[perm]                 
        y_shuffled = y[perm]                 
        
        for i in range(0, N, batch_size):
            y_batch = y_shuffled[i : min(i + batch_size, N)]
            x_batch = x_shuffled[i : min(i + batch_size, N)]
            temp_w, _ = mean_squared_error_gd(y_batch, x_batch, temp_w, max_iters, gamma)
    
        loss = compute_mse(y, tx, temp_w)
        losses.append(loss)

    return temp_w, losses


def mse_linear_training(y_train, x_train, seed, batch_size, max_iters, gamma, return_stats=False):
    """Standardize inputs and train linear SVM.

    Args:
        y_train: Training labels in {-1, +1}.
        x_train: Training feature matrix.
        seed: Random seed.
        batch_size: Observations per mini-batch.
        max_iters: Gradient steps per batch.
        gamma: Gradient step size.
        return_stats: Whether to return fitted statistics.

    Returns:
        Weights and losses
    """

    x_train, mean, std = standardize(x_train, return_stats=True)
    shape = x_train.shape[1]
    initial_w = generate_w(shape, seed)
    final_w, losses = gd_batch_train(y_train, x_train, initial_w, batch_size, max_iters, gamma, seed)

    if return_stats:
        return final_w, losses, mean, std

    return final_w, losses


def train_linear_svm(x, y, epochs=8, batch_size=4096, learning_rate=0.01, lambda_=1e-3, seed=42):
    """Train class-weighted linear SVM .

    Args:
        x: Input feature matrix.
        y: Labels in {-1, +1}.
        epochs: Number of training epochs.
        batch_size: Observations per mini-batch.
        learning_rate: Initial gradient step size.
        lambda_: Nonnegative squared L2 coefficient.
        seed: Random seed.

    Returns:
        Feature weights, bias and loss.
    """
    
    rng = np.random.default_rng(seed)
    w_with_bias = np.zeros(x.shape[1] + 1)
    regularize_mask = np.ones_like(w_with_bias)
    regularize_mask[-1] = 0.0
    losses = []

    positive_weight = y.size / (2.0 * np.sum(y == 1))
    negative_weight = y.size / (2.0 * np.sum(y == -1))

    for epoch in range(epochs):
        order = rng.permutation(y.size)
        epoch_learning_rate = learning_rate / np.sqrt(epoch + 1.0)
        epoch_loss = 0.0

        for start in range(0, y.size, batch_size):
            indices = order[start : start + batch_size]
            x_batch = np.column_stack([x[indices], np.ones(indices.size)])
            y_batch = y[indices]
            sample_weights = np.where(y_batch == 1, positive_weight, negative_weight)
            gradient, batch_loss = soft_SVM_linear(y_batch, x_batch, w_with_bias, lambda_, sample_weights=sample_weights, regularize_mask=regularize_mask)
            w_with_bias -= epoch_learning_rate * gradient
            epoch_loss += batch_loss * indices.size

        losses.append(epoch_loss / y.size)

    return w_with_bias[:-1], w_with_bias[-1], losses


def fit_rff(input_dim, n_components=128, gamma=None, seed=2026):
    """Sample Random Fourier Feature for RBF kernel.

    Args:
        input_dim: Number of input features.
        n_components: Number of random Fourier features.
        gamma: RBF kernel width; None uses the function default.
        seed: Random seed.

    Returns:
        Frequency matrix, phase and gamma.
    """

    if gamma is None:
        gamma = 1.0 / (2.0 * input_dim)

    rng = np.random.default_rng(seed)
    omega = rng.normal(0.0, np.sqrt(2.0 * gamma), size=(input_dim, n_components))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=n_components)
    return omega, phase, gamma


def transform_rff(x, omega, phase):
    """Apply a fitted Random Fourier Feature map.

    Args:
        x: Input feature matrix.
        omega: Fitted frequency matrix.
        phase: Fitted phase vector.

    Returns:
        A float32 feature matrix.
    """
        
    features = np.sqrt(2.0 / omega.shape[1]) * np.cos(x @ omega + phase)
    return features.astype(np.float32)

