import numpy as np


def compute_mse(y, tx, w):
    """Compute the mean squared error loss.

    Args:
        y: NumPy array of shape (N,) containing the target values.
        tx: NumPy array of shape (N, D) containing the input features.
        w: NumPy array of shape (D,) containing the model parameters.

    Returns:
        A scalar loss equal to ||y - tx @ w||^2 / (2N).
    """
    n_samples = y.shape[0]
    errors = y - tx @ w
    return np.sum(errors**2) / (2 * n_samples)


def compute_cross_entropy(y, tx, w):
    """Compute the mean binary cross-entropy loss.

    Args:
        y: NumPy array of shape (N,) containing binary labels in {0, 1}.
        tx: NumPy array of shape (N, D) containing the input features.
        w: NumPy array of shape (D,) containing the model parameters.

    Returns:
        The mean binary cross-entropy loss as a scalar, without a
        regularization penalty.
    """
    scores = tx @ w
    # logaddexp implements log(1 + exp(scores)) without numerical overflow.
    return np.mean(np.logaddexp(0.0, scores) - y * scores)


def compute_mse_gradient(y, tx, w):
    """Compute the gradient of the mean squared error loss.

    Args:
        y: NumPy array of shape (N,) containing the target values.
        tx: NumPy array of shape (N, D) containing the input features.
        w: NumPy array of shape (D,) containing the model parameters.

    Returns:
        The gradient as a NumPy array of shape (D,).
    """
    n_samples = y.shape[0]
    return -(tx.T @ (y - tx @ w)) / n_samples


def _sigmoid(scores):
    """Compute sigmoid probabilities using a numerically stable formula.

    Args:
        scores: NumPy array containing real-valued model scores.

    Returns:
        A NumPy array with the same shape as ``scores`` and values in [0, 1].
    """
    probabilities = np.empty_like(scores, dtype=float)
    nonnegative = scores >= 0

    probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-scores[nonnegative]))
    exp_scores = np.exp(scores[~nonnegative])
    probabilities[~nonnegative] = exp_scores / (1.0 + exp_scores)
    return probabilities


def compute_logistic_gradient(y, tx, w):
    """Compute the gradient of the mean binary cross-entropy loss.

    Args:
        y: NumPy array of shape (N,) containing binary labels in {0, 1}.
        tx: NumPy array of shape (N, D) containing the input features.
        w: NumPy array of shape (D,) containing the model parameters.

    Returns:
        The gradient as a NumPy array of shape (D,).
    """
    probabilities = _sigmoid(tx @ w)
    return tx.T @ (probabilities - y) / y.shape[0]


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Fit a linear regression model using gradient descent.

    Args:
        y: NumPy array of shape (N,) containing the target values.
        tx: NumPy array of shape (N, D) containing the input features.
        initial_w: NumPy array of shape (D,) containing the initial weights.
        max_iters: Number of gradient descent updates to perform.
        gamma: Gradient descent step size.

    Returns:
        A tuple ``(w, loss)`` where ``w`` is the final weight vector with shape
        (D,) and ``loss`` is its mean squared error as a scalar.
    """
    w = np.asarray(initial_w, dtype=float).reshape(-1).copy()

    for _ in range(max_iters):
        gradient = compute_mse_gradient(y, tx, w)
        w = w - gamma * gradient

    last_w = w
    loss = compute_mse(y, tx, last_w)
    return last_w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Fit a linear regression model using stochastic gradient descent.

    Args:
        y: NumPy array of shape (N,) containing the target values.
        tx: NumPy array of shape (N, D) containing the input features.
        initial_w: NumPy array of shape (D,) containing the initial weights.
        max_iters: Number of stochastic gradient updates to perform.
        gamma: Stochastic gradient descent step size.

    Returns:
        A tuple ``(w, loss)`` where ``w`` is the final weight vector with shape
        (D,) and ``loss`` is its mean squared error on all samples as a scalar.
    """
    w = np.asarray(initial_w, dtype=float).reshape(-1).copy()
    n_samples = y.shape[0]

    for _ in range(max_iters):
        index = np.random.randint(n_samples)
        # Slices preserve the (1,) and (1, D) shapes required for batch size 1.
        y_batch = y[index : index + 1]
        tx_batch = tx[index : index + 1]
        gradient = compute_mse_gradient(y_batch, tx_batch, w)
        w = w - gamma * gradient

    last_w = w
    loss = compute_mse(y, tx, last_w)
    return last_w, loss


def least_squares(y, tx):
    """Fit least-squares regression using the normal equations.

    Args:
        y: NumPy array of shape (N,) containing the target values.
        tx: NumPy array of shape (N, D) containing the input features.

    Returns:
        A tuple ``(w, loss)`` where ``w`` is the optimal weight vector with
        shape (D,) and ``loss`` is its mean squared error as a scalar.
    """
    last_w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = compute_mse(y, tx, last_w)
    return last_w, loss


def ridge_regression(y, tx, lambda_):
    """Fit ridge regression using the regularized normal equations.

    Args:
        y: NumPy array of shape (N,) containing the target values.
        tx: NumPy array of shape (N, D) containing the input features.
        lambda_: Nonnegative regularization parameter.

    Returns:
        A tuple ``(w, loss)`` where ``w`` is the optimal weight vector with
        shape (D,) and ``loss`` is its unregularized mean squared error as a
        scalar.
    """
    n_samples = y.shape[0]
    identity = np.eye(tx.shape[1])
    system_matrix = tx.T @ tx + 2 * n_samples * lambda_ * identity
    last_w = np.linalg.solve(system_matrix, tx.T @ y)
    loss = compute_mse(y, tx, last_w)
    return last_w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Fit logistic regression using gradient descent.

    Args:
        y: NumPy array of shape (N,) containing binary labels in {0, 1}.
        tx: NumPy array of shape (N, D) containing the input features.
        initial_w: NumPy array of shape (D,) containing the initial weights.
        max_iters: Number of gradient descent updates to perform.
        gamma: Gradient descent step size.

    Returns:
        A tuple ``(w, loss)`` where ``w`` is the final weight vector with shape
        (D,) and ``loss`` is its mean binary cross-entropy as a scalar.
    """
    w = np.asarray(initial_w, dtype=float).reshape(-1).copy()

    for _ in range(max_iters):
        gradient = compute_logistic_gradient(y, tx, w)
        w = w - gamma * gradient

    last_w = w
    loss = compute_cross_entropy(y, tx, last_w)
    return last_w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Fit L2-regularized logistic regression using gradient descent.

    Args:
        y: NumPy array of shape (N,) containing binary labels in {0, 1}.
        tx: NumPy array of shape (N, D) containing the input features.
        lambda_: Nonnegative regularization parameter for lambda * ||w||^2.
        initial_w: NumPy array of shape (D,) containing the initial weights.
        max_iters: Number of gradient descent updates to perform.
        gamma: Gradient descent step size.

    Returns:
        A tuple ``(w, loss)`` where ``w`` is the final weight vector with shape
        (D,) and ``loss`` is its unregularized mean binary cross-entropy as a
        scalar.
    """
    w = np.asarray(initial_w, dtype=float).reshape(-1).copy()

    for _ in range(max_iters):
        gradient = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient

    # The project specification requires returning the loss without the penalty.
    last_w = w
    loss = compute_cross_entropy(y, tx, last_w)
    return last_w, loss
