"""
culculate relations between features and class.
"""

import numpy as np
from data_processing import standardize
from basic_fuctions import num_not_nan

def check_corelation(tx, ty, pos_extend, neg_extend):
    """Select columns by feature-label scores.

    Args:
        tx: Input feature matrix.
        ty: Labels in {-1, +1}.
        pos_extend: Upper score cutoff.
        neg_extend: Lower score cutoff.

    Returns:
        indice: Selected indices
        corelations: score for each column.
    """

    indeces = []
    corelations = []

    for i in range(tx.shape[1]):
        num_observed = num_not_nan(tx[:, i])
        if num_observed == 0:
            corelations.append(0)
            continue

        col = standardize(tx[:, i])     # Standardize observed entries.
        corelation = np.sum(col * ty) / num_observed    # Feature-label score.
        corelations.append(corelation)  # Keep this column's score.

        if corelation > pos_extend or corelation < neg_extend:
            indeces.append(i)
            # Keep scores outside the cutoffs.

    return indeces, corelations

def fit_pca(data, variance_threshold=0.9):
    """PCA to retain cumulative variance.

    Args:
        data: Input data array.
        variance_threshold: Required cumulative variance proportion.

    Returns:
        Projected data, components, eigenvalues, variance ratios, mean and std.
    """

    standard_data, mean, std = standardize(data, return_stats=True)
    sample_num = standard_data.shape[0]


    covariance_matrix = standard_data.T @ standard_data / (sample_num - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    eigenvalues = np.maximum(eigenvalues, 0)

    total_variance = np.sum(eigenvalues)


    explained_variance_ratio = eigenvalues / total_variance
    cumulative_variance = np.cumsum(explained_variance_ratio)
    component_num = np.searchsorted(cumulative_variance, variance_threshold) + 1
    components = eigenvectors[:, :component_num]
    pca_data = standard_data @ components

    return pca_data, components, eigenvalues, explained_variance_ratio, mean, std

def transform_pca(data, mean, std, components):
    """Project new rows with PCA.

    Args:
        data: Input data array.
        mean: Mean fitted on training rows.
        std: Standard deviation fitted on training rows.
        components: Retained PCA directions as columns.

    Returns:
        pca_data: The projected data.
    """
    standard_data = data.astype(float).copy()
    isnan_index = np.isnan(standard_data)
    standard_data = (standard_data - mean) / std
    standard_data[isnan_index] = 0
    pca_data = standard_data @ components

    return pca_data
