"""
preprocessing date before training.
"""

import numpy as np

DROP_COLUMNS = {
    "Id", "SEQNO", "_PSU", "IDATE", "IMONTH", "IDAY", "IYEAR", "FMONTH",
    "DISPCODE", "QSTVER", "QSTLANG", "MSCODE", "_STSTR", "_STRWT", "_RAWRAKE",
    "_WT2RAKE", "_CHISPNC", "_CRACE1", "_CPRACE", "_CLLCPWT", "_DUALUSE",
    "_DUALCOR", "_LLCPWT", "FLSHTMY2", "HIVTSTD3", "_MICHD", "CVDINFR4", "CVDCRHD4",
}

REPLACED_COLUMNS = {
    "WEIGHT2": "WTKG3", "HEIGHT3": "HTM4", "ALCDAY5": "DROCDY3_",
    "FRUITJU1": "FTJUDA1_", "FRUIT1": "FRUTDA1_", "FVBEANS": "BEANDAY_",
    "FVGREEN": "GRENDAY_", "FVORANG": "ORNGDAY_", "VEGETAB1": "VEGEDA1_",
    "EXEROFT1": "PAFREQ1_", "EXEROFT2": "PAFREQ2_", "STRENGTH": "STRFREQ_",
}

CONTINUOUS_COLUMNS = {
    "_AGE80", "HTIN4", "HTM4", "WTKG3", "_BMI5", "PHYSHLTH", "MENTHLTH",
    "POORHLTH", "JOINPAIN", "CHILDREN", "DROCDY3_", "_DRNKWEK", "FTJUDA1_",
    "FRUTDA1_", "BEANDAY_", "GRENDAY_", "ORNGDAY_", "VEGEDA1_", "_FRUTSUM",
    "_VEGESUM", "METVL11_", "METVL21_", "MAXVO2_", "FC60_", "PADUR1_",
    "PADUR2_", "PAFREQ1_", "PAFREQ2_", "_MINAC11", "_MINAC21", "STRFREQ_",
    "PAMIN11_", "PAMIN21_", "PA1MIN_", "PAVIG11_", "PAVIG21_", "PA1VIGM_",
}

CATEGORICAL_COLUMNS = {
    "_STATE", "SEX", "MARITAL", "RENTHOM1", "EMPLOY1", "SXORIENT", "TRNSGNDR",
    "EXRACT11", "EXRACT21", "CRGVREL1", "WHRTST10", "IMFVPLAC", "_PRACE1",
    "_MRACE1", "_RACE", "_RACEGR3", "_RACE_G1",
}

def standardize(data, return_stats=False):
    """Standardize values and fill missing entries with zero.

    Args:
        data: Input data array.
        return_stats: Whether to return fitted statistics.

    Returns:
        Standardized data; also fitted mean and std when requested.
    """
    data = data.astype(float).copy()

    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)

    mean = np.asarray(mean)
    std = np.asarray(std)
    mean = np.where(np.isnan(mean), 0, mean)
    std = np.where(np.isnan(std) | (std == 0), 1, std)

    isnan_index = np.isnan(data)
    
    data = (data - mean) / std
    data[isnan_index] = 0
    
    if return_stats:
        return data, mean, std

    return data

def stratified_holdout_indices(y, valid_ratio=0.2, seed=5920):
    """Split {1, -1} into training and validation indices.

    Args:
        y: Labels in {-1, +1}.
        valid_ratio: Validation proportion.
        seed: Random seed.

    Returns:
        Training and validation index arrays.
    """
    rng = np.random.default_rng(seed)
    train_parts = []
    valid_parts = []

    for label in (-1, 1):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        n_valid = int(np.ceil(valid_ratio * indices.size))
        valid_parts.append(indices[:n_valid])
        train_parts.append(indices[n_valid:])

    train_indices = np.concatenate(train_parts)
    valid_indices = np.concatenate(valid_parts)
    rng.shuffle(train_indices)
    rng.shuffle(valid_indices)

    return train_indices, valid_indices

def stratified_kfold_indices(y, n_splits=5, seed=52):
    """Generate training and validation folds.

    Args:
        y: Labels in {-1, +1}.
        n_splits: Number of folds.
        seed: Random seed.

    Returns:
        Yield one index pair per fold.
    """

    rng = np.random.default_rng(seed)
    negative = np.flatnonzero(y == -1)
    positive = np.flatnonzero(y == 1)
    rng.shuffle(negative)
    rng.shuffle(positive)

    negative_folds = np.array_split(negative, n_splits)
    positive_folds = np.array_split(positive, n_splits)
    all_indices = np.arange(y.size)

    for fold in range(n_splits):
        valid_indices = np.concatenate([negative_folds[fold], positive_folds[fold]])
        rng.shuffle(valid_indices)

        train_mask = np.ones(y.size, dtype=bool)
        train_mask[valid_indices] = False
        yield all_indices[train_mask], valid_indices

def expend_data(data):
    """Append column of ones for bias.

    Args:
        data: Input data array.

    Returns:
        The augmented data matrix.
    """
    return np.column_stack([data, np.ones(data.shape[0])])

def make_splits(y, seed=2026, valid_ratio=0.15, holdout_ratio=0.15):
    """Build the fixed train, validation and holdout split.

    Args:
        y: Labels in {-1, +1}.
        seed: Random seed.
        valid_ratio: Validation proportion.
        holdout_ratio: Holdout proportion.

    Returns:
        train_indices
        Three index arrays; default proportions are approximately 70/15/15.
    """

    train_valid_indices, holdout_indices = stratified_holdout_indices(y, holdout_ratio, seed)
    train_local, valid_local = stratified_holdout_indices(y[train_valid_indices], valid_ratio / (1 - holdout_ratio), seed + 1)
    train_indices = train_valid_indices[train_local]
    valid_indices = train_valid_indices[valid_local]

    return train_indices, valid_indices, holdout_indices

def clean_column(values, name):
    """Copy a column and apply its named response-code rules.

    Args:
        values: One input column.
        name: Label for printed results.

    Returns:
        A float64 column with missing or non-finite entries represented by NaN.
    """
    values = np.asarray(values, dtype=np.float64).copy()
    values[~np.isfinite(values)] = np.nan
    if name in {"PHYSHLTH", "MENTHLTH", "POORHLTH"}:
        values[values == 88] = 0
        values[np.isin(values, [77, 99])] = np.nan
    elif name == "CHILDREN":
        values[values == 88] = 0
        values[values == 99] = np.nan
    elif name == "JOINPAIN":
        values[np.isin(values, [77, 99])] = np.nan
    return values

def fit_preprocessor(x, names, max_categories=32, min_category_count=20, keep_all_features=False):
    """Fit column types, categories, filling, clipping and scaling on training rows.

    Args:
        x: Input feature matrix.
        names: Raw feature names excluding Id.
        max_categories: Maximum distinct integer values for inferred categories.
        min_category_count: Minimum training count for a retained category.
        keep_all_features: Whether to bypass the original column exclusions.

    Returns:
        schema: containing fitted rules and generated input names.
    """
    schema = {"mode": "typed_missing", "raw_names": list(names), "columns": [],
              "feature_names": ["bias"], "dropped": []}
    for index, name in enumerate(names):
        if not keep_all_features and name in DROP_COLUMNS:
            schema["dropped"].append({"name": name, "reason": "administrative_or_target"})
            continue
        if not keep_all_features and name in REPLACED_COLUMNS and REPLACED_COLUMNS[name] in names:
            schema["dropped"].append({"name": name, "reason": "calculated_replacement"})
            continue
        values = clean_column(x[:, index], name)
        missing = np.isnan(values)
        observed = values[~missing]
        if observed.size == 0:
            if keep_all_features:
                schema["columns"].append({"name": name, "index": index, "kind": "continuous",
                                          "missing_rate": 1.0, "unique_count": 0, "indicator": False,
                                          "lower": 0.0, "upper": 0.0, "median": 0.0,
                                          "mean": 0.0, "std": 1.0, "feature_names": [name]})
                schema["feature_names"].append(name)
            else:
                schema["dropped"].append({"name": name, "reason": "all_missing_in_train"})
            continue
        unique, counts = np.unique(observed, return_counts=True)
        indicator = np.any(missing)
        if unique.size == 1 and not indicator and not keep_all_features:
            schema["dropped"].append({"name": name, "reason": "constant_in_train"})
            continue
        integer_codes = np.all(unique == np.floor(unique))
        categorical = name in CATEGORICAL_COLUMNS or (
            name not in CONTINUOUS_COLUMNS and integer_codes and unique.size <= max_categories)
        column = {"name": name, "index": index, "missing_rate": float(np.mean(missing)),
                  "unique_count": int(unique.size), "indicator": bool(indicator)}
        if categorical:
            reference = float(unique[np.argmax(counts)])
            kept = unique[counts >= min_category_count]
            kept = np.unique(np.r_[kept, reference])
            column.update({"kind": "categorical", "reference": reference,
                           "categories": [float(value) for value in kept if value != reference],
                           "known_categories": [float(value) for value in kept]})
            column_names = [f"{name}=={value:g}" for value in column["categories"]]
            column_names.append(name + "==OTHER")
        else:
            lower, upper = np.quantile(observed, [0.005, 0.995])
            filled = np.clip(values, lower, upper)
            median = float(np.median(np.clip(observed, lower, upper)))
            filled[missing] = median
            mean, std = float(np.mean(filled)), float(np.std(filled))
            column.update({"kind": "continuous", "lower": float(lower), "upper": float(upper),
                           "median": median, "mean": mean, "std": std if std > 1e-8 else 1.0})
            column_names = [name]
        if indicator:
            column_names.append(name + "__missing")
        column["feature_names"] = column_names
        schema["columns"].append(column)
        schema["feature_names"].extend(column_names)
    return schema

def transform_preprocessor(x, schema, names=None):
    """Apply a fitted schema without recomputing statistics.

    Args:
        x: Input feature matrix.
        schema: Preprocessing rules fitted on training rows.
        names: Raw feature names excluding Id.

    Returns:
        result: matrix with first-column bias and unchanged row order.
    """
    result = np.empty((x.shape[0], len(schema["feature_names"])), dtype=np.float32)
    result[:, 0] = 1
    output_index = 1
    for column in schema["columns"]:
        values = clean_column(x[:, column["index"]], column["name"])
        missing = np.isnan(values)
        if column["kind"] == "categorical":
            values[missing] = column["reference"]
            for category in column["categories"]:
                result[:, output_index] = values == category
                output_index += 1
            result[:, output_index] = ~np.isin(values, column["known_categories"])
            output_index += 1
        else:
            values = np.clip(values, column["lower"], column["upper"])
            values[missing] = column["median"]
            result[:, output_index] = (values - column["mean"]) / column["std"]
            output_index += 1
        if column["indicator"]:
            result[:, output_index] = missing
            output_index += 1
    return result

def select_expanded_features(x, names):
    """Keep useful raw columns

    Args:
        x: Raw features from the training subset.
        names: Raw column names.

    Returns:
        Selected raw indices.
    """
    indices = []
    signatures = {}
    for index, name in enumerate(names):
        if name in DROP_COLUMNS:
            continue
        values = clean_column(x[:, index], name)
        observed = values[~np.isnan(values)]
        if observed.size == 0:
            continue
        if np.all(values == observed[0]):
            continue

        values[values == 0] = 0.0
        signature = hash(values.tobytes())
        duplicate = None
        for previous in signatures.get(signature, []):
            previous_values = clean_column(x[:, previous], names[previous])
            if np.array_equal(values, previous_values, equal_nan=True):
                duplicate = previous
                break
        if duplicate is not None:
            continue
        indices.append(index)
        signatures.setdefault(signature, []).append(index)

    return np.asarray(indices, dtype=int)

def nn_input_names(schema, mode):
    """List input names for representation.

    Args:
        schema: Preprocessing rules fitted on training rows.
        mode: numeric, onehot or onehot_missing.

    Returns:
        Names in the exact input order.
    """
    if mode == "numeric":
        return ["bias"] + [column["name"] for column in schema["columns"]]
    if mode == "onehot":
        return [name for name in schema["feature_names"] if not name.endswith("__missing")]
    if mode == "onehot_missing":
        return list(schema["feature_names"])

def transform_nn_input(x, schema, mode, numeric_mean=None, numeric_std=None):
    """Apply fitted NN preprocessing.

    Args:
        x: Input feature matrix.
        schema: Preprocessing rules fitted on training rows.
        mode: numeric, onehot or onehot_missing.
        numeric_mean: Training means for numeric inputs.
        numeric_std: Training standard deviations for numeric inputs.

    Returns:
        result: matrix including the first-column bias.
    """
    names = nn_input_names(schema, mode)
    if mode != "numeric":
        encoded = transform_preprocessor(x, schema)
        if mode == "onehot":
            keep = [i for i, name in enumerate(schema["feature_names"])
                    if not name.endswith("__missing")]
            encoded = np.ascontiguousarray(encoded[:, keep])
        return encoded

    result = np.empty((x.shape[0], len(names)), dtype=np.float32)
    result[:, 0] = 1
    for j, column in enumerate(schema["columns"]):
        values = clean_column(x[:, column["index"]], column["name"])
        missing = np.isnan(values)
        if column["kind"] == "continuous":
            values = np.clip(values, column["lower"], column["upper"])
            values[missing] = column["median"]
        else:
            values[missing] = column["reference"]
        result[:, j + 1] = (values - numeric_mean[j]) / numeric_std[j]
    return result

