"""
Standard imputation methods (kNN, MICE, MissForest) with sklearn
Adapted from notebook with 3 strategies: numeric, mixed, categorical
"""

import pandas as pd
import numpy as np
# Enable experimental IterativeImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def impute_knn_numeric(data_missing: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """KNN with standardization for pure numeric data."""
    # Calculate mean and std from observed values
    means = data_missing.mean()
    stds = data_missing.std()

    # Standardize (keeps NaN)
    data_scaled = (data_missing - means) / stds

    # Impute
    imputer = KNNImputer(n_neighbors=k)
    data_imputed_scaled = pd.DataFrame(
        imputer.fit_transform(data_scaled),
        columns=data_scaled.columns,
        index=data_scaled.index
    )

    # Inverse transform
    data_imputed = (data_imputed_scaled * stds) + means

    return data_imputed


def impute_knn_mixed(data_missing: pd.DataFrame, k: int = 5,
                     numeric_cols: list = None, categorical_cols: list = None) -> pd.DataFrame:
    """
    KNN for mixed data (numeric + categorical).
    Strategy: Label encode → One-hot → Standardize numerics → KNN → Revert
    """
    if numeric_cols is None or categorical_cols is None:
        # Auto-detect
        numeric_cols = data_missing.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data_missing.select_dtypes(include=['object', 'category']).columns.tolist()

    # Label encode ALL categoricals
    data_encoded = data_missing.copy()
    label_mappings = {}

    for col in categorical_cols:
        unique_vals = data_encoded[col].dropna().unique()
        mapping = {val: idx for idx, val in enumerate(sorted(unique_vals))}
        reverse_mapping = {idx: val for val, idx in mapping.items()}
        label_mappings[col] = mapping
        label_mappings[f'{col}_reverse'] = reverse_mapping
        data_encoded[col] = data_encoded[col].map(mapping)

    # One-hot encode categoricals
    encoded_dfs = []
    for col in categorical_cols:
        dummies = pd.get_dummies(data_encoded[col], prefix=col, drop_first=False, dtype=float)
        # Propagate NaN
        mask = data_encoded[col].isna()
        dummies.loc[mask, :] = np.nan
        encoded_dfs.append(dummies)

    # Combine numerics + dummies
    data_for_knn = pd.concat(
        [data_encoded[numeric_cols]] + encoded_dfs,
        axis=1
    )

    # Standardize ONLY numerics
    means = data_for_knn[numeric_cols].mean()
    stds = data_for_knn[numeric_cols].std().replace(0, 1)
    data_for_knn[numeric_cols] = (data_for_knn[numeric_cols] - means) / stds

    # Impute
    imputer = KNNImputer(n_neighbors=k, weights='distance')
    data_imputed_array = imputer.fit_transform(data_for_knn)
    data_imputed_knn = pd.DataFrame(
        data_imputed_array,
        columns=data_for_knn.columns,
        index=data_for_knn.index
    )

    # Inverse transform numerics
    data_imputed_knn[numeric_cols] = (data_imputed_knn[numeric_cols] * stds) + means

    # Build result
    result = pd.DataFrame(index=data_missing.index)

    # Copy numerics
    result[numeric_cols] = data_imputed_knn[numeric_cols]

    # Revert categoricals to ORIGINAL values
    for col in categorical_cols:
        dummy_cols = [c for c in data_imputed_knn.columns if c.startswith(f"{col}_")]
        if len(dummy_cols) > 0:
            # Index of dummy with highest value
            dummy_values = data_imputed_knn[dummy_cols].values
            max_indices = dummy_values.argmax(axis=1)

            # Revert to original values
            reverse_map = label_mappings[f'{col}_reverse']
            result[col] = [reverse_map[idx] for idx in max_indices]

    # Ensure correct order
    result = result[data_missing.columns]

    return result


def impute_knn_categorical(data_missing: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """
    KNN for pure categorical data.
    Strategy: Label encode → One-hot → KNN (no standardization) → Revert
    """
    all_cols = data_missing.columns.tolist()

    # Label encode ALL features
    data_encoded = data_missing.copy()
    label_mappings = {}

    for col in all_cols:
        unique_vals = data_encoded[col].dropna().unique()
        mapping = {val: idx for idx, val in enumerate(sorted(unique_vals))}
        reverse_mapping = {idx: val for val, idx in mapping.items()}
        label_mappings[col] = mapping
        label_mappings[f'{col}_reverse'] = reverse_mapping
        data_encoded[col] = data_encoded[col].map(mapping)

    # One-hot encode ALL
    encoded_dfs = []
    for col in all_cols:
        dummies = pd.get_dummies(data_encoded[col], prefix=col, drop_first=False, dtype=float)
        mask = data_encoded[col].isna()
        dummies.loc[mask, :] = np.nan
        encoded_dfs.append(dummies)

    # Combine all dummies
    data_for_knn = pd.concat(encoded_dfs, axis=1)

    # Impute (NO standardization - they are binary)
    imputer = KNNImputer(n_neighbors=k, weights='distance')
    data_imputed_array = imputer.fit_transform(data_for_knn)
    data_imputed_knn = pd.DataFrame(
        data_imputed_array,
        columns=data_for_knn.columns,
        index=data_for_knn.index
    )

    # Revert categoricals
    result = pd.DataFrame(index=data_missing.index)

    for col in all_cols:
        dummy_cols = [c for c in data_imputed_knn.columns if c.startswith(f"{col}_")]
        if len(dummy_cols) > 0:
            # Choose category with highest value
            dummy_values = data_imputed_knn[dummy_cols].values
            max_indices = dummy_values.argmax(axis=1)
            reverse_map = label_mappings[f'{col}_reverse']
            result[col] = [reverse_map[idx] for idx in max_indices]

    return result


def impute_knn(data_missing: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """KNN imputation with automatic type detection."""
    numeric_cols = data_missing.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data_missing.select_dtypes(include=['object', 'category']).columns.tolist()

    if len(categorical_cols) == 0:
        # Pure numeric
        return impute_knn_numeric(data_missing, k)
    elif len(numeric_cols) == 0:
        # Pure categorical
        return impute_knn_categorical(data_missing, k)
    else:
        # Mixed
        return impute_knn_mixed(data_missing, k, numeric_cols, categorical_cols)


def impute_mice_numeric(data_missing: pd.DataFrame, max_iter: int = 10,
                        random_state: int = 42) -> pd.DataFrame:
    """MICE for pure numeric data."""
    imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
    data_imputed = pd.DataFrame(
        imputer.fit_transform(data_missing),
        columns=data_missing.columns,
        index=data_missing.index
    )
    return data_imputed


def impute_mice_with_categorical(data_missing: pd.DataFrame, max_iter: int = 10,
                                  random_state: int = 42) -> pd.DataFrame:
    """MICE with categorical columns - BayesianRidge."""
    data_encoded = data_missing.copy()
    label_mappings = {}
    categorical_cols = data_missing.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in categorical_cols:
        unique_vals = data_encoded[col].dropna().unique()
        mapping = {val: idx for idx, val in enumerate(sorted(unique_vals))}
        reverse_mapping = {idx: val for val, idx in mapping.items()}
        label_mappings[col] = mapping
        label_mappings[f'{col}_reverse'] = reverse_mapping
        data_encoded[col] = data_encoded[col].map(mapping)

    imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
    data_imputed_array = imputer.fit_transform(data_encoded)
    data_imputed = pd.DataFrame(
        data_imputed_array,
        columns=data_encoded.columns,
        index=data_encoded.index
    )

    for col in categorical_cols:
        n_classes = len(label_mappings[col])
        data_imputed[col] = data_imputed[col].round().clip(0, n_classes - 1).astype(int)
        reverse_map = label_mappings[f'{col}_reverse']
        data_imputed[col] = data_imputed[col].map(reverse_map)

    return data_imputed


def impute_mice(data_missing: pd.DataFrame, max_iter: int = 10,
                random_state: int = 42) -> pd.DataFrame:
    """MICE imputation with automatic type detection."""
    categorical_cols = data_missing.select_dtypes(include=['object', 'category']).columns.tolist()

    if len(categorical_cols) == 0:
        # Pure numeric
        return impute_mice_numeric(data_missing, max_iter, random_state)
    else:
        # With categoricals
        return impute_mice_with_categorical(data_missing, max_iter, random_state)


def impute_missforest_numeric(data_missing: pd.DataFrame, n_estimators: int = 10,
                               max_iter: int = 10, random_state: int = 42) -> pd.DataFrame:
    """MissForest for pure numeric data."""
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=n_estimators, random_state=random_state),
        max_iter=max_iter,
        random_state=random_state
    )
    data_imputed = pd.DataFrame(
        imputer.fit_transform(data_missing),
        columns=data_missing.columns,
        index=data_missing.index
    )
    return data_imputed


def impute_missforest_with_categorical(data_missing: pd.DataFrame, n_estimators: int = 10,
                                        max_iter: int = 10, random_state: int = 42) -> pd.DataFrame:
    """MissForest with categorical columns."""
    data_encoded = data_missing.copy()
    label_mappings = {}
    categorical_cols = data_missing.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = data_missing.select_dtypes(include=[np.number]).columns.tolist()

    for col in categorical_cols:
        unique_vals = data_encoded[col].dropna().unique()
        mapping = {val: idx for idx, val in enumerate(sorted(unique_vals))}
        reverse_mapping = {idx: val for val, idx in mapping.items()}
        label_mappings[col] = mapping
        label_mappings[f'{col}_reverse'] = reverse_mapping
        data_encoded[col] = data_encoded[col].map(mapping)

    # Use RandomForestClassifier for pure categorical, Regressor for mixed
    if len(numeric_cols) == 0:
        estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=10
        )
        max_iter = 5  # Reduce for categorical
    else:
        estimator = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state
        )

    imputer = IterativeImputer(
        estimator=estimator,
        max_iter=max_iter,
        sample_posterior=False,
        random_state=random_state
    )

    data_imputed_array = imputer.fit_transform(data_encoded)
    data_imputed = pd.DataFrame(
        data_imputed_array,
        columns=data_encoded.columns,
        index=data_encoded.index
    )

    for col in categorical_cols:
        n_classes = len(label_mappings[col])
        data_imputed[col] = data_imputed[col].round().clip(0, n_classes - 1).astype(int)
        reverse_map = label_mappings[f'{col}_reverse']
        data_imputed[col] = data_imputed[col].map(reverse_map)

    return data_imputed


def impute_missforest(data_missing: pd.DataFrame, n_estimators: int = 10,
                      max_iter: int = 10, random_state: int = 42) -> pd.DataFrame:
    """MissForest imputation with automatic type detection."""
    categorical_cols = data_missing.select_dtypes(include=['object', 'category']).columns.tolist()

    if len(categorical_cols) == 0:
        # Pure numeric
        return impute_missforest_numeric(data_missing, n_estimators, max_iter, random_state)
    else:
        # With categoricals
        return impute_missforest_with_categorical(data_missing, n_estimators, max_iter, random_state)
