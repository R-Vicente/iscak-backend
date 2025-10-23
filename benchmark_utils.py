"""
Benchmark utilities for imputation methods testing.
Includes MCAR introduction and metrics calculation.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr


def introduce_mcar(data: pd.DataFrame, missing_rate: float, seed: int) -> pd.DataFrame:
    """
    Introduce MCAR (Missing Completely At Random) missings.

    Args:
        data: Original DataFrame
        missing_rate: Proportion of missings (0 to 1)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with introduced missings
    """
    np.random.seed(seed)
    data_missing = data.copy()
    mask = np.random.rand(*data.shape) < missing_rate
    data_missing = data_missing.mask(mask)
    return data_missing


def calculate_metrics(data_original: pd.DataFrame,
                     data_imputed: pd.DataFrame,
                     missing_mask: pd.DataFrame) -> dict:
    """
    Calculate evaluation metrics for imputation.
    Returns both aggregated metrics and per-column metrics.

    Args:
        data_original: Original DataFrame (ground truth)
        data_imputed: DataFrame after imputation
        missing_mask: Boolean mask of values that were missing

    Returns:
        Dict with:
        - R2, Pearson, NRMSE, MAE (numeric) and Accuracy (categorical) - AGGREGATED
        - column_metrics: dict with individual metrics per column
    """
    numeric_cols = data_original.select_dtypes(include=[np.number]).columns
    categorical_cols = data_original.select_dtypes(exclude=[np.number]).columns

    metrics = {
        'R2': np.nan,
        'Pearson': np.nan,
        'NRMSE': np.nan,
        'MAE': np.nan,
        'Accuracy': np.nan,
        'column_metrics': {}
    }

    # ===== NUMERIC METRICS =====
    r2_scores = []
    pearson_scores = []
    nrmse_scores = []
    mae_scores = []

    for col in numeric_cols:
        col_mask = missing_mask[col]

        if not col_mask.any():
            continue

        orig_vals = data_original.loc[col_mask, col].values
        imp_vals = data_imputed.loc[col_mask, col].values

        # Remove remaining NaNs
        valid = ~(pd.isna(orig_vals) | pd.isna(imp_vals))
        orig_vals = orig_vals[valid]
        imp_vals = imp_vals[valid]

        if len(orig_vals) < 2:
            continue

        # Initialize metrics for this column
        col_metrics = {
            'type': 'numeric',
            'R2': np.nan,
            'NRMSE': np.nan,
            'MAE': np.nan,
            'Accuracy': np.nan
        }

        # R²
        if np.var(orig_vals) > 1e-10:
            col_r2 = r2_score(orig_vals, imp_vals)
            r2_scores.append(col_r2)
            col_metrics['R2'] = col_r2

        # Pearson
        if len(orig_vals) >= 3 and np.var(orig_vals) > 1e-10 and np.var(imp_vals) > 1e-10:
            pearson_scores.append(pearsonr(orig_vals, imp_vals)[0])

        # NRMSE
        rmse = np.sqrt(mean_squared_error(orig_vals, imp_vals))
        value_range = orig_vals.max() - orig_vals.min()
        if value_range > 1e-10:
            col_nrmse = rmse / value_range
            nrmse_scores.append(col_nrmse)
            col_metrics['NRMSE'] = col_nrmse

        # MAE
        col_mae = mean_absolute_error(orig_vals, imp_vals)
        mae_scores.append(col_mae)
        col_metrics['MAE'] = col_mae

        metrics['column_metrics'][col] = col_metrics

    # Aggregate numeric metrics
    if r2_scores:
        metrics['R2'] = np.mean(r2_scores)
    if pearson_scores:
        metrics['Pearson'] = np.mean(pearson_scores)
    if nrmse_scores:
        metrics['NRMSE'] = np.mean(nrmse_scores)
    if mae_scores:
        metrics['MAE'] = np.mean(mae_scores)

    # ===== CATEGORICAL METRICS =====
    accuracy_scores = []

    for col in categorical_cols:
        col_mask = missing_mask[col]

        if not col_mask.any():
            continue

        orig_vals = data_original.loc[col_mask, col]
        imp_vals = data_imputed.loc[col_mask, col]

        # Remove remaining NaNs
        valid = ~(orig_vals.isna() | imp_vals.isna())
        if valid.sum() < 1:
            continue

        orig_vals = orig_vals[valid]
        imp_vals = imp_vals[valid]

        col_accuracy = (orig_vals == imp_vals).mean()
        accuracy_scores.append(col_accuracy)

        # Store metrics for this column
        metrics['column_metrics'][col] = {
            'type': 'categorical',
            'R2': np.nan,
            'NRMSE': np.nan,
            'MAE': np.nan,
            'Accuracy': col_accuracy
        }

    # Aggregate categorical metrics
    if accuracy_scores:
        metrics['Accuracy'] = np.mean(accuracy_scores)

    return metrics
