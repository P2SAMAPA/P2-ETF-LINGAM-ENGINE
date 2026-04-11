"""
Data Preprocessing Module
=========================
Handles data cleaning, feature engineering, and normalization.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Args:
        df: DataFrame with potential missing values

    Returns:
        DataFrame with missing values handled
    """
    # Forward fill then backward fill for time series continuity
    df = df.fillna(method='ffill').fillna(method='bfill')

    # If still NaN, fill with 0 (for returns, this means no change)
    df = df.fillna(0)

    return df


def remove_outliers(
    df: pd.DataFrame,
    n_std: float = 5.0,
    method: str = 'zscore'
) -> pd.DataFrame:
    """
    Remove or clip outliers from the data.

    Args:
        df: DataFrame with returns
        n_std: Number of standard deviations for clipping
        method: 'zscore' or 'clip'

    Returns:
        DataFrame with outliers handled
    """
    if method == 'clip':
        # Clip values to n_std standard deviations
        for col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            df[col] = df[col].clip(mean - n_std * std, mean + n_std * std)
    elif method == 'zscore':
        # Replace outliers with column median
        for col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            z_scores = np.abs((df[col] - mean) / std)
            df[col] = np.where(z_scores > n_std, df[col].median(), df[col])

    return df


def normalize_features(
    df: pd.DataFrame,
    method: str = 'standard'
) -> Tuple[pd.DataFrame, object]:
    """
    Normalize features for causal discovery.

    Args:
        df: DataFrame with features
        method: 'standard' (z-score) or 'robust' (median/IQR)

    Returns:
        Tuple of (normalized_df, scaler_object)
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    normalized_df = pd.DataFrame(
        scaler.fit_transform(df),
        index=df.index,
        columns=df.columns
    )

    return normalized_df, scaler


def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical features for enhanced causal analysis.

    Args:
        df: DataFrame with price data

    Returns:
        DataFrame with additional technical features
    """
    features_df = df.copy()

    # Calculate rolling statistics for each ETF
    for col in df.columns:
        if col not in ['VIX', 'DXY', 'T10Y2Y', 'TBILL_3M', 'IG_SPREAD', 'HY_SPREAD']:
            # Rolling 20-day volatility
            features_df[f'{col}_vol_20d'] = df[col].rolling(20).std()

            # Rolling 20-day momentum
            features_df[f'{col}_mom_20d'] = df[col].pct_change(20)

            # Rolling 5-day mean
            features_df[f'{col}_ma5'] = df[col].rolling(5).mean()

    return features_df.dropna()


def prepare_causal_data(
    df: pd.DataFrame,
    assets: list,
    benchmark: str,
    include_macro: bool = True,
    normalize: bool = True
) -> pd.DataFrame:
    """
    Prepare data specifically for causal discovery.

    Args:
        df: DataFrame with returns
        assets: List of asset tickers
        benchmark: Benchmark ticker
        include_macro: Include macro variables
        normalize: Normalize the data

    Returns:
        Prepared DataFrame for causal analysis
    """
    # Select columns
    columns = [a for a in assets + [benchmark] if a in df.columns]

    if include_macro:
        macro_cols = [m for m in config.MACRO_VARIABLES if m in df.columns]
        columns.extend(macro_cols)

    causal_df = df[columns].copy()

    # Handle missing values
    causal_df = handle_missing_values(causal_df)

    # Remove outliers
    causal_df = remove_outliers(causal_df, n_std=5.0)

    # Normalize if requested
    if normalize:
        causal_df, _ = normalize_features(causal_df, method='standard')

    return causal_df


def calculate_metrics(
    returns: pd.Series,
    periods_per_year: int = 252
) -> dict:
    """
    Calculate performance metrics for a returns series.

    Args:
        returns: Series of returns
        periods_per_year: Number of periods per year (252 for daily)

    Returns:
        Dictionary of metrics
    """
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1

    # Volatility
    volatility = returns.std() * np.sqrt(periods_per_year)

    # Sharpe Ratio (assuming risk-free rate = 0)
    sharpe_ratio = annualized_return / volatility if volatility != 0 else 0

    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Win Rate
    win_rate = (returns > 0).mean()

    # Best/Worst Day
    best_day = returns.max()
    worst_day = returns.min()

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'best_day': best_day,
        'worst_day': worst_day,
    }
