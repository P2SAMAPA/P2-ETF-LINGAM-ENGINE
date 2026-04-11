"""
Data Loading Module
===================
Handles loading and initial processing of ETF data from HuggingFace.
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
from typing import Optional, Tuple
import config


def load_etf_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Load ETF data from HuggingFace dataset.

    Args:
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)

    Returns:
        DataFrame with ETF prices and index column
    """
    # Load from HuggingFace
    ds = load_dataset(config.INPUT_DATASET)
    df = pd.DataFrame(ds['train'])

    # Parse the index column as datetime
    if '__index_level_0__' in df.columns:
        df['date'] = pd.to_datetime(df['__index_level_0__'])
        df = df.drop(columns=['__index_level_0__'])
    elif 'date' not in df.columns:
        raise ValueError("Date column not found in dataset")

    # Set date as index
    df = df.set_index('date')

    # Filter by date range if specified
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]

    # Sort by date
    df = df.sort_index()

    return df


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily returns from price data.

    Args:
        df: DataFrame with ETF prices

    Returns:
        DataFrame with daily returns
    """
    returns = df.pct_change().dropna(how='all')
    return returns


def get_universe_data(
    universe: str,
    include_macro: bool = True
) -> pd.DataFrame:
    """
    Get data for a specific universe (FI/Commodity or Equity).

    Args:
        universe: 'fi_commodity' or 'equity'
        include_macro: Whether to include macro variables

    Returns:
        DataFrame with universe ETF returns + macro variables
    """
    df = load_etf_data()
    returns = calculate_returns(df)

    if universe == 'fi_commodity':
        assets = config.FI_COMMODITY_ASSETS + [config.FI_COMMODITY_BENCHMARK]
    elif universe == 'equity':
        assets = config.EQUITY_ASSETS + [config.EQUITY_BENCHMARK]
    else:
        raise ValueError(f"Unknown universe: {universe}")

    # Select columns for this universe
    available_assets = [a for a in assets if a in returns.columns]
    columns = available_assets.copy()

    if include_macro:
        available_macro = [m for m in config.MACRO_VARIABLES if m in returns.columns]
        columns.extend(available_macro)

    return returns[columns].dropna()


def split_data(
    df: pd.DataFrame,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    test_ratio: float = config.TEST_RATIO
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.

    Args:
        df: DataFrame with returns
        train_ratio: Training set ratio (default 0.8)
        val_ratio: Validation set ratio (default 0.1)
        test_ratio: Test set ratio (default 0.1)

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df


def get_rolling_window_data(
    df: pd.DataFrame,
    window_size: int = config.ROLLING_WINDOW_DAYS
) -> pd.DataFrame:
    """
    Get the most recent rolling window of data.

    Args:
        df: DataFrame with returns
        window_size: Number of days for rolling window

    Returns:
        DataFrame with rolling window data
    """
    if len(df) <= window_size:
        return df
    return df.iloc[-window_size:]


def get_shrinking_windows(
    start_years: list = config.SHRINKING_WINDOW_YEARS,
    end_date: str = config.DATA_END_DATE
) -> list:
    """
    Generate list of shrinking window date ranges.

    Args:
        start_years: List of start years for windows
        end_date: End date for all windows (YYYY-MM-DD)

    Returns:
        List of (start_date, end_date) tuples
    """
    windows = []
    for year in start_years:
        start_date = f"{year}-01-01"
        windows.append((start_date, end_date))
    return windows
