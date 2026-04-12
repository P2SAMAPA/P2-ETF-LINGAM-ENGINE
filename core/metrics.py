"""
Metrics Module
==============
Trading performance metrics calculation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def calculate_total_return(returns: pd.Series) -> float:
    """
    Calculate total return from a series of returns.

    Args:
        returns: Series of period returns

    Returns:
        Total return as decimal
    """
    return float((1 + returns).prod() - 1)


def calculate_annualized_return(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized return.

    Args:
        returns: Series of period returns
        periods_per_year: Number of periods in a year

    Returns:
        Annualized return as decimal
    """
    total = calculate_total_return(returns)
    n_periods = len(returns)
    years = n_periods / periods_per_year

    if years <= 0:
        return 0.0

    return float((1 + total) ** (1 / years) - 1)


def calculate_sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0
) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: Series of period returns
        periods_per_year: Number of periods in a year
        risk_free_rate: Annual risk-free rate

    Returns:
        Sharpe ratio
    """
    if returns.std() == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    return float(
        np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()
    )


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown.

    Args:
        returns: Series of period returns

    Returns:
        Maximum drawdown as negative decimal
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())


def calculate_win_rate(returns: pd.Series) -> float:
    """
    Calculate win rate (percentage of positive periods).

    Args:
        returns: Series of period returns

    Returns:
        Win rate as decimal (0-1)
    """
    return float((returns > 0).mean())


def calculate_best_day(returns: pd.Series) -> float:
    """
    Calculate best single period return.

    Args:
        returns: Series of period returns

    Returns:
        Best period return as decimal
    """
    return float(returns.max())


def calculate_worst_day(returns: pd.Series) -> float:
    """
    Calculate worst single period return.

    Args:
        returns: Series of period returns

    Returns:
        Worst period return as decimal
    """
    return float(returns.min())


def calculate_volatility(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized volatility.

    Args:
        returns: Series of period returns
        periods_per_year: Number of periods in a year

    Returns:
        Annualized volatility
    """
    return float(returns.std() * np.sqrt(periods_per_year))


def calculate_sortino_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    target_return: float = 0.0
) -> float:
    """
    Calculate Sortino ratio (downside risk-adjusted return).

    Args:
        returns: Series of period returns
        periods_per_year: Number of periods in a year
        risk_free_rate: Annual risk-free rate
        target_return: Target return threshold

    Returns:
        Sortino ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < target_return]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0

    downside_std = downside_returns.std() * np.sqrt(periods_per_year)
    annual_return = calculate_annualized_return(returns, periods_per_year)

    return float((annual_return - risk_free_rate) / downside_std)


def calculate_calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar ratio (return / max drawdown).

    Args:
        returns: Series of period returns
        periods_per_year: Number of periods in a year

    Returns:
        Calmar ratio
    """
    annual_return = calculate_annualized_return(returns, periods_per_year)
    max_dd = abs(calculate_max_drawdown(returns))

    if max_dd == 0:
        return 0.0

    return float(annual_return / max_dd)


def calculate_all_metrics(
    returns: pd.Series,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """
    Calculate all standard performance metrics.

    Args:
        returns: Series of period returns
        periods_per_year: Number of periods in a year

    Returns:
        Dictionary of all metrics
    """
    return {
        'total_return': calculate_total_return(returns),
        'annualized_return': calculate_annualized_return(returns, periods_per_year),
        'sharpe_ratio': calculate_sharpe_ratio(returns, periods_per_year),
        'max_drawdown': calculate_max_drawdown(returns),
        'win_rate': calculate_win_rate(returns),
        'best_day': calculate_best_day(returns),
        'worst_day': calculate_worst_day(returns),
        'volatility': calculate_volatility(returns, periods_per_year),
        'sortino_ratio': calculate_sortino_ratio(returns, periods_per_year),
        'calmar_ratio': calculate_calmar_ratio(returns, periods_per_year),
    }


def calculate_consensus_score(
    metrics: Dict[str, float],
    weights: Dict[str, float] = None
) -> float:
    """
    Calculate weighted consensus score for shrinking window selection.

    Args:
        metrics: Dictionary with annualized_return, sharpe_ratio, max_drawdown
        weights: Dictionary with weights for each metric

    Returns:
        Consensus score (0-1)
    """
    if weights is None:
        weights = config.CONSENSUS_WEIGHTS

    # Check for negative returns - zero weight if negative
    if metrics.get('annualized_return', 0) <= 0:
        return 0.0

    score = 0.0
    total_weight = 0.0

    # Annualized Return (60%) - higher is better
    ret_score = min(max(metrics['annualized_return'], 0), 1)  # Normalize to 0-1
    score += ret_score * weights.get('annualized_return', 0.6)
    total_weight += weights.get('annualized_return', 0.6)

    # Sharpe Ratio (20%) - higher is better
    sharpe = metrics.get('sharpe_ratio', 0)
    sharpe_score = min(max(sharpe, -2), 4) / 4  # Normalize -2 to 4 -> 0 to 1
    score += sharpe_score * weights.get('sharpe_ratio', 0.2)
    total_weight += weights.get('sharpe_ratio', 0.2)

    # Max Drawdown (20%) - lower is better, invert
    mdd = metrics.get('max_drawdown', 0)
    mdd_score = 1 - min(abs(mdd), 0.5) / 0.5  # Normalize 0 to -0.5 -> 1 to 0
    score += mdd_score * weights.get('max_drawdown', 0.2)
    total_weight += weights.get('max_drawdown', 0.2)

    # Normalize by total weight used
    if total_weight > 0:
        score = score / total_weight

    return float(score)
