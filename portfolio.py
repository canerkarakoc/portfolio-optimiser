from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, objective_functions, risk_models


def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """Compute daily returns from price data.

    Parameters
    ----------
    prices : pd.DataFrame
        Prices indexed by date with columns per asset.
    method : str
        "log" for log returns, otherwise simple returns.

    Returns
    -------
    pd.DataFrame
        Returns DataFrame aligned to input index (one row shorter).
    """
    prices = prices.sort_index().dropna(how="all")
    if method == "log":
        returns = np.log(prices / prices.shift(1))
    else:
        returns = prices.pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="all").dropna(axis=1, how="all")
    return returns


def compute_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute sample covariance matrix of returns (daily).

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns.

    Returns
    -------
    pd.DataFrame
        Covariance matrix.
    """
    return returns.cov()


def _weight_bounds(num_assets: int, allow_short: bool, max_weight: Optional[float]) -> Tuple[Tuple[float, float], ...]:
    if allow_short:
        lower = -1.0
    else:
        lower = 0.0
    upper = 1.0 if max_weight is None else min(1.0, float(max_weight))
    return tuple((lower, upper) for _ in range(num_assets))


def _annualize_perf(weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> Tuple[float, float, float]:
    expected_return = float(np.dot(weights, mean_returns))
    variance = float(np.dot(weights.T, np.dot(cov_matrix * 252.0, weights)))
    volatility = float(np.sqrt(max(variance, 0.0)))
    sharpe = expected_return / volatility if volatility > 0 else np.nan
    return expected_return, volatility, sharpe


def random_portfolios(
    num_portfolios: int,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float,
    allow_short: bool = False,
    max_weight: Optional[float] = None,
) -> pd.DataFrame:
    """Generate random portfolios for visualization of the frontier.

    Returns a DataFrame with columns [return, volatility, sharpe].
    """
    rng = np.random.default_rng(42)
    n = len(mean_returns)
    bounds = _weight_bounds(n, allow_short, max_weight)

    results = np.zeros((num_portfolios, 3))
    for i in range(num_portfolios):
        if allow_short:
            weights = rng.uniform(low=bounds[0][0], high=bounds[0][1], size=n)
            weights = weights / np.sum(np.abs(weights))
        else:
            weights = rng.random(n)
            weights /= weights.sum()
        # Enforce max weight per asset
        if max_weight is not None:
            weights = np.clip(weights, bounds[0][0], bounds[0][1])
            # renormalize long-only
            if not allow_short and weights.sum() > 0:
                weights /= weights.sum()
        er, vol, _ = _annualize_perf(weights, mean_returns, cov_matrix)
        sharpe = (er - risk_free_rate) / vol if vol > 0 else np.nan
        results[i] = [er, vol, sharpe]
    df = pd.DataFrame(results, columns=["return", "volatility", "sharpe"])  # type: ignore
    return df


def optimize_max_sharpe(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
    allow_short: bool = False,
    max_weight: Optional[float] = None,
) -> Tuple[Dict[str, float], Tuple[float, float, float]]:
    """Optimize portfolio for max Sharpe ratio.

    Returns weights dict and performance tuple (exp_return, volatility, sharpe).
    """
    mu = returns.mean() * 252.0
    S = risk_models.sample_cov(returns) * 252.0
    bounds = _weight_bounds(len(mu), allow_short, max_weight)

    ef = EfficientFrontier(mu, S, weight_bounds=bounds)
    if max_weight is not None:
        ef.add_objective(objective_functions.L2_reg, gamma=0.001)
    ef.max_sharpe(risk_free_rate=risk_free_rate)
    cleaned_weights = ef.clean_weights()
    perf = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
    return cleaned_weights, (float(perf[0]), float(perf[1]), float(perf[2]))


def optimize_min_volatility(
    returns: pd.DataFrame,
    allow_short: bool = False,
    max_weight: Optional[float] = None,
) -> Tuple[Dict[str, float], Tuple[float, float, float]]:
    """Optimize portfolio for minimum volatility.

    Returns weights dict and performance tuple (exp_return, volatility, sharpe).
    """
    mu = returns.mean() * 252.0
    S = risk_models.sample_cov(returns) * 252.0
    bounds = _weight_bounds(len(mu), allow_short, max_weight)

    ef = EfficientFrontier(mu, S, weight_bounds=bounds)
    if max_weight is not None:
        ef.add_objective(objective_functions.L2_reg, gamma=0.001)
    ef.min_volatility()
    cleaned_weights = ef.clean_weights()
    perf = ef.portfolio_performance(verbose=False)
    return cleaned_weights, (float(perf[0]), float(perf[1]), float(perf[2]))


