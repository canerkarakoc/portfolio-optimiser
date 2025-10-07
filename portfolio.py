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
    # Clip extreme daily moves to reduce the impact of bad ticks/splits
    returns = returns.clip(lower=-0.5, upper=0.5)
    # Ensure finite numeric values only, but avoid over-dropping data
    returns = returns.replace([np.inf, -np.inf], np.nan)
    # Drop columns (assets) that are mostly missing
    if not returns.empty:
        min_non_null = int(0.9 * len(returns))
        returns = returns.dropna(axis=1, thresh=min_non_null)
    # Forward/backward fill small gaps, then drop any remaining NaNs row-wise
    returns = returns.ffill().bfill().dropna(axis=0, how="any")
    if not returns.empty:
        returns = returns.astype(float)
    return returns


def compute_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute daily sample covariance matrix.

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


def _sanitize_moments(mu: pd.Series, S: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """Drop assets with non-finite or extreme mean/variance and ensure finite covariance.

    This protects the optimizer from exploding values due to data glitches.
    """
    mu = mu.replace([np.inf, -np.inf], np.nan)
    S = S.replace([np.inf, -np.inf], np.nan)
    mu = mu.dropna()
    # Align to intersection first
    inter = mu.index.intersection(S.index)
    S = S.loc[inter, inter]
    mu = mu.loc[inter]

    # Drop any rows/cols with NaNs in covariance
    S = S.dropna(axis=0, how="any").dropna(axis=1, how="any")
    common = mu.index.intersection(S.index)
    S = S.loc[common, common]
    mu = mu.loc[common]

    # Clip extreme covariances/variances to guard against bad ticks
    if len(S) > 0:
        S = S.astype(float)
        S = S.clip(lower=-5.0, upper=5.0)
        diag = pd.Series(np.diag(S), index=S.index)
        ok = diag[(diag >= 0) & (diag < 5.0)].index
        S = S.loc[ok, ok]
        mu = mu.loc[ok].astype(float)

    # Final guard: ensure all finite
    finite_mask = np.isfinite(S.values)
    if len(mu) == 0 or not finite_mask.all():
        # remove any remaining problematic rows/cols
        good_cols = S.columns[(~np.isnan(S.values).any(axis=0)) & (~np.isinf(S.values).any(axis=0))]
        S = S.loc[good_cols, good_cols]
        mu = mu.loc[S.index]
    return mu, S


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

    tickers = list(mean_returns.index)
    results = np.zeros((num_portfolios, 3), dtype=float)
    dominant: list[str] = []
    top_weights: list[str] = []
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
        # capture dominant asset and top-3 weights for hover
        dom_idx = int(np.argmax(np.abs(weights)))
        dominant.append(tickers[dom_idx])
        sorted_idx = np.argsort(-np.abs(weights))[:3]
        desc = ", ".join([f"{tickers[j]}: {weights[j]:.2f}" for j in sorted_idx])
        top_weights.append(desc)
    df = pd.DataFrame(results, columns=["return", "volatility", "sharpe"])  # type: ignore
    df["dominant"] = dominant
    df["top_weights"] = top_weights
    return df


def _fallback_equal_weights(
    returns: pd.DataFrame, risk_free_rate: float = 0.0
) -> Tuple[Dict[str, float], Tuple[float, float, float]]:
    """Fallback: equal-weight portfolio over available assets.

    Used when optimization cannot proceed due to data issues or <2 clean assets.
    """
    cols = list(returns.columns)
    if len(cols) == 0:
        raise ValueError("No assets available for fallback portfolio.")
    weights = {c: 1.0 / len(cols) for c in cols}
    mu = returns.mean() * 252.0
    S = returns.cov() * 252.0
    w_vec = np.array([weights[c] for c in cols])
    er = float(np.dot(w_vec, mu.loc[cols]))
    var = float(np.dot(w_vec.T, np.dot(S.loc[cols, cols], w_vec)))
    vol = float(np.sqrt(max(var, 0.0)))
    sharpe = (er - risk_free_rate) / vol if vol > 0 else np.nan
    return weights, (er, vol, sharpe)


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
    mu, S = _sanitize_moments(mu, S)
    if len(mu) < 2:
        # Fallback to equal weights on available returns
        return _fallback_equal_weights(returns, risk_free_rate)
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
    mu, S = _sanitize_moments(mu, S)
    if len(mu) < 2:
        return _fallback_equal_weights(returns)
    bounds = _weight_bounds(len(mu), allow_short, max_weight)

    ef = EfficientFrontier(mu, S, weight_bounds=bounds)
    if max_weight is not None:
        ef.add_objective(objective_functions.L2_reg, gamma=0.001)
    ef.min_volatility()
    cleaned_weights = ef.clean_weights()
    perf = ef.portfolio_performance(verbose=False)
    return cleaned_weights, (float(perf[0]), float(perf[1]), float(perf[2]))


