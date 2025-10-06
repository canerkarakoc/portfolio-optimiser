import numpy as np
import pandas as pd

from portfolio import compute_covariance, compute_returns, optimize_max_sharpe


def test_portfolio_performance_and_weights_sum_to_one():
    # Create synthetic price series for 3 assets
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=252)
    prices = pd.DataFrame(
        np.cumprod(1 + rng.normal(0.0005, 0.01, size=(252, 3))), index=dates, columns=["A", "B", "C"]
    )

    returns = compute_returns(prices)
    cov = compute_covariance(returns)

    # sanity
    assert returns.shape[0] == 251
    assert cov.shape == (3, 3)

    weights, perf = optimize_max_sharpe(returns, risk_free_rate=0.0)

    wsum = sum(weights.values())
    assert np.isclose(wsum, 1.0, atol=1e-6)
    exp_ret, vol, sharpe = perf
    assert vol >= 0
    assert isinstance(sharpe, float)


