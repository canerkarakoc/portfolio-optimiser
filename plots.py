from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_efficient_frontier(sampled: pd.DataFrame, highlight: Optional[Tuple[float, float]] = None, color_by_dominant: bool = False):
    """Plot sampled portfolios and optionally a highlighted optimal point.

    Parameters
    ----------
    sampled : pd.DataFrame
        Columns: return, volatility, sharpe, dominant (optional)
    highlight : Optional[Tuple[float, float]]
        (return, volatility) to highlight.
    color_by_dominant : bool
        If True, color by dominant asset; otherwise color by Sharpe ratio.
    """
    if color_by_dominant and "dominant" in sampled.columns:
        color_col = "dominant"
        fig = px.scatter(
            sampled,
            x="volatility",
            y="return",
            color=color_col,
            labels={"volatility": "Volatility", "return": "Return", "dominant": "Dominant Asset"},
            title="Efficient Frontier (Random Sampling)",
        )
    else:
        fig = px.scatter(
            sampled,
            x="volatility",
            y="return",
            color="sharpe",
            color_continuous_scale="Viridis",
            labels={"volatility": "Volatility", "return": "Return", "sharpe": "Sharpe"},
            title="Efficient Frontier (Random Sampling)",
        )
    if highlight is not None:
        er, vol = highlight
        fig.add_trace(
            go.Scatter(
                x=[vol],
                y=[er],
                mode="markers",
                marker=dict(size=12, color="red"),
                name="Optimal",
            )
        )
    fig.update_layout(legend=dict(orientation="h"))
    return fig


def plot_allocation_pie(weights: pd.Series):
    weights = weights[weights.abs() > 1e-6]
    fig = px.pie(weights, values=weights.values, names=weights.index, title="Portfolio Allocation")
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def plot_cumulative_returns(cumulative: pd.Series):
    fig = px.line(cumulative, title="Cumulative Returns")
    fig.update_yaxes(tickformat=",.0%")
    return fig


