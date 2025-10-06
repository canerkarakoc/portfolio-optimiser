# Portfolio Optimization Dashboard

A modular Streamlit app to fetch market data, compute returns/covariance, optimize portfolios (max Sharpe, min volatility), and visualize the efficient frontier, allocations, and cumulative performance. Provides CSV/Excel export.

## Features
- Input tickers, date range, risk-free rate, max allocation, and shorting toggle
- Cached data fetching with `yfinance`
- Portfolio optimization using `PyPortfolioOpt`
- Plotly charts for efficient frontier, allocations, cumulative returns
- CSV/Excel export buttons
- Minimal unit tests with `pytest`

## Getting Started

### 1) Create and activate a virtual environment (optional but recommended)
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Run the app
```bash
streamlit run app.py
```

Open the local URL shown in the terminal (typically `http://localhost:8501`).

## Usage
- Enter tickers like `AAPL, MSFT, GOOG, AMZN`
- Select date range (e.g., start 2018-01-01 to today)
- Set risk-free rate (annual, e.g., 0.02)
- Optionally set max allocation per asset and enable shorting
- Click "Run Optimization"

The app shows KPI cards (expected return, volatility, Sharpe), the efficient frontier with a highlighted optimal point, allocation pie + weights table, and an equal-weight cumulative return chart. Export price data to CSV/Excel with one click.

## Project Structure
```
portfolio-optimizer/
├─ app.py                      # Streamlit frontend
├─ data_fetch.py               # Fetch & cache market data
├─ portfolio.py                # Core finance & optimization functions
├─ plots.py                    # Plotly/matplotlib utilities
├─ utils.py                    # Validation and export helpers
├─ requirements.txt
├─ README.md
├─ tests/
│  ├─ test_portfolio.py        # unit tests for portfolio functions
├─ assets/                     # optional
├─ .gitignore
```

## Testing
```bash
pytest -q
```

## Notes
- Data sources via `yfinance` can have gaps or symbol naming quirks. The app drops assets with insufficient data.
- Optimizations use annualized moments (252 trading days).
- For production, consider storing data, adding authentication, and richer backtesting.

## Example Tickers
`AAPL, MSFT, GOOG, AMZN`


