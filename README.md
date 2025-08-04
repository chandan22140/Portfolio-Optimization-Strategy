# Portfolio Optimization Strategy

This project implements a backtestable portfolio optimization and trading strategy in Python with the objective of **maximizing the Sharpe Ratio**.

## 📈 Objective

Maximize the Sharpe ratio using rebalancing-based trading strategies on historical stock prices.

## 📋 Constraints

- Stock weight ∈ [1%, 5%]
- Unused cash ≤ 5% of capital
- Gross exposure ≤ 100%
- Net exposure ≤ 10%

## 📅 Rebalancing Frequency

- Monthly or Weekly (default: Monthly)

## 📂 Project Structure

```

portfolio-optimizer/
├── data/                # CSV price data (e.g., Prices.csv)
├── src/                 # Main source code
├── README.md            # This file
├── requirements.txt     # Python dependencies

````

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/chandan22140/Portfolio-Optimization-Strategy
   cd portfolio-optimizer
````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Place the `Prices.csv` file in the `data/` folder.

4. Run the backtest:

   ```bash
   python src/trading_strategy.py
   ```

## 📊 Key Outputs

* Trades dataframe (`get_trades`)
* Position holdings and value (`get_positions`, `get_portfolio_value`)
* Cash usage and constraints monitoring
