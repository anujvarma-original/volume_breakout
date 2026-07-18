# Darvas + Minervini Crypto Scanner

A Streamlit dashboard for current-state analysis of Bitcoin and Ethereum.

## Included

- BTC-USD and ETH-USD daily modes
- Configurable Darvas box detection
- Price and volume breakout confirmation
- Crypto-adapted Minervini trend template
- Dollar-volume dry-up and ATR contraction
- ETH/BTC relative strength
- Composite 100-point score
- Interactive candlestick, moving-average and volume charts
- CSV export
- No backtesting

## Run locally

```bash
python -m venv .venv
```

Activate the environment:

**Windows**

```bash
.venv\Scripts\activate
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

Install and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Put `app.py` and `requirements.txt` in a GitHub repository.
2. Open Streamlit Community Cloud.
3. Select the repository and choose `app.py` as the entry point.
4. Deploy. No API key is required for the initial yfinance version.

## Signal states

- **NO VALID BOX**: The selected window fails one or more box rules.
- **BUILDING A BOX**: A valid box exists, but price is not within 2% of the box high.
- **BREAKOUT WATCH**: A valid box exists and price is within 2% of the box high.
- **PRICE BREAKOUT / WEAK VOLUME**: Price broke out, but volume confirmation failed.
- **CONFIRMED BREAKOUT**: Box, price breakout and volume confirmation all pass.

## Important implementation detail

The box is calculated from completed candles ending one candle before the latest
available candle. The latest candle is then tested independently as the possible
breakout candle. This prevents the breakout candle itself from raising the box high.

## Data caveat

Yahoo Finance aggregates crypto pricing and volume. It is suitable for a simple
public-data scanner, but exchange-specific results may differ. Crypto trades
continuously, and the current UTC daily candle can be incomplete.
