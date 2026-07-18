# Darvas + Minervini Market Scanner

A Streamlit dashboard for current-state Darvas-box and Minervini-style analysis.

## Markets

- Bitcoin and Ethereum
- S&P 500 via SPY or ^GSPC
- Nasdaq-100 via QQQ or ^NDX
- On-demand S&P 500 constituent scanning
- On-demand Nasdaq-100 constituent scanning

## Included

- Configurable Darvas box detection
- Price and volume breakout confirmation
- 50-, 150- and 200-day trend template
- Rising 200-day moving-average test
- Distance from the 365-day high
- Dollar-volume dry-up and ATR contraction
- Relative strength versus BTC, SPY or QQQ
- Composite score out of 100
- Interactive charts and CSV export
- No backtesting

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Scanner notes

Choose **S&P 500** or **Nasdaq-100** in the sidebar, then open the **Market Scanner** tab. The default scan limit is 100 symbols. Increase it to scan the full universe. Public market-data services can occasionally rate-limit large scans.

The Nasdaq market mode uses the Nasdaq-100 rather than all Nasdaq-listed securities.

## Data caveats

Constituent lists are loaded from public Wikipedia tables. Prices are downloaded through yfinance. The latest daily candle can be incomplete while the relevant market is open; crypto candles can be incomplete before the UTC day closes.
