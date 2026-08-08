FAST DAILY DARVAS + MINERVINI SCAN

Keep your existing app.py in the repository. Add:
  fast_scan_batch.py
  send_scan_digest.py
  .github/workflows/daily-breakout-scan-fast.yml

The GitHub Actions workflow splits the S&P 500 into batches of 50 symbols,
adds BTC-USD and ETH-USD once, runs batches in parallel, saves small JSON
artifacts, and sends one combined digest email after all batches finish.

Fast scheduled mode skips historical breakout-probability replay. It still
calculates Darvas boxes, Minervini trend, volume dry-up, relative strength,
BREAKOUT WATCH / CONFIRMED BREAKOUT states, strategy score, and inexpensive
confirmed-breakout targets.

Required GitHub secrets:
  BREAKOUT_SMTP_HOST
  BREAKOUT_SMTP_PORT
  BREAKOUT_SMTP_USERNAME
  BREAKOUT_SMTP_PASSWORD
  BREAKOUT_SENDER
  BREAKOUT_RECIPIENT
  BREAKOUT_USE_SSL
