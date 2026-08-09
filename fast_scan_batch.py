"""Fast batch scanner for GitHub Actions.

Scans one slice of the S&P 500 (default 50 symbols) using the existing
Darvas + Minervini core analysis in app.py. The expensive historical breakout
probability replay and all Streamlit/dashboard rendering are skipped.

Each job writes a compact JSON result for a later digest-email aggregation job.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import app as scanner


INDEX_PROXIES = ["SPY", "QQQ", "DIA", "IWM", "MDY", "RSP"]


def finite(value: Any, default: float | None = None) -> float | None:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def analyze_fast_symbol(
    ticker: str,
    raw_df,
    settings,
    spy_df,
    btc_df,
) -> dict[str, Any]:
    """Run only the core signal calculations needed for daily alerts."""
    asset_df = scanner.add_indicators(raw_df, settings)
    minimum_rows = max(221, settings.max_base_days + 2)
    if len(asset_df) < minimum_rows:
        return {"Ticker": ticker, "Error": f"Only {len(asset_df)} candles"}

    box = scanner.detect_current_box(asset_df, settings)
    trend = scanner.evaluate_trend_template(asset_df, settings)
    dry = scanner.evaluate_volume_dry_up(asset_df, settings)

    if ticker == "BTC-USD":
        rs = scanner.evaluate_relative_strength_vs_benchmark(
            ticker, asset_df, "BTC-USD", btc_df
        )
    elif ticker == "ETH-USD":
        rs = scanner.evaluate_relative_strength_vs_benchmark(
            ticker, asset_df, "BTC-USD", btc_df
        )
    else:
        rs = scanner.evaluate_relative_strength_vs_benchmark(
            ticker, asset_df, "SPY", spy_df
        )

    score = scanner.calculate_score(box, trend, dry, rs)
    state = box.get("state", "NO VALID BOX")
    latest_close = scanner.safe_float(asset_df["Close"].iloc[-1])
    breakout_level = scanner.safe_float(box.get("breakout_level"))
    distance_pct = (
        (breakout_level - latest_close) / breakout_level * 100
        if np.isfinite(breakout_level) and breakout_level != 0
        else np.nan
    )

    # Target calculation is inexpensive. For BREAKOUT WATCH it represents
    # the hypothetical post-breakout target stack if price clears the box.
    targets = {"targets": []}
    if state in {"BREAKOUT WATCH", "PRICE BREAKOUT / WEAK VOLUME", "CONFIRMED BREAKOUT"}:
        try:
            targets = scanner.calculate_breakout_targets(asset_df, box)
        except Exception:
            targets = {"targets": []}

    row = {
        "Ticker": ticker,
        "State": state,
        "Price": finite(latest_close),
        "Strategy Score": int(score["Total"]),
        "Box High": finite(box.get("box_high")),
        "Breakout Level": finite(breakout_level),
        "Distance to Breakout %": finite(distance_pct),
        "Volume Multiple": finite(box.get("volume_multiple")),
        "Box Quality": finite(box.get("quality_score")),
        "Targets": [
            {
                "name": t.get("name", t.get("type", "Target")),
                "price": finite(t.get("price")),
                "upside_from_price_pct": finite(t.get("upside_from_price_pct")),
            }
            for t in targets.get("targets", [])[:6]
        ],
        "Latest Date": asset_df.index[-1].strftime("%Y-%m-%d"),
    }

    if state in {"BREAKOUT WATCH", "CONFIRMED BREAKOUT"}:
        # Standalone squeeze score; does not alter Strategy Score.
        try:
            sq = scanner.fetch_short_squeeze_snapshot(ticker, asset_df)
            row["Short Squeeze Potential"] = finite(sq.get("score"))
        except Exception:
            row["Short Squeeze Potential"] = None

        # Earnings enrichment only for alert candidates.
        try:
            er = scanner.fetch_earnings_snapshot(ticker)
            row["Upcoming ER"] = er.get("next_earnings")
            row["Days to ER"] = er.get("days_to_earnings")
            row["Earnings History"] = er.get("history", [])
            row["ER Beats"] = er.get("beats", 0)
            row["ER Meets"] = er.get("meets", 0)
            row["ER Misses"] = er.get("misses", 0)
            row["Avg EPS Surprise %"] = finite(er.get("avg_surprise_pct"))
        except Exception:
            pass

    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-index", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--output-dir", default="scan_results")
    args = parser.parse_args()

    settings = scanner.default_daily_settings()
    sp500 = scanner.get_sp500_tickers()

    start = args.batch_index * args.batch_size
    end = start + args.batch_size
    batch = sp500[start:end]

    # Put index ETF proxies plus BTC and ETH in the final stock batch.
    # ETF proxies are used instead of raw index symbols because this scanner
    # relies on tradable volume for dry-up and breakout confirmation.
    total_stock_batches = math.ceil(len(sp500) / args.batch_size)
    if args.batch_index == total_stock_batches - 1:
        batch = list(dict.fromkeys(batch + INDEX_PROXIES + ["BTC-USD", "ETH-USD"]))

    if not batch:
        print(f"Batch {args.batch_index}: no symbols assigned")
        return 0

    # Benchmarks are downloaded with every batch so each matrix job is independent.
    download_symbols = list(dict.fromkeys(batch + ["SPY", "BTC-USD"]))
    print(
        f"Batch {args.batch_index}: scanning {len(batch)} symbols "
        f"({start + 1}-{min(end, len(sp500))} of {len(sp500)} S&P constituents)"
    )

    data = scanner.download_market_data_batch(
        tuple(download_symbols), settings.history_period, chunk_size=args.batch_size
    )
    if "SPY" not in data or "BTC-USD" not in data:
        print("Missing SPY and/or BTC benchmark data", file=sys.stderr)
        return 2

    spy_df = scanner.add_indicators(data["SPY"], settings)
    btc_df = scanner.add_indicators(data["BTC-USD"], settings)

    alerts: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    analyzed = 0

    for ticker in batch:
        raw = data.get(ticker)
        if raw is None or raw.empty:
            errors.append({"Ticker": ticker, "Error": "No market data"})
            continue
        try:
            row = analyze_fast_symbol(ticker, raw, settings, spy_df, btc_df)
            if row.get("Error"):
                errors.append({"Ticker": ticker, "Error": str(row["Error"])})
                continue
            analyzed += 1
            if row.get("State") in {"BREAKOUT WATCH", "CONFIRMED BREAKOUT"}:
                alerts.append(row)
        except Exception as exc:
            errors.append({"Ticker": ticker, "Error": str(exc)[:200]})

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "batch_index": args.batch_index,
        "batch_size": args.batch_size,
        "assigned": len(batch),
        "analyzed": analyzed,
        "error_count": len(errors),
        "alerts": alerts,
        "errors": errors,
    }
    out_file = out_dir / f"batch_{args.batch_index:02d}.json"
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    confirmed = sum(1 for x in alerts if x.get("State") == "CONFIRMED BREAKOUT")
    watch = sum(1 for x in alerts if x.get("State") == "BREAKOUT WATCH")
    print(
        f"Batch {args.batch_index} complete: analyzed={analyzed} errors={len(errors)} "
        f"confirmed={confirmed} watch={watch} output={out_file}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
