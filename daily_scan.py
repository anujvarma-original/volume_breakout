from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import app


def configure_scan_state(scan_mode: str) -> str:
    """Point app.py at universe-specific state/history files."""
    mode = app.normalize_scan_mode(scan_mode)

    if mode == "nasdaq100":
        app.ACTIVE_SCAN_MODE = "nasdaq100"
        app.ACTIVE_SCAN_LABEL = "NASDAQ-100"
        app.ACTIVE_BENCHMARK = "QQQ"
        app.DAILY_SCAN_STATE_FILE = Path(".nasdaq100_daily_breakout_scan_state.json")
        app.SIGNAL_HISTORY_FILE = Path(".nasdaq100_breakout_signal_history.json")
    else:
        app.ACTIVE_SCAN_MODE = "sp500"
        app.ACTIVE_SCAN_LABEL = "S&P 500"
        app.ACTIVE_BENCHMARK = "SPY"
        app.DAILY_SCAN_STATE_FILE = Path(".sp500_daily_breakout_scan_state.json")
        app.SIGNAL_HISTORY_FILE = Path(".sp500_breakout_signal_history.json")

    return mode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Darvas + Minervini batch scanner without starting Streamlit."
    )
    parser.add_argument(
        "--scan",
        choices=["sp500", "nasdaq100"],
        required=True,
        help="Market universe to scan.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even if this universe already completed a scan for the current UTC date.",
    )
    parser.add_argument(
        "--no-email",
        action="store_true",
        help="Run the scan without sending the digest email.",
    )
    args = parser.parse_args()

    mode = configure_scan_state(args.scan)

    def progress(current: int, total: int, ticker: str) -> None:
        if current == 1 or current == total or current % 25 == 0:
            print(f"[{current}/{total}] {ticker}", flush=True)

    print(f"Starting {app.ACTIVE_SCAN_LABEL} Darvas scan...", flush=True)

    try:
        result = app.run_daily_market_scan(
            settings=app.default_daily_settings(),
            send_email=not args.no_email,
            force=args.force,
            progress_callback=progress,
            scan_mode=mode,
        )
    except Exception as exc:
        print(f"SCAN FAILED: {exc}", file=sys.stderr, flush=True)
        return 1

    if result.get("skipped"):
        print(
            f"Skipped {app.ACTIVE_SCAN_LABEL}: "
            f"{result.get('reason', 'already completed')}",
            flush=True,
        )
        return 0

    summary = {
        "scan_date": result.get("scan_date"),
        "universe": result.get("universe_label"),
        "benchmark": result.get("benchmark"),
        "symbols_scanned": result.get("universe_count"),
        "symbols_analyzed": result.get("analyzed_count"),
        "errors": result.get("error_count"),
        "alerts": len(result.get("alerts", [])),
        "email_sent": result.get("email_sent"),
        "email_error": result.get("email_error"),
        "signals_added": result.get("signals_added"),
        "signals_reviewed": result.get("signals_reviewed"),
    }

    print("\nScan complete:")
    print(json.dumps(summary, indent=2, default=str))

    if result.get("email_error"):
        print(f"EMAIL WARNING: {result['email_error']}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
