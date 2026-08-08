"""Unattended once-daily runner for the Darvas + Minervini S&P 500 + crypto scanner.

Place this file beside app.py in the repository. The GitHub Actions workflow runs
it once per day and supplies SMTP settings through repository secrets.
"""
from __future__ import annotations

import sys

import app as scanner


def main() -> int:
    try:
        result = scanner.run_daily_market_scan(
            settings=scanner.default_daily_settings(),
            send_email=True,
            force=True,
        )
    except Exception as exc:
        print(f"Daily scan failed: {exc}")
        return 1

    alerts = result.get("alerts", [])
    confirmed = sum(1 for r in alerts if r.get("State") == "CONFIRMED BREAKOUT")
    watches = sum(1 for r in alerts if r.get("State") == "BREAKOUT WATCH")
    print(
        f"Scan date={result.get('scan_date')} "
        f"analyzed={result.get('analyzed_count', 0)} "
        f"errors={result.get('error_count', 0)} "
        f"confirmed={confirmed} watch={watches} "
        f"email_sent={result.get('email_sent', False)}"
    )
    if result.get("email_error"):
        print(f"Email warning: {result['email_error']}")
        # Treat missing/failed email as a failed scheduled job when signals existed.
        if alerts:
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
