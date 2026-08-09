"""Merge batch JSON artifacts and send one daily breakout digest email."""
from __future__ import annotations

import json
import math
import os
import smtplib
import ssl
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any


def num(value: Any) -> float | None:
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def money(value: Any) -> str:
    x = num(value)
    if x is None:
        return "N/A"
    return f"${x:,.2f}" if abs(x) >= 1 else f"${x:,.4f}"


def pct(value: Any, digits: int = 2) -> str:
    x = num(value)
    return "N/A" if x is None else f"{x:.{digits}f}%"


def multiple(value: Any) -> str:
    x = num(value)
    return "N/A" if x is None else f"{x:.2f}x"


def email_config() -> dict[str, Any]:
    def env(name: str, default: str = "") -> str:
        return os.getenv(name, default)

    return {
        "smtp_host": env("BREAKOUT_SMTP_HOST", "smtp.gmail.com"),
        "smtp_port": int(env("BREAKOUT_SMTP_PORT", "465")),
        "smtp_username": env("BREAKOUT_SMTP_USERNAME"),
        "smtp_password": env("BREAKOUT_SMTP_PASSWORD"),
        "sender": env("BREAKOUT_SENDER") or env("BREAKOUT_SMTP_USERNAME"),
        "recipient": env("BREAKOUT_RECIPIENT"),
        "use_ssl": env("BREAKOUT_USE_SSL", "true").lower() in {"1", "true", "yes", "on"},
    }


def main() -> int:
    files = sorted(Path("scan_results").rglob("batch_*.json"))
    if not files:
        print("No batch result files found.")
        return 1

    alerts: list[dict[str, Any]] = []
    total_analyzed = 0
    total_errors = 0
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        total_analyzed += int(payload.get("analyzed", 0))
        total_errors += int(payload.get("error_count", 0))
        alerts.extend(payload.get("alerts", []))

    # De-duplicate, just in case a benchmark symbol was present in more than one batch.
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in alerts:
        deduped[(row.get("Ticker", ""), row.get("State", ""))] = row
    alerts = list(deduped.values())

    confirmed = sorted(
        [r for r in alerts if r.get("State") == "CONFIRMED BREAKOUT"],
        key=lambda r: r.get("Strategy Score", 0), reverse=True,
    )
    watches = sorted(
        [r for r in alerts if r.get("State") == "BREAKOUT WATCH"],
        key=lambda r: r.get("Strategy Score", 0), reverse=True,
    )

    scan_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    subject = f"Daily breakout scan: {len(confirmed)} confirmed / {len(watches)} watch — {scan_date}"
    lines = [
        f"Darvas + Minervini Daily S&P 500 + Index Proxies + BTC/ETH Scan — {scan_date}",
        "",
        f"Analyzed: {total_analyzed}",
        f"Errors: {total_errors}",
        f"Confirmed breakouts: {len(confirmed)}",
        f"Breakout watches: {len(watches)}",
        "",
    ]

    if confirmed:
        lines.extend(["CONFIRMED BREAKOUTS", "=" * 72])
        for r in confirmed:
            lines.append(
                f"{r['Ticker']} | Price {money(r.get('Price'))} | Score {r.get('Strategy Score', 0)}/100 | "
                f"Volume {multiple(r.get('Volume Multiple'))} | Breakout {money(r.get('Breakout Level'))}"
            )
            for target in r.get("Targets", [])[:4]:
                up = num(target.get("upside_from_price_pct"))
                up_text = "N/A" if up is None else f"{up:+.1f}%"
                lines.append(f"  - {target.get('name', 'Target')}: {money(target.get('price'))} ({up_text})")
            lines.append("")

    if watches:
        lines.extend(["BREAKOUT WATCH", "=" * 72])
        for r in watches:
            lines.append(
                f"{r['Ticker']} | Price {money(r.get('Price'))} | Score {r.get('Strategy Score', 0)}/100 | "
                f"Distance {pct(r.get('Distance to Breakout %'))} | Volume {multiple(r.get('Volume Multiple'))} | "
                f"Box Quality {num(r.get('Box Quality')) or 0:.1f}/100"
            )
            targets = r.get("Targets", [])
            if targets:
                nearest = targets[0]
                lines.append(
                    f"  - Potential post-breakout target: {money(nearest.get('price'))} "
                    f"({pct(nearest.get('upside_from_price_pct'), 1)} from current price)"
                )
                darvas = next((t for t in targets if t.get("name") == "Darvas target"), None)
                if darvas and darvas is not nearest:
                    lines.append(
                        f"  - Darvas measured target: {money(darvas.get('price'))} "
                        f"({pct(darvas.get('upside_from_price_pct'), 1)} from current price)"
                    )
        lines.append("")

    if not alerts:
        lines.append("No BREAKOUT WATCH or CONFIRMED BREAKOUT signals today.")
        lines.append("")

    lines.append("Fast daily mode skips the expensive historical breakout-probability replay.")
    lines.append("Educational signals only; not investment advice.")

    cfg = email_config()
    required = ["smtp_host", "smtp_port", "smtp_username", "smtp_password", "sender", "recipient"]
    if not all(cfg.get(k) for k in required):
        print("Email settings incomplete.")
        return 2

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg["sender"]
    msg["To"] = cfg["recipient"]
    msg.set_content("\n".join(lines))

    if cfg["use_ssl"]:
        with smtplib.SMTP_SSL(
            cfg["smtp_host"], cfg["smtp_port"], context=ssl.create_default_context(), timeout=30
        ) as server:
            server.login(cfg["smtp_username"], cfg["smtp_password"])
            server.send_message(msg)
    else:
        with smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"], timeout=30) as server:
            server.starttls(context=ssl.create_default_context())
            server.login(cfg["smtp_username"], cfg["smtp_password"])
            server.send_message(msg)

    print(f"Digest sent: analyzed={total_analyzed} errors={total_errors} confirmed={len(confirmed)} watch={len(watches)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

def earnings_text(r: dict) -> list[str]:
    out=[]
    if r.get("Upcoming ER"):
        d=r.get("Days to ER")
        s=f"  - Earnings: {r['Upcoming ER']}"
        if isinstance(d,int): s+=f" ({d} days)"
        if isinstance(d,int) and 0<=d<=7: s+="  **EARNINGS SOON**"
        out.append(s)
    hist=r.get("Earnings History") or []
    if hist:
        out.append(f"  - Last 4: {r.get('ER Beats',0)} beat / {r.get('ER Meets',0)} met / {r.get('ER Misses',0)} miss")
        for h in hist[:4]:
            def f(v):
                try: return f"{float(v):.2f}"
                except: return "N/A"
            try: sp=f"{float(h.get('Surprise %')):+.1f}%"
            except: sp="N/A"
            out.append(f"      {h.get('Date','')} | expected {f(h.get('Expected EPS'))} | actual {f(h.get('Actual EPS'))} | {sp} | {h.get('Result','')}")
    return out


