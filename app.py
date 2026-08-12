from __future__ import annotations

# BUILD: DAILY S&P 500 + CRYPTO BREAKOUT SCANNER V3

from dataclasses import dataclass
from typing import Any
import json
import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="Darvas + Minervini Volume Breakout Scanner",
    page_icon="📦",
    layout="wide",
)


ALERT_STATE_FILE = Path(".breakout_alert_state.json")

# The same Streamlit deployment can be driven by URL query parameters:
#   ?scan=sp500
#   ?scan=nasdaq100
# Add &autorun=1 when an actual Streamlit browser session should start the
# selected daily scan automatically. Daily state/history are kept separate.
def _query_value(name: str, default: str = "") -> str:
    try:
        value = st.query_params.get(name, default)
        if isinstance(value, list):
            value = value[-1] if value else default
        return str(value or default).strip()
    except Exception:
        return default


def normalize_scan_mode(value: str) -> str:
    value = str(value or "").strip().lower().replace("-", "").replace("_", "")
    return "nasdaq100" if value in {"nasdaq", "nasdaq100", "ndx", "qqq"} else "sp500"


ACTIVE_SCAN_MODE = normalize_scan_mode(_query_value("scan", "sp500"))
ACTIVE_SCAN_LABEL = "NASDAQ-100" if ACTIVE_SCAN_MODE == "nasdaq100" else "S&P 500"
ACTIVE_BENCHMARK = "QQQ" if ACTIVE_SCAN_MODE == "nasdaq100" else "SPY"
AUTO_RUN_DAILY_SCAN = _query_value("autorun", "0").lower() in {"1", "true", "yes", "on"}

DAILY_SCAN_STATE_FILE = Path(
    ".nasdaq100_daily_breakout_scan_state.json"
    if ACTIVE_SCAN_MODE == "nasdaq100"
    else ".sp500_daily_breakout_scan_state.json"
)
SIGNAL_HISTORY_FILE = Path(
    ".nasdaq100_breakout_signal_history.json"
    if ACTIVE_SCAN_MODE == "nasdaq100"
    else ".sp500_breakout_signal_history.json"
)

DEFAULT_TICKERS = "BTC-USD, ETH-USD, SPY, QQQ, NVDA, AAPL"

def parse_tickers(raw: str) -> list[str]:
    """Accept comma-, whitespace-, or newline-separated Yahoo Finance symbols."""
    normalized = raw.replace("\n", ",").replace("\t", ",").replace(" ", ",")
    tickers: list[str] = []
    for item in normalized.split(","):
        ticker = item.strip().upper()
        if ticker and ticker not in tickers:
            tickers.append(ticker)
    return tickers

def display_name_for_ticker(ticker: str) -> str:
    return {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum"}.get(ticker, ticker)


@dataclass(frozen=True)
class Settings:
    history_period: str
    min_base_days: int
    max_base_days: int
    max_box_range_pct: float
    test_tolerance_pct: float
    minimum_high_tests: int
    minimum_low_tests: int
    breakout_buffer_pct: float
    breakout_volume_multiple: float
    dry_up_days: int
    baseline_volume_days: int
    dry_up_ratio_max: float
    atr_days: int
    near_high_pct: float
    chart_days: int




def load_alert_state() -> dict[str, str]:
    try:
        if ALERT_STATE_FILE.exists():
            data = json.loads(ALERT_STATE_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def save_alert_state(state: dict[str, str]) -> None:
    try:
        ALERT_STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError as exc:
        st.warning(f"Could not save alert history: {exc}")



def load_signal_history() -> list[dict[str, Any]]:
    """Load persisted WATCH/CONFIRMED signals used for forward validation."""
    try:
        if SIGNAL_HISTORY_FILE.exists():
            data = json.loads(SIGNAL_HISTORY_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        pass
    return []


def save_signal_history(history: list[dict[str, Any]]) -> None:
    try:
        SIGNAL_HISTORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")
    except OSError as exc:
        st.warning(f"Could not save signal validation history: {exc}")


def record_new_signals(alerts: list[dict[str, Any]]) -> int:
    """Persist new WATCH/CONFIRMED episodes without duplicating the same daily signal."""
    history = load_signal_history()
    existing = {x.get("Signal ID") for x in history}
    added = 0
    for r in alerts:
        ticker = str(r.get("Ticker", ""))
        signal_date = str(r.get("Latest Date", ""))
        signal_type = str(r.get("State", ""))
        level = safe_float(r.get("Breakout Level"))
        if not ticker or not signal_date or signal_type not in {"BREAKOUT WATCH", "CONFIRMED BREAKOUT"}:
            continue
        # One observation per ticker/state/date. This preserves successive watch days for calibration.
        signal_id = f"{ticker}|{signal_type}|{signal_date}"
        if signal_id in existing:
            continue
        history.append({
            "Signal ID": signal_id,
            "Ticker": ticker,
            "Signal Date": signal_date,
            "Signal Type": signal_type,
            "Signal Price": safe_float(r.get("Price")),
            "Breakout Level": level,
            "Box High": safe_float(r.get("Box High")),
            "Overall Score": int(r.get("Strategy Score", 0)),
            "Core Score": int(r.get("Core Score", r.get("Strategy Score", 0))),
            "Short Squeeze Potential": safe_float(r.get("Short Squeeze Potential")),
            "Volume Multiple": safe_float(r.get("Volume Multiple")),
            "5-Day Probability %": safe_float(r.get("5-Day Probability %")),
            "Pre-Breakout Score": safe_float(r.get("Pre-Breakout Score")),
            "Status": "PENDING",
        })
        existing.add(signal_id)
        added += 1
    if added:
        save_signal_history(history)
    return added


def review_mature_signals(market_data: dict[str, pd.DataFrame], sessions: int = 5) -> int:
    """Grade signals after five subsequent trading sessions using close, MFE and MAE."""
    history = load_signal_history()
    changed = 0
    for rec in history:
        if rec.get("Status") == "REVIEWED":
            continue
        ticker = rec.get("Ticker")
        raw = market_data.get(ticker)
        if raw is None or raw.empty:
            continue
        try:
            signal_date = pd.Timestamp(rec.get("Signal Date"))
        except Exception:
            continue
        future = raw.loc[raw.index > signal_date].head(sessions)
        if len(future) < sessions:
            continue
        p0 = safe_float(rec.get("Signal Price"))
        if not np.isfinite(p0) or p0 <= 0:
            continue
        p5 = safe_float(future["Close"].iloc[-1])
        max_high = safe_float(future["High"].max())
        min_low = safe_float(future["Low"].min())
        ret5 = (p5 / p0 - 1) * 100
        mfe = (max_high / p0 - 1) * 100
        mae = (min_low / p0 - 1) * 100
        level = safe_float(rec.get("Breakout Level"))
        held = bool(np.isfinite(level) and p5 >= level)

        # Outcome emphasizes the actual 5-session close; MFE is retained separately
        # so a strong tradable move is not lost when price later fades.
        if ret5 >= 5:
            outcome = "STRONG SUCCESS"
        elif ret5 >= 2:
            outcome = "SUCCESS"
        elif ret5 <= -5:
            outcome = "HARD FAILURE"
        elif ret5 < -2:
            outcome = "FAILED BREAKOUT"
        else:
            outcome = "FLAT / INCONCLUSIVE"
        if rec.get("Signal Type") == "CONFIRMED BREAKOUT" and np.isfinite(level) and p5 < level and ret5 < 0:
            outcome = "HARD FAILURE" if ret5 <= -5 else "FAILED BREAKOUT"

        rec.update({
            "Status": "REVIEWED",
            "Review Date": future.index[-1].strftime("%Y-%m-%d"),
            "Day 5 Price": p5,
            "5-Day Return %": ret5,
            "Max 5-Day Gain %": mfe,
            "Max 5-Day Drawdown %": mae,
            "Held Breakout": held,
            "Outcome": outcome,
        })
        changed += 1
    if changed:
        save_signal_history(history)
    return changed


def overall_score_band(score: Any) -> str:
    """Bucket the scanner's Overall Score into stable validation bands."""
    value = safe_float(score)
    if not np.isfinite(value):
        return "Unknown"
    if value >= 90:
        return "90-100 (Exceptional)"
    if value >= 80:
        return "80-89 (Strong)"
    if value >= 70:
        return "70-79 (Good)"
    if value >= 60:
        return "60-69 (Moderate)"
    return "<60 (Weak)"


def signal_accuracy_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return reviewed signals plus accuracy by signal type and Overall Score band."""
    reviewed = [x for x in load_signal_history() if x.get("Status") == "REVIEWED"]
    if not reviewed:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    detail = pd.DataFrame(reviewed)
    detail["Successful"] = detail["Outcome"].isin(["SUCCESS", "STRONG SUCCESS"])
    detail["Overall Score Numeric"] = pd.to_numeric(detail.get("Overall Score"), errors="coerce")
    detail["Score Band"] = detail["Overall Score Numeric"].apply(overall_score_band)

    type_groups = []
    for signal_type, g in detail.groupby("Signal Type"):
        type_groups.append({
            "Signal Type": signal_type,
            "Reviewed": len(g),
            "Success Rate %": 100 * g["Successful"].mean(),
            "Avg 5-Day Return %": pd.to_numeric(g["5-Day Return %"], errors="coerce").mean(),
            "Avg Max Gain %": pd.to_numeric(g["Max 5-Day Gain %"], errors="coerce").mean(),
            "Avg Max Drawdown %": pd.to_numeric(g["Max 5-Day Drawdown %"], errors="coerce").mean(),
        })

    band_order = [
        "90-100 (Exceptional)",
        "80-89 (Strong)",
        "70-79 (Good)",
        "60-69 (Moderate)",
        "<60 (Weak)",
        "Unknown",
    ]
    band_groups = []
    for band in band_order:
        g = detail.loc[detail["Score Band"] == band]
        if g.empty:
            continue
        band_groups.append({
            "Overall Score Band": band,
            "Reviewed": len(g),
            "Success Rate %": 100 * g["Successful"].mean(),
            "Avg Overall Score": g["Overall Score Numeric"].mean(),
            "Avg 5-Day Return %": pd.to_numeric(g["5-Day Return %"], errors="coerce").mean(),
            "Avg Max Gain %": pd.to_numeric(g["Max 5-Day Gain %"], errors="coerce").mean(),
            "Avg Max Drawdown %": pd.to_numeric(g["Max 5-Day Drawdown %"], errors="coerce").mean(),
            "Breakout Hold Rate %": 100 * g["Held Breakout"].astype(bool).mean() if "Held Breakout" in g else np.nan,
        })

    return detail, pd.DataFrame(type_groups), pd.DataFrame(band_groups)


def pre_breakout_accuracy_frame() -> pd.DataFrame:
    """Summarize reviewed 5-day outcomes by the new 0-10 Pre-Breakout score."""
    reviewed = [x for x in load_signal_history() if x.get("Status") == "REVIEWED"]
    if not reviewed:
        return pd.DataFrame()

    detail = pd.DataFrame(reviewed)
    if "Pre-Breakout Score" not in detail.columns:
        return pd.DataFrame()

    detail["Pre-Breakout Numeric"] = pd.to_numeric(detail["Pre-Breakout Score"], errors="coerce")
    detail = detail.dropna(subset=["Pre-Breakout Numeric"])
    if detail.empty:
        return pd.DataFrame()

    detail["Successful"] = detail["Outcome"].isin(["SUCCESS", "STRONG SUCCESS"])

    def band(v: float) -> str:
        if v >= 9:
            return "9-10 (Very High)"
        if v >= 7:
            return "7-8 (High)"
        if v >= 5:
            return "5-6 (Moderate)"
        return "0-4 (Low)"

    detail["Pre-Breakout Band"] = detail["Pre-Breakout Numeric"].apply(band)
    order = ["9-10 (Very High)", "7-8 (High)", "5-6 (Moderate)", "0-4 (Low)"]
    rows = []
    for b in order:
        g = detail.loc[detail["Pre-Breakout Band"] == b]
        if g.empty:
            continue
        rows.append({
            "Pre-Breakout Band": b,
            "Reviewed": len(g),
            "Success Rate %": 100 * g["Successful"].mean(),
            "Avg 5-Day Return %": pd.to_numeric(g["5-Day Return %"], errors="coerce").mean(),
            "Avg Max Gain %": pd.to_numeric(g["Max 5-Day Gain %"], errors="coerce").mean(),
            "Avg Max Drawdown %": pd.to_numeric(g["Max 5-Day Drawdown %"], errors="coerce").mean(),
        })
    return pd.DataFrame(rows)



def get_email_config() -> dict[str, Any]:
    """Read SMTP settings from Streamlit secrets or environment variables."""
    secret_email = {}
    try:
        secret_email = dict(st.secrets.get("email", {}))
    except Exception:
        secret_email = {}

    def value(name: str, default: Any = "") -> Any:
        return secret_email.get(name, os.getenv(f"BREAKOUT_{name.upper()}", default))

    return {
        "smtp_host": value("smtp_host", "smtp.gmail.com"),
        "smtp_port": int(value("smtp_port", 465)),
        "smtp_username": value("smtp_username"),
        "smtp_password": value("smtp_password"),
        "sender": value("sender") or value("smtp_username"),
        "recipient": value("recipient"),
        "use_ssl": str(value("use_ssl", "true")).lower() in {"1", "true", "yes", "on"},
    }


def email_configured(config: dict[str, Any]) -> bool:
    required = ["smtp_host", "smtp_port", "smtp_username", "smtp_password", "sender", "recipient"]
    return all(config.get(key) for key in required)


def send_breakout_email(
    config: dict[str, Any],
    asset_name: str,
    ticker: str,
    candle_date: str,
    box_result: dict[str, Any],
    strategy_score: int,
    test_only: bool = False,
) -> None:
    subject = (
        f"Test: breakout alerts configured for {asset_name}"
        if test_only
        else f"Confirmed breakout: {asset_name} ({ticker})"
    )
    if test_only:
        body = (
            f"Email alerts are configured correctly for the Darvas + Minervini scanner.\n\n"
            f"Asset currently selected: {asset_name} ({ticker})\n"
            f"Test generated by the Streamlit application."
        )
    else:
        body = f"""A confirmed Darvas breakout was detected.

Asset: {asset_name}
Ticker: {ticker}
Candle date: {candle_date}
Latest close: {format_currency(box_result['latest_close'])}
Box high: {format_currency(box_result['box_high'])}
Breakout level: {format_currency(box_result['breakout_level'])}
Breakout volume: {box_result['volume_multiple']:.2f}x average
Detected base length: {box_result.get('base_days', 0)} days
Base quality: {box_result.get('quality_score', 0):.1f}/100
Strategy score: {strategy_score}/100

The alert requires both a close above the configured breakout level and the configured volume confirmation.

Educational signal only; not investment advice.
"""

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = config["sender"]
    message["To"] = config["recipient"]
    message.set_content(body)

    if config["use_ssl"]:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(config["smtp_host"], config["smtp_port"], context=context, timeout=30) as server:
            server.login(config["smtp_username"], config["smtp_password"])
            server.send_message(message)
    else:
        with smtplib.SMTP(config["smtp_host"], config["smtp_port"], timeout=30) as server:
            server.starttls(context=ssl.create_default_context())
            server.login(config["smtp_username"], config["smtp_password"])
            server.send_message(message)


def maybe_send_breakout_alert(
    enabled: bool,
    config: dict[str, Any],
    asset_name: str,
    ticker: str,
    candle_date: str,
    box_result: dict[str, Any],
    strategy_score: int,
) -> tuple[bool, str]:
    if not enabled:
        return False, "Email alerts are disabled."
    if not email_configured(config):
        return False, "Email settings are incomplete."
    if not box_result.get("confirmed_breakout", False):
        return False, "No confirmed breakout on the latest candle."

    alert_key = f"{ticker}:{candle_date}:{box_result['breakout_level']:.8f}"
    state = load_alert_state()
    if state.get(ticker) == alert_key:
        return False, "This breakout has already been emailed."

    send_breakout_email(
        config=config,
        asset_name=asset_name,
        ticker=ticker,
        candle_date=candle_date,
        box_result=box_result,
        strategy_score=strategy_score,
    )
    state[ticker] = alert_key
    save_alert_state(state)
    return True, f"Breakout alert sent to {config['recipient']}."


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@st.cache_data(ttl=900, show_spinner=False)
def download_market_data(ticker: str, period: str) -> pd.DataFrame:
    """Download daily OHLCV data and normalize yfinance output."""
    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing columns from market data: {', '.join(missing)}")

    data = data[required].copy()
    data.index = pd.to_datetime(data.index).tz_localize(None)
    data = data.apply(pd.to_numeric, errors="coerce").dropna(subset=["Open", "High", "Low", "Close"])
    data["Volume"] = data["Volume"].fillna(0)
    return data


def add_indicators(data: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    df = data.copy()

    for length in (20, 50, 150, 200):
        df[f"SMA_{length}"] = df["Close"].rolling(length).mean()

    prior_close = df["Close"].shift(1)
    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prior_close).abs(),
            (df["Low"] - prior_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["TR"] = true_range
    df["ATR"] = true_range.rolling(settings.atr_days).mean()
    df["ATR_Pct"] = (df["ATR"] / df["Close"]) * 100

    df["Volume_Avg_20"] = df["Volume"].rolling(20).mean()
    df["Dollar_Volume"] = df["Close"] * df["Volume"]
    df["Dollar_Volume_Avg_20"] = df["Dollar_Volume"].rolling(20).mean()

    df["High_365"] = df["High"].rolling(365, min_periods=180).max()
    df["Low_365"] = df["Low"].rolling(365, min_periods=180).min()
    df["Distance_From_365D_High_Pct"] = ((df["High_365"] - df["Close"]) / df["High_365"]) * 100

    df["Return_30D_Pct"] = df["Close"].pct_change(30) * 100
    df["Return_90D_Pct"] = df["Close"].pct_change(90) * 100
    df["Return_180D_Pct"] = df["Close"].pct_change(180) * 100

    # --- Pre-breakout momentum indicators ---
    # RSI(14), Wilder smoothing.
    delta = df["Close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))
    df.loc[(avg_loss == 0) & (avg_gain > 0), "RSI_14"] = 100.0
    df.loc[(avg_loss == 0) & (avg_gain == 0), "RSI_14"] = 50.0

    # On-Balance Volume.
    direction = np.sign(df["Close"].diff()).fillna(0.0)
    df["OBV"] = (direction * df["Volume"]).cumsum()

    # Bollinger Band Width(20, 2). Lower values indicate tighter compression.
    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std(ddof=0)
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["BB_Width"] = ((bb_upper - bb_lower) / bb_mid.replace(0, np.nan)) * 100
    df["BB_Width_Avg_20"] = df["BB_Width"].rolling(20).mean()

    # Chaikin Money Flow(20): positive values indicate accumulation.
    hl_range = (df["High"] - df["Low"]).replace(0, np.nan)
    money_flow_multiplier = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl_range
    money_flow_volume = money_flow_multiplier.fillna(0.0) * df["Volume"]
    volume_sum_20 = df["Volume"].rolling(20).sum().replace(0, np.nan)
    df["CMF_20"] = money_flow_volume.rolling(20).sum() / volume_sum_20

    return df


def detect_current_box(df: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    """Automatically identify the strongest active Darvas-style base.

    Candidate bases all end on the most recent completed candle. Their starting
    date varies from ``min_base_days`` to ``max_base_days``. Resistance and
    support are estimated from clusters of highs and lows, rather than from one
    isolated extreme candle. The latest candle is evaluated separately as the
    possible breakout candle.
    """
    minimum_required = settings.max_base_days + 2
    if len(df) < minimum_required:
        return {"valid": False, "reason": f"At least {minimum_required} candles are required"}

    completed = df.iloc[:-1].copy()
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    tolerance = settings.test_tolerance_pct / 100
    candidates: list[dict[str, Any]] = []

    for base_days in range(settings.min_base_days, settings.max_base_days + 1):
        base = completed.tail(base_days).copy()
        if len(base) < base_days:
            continue

        # Use price clusters to keep a one-day wick from defining the box.
        upper_seed = safe_float(base["High"].quantile(0.85))
        lower_seed = safe_float(base["Low"].quantile(0.15))
        upper_cluster = base.loc[base["High"] >= upper_seed, "High"]
        lower_cluster = base.loc[base["Low"] <= lower_seed, "Low"]
        box_high = safe_float(upper_cluster.median())
        box_low = safe_float(lower_cluster.median())

        if not np.isfinite(box_high) or not np.isfinite(box_low) or box_high <= box_low:
            continue

        midpoint = (box_high + box_low) / 2
        range_pct = ((box_high - box_low) / midpoint) * 100
        high_tests = int((base["High"].sub(box_high).abs() / box_high <= tolerance).sum())
        low_tests = int((base["Low"].sub(box_low).abs() / box_low <= tolerance).sum())

        # Slight intraday excursions are allowed; closes should remain contained.
        close_tolerance = max(tolerance, 0.01)
        upper_close_limit = box_high * (1 + close_tolerance)
        lower_close_limit = box_low * (1 - close_tolerance)
        inside_ratio = float(
            ((base["Close"] <= upper_close_limit) & (base["Close"] >= lower_close_limit)).mean()
        )

        half = max(3, base_days // 2)
        early_atr = safe_float(base["ATR_Pct"].head(half).mean())
        late_atr = safe_float(base["ATR_Pct"].tail(half).mean())
        atr_contracting = bool(np.isfinite(early_atr) and np.isfinite(late_atr) and late_atr <= early_atr * 1.05)

        range_pass = range_pct <= settings.max_box_range_pct
        high_tests_pass = high_tests >= settings.minimum_high_tests
        low_tests_pass = low_tests >= settings.minimum_low_tests
        containment_pass = inside_ratio >= 0.90
        valid = all([range_pass, high_tests_pass, low_tests_pass, containment_pass])
        if not valid:
            continue

        # Prefer tight, well-tested, contained and volatility-contracting bases.
        tightness_score = max(0.0, 1 - range_pct / settings.max_box_range_pct)
        touch_score = min(1.0, (high_tests + low_tests) / 8)
        length_score = min(1.0, base_days / 40)
        quality_score = (
            35 * tightness_score
            + 25 * touch_score
            + 20 * inside_ratio
            + 10 * length_score
            + (10 if atr_contracting else 0)
        )

        candidates.append({
            "base": base,
            "base_days": base_days,
            "box_high": box_high,
            "box_low": box_low,
            "box_range_pct": range_pct,
            "high_tests": high_tests,
            "low_tests": low_tests,
            "inside_ratio": inside_ratio,
            "atr_contracting": atr_contracting,
            "quality_score": quality_score,
        })

    if not candidates:
        return {
            "valid": False,
            "state": "NO VALID BOX",
            "reason": "No qualifying base found within the automatic search range",
            "box_high": np.nan,
            "box_low": np.nan,
            "box_range_pct": np.nan,
            "high_tests": 0,
            "low_tests": 0,
            "inside_ratio": 0.0,
            "breakout_level": np.nan,
            "latest_close": safe_float(latest["Close"]),
            "previous_close": safe_float(previous["Close"]),
            "volume_multiple": np.nan,
            "price_breakout": False,
            "volume_breakout": False,
            "confirmed_breakout": False,
            "base_days": 0,
            "quality_score": 0.0,
            "checks": {},
        }

    # A small preference for the most recent/tighter nested base is already
    # captured by tightness; quality decides among all natural base lengths.
    winner = max(candidates, key=lambda item: item["quality_score"])
    base = winner["base"]
    box_high = winner["box_high"]
    box_low = winner["box_low"]
    breakout_level = box_high * (1 + settings.breakout_buffer_pct / 100)
    near_breakout_floor = box_high * 0.98

    latest_close = safe_float(latest["Close"])
    previous_close = safe_float(previous["Close"])
    latest_volume = safe_float(latest["Volume"], 0.0)
    average_volume = safe_float(df["Volume"].iloc[-21:-1].mean(), 0.0)
    volume_multiple = latest_volume / average_volume if average_volume > 0 else np.nan

    price_breakout = latest_close > breakout_level and previous_close <= breakout_level
    volume_breakout = volume_multiple >= settings.breakout_volume_multiple
    confirmed_breakout = price_breakout and volume_breakout
    price_only_breakout = price_breakout and not volume_breakout
    breakout_watch = not price_breakout and latest_close >= near_breakout_floor

    if confirmed_breakout:
        state = "CONFIRMED BREAKOUT"
    elif price_only_breakout:
        state = "PRICE BREAKOUT / WEAK VOLUME"
    elif breakout_watch:
        state = "BREAKOUT WATCH"
    else:
        state = "BUILDING A BOX"

    return {
        "valid": True,
        "state": state,
        "box_high": box_high,
        "box_low": box_low,
        "box_range_pct": winner["box_range_pct"],
        "high_tests": winner["high_tests"],
        "low_tests": winner["low_tests"],
        "inside_ratio": winner["inside_ratio"],
        "breakout_level": breakout_level,
        "latest_close": latest_close,
        "previous_close": previous_close,
        "volume_multiple": volume_multiple,
        "price_breakout": price_breakout,
        "volume_breakout": volume_breakout,
        "confirmed_breakout": confirmed_breakout,
        "box_start": base.index[0],
        "box_end": base.index[-1],
        "base_days": winner["base_days"],
        "quality_score": winner["quality_score"],
        "atr_contracting": winner["atr_contracting"],
        "candidate_count": len(candidates),
        "checks": {
            "Range within limit": True,
            "Enough upper-bound tests": True,
            "Enough lower-bound tests": True,
            "At least 90% closes contained": True,
            "ATR stable or contracting": winner["atr_contracting"],
        },
    }

def evaluate_trend_template(df: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    latest = df.iloc[-1]

    close = safe_float(latest["Close"])
    sma_50 = safe_float(latest["SMA_50"])
    sma_150 = safe_float(latest["SMA_150"])
    sma_200 = safe_float(latest["SMA_200"])
    high_365 = safe_float(latest["High_365"])
    low_365 = safe_float(latest["Low_365"])

    sma_200_20_days_ago = safe_float(df["SMA_200"].iloc[-21]) if len(df) >= 221 else np.nan
    midpoint_365 = (high_365 + low_365) / 2 if np.isfinite(high_365) and np.isfinite(low_365) else np.nan
    distance_from_high = ((high_365 - close) / high_365) * 100 if high_365 else np.nan

    checks = {
        "Price above 50-day SMA": close > sma_50,
        "Price above 150-day SMA": close > sma_150,
        "Price above 200-day SMA": close > sma_200,
        "50-day SMA above 150-day SMA": sma_50 > sma_150,
        "150-day SMA above 200-day SMA": sma_150 > sma_200,
        "200-day SMA rising": sma_200 > sma_200_20_days_ago,
        "Price above 365-day midpoint": close > midpoint_365,
        f"Within {settings.near_high_pct:.0f}% of 365-day high": distance_from_high <= settings.near_high_pct,
    }

    passed = sum(bool(value) for value in checks.values())
    return {
        "checks": checks,
        "passed": passed,
        "total": len(checks),
        "pass_pct": passed / len(checks) * 100,
        "distance_from_high_pct": distance_from_high,
        "sma_50": sma_50,
        "sma_150": sma_150,
        "sma_200": sma_200,
    }


def evaluate_volume_dry_up(df: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    if len(df) < settings.baseline_volume_days + settings.dry_up_days + 2:
        return {"pass": False, "ratio": np.nan}

    # Exclude the latest candle because it might be the breakout candle.
    completed = df.iloc[:-1]
    recent = completed["Dollar_Volume"].tail(settings.dry_up_days)
    baseline_end = len(completed) - settings.dry_up_days
    baseline_start = max(0, baseline_end - settings.baseline_volume_days)
    baseline = completed["Dollar_Volume"].iloc[baseline_start:baseline_end]

    recent_average = safe_float(recent.mean())
    baseline_average = safe_float(baseline.mean())
    ratio = recent_average / baseline_average if baseline_average > 0 else np.nan

    recent_atr = safe_float(completed["ATR_Pct"].tail(settings.dry_up_days).mean())
    prior_atr = safe_float(
        completed["ATR_Pct"].iloc[baseline_start:baseline_end].mean()
    )
    atr_contracting = recent_atr < prior_atr if np.isfinite(prior_atr) else False

    return {
        "pass": bool(ratio <= settings.dry_up_ratio_max and atr_contracting),
        "ratio": ratio,
        "recent_average": recent_average,
        "baseline_average": baseline_average,
        "recent_atr_pct": recent_atr,
        "prior_atr_pct": prior_atr,
        "atr_contracting": atr_contracting,
    }


def evaluate_relative_strength(
    asset_ticker: str,
    asset_df: pd.DataFrame,
    btc_df: pd.DataFrame,
) -> dict[str, Any]:
    if asset_ticker == "BTC-USD":
        returns = {
            "30-day return positive": safe_float(asset_df["Return_30D_Pct"].iloc[-1]) > 0,
            "90-day return positive": safe_float(asset_df["Return_90D_Pct"].iloc[-1]) > 0,
            "180-day return positive": safe_float(asset_df["Return_180D_Pct"].iloc[-1]) > 0,
        }
        passed = sum(returns.values())
        return {"label": "BTC momentum", "checks": returns, "passed": passed,
                "total": len(returns), "ratio_series": None,
                "latest_ratio": np.nan, "ratio_column": None}

    aligned = pd.concat(
        [asset_df["Close"].rename("Asset"), btc_df["Close"].rename("BTC")],
        axis=1, join="inner"
    ).dropna()
    ratio_col = "ASSET_BTC"
    aligned[ratio_col] = aligned["Asset"] / aligned["BTC"]
    aligned["SMA_50"] = aligned[ratio_col].rolling(50).mean()
    aligned["SMA_200"] = aligned[ratio_col].rolling(200).mean()
    aligned["Return_30D"] = aligned[ratio_col].pct_change(30) * 100
    aligned["Return_90D"] = aligned[ratio_col].pct_change(90) * 100
    latest = aligned.iloc[-1]
    checks = {
        f"{asset_ticker}/BTC above 50-day average": latest[ratio_col] > latest["SMA_50"],
        f"{asset_ticker}/BTC above 200-day average": latest[ratio_col] > latest["SMA_200"],
        f"{asset_ticker}/BTC 30-day return positive": latest["Return_30D"] > 0,
        f"{asset_ticker}/BTC 90-day return positive": latest["Return_90D"] > 0,
    }
    passed = sum(bool(v) for v in checks.values())
    return {"label": f"{asset_ticker} relative strength vs BTC", "checks": checks,
            "passed": passed, "total": len(checks), "ratio_series": aligned,
            "latest_ratio": safe_float(latest[ratio_col]), "ratio_column": ratio_col}


def evaluate_pre_breakout_momentum(
    asset_df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None,
    asset_ticker: str,
    benchmark_ticker: str,
) -> dict[str, Any]:
    """Score leading/pre-breakout evidence on a separate 0-10 scale.

    Components:
      Relative-strength line leadership  0-3
      OBV accumulation                   0-2
      Bollinger-width compression        0-2
      RSI momentum                       0-2
      Chaikin money flow                 0-1

    This score is intentionally NOT added to the existing Overall Score yet.
    Keeping it separate lets the 5-day validation history determine whether it
    adds predictive value before it changes the production ranking.
    """
    result: dict[str, Any] = {
        "available": False,
        "score": 0,
        "max_score": 10,
        "label": "N/A",
        "components": {},
        "checks": {},
    }
    if asset_df is None or asset_df.empty or len(asset_df) < 60:
        return result

    latest = asset_df.iloc[-1]
    components: dict[str, int] = {}
    checks: dict[str, bool] = {}

    # 1) Relative-strength line leadership: 0-3.
    rs_points = 0
    if benchmark_df is not None and not benchmark_df.empty and asset_ticker != benchmark_ticker:
        aligned = pd.concat(
            [
                asset_df["Close"].rename("Asset"),
                benchmark_df["Close"].rename("Benchmark"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        if len(aligned) >= 60:
            aligned["RS_Line"] = aligned["Asset"] / aligned["Benchmark"]
            aligned["RS_SMA_50"] = aligned["RS_Line"].rolling(50).mean()
            aligned["RS_High_90"] = aligned["RS_Line"].rolling(90, min_periods=60).max()
            rs_latest = aligned.iloc[-1]
            rs_10_ago = safe_float(aligned["RS_Line"].iloc[-11]) if len(aligned) >= 11 else np.nan
            rs_now = safe_float(rs_latest["RS_Line"])
            rs_sma50 = safe_float(rs_latest["RS_SMA_50"])
            rs_high90 = safe_float(rs_latest["RS_High_90"])

            c1 = np.isfinite(rs_now) and np.isfinite(rs_sma50) and rs_now > rs_sma50
            c2 = np.isfinite(rs_now) and np.isfinite(rs_high90) and rs_high90 > 0 and rs_now >= rs_high90 * 0.99
            c3 = np.isfinite(rs_now) and np.isfinite(rs_10_ago) and rs_now > rs_10_ago
            checks["RS line above 50-day average"] = bool(c1)
            checks["RS line at/near 90-day high"] = bool(c2)
            checks["RS line rising over 10 sessions"] = bool(c3)
            rs_points = int(c1) + int(c2) + int(c3)
    elif asset_ticker == benchmark_ticker:
        # For the benchmark itself, use price momentum as a neutral substitute.
        close_now = safe_float(asset_df["Close"].iloc[-1])
        close_10 = safe_float(asset_df["Close"].iloc[-11]) if len(asset_df) >= 11 else np.nan
        sma50 = safe_float(asset_df["SMA_50"].iloc[-1])
        high90 = safe_float(asset_df["Close"].rolling(90, min_periods=60).max().iloc[-1])
        c1 = np.isfinite(close_now) and np.isfinite(sma50) and close_now > sma50
        c2 = np.isfinite(close_now) and np.isfinite(high90) and high90 > 0 and close_now >= high90 * 0.99
        c3 = np.isfinite(close_now) and np.isfinite(close_10) and close_now > close_10
        checks["Price above 50-day average (benchmark proxy)"] = bool(c1)
        checks["Price at/near 90-day high (benchmark proxy)"] = bool(c2)
        checks["Price rising over 10 sessions (benchmark proxy)"] = bool(c3)
        rs_points = int(c1) + int(c2) + int(c3)
    components["Relative Strength"] = rs_points

    # 2) OBV accumulation: 0-2.
    obv_now = safe_float(latest.get("OBV"))
    obv_high50 = safe_float(asset_df["OBV"].rolling(50, min_periods=30).max().iloc[-1])
    obv_10 = safe_float(asset_df["OBV"].iloc[-11]) if len(asset_df) >= 11 else np.nan
    obv_high = np.isfinite(obv_now) and np.isfinite(obv_high50) and obv_now >= obv_high50 * 0.995
    obv_rising = np.isfinite(obv_now) and np.isfinite(obv_10) and obv_now > obv_10
    checks["OBV at/near 50-day high"] = bool(obv_high)
    checks["OBV rising over 10 sessions"] = bool(obv_rising)
    components["OBV"] = int(obv_high) + int(obv_rising)

    # 3) Bollinger Band Width compression: 0-2.
    bbw_now = safe_float(latest.get("BB_Width"))
    bbw_hist = asset_df["BB_Width"].dropna().tail(126)
    bbw_p20 = safe_float(bbw_hist.quantile(0.20)) if len(bbw_hist) >= 40 else np.nan
    bbw_avg20 = safe_float(latest.get("BB_Width_Avg_20"))
    bbw_compressed = np.isfinite(bbw_now) and np.isfinite(bbw_p20) and bbw_now <= bbw_p20
    bbw_contracting = np.isfinite(bbw_now) and np.isfinite(bbw_avg20) and bbw_now < bbw_avg20
    checks["Bollinger width in lowest 20% of 6-month range"] = bool(bbw_compressed)
    checks["Bollinger width below its 20-day average"] = bool(bbw_contracting)
    components["Volatility Compression"] = int(bbw_compressed) + int(bbw_contracting)

    # 4) RSI momentum: 0-2. Strong but not extremely extended.
    rsi_now = safe_float(latest.get("RSI_14"))
    rsi_5 = safe_float(asset_df["RSI_14"].iloc[-6]) if len(asset_df) >= 6 else np.nan
    rsi_zone = np.isfinite(rsi_now) and 50 <= rsi_now <= 80
    rsi_rising = np.isfinite(rsi_now) and np.isfinite(rsi_5) and rsi_now > rsi_5
    checks["RSI in 50-80 momentum zone"] = bool(rsi_zone)
    checks["RSI rising over 5 sessions"] = bool(rsi_rising)
    components["RSI Momentum"] = int(rsi_zone) + int(rsi_rising)

    # 5) Chaikin Money Flow: 0-1.
    cmf_now = safe_float(latest.get("CMF_20"))
    cmf_positive = np.isfinite(cmf_now) and cmf_now > 0.05
    checks["CMF(20) above +0.05"] = bool(cmf_positive)
    components["Money Flow"] = int(cmf_positive)

    total = int(sum(components.values()))
    label = "VERY HIGH" if total >= 9 else "HIGH" if total >= 7 else "MODERATE" if total >= 5 else "LOW"
    result.update({
        "available": True,
        "score": total,
        "label": label,
        "components": components,
        "checks": checks,
        "rsi": rsi_now,
        "cmf": cmf_now,
        "bb_width": bbw_now,
        "bb_width_20th_pct": bbw_p20,
    })
    return result



def calculate_score(
    box_result: dict[str, Any],
    trend_result: dict[str, Any],
    dry_up_result: dict[str, Any],
    rs_result: dict[str, Any],
) -> dict[str, Any]:
    box_quality = safe_float(box_result.get("quality_score", 0.0), 0.0)
    box_points = (
        round(25 * max(0.0, min(100.0, box_quality)) / 100.0)
        if box_result["valid"]
        else 0
    )
    trend_points = round(30 * trend_result["passed"] / trend_result["total"])
    dry_up_points = 15 if dry_up_result["pass"] else 0
    rs_points = round(15 * rs_result["passed"] / rs_result["total"])

    breakout_points = 0
    if box_result["confirmed_breakout"]:
        breakout_points = 15
    elif box_result["price_breakout"]:
        breakout_points = 8
    elif box_result["state"] == "BREAKOUT WATCH":
        breakout_points = 4

    total = box_points + trend_points + dry_up_points + rs_points + breakout_points

    return {
        "Box": box_points,
        "Trend": trend_points,
        "Dry-up": dry_up_points,
        "Relative strength": rs_points,
        "Breakout": breakout_points,
        "Total": total,
    }



def _future_confirmed_breakout(
    df: pd.DataFrame,
    watch_pos: int,
    breakout_level: float,
    settings: Settings,
    horizon: int,
) -> bool:
    """Check whether a frozen watch-level confirms within the next N sessions."""
    end = min(len(df) - 1, watch_pos + horizon)
    for pos in range(watch_pos + 1, end + 1):
        close = safe_float(df["Close"].iloc[pos])
        volume = safe_float(df["Volume"].iloc[pos], 0.0)
        prior_start = max(0, pos - 20)
        prior_avg = safe_float(df["Volume"].iloc[prior_start:pos].mean(), 0.0)
        volume_multiple = volume / prior_avg if prior_avg > 0 else np.nan
        if close > breakout_level and np.isfinite(volume_multiple) and volume_multiple >= settings.breakout_volume_multiple:
            return True
    return False


@st.cache_data(ttl=900, show_spinner=False)
def estimate_breakout_probability(
    df: pd.DataFrame,
    settings: Settings,
    current_box: dict[str, Any],
    current_trend: dict[str, Any],
    current_dry_up: dict[str, Any],
    horizons: tuple[int, ...] = (3, 5, 10),
    max_samples: int = 120,
) -> dict[str, Any]:
    """Estimate watch->confirmed-breakout odds from this ticker's own history.

    This is a walk-forward analog estimator. For each historical date we only use
    candles available as of that date, identify BREAKOUT WATCH states, freeze that
    watch's breakout level, then observe whether a volume-confirmed breakout occurs
    within the requested future horizons. Similar historical setups receive larger
    weights than dissimilar ones.
    """
    if not current_box.get("valid") or current_box.get("state") != "BREAKOUT WATCH":
        return {"available": False, "reason": "Current state is not BREAKOUT WATCH", "probabilities": {}}

    warmup = max(221, settings.max_base_days + 2, settings.baseline_volume_days + settings.dry_up_days + 5)
    if len(df) < warmup + max(horizons) + 10:
        return {"available": False, "reason": "Not enough history for probability calibration", "probabilities": {}}

    latest_close = safe_float(current_box.get("latest_close"))
    breakout_level = safe_float(current_box.get("breakout_level"))
    current_distance = max(0.0, (breakout_level - latest_close) / breakout_level * 100) if breakout_level else np.nan
    current_features = {
        "distance": current_distance,
        "quality": safe_float(current_box.get("quality_score"), 0.0),
        "volume": safe_float(current_box.get("volume_multiple"), 0.0),
        "trend": safe_float(current_trend.get("pass_pct"), 0.0),
        "dryup": safe_float(current_dry_up.get("ratio"), 1.0),
        "range": safe_float(current_box.get("box_range_pct"), settings.max_box_range_pct),
    }

    samples: list[dict[str, Any]] = []
    last_watch_pos = -999
    # Sample every second session and cap the calibration window to keep Streamlit responsive.
    step = 2
    final_train_pos = len(df) - max(horizons) - 1
    first_train_pos = max(warmup - 1, final_train_pos - 520)

    for pos in range(first_train_pos, final_train_pos, step):
        # Avoid counting the same multi-day watch episode every day.
        if pos - last_watch_pos <= 2:
            continue
        hist = df.iloc[: pos + 1].copy()
        box = detect_current_box(hist, settings)
        if not box.get("valid") or box.get("state") != "BREAKOUT WATCH":
            continue

        trend = evaluate_trend_template(hist, settings)
        dry = evaluate_volume_dry_up(hist, settings)
        close = safe_float(box.get("latest_close"))
        level = safe_float(box.get("breakout_level"))
        if not np.isfinite(close) or not np.isfinite(level) or level <= 0:
            continue

        distance = max(0.0, (level - close) / level * 100)
        feat = {
            "distance": distance,
            "quality": safe_float(box.get("quality_score"), 0.0),
            "volume": safe_float(box.get("volume_multiple"), 0.0),
            "trend": safe_float(trend.get("pass_pct"), 0.0),
            "dryup": safe_float(dry.get("ratio"), 1.0),
            "range": safe_float(box.get("box_range_pct"), settings.max_box_range_pct),
        }

        # Normalized Euclidean distance. Scales are intentionally interpretable.
        d2 = (
            ((feat["distance"] - current_features["distance"]) / 1.5) ** 2
            + ((feat["quality"] - current_features["quality"]) / 20.0) ** 2
            + ((feat["volume"] - current_features["volume"]) / 0.75) ** 2
            + ((feat["trend"] - current_features["trend"]) / 25.0) ** 2
            + ((feat["dryup"] - current_features["dryup"]) / 0.35) ** 2
            + ((feat["range"] - current_features["range"]) / 7.5) ** 2
        )
        weight = 1.0 / (1.0 + d2)
        outcomes = {
            h: _future_confirmed_breakout(df, pos, level, settings, h)
            for h in horizons
        }
        samples.append({"pos": pos, "weight": weight, "outcomes": outcomes, **feat})
        last_watch_pos = pos

    if not samples:
        return {"available": False, "reason": "No historical BREAKOUT WATCH samples found", "probabilities": {}}

    samples = sorted(samples, key=lambda x: x["weight"], reverse=True)[:max_samples]
    total_weight = sum(s["weight"] for s in samples)
    probabilities = {}
    raw_rates = {}
    for h in horizons:
        probabilities[h] = 100.0 * sum(s["weight"] * int(s["outcomes"][h]) for s in samples) / total_weight
        raw_rates[h] = 100.0 * sum(int(s["outcomes"][h]) for s in samples) / len(samples)

    # Effective sample size is more informative than raw count when weighting.
    w = np.array([s["weight"] for s in samples], dtype=float)
    effective_n = float((w.sum() ** 2) / np.square(w).sum()) if np.square(w).sum() > 0 else 0.0
    p5 = probabilities.get(5, next(iter(probabilities.values())))
    confidence = "HIGH" if effective_n >= 25 else "MEDIUM" if effective_n >= 12 else "LOW"
    band = "HIGH" if p5 >= 70 else "MODERATE" if p5 >= 50 else "LOW"

    return {
        "available": True,
        "probabilities": probabilities,
        "raw_rates": raw_rates,
        "samples": len(samples),
        "effective_samples": effective_n,
        "confidence": confidence,
        "probability_band": band,
        "current_distance_pct": current_distance,
    }


def find_prior_resistance_levels(
    df: pd.DataFrame,
    box_result: dict[str, Any],
    max_levels: int = 3,
    swing_order: int = 5,
    cluster_pct: float = 2.0,
) -> list[float]:
    """Find clustered historical swing highs above the current price."""
    if not box_result.get("valid"):
        return []

    current_price = safe_float(box_result.get("latest_close"))
    breakout_level = safe_float(box_result.get("breakout_level"))
    if not np.isfinite(current_price):
        return []

    # For BREAKOUT WATCH, resistance targets should be above the hypothetical
    # breakout level, not merely above today's price.
    minimum_target_price = (
        max(current_price, breakout_level)
        if np.isfinite(breakout_level)
        else current_price
    )

    history = df.copy()
    box_start = box_result.get("box_start")
    if box_start is not None:
        history = history.loc[history.index < box_start]
    highs = history["High"].astype(float)
    if len(highs) < swing_order * 2 + 1:
        return []

    candidates: list[float] = []
    vals = highs.to_numpy()
    for i in range(swing_order, len(vals) - swing_order):
        center = vals[i]
        if not np.isfinite(center) or center <= minimum_target_price:
            continue
        if center >= np.nanmax(vals[i - swing_order:i + swing_order + 1]):
            candidates.append(float(center))

    if not candidates:
        return []

    # Cluster nearby resistance prices so repeated tests of the same zone count once.
    candidates.sort()
    clusters: list[list[float]] = []
    for price in candidates:
        if not clusters:
            clusters.append([price])
            continue
        center = float(np.mean(clusters[-1]))
        if abs(price - center) / center * 100 <= cluster_pct:
            clusters[-1].append(price)
        else:
            clusters.append([price])

    levels = [float(np.mean(c)) for c in clusters]
    levels = [x for x in levels if x > minimum_target_price]
    return levels[:max_levels]


def calculate_breakout_targets(
    df: pd.DataFrame,
    box_result: dict[str, Any],
) -> dict[str, Any]:
    """Build structural upside targets for a watch, price-only, or confirmed breakout."""
    if not box_result.get("valid"):
        return {"available": False, "targets": []}

    latest_close = safe_float(box_result.get("latest_close"))
    breakout_level = safe_float(box_result.get("breakout_level"))
    box_high = safe_float(box_result.get("box_high"))
    box_low = safe_float(box_result.get("box_low"))
    latest_atr = safe_float(df["ATR"].iloc[-1])
    if not all(np.isfinite(x) for x in [latest_close, breakout_level, box_high, box_low]):
        return {"available": False, "targets": []}

    targets: list[dict[str, Any]] = []
    for idx, level in enumerate(find_prior_resistance_levels(df, box_result), start=1):
        targets.append({"name": f"Resistance R{idx}", "price": level, "type": "Prior swing resistance"})

    box_height = max(0.0, box_high - box_low)
    if box_height > 0:
        targets.append({"name": "Darvas target", "price": breakout_level + box_height, "type": "Box-height measured move"})

    if np.isfinite(latest_atr) and latest_atr > 0:
        for mult in (1, 2, 3):
            targets.append({"name": f"{mult} ATR target", "price": breakout_level + mult * latest_atr, "type": "Volatility extension"})

    for target in targets:
        target["upside_from_price_pct"] = (target["price"] / latest_close - 1) * 100
        target["upside_from_breakout_pct"] = (target["price"] / breakout_level - 1) * 100

    minimum_target_price = max(latest_close, breakout_level)
    targets = sorted(
        [t for t in targets if np.isfinite(t["price"]) and t["price"] > minimum_target_price],
        key=lambda t: t["price"],
    )
    return {
        "available": bool(targets),
        "targets": targets,
        "nearest_resistance": next((t for t in targets if t["type"] == "Prior swing resistance"), None),
        "darvas_target": next((t for t in targets if t["name"] == "Darvas target"), None),
        "atr": latest_atr,
    }

def make_price_chart(
    df: pd.DataFrame,
    box_result: dict[str, Any],
    settings: Settings,
    asset_name: str,
) -> go.Figure:
    visible = df.tail(settings.chart_days).copy()

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=visible.index,
            open=visible["Open"],
            high=visible["High"],
            low=visible["Low"],
            close=visible["Close"],
            name=asset_name,
        )
    )

    for length in (50, 150, 200):
        fig.add_trace(
            go.Scatter(
                x=visible.index,
                y=visible[f"SMA_{length}"],
                mode="lines",
                name=f"{length}-day SMA",
                line={"width": 1.2},
            )
        )

    if box_result.get("box_start") in visible.index or box_result.get("box_end") in visible.index:
        box_start = max(box_result["box_start"], visible.index.min())
        fig.add_shape(
            type="rect",
            x0=box_start,
            x1=visible.index.max(),
            y0=box_result["box_low"],
            y1=box_result["box_high"],
            line={"width": 1.5, "dash": "dash"},
            fillcolor="rgba(120,120,120,0.10)",
        )
        fig.add_hline(
            y=box_result["breakout_level"],
            line_dash="dot",
            annotation_text="Breakout level",
        )

    fig.update_layout(
        title=f"{asset_name}: Price, Trend Averages and Current Darvas Box",
        xaxis_title=None,
        yaxis_title="USD",
        xaxis_rangeslider_visible=False,
        height=650,
        legend={"orientation": "h"},
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
    )
    return fig


def make_volume_chart(df: pd.DataFrame, settings: Settings) -> go.Figure:
    visible = df.tail(settings.chart_days).copy()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=visible.index,
            y=visible["Dollar_Volume"],
            name="Daily USD volume",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=visible.index,
            y=visible["Dollar_Volume_Avg_20"],
            mode="lines",
            name="20-day average USD volume",
        )
    )
    fig.update_layout(
        title="Dollar Volume",
        xaxis_title=None,
        yaxis_title="USD",
        height=380,
        legend={"orientation": "h"},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return fig


def render_checks(title: str, checks: dict[str, bool]) -> None:
    st.subheader(title)
    for label, passed in checks.items():
        st.write(f"{'✅' if passed else '❌'} {label}")


def format_currency(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    if abs(value) >= 1_000:
        return f"${value:,.2f}"
    return f"${value:,.4f}"



def load_daily_scan_state() -> dict[str, Any]:
    try:
        if DAILY_SCAN_STATE_FILE.exists():
            data = json.loads(DAILY_SCAN_STATE_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def save_daily_scan_state(state: dict[str, Any]) -> None:
    try:
        DAILY_SCAN_STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError:
        pass


@st.cache_data(ttl=86400, show_spinner=False)
def get_sp500_tickers() -> list[str]:
    """Return current S&P 500 Yahoo Finance symbols with two public-source fallbacks."""
    urls = [
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
    ]
    for url in urls:
        try:
            table = pd.read_csv(url)
            symbol_col = next((c for c in table.columns if str(c).lower() in {"symbol", "ticker"}), None)
            if symbol_col:
                symbols = [str(s).strip().upper().replace(".", "-") for s in table[symbol_col].dropna()]
                symbols = list(dict.fromkeys(s for s in symbols if s))
                if len(symbols) >= 450:
                    return symbols
        except Exception:
            pass

    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        for table in tables:
            symbol_col = next((c for c in table.columns if str(c).lower() in {"symbol", "ticker symbol"}), None)
            if symbol_col:
                symbols = [str(s).strip().upper().replace(".", "-") for s in table[symbol_col].dropna()]
                symbols = list(dict.fromkeys(s for s in symbols if s))
                if len(symbols) >= 450:
                    return symbols
    except Exception:
        pass
    raise RuntimeError("Could not load the S&P 500 constituent list from either source.")


@st.cache_data(ttl=86400, show_spinner=False)
def get_nasdaq100_tickers() -> list[str]:
    """Return current Nasdaq-100 Yahoo Finance symbols.

    The loader intentionally tries multiple public pages because index membership
    changes over time and any one source can occasionally be unavailable.
    """
    # CSV first: this avoids an lxml dependency on Streamlit deployments.
    csv_urls = [
        "https://raw.githubusercontent.com/Gary-Strauss/NASDAQ100_Constituents/master/data/nasdaq100_constituents.csv",
    ]
    for url in csv_urls:
        try:
            table = pd.read_csv(url)
            symbol_col = next(
                (c for c in table.columns if str(c).strip().lower() in {"symbol", "ticker", "ticker symbol"}),
                None,
            )
            if symbol_col is not None:
                symbols = [str(s).strip().upper().replace(".", "-") for s in table[symbol_col].dropna()]
                symbols = list(dict.fromkeys(s for s in symbols if s and len(s) <= 8))
                if len(symbols) >= 90:
                    return symbols
        except Exception:
            pass

    # HTML fallbacks if the CSV source is temporarily unavailable.
    urls = [
        "https://www.nasdaq.com/solutions/global-indexes/nasdaq-100/companies",
        "https://en.wikipedia.org/wiki/Nasdaq-100",
    ]
    for url in urls:
        try:
            tables = pd.read_html(url)
            for table in tables:
                symbol_col = next(
                    (c for c in table.columns if str(c).strip().lower() in {"symbol", "ticker", "ticker symbol"}),
                    None,
                )
                if symbol_col is None:
                    continue
                symbols = [str(s).strip().upper().replace(".", "-") for s in table[symbol_col].dropna()]
                symbols = list(dict.fromkeys(s for s in symbols if s and len(s) <= 8))
                # The index has 100 companies and can have more than 100 securities
                # because multiple share classes may be included.
                if len(symbols) >= 90:
                    return symbols
        except Exception:
            pass
    raise RuntimeError("Could not load the Nasdaq-100 constituent list from either source.")


def get_scan_universe(scan_mode: str) -> tuple[list[str], str, str]:
    mode = normalize_scan_mode(scan_mode)
    if mode == "nasdaq100":
        return get_nasdaq100_tickers(), "QQQ", "NASDAQ-100"
    return get_sp500_tickers(), "SPY", "S&P 500"


def normalize_download_frame(data: pd.DataFrame) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()
    required = ["Open", "High", "Low", "Close", "Volume"]
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    missing = [c for c in required if c not in data.columns]
    if missing:
        return pd.DataFrame()
    out = data[required].copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out.apply(pd.to_numeric, errors="coerce").dropna(subset=["Open", "High", "Low", "Close"])
    out["Volume"] = out["Volume"].fillna(0)
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def download_market_data_batch(tickers: tuple[str, ...], period: str, chunk_size: int = 50) -> dict[str, pd.DataFrame]:
    """Download many symbols in chunks so a 500-stock daily scan is practical."""
    results: dict[str, pd.DataFrame] = {}
    symbols = list(tickers)
    for start in range(0, len(symbols), chunk_size):
        chunk = symbols[start:start + chunk_size]
        try:
            raw = yf.download(
                chunk,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=True,
                group_by="ticker",
            )
        except Exception:
            raw = pd.DataFrame()

        if raw.empty:
            continue

        if len(chunk) == 1:
            cleaned = normalize_download_frame(raw)
            if not cleaned.empty:
                results[chunk[0]] = cleaned
            continue

        for ticker in chunk:
            try:
                part = pd.DataFrame()
                if isinstance(raw.columns, pd.MultiIndex):
                    level0 = raw.columns.get_level_values(0)
                    level1 = raw.columns.get_level_values(1)
                    if ticker in level0:
                        part = raw[ticker].copy()
                    elif ticker in level1:
                        part = raw.xs(ticker, axis=1, level=1).copy()
                cleaned = normalize_download_frame(part)
                if not cleaned.empty:
                    results[ticker] = cleaned
            except Exception:
                continue
    return results


def evaluate_relative_strength_vs_benchmark(
    asset_ticker: str,
    asset_df: pd.DataFrame,
    benchmark_ticker: str,
    benchmark_df: pd.DataFrame,
) -> dict[str, Any]:
    """Minervini-style relative-strength trend versus SPY for stocks, BTC for ETH."""
    if asset_ticker == benchmark_ticker:
        returns = {
            "30-day return positive": safe_float(asset_df["Return_30D_Pct"].iloc[-1]) > 0,
            "90-day return positive": safe_float(asset_df["Return_90D_Pct"].iloc[-1]) > 0,
            "180-day return positive": safe_float(asset_df["Return_180D_Pct"].iloc[-1]) > 0,
        }
        passed = sum(bool(v) for v in returns.values())
        return {"label": f"{asset_ticker} momentum", "checks": returns, "passed": passed,
                "total": len(returns), "ratio_series": None, "latest_ratio": np.nan}

    aligned = pd.concat(
        [asset_df["Close"].rename("Asset"), benchmark_df["Close"].rename("Benchmark")],
        axis=1, join="inner",
    ).dropna()
    if len(aligned) < 200:
        return {"label": f"{asset_ticker} vs {benchmark_ticker}", "checks": {},
                "passed": 0, "total": 4, "ratio_series": None, "latest_ratio": np.nan}
    aligned["Ratio"] = aligned["Asset"] / aligned["Benchmark"]
    aligned["SMA_50"] = aligned["Ratio"].rolling(50).mean()
    aligned["SMA_200"] = aligned["Ratio"].rolling(200).mean()
    aligned["Return_30D"] = aligned["Ratio"].pct_change(30) * 100
    aligned["Return_90D"] = aligned["Ratio"].pct_change(90) * 100
    latest = aligned.iloc[-1]
    checks = {
        f"{asset_ticker}/{benchmark_ticker} above 50-day average": latest["Ratio"] > latest["SMA_50"],
        f"{asset_ticker}/{benchmark_ticker} above 200-day average": latest["Ratio"] > latest["SMA_200"],
        f"{asset_ticker}/{benchmark_ticker} 30-day return positive": latest["Return_30D"] > 0,
        f"{asset_ticker}/{benchmark_ticker} 90-day return positive": latest["Return_90D"] > 0,
    }
    passed = sum(bool(v) for v in checks.values())
    return {"label": f"{asset_ticker} relative strength vs {benchmark_ticker}", "checks": checks,
            "passed": passed, "total": len(checks), "ratio_series": aligned,
            "latest_ratio": safe_float(latest["Ratio"])}


def default_daily_settings() -> Settings:
    return Settings(
        history_period="3y",
        min_base_days=15,
        max_base_days=90,
        max_box_range_pct=15.0,
        test_tolerance_pct=1.5,
        minimum_high_tests=2,
        minimum_low_tests=2,
        breakout_buffer_pct=0.5,
        breakout_volume_multiple=1.5,
        dry_up_days=10,
        baseline_volume_days=30,
        dry_up_ratio_max=0.70,
        atr_days=14,
        near_high_pct=25,
        chart_days=365,
    )


def analyze_daily_symbol(
    ticker: str,
    raw_df: pd.DataFrame,
    settings: Settings,
    stock_benchmark_df: pd.DataFrame,
    btc_df: pd.DataFrame,
    stock_benchmark_ticker: str = "SPY",
) -> dict[str, Any]:
    asset_df = add_indicators(raw_df, settings)
    minimum_rows = max(221, settings.max_base_days + 2)
    if len(asset_df) < minimum_rows:
        return {"Ticker": ticker, "Error": f"Only {len(asset_df)} candles"}

    box = detect_current_box(asset_df, settings)
    trend = evaluate_trend_template(asset_df, settings)
    dry = evaluate_volume_dry_up(asset_df, settings)
    if ticker == "BTC-USD":
        rs = evaluate_relative_strength_vs_benchmark(ticker, asset_df, "BTC-USD", btc_df)
    elif ticker == "ETH-USD":
        rs = evaluate_relative_strength_vs_benchmark(ticker, asset_df, "BTC-USD", btc_df)
    else:
        rs = evaluate_relative_strength_vs_benchmark(ticker, asset_df, stock_benchmark_ticker, stock_benchmark_df)
    score = calculate_score(box, trend, dry, rs)
    if ticker in {"BTC-USD", "ETH-USD"}:
        pre_benchmark_ticker = "BTC-USD"
        pre_benchmark_df = btc_df
    else:
        pre_benchmark_ticker = stock_benchmark_ticker
        pre_benchmark_df = stock_benchmark_df
    pre_breakout = evaluate_pre_breakout_momentum(
        asset_df, pre_benchmark_df, ticker, pre_benchmark_ticker
    )
    state = box.get("state", "NO VALID BOX")
    latest_close = safe_float(asset_df["Close"].iloc[-1])
    breakout_level = safe_float(box.get("breakout_level"))
    distance_pct = (
        (breakout_level - latest_close) / breakout_level * 100
        if np.isfinite(breakout_level) and breakout_level != 0 else np.nan
    )

    probability = {"available": False, "probabilities": {}}
    if state == "BREAKOUT WATCH":
        probability = estimate_breakout_probability(asset_df, settings, box, trend, dry)

    targets = {"available": False, "targets": []}
    if state in {"BREAKOUT WATCH", "PRICE BREAKOUT / WEAK VOLUME", "CONFIRMED BREAKOUT"}:
        targets = calculate_breakout_targets(asset_df, box)

    return {
        "Ticker": ticker,
        "State": state,
        "Price": latest_close,
        "Strategy Score": score["Total"],
        "Pre-Breakout Score": pre_breakout.get("score", 0),
        "Pre-Breakout Label": pre_breakout.get("label", "N/A"),
        "Pre-Breakout Components": pre_breakout.get("components", {}),
        "RSI": pre_breakout.get("rsi", np.nan),
        "CMF": pre_breakout.get("cmf", np.nan),
        "BB Width": pre_breakout.get("bb_width", np.nan),
        "Box High": safe_float(box.get("box_high")),
        "Breakout Level": breakout_level,
        "Distance to Breakout %": distance_pct,
        "Volume Multiple": safe_float(box.get("volume_multiple")),
        "Box Quality": safe_float(box.get("quality_score")),
        "5-Day Probability %": probability.get("probabilities", {}).get(5, np.nan),
        "Probability Confidence": probability.get("confidence", "N/A") if probability.get("available") else "N/A",
        "Targets": targets.get("targets", []),
        "Latest Date": asset_df.index[-1].strftime("%Y-%m-%d"),
    }


def send_daily_scan_email(
    config: dict[str, Any],
    scan_date: str,
    alerts: list[dict[str, Any]],
    universe_label: str,
) -> None:
    confirmed = [r for r in alerts if r.get("State") == "CONFIRMED BREAKOUT"]
    watches = [r for r in alerts if r.get("State") == "BREAKOUT WATCH"]
    subject = f"[{universe_label}] Daily breakout scan: {len(confirmed)} confirmed / {len(watches)} watch — {scan_date}"
    lines = [
        f"Darvas + Minervini Daily {universe_label} + Crypto Scan — {scan_date}",
        "",
        f"Confirmed breakouts: {len(confirmed)}",
        f"Breakout watches: {len(watches)}",
        "",
    ]

    if confirmed:
        lines += ["CONFIRMED BREAKOUTS", "=" * 60]
        for r in sorted(confirmed, key=lambda x: x.get("Strategy Score", 0), reverse=True):
            lines.append(
                f"{r['Ticker']} | Price {format_currency(r['Price'])} | Overall {r['Strategy Score']}/100 | "
                f"Core {r.get('Core Score', r['Strategy Score'])}/100 | "
                f"Pre-Breakout {safe_float(r.get('Pre-Breakout Score')):.0f}/10 ({r.get('Pre-Breakout Label','N/A')}) | "
                f"Squeeze {safe_float(r.get('Short Squeeze Potential')):.0f}/100 ({r.get('Short Squeeze Label','N/A')}) | "
                f"SI Bonus +{r.get('Squeeze Bonus',0)} | Volume {r['Volume Multiple']:.2f}x | Breakout {format_currency(r['Breakout Level'])}"
            )
            for target in r.get("Targets", [])[:4]:
                lines.append(
                    f"  - {target.get('name', target.get('type', 'Target'))}: "
                    f"{format_currency(safe_float(target.get('price')))} "
                    f"({safe_float(target.get('upside_from_price_pct')):+.1f}%)"
                )
            lines.append("")

    if watches:
        lines += ["BREAKOUT WATCH", "=" * 60]
        for r in sorted(watches, key=lambda x: (safe_float(x.get("5-Day Probability %"), -1), x.get("Strategy Score", 0)), reverse=True):
            prob = r.get("5-Day Probability %")
            prob_text = f"{prob:.0f}%" if np.isfinite(safe_float(prob)) else "N/A"
            lines.append(
                f"{r['Ticker']} | Price {format_currency(r['Price'])} | Overall {r['Strategy Score']}/100 | "
                f"Core {r.get('Core Score', r['Strategy Score'])}/100 | "
                f"Pre-Breakout {safe_float(r.get('Pre-Breakout Score')):.0f}/10 ({r.get('Pre-Breakout Label','N/A')}) | "
                f"Squeeze {safe_float(r.get('Short Squeeze Potential')):.0f}/100 ({r.get('Short Squeeze Label','N/A')}) | "
                f"SI Bonus +{r.get('Squeeze Bonus',0)} | Distance {r['Distance to Breakout %']:.2f}% | 5-day probability {prob_text} | "
                f"Volume {r['Volume Multiple']:.2f}x"
            )
        lines.append("")

    lines += ["Educational signals only; not investment advice."]
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = config["sender"]
    message["To"] = config["recipient"]
    message.set_content("\n".join(lines))

    if config["use_ssl"]:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(config["smtp_host"], config["smtp_port"], context=context, timeout=30) as server:
            server.login(config["smtp_username"], config["smtp_password"])
            server.send_message(message)
    else:
        with smtplib.SMTP(config["smtp_host"], config["smtp_port"], timeout=30) as server:
            server.starttls(context=ssl.create_default_context())
            server.login(config["smtp_username"], config["smtp_password"])
            server.send_message(message)


def run_daily_market_scan(
    settings: Settings | None = None,
    send_email: bool = True,
    force: bool = False,
    progress_callback=None,
    scan_mode: str | None = None,
) -> dict[str, Any]:
    """Scan the selected stock universe plus BTC/ETH and email WATCH/CONFIRMED results."""
    settings = settings or default_daily_settings()
    scan_mode = normalize_scan_mode(scan_mode or ACTIVE_SCAN_MODE)
    stock_symbols, stock_benchmark_ticker, universe_label = get_scan_universe(scan_mode)
    scan_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    daily_state = load_daily_scan_state()
    if not force and daily_state.get("last_completed_scan_utc") == scan_date:
        return {
            "skipped": True,
            "scan_date": scan_date,
            "reason": "Daily scan already completed for this UTC date.",
            "alerts": daily_state.get("last_alerts", []),
        }

    scan_universe = list(dict.fromkeys(stock_symbols + ["BTC-USD", "ETH-USD"]))
    download_symbols = list(dict.fromkeys(scan_universe + [stock_benchmark_ticker]))

    market_data = download_market_data_batch(tuple(download_symbols), settings.history_period, chunk_size=50)
    if stock_benchmark_ticker not in market_data or "BTC-USD" not in market_data:
        raise RuntimeError(
            f"Benchmark data for {stock_benchmark_ticker} and/or BTC-USD could not be downloaded."
        )
    stock_benchmark_df = add_indicators(market_data[stock_benchmark_ticker], settings)
    btc_df = add_indicators(market_data["BTC-USD"], settings)

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for idx, ticker in enumerate(scan_universe, start=1):
        if progress_callback:
            progress_callback(idx, len(scan_universe), ticker)
        raw = market_data.get(ticker)
        if raw is None or raw.empty:
            errors.append({"Ticker": ticker, "Error": "No market data"})
            continue
        try:
            result = analyze_daily_symbol(
                ticker, raw, settings, stock_benchmark_df, btc_df, stock_benchmark_ticker
            )
            if result.get("Error"):
                errors.append({"Ticker": ticker, "Error": result["Error"]})
            else:
                results.append(result)
        except Exception as exc:
            errors.append({"Ticker": ticker, "Error": str(exc)[:180]})

    alerts = [r for r in results if r.get("State") in {"BREAKOUT WATCH", "CONFIRMED BREAKOUT"}]

    # Fetch slower short-interest fundamentals only for actionable candidates. This
    # keeps the full S&P scan practical and ensures low-short-interest breakouts
    # are never filtered out before the squeeze bonus is considered.
    for r in alerts:
        ticker = r.get("Ticker", "")
        raw = market_data.get(ticker)
        asset_df = add_indicators(raw, settings) if raw is not None and not raw.empty else None
        sq = fetch_short_squeeze_snapshot(ticker, asset_df)
        core = int(r.get("Strategy Score", 0))
        bonus = get_squeeze_bonus(safe_float(sq.get("score"))) if sq.get("available") else 0
        r["Core Score"] = core
        r["Short Squeeze Potential"] = safe_float(sq.get("score")) if sq.get("available") else np.nan
        r["Short Squeeze Label"] = sq.get("label", "N/A") if sq.get("available") else "N/A"
        r["Squeeze Bonus"] = bonus
        r["Strategy Score"] = min(100, core + bonus)

    # First grade older signals with today's downloaded candles, then store today's signals.
    reviewed_now = review_mature_signals(market_data, sessions=5)
    signals_added = record_new_signals(alerts)

    email_sent = False
    email_error = ""
    config = get_email_config()
    if send_email and alerts:
        if email_configured(config):
            try:
                send_daily_scan_email(config, scan_date, alerts, universe_label)
                email_sent = True
            except Exception as exc:
                email_error = str(exc)
        else:
            email_error = "Email settings are incomplete."

    state_for_disk = {
        "last_completed_scan_utc": scan_date,
        "last_alert_count": len(alerts),
        "last_email_sent": email_sent,
        "last_alerts": [
            {k: v for k, v in r.items() if k != "Targets"}
            for r in alerts
        ],
    }
    save_daily_scan_state(state_for_disk)
    return {
        "skipped": False,
        "scan_date": scan_date,
        "scan_mode": scan_mode,
        "universe_label": universe_label,
        "benchmark": stock_benchmark_ticker,
        "universe_count": len(scan_universe),
        "analyzed_count": len(results),
        "error_count": len(errors),
        "alerts": alerts,
        "results": results,
        "errors": errors,
        "email_sent": email_sent,
        "email_error": email_error,
        "signals_added": signals_added,
        "signals_reviewed": reviewed_now,
    }



@st.cache_data(ttl=21600, show_spinner=False)
def fetch_short_squeeze_snapshot(ticker: str, asset_df=None) -> dict[str, Any]:
    r={"applicable":True,"available":False,"score":np.nan,"label":"N/A","short_percent_float":np.nan,
       "days_to_cover":np.nan,"short_change_pct":np.nan,"relative_volume":np.nan,"components":{}}
    if ticker.endswith("-USD") or ticker in {"SPY","QQQ","DIA","IWM","MDY","RSP"}:
        r["applicable"]=False; return r
    try:
        info=yf.Ticker(ticker).info or {}
        sp=safe_float(info.get("shortPercentOfFloat")); dc=safe_float(info.get("shortRatio"))
        ss=safe_float(info.get("sharesShort")); pr=safe_float(info.get("sharesShortPriorMonth"))
        fl=safe_float(info.get("floatShares"))
        ch=(ss-pr)/pr*100 if np.isfinite(ss) and np.isfinite(pr) and pr>0 else np.nan
        rv=mo=np.nan
        if asset_df is not None and len(asset_df)>=21:
            av=safe_float(asset_df["Volume"].iloc[-21:-1].mean()); lv=safe_float(asset_df["Volume"].iloc[-1])
            rv=lv/av if np.isfinite(av) and av>0 else np.nan
            c0=safe_float(asset_df["Close"].iloc[-21]); c1=safe_float(asset_df["Close"].iloc[-1])
            mo=(c1/c0-1)*100 if np.isfinite(c0) and c0>0 else np.nan
        c={}
        if np.isfinite(sp): c["sf"]=max(0,min(100,(sp-.05)/.25*100))
        if np.isfinite(dc): c["dc"]=max(0,min(100,(dc-1)/9*100))
        if np.isfinite(ch): c["chg"]=max(0,min(100,(ch+10)/40*100))
        if np.isfinite(fl) and fl>0: c["float"]=max(0,min(100,(500e6-fl)/480e6*100))
        if np.isfinite(rv): c["rv"]=max(0,min(100,(rv-.7)/1.8*100))
        if np.isfinite(mo): c["mom"]=max(0,min(100,(mo+5)/25*100))
        wt={"sf":35,"dc":20,"chg":10,"float":10,"rv":15,"mom":10}
        aw=sum(wt[k] for k in c)
        sc=sum(c[k]*wt[k] for k in c)/aw if aw else np.nan
        r.update({"available":bool(c),"score":sc,"label":"HIGH" if sc>=70 else "MODERATE" if sc>=45 else "LOW",
                  "short_percent_float":sp,"days_to_cover":dc,"short_change_pct":ch,"relative_volume":rv,"components":c})
    except Exception as e: r["error"]=str(e)
    return r

def get_squeeze_bonus(squeeze_score: float) -> int:
    """Return a 0-5 bonus. Short interest can help a setup, but never penalize it."""
    score = safe_float(squeeze_score)
    if not np.isfinite(score):
        return 0
    if score >= 90:
        return 5
    if score >= 85:
        return 4
    if score >= 75:
        return 3
    if score >= 65:
        return 2
    if score >= 50:
        return 1
    return 0

def calculate_score_with_squeeze(base, sq):
    """Preserve the original 0-100 breakout score and add only a squeeze bonus."""
    core_total = int(base.get("Total", 0))
    squeeze_score = safe_float(sq.get("score")) if sq else np.nan
    bonus = get_squeeze_bonus(squeeze_score) if sq and sq.get("available") else 0
    x = dict(base)
    x["Core Total"] = core_total
    x["Short Squeeze Score"] = squeeze_score if np.isfinite(squeeze_score) else None
    x["Short Squeeze Bonus"] = bonus
    x["Total"] = min(100, core_total + bonus)
    return x

def render_short_squeeze_snapshot(ticker, asset_df, sq=None):
    sq = sq or fetch_short_squeeze_snapshot(ticker, asset_df)
    st.subheader("Short Squeeze Potential")
    if not sq.get("applicable", True):
        st.metric("Short Squeeze Potential", "N/A")
        return
    score = safe_float(sq.get("score"))
    st.metric(
        "Short Squeeze Potential",
        f"{score:.0f}/100" if np.isfinite(score) else "N/A",
    )


def fetch_earnings_snapshot(ticker: str) -> dict[str, Any]:
    result={"applicable":True,"next_earnings":None,"days_to_earnings":None,"history":[],
            "beats":0,"meets":0,"misses":0,"avg_surprise_pct":np.nan,"error":None}
    if ticker.endswith("-USD") or ticker in {"SPY","QQQ","DIA","IWM","MDY","RSP"}:
        result["applicable"]=False
        return result
    try:
        dates=yf.Ticker(ticker).get_earnings_dates(limit=12)
        if dates is None or dates.empty:
            return result
        now=pd.Timestamp.now(tz="UTC")
        idx=pd.to_datetime(dates.index,utc=True,errors="coerce")
        future=[d for d in idx if pd.notna(d) and d>=now]
        if future:
            nxt=min(future)
            result["next_earnings"]=nxt.date().isoformat()
            result["days_to_earnings"]=int((nxt.normalize()-now.normalize()).days)
        completed=dates.copy()
        completed["_date"]=idx
        completed=completed[completed["_date"]<now].sort_values("_date",ascending=False)
        surprises=[]
        for _,r in completed.head(4).iterrows():
            est=safe_float(r.get("EPS Estimate")); act=safe_float(r.get("Reported EPS"))
            surprise=np.nan; outcome="N/A"
            if np.isfinite(est) and np.isfinite(act):
                if abs(est)>1e-12:
                    surprise=(act-est)/abs(est)*100
                    surprises.append(surprise)
                tol=max(.005,abs(est)*.0025)
                if act-est>tol: outcome="Beat"; result["beats"]+=1
                elif act-est<-tol: outcome="Miss"; result["misses"]+=1
                else: outcome="Met"; result["meets"]+=1
            dt=r.get("_date")
            result["history"].append({"Date":dt.date().isoformat() if pd.notna(dt) else "",
                "Expected EPS":est,"Actual EPS":act,"Surprise %":surprise,"Result":outcome})
        if surprises: result["avg_surprise_pct"]=float(np.mean(surprises))
    except Exception as exc:
        result["error"]=str(exc)
    return result

def render_earnings_snapshot(ticker: str) -> None:
    er=fetch_earnings_snapshot(ticker)
    st.subheader("Earnings Context")
    if not er["applicable"]:
        st.info("Corporate earnings are not applicable to this crypto or index ETF symbol.")
        return
    days=er.get("days_to_earnings")
    avg=safe_float(er.get("avg_surprise_pct"))
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Upcoming ER",er.get("next_earnings") or "Unknown")
    c2.metric("Days to ER",str(days) if isinstance(days,int) else "Unknown")
    c3.metric("Last 4",f"{er['beats']} Beat / {er['meets']} Met / {er['misses']} Miss")
    c4.metric("Avg EPS Surprise",f"{avg:+.2f}%" if np.isfinite(avg) else "N/A")
    if isinstance(days,int) and 0<=days<=7:
        st.warning(f"Earnings are scheduled in {days} day(s).")
    if er["history"]:
        st.dataframe(pd.DataFrame(er["history"]),use_container_width=True,hide_index=True)
    elif er.get("error"):
        st.caption(f"Earnings data unavailable: {er['error']}")

def main() -> None:
    st.title("📦 Darvas + Minervini Volume Breakout Scanner")
    st.caption(f"Build: Daily {ACTIVE_SCAN_LABEL} + Crypto Breakout Scanner V4")
    st.caption(
        "Scan crypto, stocks and ETFs with the same Darvas-box, Minervini trend, "
        "volume-contraction and breakout logic."
    )

    with st.sidebar:
        st.header("Tickers")
        ticker_input = st.text_area(
            "Symbols to analyze",
            value=DEFAULT_TICKERS,
            height=120,
            help="Comma-, space-, or newline-separated Yahoo Finance symbols.",
        )
        ticker_list = parse_tickers(ticker_input)
        if not ticker_list:
            st.error("Enter at least one ticker.")
            st.stop()
        ticker = st.selectbox("Ticker to display", ticker_list)
        asset_name = display_name_for_ticker(ticker)
        st.caption(f"{len(ticker_list)} ticker(s) entered.")

        st.header("Darvas Box")
        min_base_days = st.slider("Minimum base length", 10, 40, 15)
        max_base_days = st.slider("Maximum base length", 30, 150, 90)
        max_box_range_pct = st.slider("Maximum box range (%)", 3.0, 35.0, 15.0, 0.5)
        test_tolerance_pct = st.slider("Boundary test tolerance (%)", 0.25, 5.0, 1.5, 0.25)
        minimum_high_tests = st.slider("Minimum upper-bound tests", 1, 6, 2)
        minimum_low_tests = st.slider("Minimum lower-bound tests", 1, 6, 2)

        st.header("Breakout")
        breakout_buffer_pct = st.slider("Breakout buffer (%)", 0.0, 5.0, 0.5, 0.1)
        breakout_volume_multiple = st.slider(
            "Minimum volume multiple", 1.0, 5.0, 1.5, 0.1
        )

        st.header("Volume Dry-Up")
        dry_up_days = st.slider("Recent dry-up days", 3, 30, 10)
        baseline_volume_days = st.slider("Baseline volume days", 10, 90, 30)
        dry_up_ratio_max = st.slider(
            "Maximum recent/baseline volume ratio", 0.25, 1.0, 0.70, 0.05
        )

        st.header("Trend")
        near_high_pct = st.slider("Maximum distance from 365-day high (%)", 5, 50, 25)
        atr_days = st.slider("ATR period", 5, 30, 14)
        chart_days = st.slider("Chart history days", 90, 730, 365, 30)

        st.header("Email Alerts")
        email_alerts_enabled = st.toggle("Send individual confirmed-breakout email for displayed ticker", value=False)
        email_config = get_email_config()
        if email_configured(email_config):
            st.success(f"SMTP configured for {email_config['recipient']}")
            if st.button("Send test email", use_container_width=True):
                try:
                    send_breakout_email(
                        email_config, asset_name, ticker, "test",
                        {"latest_close": np.nan, "box_high": np.nan, "breakout_level": np.nan,
                         "volume_multiple": np.nan, "base_days": 0, "quality_score": 0},
                        0, test_only=True,
                    )
                    st.success("Test email sent.")
                except Exception as exc:
                    st.error(f"Test email failed: {exc}")
        else:
            st.caption("Configure SMTP credentials in .streamlit/secrets.toml before enabling alerts.")

        refresh = st.button("Refresh market data", use_container_width=True)
        if refresh:
            st.cache_data.clear()

    settings = Settings(
        history_period="3y",
        min_base_days=min_base_days,
        max_base_days=max_base_days,
        max_box_range_pct=max_box_range_pct,
        test_tolerance_pct=test_tolerance_pct,
        minimum_high_tests=minimum_high_tests,
        minimum_low_tests=minimum_low_tests,
        breakout_buffer_pct=breakout_buffer_pct,
        breakout_volume_multiple=breakout_volume_multiple,
        dry_up_days=dry_up_days,
        baseline_volume_days=baseline_volume_days,
        dry_up_ratio_max=dry_up_ratio_max,
        atr_days=atr_days,
        near_high_pct=near_high_pct,
        chart_days=chart_days,
    )

    st.subheader(f"Daily {ACTIVE_SCAN_LABEL} + Crypto Scan")
    st.caption(
        f"URL-selected universe: {ACTIVE_SCAN_LABEL}. Stocks use {ACTIVE_BENCHMARK} for relative strength; "
        "BTC-USD and ETH-USD are also scanned. The digest includes only BREAKOUT WATCH and "
        "CONFIRMED BREAKOUT signals."
    )
    st.code(
        f"?scan={ACTIVE_SCAN_MODE}" + ("&autorun=1" if AUTO_RUN_DAILY_SCAN else ""),
        language=None,
    )
    daily_left, daily_right = st.columns([1, 3])
    with daily_left:
        run_daily_now = st.button(
            f"Run full {ACTIVE_SCAN_LABEL} scan now", type="primary", use_container_width=True
        )
    with daily_right:
        st.caption("For unattended once-daily execution, use the included GitHub Actions workflow / daily_scan.py runner.")

    # autorun=1 is useful for an automation/browser session. The normal daily-state
    # guard prevents repeated emails on Streamlit reruns unless the manual button is used.
    should_run_daily = run_daily_now or AUTO_RUN_DAILY_SCAN
    if should_run_daily:
        progress = st.progress(0.0, text=f"Starting {ACTIVE_SCAN_LABEL} daily market scan...")
        def _update_progress(done: int, total: int, symbol: str) -> None:
            progress.progress(min(done / max(total, 1), 1.0), text=f"Scanning {symbol} ({done}/{total})")
        try:
            daily_result = run_daily_market_scan(
                settings=settings,
                send_email=True,
                force=bool(run_daily_now),
                progress_callback=_update_progress,
                scan_mode=ACTIVE_SCAN_MODE,
            )
            progress.progress(1.0, text="Daily scan complete")
            if daily_result.get("skipped"):
                st.info(f"{ACTIVE_SCAN_LABEL} scan already completed for {daily_result.get('scan_date')}; no duplicate email sent.")
            alerts = daily_result.get("alerts", [])
            confirmed = sum(1 for r in alerts if r.get("State") == "CONFIRMED BREAKOUT")
            watches = sum(1 for r in alerts if r.get("State") == "BREAKOUT WATCH")
            st.success(
                f"Scanned {daily_result.get('analyzed_count', 0)} symbols: "
                f"{confirmed} confirmed breakout(s), {watches} breakout watch(es)."
            )
            if daily_result.get("email_sent"):
                st.success("Daily alert digest email sent.")
            elif daily_result.get("email_error"):
                st.warning(f"Scan completed, but email was not sent: {daily_result['email_error']}")
            if alerts:
                alert_df = pd.DataFrame([{
                    "Ticker": r.get("Ticker"),
                    "State": r.get("State"),
                    "Price": r.get("Price"),
                    "Overall Score": r.get("Strategy Score"),
                    "Core Score": r.get("Core Score", r.get("Strategy Score")),
                    "Pre-Breakout": r.get("Pre-Breakout Score"),
                    "Pre-Breakout Label": r.get("Pre-Breakout Label"),
                    "RSI": r.get("RSI"),
                    "CMF": r.get("CMF"),
                    "Squeeze": r.get("Short Squeeze Potential"),
                    "SI Bonus": r.get("Squeeze Bonus", 0),
                    "Distance %": r.get("Distance to Breakout %"),
                    "Volume x": r.get("Volume Multiple"),
                    "5-Day Prob. %": r.get("5-Day Probability %"),
                } for r in alerts])
                st.dataframe(alert_df, hide_index=True, use_container_width=True)
        except Exception as exc:
            st.error(f"Daily scan failed: {exc}")

    st.subheader("5-Day Signal Accuracy")
    accuracy_detail, accuracy_summary, score_band_summary = signal_accuracy_frames()
    pending_count = sum(1 for x in load_signal_history() if x.get("Status") == "PENDING")
    if accuracy_detail.empty:
        st.info(f"No signals have completed the 5-session review yet. Pending signals: {pending_count}.")
    else:
        c1, c2, c3 = st.columns(3)
        success_rate = 100 * accuracy_detail["Successful"].mean()
        c1.metric("Reviewed Signals", len(accuracy_detail))
        c2.metric("Overall Success Rate", f"{success_rate:.1f}%")
        c3.metric("Pending 5-Day Reviews", pending_count)
        st.caption("Accuracy by signal type")
        st.dataframe(
            accuracy_summary.style.format({
                "Success Rate %": "{:.1f}%",
                "Avg 5-Day Return %": "{:+.2f}%",
                "Avg Max Gain %": "{:+.2f}%",
                "Avg Max Drawdown %": "{:+.2f}%",
            }), hide_index=True, use_container_width=True
        )

        st.markdown("#### Accuracy by Overall Score Band")
        st.caption("This tests whether higher Overall Scores actually produce better 5-day outcomes.")
        st.dataframe(
            score_band_summary.style.format({
                "Success Rate %": "{:.1f}%",
                "Avg Overall Score": "{:.1f}",
                "Avg 5-Day Return %": "{:+.2f}%",
                "Avg Max Gain %": "{:+.2f}%",
                "Avg Max Drawdown %": "{:+.2f}%",
                "Breakout Hold Rate %": "{:.1f}%",
            }), hide_index=True, use_container_width=True
        )

        prebreak_accuracy = pre_breakout_accuracy_frame()
        if not prebreak_accuracy.empty:
            st.markdown("#### Accuracy by Pre-Breakout Momentum Score")
            st.caption(
                "This is the key validation table for deciding later whether the 0-10 "
                "leading-indicator score deserves weight in Overall Score."
            )
            st.dataframe(
                prebreak_accuracy.style.format({
                    "Success Rate %": "{:.1f}%",
                    "Avg 5-Day Return %": "{:+.2f}%",
                    "Avg Max Gain %": "{:+.2f}%",
                    "Avg Max Drawdown %": "{:+.2f}%",
                }),
                hide_index=True,
                use_container_width=True,
            )

        with st.expander("Reviewed signal history"):
            cols = [c for c in [
                "Ticker", "Signal Date", "Signal Type", "Signal Price", "Overall Score", "Score Band", "Pre-Breakout Score",
                "Review Date", "Day 5 Price", "5-Day Return %", "Max 5-Day Gain %",
                "Max 5-Day Drawdown %", "Held Breakout", "Outcome"
            ] if c in accuracy_detail.columns]
            st.dataframe(accuracy_detail[cols].sort_values("Signal Date", ascending=False), hide_index=True, use_container_width=True)

    try:
        with st.spinner(f"Loading {asset_name} ({ticker}) daily candles..."):
            raw_asset = download_market_data(ticker, settings.history_period)
            raw_btc = (
                raw_asset.copy()
                if ticker == "BTC-USD"
                else download_market_data("BTC-USD", settings.history_period)
            )
    except Exception as exc:
        st.error(f"Could not load market data: {exc}")
        st.stop()

    if raw_asset.empty or raw_btc.empty:
        st.error("No market data was returned. Try Refresh market data.")
        st.stop()

    asset_df = add_indicators(raw_asset, settings)
    btc_df = add_indicators(raw_btc, settings)

    minimum_rows = max(221, settings.max_base_days + 2)
    if len(asset_df) < minimum_rows:
        st.error(f"At least {minimum_rows} daily candles are required.")
        st.stop()

    box_result = detect_current_box(asset_df, settings)
    trend_result = evaluate_trend_template(asset_df, settings)
    dry_up_result = evaluate_volume_dry_up(asset_df, settings)
    rs_result = evaluate_relative_strength(ticker, asset_df, btc_df)
    core_score = calculate_score(box_result, trend_result, dry_up_result, rs_result)

    # For the single-ticker panel, stocks use the active index benchmark (SPY/QQQ);
    # crypto uses BTC. This matches the batch scanner's pre-breakout logic.
    if ticker in {"BTC-USD", "ETH-USD"}:
        pre_benchmark_ticker = "BTC-USD"
        pre_benchmark_df = btc_df
    else:
        try:
            raw_pre_benchmark = download_market_data(ACTIVE_BENCHMARK, settings.history_period)
            pre_benchmark_df = add_indicators(raw_pre_benchmark, settings) if not raw_pre_benchmark.empty else None
        except Exception:
            pre_benchmark_df = None
        pre_benchmark_ticker = ACTIVE_BENCHMARK

    pre_breakout = evaluate_pre_breakout_momentum(
        asset_df, pre_benchmark_df, ticker, pre_benchmark_ticker
    )

    squeeze_snapshot = fetch_short_squeeze_snapshot(ticker, asset_df)
    score = calculate_score_with_squeeze(core_score, squeeze_snapshot)

    breakout_probability = (
        estimate_breakout_probability(asset_df, settings, box_result, trend_result, dry_up_result)
        if box_result.get("state") == "BREAKOUT WATCH"
        else {"available": False, "probabilities": {}}
    )
    breakout_targets = (
        calculate_breakout_targets(asset_df, box_result)
        if box_result.get("state") in {"BREAKOUT WATCH", "PRICE BREAKOUT / WEAK VOLUME", "CONFIRMED BREAKOUT"}
        else {"available": False, "targets": []}
    )

    latest = asset_df.iloc[-1]
    prior = asset_df.iloc[-2]
    daily_change = (latest["Close"] / prior["Close"] - 1) * 100
    latest_date = asset_df.index[-1].strftime("%Y-%m-%d")

    try:
        alert_sent, alert_message = maybe_send_breakout_alert(
            enabled=email_alerts_enabled,
            config=email_config,
            asset_name=asset_name,
            ticker=ticker,
            candle_date=latest_date,
            box_result=box_result,
            strategy_score=score["Total"],
        )
        if alert_sent:
            st.success(alert_message)
    except Exception as exc:
        st.error(f"Breakout detected, but email delivery failed: {exc}")

    st.info(
        f"Latest available daily candle: **{latest_date}**. "
        "Crypto daily candles from the data source may still be incomplete before the UTC day closes."
    )

    metric_columns = st.columns(10)
    metric_columns[0].metric("Price", format_currency(latest["Close"]), f"{daily_change:.2f}%")
    metric_columns[1].metric("Overall Score", f"{score['Total']}/100")
    metric_columns[2].metric("Core Score", f"{score['Core Total']}/100")
    sq_display = safe_float(squeeze_snapshot.get("score"))
    metric_columns[3].metric(
        "Pre-Breakout",
        f"{pre_breakout.get('score', 0)}/10",
        pre_breakout.get("label", "N/A"),
    )
    metric_columns[4].metric(
        "Short Squeeze",
        f"{sq_display:.0f}/100" if np.isfinite(sq_display) else "N/A",
        f"+{score['Short Squeeze Bonus']} bonus" if score['Short Squeeze Bonus'] else None,
    )
    metric_columns[5].metric("State", box_result.get("state", "N/A"))
    metric_columns[6].metric("Box High", format_currency(safe_float(box_result.get("box_high"))))
    metric_columns[7].metric("Box Low", format_currency(safe_float(box_result.get("box_low"))))
    metric_columns[8].metric(
        "Breakout Volume",
        f"{safe_float(box_result.get('volume_multiple')):.2f}×"
        if np.isfinite(safe_float(box_result.get("volume_multiple")))
        else "N/A",
    )
    if breakout_probability.get("available"):
        metric_columns[9].metric(
            "5-Day Breakout Prob.",
            f"{breakout_probability['probabilities'].get(5, np.nan):.0f}%",
            breakout_probability.get("confidence", ""),
        )
    elif breakout_targets.get("available"):
        first_target = breakout_targets["targets"][0]
        metric_columns[9].metric(
            "Nearest Target",
            format_currency(first_target["price"]),
            f"{first_target['upside_from_price_pct']:+.1f}%",
        )
    else:
        metric_columns[9].metric("Forecast", "N/A")

    tabs = st.tabs(
        [
            "Overview",
            "Darvas Box",
            "Trend Template",
            "Volume Dry-Up",
            "Relative Strength",
            "Pre-Breakout",
            "Breakout Forecast",
            "Raw Data",
        ]
    )

    with tabs[0]:
        st.plotly_chart(
            make_price_chart(asset_df, box_result, settings, asset_name),
            use_container_width=True,
        )
        st.plotly_chart(
            make_volume_chart(asset_df, settings),
            use_container_width=True,
        )

        st.subheader("Score Breakdown")
        score_table = pd.DataFrame(
            [{"Component": key, "Points": value} for key, value in score.items() if key != "Total"]
        )
        st.dataframe(score_table, hide_index=True, use_container_width=True)

        if box_result["confirmed_breakout"]:
            st.success(
                "The latest candle meets the configured box, price-breakout and volume-confirmation rules."
            )
        elif box_result["state"] == "PRICE BREAKOUT / WEAK VOLUME":
            st.warning(
                "Price has cleared the breakout level, but volume has not reached the configured confirmation multiple."
            )
        elif box_result["state"] == "BREAKOUT WATCH":
            st.warning("Price is within 2% of the current box high.")
        elif box_result["valid"]:
            st.info("A valid box is present, but price is not yet near a confirmed breakout.")
        else:
            st.error("The selected lookback does not currently form a valid Darvas box.")

    with tabs[1]:
        details = {
            "Status": box_result["state"],
            "Detected base length": f"{box_result.get('base_days', 0)} days",
            "Base quality": f"{box_result.get('quality_score', 0):.1f}/100",
            "Box start": box_result["box_start"].strftime("%Y-%m-%d"),
            "Box end": box_result["box_end"].strftime("%Y-%m-%d"),
            "Box high": format_currency(box_result["box_high"]),
            "Box low": format_currency(box_result["box_low"]),
            "Box range": f"{box_result['box_range_pct']:.2f}%",
            "Upper-bound tests": box_result["high_tests"],
            "Lower-bound tests": box_result["low_tests"],
            "Closes contained": f"{box_result['inside_ratio'] * 100:.1f}%",
            "Breakout level": format_currency(box_result["breakout_level"]),
            "Price breakout": "Yes" if box_result["price_breakout"] else "No",
            "Volume confirmation": "Yes" if box_result["volume_breakout"] else "No",
        }
        st.dataframe(
            pd.DataFrame(details.items(), columns=["Measure", "Value"]),
            hide_index=True,
            use_container_width=True,
        )
        render_checks("Box Qualification", box_result["checks"])

    with tabs[2]:
        left, right = st.columns([1, 1])
        with left:
            render_checks(
                f"Trend Rules: {trend_result['passed']}/{trend_result['total']} Passed",
                trend_result["checks"],
            )
        with right:
            trend_values = pd.DataFrame(
                [
                    {"Measure": "50-day SMA", "Value": format_currency(trend_result["sma_50"])},
                    {"Measure": "150-day SMA", "Value": format_currency(trend_result["sma_150"])},
                    {"Measure": "200-day SMA", "Value": format_currency(trend_result["sma_200"])},
                    {
                        "Measure": "Distance from 365-day high",
                        "Value": f"{trend_result['distance_from_high_pct']:.2f}%",
                    },
                ]
            )
            st.dataframe(trend_values, hide_index=True, use_container_width=True)

    with tabs[3]:
        dryup_values = pd.DataFrame(
            [
                {
                    "Measure": "Recent/baseline dollar-volume ratio",
                    "Value": (
                        f"{dry_up_result['ratio']:.2f}"
                        if np.isfinite(dry_up_result["ratio"])
                        else "N/A"
                    ),
                },
                {
                    "Measure": "Recent average dollar volume",
                    "Value": format_currency(dry_up_result.get("recent_average", np.nan)),
                },
                {
                    "Measure": "Baseline average dollar volume",
                    "Value": format_currency(dry_up_result.get("baseline_average", np.nan)),
                },
                {
                    "Measure": "Recent average ATR %",
                    "Value": f"{dry_up_result.get('recent_atr_pct', np.nan):.2f}%",
                },
                {
                    "Measure": "Prior average ATR %",
                    "Value": f"{dry_up_result.get('prior_atr_pct', np.nan):.2f}%",
                },
                {
                    "Measure": "Volume + ATR dry-up passes",
                    "Value": "Yes" if dry_up_result["pass"] else "No",
                },
            ]
        )
        st.dataframe(dryup_values, hide_index=True, use_container_width=True)
        st.caption(
            "Dollar volume is used instead of raw share/coin volume so activity is more comparable across symbols."
        )

    with tabs[4]:
        render_checks(
            f"{rs_result['label']}: {rs_result['passed']}/{rs_result['total']} Passed",
            rs_result["checks"],
        )

        if rs_result["ratio_series"] is not None:
            ratio = rs_result["ratio_series"].tail(settings.chart_days)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=ratio.index,
                    y=ratio[rs_result["ratio_column"]],
                    mode="lines",
                    name=f"{ticker}/BTC",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=ratio.index,
                    y=ratio["SMA_50"],
                    mode="lines",
                    name="50-day average",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=ratio.index,
                    y=ratio["SMA_200"],
                    mode="lines",
                    name="200-day average",
                )
            )
            fig.update_layout(
                title=f"{ticker} Relative Strength Versus Bitcoin",
                height=450,
                yaxis_title=f"{ticker}/BTC ratio",
                legend={"orientation": "h"},
            )
            st.plotly_chart(fig, use_container_width=True)

    with tabs[5]:
        st.subheader("Pre-Breakout Momentum")
        st.caption(
            "A separate 0-10 leading-indicator score. It is not added to Overall Score yet; "
            "the 5-day validation history will tell us whether it deserves production weight."
        )
        pb_components = pre_breakout.get("components", {})
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("RS Leadership", f"{pb_components.get('Relative Strength', 0)}/3")
        c2.metric("OBV", f"{pb_components.get('OBV', 0)}/2")
        c3.metric("Compression", f"{pb_components.get('Volatility Compression', 0)}/2")
        c4.metric("RSI Momentum", f"{pb_components.get('RSI Momentum', 0)}/2")
        c5.metric("Money Flow", f"{pb_components.get('Money Flow', 0)}/1")

        detail_rows = [
            {"Measure": "Total Pre-Breakout Score", "Value": f"{pre_breakout.get('score', 0)}/10"},
            {"Measure": "Signal Band", "Value": pre_breakout.get("label", "N/A")},
            {"Measure": "RSI(14)", "Value": f"{safe_float(pre_breakout.get('rsi')):.1f}" if np.isfinite(safe_float(pre_breakout.get('rsi'))) else "N/A"},
            {"Measure": "CMF(20)", "Value": f"{safe_float(pre_breakout.get('cmf')):.3f}" if np.isfinite(safe_float(pre_breakout.get('cmf'))) else "N/A"},
            {"Measure": "Bollinger Width", "Value": f"{safe_float(pre_breakout.get('bb_width')):.2f}%" if np.isfinite(safe_float(pre_breakout.get('bb_width'))) else "N/A"},
        ]
        st.dataframe(pd.DataFrame(detail_rows), hide_index=True, use_container_width=True)

        for label, passed in pre_breakout.get("checks", {}).items():
            st.write(f"{'✅' if passed else '❌'} {label}")

    with tabs[6]:
        st.subheader("Breakout Forecast")

        # Compact upside summary for WATCH / CONFIRMED states
        forecast_targets = calculate_breakout_targets(asset_df, box_result)
        if box_result.get("state") in {
            "BREAKOUT WATCH",
            "PRICE BREAKOUT / WEAK VOLUME",
            "CONFIRMED BREAKOUT",
        } and forecast_targets.get("targets"):
            projected = forecast_targets["targets"][0]
            projected_price = safe_float(projected.get("price"))
            current_price = safe_float(box_result.get("latest_close"))
            upside_pct = (
                (projected_price / current_price - 1) * 100
                if np.isfinite(projected_price) and np.isfinite(current_price) and current_price > 0
                else np.nan
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Projected Target", format_currency(projected_price))
            c2.metric(
                "Potential Upside",
                f"{upside_pct:.2f}%"
                if np.isfinite(upside_pct)
                else "N/A",
            )
            c3.metric("Target Basis", projected.get("name", "Structural target"))

            darvas_target = next(
                (t for t in forecast_targets["targets"] if t.get("name") == "Darvas target"),
                None,
            )
            if darvas_target:
                darvas_price = safe_float(darvas_target.get("price"))
                darvas_upside = (
                    (darvas_price / current_price - 1) * 100
                    if np.isfinite(darvas_price) and np.isfinite(current_price) and current_price > 0
                    else np.nan
                )
                if np.isfinite(darvas_upside):
                    st.caption(
                        f"Darvas measured target: {format_currency(darvas_price)} "
                        f"({darvas_upside:.2f}% potential upside)"
                    )
                else:
                    st.caption(
                        f"Darvas measured target: {format_currency(darvas_price)}"
                    )

        if box_result.get("state") == "BREAKOUT WATCH":
            if breakout_probability.get("available"):
                probs = breakout_probability["probabilities"]
                cols = st.columns(3)
                cols[0].metric("Within 3 sessions", f"{probs.get(3, np.nan):.1f}%")
                cols[1].metric("Within 5 sessions", f"{probs.get(5, np.nan):.1f}%")
                cols[2].metric("Within 10 sessions", f"{probs.get(10, np.nan):.1f}%")
                st.write(
                    f"**Probability band:** {breakout_probability['probability_band']}  |  "
                    f"**Calibration confidence:** {breakout_probability['confidence']}  |  "
                    f"**Historical watch samples:** {breakout_probability['samples']}  |  "
                    f"**Effective weighted samples:** {breakout_probability['effective_samples']:.1f}"
                )
                st.caption(
                    "These are ticker-specific walk-forward historical analog probabilities. "
                    "A success requires a future close above the frozen breakout level plus the configured volume multiple. "
                    "They are estimates, not guaranteed probabilities."
                )
            else:
                st.info(breakout_probability.get("reason", "Historical probability is not available."))

            resistance_preview = find_prior_resistance_levels(asset_df, box_result)
            if resistance_preview:
                preview = pd.DataFrame([
                    {
                        "Resistance": f"R{i}",
                        "Price": level,
                        "Room from current price (%)": (level / safe_float(box_result["latest_close"]) - 1) * 100,
                    }
                    for i, level in enumerate(resistance_preview, start=1)
                ])
                st.subheader("Overhead Resistance if Breakout Occurs")
                st.dataframe(
                    preview.style.format({"Price": "${:,.2f}", "Room from current price (%)": "{:.2f}%"}),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.caption("No clear prior swing resistance above the current price was found in the available history.")

        elif box_result.get("confirmed_breakout") or box_result.get("price_breakout"):
            if breakout_targets.get("available"):
                rows = [{
                    "Target": t["name"],
                    "Price": t["price"],
                    "Upside from Current (%)": t["upside_from_price_pct"],
                    "Upside from Breakout (%)": t["upside_from_breakout_pct"],
                    "Basis": t["type"],
                } for t in breakout_targets["targets"]]
                target_df = pd.DataFrame(rows)
                st.dataframe(
                    target_df.style.format({
                        "Price": "${:,.2f}",
                        "Upside from Current (%)": "{:+.2f}%",
                        "Upside from Breakout (%)": "{:+.2f}%",
                    }),
                    hide_index=True,
                    use_container_width=True,
                )
                nearest = breakout_targets.get("nearest_resistance")
                darvas = breakout_targets.get("darvas_target")
                if nearest:
                    st.info(
                        f"Nearest historical resistance is {format_currency(nearest['price'])} "
                        f"({nearest['upside_from_price_pct']:+.1f}% from the latest close)."
                    )
                if darvas:
                    st.info(
                        f"Darvas measured-move target is {format_currency(darvas['price'])} "
                        f"({darvas['upside_from_price_pct']:+.1f}% from the latest close)."
                    )
                st.caption(
                    "Targets are structural reference levels, not a prediction that price will reach them. "
                    "Prior resistance is based on clustered historical swing highs; ATR targets adapt to current volatility."
                )
            else:
                st.info("No upside target above the current price was found from the available history and volatility inputs.")
        else:
            st.info("Forecast outputs appear when the state is BREAKOUT WATCH or a price/confirmed breakout is detected.")

    with tabs[7]:
        display_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Dollar_Volume",
            "SMA_50",
            "SMA_150",
            "SMA_200",
            "ATR_Pct",
            "RSI_14",
            "OBV",
            "BB_Width",
            "CMF_20",
            "Distance_From_365D_High_Pct",
        ]
        export = asset_df[display_columns].copy().sort_index(ascending=False)
        st.dataframe(export, use_container_width=True)
        st.download_button(
            "Download analyzed CSV",
            data=export.to_csv().encode("utf-8"),
            file_name=f"{ticker.replace('-', '_')}_darvas_minervini.csv",
            mime="text/csv",
        )

    with st.expander("How the crypto adaptation works"):
        st.markdown(
            """
            - **Dynamic Darvas component:** The app searches every candidate base between the configured
              minimum and maximum lengths, then selects the strongest current structure. Resistance and
              support come from clusters of highs and lows, so a single wick does not define the box.
            - **Breakout:** The latest close must exceed the prior box high plus a buffer.
              Volume confirmation compares the latest coin volume with the previous 20 completed days.
            - **Minervini-style trend template:** Uses 50-, 150- and 200-day averages, a rising
              200-day average, the 365-day range midpoint and distance from the 365-day high.
            - **Volume dry-up:** Compares recent average USD trading volume with an earlier baseline
              and also requires average ATR percentage to contract.
            - **Relative strength:** Every non-BTC ticker is evaluated against Bitcoin using its
              ticker/BTC ratio. Bitcoin uses positive 30-, 90- and 180-day momentum as the benchmark.
            - **Breakout Watch probability:** Replays historical watch states for the selected ticker and
              estimates 3-, 5- and 10-session confirmation rates, weighting setups that most closely match
              the current box quality, breakout distance, volume, trend, dry-up and box range.
            - **Confirmed-breakout targets:** Shows clustered prior swing resistance, a Darvas box-height
              measured move, and 1/2/3-ATR volatility extensions. These are reference levels, not guarantees.
            """
        )

    with st.expander("Email alert setup"):
        st.markdown(
            """
Create `.streamlit/secrets.toml` beside `app.py`:

```toml
[email]
smtp_host = "smtp.gmail.com"
smtp_port = 465
smtp_username = "your-email@gmail.com"
smtp_password = "your-app-password"
sender = "your-email@gmail.com"
recipient = "destination@example.com"
use_ssl = true
```

For Gmail, use an **App Password**, not your normal account password. The displayed-ticker alert sends once for each unique ticker/candle/breakout level. The daily batch scanner sends one universe-labeled digest containing all BREAKOUT WATCH and CONFIRMED BREAKOUT states. For dependable unattended daily execution, run daily_scan.py from the included GitHub Actions workflow.
            """
        )

    st.caption(
        "Educational scanner only—not investment advice. Signals can fail, volume data varies by source, "
        "and the current UTC daily candle may be incomplete."
    )

    render_short_squeeze_snapshot(ticker, asset_df, squeeze_snapshot)
    render_earnings_snapshot(ticker)


if __name__ == "__main__":
    main()
