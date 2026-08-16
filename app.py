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
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# Historical analog probability is expensive; reserve it for stronger WATCH setups.
MIN_WATCH_SCORE_FOR_PROBABILITY = 60

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


def format_direction_bias(bias: Any) -> str:
    """Return an explicit, color-coded direction label for Streamlit UI."""
    value = str(bias or "NEUTRAL").strip().upper()
    if value == "BULLISH":
        return "🟢 ↑ BULLISH"
    if value == "BEARISH":
        return "🔴 ↓ BEARISH"
    return "⚪ → NEUTRAL"


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


def detect_momentum_breakout(df: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    """Detect a breakout that is accelerating without a classical Darvas base.

    This is deliberately a parallel path, not a relaxed Darvas box.  It uses prior
    20/55-session highs as resistance, recent price acceleration, and relative
    volume.  The returned shape mirrors ``detect_current_box`` so the existing
    score/target/email machinery can reuse it.
    """
    if len(df) < 60:
        return {"valid": False, "state": "NO MOMENTUM SETUP", "reason": "At least 60 candles are required"}

    latest = df.iloc[-1]
    previous = df.iloc[-2]
    prior = df.iloc[:-1]

    latest_close = safe_float(latest["Close"])
    previous_close = safe_float(previous["Close"])
    prior_high_20 = safe_float(prior["High"].tail(20).max())
    prior_high_55 = safe_float(prior["High"].tail(55).max())
    prior_low_20 = safe_float(prior["Low"].tail(20).min())

    latest_volume = safe_float(latest["Volume"], 0.0)
    average_volume = safe_float(prior["Volume"].tail(20).mean(), 0.0)
    volume_multiple = latest_volume / average_volume if average_volume > 0 else np.nan

    ret_5 = (latest_close / safe_float(df["Close"].iloc[-6]) - 1) * 100 if len(df) >= 6 else np.nan
    ret_20 = (latest_close / safe_float(df["Close"].iloc[-21]) - 1) * 100 if len(df) >= 21 else np.nan

    # Use the nearer 20-day ceiling for entry detection while rewarding a 55-day high.
    breakout_level = prior_high_20 * (1 + settings.breakout_buffer_pct / 100)
    near_breakout_floor = prior_high_20 * 0.98
    price_breakout = bool(np.isfinite(breakout_level) and latest_close > breakout_level)
    was_below_or_near = bool(np.isfinite(prior_high_20) and previous_close <= prior_high_20 * 1.015)
    volume_breakout = bool(np.isfinite(volume_multiple) and volume_multiple >= settings.breakout_volume_multiple)

    near_55_high = bool(np.isfinite(prior_high_55) and latest_close >= prior_high_55 * 0.98)
    strong_5d = bool(np.isfinite(ret_5) and ret_5 >= 3.0)
    strong_20d = bool(np.isfinite(ret_20) and ret_20 >= 7.0)
    acceleration = strong_5d or strong_20d

    # A momentum breakout must either clear the 20-day ceiling or be within 2% of it,
    # and it must show genuine recent acceleration.
    actionable = acceleration and (price_breakout or latest_close >= near_breakout_floor)
    confirmed_breakout = bool(actionable and price_breakout and volume_breakout)
    price_only_breakout = bool(actionable and price_breakout and not volume_breakout)
    breakout_watch = bool(actionable and not price_breakout and latest_close >= near_breakout_floor)

    if confirmed_breakout:
        state = "CONFIRMED BREAKOUT"
    elif price_only_breakout:
        state = "PRICE BREAKOUT / WEAK VOLUME"
    elif breakout_watch:
        state = "BREAKOUT WATCH"
    else:
        state = "NO MOMENTUM SETUP"

    quality_score = 0.0
    if actionable:
        quality_score += 30 if price_breakout else 18
        quality_score += 20 if near_55_high else 0
        quality_score += min(25.0, max(0.0, safe_float(ret_5, 0.0)) * 2.5)
        quality_score += min(15.0, max(0.0, safe_float(ret_20, 0.0)))
        quality_score += 10 if volume_breakout else min(7.0, max(0.0, safe_float(volume_multiple, 0.0)) * 4)
        quality_score = min(100.0, quality_score)

    return {
        "valid": actionable,
        "structure_type": "MOMENTUM",
        "state": state,
        "reason": "20/55-day momentum breakout path",
        "box_high": prior_high_20,
        "box_low": prior_low_20,
        "box_range_pct": ((prior_high_20 - prior_low_20) / ((prior_high_20 + prior_low_20) / 2) * 100) if np.isfinite(prior_high_20) and np.isfinite(prior_low_20) and prior_high_20 > prior_low_20 else np.nan,
        "high_tests": 0,
        "low_tests": 0,
        "inside_ratio": np.nan,
        "breakout_level": breakout_level,
        "latest_close": latest_close,
        "previous_close": previous_close,
        "volume_multiple": volume_multiple,
        "price_breakout": price_breakout,
        "volume_breakout": volume_breakout,
        "confirmed_breakout": confirmed_breakout,
        "base_days": 20,
        "quality_score": quality_score,
        "atr_contracting": False,
        "candidate_count": 1 if actionable else 0,
        "momentum_5d_pct": ret_5,
        "momentum_20d_pct": ret_20,
        "prior_high_55": prior_high_55,
        "near_55_high": near_55_high,
        "checks": {
            "At/above 20-day resistance": bool(price_breakout or latest_close >= near_breakout_floor),
            "Recent price acceleration": acceleration,
            "Near 55-day high": near_55_high,
            "Volume confirmation": volume_breakout,
        },
    }


def _rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _adx_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([(high-low), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di-minus_di).abs() / (plus_di+minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/period, adjust=False).mean()


def evaluate_momentum_box(asset_df: pd.DataFrame, benchmark_df: pd.DataFrame | None = None, include_history: bool = True) -> dict[str, Any]:
    """Independent 0-100 momentum regime score; does not require a Darvas box."""
    if asset_df is None or asset_df.empty or len(asset_df) < 60:
        return {"available": False, "score": 0, "label": "N/A", "trajectory": "N/A", "components": {}, "measurements": {}, "history": []}

    df = asset_df.copy()
    close = df["Close"]
    rsi = _rsi_series(close)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal
    adx = _adx_series(df)
    vol_avg20 = df["Volume"].rolling(20).mean()
    rel_vol = df["Volume"] / vol_avg20.replace(0, np.nan)
    ret5 = close.pct_change(5) * 100
    ret20 = close.pct_change(20) * 100

    rs20 = np.nan
    if benchmark_df is not None and not benchmark_df.empty:
        aligned = pd.concat([close.rename("a"), benchmark_df["Close"].rename("b")], axis=1, join="inner").dropna()
        if len(aligned) >= 21:
            rs20 = ((aligned["a"].iloc[-1]/aligned["a"].iloc[-21]-1) - (aligned["b"].iloc[-1]/aligned["b"].iloc[-21]-1))*100

    # Higher-high / higher-low test using recent 5-session windows.
    hh = safe_float(df["High"].tail(5).max()) > safe_float(df["High"].iloc[-10:-5].max())
    hl = safe_float(df["Low"].tail(5).min()) > safe_float(df["Low"].iloc[-10:-5].min())

    rv = safe_float(rel_vol.iloc[-1])
    r = safe_float(rsi.iloc[-1])
    r_prev = safe_float(rsi.iloc[-3])
    mh = safe_float(macd_hist.iloc[-1])
    mh_prev = safe_float(macd_hist.iloc[-3])
    a = safe_float(adx.iloc[-1])
    a_prev = safe_float(adx.iloc[-3])
    r5 = safe_float(ret5.iloc[-1])
    r20 = safe_float(ret20.iloc[-1])

    components = {}
    # RSI 15
    components["RSI"] = 15 if (r >= 60 and r > r_prev) else 12 if r >= 60 else 9 if (r >= 50 and r > r_prev) else 5 if r >= 45 else 0
    # MACD 15
    components["MACD"] = 15 if (mh > 0 and mh > mh_prev) else 11 if mh > 0 else 6 if mh > mh_prev else 0
    # ADX 15
    components["ADX"] = 15 if (a >= 25 and a > a_prev) else 11 if a >= 25 else 7 if (a >= 20 and a > a_prev) else 3 if a >= 15 else 0
    # Relative volume 15; don't punish quiet accumulation too harshly
    components["Relative Volume"] = 15 if rv >= 1.5 else 12 if rv >= 1.2 else 9 if rv >= 1.0 else 6 if rv >= 0.8 else 2
    # Price momentum 15
    pm = 0
    if np.isfinite(r5): pm += 8 if r5 >= 5 else 6 if r5 >= 3 else 4 if r5 > 0 else 0
    if np.isfinite(r20): pm += 7 if r20 >= 10 else 5 if r20 >= 5 else 3 if r20 > 0 else 0
    components["Price Momentum"] = min(15, pm)
    # Relative strength 15
    components["Relative Strength"] = 15 if np.isfinite(rs20) and rs20 >= 7 else 12 if np.isfinite(rs20) and rs20 >= 3 else 8 if np.isfinite(rs20) and rs20 > 0 else 4 if not np.isfinite(rs20) else 0
    # Price structure 10
    components["Price Structure"] = 10 if (hh and hl) else 6 if (hh or hl) else 0

    score = int(round(sum(components.values())))
    if score >= 80: label = "EXPLOSIVE MOMENTUM"
    elif score >= 65: label = "STRONG BULLISH MOMENTUM"
    elif score >= 50: label = "BUILDING MOMENTUM"
    elif score >= 35: label = "NEUTRAL MOMENTUM"
    elif score >= 20: label = "WEAK / BEARISH MOMENTUM"
    else: label = "STRONG BEARISH MOMENTUM"

    # Full trajectory is useful in the UI but expensive across hundreds of stocks.
    # Batch mode requests only the current score and labels trajectory as CURRENT.
    hist=[]
    trajectory="CURRENT"
    if include_history:
        for offset in (3,2,1,0):
            pos = len(df) - 1 - offset
            if pos < 59:
                continue
            rr=safe_float(rsi.iloc[pos]); rrp=safe_float(rsi.iloc[max(0,pos-2)])
            hhst=safe_float(macd_hist.iloc[pos]); hp=safe_float(macd_hist.iloc[max(0,pos-2)])
            aa=safe_float(adx.iloc[pos]); ap=safe_float(adx.iloc[max(0,pos-2)])
            vv=safe_float(rel_vol.iloc[pos]); q5=safe_float(ret5.iloc[pos]); q20=safe_float(ret20.iloc[pos])
            left=max(0,pos-4); prev_left=max(0,pos-9); prev_right=max(0,pos-4)
            hh2=safe_float(df["High"].iloc[left:pos+1].max()) > safe_float(df["High"].iloc[prev_left:prev_right+1].max())
            hl2=safe_float(df["Low"].iloc[left:pos+1].min()) > safe_float(df["Low"].iloc[prev_left:prev_right+1].min())
            c1=15 if (rr>=60 and rr>rrp) else 12 if rr>=60 else 9 if (rr>=50 and rr>rrp) else 5 if rr>=45 else 0
            c2=15 if (hhst>0 and hhst>hp) else 11 if hhst>0 else 6 if hhst>hp else 0
            c3=15 if (aa>=25 and aa>ap) else 11 if aa>=25 else 7 if (aa>=20 and aa>ap) else 3 if aa>=15 else 0
            c4=15 if vv>=1.5 else 12 if vv>=1.2 else 9 if vv>=1.0 else 6 if vv>=0.8 else 2
            c5=(8 if q5>=5 else 6 if q5>=3 else 4 if q5>0 else 0)+(7 if q20>=10 else 5 if q20>=5 else 3 if q20>0 else 0)
            c6=components["Relative Strength"]
            c7=10 if (hh2 and hl2) else 6 if (hh2 or hl2) else 0
            hist.append({"Date": df.index[pos].strftime("%Y-%m-%d"), "Score": int(round(c1+c2+c3+c4+c5+c6+c7))})
        trajectory="STABLE →"
        if len(hist)>=2:
            delta=hist[-1]["Score"]-hist[0]["Score"]
            trajectory="ACCELERATING ↑" if delta>=10 else "IMPROVING ↑" if delta>=4 else "DECELERATING ↓" if delta<=-10 else "SOFTENING ↓" if delta<=-4 else "STABLE →"

    measurements={
        "RSI (14)": r, "MACD histogram": mh, "ADX (14)": a, "Relative volume": rv,
        "5-day return %": r5, "20-day return %": r20, "Relative strength vs benchmark (20D %)": rs20,
        "Higher high": hh, "Higher low": hl,
    }
    return {"available": True, "score": score, "label": label, "trajectory": trajectory, "components": components, "measurements": measurements, "history": hist}


def calculate_momentum_targets(asset_df: pd.DataFrame, momentum_box: dict[str, Any], horizon_sessions: int = 5) -> dict[str, Any]:
    """Estimate a momentum-skewed high/low price envelope over the next few sessions.

    Uses current ATR as the volatility unit and skews the upside/downside multiples
    according to the 0-100 Momentum Box score and its trajectory. These are reference
    levels for scenario planning, not guaranteed forecasts.
    """
    if asset_df is None or asset_df.empty or not momentum_box.get("available"):
        return {"available": False}

    price = safe_float(asset_df["Close"].iloc[-1])
    atr = safe_float(asset_df["ATR"].iloc[-1]) if "ATR" in asset_df.columns else np.nan
    score = safe_float(momentum_box.get("score"), 50.0)
    if not (np.isfinite(price) and price > 0 and np.isfinite(atr) and atr > 0):
        return {"available": False}

    strength = max(0.0, min(1.0, score / 100.0))
    trajectory = str(momentum_box.get("trajectory", ""))

    # Strong momentum widens the upside envelope and tightens the downside envelope.
    # Weak momentum does the opposite. The trajectory contributes a modest adjustment.
    up_mult = 1.15 + 1.85 * strength
    down_mult = 2.25 - 1.20 * strength
    if "ACCELERATING" in trajectory:
        up_mult += 0.35
        down_mult -= 0.15
    elif "IMPROVING" in trajectory:
        up_mult += 0.20
        down_mult -= 0.10
    elif "DECELERATING" in trajectory:
        up_mult -= 0.25
        down_mult += 0.25
    elif "SOFTENING" in trajectory:
        up_mult -= 0.15
        down_mult += 0.15

    # Normalize gently for the requested horizon. Five sessions is the baseline.
    horizon_scale = max(0.5, min(2.0, (max(1, horizon_sessions) / 5.0) ** 0.5))
    up_mult *= horizon_scale
    down_mult *= horizon_scale
    up_mult = max(0.75, up_mult)
    down_mult = max(0.75, down_mult)

    high_target = price + atr * up_mult
    low_target = max(0.01, price - atr * down_mult)

    # Nearby structure is informative, but momentum/ATR remains the primary driver.
    prior20_high = safe_float(asset_df["High"].iloc[-21:-1].max()) if len(asset_df) >= 21 else np.nan
    prior20_low = safe_float(asset_df["Low"].iloc[-21:-1].min()) if len(asset_df) >= 21 else np.nan

    return {
        "available": True,
        "horizon_sessions": int(horizon_sessions),
        "current_price": price,
        "atr": atr,
        "high_target": high_target,
        "low_target": low_target,
        "high_upside_pct": (high_target / price - 1.0) * 100.0,
        "low_downside_pct": (low_target / price - 1.0) * 100.0,
        "up_atr_multiple": up_mult,
        "down_atr_multiple": down_mult,
        "prior_20d_high": prior20_high,
        "prior_20d_low": prior20_low,
    }



def _compression_points(ratio: float, bands: list[tuple[float, int]], default: int = 0) -> int:
    """Return points for a lower-is-tighter ratio using ordered threshold bands."""
    if not np.isfinite(ratio):
        return default
    for threshold, points in bands:
        if ratio <= threshold:
            return points
    return default


def evaluate_momentum_compression(
    asset_df: pd.DataFrame,
    momentum_box: dict[str, Any],
) -> dict[str, Any]:
    """Score 0-100 momentum compression independently of a Darvas box.

    Compression measures stored-energy conditions: contracting ATR/range/volume,
    tightly clustered closes, and momentum that is being retained while price
    volatility contracts. Direction bias is deliberately separate from compression.
    """
    if asset_df is None or asset_df.empty or len(asset_df) < 45 or not momentum_box.get("available"):
        return {"available": False, "score": 0, "status": "N/A", "bias": "NEUTRAL", "components": {}, "measurements": {}}

    df = asset_df.copy()
    recent = df.tail(5)
    prior = df.iloc[-25:-5]
    if len(prior) < 15:
        return {"available": False, "score": 0, "status": "N/A", "bias": "NEUTRAL", "components": {}, "measurements": {}}

    recent_atr = safe_float(recent["ATR_Pct"].mean())
    prior_atr = safe_float(prior["ATR_Pct"].mean())
    atr_ratio = recent_atr / prior_atr if np.isfinite(prior_atr) and prior_atr > 0 else np.nan

    recent_mid = safe_float(recent["Close"].mean())
    prior_mid = safe_float(prior["Close"].mean())
    recent_range_pct = ((safe_float(recent["High"].max()) - safe_float(recent["Low"].min())) / recent_mid * 100) if recent_mid > 0 else np.nan
    prior_range_pct = ((safe_float(prior["High"].max()) - safe_float(prior["Low"].min())) / prior_mid * 100) if prior_mid > 0 else np.nan
    range_ratio = recent_range_pct / prior_range_pct if np.isfinite(prior_range_pct) and prior_range_pct > 0 else np.nan

    recent_vol = safe_float(recent["Volume"].mean())
    prior_vol = safe_float(prior["Volume"].mean())
    volume_ratio = recent_vol / prior_vol if prior_vol > 0 else np.nan

    close_std_recent = safe_float(recent["Close"].std(ddof=0))
    close_std_prior = safe_float(prior["Close"].std(ddof=0))
    recent_cv = close_std_recent / recent_mid if recent_mid > 0 else np.nan
    prior_cv = close_std_prior / prior_mid if prior_mid > 0 else np.nan
    cluster_ratio = recent_cv / prior_cv if np.isfinite(prior_cv) and prior_cv > 0 else np.nan

    momentum_score = safe_float(momentum_box.get("score"), 0.0)
    trajectory = str(momentum_box.get("trajectory", ""))

    components = {
        "ATR Contraction": _compression_points(atr_ratio, [(0.70, 25), (0.82, 21), (0.92, 16), (1.00, 10)], 2),
        "Price Range Compression": _compression_points(range_ratio, [(0.45, 25), (0.60, 21), (0.75, 16), (0.90, 10)], 2),
        "Volume Contraction": _compression_points(volume_ratio, [(0.60, 20), (0.75, 16), (0.90, 11), (1.00, 6)], 1),
        "Momentum Retention": 20 if momentum_score >= 65 else 16 if momentum_score >= 55 else 12 if momentum_score >= 45 else 7 if momentum_score >= 35 else 2,
        "Close Clustering": _compression_points(cluster_ratio, [(0.50, 10), (0.70, 8), (0.85, 6), (1.00, 4)], 1),
    }
    if "ACCELERATING" in trajectory or "IMPROVING" in trajectory:
        components["Momentum Retention"] = min(20, components["Momentum Retention"] + 2)
    elif "DECELERATING" in trajectory or "SOFTENING" in trajectory:
        components["Momentum Retention"] = max(0, components["Momentum Retention"] - 2)

    score = int(round(sum(components.values())))
    expanding = bool(
        np.isfinite(atr_ratio) and np.isfinite(range_ratio)
        and atr_ratio > 1.08 and range_ratio > 1.05
    )
    if expanding:
        status = "MOMENTUM EXPANDING"
    elif score >= 75:
        status = "COILED MOMENTUM"
    elif score >= 60:
        status = "MOMENTUM COMPRESSION BUILDING"
    elif score >= 45:
        status = "MILD COMPRESSION"
    else:
        status = "NORMAL / LOOSE MOMENTUM"

    rsi = safe_float(momentum_box.get("measurements", {}).get("RSI (14)"))
    macd_hist = safe_float(momentum_box.get("measurements", {}).get("MACD histogram"))
    if momentum_score >= 50 and (not np.isfinite(rsi) or rsi >= 48) and (not np.isfinite(macd_hist) or macd_hist >= 0):
        bias = "BULLISH"
    elif momentum_score < 35 or (np.isfinite(rsi) and rsi < 42 and np.isfinite(macd_hist) and macd_hist < 0):
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    return {
        "available": True,
        "score": score,
        "status": status,
        "bias": bias,
        "components": components,
        "measurements": {
            "Recent/Prior ATR Ratio": atr_ratio,
            "Recent/Prior Range Ratio": range_ratio,
            "Recent/Prior Volume Ratio": volume_ratio,
            "Close Clustering Ratio": cluster_ratio,
        },
    }


def evaluate_darvas_compression(
    asset_df: pd.DataFrame,
    darvas_box: dict[str, Any],
    momentum_box: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    """Score a valid Darvas base for pre-breakout compression pressure (0-100)."""
    if asset_df is None or asset_df.empty or not darvas_box.get("valid"):
        return {"available": False, "score": 0, "status": "NO ACTIVE DARVAS COMPRESSION", "bias": "NEUTRAL", "components": {}, "measurements": {}}

    box_range = safe_float(darvas_box.get("box_range_pct"))
    base_days = safe_float(darvas_box.get("base_days"), 0.0)
    box_high = safe_float(darvas_box.get("box_high"))
    box_low = safe_float(darvas_box.get("box_low"))
    close = safe_float(asset_df["Close"].iloc[-1])

    tight_ratio = box_range / max(settings.max_box_range_pct, 0.01) if np.isfinite(box_range) else np.nan
    tight_points = _compression_points(tight_ratio, [(0.45, 30), (0.60, 26), (0.75, 21), (0.90, 15), (1.00, 10)], 2)
    maturity_points = 15 if base_days >= 30 else 12 if base_days >= 20 else 9 if base_days >= 12 else 5

    completed = asset_df.iloc[:-1]
    recent_vol = safe_float(completed["Volume"].tail(5).mean())
    baseline_vol = safe_float(completed["Volume"].iloc[-25:-5].mean()) if len(completed) >= 25 else safe_float(completed["Volume"].tail(20).mean())
    vol_ratio = recent_vol / baseline_vol if baseline_vol > 0 else np.nan
    volume_points = _compression_points(vol_ratio, [(0.60, 20), (0.75, 17), (0.90, 13), (1.00, 8)], 2)

    distance_to_high = ((box_high - close) / box_high * 100) if np.isfinite(box_high) and box_high > 0 else np.nan
    proximity_points = 20 if np.isfinite(distance_to_high) and distance_to_high <= 1 else 17 if np.isfinite(distance_to_high) and distance_to_high <= 2 else 12 if np.isfinite(distance_to_high) and distance_to_high <= 4 else 6 if np.isfinite(distance_to_high) and distance_to_high <= 7 else 2

    mb_score = safe_float(momentum_box.get("score"), 0.0)
    momentum_points = 15 if mb_score >= 65 else 12 if mb_score >= 55 else 9 if mb_score >= 45 else 5 if mb_score >= 35 else 1

    components = {
        "Box Tightness": tight_points,
        "Box Maturity": maturity_points,
        "Volume Compression": volume_points,
        "Proximity to Box High": proximity_points,
        "Momentum Confirmation": momentum_points,
    }
    score = int(round(sum(components.values())))
    if score >= 75:
        status = "COILED / BREAKOUT WATCH"
    elif score >= 60:
        status = "DARVAS COMPRESSION BUILDING"
    elif score >= 45:
        status = "MILD DARVAS COMPRESSION"
    else:
        status = "NORMAL / LOOSE DARVAS BOX"

    midpoint = (box_high + box_low) / 2 if np.isfinite(box_high) and np.isfinite(box_low) else np.nan
    if mb_score >= 50 and np.isfinite(midpoint) and close >= midpoint:
        bias = "BULLISH"
    elif mb_score < 35 and np.isfinite(midpoint) and close < midpoint:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    return {
        "available": True,
        "score": score,
        "status": status,
        "bias": bias,
        "components": components,
        "measurements": {
            "Box Range %": box_range,
            "Base Days": base_days,
            "Recent/Baseline Volume Ratio": vol_ratio,
            "Distance to Box High %": distance_to_high,
        },
    }


def _cheap_compression_gate(asset_df: pd.DataFrame, settings: Settings) -> bool:
    """Cheap fast-batch gate that allows tight bases through to full Darvas search."""
    if asset_df is None or len(asset_df) < 30:
        return False
    prior = asset_df.iloc[:-1].tail(20)
    mid = safe_float(prior["Close"].mean())
    if not np.isfinite(mid) or mid <= 0:
        return False
    range_pct = (safe_float(prior["High"].max()) - safe_float(prior["Low"].min())) / mid * 100
    recent_atr = safe_float(prior["ATR_Pct"].tail(5).mean())
    earlier_atr = safe_float(prior["ATR_Pct"].head(15).mean())
    return bool(
        np.isfinite(range_pct)
        and range_pct <= settings.max_box_range_pct * 1.30
        and np.isfinite(recent_atr) and np.isfinite(earlier_atr)
        and recent_atr <= earlier_atr * 1.02
    )


def compression_category_flags(row: dict[str, Any]) -> dict[str, bool]:
    """Centralized category rules used by both UI and email."""
    d_score = safe_float(row.get("Darvas Breakout Pressure"), -1)
    m_score = safe_float(row.get("Momentum Compression Score"), -1)
    mb_score = safe_float(row.get("Momentum Box Score"), -1)
    sq_score = safe_float(row.get("Short Squeeze Potential"), -1)
    d_bull = str(row.get("Darvas Compression Bias", "")).upper() == "BULLISH"
    m_bull = str(row.get("Momentum Compression Bias", "")).upper() == "BULLISH"

    darvas = d_score >= 65
    momentum = m_score >= 65
    dual = darvas and momentum and d_bull and m_bull
    squeeze_momentum = sq_score >= 70 and momentum and m_bull and mb_score >= 50
    triple = dual and sq_score >= 70 and mb_score >= 50
    return {
        "darvas": darvas,
        "momentum": momentum,
        "dual": dual,
        "squeeze_momentum": squeeze_momentum,
        "triple": triple,
    }

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



def score_resistance_strength(
    df: pd.DataFrame,
    level: float,
    box_result: dict[str, Any],
    tolerance_pct: float = 1.5,
) -> dict[str, Any]:
    """Estimate historical strength of one resistance level without altering target discovery."""
    empty = {
        "score": np.nan, "rating": "N/A", "tests": 0, "last_test": "N/A",
        "avg_rejection_pct": np.nan, "avg_volume_multiple": np.nan, "confluence": 0,
    }
    if df is None or df.empty or not np.isfinite(safe_float(level)):
        return empty

    history = df.copy()
    box_start = box_result.get("box_start")
    if box_start is not None:
        try:
            history = history.loc[history.index < box_start]
        except Exception:
            pass
    if len(history) < 10:
        return empty

    lvl = float(level)
    tol = max(0.0025, tolerance_pct / 100.0)
    highs = pd.to_numeric(history["High"], errors="coerce")
    volumes = pd.to_numeric(history["Volume"], errors="coerce").fillna(0.0)
    touch_mask = ((highs - lvl).abs() / max(lvl, 1e-9)) <= tol
    touch_positions = np.flatnonzero(touch_mask.to_numpy())
    if len(touch_positions) == 0:
        return empty

    # De-duplicate adjacent candles from the same test episode.
    episodes = []
    for pos in touch_positions:
        if not episodes or pos - episodes[-1] > 3:
            episodes.append(int(pos))
    tests = len(episodes)

    rejection_values = []
    volume_values = []
    for pos in episodes:
        after = history.iloc[pos + 1:min(len(history), pos + 6)]
        if not after.empty:
            future_low = safe_float(pd.to_numeric(after["Low"], errors="coerce").min())
            if np.isfinite(future_low) and lvl > 0:
                rejection_values.append(max(0.0, (lvl - future_low) / lvl * 100.0))
        prior_vol = safe_float(volumes.iloc[max(0, pos - 20):pos].mean(), 0.0)
        current_vol = safe_float(volumes.iloc[pos], 0.0)
        if prior_vol > 0:
            volume_values.append(current_vol / prior_vol)

    avg_rejection = float(np.mean(rejection_values)) if rejection_values else 0.0
    avg_vm = float(np.mean(volume_values)) if volume_values else np.nan
    last_pos = episodes[-1]
    sessions_ago = max(0, len(history) - 1 - last_pos)
    span_sessions = episodes[-1] - episodes[0] if tests > 1 else 0

    touch_points = min(25.0, tests / 4.0 * 25.0)
    rejection_points = min(20.0, avg_rejection / 5.0 * 20.0)
    volume_points = min(15.0, max(0.0, (safe_float(avg_vm, 0.0) - 0.5) * 15.0))
    recency_points = 10.0 if sessions_ago <= 20 else 8.0 if sessions_ago <= 60 else 5.0 if sessions_ago <= 120 else 2.0 if sessions_ago <= 250 else 0.0
    significance_points = 15.0 if (tests >= 3 and span_sessions >= 60) else 10.0 if (tests >= 2 and span_sessions >= 30) else 5.0 if tests >= 2 else 0.0

    confluence = 0
    for window in (90, 180, 365):
        if len(history) >= 20:
            major_high = safe_float(pd.to_numeric(history["High"].tail(window), errors="coerce").max())
            if np.isfinite(major_high) and major_high > 0 and abs(lvl - major_high) / major_high <= 0.015:
                confluence += 1
    confluence_points = min(15.0, confluence * 5.0)

    score = int(round(min(100.0, touch_points + rejection_points + volume_points + recency_points + significance_points + confluence_points)))
    rating = "VERY STRONG" if score >= 80 else "STRONG" if score >= 60 else "MODERATE" if score >= 40 else "WEAK"
    last_test = history.index[last_pos]
    return {
        "score": score,
        "rating": rating,
        "tests": tests,
        "last_test": last_test.strftime("%Y-%m-%d") if hasattr(last_test, "strftime") else str(last_test),
        "avg_rejection_pct": avg_rejection,
        "avg_volume_multiple": avg_vm,
        "confluence": confluence,
    }


def add_resistance_break_scores(
    targets: list[dict[str, Any]],
    momentum_box: dict[str, Any] | None,
    momentum_compression: dict[str, Any] | None,
    short_squeeze_score: float = np.nan,
    current_volume_multiple: float = np.nan,
) -> list[dict[str, Any]]:
    """Add heuristic 0-100 barrier-break scores to resistance targets."""
    mb = safe_float((momentum_box or {}).get("score"), 0.0)
    mc = safe_float((momentum_compression or {}).get("score"), 0.0)
    squeeze = safe_float(short_squeeze_score, 0.0)
    vol_mult = safe_float(current_volume_multiple, 0.0)
    volume_score = min(100.0, max(0.0, vol_mult / 2.0 * 100.0))
    bias = str((momentum_compression or {}).get("bias", "NEUTRAL")).upper()
    for target in targets:
        if target.get("type") != "Prior swing resistance":
            continue
        strength = safe_float(target.get("resistance_strength"), 50.0)
        score = 0.35 * (100.0 - strength) + 0.25 * mb + 0.15 * mc + 0.15 * squeeze + 0.10 * volume_score
        if bias == "BULLISH":
            score += 5.0
        elif bias == "BEARISH":
            score -= 10.0
        score = int(round(max(0.0, min(100.0, score))))
        target["resistance_break_score"] = score
        target["resistance_break_label"] = "HIGH" if score >= 75 else "FAVORABLE" if score >= 60 else "MIXED" if score >= 45 else "DIFFICULT"
    return targets

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
        strength = score_resistance_strength(df, level, box_result)
        targets.append({
            "name": f"Resistance R{idx}",
            "price": level,
            "type": "Prior swing resistance",
            "resistance_strength": strength.get("score"),
            "resistance_rating": strength.get("rating"),
            "resistance_tests": strength.get("tests"),
            "resistance_last_test": strength.get("last_test"),
            "resistance_avg_rejection_pct": strength.get("avg_rejection_pct"),
            "resistance_test_volume_multiple": strength.get("avg_volume_multiple"),
            "resistance_confluence_count": strength.get("confluence"),
        })

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

    box_start_value = box_result.get("box_start")
    box_end_value = box_result.get("box_end")
    box_low_value = safe_float(box_result.get("box_low"))
    box_high_value = safe_float(box_result.get("box_high"))
    breakout_level_value = safe_float(box_result.get("breakout_level"))
    is_darvas_structure = str(box_result.get("structure_type", "DARVAS")).upper() == "DARVAS"

    if (
        is_darvas_structure
        and box_start_value is not None
        and (box_start_value in visible.index or box_end_value in visible.index)
        and np.isfinite(box_low_value)
        and np.isfinite(box_high_value)
    ):
        box_start = max(box_start_value, visible.index.min())
        fig.add_shape(
            type="rect",
            x0=box_start,
            x1=visible.index.max(),
            y0=box_low_value,
            y1=box_high_value,
            line={"width": 1.5, "dash": "dash"},
            fillcolor="rgba(120,120,120,0.10)",
        )
        if np.isfinite(breakout_level_value):
            fig.add_hline(
                y=breakout_level_value,
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
    fast_batch: bool = False,
) -> dict[str, Any]:
    """Analyze one symbol.

    fast_batch=True is optimized for large-universe scans:
      * evaluate the cheap momentum path first;
      * only run the exhaustive Darvas-base search when price is near recent resistance;
      * defer historical analog probability and target generation to stage 2.
    The interactive single-ticker UI keeps the full analysis path.
    """
    asset_df = add_indicators(raw_df, settings)
    minimum_rows = max(221, settings.max_base_days + 2)
    if len(asset_df) < minimum_rows:
        return {"Ticker": ticker, "Error": f"Only {len(asset_df)} candles"}

    momentum_structure = detect_momentum_breakout(asset_df, settings)
    momentum_actionable = momentum_structure.get("state") in {
        "BREAKOUT WATCH", "PRICE BREAKOUT / WEAK VOLUME", "CONFIRMED BREAKOUT"
    }

    # Fast universe scans should not spend 15..90 base-length iterations on stocks
    # that are nowhere near recent resistance. A 5% cushion is deliberately loose
    # so classical setups are not screened out merely because of a small wick.
    run_darvas = True
    if fast_batch and not momentum_actionable:
        prior = asset_df.iloc[:-1]
        prior_high_20 = safe_float(prior["High"].tail(20).max())
        prior_high_55 = safe_float(prior["High"].tail(55).max())
        latest_close_gate = safe_float(asset_df["Close"].iloc[-1])
        near_20 = np.isfinite(prior_high_20) and latest_close_gate >= prior_high_20 * 0.95
        near_55 = np.isfinite(prior_high_55) and latest_close_gate >= prior_high_55 * 0.95
        compressed_gate = _cheap_compression_gate(asset_df, settings)
        run_darvas = bool(near_20 or near_55 or compressed_gate)

    if run_darvas:
        darvas_box = detect_current_box(asset_df, settings)
    else:
        darvas_box = {
            "valid": False,
            "state": "NO VALID BOX",
            "reason": "Fast batch prefilter: price is not near recent resistance",
            "box_high": np.nan,
            "box_low": np.nan,
            "quality_score": 0.0,
            "breakout_level": np.nan,
            "volume_multiple": np.nan,
            "confirmed_breakout": False,
            "price_breakout": False,
        }

    darvas_actionable = darvas_box.get("state") in {
        "BREAKOUT WATCH", "PRICE BREAKOUT / WEAK VOLUME", "CONFIRMED BREAKOUT"
    }
    if darvas_actionable:
        box = dict(darvas_box)
        box["structure_type"] = "DARVAS"
    elif momentum_actionable:
        box = dict(momentum_structure)
        box["structure_type"] = "MOMENTUM"
    else:
        box = dict(darvas_box)
        box["structure_type"] = "DARVAS" if darvas_box.get("valid") else "NONE"

    trend = evaluate_trend_template(asset_df, settings)
    dry = evaluate_volume_dry_up(asset_df, settings)
    if ticker == "BTC-USD":
        rs = evaluate_relative_strength_vs_benchmark(ticker, asset_df, "BTC-USD", btc_df)
        mb_benchmark = btc_df
    elif ticker == "ETH-USD":
        rs = evaluate_relative_strength_vs_benchmark(ticker, asset_df, "BTC-USD", btc_df)
        mb_benchmark = btc_df
    else:
        rs = evaluate_relative_strength_vs_benchmark(
            ticker, asset_df, stock_benchmark_ticker, stock_benchmark_df
        )
        mb_benchmark = stock_benchmark_df

    score = calculate_score(box, trend, dry, rs)
    pre_breakout = evaluate_pre_breakout_momentum(
        asset_df, mb_benchmark, ticker,
        "BTC-USD" if ticker.endswith("-USD") else stock_benchmark_ticker,
    )
    state = box.get("state", "NO VALID BOX")
    latest_close = safe_float(asset_df["Close"].iloc[-1])
    breakout_level = safe_float(box.get("breakout_level"))
    distance_pct = (
        (breakout_level - latest_close) / breakout_level * 100
        if np.isfinite(breakout_level) and breakout_level != 0 else np.nan
    )

    # Momentum Box is useful across the whole universe, but batch mode avoids the
    # trajectory/history recomputation. The UI gets the full version later.
    momentum_box = evaluate_momentum_box(asset_df, mb_benchmark, include_history=not fast_batch)
    momentum_targets = calculate_momentum_targets(asset_df, momentum_box, horizon_sessions=5)
    momentum_compression = evaluate_momentum_compression(asset_df, momentum_box)
    darvas_compression = evaluate_darvas_compression(asset_df, darvas_box, momentum_box, settings)

    probability = {"available": False, "probabilities": {}}
    targets = {"available": False, "targets": []}
    if not fast_batch:
        if state == "BREAKOUT WATCH":
            probability = estimate_breakout_probability(asset_df, settings, box, trend, dry)
        if state in {"BREAKOUT WATCH", "PRICE BREAKOUT / WEAK VOLUME", "CONFIRMED BREAKOUT"}:
            targets = calculate_breakout_targets(asset_df, box)

    return {
        "Ticker": ticker,
        "State": state,
        "Breakout Type": box.get("structure_type", "NONE"),
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
        "Momentum Box Score": int(momentum_box.get("score", 0)) if momentum_box.get("available") else 0,
        "Momentum Box Label": momentum_box.get("label", "N/A"),
        "Momentum Box Trajectory": momentum_box.get("trajectory", "N/A"),
        "Momentum High Target": safe_float(momentum_targets.get("high_target")),
        "Momentum Low Target": safe_float(momentum_targets.get("low_target")),
        "Momentum High Upside %": safe_float(momentum_targets.get("high_upside_pct")),
        "Momentum Low Downside %": safe_float(momentum_targets.get("low_downside_pct")),
        "Momentum Compression Score": int(momentum_compression.get("score", 0)) if momentum_compression.get("available") else 0,
        "Momentum Compression Status": momentum_compression.get("status", "N/A"),
        "Momentum Compression Bias": momentum_compression.get("bias", "NEUTRAL"),
        "Darvas Breakout Pressure": int(darvas_compression.get("score", 0)) if darvas_compression.get("available") else 0,
        "Darvas Compression Status": darvas_compression.get("status", "N/A"),
        "Darvas Compression Bias": darvas_compression.get("bias", "NEUTRAL"),
        "Latest Date": asset_df.index[-1].strftime("%Y-%m-%d"),
    }


def send_daily_scan_email(
    config: dict[str, Any],
    scan_date: str,
    alerts: list[dict[str, Any]],
    universe_label: str,
) -> None:
    confirmed = [r for r in alerts if r.get("State") == "CONFIRMED BREAKOUT"]
    watches = [r for r in alerts if r.get("State") in {"BREAKOUT WATCH", "PRICE BREAKOUT / WEAK VOLUME"}]

    # High-conviction squeeze setup: strong strategy score + strong short-squeeze score.
    # Keep this as a separate ranked view; do not remove these names from the normal
    # Confirmed/Watch sections below.
    HIGH_OVERALL_SCORE = 80
    HIGH_SQUEEZE_SCORE = 70
    high_squeeze = [
        r for r in alerts
        if safe_float(r.get("Strategy Score"), -1) >= HIGH_OVERALL_SCORE
        and safe_float(r.get("Short Squeeze Potential"), -1) >= HIGH_SQUEEZE_SCORE
    ]
    high_squeeze = sorted(
        high_squeeze,
        key=lambda x: (
            safe_float(x.get("Strategy Score"), -1),
            safe_float(x.get("Short Squeeze Potential"), -1),
            safe_float(x.get("5-Day Probability %"), -1),
        ),
        reverse=True,
    )

    darvas_compressed = sorted(
        [r for r in alerts if compression_category_flags(r)["darvas"]],
        key=lambda x: safe_float(x.get("Darvas Breakout Pressure"), -1),
        reverse=True,
    )
    momentum_compressed = sorted(
        [r for r in alerts if compression_category_flags(r)["momentum"]],
        key=lambda x: safe_float(x.get("Momentum Compression Score"), -1),
        reverse=True,
    )
    dual_coiled = sorted(
        [r for r in alerts if compression_category_flags(r)["dual"]],
        key=lambda x: (safe_float(x.get("Darvas Breakout Pressure"), -1), safe_float(x.get("Momentum Compression Score"), -1)),
        reverse=True,
    )
    squeeze_momentum = sorted(
        [r for r in alerts if compression_category_flags(r)["squeeze_momentum"]],
        key=lambda x: safe_float(x.get("Squeeze-Momentum Score"), -1),
        reverse=True,
    )
    triple_alignment = sorted(
        [r for r in alerts if compression_category_flags(r)["triple"]],
        key=lambda x: (safe_float(x.get("Squeeze-Momentum Score"), -1), safe_float(x.get("Darvas Breakout Pressure"), -1)),
        reverse=True,
    )

    subject = (
        f"[{universe_label}] Daily breakout scan: {len(confirmed)} confirmed / "
        f"{len(watches)} watch / {len(high_squeeze)} high-squeeze — {scan_date}"
    )
    lines = [
        f"Darvas + Minervini Daily {universe_label} + Crypto Scan — {scan_date}",
        "",
        f"Confirmed breakouts: {len(confirmed)}",
        f"Breakout watches: {len(watches)}",
        f"High Squeeze + High Overall Score: {len(high_squeeze)} "
        f"(Overall >= {HIGH_OVERALL_SCORE}, Squeeze >= {HIGH_SQUEEZE_SCORE})",
        f"Triple Alignment: {len(triple_alignment)}",
        f"Dual Coiled + Bullish: {len(dual_coiled)}",
        f"Squeeze + Momentum Compression: {len(squeeze_momentum)}",
        f"Darvas Compression: {len(darvas_compressed)}",
        f"Momentum Compression: {len(momentum_compressed)}",
        "",
    ]

    def _compression_line(r: dict[str, Any]) -> str:
        return (
            f"{r.get('Ticker','')} | Price {format_currency(safe_float(r.get('Price')))} | "
            f"Darvas {safe_float(r.get('Darvas Breakout Pressure')):.0f}/100 ({r.get('Darvas Compression Bias','N/A')}) | "
            f"Momentum Compression {safe_float(r.get('Momentum Compression Score')):.0f}/100 ({r.get('Momentum Compression Bias','N/A')}) | "
            f"Momentum Box {safe_float(r.get('Momentum Box Score')):.0f}/100 | "
            f"Squeeze {safe_float(r.get('Short Squeeze Potential')):.0f}/100 | "
            f"Squeeze-Momentum {safe_float(r.get('Squeeze-Momentum Score')):.1f}/100"
        )

    if triple_alignment:
        lines += ["TRIPLE ALIGNMENT — DARVAS + MOMENTUM + SHORT SQUEEZE", "=" * 60]
        lines.extend(_compression_line(r) for r in triple_alignment)
        lines.append("")

    if squeeze_momentum:
        lines += ["SHORT SQUEEZE + MOMENTUM COMPRESSION", "=" * 60]
        lines.extend(_compression_line(r) for r in squeeze_momentum)
        lines.append("")

    if dual_coiled:
        lines += ["DUAL COILED + BULLISH", "=" * 60]
        lines.extend(_compression_line(r) for r in dual_coiled)
        lines.append("")

    if darvas_compressed:
        lines += ["DARVAS COMPRESSION", "=" * 60]
        lines.extend(_compression_line(r) for r in darvas_compressed)
        lines.append("")

    if momentum_compressed:
        lines += ["MOMENTUM COMPRESSION", "=" * 60]
        lines.extend(_compression_line(r) for r in momentum_compressed)
        lines.append("")

    if high_squeeze:
        lines += [
            "HIGH SHORT SQUEEZE + HIGH OVERALL SCORE",
            "=" * 60,
            f"Thresholds: Overall >= {HIGH_OVERALL_SCORE}/100 and Squeeze >= {HIGH_SQUEEZE_SCORE}/100",
        ]
        for r in high_squeeze:
            prob = safe_float(r.get("5-Day Probability %"))
            prob_text = f"{prob:.0f}%" if np.isfinite(prob) else "N/A"
            lines.append(
                f"{r['Ticker']} | {r.get('State','')} | Price {format_currency(r['Price'])} | "
                f"Overall {r['Strategy Score']}/100 | Core {r.get('Core Score', r['Strategy Score'])}/100 | "
                f"Squeeze {safe_float(r.get('Short Squeeze Potential')):.0f}/100 "
                f"({r.get('Short Squeeze Label','N/A')}) | SI Bonus +{r.get('Squeeze Bonus',0)} | "
                f"5-day probability {prob_text} | Volume {safe_float(r.get('Volume Multiple')):.2f}x"
            )
        lines.append("")

    if confirmed:
        lines += ["CONFIRMED BREAKOUTS", "=" * 60]
        for r in sorted(confirmed, key=lambda x: x.get("Strategy Score", 0), reverse=True):
            lines.append(
                f"{r['Ticker']} | Price {format_currency(r['Price'])} | Overall {r['Strategy Score']}/100 | "
                f"Core {r.get('Core Score', r['Strategy Score'])}/100 | "
                f"Squeeze {safe_float(r.get('Short Squeeze Potential')):.0f}/100 ({r.get('Short Squeeze Label','N/A')}) | "
                f"SI Bonus +{r.get('Squeeze Bonus',0)} | Volume {r['Volume Multiple']:.2f}x | Breakout {format_currency(r['Breakout Level'])} | "
                f"Momentum Range {format_currency(safe_float(r.get('Momentum Low Target')))} to {format_currency(safe_float(r.get('Momentum High Target')))}"
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
                f"Squeeze {safe_float(r.get('Short Squeeze Potential')):.0f}/100 ({r.get('Short Squeeze Label','N/A')}) | "
                f"SI Bonus +{r.get('Squeeze Bonus',0)} | Distance {r['Distance to Breakout %']:.2f}% | 5-day probability {prob_text} | "
                f"Volume {r['Volume Multiple']:.2f}x | "
                f"Momentum Range {format_currency(safe_float(r.get('Momentum Low Target')))} to {format_currency(safe_float(r.get('Momentum High Target')))}"
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
    enrichment_progress_callback=None,
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
                ticker, raw, settings, stock_benchmark_df, btc_df, stock_benchmark_ticker,
                fast_batch=True,
            )
            if result.get("Error"):
                errors.append({"Ticker": ticker, "Error": result["Error"]})
            else:
                results.append(result)
        except Exception as exc:
            errors.append({"Ticker": ticker, "Error": str(exc)[:180]})

    # Retain normal breakout alerts plus strong compression setups. This lets the
    # daily digest surface pre-breakout coils without weakening confirmed-breakout rules.
    alerts = [
        r for r in results
        if r.get("State") in {"BREAKOUT WATCH", "PRICE BREAKOUT / WEAK VOLUME", "CONFIRMED BREAKOUT"}
        or safe_float(r.get("Darvas Breakout Pressure"), -1) >= 65
        or safe_float(r.get("Momentum Compression Score"), -1) >= 65
    ]

    # Stage 2 enrichment: prefetch slow Yahoo fundamentals concurrently using plain
    # Python worker functions. Avoid Streamlit caching/context inside worker threads.
    prefetched_short_interest: dict[str, dict[str, Any]] = {}
    enrichment_tickers = [str(r.get("Ticker", "")) for r in alerts
                          if str(r.get("Ticker", "")) and not str(r.get("Ticker", "")).endswith("-USD")
                          and str(r.get("Ticker", "")) not in {"SPY","QQQ","DIA","IWM","MDY","RSP"}]
    enrichment_tickers = list(dict.fromkeys(enrichment_tickers))
    if enrichment_tickers:
        with ThreadPoolExecutor(max_workers=min(6, len(enrichment_tickers))) as pool:
            futures = {pool.submit(fetch_short_interest_fundamentals_plain, t): t for t in enrichment_tickers}
            done_count = 0
            for future in as_completed(futures):
                t = futures[future]
                try: prefetched_short_interest[t] = future.result()
                except Exception as exc: prefetched_short_interest[t] = {"available": False, "error": str(exc)}
                done_count += 1
                if enrichment_progress_callback:
                    enrichment_progress_callback(done_count, len(enrichment_tickers), t)

    for r in alerts:
        ticker = r.get("Ticker", "")
        raw = market_data.get(ticker)
        asset_df = add_indicators(raw, settings) if raw is not None and not raw.empty else None

        if asset_df is not None and not asset_df.empty:
            # Reconstruct only what the expensive enrichment needs.
            if r.get("State") == "BREAKOUT WATCH":
                core_for_probability = int(r.get("Strategy Score", 0))
                if core_for_probability >= MIN_WATCH_SCORE_FOR_PROBABILITY:
                    if r.get("Breakout Type") == "MOMENTUM":
                        current_structure = detect_momentum_breakout(asset_df, settings)
                    else:
                        current_structure = detect_current_box(asset_df, settings)
                    current_trend = evaluate_trend_template(asset_df, settings)
                    current_dry = evaluate_volume_dry_up(asset_df, settings)
                    prob = estimate_breakout_probability(
                        asset_df, settings, current_structure, current_trend, current_dry
                    )
                    r["5-Day Probability %"] = prob.get("probabilities", {}).get(5, np.nan)
                    r["Probability Confidence"] = prob.get("confidence", "N/A") if prob.get("available") else "N/A"
                    r["Probability Status"] = "Calculated" if prob.get("available") else prob.get("reason", "Not available")
                else:
                    r["5-Day Probability %"] = np.nan
                    r["Probability Confidence"] = "N/A"
                    r["Probability Status"] = (
                        f"Skipped for speed: Core Score {core_for_probability}/100 is below "
                        f"{MIN_WATCH_SCORE_FOR_PROBABILITY}"
                    )

            if r.get("Breakout Type") == "MOMENTUM":
                target_structure = detect_momentum_breakout(asset_df, settings)
            else:
                target_structure = detect_current_box(asset_df, settings)
            target_data = calculate_breakout_targets(asset_df, target_structure)
            r["Targets"] = target_data.get("targets", [])

        sq = fetch_short_squeeze_snapshot(ticker, asset_df, prefetched_short_interest.get(ticker))
        core = int(r.get("Strategy Score", 0))
        bonus = get_squeeze_bonus(safe_float(sq.get("score"))) if sq.get("available") else 0
        r["Core Score"] = core
        r["Short Squeeze Potential"] = safe_float(sq.get("score")) if sq.get("available") else np.nan
        r["Short Squeeze Label"] = sq.get("label", "N/A") if sq.get("available") else "N/A"
        r["Squeeze Bonus"] = bonus
        r["Strategy Score"] = min(100, core + bonus)
        mcomp = safe_float(r.get("Momentum Compression Score"), 0.0)
        mbox = safe_float(r.get("Momentum Box Score"), 0.0)
        sqscore = safe_float(r.get("Short Squeeze Potential"), 0.0)
        r["Squeeze-Momentum Score"] = round(0.45 * sqscore + 0.35 * mcomp + 0.20 * mbox, 1)
        r["Targets"] = add_resistance_break_scores(
            r.get("Targets", []),
            {"score": mbox},
            {"score": mcomp, "bias": r.get("Momentum Compression Bias", "NEUTRAL")},
            short_squeeze_score=sqscore,
            current_volume_multiple=safe_float(r.get("Volume Multiple")),
        )
        flags = compression_category_flags(r)
        r["Triple Alignment"] = flags["triple"]
        r["Dual Coiled Bullish"] = flags["dual"]
        r["Squeeze + Momentum Compression"] = flags["squeeze_momentum"]

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




def run_short_squeeze_scan(
    scan_mode: str = "sp500",
    threshold: float = 70.0,
    progress_callback=None,
    max_workers: int = 6,
) -> dict[str, Any]:
    """Scan one stock universe specifically for high short-squeeze potential.

    OHLCV is downloaded in 50-symbol batches. Short-interest fundamentals require
    per-ticker Yahoo lookups, so those calls are parallelized conservatively.
    This scan is independent of Darvas state: a stock does not need to be on
    BREAKOUT WATCH to appear here.
    """
    mode = normalize_scan_mode(scan_mode)
    label = "NASDAQ-100" if mode == "nasdaq100" else "S&P 500"
    stock_symbols = get_nasdaq100_tickers() if mode == "nasdaq100" else get_sp500_tickers()

    # No crypto/ETF symbols: the short-squeeze model is stock-specific.
    stock_symbols = list(dict.fromkeys(stock_symbols))
    market_data = download_market_data_batch(
        tuple(stock_symbols),
        default_daily_settings().history_period,
        chunk_size=50,
    )

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    total = len(stock_symbols)
    completed = 0

    def _one(ticker: str) -> tuple[str, dict[str, Any]]:
        raw = market_data.get(ticker)
        if raw is None or raw.empty:
            return ticker, {"available": False, "error": "No market data"}
        sq = fetch_short_squeeze_snapshot(ticker, raw)
        return ticker, sq

    # Conservative parallelism reduces runtime without hammering Yahoo too hard.
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
        futures = {pool.submit(_one, ticker): ticker for ticker in stock_symbols}
        for future in as_completed(futures):
            ticker = futures[future]
            completed += 1
            if progress_callback:
                progress_callback(completed, total, ticker)
            try:
                _, sq = future.result()
            except Exception as exc:
                errors.append({"Ticker": ticker, "Error": str(exc)[:180]})
                continue

            if not sq.get("available"):
                if sq.get("error"):
                    errors.append({"Ticker": ticker, "Error": str(sq.get("error"))[:180]})
                continue

            score = safe_float(sq.get("score"))
            if not np.isfinite(score) or score < threshold:
                continue

            raw = market_data.get(ticker)
            latest_price = safe_float(raw["Close"].iloc[-1]) if raw is not None and not raw.empty else np.nan
            return_20d = np.nan
            if raw is not None and len(raw) >= 21:
                p0 = safe_float(raw["Close"].iloc[-21])
                p1 = safe_float(raw["Close"].iloc[-1])
                if np.isfinite(p0) and p0 > 0 and np.isfinite(p1):
                    return_20d = (p1 / p0 - 1) * 100

            rows.append({
                "Ticker": ticker,
                "Price": latest_price,
                "Squeeze Score": score,
                "Squeeze Label": sq.get("label", "N/A"),
                "Short % Float": (
                    safe_float(sq.get("short_percent_float")) * 100
                    if np.isfinite(safe_float(sq.get("short_percent_float")))
                    else np.nan
                ),
                "Days to Cover": safe_float(sq.get("days_to_cover")),
                "Short Interest Change %": safe_float(sq.get("short_change_pct")),
                "Relative Volume": safe_float(sq.get("relative_volume")),
                "20-Day Return %": return_20d,
            })

    rows.sort(
        key=lambda r: (
            safe_float(r.get("Squeeze Score"), -1),
            safe_float(r.get("Short % Float"), -1),
            safe_float(r.get("Days to Cover"), -1),
        ),
        reverse=True,
    )

    return {
        "universe": label,
        "universe_count": total,
        "threshold": threshold,
        "candidates": rows,
        "errors": errors,
    }



@st.cache_data(ttl=21600, show_spinner=False)
def fetch_short_interest_fundamentals_plain(ticker: str) -> dict[str, Any]:
    """Fetch Yahoo short-interest fundamentals without Streamlit caching."""
    result = {"applicable": True, "available": False, "short_percent_float": float("nan"),
              "days_to_cover": float("nan"), "short_change_pct": float("nan"), "float_shares": float("nan")}
    if ticker.endswith("-USD") or ticker in {"SPY","QQQ","DIA","IWM","MDY","RSP"}:
        result["applicable"] = False
        return result
    try:
        info = yf.Ticker(ticker).info or {}
        sp = safe_float(info.get("shortPercentOfFloat")); dc = safe_float(info.get("shortRatio"))
        ss = safe_float(info.get("sharesShort")); pr = safe_float(info.get("sharesShortPriorMonth")); fl = safe_float(info.get("floatShares"))
        ch = (ss-pr)/pr*100 if np.isfinite(ss) and np.isfinite(pr) and pr>0 else np.nan
        result.update({"available": any(np.isfinite(x) for x in (sp,dc,ch,fl)),
                       "short_percent_float": sp, "days_to_cover": dc,
                       "short_change_pct": ch, "float_shares": fl})
    except Exception as exc:
        result["error"] = str(exc)
    return result


def fetch_short_squeeze_snapshot(ticker: str, asset_df=None, fundamentals: dict[str, Any] | None = None) -> dict[str, Any]:
    r={"applicable":True,"available":False,"score":np.nan,"label":"N/A","short_percent_float":np.nan,
       "days_to_cover":np.nan,"short_change_pct":np.nan,"relative_volume":np.nan,"components":{}}
    fundamentals = fundamentals if fundamentals is not None else fetch_short_interest_fundamentals_plain(ticker)
    if not fundamentals.get("applicable", True):
        r["applicable"] = False; return r
    if fundamentals.get("error"): r["error"] = fundamentals.get("error")
    sp=safe_float(fundamentals.get("short_percent_float")); dc=safe_float(fundamentals.get("days_to_cover"))
    ch=safe_float(fundamentals.get("short_change_pct")); fl=safe_float(fundamentals.get("float_shares"))
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
    wt={"sf":35,"dc":20,"chg":10,"float":10,"rv":15,"mom":10}; aw=sum(wt[k] for k in c)
    sc=sum(c[k]*wt[k] for k in c)/aw if aw else np.nan
    r.update({"available":bool(c),"score":sc,"label":"HIGH" if sc>=70 else "MODERATE" if sc>=45 else "LOW",
              "short_percent_float":sp,"days_to_cover":dc,"short_change_pct":ch,"relative_volume":rv,"components":c})
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
    st.caption("Build: Dual-Path Momentum Fix 2026-08-14-B")
    st.caption(f"Build: Momentum Targets 2026-08-14-D — Daily {ACTIVE_SCAN_LABEL} + Crypto Breakout Scanner")
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
        # Give the ticker selector an explicit session-state key so changing the
        # dropdown deterministically triggers analysis for the newly selected symbol.
        if st.session_state.get("ticker_to_display") not in ticker_list:
            st.session_state["ticker_to_display"] = ticker_list[0]

        def _mark_ticker_changed() -> None:
            st.session_state["ticker_selection_changed"] = True

        ticker = st.selectbox(
            "Ticker to display",
            ticker_list,
            key="ticker_to_display",
            on_change=_mark_ticker_changed,
        )
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
    scan_col1, scan_col2, scan_col3 = st.columns(3)
    with scan_col1:
        run_sp500_now = st.button(
            "Run Full S&P 500 Scan Now", type="primary", use_container_width=True
        )
    with scan_col2:
        run_nasdaq100_now = st.button(
            "Run Full Nasdaq 100 Scan Now", type="primary", use_container_width=True
        )
    with scan_col3:
        run_squeeze_now = st.button(
            "🔥 Scan Short Squeeze Candidates", use_container_width=True
        )

    st.caption(
        "Short Squeeze scan uses the URL-selected universe "
        f"({ACTIVE_SCAN_LABEL}) and is independent of Darvas breakout state. "
        "For unattended once-daily breakout execution, use the GitHub Actions workflow / daily_scan.py runner."
    )

    if run_squeeze_now:
        squeeze_threshold = 70.0
        squeeze_progress = st.progress(
            0.0,
            text=f"Starting {ACTIVE_SCAN_LABEL} short-squeeze scan..."
        )

        def _update_squeeze_progress(done: int, total: int, symbol: str) -> None:
            squeeze_progress.progress(
                min(done / max(total, 1), 1.0),
                text=f"Evaluating short squeeze: {symbol} ({done}/{total})",
            )

        try:
            squeeze_result = run_short_squeeze_scan(
                scan_mode=ACTIVE_SCAN_MODE,
                threshold=squeeze_threshold,
                progress_callback=_update_squeeze_progress,
                max_workers=6,
            )
            squeeze_progress.progress(1.0, text="Short-squeeze scan complete")
            squeeze_candidates = squeeze_result.get("candidates", [])

            st.markdown(f"## 🔥 {ACTIVE_SCAN_LABEL} Short Squeeze Potential")
            st.caption(
                f"Candidates with Short Squeeze Potential >= {squeeze_threshold:.0f}/100. "
                "This scan does not require a Darvas or Momentum breakout setup."
            )

            if squeeze_candidates:
                squeeze_df = pd.DataFrame(squeeze_candidates)
                st.dataframe(
                    squeeze_df.style.format({
                        "Price": "${:,.2f}",
                        "Squeeze Score": "{:.1f}",
                        "Short % Float": "{:.1f}%",
                        "Days to Cover": "{:.2f}",
                        "Short Interest Change %": "{:+.1f}%",
                        "Relative Volume": "{:.2f}x",
                        "20-Day Return %": "{:+.1f}%",
                    }),
                    hide_index=True,
                    use_container_width=True,
                )
                st.success(
                    f"Found {len(squeeze_candidates)} high-squeeze candidate(s) "
                    f"from {squeeze_result.get('universe_count', 0)} {ACTIVE_SCAN_LABEL} securities."
                )
            else:
                st.info(
                    f"No {ACTIVE_SCAN_LABEL} stocks currently have a Short Squeeze Potential "
                    f"score of {squeeze_threshold:.0f}/100 or higher."
                )

            if squeeze_result.get("errors"):
                with st.expander(f"Squeeze scan warnings ({len(squeeze_result['errors'])})"):
                    st.dataframe(
                        pd.DataFrame(squeeze_result["errors"]),
                        hide_index=True,
                        use_container_width=True,
                    )
        except Exception as exc:
            st.error(f"Short-squeeze scan failed: {exc}")

    # Manual buttons explicitly choose their universe. autorun=1 continues to use
    # the URL-selected ACTIVE_SCAN_MODE.
    manual_scan_mode = (
        "sp500" if run_sp500_now
        else "nasdaq100" if run_nasdaq100_now
        else None
    )
    run_daily_now = manual_scan_mode is not None
    requested_scan_mode = manual_scan_mode or ACTIVE_SCAN_MODE
    requested_scan_label = "NASDAQ-100" if requested_scan_mode == "nasdaq100" else "S&P 500"

    should_run_daily = run_daily_now or AUTO_RUN_DAILY_SCAN
    if should_run_daily:
        progress = st.progress(0.0, text=f"Starting {requested_scan_label} daily market scan...")
        def _update_progress(done: int, total: int, symbol: str) -> None:
            progress.progress(min(done / max(total, 1), 1.0), text=f"Scanning {symbol} ({done}/{total})")
        try:
            daily_result = run_daily_market_scan(
                settings=settings,
                send_email=True,
                force=bool(run_daily_now),
                progress_callback=_update_progress,
                scan_mode=requested_scan_mode,
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
                HIGH_OVERALL_SCORE = 80
                HIGH_SQUEEZE_SCORE = 70

                def _alert_row(r: dict[str, Any]) -> dict[str, Any]:
                    return {
                        "Ticker": r.get("Ticker"),
                        "State": r.get("State"),
                        "Price": r.get("Price"),
                        "Overall Score": r.get("Strategy Score"),
                        "Core Score": r.get("Core Score", r.get("Strategy Score")),
                        "Squeeze": r.get("Short Squeeze Potential"),
                        "SI Bonus": r.get("Squeeze Bonus", 0),
                        "Distance %": r.get("Distance to Breakout %"),
                        "Volume x": r.get("Volume Multiple"),
                        "5-Day Prob. %": r.get("5-Day Probability %"),
                        "Momentum Box": r.get("Momentum Box Score"),
                        "Momentum High": r.get("Momentum High Target"),
                        "Momentum Low": r.get("Momentum Low Target"),
                        "Darvas Pressure": r.get("Darvas Breakout Pressure"),
                        "Darvas Bias": r.get("Darvas Compression Bias"),
                        "Momentum Compression": r.get("Momentum Compression Score"),
                        "Momentum Bias": r.get("Momentum Compression Bias"),
                        "Squeeze-Momentum": r.get("Squeeze-Momentum Score"),
                    }

                high_squeeze = [
                    r for r in alerts
                    if safe_float(r.get("Strategy Score"), -1) >= HIGH_OVERALL_SCORE
                    and safe_float(r.get("Short Squeeze Potential"), -1) >= HIGH_SQUEEZE_SCORE
                ]
                high_squeeze = sorted(
                    high_squeeze,
                    key=lambda x: (
                        safe_float(x.get("Strategy Score"), -1),
                        safe_float(x.get("Short Squeeze Potential"), -1),
                        safe_float(x.get("5-Day Probability %"), -1),
                    ),
                    reverse=True,
                )
                confirmed_rows = sorted(
                    [r for r in alerts if r.get("State") == "CONFIRMED BREAKOUT"],
                    key=lambda x: safe_float(x.get("Strategy Score"), -1),
                    reverse=True,
                )
                watch_rows = sorted(
                    [r for r in alerts if r.get("State") == "BREAKOUT WATCH"],
                    key=lambda x: (
                        safe_float(x.get("5-Day Probability %"), -1),
                        safe_float(x.get("Strategy Score"), -1),
                    ),
                    reverse=True,
                )

                triple_rows = sorted(
                    [r for r in alerts if compression_category_flags(r)["triple"]],
                    key=lambda x: (safe_float(x.get("Squeeze-Momentum Score"), -1), safe_float(x.get("Darvas Breakout Pressure"), -1)),
                    reverse=True,
                )
                squeeze_momentum_rows = sorted(
                    [r for r in alerts if compression_category_flags(r)["squeeze_momentum"]],
                    key=lambda x: safe_float(x.get("Squeeze-Momentum Score"), -1),
                    reverse=True,
                )
                dual_rows = sorted(
                    [r for r in alerts if compression_category_flags(r)["dual"]],
                    key=lambda x: (safe_float(x.get("Darvas Breakout Pressure"), -1), safe_float(x.get("Momentum Compression Score"), -1)),
                    reverse=True,
                )
                darvas_rows = sorted(
                    [r for r in alerts if compression_category_flags(r)["darvas"]],
                    key=lambda x: safe_float(x.get("Darvas Breakout Pressure"), -1),
                    reverse=True,
                )
                momentum_rows = sorted(
                    [r for r in alerts if compression_category_flags(r)["momentum"]],
                    key=lambda x: safe_float(x.get("Momentum Compression Score"), -1),
                    reverse=True,
                )

                st.markdown("## 💥 Triple Alignment — Darvas + Momentum + Short Squeeze")
                st.caption("Highest-priority view: bullish Darvas compression + bullish Momentum compression + Short Squeeze Potential >= 70.")
                if triple_rows:
                    st.dataframe(pd.DataFrame([_alert_row(r) for r in triple_rows]), hide_index=True, use_container_width=True)
                else:
                    st.info("No Triple Alignment candidates in this scan.")

                st.markdown("### 🔥 Short Squeeze + Momentum Compression")
                st.caption("Independent of Darvas: Squeeze >= 70, Momentum Compression >= 65, bullish momentum bias, Momentum Box >= 50.")
                if squeeze_momentum_rows:
                    st.dataframe(pd.DataFrame([_alert_row(r) for r in squeeze_momentum_rows]), hide_index=True, use_container_width=True)
                else:
                    st.info("No Short Squeeze + Momentum Compression candidates in this scan.")

                st.markdown("### 🚀 Dual Coiled + Bullish")
                if dual_rows:
                    st.dataframe(pd.DataFrame([_alert_row(r) for r in dual_rows]), hide_index=True, use_container_width=True)
                else:
                    st.info("No Dual Coiled + Bullish candidates in this scan.")

                st.markdown("### ⚡ Darvas Compression")
                if darvas_rows:
                    st.dataframe(pd.DataFrame([_alert_row(r) for r in darvas_rows]), hide_index=True, use_container_width=True)
                else:
                    st.info("No strong Darvas Compression candidates in this scan.")

                st.markdown("### 🌀 Momentum Compression")
                if momentum_rows:
                    st.dataframe(pd.DataFrame([_alert_row(r) for r in momentum_rows]), hide_index=True, use_container_width=True)
                else:
                    st.info("No strong Momentum Compression candidates in this scan.")

                st.markdown("### 🔥 High Short Squeeze + High Overall Score")
                st.caption(
                    f"Breakout watches or confirmed breakouts with Overall Score >= {HIGH_OVERALL_SCORE} "
                    f"and Short Squeeze Potential >= {HIGH_SQUEEZE_SCORE}. Ranked by Overall Score, then Squeeze."
                )
                if high_squeeze:
                    st.dataframe(
                        pd.DataFrame([_alert_row(r) for r in high_squeeze]),
                        hide_index=True,
                        use_container_width=True,
                    )
                else:
                    st.info("No current alerts meet both high-score/high-squeeze thresholds.")

                st.markdown("### ✅ Confirmed Breakouts")
                if confirmed_rows:
                    st.dataframe(
                        pd.DataFrame([_alert_row(r) for r in confirmed_rows]),
                        hide_index=True,
                        use_container_width=True,
                    )
                else:
                    st.info("No confirmed breakouts in this scan.")

                st.markdown("### 👀 Breakout Candidates")
                if watch_rows:
                    st.dataframe(
                        pd.DataFrame([_alert_row(r) for r in watch_rows]),
                        hide_index=True,
                        use_container_width=True,
                    )
                else:
                    st.info("No breakout-watch candidates in this scan.")
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
                "Use this table to decide later whether the 0-10 leading-indicator score "
                "deserves weight in Overall Score."
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

    if st.session_state.pop("ticker_selection_changed", False):
        st.info(f"Ticker changed to **{ticker}** — running single-ticker analysis now.")

    try:
        with st.spinner(f"Loading {asset_name} ({ticker}) daily candles..."):
            raw_asset = download_market_data(ticker, settings.history_period)
            raw_btc = (
                raw_asset.copy()
                if ticker == "BTC-USD"
                else download_market_data("BTC-USD", settings.history_period)
            )
            raw_momentum_benchmark = (
                raw_btc.copy()
                if ticker.endswith("-USD")
                else download_market_data(ACTIVE_BENCHMARK, settings.history_period)
            )
    except Exception as exc:
        st.error(f"Could not load market data: {exc}")
        st.stop()

    if raw_asset.empty or raw_btc.empty:
        st.error("No market data was returned. Try Refresh market data.")
        st.stop()

    asset_df = add_indicators(raw_asset, settings)
    btc_df = add_indicators(raw_btc, settings)
    momentum_benchmark_df = add_indicators(raw_momentum_benchmark, settings)

    minimum_rows = max(221, settings.max_base_days + 2)
    if len(asset_df) < minimum_rows:
        st.error(f"At least {minimum_rows} daily candles are required.")
        st.stop()

    # Run both structure detectors for the interactive UI, matching the batch scanner.
    darvas_result = detect_current_box(asset_df, settings)
    momentum_result = detect_momentum_breakout(asset_df, settings)
    darvas_actionable = darvas_result.get("state") in {"BREAKOUT WATCH", "PRICE BREAKOUT / WEAK VOLUME", "CONFIRMED BREAKOUT"}
    momentum_actionable = momentum_result.get("state") in {"BREAKOUT WATCH", "PRICE BREAKOUT / WEAK VOLUME", "CONFIRMED BREAKOUT"}
    if darvas_actionable:
        box_result = dict(darvas_result)
        box_result["structure_type"] = "DARVAS"
    elif momentum_actionable:
        box_result = dict(momentum_result)
        box_result["structure_type"] = "MOMENTUM"
    else:
        box_result = dict(darvas_result)
        box_result["structure_type"] = "DARVAS" if darvas_result.get("valid") else "NONE"

    trend_result = evaluate_trend_template(asset_df, settings)
    dry_up_result = evaluate_volume_dry_up(asset_df, settings)
    rs_result = evaluate_relative_strength(ticker, asset_df, btc_df)
    momentum_box = evaluate_momentum_box(asset_df, momentum_benchmark_df)
    momentum_targets = calculate_momentum_targets(asset_df, momentum_box, horizon_sessions=5)
    momentum_compression = evaluate_momentum_compression(asset_df, momentum_box)
    darvas_compression = evaluate_darvas_compression(asset_df, darvas_result, momentum_box, settings)
    core_score = calculate_score(box_result, trend_result, dry_up_result, rs_result)
    pre_breakout = evaluate_pre_breakout_momentum(
        asset_df,
        momentum_benchmark_df,
        ticker,
        "BTC-USD" if ticker.endswith("-USD") else ACTIVE_BENCHMARK,
    )
    # Keep the ticker dropdown responsive: cache slow Yahoo short-interest fundamentals
    # in Streamlit session state and reuse them across ordinary widget reruns.
    _si_cache = st.session_state.setdefault("short_interest_fundamentals_cache", {})
    fundamentals = _si_cache.get(ticker)
    if fundamentals is None:
        fundamentals = fetch_short_interest_fundamentals_plain(ticker)
        _si_cache[ticker] = fundamentals
    squeeze_snapshot = fetch_short_squeeze_snapshot(ticker, asset_df, fundamentals)
    score = calculate_score_with_squeeze(core_score, squeeze_snapshot)

    breakout_probability = (
        estimate_breakout_probability(asset_df, settings, box_result, trend_result, dry_up_result)
        if (
            box_result.get("state") == "BREAKOUT WATCH"
            and core_score["Total"] >= MIN_WATCH_SCORE_FOR_PROBABILITY
        )
        else {
            "available": False,
            "probabilities": {},
            "reason": (
                f"Skipped for speed: Core Score {core_score['Total']}/100 is below "
                f"{MIN_WATCH_SCORE_FOR_PROBABILITY}"
                if box_result.get("state") == "BREAKOUT WATCH"
                else "Current state is not BREAKOUT WATCH"
            ),
        }
    )
    breakout_targets = (
        calculate_breakout_targets(asset_df, box_result)
        if box_result.get("state") in {"BREAKOUT WATCH", "PRICE BREAKOUT / WEAK VOLUME", "CONFIRMED BREAKOUT"}
        else {"available": False, "targets": []}
    )

    if breakout_targets.get("targets"):
        breakout_targets["targets"] = add_resistance_break_scores(
            breakout_targets["targets"],
            momentum_box,
            momentum_compression,
            short_squeeze_score=safe_float(squeeze_snapshot.get("score")) if squeeze_snapshot.get("available") else np.nan,
            current_volume_multiple=safe_float(box_result.get("volume_multiple")),
        )
        breakout_targets["nearest_resistance"] = next(
            (t for t in breakout_targets["targets"] if t.get("type") == "Prior swing resistance"), None
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

    metric_columns = st.columns(12)
    metric_columns[0].metric("Price", format_currency(latest["Close"]), f"{daily_change:.2f}%")
    metric_columns[1].metric("Overall Score", f"{score['Total']}/100")
    metric_columns[2].metric("Momentum Box", f"{momentum_box.get('score',0)}/100", momentum_box.get('trajectory','N/A'))
    metric_columns[4].metric("Core Score", f"{score['Core Total']}/100")
    sq_display = safe_float(squeeze_snapshot.get("score"))
    metric_columns[3].metric(
        "Pre-Breakout",
        f"{pre_breakout.get('score', 0)}/10",
        pre_breakout.get("label", "N/A"),
    )
    # Short squeeze is still shown in the dedicated panel/tab below.

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
        metric_columns[8].metric(
            "Nearest Target",
            format_currency(first_target["price"]),
            f"{first_target['upside_from_price_pct']:+.1f}%",
        )
    else:
        metric_columns[8].metric("Forecast", "N/A")
    metric_columns[10].metric(
        "Darvas Pressure",
        f"{darvas_compression.get('score', 0)}/100" if darvas_compression.get("available") else "N/A",
        format_direction_bias(darvas_compression.get("bias", "NEUTRAL")),
        delta_color="off",
    )
    metric_columns[11].metric(
        "Momentum Compression",
        f"{momentum_compression.get('score', 0)}/100" if momentum_compression.get("available") else "N/A",
        format_direction_bias(momentum_compression.get("bias", "NEUTRAL")),
        delta_color="off",
    )

    tabs = st.tabs(
        [
            "Overview",
            "Structure Details",
            "Trend Template",
            "Volume Dry-Up",
            "Relative Strength",
            "Breakout Forecast",
            "Pre-Breakout",
            "Momentum Box",
            "Compression",
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

        structure_type = box_result.get("structure_type", "DARVAS")
        st.caption(f"Active breakout path: **{structure_type}**")
        if box_result.get("confirmed_breakout"):
            if structure_type == "MOMENTUM":
                st.success("Momentum breakout confirmed: price cleared recent resistance with the configured volume confirmation.")
            else:
                st.success("Darvas breakout confirmed: price cleared the box with the configured volume confirmation.")
        elif box_result.get("state") == "PRICE BREAKOUT / WEAK VOLUME":
            st.warning("Price has cleared resistance, but volume has not reached the configured confirmation multiple.")
        elif box_result.get("state") == "BREAKOUT WATCH":
            if structure_type == "MOMENTUM":
                st.warning("Momentum setup is within 2% of the recent 20-day resistance level.")
            else:
                st.warning("Price is within 2% of the current Darvas box high.")
        elif box_result.get("valid"):
            st.info("A valid Darvas structure is present, but price is not yet near a breakout.")
        else:
            st.info("No actionable Darvas or momentum breakout structure is active on the latest daily candle.")

    with tabs[1]:
        structure_type = box_result.get("structure_type", "DARVAS")
        if structure_type == "MOMENTUM":
            details = {
                "Breakout path": "MOMENTUM",
                "Status": box_result.get("state", "N/A"),
                "20-day resistance": format_currency(box_result.get("box_high")),
                "55-day prior high": format_currency(box_result.get("prior_high_55")),
                "20-day range low": format_currency(box_result.get("box_low")),
                "Breakout level": format_currency(box_result.get("breakout_level")),
                "5-day momentum": f"{safe_float(box_result.get('momentum_5d_pct')):+.2f}%",
                "20-day momentum": f"{safe_float(box_result.get('momentum_20d_pct')):+.2f}%",
                "Near 55-day high": "Yes" if box_result.get("near_55_high") else "No",
                "Price breakout": "Yes" if box_result.get("price_breakout") else "No",
                "Volume multiple": f"{safe_float(box_result.get('volume_multiple')):.2f}x",
                "Volume confirmation": "Yes" if box_result.get("volume_breakout") else "No",
                "Momentum quality": f"{safe_float(box_result.get('quality_score'), 0):.1f}/100",
            }
            st.subheader("Momentum Breakout Structure")
            st.dataframe(pd.DataFrame(details.items(), columns=["Measure", "Value"]), hide_index=True, use_container_width=True)
            render_checks("Momentum Qualification", box_result.get("checks", {}))
        elif box_result.get("valid"):
            box_start = box_result.get("box_start")
            box_end = box_result.get("box_end")
            details = {
                "Breakout path": "DARVAS",
                "Status": box_result.get("state", "N/A"),
                "Detected base length": f"{box_result.get('base_days', 0)} days",
                "Base quality": f"{safe_float(box_result.get('quality_score'), 0):.1f}/100",
                "Box start": box_start.strftime("%Y-%m-%d") if hasattr(box_start, "strftime") else "N/A",
                "Box end": box_end.strftime("%Y-%m-%d") if hasattr(box_end, "strftime") else "N/A",
                "Box high": format_currency(box_result.get("box_high")),
                "Box low": format_currency(box_result.get("box_low")),
                "Box range": f"{safe_float(box_result.get('box_range_pct')):.2f}%",
                "Upper-bound tests": box_result.get("high_tests", 0),
                "Lower-bound tests": box_result.get("low_tests", 0),
                "Closes contained": f"{safe_float(box_result.get('inside_ratio'), 0) * 100:.1f}%",
                "Breakout level": format_currency(box_result.get("breakout_level")),
                "Price breakout": "Yes" if box_result.get("price_breakout") else "No",
                "Volume confirmation": "Yes" if box_result.get("volume_breakout") else "No",
            }
            st.subheader("Darvas Box Structure")
            st.dataframe(pd.DataFrame(details.items(), columns=["Measure", "Value"]), hide_index=True, use_container_width=True)
            render_checks("Box Qualification", box_result.get("checks", {}))
        else:
            st.info("No valid Darvas box and no actionable momentum breakout structure were found for the latest daily candle.")
            if darvas_result.get("reason"):
                st.caption(f"Darvas: {darvas_result.get('reason')}")
            if momentum_result.get("reason"):
                st.caption(f"Momentum: {momentum_result.get('reason')}")

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
                    "Resistance Strength": (
                        f"{safe_float(t.get('resistance_strength')):.0f}/100 {t.get('resistance_rating','')}"
                        if t.get("type") == "Prior swing resistance" and np.isfinite(safe_float(t.get("resistance_strength"))) else "—"
                    ),
                    "Break Score": (
                        f"{safe_float(t.get('resistance_break_score')):.0f}/100 {t.get('resistance_break_label','')}"
                        if t.get("type") == "Prior swing resistance" and np.isfinite(safe_float(t.get("resistance_break_score"))) else "—"
                    ),
                    "Tests": t.get("resistance_tests", "—") if t.get("type") == "Prior swing resistance" else "—",
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
                    strength = safe_float(nearest.get("resistance_strength"))
                    break_score = safe_float(nearest.get("resistance_break_score"))
                    extra = ""
                    if np.isfinite(strength):
                        extra += f" Strength {strength:.0f}/100 ({nearest.get('resistance_rating','N/A')})."
                    if np.isfinite(break_score):
                        extra += f" Break Score {break_score:.0f}/100 ({nearest.get('resistance_break_label','N/A')})."
                    st.info(
                        f"Nearest historical resistance is {format_currency(nearest['price'])} "
                        f"({nearest['upside_from_price_pct']:+.1f}% from the latest close)." + extra
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

    with tabs[6]:
        st.subheader("Pre-Breakout Momentum")
        st.caption(
            "Separate 0-10 leading-indicator score. It remains outside Overall Score "
            "until the 5-day validation history shows whether it improves prediction."
        )
        pb_components = pre_breakout.get("components", {})
        pb1, pb2, pb3, pb4, pb5 = st.columns(5)
        pb1.metric("RS Leadership", f"{pb_components.get('Relative Strength', 0)}/3")
        pb2.metric("OBV", f"{pb_components.get('OBV', 0)}/2")
        pb3.metric("Compression", f"{pb_components.get('Volatility Compression', 0)}/2")
        pb4.metric("RSI Momentum", f"{pb_components.get('RSI Momentum', 0)}/2")
        pb5.metric("Money Flow", f"{pb_components.get('Money Flow', 0)}/1")

        detail_rows = [
            {"Measure": "Pre-Breakout Score", "Value": f"{pre_breakout.get('score', 0)}/10"},
            {"Measure": "Signal Band", "Value": pre_breakout.get("label", "N/A")},
            {"Measure": "RSI(14)", "Value": f"{safe_float(pre_breakout.get('rsi')):.1f}" if np.isfinite(safe_float(pre_breakout.get('rsi'))) else "N/A"},
            {"Measure": "CMF(20)", "Value": f"{safe_float(pre_breakout.get('cmf')):.3f}" if np.isfinite(safe_float(pre_breakout.get('cmf'))) else "N/A"},
            {"Measure": "Bollinger Width", "Value": f"{safe_float(pre_breakout.get('bb_width')):.2f}%" if np.isfinite(safe_float(pre_breakout.get('bb_width'))) else "N/A"},
        ]
        st.dataframe(pd.DataFrame(detail_rows), hide_index=True, use_container_width=True)
        for label, passed in pre_breakout.get("checks", {}).items():
            st.write(f"{'✅' if passed else '❌'} {label}")

    with tabs[7]:
        st.subheader("Momentum Box")
        if not momentum_box.get("available"):
            st.info("Momentum Box requires at least 60 daily candles.")
        else:
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Momentum Score", f"{momentum_box['score']}/100")
            mc2.metric("Momentum State", momentum_box['label'])
            mc3.metric("Trajectory", momentum_box['trajectory'])

            if momentum_targets.get("available"):
                st.markdown("#### Estimated 5-Day Momentum Range")
                mt1, mt2, mt3 = st.columns(3)
                mt1.metric(
                    "Momentum High Target",
                    format_currency(momentum_targets["high_target"]),
                    f"{momentum_targets['high_upside_pct']:+.1f}%",
                )
                mt2.metric(
                    "Current Price",
                    format_currency(momentum_targets["current_price"]),
                    f"ATR {format_currency(momentum_targets['atr'])}",
                )
                mt3.metric(
                    "Momentum Low Target",
                    format_currency(momentum_targets["low_target"]),
                    f"{momentum_targets['low_downside_pct']:+.1f}%",
                )
                st.caption(
                    "Momentum targets are a 5-session ATR/momentum scenario envelope. "
                    "Stronger or accelerating momentum expands the high target and tightens the low target; "
                    "weakening momentum does the reverse. These are reference levels, not guaranteed forecasts."
                )

            score_value = momentum_box['score']
            if score_value >= 80:
                st.success("🔥 EXPLOSIVE MOMENTUM — strong multi-factor bullish momentum alignment.")
            elif score_value >= 65:
                st.success("🟢 STRONG BULLISH MOMENTUM — momentum conditions are favorable.")
            elif score_value >= 50:
                st.warning("🟡 BUILDING MOMENTUM — bullish momentum is developing but not yet strong.")
            elif score_value >= 35:
                st.info("⚪ NEUTRAL MOMENTUM — no strong directional momentum regime.")
            else:
                st.error("🔴 BEARISH / WEAK MOMENTUM — momentum conditions are unfavorable.")

            st.markdown("#### Component Scores")
            component_df = pd.DataFrame([
                {"Component": k, "Points": v, "Max": {"RSI":15,"MACD":15,"ADX":15,"Relative Volume":15,"Price Momentum":15,"Relative Strength":15,"Price Structure":10}.get(k,0)}
                for k,v in momentum_box["components"].items()
            ])
            st.dataframe(component_df, hide_index=True, use_container_width=True)

            st.markdown("#### Current Measurements")
            measurement_rows=[]
            for k,v in momentum_box["measurements"].items():
                if isinstance(v, (bool, np.bool_)):
                    disp = "Yes" if v else "No"
                elif np.isfinite(safe_float(v)):
                    disp = f"{safe_float(v):.2f}"
                else:
                    disp = "N/A"
                measurement_rows.append({"Measure": k, "Value": disp})
            st.dataframe(pd.DataFrame(measurement_rows), hide_index=True, use_container_width=True)

            if momentum_box.get("history"):
                st.markdown("#### Recent Momentum Score Trajectory")
                hist_df = pd.DataFrame(momentum_box["history"])
                st.dataframe(hist_df, hide_index=True, use_container_width=True)
                st.line_chart(hist_df.set_index("Date")["Score"])

    with tabs[8]:
        st.subheader("Compression Signals")
        d1, d2, d3 = st.columns(3)
        d1.metric("Darvas Breakout Pressure", f"{darvas_compression.get('score', 0)}/100" if darvas_compression.get("available") else "N/A")
        d2.metric("Darvas Compression Status", darvas_compression.get("status", "N/A"))
        d3.metric("Darvas Bias", format_direction_bias(darvas_compression.get("bias", "NEUTRAL")))
        if darvas_compression.get("available"):
            st.dataframe(
                pd.DataFrame([{"Component": k, "Points": v} for k, v in darvas_compression.get("components", {}).items()]),
                hide_index=True, use_container_width=True,
            )

        st.markdown("#### Momentum Compression")
        m1, m2, m3 = st.columns(3)
        m1.metric("Momentum Compression Score", f"{momentum_compression.get('score', 0)}/100" if momentum_compression.get("available") else "N/A")
        m2.metric("Momentum Compression Status", momentum_compression.get("status", "N/A"))
        m3.metric("Momentum Bias", format_direction_bias(momentum_compression.get("bias", "NEUTRAL")))
        if momentum_compression.get("available"):
            st.dataframe(
                pd.DataFrame([{"Component": k, "Points": v} for k, v in momentum_compression.get("components", {}).items()]),
                hide_index=True, use_container_width=True,
            )

        local_row = {
            "Darvas Breakout Pressure": darvas_compression.get("score", 0),
            "Darvas Compression Bias": darvas_compression.get("bias", "NEUTRAL"),
            "Momentum Compression Score": momentum_compression.get("score", 0),
            "Momentum Compression Bias": momentum_compression.get("bias", "NEUTRAL"),
            "Momentum Box Score": momentum_box.get("score", 0),
            "Short Squeeze Potential": safe_float(squeeze_snapshot.get("score")),
        }
        local_row["Squeeze-Momentum Score"] = round(
            0.45 * max(0.0, safe_float(local_row.get("Short Squeeze Potential"), 0.0))
            + 0.35 * safe_float(local_row.get("Momentum Compression Score"), 0.0)
            + 0.20 * safe_float(local_row.get("Momentum Box Score"), 0.0),
            1,
        )
        local_flags = compression_category_flags(local_row)
        st.markdown("#### Current Category Membership")

        category_rows = [
            ("triple", "💥 Triple Alignment"),
            ("squeeze_momentum", "🔥 Short Squeeze + Momentum Compression"),
            ("dual", "🚀 Dual Coiled + Bullish"),
            ("darvas", "⚡ Darvas Compression"),
            ("momentum", "🌀 Momentum Compression"),
        ]
        active_categories = [label for key, label in category_rows if bool(local_flags.get(key, False))]

        if active_categories:
            st.markdown("**Active categories:** " + " · ".join(active_categories))
        else:
            st.caption("No compression categories are currently active for this ticker.")

        for key, label in category_rows:
            if bool(local_flags.get(key, False)):
                st.markdown(f"✅ **ACTIVE** — {label}")
            else:
                st.caption(f"❌ NOT ACTIVE — {label}")

    with tabs[9]:
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
