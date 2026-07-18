from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="Darvas + Minervini Market Scanner",
    page_icon="📦",
    layout="wide",
)


ASSET_GROUPS = {
    "Crypto": {
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD",
    },
    "S&P 500": {
        "S&P 500 ETF (SPY)": "SPY",
        "S&P 500 Index (^GSPC)": "^GSPC",
    },
    "Nasdaq-100": {
        "Nasdaq-100 ETF (QQQ)": "QQQ",
        "Nasdaq-100 Index (^NDX)": "^NDX",
    },
}

BENCHMARKS = {
    "Crypto": "BTC-USD",
    "S&P 500": "SPY",
    "Nasdaq-100": "QQQ",
}

UNIVERSE_URLS = {
    "S&P 500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "Nasdaq-100": "https://en.wikipedia.org/wiki/Nasdaq-100",
}


@dataclass(frozen=True)
class Settings:
    history_period: str
    box_days: int
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

    return df


def detect_current_box(df: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    """
    Detect a Darvas-style consolidation using completed candles before the latest candle.

    The latest candle is excluded so that it can independently qualify as a breakout.
    """
    if len(df) < settings.box_days + 2:
        return {"valid": False, "reason": "Not enough data"}

    box = df.iloc[-(settings.box_days + 1):-1].copy()
    latest = df.iloc[-1]
    previous = df.iloc[-2]

    box_high = safe_float(box["High"].max())
    box_low = safe_float(box["Low"].min())
    midpoint = (box_high + box_low) / 2
    range_pct = ((box_high - box_low) / midpoint) * 100 if midpoint else np.nan

    tolerance = settings.test_tolerance_pct / 100
    high_test_level = box_high * (1 - tolerance)
    low_test_level = box_low * (1 + tolerance)

    high_tests = int((box["High"] >= high_test_level).sum())
    low_tests = int((box["Low"] <= low_test_level).sum())

    inside_ratio = float(
        ((box["Close"] <= box_high) & (box["Close"] >= box_low)).mean()
    )

    max_range_pass = range_pct <= settings.max_box_range_pct
    high_tests_pass = high_tests >= settings.minimum_high_tests
    low_tests_pass = low_tests >= settings.minimum_low_tests
    containment_pass = inside_ratio >= 0.90

    box_valid = all(
        [max_range_pass, high_tests_pass, low_tests_pass, containment_pass]
    )

    breakout_level = box_high * (1 + settings.breakout_buffer_pct / 100)
    near_breakout_floor = box_high * 0.98

    latest_close = safe_float(latest["Close"])
    previous_close = safe_float(previous["Close"])
    latest_volume = safe_float(latest["Volume"], 0.0)
    average_volume = safe_float(df["Volume"].iloc[-21:-1].mean(), 0.0)
    volume_multiple = latest_volume / average_volume if average_volume > 0 else np.nan

    price_breakout = latest_close > breakout_level and previous_close <= breakout_level
    volume_breakout = volume_multiple >= settings.breakout_volume_multiple
    confirmed_breakout = box_valid and price_breakout and volume_breakout
    price_only_breakout = box_valid and price_breakout and not volume_breakout
    breakout_watch = box_valid and not price_breakout and latest_close >= near_breakout_floor

    if confirmed_breakout:
        state = "CONFIRMED BREAKOUT"
    elif price_only_breakout:
        state = "PRICE BREAKOUT / WEAK VOLUME"
    elif breakout_watch:
        state = "BREAKOUT WATCH"
    elif box_valid:
        state = "BUILDING A BOX"
    else:
        state = "NO VALID BOX"

    return {
        "valid": box_valid,
        "state": state,
        "box_high": box_high,
        "box_low": box_low,
        "box_range_pct": range_pct,
        "high_tests": high_tests,
        "low_tests": low_tests,
        "inside_ratio": inside_ratio,
        "breakout_level": breakout_level,
        "latest_close": latest_close,
        "previous_close": previous_close,
        "volume_multiple": volume_multiple,
        "price_breakout": price_breakout,
        "volume_breakout": volume_breakout,
        "confirmed_breakout": confirmed_breakout,
        "box_start": box.index[0],
        "box_end": box.index[-1],
        "checks": {
            "Range within limit": max_range_pass,
            "Enough upper-bound tests": high_tests_pass,
            "Enough lower-bound tests": low_tests_pass,
            "At least 90% closes contained": containment_pass,
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
    benchmark_ticker: str,
    benchmark_df: pd.DataFrame,
) -> dict[str, Any]:
    if asset_ticker == benchmark_ticker:
        returns = {
            "30-day return positive": safe_float(asset_df["Return_30D_Pct"].iloc[-1]) > 0,
            "90-day return positive": safe_float(asset_df["Return_90D_Pct"].iloc[-1]) > 0,
            "180-day return positive": safe_float(asset_df["Return_180D_Pct"].iloc[-1]) > 0,
        }
        passed = sum(returns.values())
        return {
            "label": f"{asset_ticker} momentum",
            "checks": returns,
            "passed": passed,
            "total": len(returns),
            "ratio_series": None,
            "latest_ratio": np.nan,
            "ratio_name": None,
        }

    aligned = pd.concat(
        [
            asset_df["Close"].rename("Asset"),
            benchmark_df["Close"].rename("Benchmark"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    aligned["Ratio"] = aligned["Asset"] / aligned["Benchmark"]
    aligned["SMA_50"] = aligned["Ratio"].rolling(50).mean()
    aligned["SMA_200"] = aligned["Ratio"].rolling(200).mean()
    aligned["Return_30D"] = aligned["Ratio"].pct_change(30) * 100
    aligned["Return_90D"] = aligned["Ratio"].pct_change(90) * 100

    latest = aligned.iloc[-1]
    ratio_name = f"{asset_ticker}/{benchmark_ticker}"
    checks = {
        f"{ratio_name} above 50-day average": latest["Ratio"] > latest["SMA_50"],
        f"{ratio_name} above 200-day average": latest["Ratio"] > latest["SMA_200"],
        f"{ratio_name} 30-day return positive": latest["Return_30D"] > 0,
        f"{ratio_name} 90-day return positive": latest["Return_90D"] > 0,
    }

    passed = sum(bool(value) for value in checks.values())
    return {
        "label": f"Relative strength versus {benchmark_ticker}",
        "checks": checks,
        "passed": passed,
        "total": len(checks),
        "ratio_series": aligned,
        "latest_ratio": safe_float(latest["Ratio"]),
        "ratio_name": ratio_name,
    }


def calculate_score(
    box_result: dict[str, Any],
    trend_result: dict[str, Any],
    dry_up_result: dict[str, Any],
    rs_result: dict[str, Any],
) -> dict[str, Any]:
    box_points = 25 if box_result["valid"] else 0
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


@st.cache_data(ttl=86400, show_spinner=False)
def load_universe(market: str) -> pd.DataFrame:
    """Load current S&P 500 or Nasdaq-100 constituents from Wikipedia."""
    if market == "S&P 500":
        table = pd.read_html(UNIVERSE_URLS[market])[0]
        result = table[["Symbol", "Security", "GICS Sector"]].copy()
        result.columns = ["Ticker", "Company", "Sector"]
    elif market == "Nasdaq-100":
        tables = pd.read_html(UNIVERSE_URLS[market])
        candidates = [t for t in tables if "Ticker" in t.columns and len(t) >= 90]
        if not candidates:
            raise ValueError("Could not locate the Nasdaq-100 constituents table.")
        table = candidates[0]
        company_col = "Company" if "Company" in table.columns else table.columns[0]
        result = table[["Ticker", company_col]].copy()
        result.columns = ["Ticker", "Company"]
        result["Sector"] = "Nasdaq-100"
    else:
        raise ValueError(f"No stock universe is defined for {market}.")

    result["Ticker"] = result["Ticker"].astype(str).str.replace(".", "-", regex=False)
    return result.drop_duplicates("Ticker").reset_index(drop=True)


@st.cache_data(ttl=900, show_spinner=False)
def download_batch_closes(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    data = yf.download(
        list(tickers), period=period, interval="1d", auto_adjust=False,
        progress=False, threads=True, group_by="column"
    )
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            return data["Close"].copy()
    if len(tickers) == 1 and "Close" in data.columns:
        return data[["Close"]].rename(columns={"Close": tickers[0]})
    return pd.DataFrame()


def scan_symbol(
    ticker: str,
    company: str,
    sector: str,
    settings: Settings,
    benchmark_ticker: str,
    benchmark_df: pd.DataFrame,
) -> dict[str, Any] | None:
    try:
        raw = download_market_data(ticker, settings.history_period)
        if len(raw) < max(221, settings.box_days + 2):
            return None
        df = add_indicators(raw, settings)
        box = detect_current_box(df, settings)
        trend = evaluate_trend_template(df, settings)
        dry = evaluate_volume_dry_up(df, settings)
        rs = evaluate_relative_strength(ticker, df, benchmark_ticker, benchmark_df)
        score = calculate_score(box, trend, dry, rs)
        latest = df.iloc[-1]
        return {
            "Ticker": ticker,
            "Company": company,
            "Sector": sector,
            "State": box["state"],
            "Score": score["Total"],
            "Price": safe_float(latest["Close"]),
            "Box High": box["box_high"],
            "Box Low": box["box_low"],
            "Box Range %": box["box_range_pct"],
            "Breakout Volume ×": box["volume_multiple"],
            "Trend Passed": f"{trend['passed']}/{trend['total']}",
            "Dry-Up": dry["pass"],
            "RS Passed": f"{rs['passed']}/{rs['total']}",
            "Distance from 365D High %": trend["distance_from_high_pct"],
            "Latest Date": df.index[-1].strftime("%Y-%m-%d"),
        }
    except Exception:
        return None


def render_asset_analysis(
    group: str,
    asset_name: str,
    ticker: str,
    settings: Settings,
) -> None:
    benchmark_ticker = BENCHMARKS[group]
    try:
        with st.spinner(f"Loading {asset_name} daily candles..."):
            raw_asset = download_market_data(ticker, settings.history_period)
            raw_benchmark = (
                raw_asset.copy() if ticker == benchmark_ticker
                else download_market_data(benchmark_ticker, settings.history_period)
            )
    except Exception as exc:
        st.error(f"Could not load market data: {exc}")
        return

    if raw_asset.empty or raw_benchmark.empty:
        st.error("No market data was returned. Try Refresh market data.")
        return

    asset_df = add_indicators(raw_asset, settings)
    benchmark_df = add_indicators(raw_benchmark, settings)
    if len(asset_df) < max(221, settings.box_days + 2):
        st.error("Insufficient daily history for the selected analysis.")
        return

    box_result = detect_current_box(asset_df, settings)
    trend_result = evaluate_trend_template(asset_df, settings)
    dry_up_result = evaluate_volume_dry_up(asset_df, settings)
    rs_result = evaluate_relative_strength(ticker, asset_df, benchmark_ticker, benchmark_df)
    score = calculate_score(box_result, trend_result, dry_up_result, rs_result)

    latest = asset_df.iloc[-1]
    prior = asset_df.iloc[-2]
    daily_change = (latest["Close"] / prior["Close"] - 1) * 100
    latest_date = asset_df.index[-1].strftime("%Y-%m-%d")
    caveat = (
        "Crypto candles may be incomplete before the UTC day closes."
        if group == "Crypto" else
        "For stocks and ETFs, the latest candle may be incomplete while the U.S. market is open."
    )
    st.info(f"Latest available daily candle: **{latest_date}**. {caveat}")

    metrics = st.columns(6)
    metrics[0].metric("Price", format_currency(latest["Close"]), f"{daily_change:.2f}%")
    metrics[1].metric("Strategy Score", f"{score['Total']}/100")
    metrics[2].metric("State", box_result["state"])
    metrics[3].metric("Box High", format_currency(box_result["box_high"]))
    metrics[4].metric("Box Low", format_currency(box_result["box_low"]))
    metrics[5].metric("Breakout Volume", f"{box_result['volume_multiple']:.2f}×")

    tabs = st.tabs(["Overview", "Darvas Box", "Trend Template", "Volume Dry-Up", "Relative Strength", "Raw Data"])
    with tabs[0]:
        st.plotly_chart(make_price_chart(asset_df, box_result, settings, asset_name), use_container_width=True)
        st.plotly_chart(make_volume_chart(asset_df, settings), use_container_width=True)
        st.subheader("Score Breakdown")
        st.dataframe(pd.DataFrame([{"Component": k, "Points": v} for k, v in score.items() if k != "Total"]), hide_index=True, use_container_width=True)
        if box_result["confirmed_breakout"]:
            st.success("The latest candle meets the configured box, price-breakout and volume-confirmation rules.")
        elif box_result["state"] == "PRICE BREAKOUT / WEAK VOLUME":
            st.warning("Price cleared the breakout level, but volume confirmation failed.")
        elif box_result["state"] == "BREAKOUT WATCH":
            st.warning("Price is within 2% of the current box high.")
        elif box_result["valid"]:
            st.info("A valid box is present, but price is not yet near a confirmed breakout.")
        else:
            st.error("The selected lookback does not currently form a valid Darvas box.")

    with tabs[1]:
        details = {
            "Status": box_result["state"], "Box start": box_result["box_start"].strftime("%Y-%m-%d"),
            "Box end": box_result["box_end"].strftime("%Y-%m-%d"), "Box high": format_currency(box_result["box_high"]),
            "Box low": format_currency(box_result["box_low"]), "Box range": f"{box_result['box_range_pct']:.2f}%",
            "Upper-bound tests": box_result["high_tests"], "Lower-bound tests": box_result["low_tests"],
            "Closes contained": f"{box_result['inside_ratio'] * 100:.1f}%", "Breakout level": format_currency(box_result["breakout_level"]),
            "Price breakout": "Yes" if box_result["price_breakout"] else "No", "Volume confirmation": "Yes" if box_result["volume_breakout"] else "No",
        }
        st.dataframe(pd.DataFrame(details.items(), columns=["Measure", "Value"]), hide_index=True, use_container_width=True)
        render_checks("Box Qualification", box_result["checks"])

    with tabs[2]:
        left, right = st.columns(2)
        with left:
            render_checks(f"Trend Rules: {trend_result['passed']}/{trend_result['total']} Passed", trend_result["checks"])
        with right:
            st.dataframe(pd.DataFrame([
                {"Measure": "50-day SMA", "Value": format_currency(trend_result["sma_50"])},
                {"Measure": "150-day SMA", "Value": format_currency(trend_result["sma_150"])},
                {"Measure": "200-day SMA", "Value": format_currency(trend_result["sma_200"])},
                {"Measure": "Distance from 365-day high", "Value": f"{trend_result['distance_from_high_pct']:.2f}%"},
            ]), hide_index=True, use_container_width=True)

    with tabs[3]:
        st.dataframe(pd.DataFrame([
            {"Measure": "Recent/baseline dollar-volume ratio", "Value": f"{dry_up_result['ratio']:.2f}"},
            {"Measure": "Recent average dollar volume", "Value": format_currency(dry_up_result.get("recent_average", np.nan))},
            {"Measure": "Baseline average dollar volume", "Value": format_currency(dry_up_result.get("baseline_average", np.nan))},
            {"Measure": "Recent average ATR %", "Value": f"{dry_up_result.get('recent_atr_pct', np.nan):.2f}%"},
            {"Measure": "Prior average ATR %", "Value": f"{dry_up_result.get('prior_atr_pct', np.nan):.2f}%"},
            {"Measure": "Volume + ATR dry-up passes", "Value": "Yes" if dry_up_result["pass"] else "No"},
        ]), hide_index=True, use_container_width=True)

    with tabs[4]:
        render_checks(f"{rs_result['label']}: {rs_result['passed']}/{rs_result['total']} Passed", rs_result["checks"])
        if rs_result["ratio_series"] is not None:
            ratio = rs_result["ratio_series"].tail(settings.chart_days)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ratio.index, y=ratio["Ratio"], mode="lines", name=rs_result["ratio_name"]))
            fig.add_trace(go.Scatter(x=ratio.index, y=ratio["SMA_50"], mode="lines", name="50-day average"))
            fig.add_trace(go.Scatter(x=ratio.index, y=ratio["SMA_200"], mode="lines", name="200-day average"))
            fig.update_layout(title=f"Relative Strength: {rs_result['ratio_name']}", height=450, legend={"orientation": "h"})
            st.plotly_chart(fig, use_container_width=True)

    with tabs[5]:
        cols = ["Open", "High", "Low", "Close", "Volume", "Dollar_Volume", "SMA_50", "SMA_150", "SMA_200", "ATR_Pct", "Distance_From_365D_High_Pct"]
        export = asset_df[cols].copy().sort_index(ascending=False)
        st.dataframe(export, use_container_width=True)
        st.download_button("Download analyzed CSV", export.to_csv().encode("utf-8"), f"{ticker.replace('-', '_').replace('^','')}_darvas_minervini.csv", "text/csv")


def render_universe_scanner(market: str, settings: Settings) -> None:
    st.subheader(f"{market} Constituent Scanner")
    st.caption("The scan runs only when you press the button. Results are current-state signals, not backtests.")

    try:
        universe = load_universe(market)
    except Exception as exc:
        st.error(f"Could not load the {market} constituent list: {exc}")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        states = st.multiselect("States to display", ["CONFIRMED BREAKOUT", "PRICE BREAKOUT / WEAK VOLUME", "BREAKOUT WATCH", "BUILDING A BOX", "NO VALID BOX"], default=["CONFIRMED BREAKOUT", "PRICE BREAKOUT / WEAK VOLUME", "BREAKOUT WATCH", "BUILDING A BOX"])
    with c2:
        minimum_score = st.slider("Minimum score", 0, 100, 55)
    with c3:
        scan_limit = st.number_input("Maximum symbols to scan", 10, len(universe), min(len(universe), 100), 10)

    st.caption(f"Universe currently contains {len(universe)} symbols. Increase the limit to scan the full universe; free data downloads can take longer or occasionally rate-limit.")
    key = f"scan_results_{market}"
    if st.button(f"Scan {market}", type="primary", use_container_width=True):
        benchmark_ticker = BENCHMARKS[market]
        raw_benchmark = download_market_data(benchmark_ticker, settings.history_period)
        benchmark_df = add_indicators(raw_benchmark, settings)
        rows = []
        progress = st.progress(0)
        status = st.empty()
        subset = universe.head(int(scan_limit))
        for i, record in subset.iterrows():
            status.write(f"Scanning {record['Ticker']} — {i + 1} of {len(subset)}")
            row = scan_symbol(record["Ticker"], record["Company"], record["Sector"], settings, benchmark_ticker, benchmark_df)
            if row:
                rows.append(row)
            progress.progress((i + 1) / len(subset))
        status.empty(); progress.empty()
        st.session_state[key] = pd.DataFrame(rows)

    results = st.session_state.get(key)
    if isinstance(results, pd.DataFrame) and not results.empty:
        filtered = results[(results["Score"] >= minimum_score) & (results["State"].isin(states))].copy()
        filtered = filtered.sort_values(["Score", "State", "Ticker"], ascending=[False, True, True])
        st.metric("Matching candidates", len(filtered), f"of {len(results)} successfully analyzed")
        st.dataframe(filtered, hide_index=True, use_container_width=True, column_config={
            "Price": st.column_config.NumberColumn(format="$%.2f"),
            "Box High": st.column_config.NumberColumn(format="$%.2f"),
            "Box Low": st.column_config.NumberColumn(format="$%.2f"),
            "Box Range %": st.column_config.NumberColumn(format="%.2f%%"),
            "Breakout Volume ×": st.column_config.NumberColumn(format="%.2fx"),
            "Distance from 365D High %": st.column_config.NumberColumn(format="%.2f%%"),
        })
        st.download_button(f"Download {market} scan CSV", filtered.to_csv(index=False).encode("utf-8"), f"{market.lower().replace(' ','_').replace('&','and')}_darvas_minervini_scan.csv", "text/csv")
    elif isinstance(results, pd.DataFrame):
        st.warning("The scan completed, but no symbols were successfully analyzed.")


def main() -> None:
    st.title("📦 Darvas + Minervini Market Scanner")
    st.caption("Current-state analysis for BTC, ETH, S&P 500 and Nasdaq-100 markets. No backtesting and no return prediction.")

    with st.sidebar:
        st.header("Market")
        group = st.selectbox("Market mode", list(ASSET_GROUPS))
        asset_name = st.selectbox("Asset or index", list(ASSET_GROUPS[group]))
        ticker = ASSET_GROUPS[group][asset_name]

        st.header("Darvas Box")
        default_range = 15.0 if group == "Crypto" else 12.0
        box_days = st.slider("Box lookback days", 10, 90, 30)
        max_box_range_pct = st.slider("Maximum box range (%)", 3.0, 35.0, default_range, 0.5)
        test_tolerance_pct = st.slider("Boundary test tolerance (%)", 0.25, 5.0, 1.5, 0.25)
        minimum_high_tests = st.slider("Minimum upper-bound tests", 1, 6, 2)
        minimum_low_tests = st.slider("Minimum lower-bound tests", 1, 6, 2)

        st.header("Breakout")
        breakout_buffer_pct = st.slider("Breakout buffer (%)", 0.0, 5.0, 0.5, 0.1)
        breakout_volume_multiple = st.slider("Minimum volume multiple", 1.0, 5.0, 1.5, 0.1)

        st.header("Volume Dry-Up")
        dry_up_days = st.slider("Recent dry-up days", 3, 30, 10)
        baseline_volume_days = st.slider("Baseline volume days", 10, 90, 30)
        dry_up_ratio_max = st.slider("Maximum recent/baseline volume ratio", 0.25, 1.0, 0.70, 0.05)

        st.header("Trend")
        near_high_pct = st.slider("Maximum distance from 365-day high (%)", 5, 50, 25)
        atr_days = st.slider("ATR period", 5, 30, 14)
        chart_days = st.slider("Chart history days", 90, 730, 365, 30)
        if st.button("Refresh market data", use_container_width=True):
            st.cache_data.clear()

    settings = Settings("3y", box_days, max_box_range_pct, test_tolerance_pct, minimum_high_tests, minimum_low_tests, breakout_buffer_pct, breakout_volume_multiple, dry_up_days, baseline_volume_days, dry_up_ratio_max, atr_days, near_high_pct, chart_days)

    analysis_tab, scanner_tab = st.tabs(["Asset / Index Analysis", "Market Scanner"])
    with analysis_tab:
        render_asset_analysis(group, asset_name, ticker, settings)
    with scanner_tab:
        if group == "Crypto":
            st.info("The market scanner is available for the S&P 500 and Nasdaq-100 modes. BTC and ETH remain available in the analysis tab.")
        else:
            render_universe_scanner(group, settings)

    with st.expander("Methodology and limitations"):
        st.markdown("""
        - The Darvas box uses completed candles before the latest candle; the latest candle is evaluated independently as the possible breakout.
        - The Minervini-style trend template uses 50-, 150- and 200-day averages, a rising 200-day average, the 365-day range midpoint and distance from the 365-day high.
        - S&P 500 stocks are compared with SPY. Nasdaq-100 stocks are compared with QQQ. Ethereum is compared with Bitcoin.
        - Volume dry-up uses recent dollar volume versus an earlier baseline and also requires ATR percentage contraction.
        - The scanner downloads public market data on demand. Constituents and market data can occasionally fail or be rate-limited.
        """)
    st.caption("Educational scanner only—not investment advice. Signals can fail and market data may be delayed or incomplete.")


if __name__ == "__main__":
    main()
