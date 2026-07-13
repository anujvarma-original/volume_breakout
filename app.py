from __future__ import annotations

import time
from dataclasses import dataclass
from io import StringIO

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="S&P 500 Volume Breakout Scanner",
    page_icon="📈",
    layout="wide",
)

SP500_CSV_URL = (
    "https://raw.githubusercontent.com/datasets/"
    "s-and-p-500-companies/main/data/constituents.csv"
)


@dataclass(frozen=True)
class ScannerConfig:
    quiet_days: int
    baseline_days: int
    recent_days: int
    spike_multiplier: float
    quiet_ratio: float
    low_day_fraction: float
    max_quiet_cv: float
    min_median_volume: int
    require_price_breakout: bool
    require_bullish_candle: bool
    batch_size: int
    pause_seconds: float


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def load_sp500_constituents() -> pd.DataFrame:
    """Load the current S&P 500 constituent list."""
    response = requests.get(SP500_CSV_URL, timeout=30)
    response.raise_for_status()

    frame = pd.read_csv(StringIO(response.text))

    required = {"Symbol", "Security", "GICS Sector"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Constituent file is missing columns: {sorted(missing)}")

    frame = frame.rename(
        columns={
            "Symbol": "Ticker",
            "Security": "Company",
            "GICS Sector": "Sector",
        }
    )

    frame["Ticker"] = (
        frame["Ticker"]
        .astype(str)
        .str.strip()
        .str.replace(".", "-", regex=False)
    )

    return (
        frame[["Ticker", "Company", "Sector"]]
        .drop_duplicates("Ticker")
        .sort_values("Ticker")
        .reset_index(drop=True)
    )


def download_batch(tickers: list[str], period: str = "6mo") -> pd.DataFrame:
    """Download one batch without caching partial failures."""
    return yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        actions=False,
        threads=True,
        progress=False,
        timeout=30,
    )


def extract_ticker_frame(
    downloaded: pd.DataFrame,
    ticker: str,
    batch_length: int,
) -> pd.DataFrame:
    """Extract and normalize one ticker from a yfinance batch response."""
    if downloaded is None or downloaded.empty:
        raise ValueError("No data returned")

    if batch_length == 1:
        frame = downloaded.copy()
    else:
        if not isinstance(downloaded.columns, pd.MultiIndex):
            raise ValueError("Unexpected multi-ticker response format")

        level_zero = downloaded.columns.get_level_values(0)
        if ticker not in level_zero:
            raise ValueError("Ticker missing from batch response")

        frame = downloaded[ticker].copy()

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")

    frame = frame[required].copy()
    frame = frame.dropna(subset=required)
    frame = frame[frame["Volume"] > 0]
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame = frame.sort_index()

    if frame.empty:
        raise ValueError("No usable OHLCV rows")

    return frame


def quiet_metrics(
    quiet_window: pd.DataFrame,
    baseline_window: pd.DataFrame,
) -> dict[str, float]:
    quiet_median = float(quiet_window["Volume"].median())
    baseline_median = float(baseline_window["Volume"].median())
    quiet_mean = float(quiet_window["Volume"].mean())
    quiet_std = float(quiet_window["Volume"].std(ddof=0))

    return {
        "quiet_median": quiet_median,
        "baseline_median": baseline_median,
        "quiet_ratio": (
            quiet_median / baseline_median if baseline_median > 0 else np.inf
        ),
        "low_day_fraction": float(
            (quiet_window["Volume"] < baseline_median).mean()
        ),
        "quiet_cv": quiet_std / quiet_mean if quiet_mean > 0 else np.inf,
    }


def is_quiet_setup(metrics: dict[str, float], config: ScannerConfig) -> bool:
    return (
        metrics["quiet_median"] >= config.min_median_volume
        and metrics["quiet_ratio"] <= config.quiet_ratio
        and metrics["low_day_fraction"] >= config.low_day_fraction
        and metrics["quiet_cv"] <= config.max_quiet_cv
    )


def analyze_recent_spikes(
    ticker: str,
    company: str,
    sector: str,
    frame: pd.DataFrame,
    config: ScannerConfig,
) -> list[dict]:
    """
    Test each of the latest N sessions independently.

    For every candidate spike date, the quiet and baseline windows are taken
    from dates strictly before that candle. This prevents look-ahead bias.
    """
    results: list[dict] = []
    minimum_rows = config.baseline_days + config.quiet_days + config.recent_days

    if len(frame) < minimum_rows:
        return results

    first_candidate = len(frame) - config.recent_days

    for spike_position in range(first_candidate, len(frame)):
        quiet_end = spike_position
        quiet_start = quiet_end - config.quiet_days
        baseline_end = quiet_start
        baseline_start = baseline_end - config.baseline_days

        if baseline_start < 0:
            continue

        baseline = frame.iloc[baseline_start:baseline_end]
        quiet = frame.iloc[quiet_start:quiet_end]
        spike = frame.iloc[spike_position]

        metrics = quiet_metrics(quiet, baseline)
        if not is_quiet_setup(metrics, config):
            continue

        volume_multiple = (
            float(spike["Volume"]) / metrics["quiet_median"]
            if metrics["quiet_median"] > 0
            else 0.0
        )

        if volume_multiple < config.spike_multiplier:
            continue

        quiet_high = float(quiet["High"].max())
        price_breakout = float(spike["Close"]) > quiet_high
        bullish = float(spike["Close"]) > float(spike["Open"])

        if config.require_price_breakout and not price_breakout:
            continue
        if config.require_bullish_candle and not bullish:
            continue

        candle_range = float(spike["High"] - spike["Low"])
        body = abs(float(spike["Close"] - spike["Open"]))

        results.append(
            {
                "Ticker": ticker,
                "Company": company,
                "Sector": sector,
                "Spike Date": frame.index[spike_position].date(),
                "Trading Days Ago": len(frame) - 1 - spike_position,
                "Volume Multiple": round(volume_multiple, 2),
                "Spike Volume": int(spike["Volume"]),
                "Quiet Median Volume": int(metrics["quiet_median"]),
                "Quiet/Baseline Ratio": round(metrics["quiet_ratio"], 3),
                "Low-Volume Days %": round(
                    metrics["low_day_fraction"] * 100, 1
                ),
                "Quiet Volume CV": round(metrics["quiet_cv"], 3),
                "Open": round(float(spike["Open"]), 2),
                "High": round(float(spike["High"]), 2),
                "Low": round(float(spike["Low"]), 2),
                "Close": round(float(spike["Close"]), 2),
                "Candle Change %": round(
                    (float(spike["Close"]) / float(spike["Open"]) - 1) * 100,
                    2,
                ),
                "Bullish Candle": bullish,
                "Price Breakout": price_breakout,
                "Close vs Quiet High %": round(
                    (float(spike["Close"]) / quiet_high - 1) * 100,
                    2,
                ),
                "Body/Range": round(body / candle_range, 3)
                if candle_range > 0
                else 0.0,
            }
        )

    return results


def analyze_current_quiet_setup(
    ticker: str,
    company: str,
    sector: str,
    frame: pd.DataFrame,
    config: ScannerConfig,
) -> dict | None:
    """Check whether the latest completed sessions are still in a quiet setup."""
    required_rows = config.baseline_days + config.quiet_days
    if len(frame) < required_rows:
        return None

    quiet = frame.iloc[-config.quiet_days:]
    baseline = frame.iloc[
        -(config.baseline_days + config.quiet_days):-config.quiet_days
    ]

    metrics = quiet_metrics(quiet, baseline)
    if not is_quiet_setup(metrics, config):
        return None

    latest = frame.iloc[-1]
    recent_max_multiple = float(
        (quiet["Volume"] / metrics["quiet_median"]).max()
    )

    # "Still quiet" means the current quiet window itself has not already
    # produced a candle meeting the configured huge-spike threshold.
    if recent_max_multiple >= config.spike_multiplier:
        return None

    quiet_high = float(quiet["High"].max())
    latest_close = float(latest["Close"])

    return {
        "Ticker": ticker,
        "Company": company,
        "Sector": sector,
        "Latest Date": frame.index[-1].date(),
        "Latest Close": round(latest_close, 2),
        "Latest Volume": int(latest["Volume"]),
        "Quiet Median Volume": int(metrics["quiet_median"]),
        "Latest Volume Multiple": round(
            float(latest["Volume"]) / metrics["quiet_median"], 2
        ),
        "Quiet/Baseline Ratio": round(metrics["quiet_ratio"], 3),
        "Low-Volume Days %": round(metrics["low_day_fraction"] * 100, 1),
        "Quiet Volume CV": round(metrics["quiet_cv"], 3),
        "Quiet Range High": round(quiet_high, 2),
        "Distance to Quiet High %": round(
            (quiet_high / latest_close - 1) * 100, 2
        )
        if latest_close > 0
        else np.nan,
    }


def run_scan(
    constituents: pd.DataFrame,
    config: ScannerConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    breakout_rows: list[dict] = []
    quiet_rows: list[dict] = []
    error_rows: list[dict] = []

    progress = st.progress(0, text="Preparing scan...")
    status = st.empty()

    tickers = constituents["Ticker"].tolist()
    lookup = constituents.set_index("Ticker")
    batches = [
        tickers[i:i + config.batch_size]
        for i in range(0, len(tickers), config.batch_size)
    ]

    processed = 0

    for batch_number, batch in enumerate(batches, start=1):
        status.write(
            f"Downloading batch {batch_number} of {len(batches)} "
            f"({len(batch)} stocks)"
        )

        try:
            downloaded = download_batch(batch)
        except Exception as exc:
            for ticker in batch:
                error_rows.append(
                    {
                        "Ticker": ticker,
                        "Stage": "Download",
                        "Error": str(exc),
                    }
                )
            processed += len(batch)
            progress.progress(
                min(processed / len(tickers), 1.0),
                text=f"Processed {processed} of {len(tickers)} stocks",
            )
            continue

        for ticker in batch:
            try:
                frame = extract_ticker_frame(downloaded, ticker, len(batch))
                metadata = lookup.loc[ticker]

                breakout_rows.extend(
                    analyze_recent_spikes(
                        ticker=ticker,
                        company=metadata["Company"],
                        sector=metadata["Sector"],
                        frame=frame,
                        config=config,
                    )
                )

                quiet_result = analyze_current_quiet_setup(
                    ticker=ticker,
                    company=metadata["Company"],
                    sector=metadata["Sector"],
                    frame=frame,
                    config=config,
                )
                if quiet_result:
                    quiet_rows.append(quiet_result)

            except Exception as exc:
                error_rows.append(
                    {
                        "Ticker": ticker,
                        "Stage": "Analysis",
                        "Error": str(exc),
                    }
                )

            processed += 1
            progress.progress(
                min(processed / len(tickers), 1.0),
                text=f"Processed {processed} of {len(tickers)} stocks",
            )

        if batch_number < len(batches) and config.pause_seconds > 0:
            time.sleep(config.pause_seconds)

    progress.empty()
    status.empty()

    breakouts = pd.DataFrame(breakout_rows)
    quiet = pd.DataFrame(quiet_rows)
    errors = pd.DataFrame(error_rows)

    if not breakouts.empty:
        # Keep the strongest qualifying spike per ticker.
        breakouts = (
            breakouts.sort_values(
                ["Ticker", "Volume Multiple", "Spike Date"],
                ascending=[True, False, False],
            )
            .drop_duplicates("Ticker", keep="first")
            .sort_values(
                ["Volume Multiple", "Candle Change %"],
                ascending=[False, False],
            )
            .reset_index(drop=True)
        )

    if not quiet.empty:
        quiet = quiet.sort_values(
            ["Quiet/Baseline Ratio", "Distance to Quiet High %"],
            ascending=[True, True],
        ).reset_index(drop=True)

    return breakouts, quiet, errors


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


st.title("📈 S&P 500 Volume Breakout Scanner")
st.caption(
    "Find stocks emerging from consistently low volume, plus stocks that "
    "remain compressed and may be worth monitoring."
)

with st.sidebar:
    st.header("Scanner settings")

    quiet_days = st.number_input(
        "Quiet period (trading days)",
        min_value=5,
        max_value=60,
        value=20,
        step=1,
    )

    baseline_days = st.number_input(
        "Historical baseline (trading days)",
        min_value=20,
        max_value=180,
        value=60,
        step=5,
    )

    recent_days = st.number_input(
        "Recent spike window (trading days)",
        min_value=1,
        max_value=20,
        value=7,
        step=1,
    )

    spike_multiplier = st.slider(
        "Huge-volume multiplier",
        min_value=2.0,
        max_value=10.0,
        value=4.0,
        step=0.25,
    )

    quiet_ratio = st.slider(
        "Maximum quiet/baseline volume ratio",
        min_value=0.20,
        max_value=1.00,
        value=0.65,
        step=0.05,
        help=(
            "A value of 0.65 means quiet-period median volume must be "
            "65% or less of the earlier baseline median."
        ),
    )

    low_day_percent = st.slider(
        "Minimum low-volume days",
        min_value=50,
        max_value=100,
        value=80,
        step=5,
    )

    max_quiet_cv = st.slider(
        "Maximum quiet-volume variability",
        min_value=0.10,
        max_value=1.00,
        value=0.40,
        step=0.05,
        help=(
            "Lower values require more consistent volume during the "
            "quiet period."
        ),
    )

    min_median_volume = st.number_input(
        "Minimum median daily volume",
        min_value=0,
        max_value=10_000_000,
        value=100_000,
        step=50_000,
    )

    require_price_breakout = st.checkbox(
        "Require close above quiet-period high",
        value=False,
    )

    require_bullish_candle = st.checkbox(
        "Require bullish spike candle",
        value=False,
    )

    with st.expander("Download settings"):
        batch_size = st.number_input(
            "Tickers per batch",
            min_value=10,
            max_value=150,
            value=50,
            step=10,
        )
        pause_seconds = st.number_input(
            "Pause between batches (seconds)",
            min_value=0.0,
            max_value=5.0,
            value=0.5,
            step=0.5,
        )

    run_button = st.button(
        "Run S&P 500 scan",
        type="primary",
        use_container_width=True,
    )

st.info(
    "**Volume breakout** means a huge daily volume candle after a qualifying "
    "quiet period. It does not require a price breakout unless that sidebar "
    "option is enabled."
)

try:
    constituents = load_sp500_constituents()
    st.caption(f"Universe loaded: {len(constituents)} S&P 500 securities.")
except Exception as exc:
    st.error(f"Could not load the S&P 500 list: {exc}")
    st.stop()

if run_button:
    config = ScannerConfig(
        quiet_days=int(quiet_days),
        baseline_days=int(baseline_days),
        recent_days=int(recent_days),
        spike_multiplier=float(spike_multiplier),
        quiet_ratio=float(quiet_ratio),
        low_day_fraction=float(low_day_percent) / 100,
        max_quiet_cv=float(max_quiet_cv),
        min_median_volume=int(min_median_volume),
        require_price_breakout=bool(require_price_breakout),
        require_bullish_candle=bool(require_bullish_candle),
        batch_size=int(batch_size),
        pause_seconds=float(pause_seconds),
    )

    try:
        breakouts, quiet, errors = run_scan(constituents, config)
        st.session_state["breakouts"] = breakouts
        st.session_state["quiet"] = quiet
        st.session_state["errors"] = errors
        st.session_state["last_config"] = config
    except Exception as exc:
        st.exception(exc)

if "breakouts" in st.session_state:
    breakouts = st.session_state["breakouts"]
    quiet = st.session_state["quiet"]
    errors = st.session_state["errors"]

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Recent spike stocks", len(breakouts))
    metric_2.metric("Still-low-volume stocks", len(quiet))
    metric_3.metric("Download/analysis errors", len(errors))

    breakout_tab, quiet_tab, error_tab, rules_tab = st.tabs(
        [
            "Recent volume spikes",
            "Still low volume",
            "Errors",
            "Rule definitions",
        ]
    )

    with breakout_tab:
        st.subheader("Recent huge-volume spike candles")

        if breakouts.empty:
            st.warning("No stocks matched the current breakout settings.")
        else:
            sectors = sorted(breakouts["Sector"].dropna().unique())
            chosen_sectors = st.multiselect(
                "Filter breakout sectors",
                sectors,
                default=sectors,
                key="breakout_sector_filter",
            )
            visible_breakouts = breakouts[
                breakouts["Sector"].isin(chosen_sectors)
            ]

            st.dataframe(
                visible_breakouts,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Volume Multiple": st.column_config.NumberColumn(
                        format="%.2fx"
                    ),
                    "Candle Change %": st.column_config.NumberColumn(
                        format="%.2f%%"
                    ),
                    "Close vs Quiet High %": st.column_config.NumberColumn(
                        format="%.2f%%"
                    ),
                },
            )

            st.download_button(
                "Download breakout results",
                data=csv_bytes(visible_breakouts),
                file_name="recent_volume_spikes.csv",
                mime="text/csv",
            )

    with quiet_tab:
        st.subheader("Stocks still in a low-volume setup")

        if quiet.empty:
            st.warning("No stocks matched the current quiet-volume settings.")
        else:
            sectors = sorted(quiet["Sector"].dropna().unique())
            chosen_sectors = st.multiselect(
                "Filter quiet-setup sectors",
                sectors,
                default=sectors,
                key="quiet_sector_filter",
            )
            visible_quiet = quiet[quiet["Sector"].isin(chosen_sectors)]

            st.dataframe(
                visible_quiet,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Latest Volume Multiple": st.column_config.NumberColumn(
                        format="%.2fx"
                    ),
                    "Distance to Quiet High %": st.column_config.NumberColumn(
                        format="%.2f%%"
                    ),
                },
            )

            st.download_button(
                "Download quiet-stock results",
                data=csv_bytes(visible_quiet),
                file_name="still_low_volume.csv",
                mime="text/csv",
            )

    with error_tab:
        if errors.empty:
            st.success("No download or analysis errors were recorded.")
        else:
            st.dataframe(errors, use_container_width=True, hide_index=True)
            st.download_button(
                "Download error report",
                data=csv_bytes(errors),
                file_name="scan_errors.csv",
                mime="text/csv",
            )

    with rules_tab:
        st.markdown(
            """
            ### Recent volume spike

            For each of the latest configured sessions, the scanner:

            1. Looks backward over the immediately preceding quiet period.
            2. Compares that period with an older historical baseline.
            3. Confirms that volume was low and reasonably consistent.
            4. Tests whether the candidate candle's volume exceeds the quiet
               median by the configured spike multiple.
            5. Optionally requires a bullish candle or a close above the
               quiet-period high.

            Each candidate date uses only data available before that candle.

            ### Still low volume

            A stock appears here when its latest quiet-period window passes
            the low-volume tests and no candle inside that window has already
            reached the configured huge-volume threshold.
            """
        )

else:
    st.write("Choose the scanner settings and select **Run S&P 500 scan**.")

st.divider()
st.caption(
    "Market data is obtained through yfinance for research and educational "
    "use. Screening results are not investment advice."
)
