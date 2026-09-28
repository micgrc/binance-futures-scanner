"""Binance USDT-M Futures Trend Scanner & Backtester.

Run locally:
    streamlit run app.py

This application uses public Binance Futures REST endpoints only. It does not
need API keys and does not place orders.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests
import streamlit as st


BINANCE_FUTURES_URL = "https://fapi.binance.com"
REQUEST_TIMEOUT = 15
KLINE_LIMIT = 500
VOLUME_MA_LENGTH = 20
TP_R_MULTIPLE = 2.0
TP1_POSITION_FRACTION = 0.50
# Treat 0.04% as the all-in trading cost on each executed side.
COST_PER_SIDE = 0.0004


st.set_page_config(
    page_title="Binance Futures Trend Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1200px;}
      [data-testid="stMetric"] {background: #111827; border: 1px solid #263244;
          border-radius: 12px; padding: .8rem;}
      [data-testid="stDataFrame"] {border-radius: 12px; overflow: hidden;}
      .small-note {color: #94a3b8; font-size: .82rem; line-height: 1.4;}
      @media (max-width: 640px) {
        .block-container {padding: .7rem .55rem 1.5rem;}
        h1 {font-size: 1.65rem !important; line-height: 1.15 !important;}
        [data-testid="stMetric"] {padding: .55rem;}
        [data-testid="stMetricValue"] {font-size: 1.15rem;}
        button[kind="primary"] {min-height: 3rem; width: 100%;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)


class BinanceAPIError(RuntimeError):
    """Raised when Binance returns an error or unusable response."""


@dataclass
class Trade:
    """A completed backtest trade."""

    direction: int  # 1 for long, -1 for short
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    net_return: float


def _get_json(path: str, params: dict[str, Any] | None = None) -> Any:
    """GET JSON from Binance with small retries for transient failures."""
    last_error: Exception | None = None
    headers = {"User-Agent": "Mozilla/5.0 Streamlit-Binance-Scanner/1.0"}

    for attempt in range(3):
        try:
            response = requests.get(
                f"{BINANCE_FUTURES_URL}{path}",
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.6 * (2**attempt))

    raise BinanceAPIError(f"Binance request failed: {last_error}")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_active_usdt_perpetuals() -> list[str]:
    """Return every currently trading USDT-margined perpetual symbol."""
    payload = _get_json("/fapi/v1/exchangeInfo")
    symbols = [
        item["symbol"]
        for item in payload.get("symbols", [])
        if item.get("status") == "TRADING"
        and item.get("contractType") == "PERPETUAL"
        and item.get("quoteAsset") == "USDT"
    ]
    if not symbols:
        raise BinanceAPIError("Binance returned no active USDT perpetual pairs.")
    return symbols


@st.cache_data(ttl=120, show_spinner=False)
def fetch_top_symbols(limit: int) -> list[str]:
    """Select the most liquid active contracts by 24-hour quote volume."""
    active = set(fetch_active_usdt_perpetuals())
    tickers = _get_json("/fapi/v1/ticker/24hr")
    ranked: list[tuple[str, float]] = []

    for ticker in tickers:
        symbol = ticker.get("symbol", "")
        if symbol not in active:
            continue
        try:
            ranked.append((symbol, float(ticker.get("quoteVolume", 0))))
        except (TypeError, ValueError):
            continue

    ranked.sort(key=lambda item: item[1], reverse=True)
    return [symbol for symbol, _ in ranked[:limit]]


@st.cache_data(ttl=300, show_spinner=False)
def fetch_klines(symbol: str, interval: str, limit: int = KLINE_LIMIT) -> pd.DataFrame:
    """Download and normalize public futures candlesticks."""
    rows = _get_json(
        "/fapi/v1/klines",
        {"symbol": symbol, "interval": interval, "limit": limit},
    )
    if not isinstance(rows, list) or not rows:
        raise BinanceAPIError(f"No candle data returned for {symbol}.")

    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trade_count", "taker_base",
        "taker_quote", "ignore",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    numeric = ["open", "high", "low", "close", "volume"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame = frame[["open_time", *numeric]].dropna().reset_index(drop=True)
    return frame


def add_indicators(
    frame: pd.DataFrame,
    fast_span: int,
    slow_span: int,
    atr_length: int = 14,
) -> pd.DataFrame:
    """Calculate EMA, volume MA and Wilder-style ATR without extra packages."""
    data = frame.copy()
    data["fast_ema"] = data["close"].ewm(span=fast_span, adjust=False).mean()
    data["slow_ema"] = data["close"].ewm(span=slow_span, adjust=False).mean()
    data["volume_ma"] = data["volume"].rolling(VOLUME_MA_LENGTH).mean()

    previous_close = data["close"].shift(1)
    true_range = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    data["atr"] = true_range.ewm(
        alpha=1 / atr_length,
        adjust=False,
        min_periods=atr_length,
    ).mean()

    volume_ok = data["volume"] > data["volume_ma"]
    data["signal"] = 0
    data.loc[(data["fast_ema"] > data["slow_ema"]) & volume_ok, "signal"] = 1
    data.loc[(data["fast_ema"] < data["slow_ema"]) & volume_ok, "signal"] = -1
    return data


def _leg_return(direction: int, entry: float, exit_price: float, fraction: float) -> float:
    """Net contribution of one exit leg, including proportional exit cost."""
    gross = direction * (exit_price - entry) / entry
    return fraction * gross - fraction * COST_PER_SIDE


def backtest(frame: pd.DataFrame, atr_multiplier: float) -> list[Trade]:
    """Backtest one position at a time using next-open entries.

    Intrabar ambiguity is handled conservatively: if stop and target are both
    touched in one candle, the stop is assumed to have occurred first.
    """
    trades: list[Trade] = []
    warmup = max(VOLUME_MA_LENGTH, 14) + 1
    i = warmup

    while i < len(frame) - 1:
        signal = int(frame.at[i, "signal"])
        signal_atr = float(frame.at[i, "atr"])
        if signal == 0 or pd.isna(signal_atr) or signal_atr <= 0:
            i += 1
            continue

        entry_i = i + 1
        entry = float(frame.at[entry_i, "open"])
        risk_distance = atr_multiplier * signal_atr
        if entry <= 0 or risk_distance <= 0:
            i += 1
            continue

        stop = entry - signal * risk_distance
        target = entry + signal * TP_R_MULTIPLE * risk_distance
        remaining = 1.0
        trade_return = -COST_PER_SIDE  # Entry cost applies to the full position.
        exit_i = entry_i

        for j in range(entry_i, len(frame)):
            high = float(frame.at[j, "high"])
            low = float(frame.at[j, "low"])
            close = float(frame.at[j, "close"])
            bar_atr = float(frame.at[j, "atr"])
            stop_touched = low <= stop if signal == 1 else high >= stop
            target_touched = (
                remaining == 1.0
                and (high >= target if signal == 1 else low <= target)
            )

            if stop_touched:
                trade_return += _leg_return(signal, entry, stop, remaining)
                remaining = 0.0
                exit_i = j
                break

            if target_touched:
                trade_return += _leg_return(
                    signal, entry, target, TP1_POSITION_FRACTION
                )
                remaining -= TP1_POSITION_FRACTION

            # Ratchet the stop only after evaluating this candle, avoiding the
            # impossible assumption that its high/low happened before its low/high.
            if pd.notna(bar_atr) and bar_atr > 0:
                candidate = (
                    high - atr_multiplier * bar_atr
                    if signal == 1
                    else low + atr_multiplier * bar_atr
                )
                stop = max(stop, candidate) if signal == 1 else min(stop, candidate)

            exit_i = j
            if j == len(frame) - 1:
                trade_return += _leg_return(signal, entry, close, remaining)
                remaining = 0.0

        trades.append(
            Trade(
                direction=signal,
                entry_time=frame.at[entry_i, "open_time"],
                exit_time=frame.at[exit_i, "open_time"],
                net_return=trade_return,
            )
        )
        # Do not open overlapping trades. Resume after the completed trade.
        i = max(exit_i + 1, entry_i + 1)

    return trades


def summarize(symbol: str, trades: list[Trade]) -> dict[str, Any]:
    """Produce one scanner table row."""
    if not trades:
        return {
            "Pair": symbol,
            "Total Trades": 0,
            "Win Rate (%)": 0.0,
            "Net Expectancy (%)": 0.0,
        }
    returns = pd.Series([trade.net_return for trade in trades], dtype="float64")
    return {
        "Pair": symbol,
        "Total Trades": len(trades),
        "Win Rate (%)": round(float((returns > 0).mean() * 100), 2),
        "Net Expectancy (%)": round(float(returns.mean() * 100), 3),
    }


def scan_market(
    symbols: list[str],
    interval: str,
    fast_span: int,
    slow_span: int,
    atr_multiplier: float,
    progress_bar: Any,
    status_box: Any,
) -> tuple[pd.DataFrame, list[str]]:
    """Fetch, backtest and summarize each selected contract."""
    results: list[dict[str, Any]] = []
    failures: list[str] = []

    for index, symbol in enumerate(symbols, start=1):
        status_box.caption(f"Scanning {index}/{len(symbols)} · {symbol}")
        try:
            candles = fetch_klines(symbol, interval)
            prepared = add_indicators(candles, fast_span, slow_span)
            trades = backtest(prepared, atr_multiplier)
            results.append(summarize(symbol, trades))
        except Exception as exc:  # Continue scanning if one contract is unavailable.
            failures.append(f"{symbol}: {exc}")
        progress_bar.progress(index / len(symbols))

    output = pd.DataFrame(results)
    if not output.empty:
        output = output.sort_values(
            ["Net Expectancy (%)", "Win Rate (%)", "Total Trades"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        output.index = output.index + 1
    return output, failures


def main() -> None:
    st.title("📈 Binance Futures Trend Scanner")
    st.write(
        "Backtest a transparent EMA trend model across the most liquid active "
        "Binance USDT-M perpetual contracts. Public market data only—no API key required."
    )

    with st.sidebar:
        st.header("Scanner settings")
        timeframe = st.selectbox("Timeframe", ["15m", "1h"], index=0)
        pair_count = st.slider("Number of pairs to scan", 10, 100, 30, step=10)
        st.subheader("Indicator parameters")
        fast_ema = st.number_input("Fast EMA span", 2, 100, 9, step=1)
        slow_ema = st.number_input("Slow EMA span", 3, 250, 21, step=1)
        atr_multiplier = st.number_input(
            "ATR trailing-stop multiplier", 0.5, 5.0, 1.5, step=0.1
        )
        st.caption(
            f"Fixed rules: Volume MA {VOLUME_MA_LENGTH} · ATR 14 · "
            f"TP1 {TP_R_MULTIPLE:.1f}R on 50% · Cost 0.04% per side"
        )

    if fast_ema >= slow_ema:
        st.warning("Fast EMA must be smaller than Slow EMA.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Universe", f"Top {pair_count} liquid")
    col2.metric("Timeframe", timeframe)
    col3.metric("Candle history", KLINE_LIMIT)

    run_scan = st.button("Run market scan", type="primary", use_container_width=True)

    if not run_scan:
        st.info("Open the sidebar to adjust the model, then tap **Run market scan**.")
        st.markdown(
            "<p class='small-note'>Ranking uses average net trade return, not total "
            "profit. Results are historical simulations and are not financial advice.</p>",
            unsafe_allow_html=True,
        )
        return

    progress = st.progress(0.0)
    status = st.empty()

    try:
        status.caption("Loading the active contract universe…")
        symbols = fetch_top_symbols(pair_count)
        if not symbols:
            raise BinanceAPIError("No liquid active symbols were returned.")
        table, failures = scan_market(
            symbols,
            timeframe,
            int(fast_ema),
            int(slow_ema),
            float(atr_multiplier),
            progress,
            status,
        )
    except Exception as exc:
        progress.empty()
        status.empty()
        st.error(f"The scan could not start. {exc}")
        st.caption(
            "Some hosting regions restrict Binance. If this persists on Streamlit "
            "Community Cloud, run the same app locally or in a Binance-supported region."
        )
        return

    status.success(f"Scan complete · {len(table)} pairs analyzed")
    if table.empty:
        st.warning("No backtest results were produced. Try again in a moment.")
        return

    st.subheader("Ranked backtest results")
    st.dataframe(
        table,
        use_container_width=True,
        height=min(620, 38 * (len(table) + 1)),
        column_config={
            "Pair": st.column_config.TextColumn("Pair", width="medium"),
            "Total Trades": st.column_config.NumberColumn("Trades", format="%d"),
            "Win Rate (%)": st.column_config.NumberColumn("Win %", format="%.2f%%"),
            "Net Expectancy (%)": st.column_config.NumberColumn(
                "Expectancy %", format="%.3f%%"
            ),
        },
    )

    csv_bytes = table.reset_index(drop=True).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download results as CSV",
        data=csv_bytes,
        file_name=f"binance_trend_scan_{timeframe}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if failures:
        with st.expander(f"Skipped pairs ({len(failures)})"):
            st.code("\n".join(failures))

    st.markdown(
        """
        <p class='small-note'>Method: signals use completed candles and enter at the
        next open. TP1 closes 50% at 2R; the remainder follows a 1.5× ATR-style
        ratcheting stop (or your selected multiplier). When a candle touches both
        stop and target, the backtest assumes the stop was hit first. Expectancy is
        average net return per completed trade after 0.04% cost per executed side.
        This simplified candle-level model does not include funding, spread,
        liquidation, latency, or market impact.</p>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
