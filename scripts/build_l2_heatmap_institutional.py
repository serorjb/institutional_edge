#!/usr/bin/env python3
"""
Institutional L2 Order Book Heatmap

Builds a professional HTML dashboard from level-2 order-book updates:
- Reconstructs the displayed book by applying INSERT / UPDATE / DELETE events by side and position.
- Samples book state into time buckets.
- Renders a focused net-liquidity heatmap with restrained institutional styling.

Example:
  python build_l2_heatmap_institutional.py \
    --input C:\\Users\\jsero\\Documents\\Dev\\lifestyles\\L2_work\\cl_l2_orderbook.parquet \
    --output l2_heatmap_institutional.html
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go


BRAND = {
    "ink": "#111111",
    "blue": "#0A2540",
    "blue_soft": "#DCE9F3",
    "paper": "#F8F9FA",
    "line": "#DDE3E9",
    "muted": "#66717E",
    "bid": "#0A2540",
    "ask": "#A94D4D",
    "gold": "#B88A44",
    "green": "#2F7D69",
}


@dataclass(frozen=True)
class DashboardConfig:
    input_file: Path
    output_file: Path
    instrument: str
    context: str
    time_bucket_seconds: int
    max_price_levels: int
    sweep_levels: int
    target_contracts: int
    include_plotlyjs: str


def parse_args() -> DashboardConfig:
    parser = argparse.ArgumentParser(description="Build an institutional L2 heatmap dashboard.")
    parser.add_argument("--input", "-i", default="cl_l2_orderbook.parquet", help="Input parquet file.")
    parser.add_argument("--output", "-o", default="l2_heatmap_institutional.html", help="Output HTML file.")
    parser.add_argument("--instrument", default="ICE CL Crude Oil Futures", help="Instrument title.")
    parser.add_argument(
        "--context",
        default="Real level-2 sample, reconstructed from order-book events",
        help="Subtitle/context line.",
    )
    parser.add_argument("--bucket-seconds", type=int, default=5, help="Time bucket size in seconds.")
    parser.add_argument("--max-price-levels", type=int, default=34, help="Maximum price rows in heatmap.")
    parser.add_argument("--sweep-levels", type=int, default=5, help="Top N levels retained for book diagnostics.")
    parser.add_argument("--target-contracts", type=int, default=25, help=argparse.SUPPRESS)
    parser.add_argument(
        "--plotlyjs",
        default="cdn",
        choices=["cdn", "include", "directory", "require"],
        help="How Plotly JS is included in the HTML.",
    )
    args = parser.parse_args()
    return DashboardConfig(
        input_file=Path(args.input),
        output_file=Path(args.output),
        instrument=args.instrument,
        context=args.context,
        time_bucket_seconds=args.bucket_seconds,
        max_price_levels=args.max_price_levels,
        sweep_levels=args.sweep_levels,
        target_contracts=args.target_contracts,
        include_plotlyjs=args.plotlyjs,
    )


def load_events(path: Path) -> pd.DataFrame:
    required = {"timestamp", "position", "operation", "side", "price", "size"}
    df = pd.read_parquet(path)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["dt"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["dt"])
    df["side"] = df["side"].astype(str).str.upper()
    df["operation"] = df["operation"].astype(str).str.upper()
    df = df[df["side"].isin(["BID", "ASK"])]
    df = df[df["operation"].isin(["INSERT", "UPDATE", "DELETE"])]
    df["position"] = df["position"].astype(int)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df["size"] = pd.to_numeric(df["size"], errors="coerce").fillna(0.0)
    return df.sort_values(["dt", "side", "position"]).reset_index(drop=True)


def apply_l2_event(book: dict[str, list[tuple[float, float]]], row: pd.Series) -> None:
    side = row["side"]
    position = int(row["position"])
    operation = row["operation"]
    level = (float(row["price"]), float(row["size"]))
    levels = book[side]

    if operation == "INSERT":
        position = min(max(position, 0), len(levels))
        levels.insert(position, level)
    elif operation == "UPDATE":
        if 0 <= position < len(levels):
            if level[0] > 0 and level[1] > 0:
                levels[position] = level
            else:
                levels.pop(position)
        elif level[0] > 0 and level[1] > 0:
            levels.append(level)
    elif operation == "DELETE":
        if 0 <= position < len(levels):
            levels.pop(position)


def level_dict(levels: Iterable[tuple[float, float]]) -> dict[float, float]:
    result: dict[float, float] = {}
    for price, size in levels:
        if price > 0 and size > 0:
            result[price] = result.get(price, 0.0) + size
    return result


def sweep_notional(levels: Iterable[tuple[float, float]], n_levels: int, descending: bool) -> float:
    valid = [(p, s) for p, s in levels if p > 0 and s > 0]
    valid.sort(key=lambda item: item[0], reverse=descending)
    return float(sum(price * size for price, size in valid[:n_levels]))


def price_to_fill(levels: Iterable[tuple[float, float]], target_size: float, descending: bool) -> float | None:
    valid = [(p, s) for p, s in levels if p > 0 and s > 0]
    valid.sort(key=lambda item: item[0], reverse=descending)
    remaining = target_size
    if remaining <= 0:
        return None
    for price, size in valid:
        remaining -= size
        if remaining <= 0:
            return float(price)
    return None


def reconstruct_snapshots(df: pd.DataFrame, bucket_seconds: int) -> pd.DataFrame:
    book: dict[str, list[tuple[float, float]]] = {"BID": [], "ASK": []}
    bucket = pd.Timedelta(seconds=bucket_seconds)
    next_bucket = df["dt"].min().floor(f"{bucket_seconds}s")
    end_time = df["dt"].max().ceil(f"{bucket_seconds}s")
    rows: list[dict[str, object]] = []

    def emit_snapshot(ts: pd.Timestamp) -> None:
        bid_levels = level_dict(book["BID"])
        ask_levels = level_dict(book["ASK"])
        best_bid = max(bid_levels, default=np.nan)
        best_ask = min(ask_levels, default=np.nan)
        mid = (best_bid + best_ask) / 2 if np.isfinite(best_bid) and np.isfinite(best_ask) else np.nan
        spread = best_ask - best_bid if np.isfinite(best_bid) and np.isfinite(best_ask) else np.nan
        bid_top = sum(size for _, size in sorted(bid_levels.items(), reverse=True)[:5])
        ask_top = sum(size for _, size in sorted(ask_levels.items())[:5])
        imbalance = (bid_top - ask_top) / (bid_top + ask_top) if (bid_top + ask_top) else np.nan
        rows.append(
            {
                "time_bin": ts,
                "bids": bid_levels,
                "asks": ask_levels,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "mid": mid,
                "spread": spread,
                "imbalance": imbalance,
                "sweep_bid": sweep_notional(bid_levels.items(), 5, descending=True),
                "sweep_ask": sweep_notional(ask_levels.items(), 5, descending=False),
            }
        )

    for _, row in df.iterrows():
        while row["dt"] >= next_bucket + bucket:
            emit_snapshot(next_bucket)
            next_bucket += bucket
        apply_l2_event(book, row)

    while next_bucket <= end_time:
        emit_snapshot(next_bucket)
        next_bucket += bucket

    snapshots = pd.DataFrame(rows)
    active = snapshots["bids"].map(bool) | snapshots["asks"].map(bool)
    return snapshots[active].reset_index(drop=True)


def choose_price_ladder(snapshots: pd.DataFrame, max_levels: int) -> list[float]:
    total: dict[float, float] = {}
    for _, row in snapshots.iterrows():
        for price, size in row["bids"].items():
            total[price] = total.get(price, 0.0) + size
        for price, size in row["asks"].items():
            total[price] = total.get(price, 0.0) + size

    if not total:
        return []

    all_prices = sorted(total)
    anchor = max(total, key=total.get)
    if len(all_prices) <= max_levels:
        return all_prices

    idx = all_prices.index(anchor)
    start = max(0, min(idx - max_levels // 2, len(all_prices) - max_levels))
    return all_prices[start : start + max_levels]


def build_matrices(
    snapshots: pd.DataFrame, price_ladder: list[float], target_contracts: int
) -> tuple[list[list[float | None]], list[float], list[float], list[float | None]]:
    heatmap: list[list[float | None]] = []
    sell_fill: list[float | None] = []
    buy_fill: list[float | None] = []

    for price in price_ladder:
        row = []
        for _, snap in snapshots.iterrows():
            bid = snap["bids"].get(price, 0.0)
            ask = snap["asks"].get(price, 0.0)
            row.append(None if bid == 0 and ask == 0 else bid - ask)
        heatmap.append(row)

    for _, snap in snapshots.iterrows():
        sell_px = price_to_fill(snap["bids"].items(), target_contracts, descending=True)
        buy_px = price_to_fill(snap["asks"].items(), target_contracts, descending=False)
        mid = snap["mid"]
        if np.isfinite(mid) and sell_px:
            sell_fill.append((mid - sell_px) / mid * 10_000)
        else:
            sell_fill.append(None)
        if np.isfinite(mid) and buy_px:
            buy_fill.append((buy_px - mid) / mid * 10_000)
        else:
            buy_fill.append(None)

    return heatmap, sell_fill, buy_fill


def colorscale() -> list[tuple[float, str]]:
    return [
        [0.00, BRAND["ask"]],
        [0.48, "#F3F5F7"],
        [0.50, "#F8F9FA"],
        [0.52, "#EEF3F7"],
        [1.00, BRAND["blue"]],
    ]


def make_dashboard(config: DashboardConfig) -> None:
    df = load_events(config.input_file)
    snapshots = reconstruct_snapshots(df, config.time_bucket_seconds)
    if snapshots.empty:
        raise ValueError("No active order-book snapshots could be reconstructed.")

    price_ladder = choose_price_ladder(snapshots, config.max_price_levels)
    heatmap, _, _ = build_matrices(snapshots, price_ladder, config.target_contracts)
    times = snapshots["time_bin"]
    z_values = [value for row in heatmap for value in row if value is not None]
    z_abs = max(abs(min(z_values, default=-1)), abs(max(z_values, default=1)), 1)

    title_range = f"{df['dt'].min():%Y-%m-%d %H:%M:%S} - {df['dt'].max():%H:%M:%S UTC}"
    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=heatmap,
            x=times,
            y=price_ladder,
            zmid=0,
            zmin=-z_abs,
            zmax=z_abs,
            colorscale=colorscale(),
            showscale=False,
            hovertemplate="Time: %{x|%H:%M:%S}<br>Price: %{y:.2f}<br>Net size: %{z:,.0f}<extra></extra>",
        ),
    )

    tick_step = max(1, len(times) // 12)
    tick_values = list(times.iloc[::tick_step])
    tick_text = [ts.strftime("%H:%M:%S") for ts in tick_values]
    fig.update_xaxes(
        title_text="Time (UTC)",
        tickvals=tick_values,
        ticktext=tick_text,
        tickangle=-35,
        gridcolor=BRAND["line"],
    )
    fig.update_yaxes(title_text="Price", tickformat=".2f", autorange="reversed", gridcolor=BRAND["line"])

    fig.update_layout(
        template="plotly_white",
        width=1320,
        height=780,
        margin=dict(l=78, r=68, t=132, b=72),
        paper_bgcolor=BRAND["paper"],
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, Arial, sans-serif", color=BRAND["ink"], size=12),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=BRAND["line"], font_size=11),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.015,
            xanchor="left",
            x=0,
            bgcolor="rgba(248,249,250,.92)",
            bordercolor=BRAND["line"],
            borderwidth=1,
        ),
        title=dict(
            text=(
                f"<span style='font-family:Georgia,serif;font-size:30px;color:{BRAND['blue']}'>{config.instrument}</span>"
                f"<br><span style='font-size:13px;color:{BRAND['muted']}'>{config.context} | {title_range}</span>"
            ),
            x=0.01,
            xanchor="left",
        ),
    )

    config.output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        config.output_file,
        include_plotlyjs=config.include_plotlyjs,
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "toImageButtonOptions": {
                "format": "png",
                "filename": "institutional_l2_heatmap",
                "height": 780,
                "width": 1320,
                "scale": 2,
            },
        },
    )

    print(f"Saved {config.output_file}")
    print(f"Updates: {len(df):,}")
    print(f"Snapshots: {len(snapshots):,} at {config.time_bucket_seconds}s buckets")
    print(f"Price levels: {len(price_ladder)}")
    print(f"Time range: {title_range}")


def main() -> None:
    make_dashboard(parse_args())


if __name__ == "__main__":
    main()
