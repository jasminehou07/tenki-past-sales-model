#!/usr/bin/env python3
"""Train a daily genre-level Rakuten sales model with event features.

The source TENKI files are item-level parquet exports. This script aggregates
them to genre x day, adds calendar/event/ranking features, creates lagged sales
signals, and evaluates a time-based holdout.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("work/model_cache/matplotlib").resolve()))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


DEFAULT_DATA_DIR = Path("/Users/jasminehou/Downloads/TENKI")
DEFAULT_CACHE_DIR = Path("work/model_cache")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_HOLIDAY_FILE = Path("data/japan_holidays.csv")
DEFAULT_PROMOTION_EFFECT_DIR = Path("data/promotion_effects")
DEFAULT_EVENT_STRENGTH_FILE = Path("data/rakuten_event_strength.csv")
CACHE_NAME = "daily_genre_dataset_v6.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--holiday-file", type=Path, default=DEFAULT_HOLIDAY_FILE)
    parser.add_argument("--promotion-effect-dir", type=Path, default=DEFAULT_PROMOTION_EFFECT_DIR)
    parser.add_argument("--event-strength-file", type=Path, default=DEFAULT_EVENT_STRENGTH_FILE)
    parser.add_argument(
        "--regime-cutoff",
        type=str,
        default="2024-01-01",
        help="Date where the marketplace changed enough to train a separate later-years model.",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--test-days", type=int, default=180)
    parser.add_argument(
        "--max-importance-rows",
        type=int,
        default=5000,
        help="Rows sampled from the test set for permutation importance.",
    )
    return parser.parse_args()


def active_events_by_day(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for rec in events.itertuples(index=False):
        days = pd.date_range(pd.Timestamp(rec.start).normalize(), pd.Timestamp(rec.end).normalize(), freq="D")
        rows.append(pd.DataFrame({"date": days, f"event_{rec.name}": 1}))
    if not rows:
        return pd.DataFrame(columns=["date"])

    expanded = pd.concat(rows, ignore_index=True)
    expanded = expanded.groupby("date", as_index=False).max()
    event_cols = [c for c in expanded.columns if c.startswith("event_")]
    expanded["event_count"] = expanded[event_cols].sum(axis=1)
    expanded["any_event"] = (expanded["event_count"] > 0).astype(int)
    return expanded


def add_event_timing_features(event_daily: pd.DataFrame, min_date: pd.Timestamp, max_date: pd.Timestamp) -> pd.DataFrame:
    all_dates = pd.DataFrame({"date": pd.date_range(min_date, max_date, freq="D")})
    out = all_dates.merge(event_daily, on="date", how="left")
    event_cols = [c for c in out.columns if c.startswith("event_")]
    for col in event_cols + ["event_count", "any_event"]:
        out[col] = out[col].fillna(0)

    timing_features: dict[str, pd.Series] = {}
    for col in event_cols:
        for days in [1, 2, 3, 7]:
            timing_features[f"{col}_in_{days}d"] = out[col].shift(-days).fillna(0)
            timing_features[f"{col}_after_{days}d"] = out[col].shift(days).fillna(0)
    for window in [3, 7, 14]:
        timing_features[f"event_count_next_{window}d"] = (
            out["event_count"].shift(-1).rolling(window, min_periods=1).sum().shift(-(window - 1)).fillna(0)
        )
        timing_features[f"event_count_prev_{window}d"] = (
            out["event_count"].shift(1).rolling(window, min_periods=1).sum().fillna(0)
        )
    if timing_features:
        out = pd.concat([out, pd.DataFrame(timing_features)], axis=1)

    event_dates = out.loc[out["any_event"].eq(1), "date"].to_numpy()
    days_since: list[int] = []
    days_to: list[int] = []
    for date in out["date"].to_numpy():
        diffs = (event_dates - date).astype("timedelta64[D]").astype(int)
        past = diffs[diffs <= 0]
        future = diffs[diffs >= 0]
        days_since.append(abs(past.max()) if len(past) else 999)
        days_to.append(future.min() if len(future) else 999)
    out["days_since_event"] = np.clip(days_since, 0, 30)
    out["days_to_event"] = np.clip(days_to, 0, 30)
    return out


def add_event_strength_features(event_daily: pd.DataFrame, strength_file: Path) -> pd.DataFrame:
    out = event_daily.copy()
    for col in [
        "event_max_point_multiplier",
        "event_bonus_point_multiplier",
        "event_point_cap",
        "event_strength_score",
        "event_shoparound_strength",
        "event_requires_entry_count",
    ]:
        out[col] = 0.0
    if not strength_file.exists():
        return out

    strengths = pd.read_csv(strength_file)
    for row in strengths.itertuples(index=False):
        event_col = f"event_{row.event_name}"
        if event_col not in out.columns:
            continue
        active = out[event_col].fillna(0)
        max_multiplier = float(row.max_point_multiplier)
        bonus_multiplier = float(row.bonus_point_multiplier)
        point_cap = float(row.point_cap)
        strength_score = bonus_multiplier + np.log1p(point_cap) / 5
        out["event_max_point_multiplier"] += active * max_multiplier
        out["event_bonus_point_multiplier"] += active * bonus_multiplier
        out["event_point_cap"] += active * point_cap
        out["event_strength_score"] += active * strength_score
        out["event_requires_entry_count"] += active * float(row.requires_entry)
        if row.scope == "shop_around":
            out["event_shoparound_strength"] += active * bonus_multiplier
    return out


def add_holiday_features(holiday_file: Path, min_date: pd.Timestamp, max_date: pd.Timestamp) -> pd.DataFrame:
    holidays = pd.read_csv(holiday_file)
    holidays["date"] = pd.to_datetime(holidays["date"])
    out = pd.DataFrame({"date": pd.date_range(min_date, max_date, freq="D")})
    out = out.merge(holidays.assign(is_holiday=1), on="date", how="left")
    out["is_holiday"] = out["is_holiday"].fillna(0)
    out["holiday_name"] = out["holiday_name"].fillna("")

    timing_features: dict[str, pd.Series] = {}
    for days in [1, 2, 3, 7, 14]:
        timing_features[f"holiday_in_{days}d"] = out["is_holiday"].shift(-days).fillna(0)
        timing_features[f"holiday_after_{days}d"] = out["is_holiday"].shift(days).fillna(0)
    for window in [3, 7, 14]:
        timing_features[f"holiday_count_next_{window}d"] = (
            out["is_holiday"].shift(-1).rolling(window, min_periods=1).sum().shift(-(window - 1)).fillna(0)
        )
        timing_features[f"holiday_count_prev_{window}d"] = (
            out["is_holiday"].shift(1).rolling(window, min_periods=1).sum().fillna(0)
        )

    holiday_dates = out.loc[out["is_holiday"].eq(1), "date"].to_numpy()
    days_since: list[int] = []
    days_to: list[int] = []
    for date in out["date"].to_numpy():
        diffs = (holiday_dates - date).astype("timedelta64[D]").astype(int)
        past = diffs[diffs <= 0]
        future = diffs[diffs >= 0]
        days_since.append(abs(past.max()) if len(past) else 999)
        days_to.append(future.min() if len(future) else 999)
    timing_features["days_since_holiday"] = np.clip(days_since, 0, 30)
    timing_features["days_to_holiday"] = np.clip(days_to, 0, 30)
    return pd.concat([out.drop(columns=["holiday_name"]), pd.DataFrame(timing_features)], axis=1)


def add_combined_upcoming_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for days in [1, 2, 3, 7, 14]:
        event_col = f"event_count_next_{days}d"
        holiday_col = f"holiday_count_next_{days}d"
        if event_col in out.columns and holiday_col in out.columns:
            out[f"promo_or_holiday_count_next_{days}d"] = out[event_col] + out[holiday_col]
            out[f"any_promo_or_holiday_next_{days}d"] = (out[f"promo_or_holiday_count_next_{days}d"] > 0).astype(int)
    if "days_to_event" in out.columns and "days_to_holiday" in out.columns:
        out["days_to_promo_or_holiday"] = np.minimum(out["days_to_event"], out["days_to_holiday"])
    for col in [
        "event_strength_score",
        "event_bonus_point_multiplier",
        "event_shoparound_strength",
        "event_point_cap",
    ]:
        if col not in out.columns:
            continue
        for days in [1, 2, 3, 7]:
            out[f"{col}_in_{days}d"] = out.groupby("genre_id")[col].shift(-days).fillna(0)
        for window in [3, 7]:
            out[f"{col}_next_{window}d"] = (
                out.groupby("genre_id")[col]
                .shift(-1)
                .groupby(out["genre_id"])
                .rolling(window, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
                .groupby(out["genre_id"])
                .shift(-(window - 1))
                .fillna(0)
            )
    return out


def add_promotion_effect_features(df: pd.DataFrame, effect_dir: Path) -> pd.DataFrame:
    out = df.copy()
    lookup_file = effect_dir / "ranking_group_lookup.csv"
    group_effect_file = effect_dir / "event_impact_by_group.csv"
    summary_effect_file = effect_dir / "event_impact_summary.csv"

    out["genre_id"] = out["genre_id"].astype(str)
    if lookup_file.exists():
        groups = pd.read_csv(lookup_file, usecols=["genre_id", "ranking_group"])
        groups["genre_id"] = groups["genre_id"].astype(str)
        out = out.merge(groups, on="genre_id", how="left")
    else:
        out["ranking_group"] = "Unknown"
    out["ranking_group"] = out["ranking_group"].fillna("Unknown")

    lift_cols = [
        "sales_promo_lift_today",
        "items_promo_lift_today",
        "sales_promo_lift_next_3d",
        "items_promo_lift_next_3d",
        "sales_promo_lift_next_7d",
        "items_promo_lift_next_7d",
    ]
    for col in lift_cols:
        out[col] = 0.0

    if not group_effect_file.exists():
        return out

    group_effects = pd.read_csv(group_effect_file)
    group_effects = group_effects[group_effects["occurrences"].fillna(0) >= 3].copy()
    if group_effects.empty:
        return out

    group_effects["sales_lift_pct"] = group_effects["sales_lift_pct"].clip(lower=-80, upper=400)
    group_effects["items_lift_pct"] = group_effects["items_lift_pct"].clip(lower=-80, upper=400)
    global_sales_lift: dict[str, float] = {}
    global_items_lift: dict[str, float] = {}
    if summary_effect_file.exists():
        summary = pd.read_csv(summary_effect_file)
        global_sales_lift = summary.set_index("event_name")["sales_lift_pct"].clip(lower=-80, upper=400).to_dict()
        global_items_lift = summary.set_index("event_name")["items_lift_pct"].clip(lower=-80, upper=400).to_dict()

    sales_maps = {
        event: values.set_index("ranking_group")["sales_lift_pct"].to_dict()
        for event, values in group_effects.groupby("event_name")
    }
    items_maps = {
        event: values.set_index("ranking_group")["items_lift_pct"].to_dict()
        for event, values in group_effects.groupby("event_name")
    }

    for event_name, sales_map in sales_maps.items():
        event_col = f"event_{event_name}"
        if event_col not in out.columns:
            continue
        sales_lift = out["ranking_group"].map(sales_map).fillna(global_sales_lift.get(event_name, 0.0))
        items_lift = out["ranking_group"].map(items_maps.get(event_name, {})).fillna(global_items_lift.get(event_name, 0.0))
        out["sales_promo_lift_today"] += out[event_col] * sales_lift
        out["items_promo_lift_today"] += out[event_col] * items_lift
        for days in [1, 2, 3, 7]:
            upcoming_col = f"{event_col}_in_{days}d"
            if upcoming_col not in out.columns:
                continue
            if days <= 3:
                out["sales_promo_lift_next_3d"] += out[upcoming_col] * sales_lift
                out["items_promo_lift_next_3d"] += out[upcoming_col] * items_lift
            out["sales_promo_lift_next_7d"] += out[upcoming_col] * sales_lift
            out["items_promo_lift_next_7d"] += out[upcoming_col] * items_lift
    return out


def build_sales_daily(sales_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    sales_cols = [
        "date",
        "item_genre",
        "shop",
        "item",
        "sales",
        "sales_items",
        "pv",
        "uv",
        "sales_number",
        "reviews_posted",
        "reviews_total",
    ]
    for path in sorted(sales_dir.glob("*.parquet")):
        df = pd.read_parquet(path, columns=sales_cols)
        df["date"] = pd.to_datetime(df["date"])
        grouped = (
            df.groupby(["item_genre", "date"], as_index=False)
            .agg(
                sales=("sales", "sum"),
                sales_items=("sales_items", "sum"),
                pv=("pv", "sum"),
                uv=("uv", "sum"),
                orders=("sales_number", "sum"),
                reviews_posted=("reviews_posted", "sum"),
                reviews_total=("reviews_total", "max"),
                active_shops=("shop", "nunique"),
                active_items=("item", "nunique"),
            )
            .rename(columns={"item_genre": "genre_id"})
        )
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True)


def build_ranking_daily(ranking_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    rank_cols = ["date", "genre_id", "shop", "item", "rank", "price"]
    for path in sorted(ranking_dir.glob("*.parquet")):
        df = pd.read_parquet(path, columns=rank_cols)
        df["date"] = pd.to_datetime(df["date"])
        grouped = df.groupby(["genre_id", "date"], as_index=False).agg(
            ranked_items=("item", "nunique"),
            ranked_shops=("shop", "nunique"),
            best_rank=("rank", "min"),
            mean_rank=("rank", "mean"),
            median_price=("price", "median"),
            min_price=("price", "min"),
            max_price=("price", "max"),
        )
        frames.append(grouped)
    if not frames:
        return pd.DataFrame(columns=["genre_id", "date"])
    return pd.concat(frames, ignore_index=True)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["day_of_week"] = out["date"].dt.dayofweek
    out["day_of_month"] = out["date"].dt.day
    out["week_of_year"] = out["date"].dt.isocalendar().week.astype(int)
    out["month"] = out["date"].dt.month
    out["quarter"] = out["date"].dt.quarter
    out["year"] = out["date"].dt.year
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["is_month_start"] = out["date"].dt.is_month_start.astype(int)
    out["is_month_end"] = out["date"].dt.is_month_end.astype(int)
    out["sin_dow"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["cos_dow"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    out["sin_month"] = np.sin(2 * np.pi * out["month"] / 12)
    out["cos_month"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["genre_id", "date"]).copy()
    grouped = out.groupby("genre_id", group_keys=False)
    for lag in [1, 7, 14, 28]:
        out[f"sales_lag_{lag}d"] = grouped["sales"].shift(lag)
        out[f"items_lag_{lag}d"] = grouped["sales_items"].shift(lag)
    for window in [7, 28]:
        shifted_sales = grouped["sales"].shift(1)
        shifted_items = grouped["sales_items"].shift(1)
        out[f"sales_roll_mean_{window}d"] = shifted_sales.groupby(out["genre_id"]).rolling(window).mean().reset_index(level=0, drop=True)
        out[f"items_roll_mean_{window}d"] = shifted_items.groupby(out["genre_id"]).rolling(window).mean().reset_index(level=0, drop=True)
    return out


def build_dataset(
    data_dir: Path,
    cache_dir: Path,
    rebuild_cache: bool,
    holiday_file: Path = DEFAULT_HOLIDAY_FILE,
    promotion_effect_dir: Path = DEFAULT_PROMOTION_EFFECT_DIR,
    event_strength_file: Path = DEFAULT_EVENT_STRENGTH_FILE,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / CACHE_NAME
    if cache_path.exists() and not rebuild_cache:
        return pd.read_parquet(cache_path)

    sales = build_sales_daily(data_dir / "genre-sales")
    rankings = build_ranking_daily(data_dir / "genre-ranking")
    events = pd.read_parquet(data_dir / "events" / "events.parquet")
    event_daily = add_event_timing_features(
        active_events_by_day(events),
        pd.Timestamp(sales["date"].min()),
        pd.Timestamp(sales["date"].max()),
    )
    event_daily = add_event_strength_features(event_daily, event_strength_file)
    holiday_daily = add_holiday_features(
        holiday_file,
        pd.Timestamp(sales["date"].min()),
        pd.Timestamp(sales["date"].max()),
    )

    dataset = sales.merge(rankings, on=["genre_id", "date"], how="left")
    dataset = dataset.merge(event_daily, on="date", how="left")
    dataset = dataset.merge(holiday_daily, on="date", how="left")
    event_cols = [c for c in dataset.columns if c.startswith("event_")]
    for col in event_cols + ["event_count", "any_event"]:
        dataset[col] = dataset[col].fillna(0)
    holiday_cols = [c for c in dataset.columns if "holiday" in c]
    for col in holiday_cols:
        dataset[col] = dataset[col].fillna(0)
    dataset = dataset.sort_values(["genre_id", "date"]).reset_index(drop=True)
    dataset = add_combined_upcoming_features(dataset)
    dataset = add_promotion_effect_features(dataset, promotion_effect_dir)
    dataset = add_calendar_features(dataset)
    dataset = add_lag_features(dataset)
    dataset = dataset.sort_values(["date", "genre_id"]).reset_index(drop=True)
    dataset.to_parquet(cache_path, index=False)
    return dataset


def metrics_frame(y_true: pd.Series, pred: np.ndarray) -> dict[str, float]:
    rmse = mean_squared_error(y_true, pred) ** 0.5
    mae = mean_absolute_error(y_true, pred)
    denominator = np.maximum(np.abs(y_true), 1)
    wape = np.abs(y_true - pred).sum() / np.maximum(np.abs(y_true).sum(), 1)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "wape": float(wape),
        "median_absolute_pct_error": float(np.median(np.abs(y_true - pred) / denominator)),
        "r2": float(r2_score(y_true, pred)),
    }


def readable_feature_name(name: str) -> str:
    def days_text(days: str) -> str:
        return "1 day" if str(days) == "1" else f"{days} days"

    event_names = {
        "zero-five": "5 and 0 day",
        "marathon": "Shopping Marathon",
        "supersale": "Rakuten Super Sale",
        "black-friday": "Black Friday",
        "ichiba-day": "18th Ichiba Day",
        "wonderful-day": "Wonderful Day",
        "thank-you": "Thanksgiving Festival",
        "eagles": "Sports win campaign",
        "39shop": "39 Shop campaign",
        "point-back": "Point-back campaign",
        "point-up": "Point-up campaign",
    }
    direct_names = {
        "genre_id": "Genre",
        "ranking_group": "Ranking group",
        "active_items": "Active items that day",
        "active_shops": "Active shops that day",
        "ranked_items": "Ranked items that day",
        "ranked_shops": "Ranked shops that day",
        "best_rank": "Best ranking position",
        "mean_rank": "Average ranking position",
        "median_price": "Median ranked item price",
        "min_price": "Lowest ranked item price",
        "max_price": "Highest ranked item price",
        "event_count": "Number of active Rakuten events",
        "any_event": "Any Rakuten event active",
        "event_max_point_multiplier": "Largest active event point multiplier",
        "event_bonus_point_multiplier": "Active event bonus point multiplier",
        "event_point_cap": "Active event point cap",
        "event_strength_score": "Active event strength score",
        "event_shoparound_strength": "Active shop-around promotion strength",
        "event_requires_entry_count": "Active events requiring entry",
        "days_since_event": "Days since previous Rakuten event",
        "days_to_event": "Days until next Rakuten event",
        "is_holiday": "Japan holiday that day",
        "days_since_holiday": "Days since previous Japan holiday",
        "days_to_holiday": "Days until next Japan holiday",
        "days_to_promo_or_holiday": "Days until next promotion or holiday",
        "sales_promo_lift_today": "Expected sales lift from today's promotion",
        "items_promo_lift_today": "Expected quantity lift from today's promotion",
        "sales_promo_lift_next_3d": "Expected sales lift from promotions in next 3 days",
        "items_promo_lift_next_3d": "Expected quantity lift from promotions in next 3 days",
        "sales_promo_lift_next_7d": "Expected sales lift from promotions in next 7 days",
        "items_promo_lift_next_7d": "Expected quantity lift from promotions in next 7 days",
        "day_of_week": "Day of week",
        "day_of_month": "Day of month",
        "week_of_year": "Week of year",
        "month": "Month",
        "quarter": "Quarter",
        "year": "Year",
        "is_weekend": "Weekend",
        "is_month_start": "Start of month",
        "is_month_end": "End of month",
        "sin_dow": "Weekly seasonality pattern",
        "cos_dow": "Weekly seasonality pattern",
        "sin_month": "Yearly seasonality pattern",
        "cos_month": "Yearly seasonality pattern",
    }
    if name in direct_names:
        return direct_names[name]

    strength_match = re.fullmatch(r"event_(max_point_multiplier|bonus_point_multiplier|point_cap|strength_score|shoparound_strength)_(in|next)_(\d+)d", name)
    if strength_match:
        metric = direct_names.get(f"event_{strength_match.group(1)}", strength_match.group(1).replace("_", " "))
        if strength_match.group(2) == "next":
            return f"{metric} over next {days_text(strength_match.group(3))}"
        return f"{metric} in {days_text(strength_match.group(3))}"

    for prefix, label in [("sales", "sales"), ("items", "quantity")]:
        lag_match = re.fullmatch(rf"{prefix}_lag_(\d+)d", name)
        if lag_match:
            return f"{label.title()} from {days_text(lag_match.group(1))} earlier"
        roll_match = re.fullmatch(rf"{prefix}_roll_mean_(\d+)d", name)
        if roll_match:
            return f"Average {label} over previous {days_text(roll_match.group(1))}"

    event_match = re.fullmatch(r"event_(.+?)(?:_in_(\d+)d|_after_(\d+)d)?", name)
    if event_match:
        event_label = event_names.get(event_match.group(1), event_match.group(1).replace("-", " ").title())
        if event_match.group(2):
            return f"{event_label} in {days_text(event_match.group(2))}"
        if event_match.group(3):
            return f"{event_label} {days_text(event_match.group(3))} ago"
        return f"{event_label} active today"

    count_match = re.fullmatch(r"(event|holiday)_count_(next|prev)_(\d+)d", name)
    if count_match:
        kind = "Rakuten events" if count_match.group(1) == "event" else "Japan holidays"
        direction = "next" if count_match.group(2) == "next" else "previous"
        return f"Number of {kind} in {direction} {days_text(count_match.group(3))}"

    holiday_match = re.fullmatch(r"holiday_(in|after)_(\d+)d", name)
    if holiday_match:
        if holiday_match.group(1) == "in":
            return f"Japan holiday in {days_text(holiday_match.group(2))}"
        return f"Japan holiday {days_text(holiday_match.group(2))} ago"

    promo_match = re.fullmatch(r"promo_or_holiday_count_next_(\d+)d", name)
    if promo_match:
        return f"Promotions or holidays in next {days_text(promo_match.group(1))}"
    any_match = re.fullmatch(r"any_promo_or_holiday_next_(\d+)d", name)
    if any_match:
        return f"Any promotion or holiday in next {days_text(any_match.group(1))}"

    return name.replace("_", " ").replace("-", " ").title()


def make_preprocessor(categorical_cols: list[str], numeric_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("numeric", "passthrough", numeric_cols),
        ],
        verbose_feature_names_out=False,
    )


def make_sales_model(preprocessor: ColumnTransformer) -> object:
    return make_pipeline(
        preprocessor,
        HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.04,
            max_iter=700,
            max_leaf_nodes=63,
            l2_regularization=0.03,
            random_state=42,
        ),
    )


def train_log_model(model: object, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
    model.fit(X_train, np.log1p(y_train))
    return np.expm1(model.predict(X_test)).clip(min=0)


def predict_log_model(model: object, X_test: pd.DataFrame) -> np.ndarray:
    return np.expm1(model.predict(X_test)).clip(min=0)


def write_model_analysis_outputs(
    output_dir: Path,
    test: pd.DataFrame,
    pred: np.ndarray,
    quantity_pred: np.ndarray,
    event_strength_file: Path,
) -> None:
    analysis = test[["date", "genre_id", "ranking_group", "sales", "sales_items"]].copy()
    analysis["predicted_sales"] = pred
    analysis["sales_error"] = (analysis["sales"] - analysis["predicted_sales"]).abs()
    analysis["predicted_sales_items"] = quantity_pred
    analysis["quantity_error"] = (analysis["sales_items"] - analysis["predicted_sales_items"]).abs()

    struggle = (
        analysis.groupby(["genre_id", "ranking_group"], as_index=False)
        .agg(
            rows=("date", "count"),
            actual_sales=("sales", "sum"),
            predicted_sales=("predicted_sales", "sum"),
            sales_mae=("sales_error", "mean"),
            sales_total_error=("sales_error", "sum"),
            actual_quantity=("sales_items", "sum"),
            predicted_quantity=("predicted_sales_items", "sum"),
            quantity_mae=("quantity_error", "mean"),
            quantity_total_error=("quantity_error", "sum"),
        )
    )
    struggle["sales_wape"] = struggle["sales_total_error"] / struggle["actual_sales"].clip(lower=1)
    struggle["quantity_wape"] = struggle["quantity_total_error"] / struggle["actual_quantity"].clip(lower=1)
    struggle["sales_bias"] = (struggle["predicted_sales"] - struggle["actual_sales"]) / struggle["actual_sales"].clip(lower=1)
    struggle["quantity_bias"] = (
        (struggle["predicted_quantity"] - struggle["actual_quantity"]) / struggle["actual_quantity"].clip(lower=1)
    )
    struggle.sort_values(["sales_wape", "sales_total_error"], ascending=False).to_csv(
        output_dir / "model_struggles.csv",
        index=False,
    )

    if not event_strength_file.exists():
        return
    strengths = pd.read_csv(event_strength_file)
    promo_rows: list[dict[str, object]] = []
    for event in strengths.itertuples(index=False):
        event_col = f"event_{event.event_name}"
        if event_col not in test.columns:
            continue
        mask = test[event_col].fillna(0).gt(0)
        if not mask.any():
            continue
        subset = analysis.loc[mask]
        sales_error = subset["sales_error"].sum()
        quantity_error = subset["quantity_error"].sum()
        promo_rows.append(
            {
                "event_name": event.event_name,
                "display_name": event.display_name,
                "rows": int(len(subset)),
                "days": int(test.loc[mask, "date"].nunique()),
                "actual_sales": float(subset["sales"].sum()),
                "predicted_sales": float(subset["predicted_sales"].sum()),
                "sales_wape": float(sales_error / max(subset["sales"].sum(), 1)),
                "actual_quantity": float(subset["sales_items"].sum()),
                "predicted_quantity": float(subset["predicted_sales_items"].sum()),
                "quantity_wape": float(quantity_error / max(subset["sales_items"].sum(), 1)),
                "max_point_multiplier": float(event.max_point_multiplier),
                "bonus_point_multiplier": float(event.bonus_point_multiplier),
                "point_cap": float(event.point_cap),
                "scope": event.scope,
                "source_url": event.source_url,
            }
        )
    pd.DataFrame(promo_rows).sort_values(["actual_sales", "rows"], ascending=False).to_csv(
        output_dir / "promotion_impact.csv",
        index=False,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(
        args.data_dir,
        args.cache_dir,
        args.rebuild_cache,
        args.holiday_file,
        args.promotion_effect_dir,
        args.event_strength_file,
    )
    dataset = dataset.dropna(subset=["sales_lag_28d", "sales_roll_mean_28d"]).copy()
    dataset = dataset.fillna(
        {
            "ranked_items": 0,
            "ranked_shops": 0,
            "best_rank": 999,
            "mean_rank": 999,
            "median_price": 0,
            "min_price": 0,
            "max_price": 0,
        }
    )

    cutoff = dataset["date"].max() - pd.Timedelta(days=args.test_days)
    regime_cutoff = pd.Timestamp(args.regime_cutoff)
    train = dataset[dataset["date"] <= cutoff].copy()
    test = dataset[dataset["date"] > cutoff].copy()
    early_train = train[train["date"] < regime_cutoff].copy()
    late_train = train[train["date"] >= regime_cutoff].copy()
    if early_train.empty or late_train.empty:
        raise ValueError(f"Regime cutoff {regime_cutoff.date()} must leave training rows on both sides.")

    leakage_cols = {
        "date",
        "sales",
        "sales_items",
        "pv",
        "uv",
        "orders",
        "reviews_posted",
        "reviews_total",
    }
    feature_cols = [c for c in dataset.columns if c not in leakage_cols]
    categorical_cols = ["genre_id", "ranking_group"]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    X_test = test[feature_cols]

    early_sales_model = make_sales_model(make_preprocessor(categorical_cols, numeric_cols))
    late_sales_model = make_sales_model(make_preprocessor(categorical_cols, numeric_cols))
    y_test = test["sales"]
    early_sales_model.fit(early_train[feature_cols], np.log1p(early_train["sales"]))
    late_sales_model.fit(late_train[feature_cols], np.log1p(late_train["sales"]))
    pred = np.zeros(len(test))
    early_test_mask = test["date"] < regime_cutoff
    late_test_mask = ~early_test_mask
    if early_test_mask.any():
        pred[early_test_mask.to_numpy()] = predict_log_model(early_sales_model, test.loc[early_test_mask, feature_cols])
    if late_test_mask.any():
        pred[late_test_mask.to_numpy()] = predict_log_model(late_sales_model, test.loc[late_test_mask, feature_cols])

    early_quantity_model = make_sales_model(make_preprocessor(categorical_cols, numeric_cols))
    late_quantity_model = make_sales_model(make_preprocessor(categorical_cols, numeric_cols))
    quantity_y_test = test["sales_items"]
    early_quantity_model.fit(early_train[feature_cols], np.log1p(early_train["sales_items"]))
    late_quantity_model.fit(late_train[feature_cols], np.log1p(late_train["sales_items"]))
    quantity_pred = np.zeros(len(test))
    if early_test_mask.any():
        quantity_pred[early_test_mask.to_numpy()] = predict_log_model(
            early_quantity_model,
            test.loc[early_test_mask, feature_cols],
        )
    if late_test_mask.any():
        quantity_pred[late_test_mask.to_numpy()] = predict_log_model(
            late_quantity_model,
            test.loc[late_test_mask, feature_cols],
        )

    metrics = metrics_frame(y_test, pred)
    metrics.update(
        {
            "train_rows": int(len(train)),
            "train_rows_early": int(len(early_train)),
            "train_rows_late": int(len(late_train)),
            "test_rows": int(len(test)),
            "test_rows_early_model": int(early_test_mask.sum()),
            "test_rows_late_model": int(late_test_mask.sum()),
            "genres": int(dataset["genre_id"].nunique()),
            "min_date": str(dataset["date"].min().date()),
            "max_date": str(dataset["date"].max().date()),
            "test_start": str(test["date"].min().date()),
            "regime_cutoff": str(regime_cutoff.date()),
            "target": "daily genre sales yen",
            "model": "two-regime pre-2024 and 2024+ models with online Rakuten event strength, promotion-effect dashboard lift features, upcoming holiday features, and absolute-error histogram gradient boosting on log sales",
        }
    )

    predictions = test[["date", "genre_id", "sales"]].copy()
    predictions["predicted_sales"] = pred
    predictions["absolute_error"] = (predictions["sales"] - predictions["predicted_sales"]).abs()
    predictions.to_csv(args.output_dir / "sales_event_predictions.csv", index=False)

    quantity_metrics = metrics_frame(quantity_y_test, quantity_pred)
    quantity_metrics.update(
        {
            "train_rows": int(len(train)),
            "train_rows_early": int(len(early_train)),
            "train_rows_late": int(len(late_train)),
            "test_rows": int(len(test)),
            "test_rows_early_model": int(early_test_mask.sum()),
            "test_rows_late_model": int(late_test_mask.sum()),
            "genres": int(dataset["genre_id"].nunique()),
            "min_date": str(dataset["date"].min().date()),
            "max_date": str(dataset["date"].max().date()),
            "test_start": str(test["date"].min().date()),
            "regime_cutoff": str(regime_cutoff.date()),
            "target": "daily genre quantity sold",
            "model": "two-regime pre-2024 and 2024+ models with online Rakuten event strength, promotion-effect dashboard lift features, upcoming holiday features, and absolute-error histogram gradient boosting on log quantity",
        }
    )

    quantity_predictions = test[["date", "genre_id", "sales_items"]].copy()
    quantity_predictions["predicted_sales_items"] = quantity_pred
    quantity_predictions["absolute_error"] = (
        quantity_predictions["sales_items"] - quantity_predictions["predicted_sales_items"]
    ).abs()
    quantity_predictions.to_csv(args.output_dir / "quantity_event_predictions.csv", index=False)

    with (args.output_dir / "sales_event_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    with (args.output_dir / "quantity_event_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(quantity_metrics, fh, indent=2)
    write_model_analysis_outputs(args.output_dir, test, pred, quantity_pred, args.event_strength_file)

    sample = X_test
    sample_y = np.log1p(y_test)
    if len(sample) > args.max_importance_rows:
        sample = sample.sample(args.max_importance_rows, random_state=42)
        sample_y = sample_y.loc[sample.index]
    importance = permutation_importance(
        late_sales_model,
        sample,
        sample_y,
        n_repeats=5,
        random_state=42,
        scoring="neg_mean_absolute_error",
    )
    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "display_name": [readable_feature_name(feature) for feature in feature_cols],
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(args.output_dir / "sales_event_feature_importance.csv", index=False)

    joblib.dump(
        {
            "sales_model_early": early_sales_model,
            "sales_model_late": late_sales_model,
            "quantity_model_early": early_quantity_model,
            "quantity_model_late": late_quantity_model,
            "feature_cols": feature_cols,
            "regime_cutoff": str(regime_cutoff.date()),
        },
        args.output_dir / "sales_event_model.joblib",
    )

    top = importance_df.head(20).sort_values("importance_mean")
    plt.figure(figsize=(9, 7))
    plt.barh(top["display_name"], top["importance_mean"])
    plt.title("Top sales model features")
    plt.xlabel("Permutation importance")
    plt.tight_layout()
    plt.savefig(args.output_dir / "sales_event_feature_importance.png", dpi=160)
    plt.close()

    print(json.dumps({"sales": metrics, "quantity": quantity_metrics}, indent=2))


if __name__ == "__main__":
    main()
