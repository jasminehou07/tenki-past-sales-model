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
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("work/model_cache/matplotlib").resolve()))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DEFAULT_DATA_DIR = Path("/Users/jasminehou/Downloads/TENKI")
DEFAULT_CACHE_DIR = Path("work/model_cache")
DEFAULT_OUTPUT_DIR = Path("outputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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


def build_dataset(data_dir: Path, cache_dir: Path, rebuild_cache: bool) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "daily_genre_dataset.parquet"
    if cache_path.exists() and not rebuild_cache:
        return pd.read_parquet(cache_path)

    sales = build_sales_daily(data_dir / "genre-sales")
    rankings = build_ranking_daily(data_dir / "genre-ranking")
    events = pd.read_parquet(data_dir / "events" / "events.parquet")
    event_daily = active_events_by_day(events)

    dataset = sales.merge(rankings, on=["genre_id", "date"], how="left")
    dataset = dataset.merge(event_daily, on="date", how="left")
    event_cols = [c for c in dataset.columns if c.startswith("event_")]
    for col in event_cols + ["event_count", "any_event"]:
        dataset[col] = dataset[col].fillna(0)
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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args.data_dir, args.cache_dir, args.rebuild_cache)
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
    train = dataset[dataset["date"] <= cutoff].copy()
    test = dataset[dataset["date"] > cutoff].copy()

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
    X_train = train[feature_cols]
    y_train = np.log1p(train["sales"])
    X_test = test[feature_cols]
    y_test = test["sales"]

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=500,
        l2_regularization=0.05,
        random_state=42,
    )
    model.fit(X_train, y_train)
    pred = np.expm1(model.predict(X_test)).clip(min=0)

    metrics = metrics_frame(y_test, pred)
    metrics.update(
        {
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "genres": int(dataset["genre_id"].nunique()),
            "min_date": str(dataset["date"].min().date()),
            "max_date": str(dataset["date"].max().date()),
            "test_start": str(test["date"].min().date()),
            "target": "daily genre sales yen",
        }
    )

    predictions = test[["date", "genre_id", "sales"]].copy()
    predictions["predicted_sales"] = pred
    predictions["absolute_error"] = (predictions["sales"] - predictions["predicted_sales"]).abs()
    predictions.to_csv(args.output_dir / "sales_event_predictions.csv", index=False)

    with (args.output_dir / "sales_event_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    sample = X_test
    sample_y = np.log1p(y_test)
    if len(sample) > args.max_importance_rows:
        sample = sample.sample(args.max_importance_rows, random_state=42)
        sample_y = sample_y.loc[sample.index]
    importance = permutation_importance(
        model,
        sample,
        sample_y,
        n_repeats=5,
        random_state=42,
        scoring="neg_mean_absolute_error",
    )
    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(args.output_dir / "sales_event_feature_importance.csv", index=False)

    joblib.dump({"model": model, "feature_cols": feature_cols}, args.output_dir / "sales_event_model.joblib")

    top = importance_df.head(20).sort_values("importance_mean")
    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["importance_mean"])
    plt.title("Top sales model features")
    plt.xlabel("Permutation importance")
    plt.tight_layout()
    plt.savefig(args.output_dir / "sales_event_feature_importance.png", dpi=160)
    plt.close()

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
