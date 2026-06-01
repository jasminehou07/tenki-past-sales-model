# Rakuten TENKI Sales Event Model

This repository contains a first-pass model for predicting historical daily
Rakuten sales by genre using TENKI sales, ranking, and event parquet exports.

The pipeline aggregates item-level sales to `genre_id x date`, adds calendar,
Rakuten event, Japan holiday, ranking, price, and lagged demand features, then
trains a time-based backtest model on earlier history and evaluates it on the
final 180 days.

## Contents

- `index.html`, `styles.css`, `app.js`: interactive model dashboard
- `outputs/sales_event_model.py`: training and evaluation pipeline
- `outputs/README_sales_event_model.md`: detailed model notes
- `outputs/sales_event_metrics.json`: latest holdout metrics
- `outputs/sales_event_predictions.csv`: actual vs predicted holdout sales
- `outputs/quantity_event_metrics.json`: latest holdout metrics for quantity sold
- `outputs/quantity_event_predictions.csv`: actual vs predicted holdout quantity sold
- `outputs/sales_event_feature_importance.csv`: feature importance table
- `outputs/sales_event_feature_importance.png`: feature importance chart
- `outputs/sales_event_model.joblib`: trained model artifact
- `data/japan_holidays.csv`: official Japan holiday calendar used for holiday lookahead features

## Data

The raw TENKI parquet files are not committed. The script expects this local
layout by default:

```text
/Users/jasminehou/Downloads/TENKI/
  events/events.parquet
  genre-sales/*.parquet
  genre-ranking/*.parquet
```

Use `--data-dir` to point at a different copy of the same folder structure.

## Setup

```bash
python3 -m venv work/.venv
work/.venv/bin/python -m pip install -r requirements.txt
```

## Run

Open the interactive page on GitHub Pages:

```text
https://jasminehou07.github.io/tenki-past-sales-model/
```

No local Python server is needed for the GitHub Pages version.

To open the interactive page locally:

```bash
python3 -m http.server 8877
```

Then visit:

```text
http://127.0.0.1:8877/
```

Rebuild the model outputs:

```bash
work/.venv/bin/python outputs/sales_event_model.py --rebuild-cache
```

After the first run, omit `--rebuild-cache` to reuse the aggregated daily cache:

```bash
work/.venv/bin/python outputs/sales_event_model.py
```

## Latest Backtest

- Test period starts: 2025-12-03
- Data through: 2026-05-31
- Genres: 100
- Sales R2: 0.841
- Sales WAPE: 28.2%
- Sales MAE: 110,585 yen daily genre sales
- Quantity R2: 0.815
- Quantity WAPE: 25.9%
- Quantity MAE: 14.6 items per daily genre row

This pass uses genre one-hot encoding, all event types from `events.parquet`,
pre-event and post-event timing windows, official Japan holidays, combined
promo-or-holiday lookahead features, and an absolute-error gradient boosting
objective. The strongest signals are recent demand, active item count, Rakuten
event timing, and distance to upcoming promo/holiday days.
