# Rakuten TENKI Sales Event Model

This repository contains a first-pass model for predicting historical daily
Rakuten sales by genre using TENKI sales, ranking, and event parquet exports.

The pipeline aggregates item-level sales to `genre_id x date`, adds calendar,
Rakuten event, ranking, price, and lagged demand features, then trains a
time-based backtest model on earlier history and evaluates it on the final 180
days.

## Contents

- `index.html`, `styles.css`, `app.js`: interactive model dashboard
- `outputs/sales_event_model.py`: training and evaluation pipeline
- `outputs/README_sales_event_model.md`: detailed model notes
- `outputs/sales_event_metrics.json`: latest holdout metrics
- `outputs/sales_event_predictions.csv`: actual vs predicted holdout sales
- `outputs/sales_event_feature_importance.csv`: feature importance table
- `outputs/sales_event_feature_importance.png`: feature importance chart
- `outputs/sales_event_model.joblib`: trained model artifact

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

Open the interactive page locally:

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
- R2: 0.773
- WAPE: 34.1%
- MAE: 133,704 yen daily genre sales

The strongest signals in this pass are recent demand, active item count,
`zero-five`, `marathon`, day-of-month effects, and overall event count.
