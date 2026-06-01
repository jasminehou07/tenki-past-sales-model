# Sales Event Model

This starter model predicts historical daily Rakuten sales at the `genre_id x date`
level using:

- calendar seasonality: day of week, month, quarter, month start/end
- Rakuten events: `zero-five`, `marathon`, `supersale`, `ichiba-day`, and the other event windows in `events/events.parquet`
- ranking context: ranked item count, shop count, rank, and price summaries
- recent demand: lagged and rolling sales/sales item features

The model intentionally excludes same-day `pv`, `uv`, `orders`, and other
conversion metrics from the feature set because those are usually outcomes of
the event day, not inputs you would know before predicting sales.

## Run

From the project folder:

```bash
work/.venv/bin/python outputs/sales_event_model.py --rebuild-cache
```

Outputs:

- `sales_event_metrics.json`: time-based holdout metrics
- `sales_event_predictions.csv`: actual vs predicted sales for the holdout period
- `sales_event_feature_importance.csv`: model feature influence
- `sales_event_feature_importance.png`: top feature chart
- `sales_event_model.joblib`: trained model and feature column list

## Modeling Notes

The default validation split uses the final 180 days as the holdout. This keeps
the evaluation honest for forecasting-style use: the model trains on earlier
history and predicts later history.

For SKU/item-level prediction, keep this genre-level model as the baseline first.
The item-level table is sparse, so an item model should usually add item
identity, availability, price, promotion exposure, and stronger lag features.
