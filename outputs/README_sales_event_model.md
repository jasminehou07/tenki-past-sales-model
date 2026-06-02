# Sales Event Model

This starter model predicts historical daily Rakuten sales at the `genre_id x date`
level using:

- calendar seasonality: day of week, month, quarter, month start/end
- Rakuten events: `zero-five`, `marathon`, `supersale`, `ichiba-day`, and the other event windows in `events/events.parquet`
- Japan holidays: same-day flags, nearby holiday windows, and days to/from the next holiday
- combined lookahead: whether a promotion or holiday is coming in the next 3, 7, or 14 days
- promotion lift estimates: ranking-group-specific sales and quantity lifts from the TENKI dashboard
- online event strength: Rakuten event point multipliers, bonus multipliers, point caps, and shop-around scope
- ranking context: ranked item count, shop count, rank, and price summaries
- recent demand: lagged and rolling sales/sales item features
- marketplace regime: separate pre-2024 and 2024+ models to avoid blending the older low-seller/COVID period with the later marketplace

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
- `model_struggles.csv`: genre-level holdout error summary
- `promotion_impact.csv`: promotion-level holdout error summary
- `promotion_regression_effects.csv`: train-period promotion regression/correlation table and keep/drop decision
- `sales_event_model.joblib`: trained model and feature column list

## Modeling Notes

The default validation split uses the final 180 days as the holdout. This keeps
the evaluation honest for forecasting-style use: the model trains on earlier
history and predicts later history.

Promotion regressions are calculated on the 2024+ training regime before the
holdout. Individual promotion event features are kept only when their absolute
sales/quantity correlation passes the configured threshold.

For SKU/item-level prediction, keep this genre-level model as the baseline first.
The item-level table is sparse, so an item model should usually add item
identity, availability, price, promotion exposure, and stronger lag features.
