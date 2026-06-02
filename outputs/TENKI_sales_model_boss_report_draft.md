# TENKI Rakuten Sales Prediction Model

## Executive Summary

This project builds an interactive model dashboard for analyzing and predicting historical Rakuten sales performance. The model estimates both sales value and quantity sold by learning from seasonality, Japanese holidays, Rakuten promotion events, and category-level sales behavior. The main business goal is to move toward a future-facing tool that helps sellers prepare inventory before major demand changes, especially during Rakuten campaigns such as Shopping Marathon, Super Sale, Black Friday, Thanksgiving Festival, and 5/0 point days.

Current dashboard:

- **Interactive model link:** `https://jasminehou07.github.io/tenki-past-sales-model/`

## Business Question

Can we use historical Rakuten sales, promotion timing, holidays, and category behavior to estimate:

- How much sales volume a seller should expect during a selected period
- How many units the seller should prepare to sell
- Which promotions appear to have the strongest demand impact
- Which categories are currently less reliable and need more data or better features

## Current Model Performance

| Target | R2 | WAPE | MAE |
|---|---:|---:|---:|
| Sales value | 0.855 | 27.7% | ¥108,590 |
| Quantity sold | 0.861 | 23.5% | 13.3 units |

Interpretation:

- **R2** shows how much of the sales pattern the model explains. Values around 0.85 mean the model is capturing a large share of the broad pattern.
- **WAPE** shows weighted percent error. Sales WAPE of 27.7% means total prediction error is about 27.7% of actual sales volume in the holdout period.
- **MAE** shows the average absolute prediction miss per row. For quantity, the model misses by about 13 units per prediction row on average.

## Data and Validation Setup

| Detail | Value |
|---|---|
| Full data range | 2018-10-29 to 2026-05-31 |
| Backtest period | 2025-12-03 to 2026-05-31 |
| Training rows | 222,033 |
| Holdout rows | 17,897 |
| Genres tested | 100 |
| Model split | Pre-2024 model and 2024+ model |
| Split date | 2024-01-01 |

The model uses two regimes because earlier years and later years appear meaningfully different. Before 2024, Rakuten marketplace behavior was affected by fewer sellers and COVID-era changes. The later-period model is more relevant for future forecasting.

## Promotion Impact

The table below compares promotions that were used in the model with additional promotions that were reviewed but not included. Promotions were kept when their relationship with sales or quantity movement was stronger in the regression/correlation analysis.

| Promotion | Correlation | Estimated Sales Lift | Estimated Quantity Lift | Used in Model |
|---|---:|---:|---:|---|
| 5 and 0 day | 0.218 | 227.1% | 123.8% | Yes |
| Rakuten Super Sale | 0.140 | 172.4% | 112.1% | Yes |
| Shopping Marathon | 0.084 | 43.2% | 29.4% | Yes |
| Black Friday | 0.059 | 165.4% | 81.7% | Yes |
| Thanksgiving Festival | 0.051 | 339.9% | 102.7% | Yes |
| Wonderful Day | 0.020 | 42.5% | 17.7% | No |
| Rakuten Fashion sale | 0.014 | -12.5% | -7.0% | No |
| Sports win campaign | 0.009 | 33.1% | 11.4% | No |
| 18th Ichiba Day | 0.007 | -10.3% | -5.5% | No |

These results suggest that promotion timing and promotion type are important for predicting fluctuations, especially for quantity sold and inventory planning. Promotions marked "No" were still analyzed, but their measured relationship was weaker or less consistent, so they were not used as primary model features.

## Dashboard Sections

The dashboard is organized for both business review and model inspection:

- **Report Summary**: gives a clear summary for the selected date range, genre, and item.
- **Sales Metrics**: shows sales model accuracy.
- **Confidence Range**: estimates predicted sales range and units the seller should prepare to sell.
- **Actual vs Predicted Charts**: compares sales and quantity predictions against actual holdout results.
- **Where the Model Struggles**: highlights categories where prediction error is highest.
- **Promotion Impact**: shows how the model performs during major Rakuten events.
- **Model Details**: keeps technical validation details at the bottom so they do not distract from the business story.

## Current Limitations

The model is useful for backtesting and early demand-planning research, but it is not yet ready to be used as a final client inventory-ordering system.

Main limitations:

- Item names are not available yet, so item-level dropdowns still rely on item IDs or fallback labels.
- Current inventory output is based on predicted quantity sold, not true warehouse stock or available inventory.
- Future forecasting will require a confirmed future Rakuten promotion calendar.
- Some categories have high error because of volatile demand, sparse sales, or major product mix changes.

## Next Steps Plan

1. Build the next version of the model to predict future client sales, not only backtest past sales.
2. Add client-specific product, store, historical sales, quantity, and promotion participation data.
3. Include future-known drivers such as Rakuten promotion dates, point multipliers, coupons, discounts, holidays, and pricing changes.
4. Validate the future forecast by testing on later unseen periods before using it for client planning.
5. Turn the output into a client-ready planning view showing predicted sales, units to prepare, confidence range, and caution areas.
