# TENKI Rakuten Sales Prediction Model

## Executive Summary

This project builds an interactive model dashboard for analyzing and predicting historical Rakuten sales performance. The model estimates both sales value and quantity sold by learning from seasonality, Japanese holidays, Rakuten promotion events, and category-level sales behavior.

The main business goal is to move toward a future-facing tool that helps sellers prepare inventory before major demand changes, especially during Rakuten campaigns such as Shopping Marathon, Super Sale, Black Friday, Thanksgiving Festival, and 5/0 point days.

Current dashboard:

- Local page: `http://127.0.0.1:8877/?v=organized-dashboard`
- GitHub Pages: `https://jasminehou07.github.io/tenki-past-sales-model/`

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

Plain-English interpretation:

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

The model currently uses promotions that showed stronger relationship with sales or quantity movement. These were selected using promotion regression/correlation analysis.

| Promotion | Estimated Sales Lift | Estimated Quantity Lift | Used in Model |
|---|---:|---:|---|
| 5 and 0 day | 227.1% | 123.8% | Yes |
| Rakuten Super Sale | 172.4% | 112.1% | Yes |
| Shopping Marathon | 43.2% | 29.4% | Yes |
| Black Friday | 165.4% | 81.7% | Yes |
| Thanksgiving Festival | 339.9% | 102.7% | Yes |

These results suggest that promotion timing and promotion type are important for predicting fluctuations, especially for quantity sold and inventory planning.

## Dashboard Sections

The dashboard is organized for both business review and model inspection:

- **Report Summary**: gives a boss-ready summary for the selected date range, genre, and item.
- **Sales Metrics**: shows high-level sales model accuracy.
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
- Price, discount, coupon, shipping, and stockout data are not included yet.
- Future forecasting will require a confirmed future Rakuten promotion calendar.
- Some categories have high error because of volatile demand, sparse sales, or major product mix changes.

## Item Name File Needed

The next important file from the team should map item IDs to readable item names or FTP titles.

Ideal columns:

| Column | Why It Helps |
|---|---|
| `item_id` | Connects the file to the current TENKI item IDs |
| `item_name` or `ftp_title` | Lets the dashboard show actual product names |
| `genre_id` | Confirms the item belongs to the right category |
| `shop_id` or `shop_code` | Allows seller/store-level modeling later |
| `price` or `list_price` | Helps model demand changes from price |
| `jan` / SKU / product code | Helps identify the same product across files |

Once this file is available, the dashboard can replace fallback item labels with real product names and support more useful item-level reporting.

## Recommended Next Steps

1. Add the item-name / FTP-title file and update the item dropdown.
2. Add price, discount, coupon, and point-back features.
3. Add stockout or inventory availability data if available.
4. Build a future promotion calendar input.
5. Create client/store-specific model views once seller identifiers are available.
6. Calibrate confidence ranges using true forecast error, not only display caps.
7. Convert this report into a short boss-facing PDF or slide deck after item names are added.

## Suggested Boss Talking Points

- The model already captures broad sales and quantity patterns with R2 around 0.86.
- Promotion events clearly matter and should be included in any future demand forecast.
- The dashboard can estimate how many units sellers should prepare for a selected date range.
- Accuracy is promising for a first backtesting tool, but production forecasting needs better product, price, and stock data.
- The next highest-value improvement is mapping item IDs to real item names or FTP titles.
