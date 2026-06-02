const state = {
  predictions: [],
  quantityPredictions: [],
  features: [],
  struggles: [],
  promotions: [],
  itemOptions: [],
  itemActuals: [],
  genreLabels: new Map(),
  metrics: {},
  quantityMetrics: {},
  filtered: [],
  filteredQuantity: [],
};

const fmtCurrency = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "JPY",
  maximumFractionDigits: 0,
});

const fmtNumber = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const fmtPct = new Intl.NumberFormat("en-US", {
  style: "percent",
  maximumFractionDigits: 1,
});

const el = {
  status: document.getElementById("status"),
  genreSelect: document.getElementById("genreSelect"),
  itemSelect: document.getElementById("itemSelect"),
  startDate: document.getElementById("startDate"),
  endDate: document.getElementById("endDate"),
  viewMode: document.getElementById("viewMode"),
  salesChart: document.getElementById("salesChart"),
  quantityChart: document.getElementById("quantityChart"),
  errorChart: document.getElementById("errorChart"),
  featureBars: document.getElementById("featureBars"),
  struggleRows: document.getElementById("struggleRows"),
  promotionRows: document.getElementById("promotionRows"),
  rows: document.getElementById("predictionRows"),
  downloadCsv: document.getElementById("downloadCsv"),
  chartCaption: document.getElementById("chartCaption"),
  quantityCaption: document.getElementById("quantityCaption"),
  errorCaption: document.getElementById("errorCaption"),
  tableCaption: document.getElementById("tableCaption"),
  confidenceCaption: document.getElementById("confidenceCaption"),
  salesRange: document.getElementById("salesRange"),
  actualSalesRange: document.getElementById("actualSalesRange"),
  inventoryRange: document.getElementById("inventoryRange"),
  actualInventoryRange: document.getElementById("actualInventoryRange"),
  metricR2: document.getElementById("metricR2"),
  metricWape: document.getElementById("metricWape"),
  metricMae: document.getElementById("metricMae"),
  metricRows: document.getElementById("metricRows"),
  quantityMetricR2: document.getElementById("quantityMetricR2"),
  quantityMetricWape: document.getElementById("quantityMetricWape"),
  quantityMetricMae: document.getElementById("quantityMetricMae"),
  quantityMetricRows: document.getElementById("quantityMetricRows"),
};

function parseCsv(text) {
  const [headerLine, ...lines] = text.trim().split(/\r?\n/);
  const headers = parseCsvLine(headerLine);
  return lines.map((line) => {
    const values = parseCsvLine(line);
    return Object.fromEntries(headers.map((header, index) => [header, values[index]]));
  });
}

function parseCsvLine(line) {
  const values = [];
  let current = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    const next = line[i + 1];
    if (char === '"' && quoted && next === '"') {
      current += '"';
      i += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      values.push(current);
      current = "";
    } else {
      current += char;
    }
  }
  values.push(current);
  return values;
}

async function loadData() {
  const [
    predictionText,
    quantityText,
    featureText,
    struggleText,
    promotionText,
    itemOptionText,
    itemActualText,
    labelText,
    metrics,
    quantityMetrics,
  ] = await Promise.all([
    fetchText("outputs/sales_event_predictions.csv"),
    fetchText("outputs/quantity_event_predictions.csv"),
    fetchText("outputs/sales_event_feature_importance.csv"),
    fetchText("outputs/model_struggles.csv"),
    fetchText("outputs/promotion_impact.csv"),
    fetchText("outputs/item_options.csv"),
    fetchText("outputs/item_holdout_actuals.csv"),
    fetchText("data/genre_labels.csv"),
    fetchJson("outputs/sales_event_metrics.json"),
    fetchJson("outputs/quantity_event_metrics.json"),
  ]);

  state.predictions = parseCsv(predictionText).map((row) => ({
    date: row.date,
    genre: row.genre_id,
    sales: Number(row.sales),
    predicted: Number(row.predicted_sales),
    error: Number(row.absolute_error),
  }));
  state.quantityPredictions = parseCsv(quantityText).map((row) => ({
    date: row.date,
    genre: row.genre_id,
    sales: Number(row.sales_items),
    predicted: Number(row.predicted_sales_items),
    error: Number(row.absolute_error),
  }));
  state.features = parseCsv(featureText).map((row) => ({
    name: row.feature,
    displayName: row.display_name || friendlyFeature(row.feature),
    value: Number(row.importance_mean),
  }));
  state.struggles = parseCsv(struggleText).map((row) => ({
    genre: row.genre_id,
    group: row.ranking_group,
    salesWape: Number(row.sales_wape),
    quantityWape: Number(row.quantity_wape),
    salesBias: Number(row.sales_bias),
    quantityBias: Number(row.quantity_bias),
    actualSales: Number(row.actual_sales),
  }));
  state.promotions = parseCsv(promotionText).map((row) => ({
    name: row.display_name || row.event_name,
    eventName: row.event_name,
    days: Number(row.days),
    actualSales: Number(row.actual_sales),
    salesWape: Number(row.sales_wape),
    quantityWape: Number(row.quantity_wape),
    maxPoint: Number(row.max_point_multiplier),
    bonusPoint: Number(row.bonus_point_multiplier),
    scope: row.scope,
    source: row.source_url,
  }));
  state.itemOptions = parseCsv(itemOptionText).map((row) => ({
    genre: row.genre_id,
    item: row.item,
    label: row.item_label,
    totalSales: Number(row.total_sales),
    totalQuantity: Number(row.total_quantity),
    activeDays: Number(row.active_days),
  }));
  state.itemActuals = parseCsv(itemActualText).map((row) => ({
    date: row.date,
    genre: row.genre_id,
    item: row.item,
    sales: Number(row.sales),
    quantity: Number(row.quantity),
  }));
  state.genreLabels = new Map(parseCsv(labelText).map((row) => [row.genre_id, row.label]));
  state.metrics = metrics;
  state.quantityMetrics = quantityMetrics;
}

function fetchText(path) {
  return fetch(`${path}?v=event-strength-struggles`, { cache: "no-store" }).then((res) => res.text());
}

function fetchJson(path) {
  return fetch(`${path}?v=event-strength-struggles`, { cache: "no-store" }).then((res) => res.json());
}

function setupControls() {
  const genres = [...new Set(state.predictions.map((row) => row.genre))].sort((a, b) =>
    genreLabel(a).localeCompare(genreLabel(b), "en"),
  );
  el.genreSelect.innerHTML = [
    `<option value="all">All genres</option>`,
    ...genres.map((genre) => `<option value="${genre}">${escapeHtml(genreLabel(genre))}</option>`),
  ].join("");
  updateItemOptions();

  const dates = state.predictions.map((row) => row.date).sort();
  el.startDate.value = dates[0];
  el.startDate.min = dates[0];
  el.startDate.max = dates.at(-1);
  el.endDate.value = dates.at(-1);
  el.endDate.min = dates[0];
  el.endDate.max = dates.at(-1);

  [el.startDate, el.endDate, el.viewMode].forEach((control) => {
    control.addEventListener("change", updateView);
  });
  el.genreSelect.addEventListener("change", () => {
    updateItemOptions();
    updateView();
  });
  el.itemSelect.addEventListener("change", updateView);
  el.downloadCsv.addEventListener("click", downloadFilteredCsv);
}

function updateItemOptions() {
  const genre = el.genreSelect.value;
  const items = state.itemOptions
    .filter((item) => genre !== "all" && item.genre === genre)
    .sort((a, b) => b.totalSales - a.totalSales);
  el.itemSelect.disabled = genre === "all" || !items.length;
  el.itemSelect.innerHTML = [
    `<option value="all">All items in genre</option>`,
    ...items.map(
      (item) =>
        `<option value="${item.item}">${escapeHtml(item.label)} - ${fmtNumber.format(item.totalQuantity)} units</option>`,
    ),
  ].join("");
}

function updateMetrics() {
  el.metricR2.textContent = Number(state.metrics.r2).toFixed(3);
  el.metricWape.textContent = fmtPct.format(Number(state.metrics.wape));
  el.metricMae.textContent = fmtCurrency.format(Number(state.metrics.mae));
  el.metricRows.textContent = fmtNumber.format(Number(state.metrics.test_rows));
  el.quantityMetricR2.textContent = Number(state.quantityMetrics.r2).toFixed(3);
  el.quantityMetricWape.textContent = fmtPct.format(Number(state.quantityMetrics.wape));
  el.quantityMetricMae.textContent = `${Number(state.quantityMetrics.mae).toFixed(1)} items`;
  el.quantityMetricRows.textContent = fmtNumber.format(Number(state.quantityMetrics.test_rows));
}

function filterRows() {
  return filterPredictionRows(state.predictions);
}

function filterQuantityRows() {
  return filterPredictionRows(state.quantityPredictions);
}

function filterPredictionRows(rows) {
  const genre = el.genreSelect.value;
  const start = el.startDate.value;
  const end = el.endDate.value;
  let selected = rows.filter((row) => {
    return (genre === "all" || row.genre === genre) && row.date >= start && row.date <= end;
  });
  if (genre === "all") selected = aggregateByDate(selected);
  return el.viewMode.value === "weekly" ? toWeekly(selected) : selected;
}

function aggregateByDate(rows) {
  const buckets = new Map();
  rows.forEach((row) => {
    const current = buckets.get(row.date) || {
      date: row.date,
      genre: "all",
      sales: 0,
      predicted: 0,
      error: 0,
    };
    current.sales += row.sales;
    current.predicted += row.predicted;
    buckets.set(row.date, current);
  });
  return [...buckets.values()]
    .map((row) => ({ ...row, error: Math.abs(row.sales - row.predicted) }))
    .sort((a, b) => a.date.localeCompare(b.date));
}

function toWeekly(rows) {
  const buckets = new Map();
  rows.forEach((row) => {
    const date = new Date(`${row.date}T00:00:00`);
    const day = date.getDay();
    const diffToMonday = (day + 6) % 7;
    date.setDate(date.getDate() - diffToMonday);
    const week = date.toISOString().slice(0, 10);
    const key = `${week}|${row.genre}`;
    const current = buckets.get(key) || {
      date: week,
      genre: row.genre,
      sales: 0,
      predicted: 0,
      error: 0,
    };
    current.sales += row.sales;
    current.predicted += row.predicted;
    buckets.set(key, current);
  });
  return [...buckets.values()]
    .map((row) => ({ ...row, error: Math.abs(row.sales - row.predicted) }))
    .sort((a, b) => a.date.localeCompare(b.date));
}

function summarizeRows(rows) {
  const sales = rows.reduce((sum, row) => sum + row.sales, 0);
  const predicted = rows.reduce((sum, row) => sum + row.predicted, 0);
  const error = rows.reduce((sum, row) => sum + Math.abs(row.sales - row.predicted), 0);
  const wape = error / Math.max(sales, 1);
  return { sales, predicted, error, wape };
}

function updateView() {
  state.filtered = filterRows();
  state.filteredQuantity = filterQuantityRows();
  const summary = summarizeRows(state.filtered);
  const quantitySummary = summarizeRows(state.filteredQuantity);
  const selectedGenreLabel =
    el.genreSelect.value === "all" ? "all genres" : genreLabel(el.genreSelect.value);
  const unit = el.viewMode.value === "weekly" ? "weekly" : "daily";

  el.status.textContent = `${fmtNumber.format(state.filtered.length)} ${unit} rows`;
  el.chartCaption.textContent = `${selectedGenreLabel}, ${fmtCurrency.format(summary.sales)} actual sales`;
  el.quantityCaption.textContent =
    `${selectedGenreLabel}, ${fmtNumber.format(Math.round(quantitySummary.sales))} actual items, ` +
    `${fmtPct.format(quantitySummary.wape)} WAPE`;
  el.errorCaption.textContent = `${fmtPct.format(summary.wape)} WAPE in selected range`;
  el.tableCaption.textContent = `${unit} rows sorted by largest error`;

  drawLineChart(el.salesChart, state.filtered, [
    { key: "sales", label: "Actual", color: "#246b52" },
    { key: "predicted", label: "Predicted", color: "#b97912" },
  ]);
  drawLineChart(
    el.quantityChart,
    state.filteredQuantity,
    [
      { key: "sales", label: "Actual", color: "#246b52" },
      { key: "predicted", label: "Predicted", color: "#b97912" },
    ],
    shortCount,
  );
  drawLineChart(el.errorChart, state.filtered, [
    { key: "error", label: "Absolute Error", color: "#b23b3b" },
  ]);
  renderFeatureBars();
  renderConfidenceRange(summary, quantitySummary, selectedGenreLabel);
  renderStruggles();
  renderPromotions();
  renderRows();
}

function renderConfidenceRange(summary, quantitySummary, selectedGenreLabel) {
  const item = el.itemSelect.value;
  const itemData = getSelectedItemData();
  const salesBase = itemData ? itemData.predictedSales : summary.predicted;
  const quantityBase = itemData ? itemData.predictedQuantity : quantitySummary.predicted;
  const actualSales = itemData ? itemData.actualSales : summary.sales;
  const actualQuantity = itemData ? itemData.actualQuantity : quantitySummary.sales;
  const confidence = confidenceForSelection(itemData);

  el.salesRange.textContent = `${fmtCurrency.format(salesBase * confidence.salesLow)} - ${fmtCurrency.format(
    salesBase * confidence.salesHigh,
  )}`;
  el.actualSalesRange.textContent = fmtCurrency.format(actualSales);
  el.inventoryRange.textContent = `${fmtNumber.format(Math.round(quantityBase * confidence.quantityLow))} - ${fmtNumber.format(
    Math.round(quantityBase * confidence.quantityHigh),
  )} units`;
  el.actualInventoryRange.textContent = `${fmtNumber.format(Math.round(actualQuantity))} units`;
  const itemLabel = itemData ? `, ${itemData.label}` : "";
  el.confidenceCaption.textContent = `${selectedGenreLabel}${itemLabel}, ${el.startDate.value} to ${el.endDate.value}`;
}

function confidenceForSelection(itemData) {
  const genre = el.genreSelect.value;
  const sourceRows =
    genre === "all" ? state.predictions : state.predictions.filter((row) => row.genre === genre);
  const quantityRows =
    genre === "all"
      ? state.quantityPredictions
      : state.quantityPredictions.filter((row) => row.genre === genre);
  const salesRatios = sourceRows
    .filter((row) => row.predicted > 0 && row.sales > 0)
    .map((row) => clamp(row.sales / row.predicted, 0.15, 3));
  const quantityRatios = quantityRows
    .filter((row) => row.predicted > 0 && row.sales > 0)
    .map((row) => clamp(row.sales / row.predicted, 0.15, 3));
  const fallback = itemData ? 0.35 : 0.25;
  return {
    salesLow: quantile(salesRatios, 0.1) || 1 - fallback,
    salesHigh: quantile(salesRatios, 0.9) || 1 + fallback,
    quantityLow: quantile(quantityRatios, 0.1) || 1 - fallback,
    quantityHigh: quantile(quantityRatios, 0.9) || 1 + fallback,
  };
}

function getSelectedItemData() {
  const genre = el.genreSelect.value;
  const item = el.itemSelect.value;
  if (genre === "all" || item === "all") return null;
  const option = state.itemOptions.find((row) => row.genre === genre && row.item === item);
  const itemRows = state.itemActuals.filter(
    (row) => row.genre === genre && row.item === item && row.date >= el.startDate.value && row.date <= el.endDate.value,
  );
  const actualSales = itemRows.reduce((sum, row) => sum + row.sales, 0);
  const actualQuantity = itemRows.reduce((sum, row) => sum + row.quantity, 0);
  const genreRows = state.itemActuals.filter(
    (row) => row.genre === genre && row.date >= el.startDate.value && row.date <= el.endDate.value,
  );
  const genreSales = genreRows.reduce((sum, row) => sum + row.sales, 0);
  const genreQuantity = genreRows.reduce((sum, row) => sum + row.quantity, 0);
  const genrePredSales = state.filtered.reduce((sum, row) => sum + row.predicted, 0);
  const genrePredQuantity = state.filteredQuantity.reduce((sum, row) => sum + row.predicted, 0);
  const salesShare = genreSales > 0 ? actualSales / genreSales : 0;
  const quantityShare = genreQuantity > 0 ? actualQuantity / genreQuantity : salesShare;
  return {
    label: option?.label || `Item #${item}`,
    actualSales,
    actualQuantity,
    predictedSales: genrePredSales * salesShare,
    predictedQuantity: genrePredQuantity * quantityShare,
  };
}

function quantile(values, q) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (sorted[base + 1] === undefined) return sorted[base];
  return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function drawLineChart(canvas, rows, series, valueFormatter = shortMoney) {
  const ctx = canvas.getContext("2d");
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(320, Math.floor(rect.width * ratio));
  canvas.height = Math.floor(Number(canvas.getAttribute("height")) * ratio);
  ctx.scale(ratio, ratio);

  const width = canvas.width / ratio;
  const height = canvas.height / ratio;
  const pad = { top: 20, right: 20, bottom: 34, left: 72 };
  ctx.clearRect(0, 0, width, height);

  if (!rows.length) {
    ctx.fillStyle = "#647067";
    ctx.font = "14px system-ui";
    ctx.fillText("No rows for this selection", pad.left, pad.top + 20);
    return;
  }

  const allValues = rows.flatMap((row) => series.map((item) => row[item.key]));
  const max = Math.max(...allValues, 1);
  const min = Math.min(...allValues, 0);
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const xFor = (index) => pad.left + (plotW * index) / Math.max(rows.length - 1, 1);
  const yFor = (value) => pad.top + plotH - ((value - min) / (max - min || 1)) * plotH;

  ctx.strokeStyle = "#e1e7e1";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#647067";
  ctx.font = "12px system-ui";
  for (let tick = 0; tick <= 4; tick += 1) {
    const value = min + ((max - min) * tick) / 4;
    const y = yFor(value);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    ctx.fillText(valueFormatter(value), 10, y + 4);
  }

  series.forEach((item) => {
    ctx.beginPath();
    rows.forEach((row, index) => {
      const x = xFor(index);
      const y = yFor(row[item.key]);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = item.color;
    ctx.lineWidth = 2.5;
    ctx.stroke();
  });

  const dateMarks = [0, Math.floor((rows.length - 1) / 2), rows.length - 1];
  ctx.fillStyle = "#647067";
  dateMarks.forEach((index) => {
    const label = rows[index]?.date;
    if (!label) return;
    const x = xFor(index);
    ctx.fillText(label.slice(5), Math.min(x, width - pad.right - 36), height - 10);
  });
}

function shortMoney(value) {
  const abs = Math.abs(value);
  if (abs >= 1_000_000_000) return `¥${(value / 1_000_000_000).toFixed(1)}B`;
  if (abs >= 1_000_000) return `¥${(value / 1_000_000).toFixed(1)}M`;
  if (abs >= 1_000) return `¥${(value / 1_000).toFixed(0)}K`;
  return `¥${value.toFixed(0)}`;
}

function shortCount(value) {
  const abs = Math.abs(value);
  if (abs >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (abs >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return fmtNumber.format(Math.round(value));
}

function renderFeatureBars() {
  const top = state.features.slice(0, 10);
  const max = Math.max(...top.map((item) => item.value), 1);
  el.featureBars.innerHTML = top
    .map((item) => {
      const width = `${Math.max(2, (item.value / max) * 100)}%`;
      return `
        <div class="feature" title="${item.name}">
          <span class="feature-name">${escapeHtml(item.displayName)}</span>
          <span class="track"><span class="fill" style="width:${width}"></span></span>
          <span class="feature-value">${item.value.toFixed(3)}</span>
        </div>
      `;
    })
    .join("");
}

function renderStruggles() {
  const rows = state.struggles.slice(0, 12);
  el.struggleRows.innerHTML = rows
    .map(
      (row) => `
        <tr>
          <td>${escapeHtml(genreLabel(row.genre))}</td>
          <td>${escapeHtml(row.group || "Unknown")}</td>
          <td>${fmtPct.format(row.salesWape)}</td>
          <td>${formatBias(row.salesBias)}</td>
          <td>${fmtPct.format(row.quantityWape)}</td>
          <td>${fmtCurrency.format(row.actualSales)}</td>
        </tr>
      `,
    )
    .join("");
}

function renderPromotions() {
  const rows = [...state.promotions].sort((a, b) => b.actualSales - a.actualSales).slice(0, 12);
  el.promotionRows.innerHTML = rows
    .map((row) => {
      const source = row.source
        ? `<a href="${escapeHtml(row.source)}" target="_blank" rel="noreferrer">Rules</a>`
        : "--";
      return `
        <tr>
          <td>${escapeHtml(row.name)}</td>
          <td>${row.maxPoint ? `${row.maxPoint.toFixed(0)}x` : "--"}</td>
          <td>${escapeHtml(friendlyScope(row.scope))}</td>
          <td>${fmtNumber.format(row.days)}</td>
          <td>${fmtCurrency.format(row.actualSales)}</td>
          <td>${fmtPct.format(row.salesWape)}</td>
          <td>${fmtPct.format(row.quantityWape)}</td>
          <td>${source}</td>
        </tr>
      `;
    })
    .join("");
}

function formatBias(value) {
  const label = value > 0 ? "over" : "under";
  return `${fmtPct.format(Math.abs(value))} ${label}`;
}

function friendlyScope(value) {
  return String(value || "mixed").replaceAll("_", " ");
}

function friendlyFeature(name) {
  return name
    .replace(/^event_/, "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function genreLabel(id) {
  if (id === "all") return "All genres";
  return state.genreLabels.get(String(id)) || `Genre ${id}`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderRows() {
  const rows = [...state.filtered].sort((a, b) => b.error - a.error).slice(0, 250);
  const quantityByKey = new Map(
    state.filteredQuantity.map((row) => [`${row.date}|${row.genre}`, row]),
  );
  el.rows.innerHTML = rows
    .map((row) => {
      const quantity = quantityByKey.get(`${row.date}|${row.genre}`);
      return `
        <tr>
          <td>${row.date}</td>
          <td>${escapeHtml(genreLabel(row.genre))}</td>
          <td>${fmtCurrency.format(row.sales)}</td>
          <td>${fmtCurrency.format(row.predicted)}</td>
          <td>${fmtCurrency.format(row.error)}</td>
          <td>${quantity ? fmtNumber.format(Math.round(quantity.sales)) : "--"}</td>
          <td>${quantity ? fmtNumber.format(Math.round(quantity.predicted)) : "--"}</td>
          <td>${quantity ? fmtNumber.format(Math.round(quantity.error)) : "--"}</td>
        </tr>
      `;
    })
    .join("");
}

function downloadFilteredCsv() {
  const quantityByKey = new Map(
    state.filteredQuantity.map((row) => [`${row.date}|${row.genre}`, row]),
  );
  const header =
    "date,genre_id,genre_label,sales,predicted_sales,sales_absolute_error,quantity,predicted_quantity,quantity_absolute_error";
  const body = state.filtered
    .map((row) => {
      const quantity = quantityByKey.get(`${row.date}|${row.genre}`);
      return [
        row.date,
        row.genre,
        csvCell(genreLabel(row.genre)),
        row.sales,
        row.predicted,
        row.error,
        quantity?.sales ?? "",
        quantity?.predicted ?? "",
        quantity?.error ?? "",
      ].join(",");
    })
    .join("\n");
  const blob = new Blob([`${header}\n${body}\n`], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "filtered_sales_event_predictions.csv";
  link.click();
  URL.revokeObjectURL(url);
}

function csvCell(value) {
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

window.addEventListener("resize", () => {
  if (state.filtered.length) {
    drawLineChart(el.salesChart, state.filtered, [
      { key: "sales", label: "Actual", color: "#246b52" },
      { key: "predicted", label: "Predicted", color: "#b97912" },
    ]);
    drawLineChart(
      el.quantityChart,
      state.filteredQuantity,
      [
        { key: "sales", label: "Actual", color: "#246b52" },
        { key: "predicted", label: "Predicted", color: "#b97912" },
      ],
      shortCount,
    );
    drawLineChart(el.errorChart, state.filtered, [
      { key: "error", label: "Absolute Error", color: "#b23b3b" },
    ]);
  }
});

loadData()
  .then(() => {
    setupControls();
    updateMetrics();
    updateView();
  })
  .catch((error) => {
    el.status.textContent = "Could not load model outputs";
    console.error(error);
  });
