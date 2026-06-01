const state = {
  predictions: [],
  features: [],
  genreLabels: new Map(),
  metrics: {},
  filtered: [],
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
  startDate: document.getElementById("startDate"),
  endDate: document.getElementById("endDate"),
  viewMode: document.getElementById("viewMode"),
  salesChart: document.getElementById("salesChart"),
  errorChart: document.getElementById("errorChart"),
  featureBars: document.getElementById("featureBars"),
  rows: document.getElementById("predictionRows"),
  downloadCsv: document.getElementById("downloadCsv"),
  chartCaption: document.getElementById("chartCaption"),
  errorCaption: document.getElementById("errorCaption"),
  tableCaption: document.getElementById("tableCaption"),
  metricR2: document.getElementById("metricR2"),
  metricWape: document.getElementById("metricWape"),
  metricMae: document.getElementById("metricMae"),
  metricRows: document.getElementById("metricRows"),
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
  const [predictionText, featureText, labelText, metrics] = await Promise.all([
    fetch("outputs/sales_event_predictions.csv").then((res) => res.text()),
    fetch("outputs/sales_event_feature_importance.csv").then((res) => res.text()),
    fetch("data/genre_labels.csv").then((res) => res.text()),
    fetch("outputs/sales_event_metrics.json").then((res) => res.json()),
  ]);

  state.predictions = parseCsv(predictionText).map((row) => ({
    date: row.date,
    genre: row.genre_id,
    sales: Number(row.sales),
    predicted: Number(row.predicted_sales),
    error: Number(row.absolute_error),
  }));
  state.features = parseCsv(featureText).map((row) => ({
    name: row.feature,
    value: Number(row.importance_mean),
  }));
  state.genreLabels = new Map(parseCsv(labelText).map((row) => [row.genre_id, row.label]));
  state.metrics = metrics;
}

function setupControls() {
  const genres = [...new Set(state.predictions.map((row) => row.genre))].sort((a, b) =>
    genreLabel(a).localeCompare(genreLabel(b), "en"),
  );
  el.genreSelect.innerHTML = [
    `<option value="all">All genres</option>`,
    ...genres.map((genre) => `<option value="${genre}">${escapeHtml(genreLabel(genre))}</option>`),
  ].join("");

  const dates = state.predictions.map((row) => row.date).sort();
  el.startDate.value = dates[0];
  el.startDate.min = dates[0];
  el.startDate.max = dates.at(-1);
  el.endDate.value = dates.at(-1);
  el.endDate.min = dates[0];
  el.endDate.max = dates.at(-1);

  [el.genreSelect, el.startDate, el.endDate, el.viewMode].forEach((control) => {
    control.addEventListener("change", updateView);
  });
  el.downloadCsv.addEventListener("click", downloadFilteredCsv);
}

function updateMetrics() {
  el.metricR2.textContent = Number(state.metrics.r2).toFixed(3);
  el.metricWape.textContent = fmtPct.format(Number(state.metrics.wape));
  el.metricMae.textContent = fmtCurrency.format(Number(state.metrics.mae));
  el.metricRows.textContent = fmtNumber.format(Number(state.metrics.test_rows));
}

function filterRows() {
  const genre = el.genreSelect.value;
  const start = el.startDate.value;
  const end = el.endDate.value;
  let selected = state.predictions.filter((row) => {
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
  const summary = summarizeRows(state.filtered);
  const selectedGenreLabel =
    el.genreSelect.value === "all" ? "all genres" : genreLabel(el.genreSelect.value);
  const unit = el.viewMode.value === "weekly" ? "weekly" : "daily";

  el.status.textContent = `${fmtNumber.format(state.filtered.length)} ${unit} rows`;
  el.chartCaption.textContent = `${selectedGenreLabel}, ${fmtCurrency.format(summary.sales)} actual sales`;
  el.errorCaption.textContent = `${fmtPct.format(summary.wape)} WAPE in selected range`;
  el.tableCaption.textContent = `${unit} rows sorted by largest error`;

  drawLineChart(el.salesChart, state.filtered, [
    { key: "sales", label: "Actual", color: "#246b52" },
    { key: "predicted", label: "Predicted", color: "#b97912" },
  ]);
  drawLineChart(el.errorChart, state.filtered, [
    { key: "error", label: "Absolute Error", color: "#b23b3b" },
  ]);
  renderFeatureBars();
  renderRows();
}

function drawLineChart(canvas, rows, series) {
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
    ctx.fillText(shortMoney(value), 10, y + 4);
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

function renderFeatureBars() {
  const top = state.features.slice(0, 12);
  const max = Math.max(...top.map((item) => item.value), 1);
  el.featureBars.innerHTML = top
    .map((item) => {
      const width = `${Math.max(2, (item.value / max) * 100)}%`;
      return `
        <div class="feature" title="${item.name}">
          <span class="feature-name">${friendlyFeature(item.name)}</span>
          <span class="track"><span class="fill" style="width:${width}"></span></span>
          <span class="feature-value">${item.value.toFixed(3)}</span>
        </div>
      `;
    })
    .join("");
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
  el.rows.innerHTML = rows
    .map(
      (row) => `
        <tr>
          <td>${row.date}</td>
          <td>${escapeHtml(genreLabel(row.genre))}</td>
          <td>${fmtCurrency.format(row.sales)}</td>
          <td>${fmtCurrency.format(row.predicted)}</td>
          <td>${fmtCurrency.format(row.error)}</td>
        </tr>
      `,
    )
    .join("");
}

function downloadFilteredCsv() {
  const header = "date,genre_id,genre_label,sales,predicted_sales,absolute_error";
  const body = state.filtered
    .map((row) =>
      [row.date, row.genre, csvCell(genreLabel(row.genre)), row.sales, row.predicted, row.error].join(","),
    )
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
