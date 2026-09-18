// Forecast card: history line, median forecast, p30-p70 and p10-p90 fan,
// crosshair tooltip, backtest "trust" strip and a table view of the numbers.
// All labels arrive from tools/APIs, so they only ever enter the DOM as text.

const NS = "http://www.w3.org/2000/svg";
const H = 270; // includes the x-axis band
const M = { top: 18, right: 64, bottom: 26, left: 46 };
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};
const svg = (tag, attrs = {}) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
};

export function compact(v) {
  const a = Math.abs(v);
  if (a >= 1e9) return trim(v / 1e9) + "B";
  if (a >= 1e6) return trim(v / 1e6) + "M";
  if (a >= 1e4) return trim(v / 1e3) + "K";
  if (a >= 100) return Math.round(v).toLocaleString("en-US");
  if (a >= 1) return trim(v);
  return v === 0 ? "0" : Number(v.toPrecision(3)).toString();
}
const trim = (v) => String(Number(v.toFixed(Math.abs(v) >= 100 ? 0 : Math.abs(v) >= 10 ? 1 : 2)));
const full = (v) =>
  Math.abs(v) >= 100 ? Math.round(v).toLocaleString("en-US") : Number(v.toPrecision(4)).toLocaleString("en-US", { maximumFractionDigits: 8 });
const day = (iso) => {
  const [y, m, d] = iso.split("-").map(Number);
  return `${d} ${MONTHS[m - 1]} ${y}`;
};

function niceTicks(lo, hi, count = 4) {
  const span = hi - lo || Math.abs(hi) || 1;
  const raw = span / count;
  const pow = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * pow).find((s) => s >= raw);
  const start = Math.floor(lo / step) * step;
  const ticks = [];
  for (let t = start; t < hi + step * 0.999; t += step) ticks.push(Number(t.toPrecision(12)));
  return ticks;
}

export function forecastCard(chart) {
  const card = el("article", "card");
  card.append(el("h3", "", chart.label));

  const sub = el("p", "sub");
  sub.append(`${chart.unit} · ${chart.forecast.dates.length}-day forecast by TimesFM 3.0 · `);
  const link = el("a", "", "source");
  if (/^https:\/\//.test(chart.source)) link.href = chart.source;
  link.target = "_blank";
  link.rel = "noopener";
  sub.append(link);
  card.append(sub);

  const key = el("div", "key");
  for (const [cls, label] of [["", "Actual"], ["f", "Forecast (median)"], ["b", "Likely range (p30–p70)"], ["b2", "Wide range (p10–p90)"]]) {
    const item = el("span");
    item.append(el("i", cls), label);
    key.append(item);
  }
  card.append(key);

  const plot = el("div", "plot");
  const tip = el("div", "tip");
  tip.hidden = true;
  plot.append(tip);
  card.append(plot);

  const draw = () => render(plot, tip, chart);
  new ResizeObserver(draw).observe(plot);

  if (chart.backtest) card.append(trustStrip(chart.backtest));
  card.append(dataTable(chart));
  return card;
}

function render(plot, tip, chart) {
  const W = Math.max(280, plot.clientWidth);
  plot.querySelector("svg")?.remove();

  const hist = chart.history, fc = chart.forecast;
  const nH = hist.values.length, nF = fc.median.length, n = nH + nF;
  const dates = [...hist.dates, ...fc.dates];

  let lo = Math.min(...hist.values, ...fc.p10), hi = Math.max(...hist.values, ...fc.p90);
  if (lo >= 0 && lo < hi * 0.35) lo = 0;
  const ticks = niceTicks(lo, hi);
  lo = Math.min(lo, ticks[0]);
  hi = Math.max(hi, ticks[ticks.length - 1]);

  const x = (i) => M.left + (i / (n - 1)) * (W - M.left - M.right);
  const y = (v) => M.top + (1 - (v - lo) / (hi - lo)) * (H - M.top - M.bottom);
  const path = (pts) => pts.map(([i, v], k) => `${k ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");

  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img" });
  root.append(Object.assign(svg("title"), { textContent: `${chart.label}: history and forecast` }));

  // Grid + y ticks (recessive hairlines)
  for (const t of ticks) {
    root.append(svg("line", { x1: M.left, x2: W - M.right, y1: y(t), y2: y(t), stroke: "var(--grid)", "stroke-width": 1 }));
    const label = svg("text", { x: M.left - 8, y: y(t) + 3.5, "text-anchor": "end" });
    label.textContent = compact(t);
    root.append(label);
  }

  // X ticks: month starts, thinned so labels never collide
  const monthStarts = dates.map((d, i) => [d, i]).filter(([d]) => d.endsWith("-01"));
  const every = Math.ceil(monthStarts.length / Math.max(2, Math.floor((W - M.left - M.right) / 70)));
  monthStarts.filter((_, k) => k % every === 0).forEach(([d, i]) => {
    const [yy, mm] = d.split("-").map(Number);
    const label = svg("text", { x: x(i), y: H - 8, "text-anchor": "middle" });
    label.textContent = mm === 1 ? `${MONTHS[0]} ${yy}` : MONTHS[mm - 1];
    root.append(label);
  });

  // Fan: both bands open from the last observed point
  const last = nH - 1, anchor = hist.values[last];
  const band = (lower, upper, opacity) => {
    const top = [[last, anchor], ...upper.map((v, k) => [nH + k, v])];
    const bottom = [[last, anchor], ...lower.map((v, k) => [nH + k, v])].reverse();
    root.append(svg("path", { d: path(top) + path(bottom).replace("M", "L") + "Z", fill: "var(--forecast)", opacity }));
  };
  band(fc.p10, fc.p90, 0.12);
  band(fc.p30, fc.p70, 0.18);

  // "Now" divider
  root.append(svg("line", { x1: x(last), x2: x(last), y1: M.top - 6, y2: H - M.bottom, stroke: "var(--axis)", "stroke-width": 1 }));
  const now = svg("text", { x: x(last) + 6, y: M.top - 8, class: "now" });
  now.textContent = "forecast →";
  root.append(now);

  const line = (pts, color) =>
    svg("path", { d: path(pts), fill: "none", stroke: color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" });
  root.append(line(hist.values.map((v, i) => [i, v]), "var(--history)"));
  root.append(line([[last, anchor], ...fc.median.map((v, k) => [nH + k, v])], "var(--forecast)"));

  // End dot + the one direct label: where the median lands
  const endV = fc.median[nF - 1];
  root.append(svg("circle", { cx: x(n - 1), cy: y(endV), r: 4, fill: "var(--forecast)", stroke: "var(--surface)", "stroke-width": 2 }));
  const endLabel = svg("text", { x: x(n - 1) + 9, y: y(endV) + 4, class: "end" });
  endLabel.textContent = compact(endV);
  root.append(endLabel);

  // Hover layer: crosshair snaps to the nearest day
  const cross = svg("line", { y1: M.top, y2: H - M.bottom, stroke: "var(--muted)", "stroke-width": 1, visibility: "hidden" });
  const dot = svg("circle", { r: 4, stroke: "var(--surface)", "stroke-width": 2, visibility: "hidden" });
  const hit = svg("rect", { x: M.left, y: 0, width: W - M.left - M.right, height: H, fill: "transparent" });
  root.append(cross, dot, hit);

  const show = (clientX) => {
    const box = root.getBoundingClientRect();
    const px = ((clientX - box.left) / box.width) * W;
    const i = Math.max(0, Math.min(n - 1, Math.round(((px - M.left) / (W - M.left - M.right)) * (n - 1))));
    const isF = i >= nH, k = i - nH;
    const v = isF ? fc.median[k] : hist.values[i];

    cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible");
    dot.setAttribute("cx", x(i)); dot.setAttribute("cy", y(v));
    dot.setAttribute("fill", isF ? "var(--forecast)" : "var(--history)"); dot.setAttribute("visibility", "visible");

    tip.replaceChildren(el("div", "d", day(dates[i]) + (isF ? " · forecast" : "")));
    const value = el("div", "v");
    value.append(el("i", isF ? "f" : ""), full(v));
    tip.append(value);
    if (isF) {
      tip.append(el("div", "r", `likely ${compact(fc.p30[k])} – ${compact(fc.p70[k])}`));
      tip.append(el("div", "r", `wide ${compact(fc.p10[k])} – ${compact(fc.p90[k])}`));
    }
    tip.hidden = false;
    const left = (x(i) / W) * box.width;
    const flip = left + tip.offsetWidth + 16 > box.width;
    tip.style.left = `${flip ? left - tip.offsetWidth - 12 : left + 12}px`;
  };
  const hide = () => {
    tip.hidden = true;
    cross.setAttribute("visibility", "hidden");
    dot.setAttribute("visibility", "hidden");
  };
  hit.addEventListener("pointermove", (e) => show(e.clientX));
  hit.addEventListener("pointerdown", (e) => show(e.clientX));
  hit.addEventListener("pointerleave", hide);

  plot.append(root);
}

function trustStrip(bt) {
  const strip = el("dl", "trust");
  const cell = (term, main, small) => {
    const box = el("div");
    const dd = el("dd");
    dd.append(main);
    if (small) dd.append(el("small", "", small));
    box.append(el("dt", "", term), dd);
    strip.append(box);
  };

  const beat = bt.beats_naive_baseline;
  const flag = el("span", "flag");
  flag.append(statusIcon(beat), beat ? "Beat the baseline" : "Lost to the baseline");
  cell("Backtest on the latest period", flag, "vs. simply repeating last week");

  cell(
    "Average miss",
    `${bt.model_error_pct_of_typical_value}%`,
    `baseline missed by ${bt.naive_repeat_last_week_error_pct}%`,
  );
  cell(
    "Wide range held",
    `${bt.p10_p90_band_coverage_pct}% of days`,
    `a calibrated band holds ~${bt.band_coverage_target_pct}%`,
  );
  return strip;
}

function statusIcon(good) {
  const icon = svg("svg", { viewBox: "0 0 14 14", width: 14, height: 14, "aria-hidden": "true" });
  icon.append(svg("circle", { cx: 7, cy: 7, r: 7, fill: good ? "var(--good)" : "var(--serious)" }));
  icon.append(svg("path", {
    d: good ? "M4 7.2 6.2 9.4 10 5" : "M4.5 4.5l5 5M9.5 4.5l-5 5",
    fill: "none", stroke: "#fff", "stroke-width": 1.6, "stroke-linecap": "round", "stroke-linejoin": "round",
  }));
  return icon;
}

function dataTable(chart) {
  const details = el("details", "data");
  details.append(el("summary", "", "View forecast as a table"));
  const table = el("table");
  const head = el("tr");
  for (const h of ["Date", "Median", "p10", "p30", "p70", "p90"]) head.append(el("th", "", h));
  table.append(head);
  chart.forecast.dates.forEach((d, k) => {
    const row = el("tr");
    row.append(el("td", "", day(d)));
    for (const q of ["median", "p10", "p30", "p70", "p90"]) row.append(el("td", "", full(chart.forecast[q][k])));
    table.append(row);
  });
  const scroll = el("div", "scroll");
  scroll.append(table);
  details.append(scroll);
  return details;
}
