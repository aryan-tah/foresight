import { forecastCard } from "./chart.js";

const thread = document.getElementById("thread");
const intro = document.getElementById("intro");
const form = document.getElementById("composer");
const input = document.getElementById("input");
const send = document.getElementById("send");
const statusEl = document.getElementById("status");

const sessionId = crypto.randomUUID();
let busy = false;

const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

// What each tool call looks like in the agent's working trace.
const describe = (name, a = {}) =>
  ({
    search_wikipedia: `Looking up “${a.query}” on Wikipedia`,
    get_wikipedia_pageviews: `Fetching Wikipedia pageviews · ${String(a.article ?? "").replaceAll("_", " ")}`,
    get_package_downloads: `Fetching ${a.registry} downloads · ${a.package}`,
    get_crypto_prices: `Fetching ${a.coin_id} prices`,
    get_weather_history: `Fetching ${a.variable} history · ${a.city}`,
    forecast: `Forecasting ${a.horizon_days} days with TimesFM 3.0 · ${a.series_id}`,
  })[name] ?? name;

async function pollHealth() {
  try {
    const h = await (await fetch("/api/health")).json();
    const label = statusEl.querySelector("span");
    if (!h.llm_configured) {
      statusEl.className = "status down";
      label.textContent = "No Anthropic API key";
    } else if (h.model_ready) {
      statusEl.className = "status ready";
      label.textContent = "Claude + TimesFM 3.0 ready";
      return;
    }
  } catch {
    statusEl.className = "status down";
    statusEl.querySelector("span").textContent = "Server unreachable";
  }
  setTimeout(pollHealth, 2000);
}

// Plain text with **bold** only; everything enters the DOM as text nodes.
function appendText(node, text) {
  node.dataset.raw = (node.dataset.raw ?? "") + text;
  node.replaceChildren(
    ...node.dataset.raw.split("**").map((part, i) => (i % 2 ? el("strong", "", part) : document.createTextNode(part))),
  );
}

async function ask(question) {
  if (busy || !question.trim()) return;
  busy = true;
  send.disabled = true;
  intro?.remove();

  const user = el("div", "turn user");
  user.append(el("p", "", question.trim()));
  const agent = el("div", "turn agent");
  const thinking = el("p", "say thinking", "Thinking…");
  agent.append(thinking);
  thread.append(user, agent);
  user.scrollIntoView({ behavior: "smooth", block: "start" });

  let say = null; // current text block
  let trace = null; // current tool list
  const steps = new Map();

  const handle = (ev) => {
    if (ev.type !== "tool_end") thinking.remove();
    if (ev.type === "text") {
      if (!say) agent.append((say = el("p", "say")));
      appendText(say, ev.text);
      trace = null;
    } else if (ev.type === "tool_start") {
      if (!trace) agent.append((trace = el("ul", "trace")));
      const li = el("li", "run");
      li.append(el("span", "tick", "●"), el("span", "", describe(ev.name, ev.input)));
      steps.set(ev.id, li);
      trace.append(li);
      say = null;
    } else if (ev.type === "tool_end") {
      const li = steps.get(ev.id);
      if (!li) return;
      li.className = ev.ok ? "ok" : "bad";
      li.querySelector(".tick").textContent = ev.ok ? "✓" : "✕";
      if (ev.note) li.append(el("span", "note", `— ${ev.note}`));
    } else if (ev.type === "chart") {
      agent.append(forecastCard(ev.chart));
      say = trace = null;
    } else if (ev.type === "error") {
      agent.append(el("p", "say err", ev.message));
    }
  };

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message: question.trim() }),
    });
    if (!res.ok) throw new Error(`Server returned ${res.status}`);

    const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += value;
      const frames = buffer.split("\n\n");
      buffer = frames.pop();
      for (const frame of frames) {
        if (frame.startsWith("data: ")) handle(JSON.parse(frame.slice(6)));
      }
    }
  } catch (e) {
    handle({ type: "error", message: `Something went wrong: ${e.message}` });
  } finally {
    thinking.remove();
    busy = false;
    send.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = input.value;
  input.value = "";
  input.style.height = "auto";
  ask(q);
});
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${input.scrollHeight}px`;
});
document.getElementById("examples").addEventListener("click", (e) => {
  if (e.target.tagName === "BUTTON") ask(e.target.textContent);
});

pollHealth();
input.focus();

// Shareable links: /?q=Will+it+rain asks the question on load.
const shared = new URLSearchParams(location.search).get("q");
if (shared) ask(shared);
