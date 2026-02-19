const filterSymbolEl = document.getElementById("filterSymbol");
const filterStatusEl = document.getElementById("filterStatus");
const listEl = document.getElementById("list");
const coinStatsEl = document.getElementById("coinStats");
const prevBtnEl = document.getElementById("prevBtn");
const nextBtnEl = document.getElementById("nextBtn");
const pageLabelEl = document.getElementById("pageLabel");
const POPUP_LIST_CACHE_KEY = "sim_popup_list_cache_v1";
const POPUP_STATS_CACHE_KEY = "sim_popup_stats_cache_v1";

const SYMBOLS = ["ALL", "BTCUSDT", "ETHUSDT", "XRPUSDT", "DOGEUSDT", "SUIUSDT", "SOLUSDT", "CROSSUSDT"];
const STATUSES = [
  { v: "ALL", t: "전체 상태" },
  { v: "UNFILLED", t: "미체결" },
  { v: "IN_PROGRESS", t: "거래중" },
  { v: "TP", t: "익절" },
  { v: "SL", t: "손절" },
  { v: "FAIL", t: "예측실패" },
];

let page = 1;
let hasNext = false;
let activeLoadSeq = 0;

function fmtPrice(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "-";
  if (n >= 1000) return n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  if (n >= 1) return n.toLocaleString("en-US", { minimumFractionDigits: 3, maximumFractionDigits: 3 });
  return n.toLocaleString("en-US", { minimumFractionDigits: 6, maximumFractionDigits: 6 });
}

function fmtTs(ms) {
  const n = Number(ms);
  if (!Number.isFinite(n) || n <= 0) return "-";
  return new Date(n).toLocaleString("ko-KR", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function statusMeta(status) {
  const s = String(status || "").toUpperCase();
  if (s === "TP") return { cls: "b-tp", text: "익절" };
  if (s === "SL") return { cls: "b-sl", text: "손절" };
  if (s === "FAIL") return { cls: "b-fail", text: "예측실패" };
  if (s === "UNFILLED") return { cls: "b-unfilled", text: "미체결" };
  return { cls: "b-open", text: "거래중" };
}

async function fetchJSON(url, options) {
  const r = await fetch(url, options);
  const text = await r.text();
  if (!r.ok) throw new Error(text || `HTTP ${r.status}`);
  return text ? JSON.parse(text) : null;
}

function render(trades) {
  if (!Array.isArray(trades) || trades.length === 0) {
    listEl.innerHTML = `<div class="item">표시할 기록이 없습니다.</div>`;
    return;
  }
  const rows = trades
    .slice()
    .sort((a, b) => Number(b?.created_ms || 0) - Number(a?.created_ms || 0));
  listEl.innerHTML = rows
    .map((t) => {
      const m = statusMeta(t.status);
      return `<div class="item">
        <div class="row">
          <div class="symbol">${t.symbol}</div>
          <div class="head-right">
            <span class="created">등록 ${fmtTs(t.created_ms)}</span>
            <span class="badge ${m.cls}">${m.text}</span>
          </div>
        </div>
        <div class="prices">
          <div class="p"><span>진입</span>${fmtPrice(t.entry_price)}</div>
          <div class="p"><span>익절</span>${fmtPrice(t.take_profit)}</div>
          <div class="p"><span>손절</span>${fmtPrice(t.stop_loss)}</div>
        </div>
      </div>`;
    })
    .join("");
}

function renderStats(items) {
  if (!coinStatsEl) return;
  if (!Array.isArray(items) || items.length === 0) {
    coinStatsEl.innerHTML = "";
    return;
  }
  coinStatsEl.innerHTML = items
    .map((s) => {
      const total = Number(s.total || 0);
      const tp = Number(s.tp || 0);
      const sl = Number(s.sl || 0);
      const fail = Number(s.fail || 0);
      const unfilled = Number(s.unfilled || 0);
      const inProgress = Number(s.in_progress || 0);
      const winRate = Number(s.win_rate || 0);
      return `<div class="stat-card">
        <div class="stat-top">
          <div class="stat-symbol">${s.symbol}</div>
          <div class="stat-total">총 ${total}건</div>
        </div>
        <div class="stat-row">익절 ${tp} · 손절 ${sl} · 예측실패 ${fail}</div>
        <div class="stat-row">미체결 ${unfilled} · 거래중 ${inProgress}</div>
        <div class="stat-win">익절률 ${winRate.toFixed(1)}%</div>
      </div>`;
    })
    .join("");
}

async function loadStats() {
  const data = await fetchJSON("/api/sim/stats?sync=0");
  const stats = data?.stats || [];
  renderStats(stats);
  try {
    localStorage.setItem(POPUP_STATS_CACHE_KEY, JSON.stringify({ stats, ts: Date.now() }));
  } catch (_) {}
}

async function load() {
  const seq = ++activeLoadSeq;
  const symbol = filterSymbolEl.value;
  const status = filterStatusEl.value;
  const q = new URLSearchParams({ limit: "20", page: String(page) });
  if (symbol && symbol !== "ALL") q.set("symbol", symbol);
  if (status && status !== "ALL") q.set("status", status);
  q.set("sync", "0");
  const data = await fetchJSON(`/api/sim/trades?${q.toString()}`);
  if (seq !== activeLoadSeq) return;
  const trades = data?.trades || [];
  render(trades);
  hasNext = Boolean(data?.has_next);
  pageLabelEl.textContent = `${page} 페이지`;
  prevBtnEl.disabled = page <= 1;
  nextBtnEl.disabled = !hasNext;
  try {
    if (page === 1 && (!symbol || symbol === "ALL") && (!status || status === "ALL")) {
      localStorage.setItem(
        POPUP_LIST_CACHE_KEY,
        JSON.stringify({
          trades,
          has_next: hasNext,
          ts: Date.now(),
        })
      );
    }
  } catch (_) {}
}

function initFilters() {
  filterSymbolEl.innerHTML = SYMBOLS.map((s) => `<option value="${s}">${s === "ALL" ? "전체 코인" : s}</option>`).join("");
  filterStatusEl.innerHTML = STATUSES.map((s) => `<option value="${s.v}">${s.t}</option>`).join("");
}

function onFilterChange() {
  page = 1;
  load().catch((e) => alert(e.message || e));
}

function hydrateFromCache() {
  try {
    const rawList = localStorage.getItem(POPUP_LIST_CACHE_KEY);
    if (rawList) {
      const parsed = JSON.parse(rawList);
      if (Array.isArray(parsed?.trades)) {
        render(parsed.trades);
        hasNext = Boolean(parsed?.has_next);
        pageLabelEl.textContent = "1 페이지";
        prevBtnEl.disabled = true;
        nextBtnEl.disabled = !hasNext;
      }
    }
  } catch (_) {}
  try {
    const rawStats = localStorage.getItem(POPUP_STATS_CACHE_KEY);
    if (rawStats) {
      const parsed = JSON.parse(rawStats);
      if (Array.isArray(parsed?.stats)) renderStats(parsed.stats);
    }
  } catch (_) {}
}
filterSymbolEl.addEventListener("change", onFilterChange);
filterStatusEl.addEventListener("change", onFilterChange);
prevBtnEl.addEventListener("click", () => {
  if (page <= 1) return;
  page -= 1;
  load().catch((e) => alert(e.message || e));
});
nextBtnEl.addEventListener("click", () => {
  if (!hasNext) return;
  page += 1;
  load().catch((e) => alert(e.message || e));
});

initFilters();
hydrateFromCache();
listEl.innerHTML = listEl.innerHTML || `<div class="item">불러오는 중...</div>`;
load()
  .then(() => loadStats().catch(() => {}))
  .catch((e) => alert(e.message || e));
