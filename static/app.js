const analysisIntervalEl = document.getElementById("analysisInterval");
const chartIntervalEl = document.getElementById("chartInterval");
const analysisModeEl = document.getElementById("analysisMode");
const refreshEl = document.getElementById("refresh");
const headerAutoBtnEl = document.getElementById("headerAutoBtn");
const fibToggleEl = document.getElementById("fibToggle");
const avwapToggleEl = document.getElementById("avwapToggle");
const vpToggleEl = document.getElementById("vpToggle");
const volumeToggleEl = document.getElementById("volumeToggle");
const coinListEl = document.getElementById("coinList");
const newsSummaryEl = document.getElementById("newsSummary");
const newsListEl = document.getElementById("newsList");
const newsRefreshEl = document.getElementById("newsRefresh");
const probAssetEl = document.getElementById("probAsset");
const simSymbolEl = document.getElementById("simSymbol");
const simUseCurrentEl = document.getElementById("simUseCurrent");
const simEntryEl = document.getElementById("simEntry");
const simTpEl = document.getElementById("simTp");
const simSlEl = document.getElementById("simSl");
const simAddBtnEl = document.getElementById("simAddBtn");
const simToggleBtnEl = document.getElementById("simToggleBtn");
const simFormEl = document.getElementById("simForm");
const simRuleNoteEl = document.getElementById("simRuleNote");
const simStatusEl = document.getElementById("simStatus");
const simListEl = document.getElementById("simList");
const simOpenAllBtnEl = document.getElementById("simOpenAllBtn");
const autoEnabledEl = document.getElementById("autoEnabled");
const autoModeEl = document.getElementById("autoMode");
const autoIntervalEl = document.getElementById("autoInterval");
const autoOrderSizeEl = document.getElementById("autoOrderSize");
const autoDailyLossEl = document.getElementById("autoDailyLoss");
const autoTpPctEl = document.getElementById("autoTpPct");
const autoSlPctEl = document.getElementById("autoSlPct");
const autoCooldownMinEl = document.getElementById("autoCooldownMin");
const autoMaxOpenEl = document.getElementById("autoMaxOpen");
const autoSaveBtnEl = document.getElementById("autoSaveBtn");
const autoRunBtnEl = document.getElementById("autoRunBtn");
const autoOpenAllBtnEl = document.getElementById("autoOpenAllBtn");
const autoStatusEl = document.getElementById("autoStatus");
const autoListEl = document.getElementById("autoList");
let toastWrapEl = null;
const SIM_TRADES_CACHE_KEY = "sim_trades_cache_v1";
const AUTO_TRADES_CACHE_KEY = "auto_trades_cache_v1";
let latestVolumeStates = {};
let latestDecisionKlines = [];

const buyPctEl = document.getElementById("buyPct");
const sellPctEl = document.getElementById("sellPct");
const confEl = document.getElementById("confidence");
const regimeEl = document.getElementById("regime");
const positionStateEl = document.getElementById("positionState");
const whaleStateEl = document.getElementById("whaleState");
const volumeStateEl = document.getElementById("volumeState");
const asofEl = document.getElementById("asof");
const explainDetailEl = document.getElementById("explainDetail");
const actionBadgeEl = document.getElementById("actionBadge");
const actionTitleEl = document.getElementById("actionTitle");
const actionSubtitleEl = document.getElementById("actionSubtitle");
const actionExplainEl = document.getElementById("actionExplain");
const actionReadGuideEl = document.getElementById("actionReadGuide");
const entryZoneEl = document.getElementById("entryZone");
const targetZoneEl = document.getElementById("targetZone");
const stopZoneEl = document.getElementById("stopZone");
const entryPlanMetricsEl = document.getElementById("entryPlanMetrics");
const entryLabelEl = document.getElementById("entryLabel");
const targetLabelEl = document.getElementById("targetLabel");
const stopLabelEl = document.getElementById("stopLabel");
const targetDescEl = document.getElementById("targetDesc");
const entryTableSwingEl = document.getElementById("entryTableSwing");
const entryTableScenarioEl = document.getElementById("entryTableScenario");
const entryFibLevelsEl = document.getElementById("entryFibLevels");
const entryFibZonesEl = document.getElementById("entryFibZones");
const stepProbStatusEl = document.getElementById("stepProbStatus");
const stepProbDetailEl = document.getElementById("stepProbDetail");
const stepFibStatusEl = document.getElementById("stepFibStatus");
const stepFibDetailEl = document.getElementById("stepFibDetail");
const stepExecStatusEl = document.getElementById("stepExecStatus");
const stepExecDetailEl = document.getElementById("stepExecDetail");
const stepFinalStatusEl = document.getElementById("stepFinalStatus");
const decisionFinalEl = document.getElementById("decisionFinal");
const passCheckSummaryEl = document.getElementById("passCheckSummary");
const passCheckPeriodEl = document.getElementById("passCheckPeriod");
const passCheckModeNoteEl = document.getElementById("passCheckModeNote");
const passCheckBarsEl = document.getElementById("passCheckBars");
const passCheckPassesEl = document.getElementById("passCheckPasses");
const passCheckHitRateEl = document.getElementById("passCheckHitRate");
const entrySwingBadgeEl = document.getElementById("entrySwingBadge");
const coinSideStickyEl = document.querySelector(".coin-side-sticky");
const coinSidePanelEl = document.querySelector(".coin-side");
const mainLayoutEl = document.querySelector(".main");
let sidebarPinRaf = 0;
let sidebarPinObserver = null;
let sidebarScrollHost = null;
let sidebarPinTimer = null;
let suppressCoinClickUntil = 0;
let coinSwitchTimer = null;
let coinSwitchSeq = 0;
let coinListDirty = false;
let coinListDeferredTimer = null;
let lastScrollActivityTs = 0;
let lastTouchPointerTs = 0;
let isCoinListScrolling = false;
let coinListNodes = new Map();
const COIN_LIST_SCROLL_DEFER_MS = 380;

function clearSidebarPin() {
  if (!coinSideStickyEl) return;
  coinSideStickyEl.style.transform = "";
  coinSideStickyEl.style.height = "";
  if (!coinSidePanelEl) return;
  coinSidePanelEl.style.position = "";
  coinSidePanelEl.style.left = "";
  coinSidePanelEl.style.top = "";
  coinSidePanelEl.style.width = "";
  coinSidePanelEl.style.maxHeight = "";
  coinSidePanelEl.style.height = "";
  coinSidePanelEl.style.overflowY = "";
  coinSidePanelEl.style.overflowX = "";
}

function flushDeferredCoinListRender() {
  if (!coinListDirty) return;
  coinListDirty = false;
  renderCoinList();
}

function noteCoinListScrollActivity() {
  if (window.innerWidth > 1100) return;
  lastScrollActivityTs = Date.now();
  isCoinListScrolling = true;
  if (coinListEl) coinListEl.classList.add("is-scrolling");
  if (coinListDeferredTimer) clearTimeout(coinListDeferredTimer);
  coinListDeferredTimer = setTimeout(() => {
    coinListDeferredTimer = null;
    isCoinListScrolling = false;
    if (coinListEl) coinListEl.classList.remove("is-scrolling");
    flushDeferredCoinListRender();
  }, COIN_LIST_SCROLL_DEFER_MS);
}

function requestCoinListRender(force = false) {
  if (force) {
    coinListDirty = false;
    renderCoinList();
    return;
  }
  const defer = window.innerWidth <= 1100 && Date.now() - lastScrollActivityTs < COIN_LIST_SCROLL_DEFER_MS;
  if (defer) {
    coinListDirty = true;
    if (!coinListDeferredTimer) {
      coinListDeferredTimer = setTimeout(() => {
        coinListDeferredTimer = null;
        flushDeferredCoinListRender();
      }, COIN_LIST_SCROLL_DEFER_MS);
    }
    return;
  }
  renderCoinList();
}

function initCoinListInteractionGuard() {
  const onActivity = () => noteCoinListScrollActivity();
  window.addEventListener("scroll", onActivity, { passive: true });
  window.addEventListener("touchmove", onActivity, { passive: true });
  if (coinListEl) {
    coinListEl.addEventListener("touchmove", onActivity, { passive: true });
    coinListEl.addEventListener("wheel", onActivity, { passive: true });
  }
  const coinSideEl = document.querySelector(".coin-side");
  if (coinSideEl) {
    coinSideEl.addEventListener("scroll", onActivity, { passive: true });
    coinSideEl.addEventListener("touchmove", onActivity, { passive: true });
  }
}

function isLikelyMobileBrowser() {
  const ua = String(navigator?.userAgent || "");
  return /Android|iPhone|iPad|iPod|Mobile|Windows Phone/i.test(ua);
}

function updateDesktopModeClass() {
  const desktop = !isLikelyMobileBrowser() && window.innerWidth > 1100;
  document.body.classList.toggle("desktop-mode", desktop);
  scheduleSidebarPinUpdate();
}

function findScrollHost(el) {
  let p = el ? el.parentElement : null;
  while (p && p !== document.body && p !== document.documentElement) {
    const cs = window.getComputedStyle(p);
    const oy = String(cs.overflowY || "");
    if ((oy.includes("auto") || oy.includes("scroll") || oy.includes("overlay")) && p.scrollHeight > p.clientHeight + 1) {
      return p;
    }
    p = p.parentElement;
  }
  return window;
}

function getScrollTopAndRect(host) {
  if (!host || host === window) {
    return {
      scrollTop: window.scrollY || window.pageYOffset || 0,
      hostTop: 0,
    };
  }
  const r = host.getBoundingClientRect();
  return {
    scrollTop: host.scrollTop || 0,
    hostTop: r.top || 0,
  };
}

function updateSidebarPinNow() {
  if (!coinSideStickyEl || !coinSidePanelEl || !mainLayoutEl) return;
  const desktopMode = document.body.classList.contains("desktop-mode");
  if (!desktopMode || window.innerWidth <= 900) {
    clearSidebarPin();
    return;
  }
  const mainRect = mainLayoutEl.getBoundingClientRect();
  const wrapRect = coinSideStickyEl.getBoundingClientRect();
  if (mainRect.width <= 0 || wrapRect.width <= 0 || mainLayoutEl.offsetHeight <= 0) return;

  const topGap = 12;
  const bottomGap = 12;
  const blockHeight = Math.max(240, Math.round(window.innerHeight - topGap - bottomGap));
  const mainTopDoc = window.scrollY + mainRect.top;
  const mainBottomDoc = mainTopDoc + mainLayoutEl.offsetHeight;

  let topPx = topGap;
  if (window.scrollY + topGap < mainTopDoc) topPx = mainTopDoc - window.scrollY;
  const maxTopPx = mainBottomDoc - window.scrollY - blockHeight - bottomGap;
  if (Number.isFinite(maxTopPx)) topPx = Math.min(topPx, maxTopPx);
  topPx = Math.max(0, topPx);

  coinSideStickyEl.style.height = `${blockHeight}px`;
  coinSidePanelEl.style.position = "fixed";
  coinSidePanelEl.style.left = `${Math.round(mainRect.left)}px`;
  coinSidePanelEl.style.top = `${Math.round(topPx)}px`;
  coinSidePanelEl.style.width = `${Math.round(wrapRect.width)}px`;
  coinSidePanelEl.style.maxHeight = `${blockHeight}px`;
  coinSidePanelEl.style.height = `${blockHeight}px`;
  coinSidePanelEl.style.overflowY = "auto";
  coinSidePanelEl.style.overflowX = "hidden";
}

function scheduleSidebarPinUpdate() {
  if (sidebarPinRaf) cancelAnimationFrame(sidebarPinRaf);
  sidebarPinRaf = requestAnimationFrame(() => {
    sidebarPinRaf = 0;
    updateSidebarPinNow();
  });
}

function initSidebarPinFallback() {
  if (!coinSideStickyEl || !mainLayoutEl) return;
  sidebarScrollHost = findScrollHost(mainLayoutEl);
  window.addEventListener("resize", scheduleSidebarPinUpdate, { passive: true });
  window.addEventListener("scroll", scheduleSidebarPinUpdate, { passive: true });
  document.addEventListener("scroll", scheduleSidebarPinUpdate, true);
  if (sidebarScrollHost && sidebarScrollHost !== window) {
    sidebarScrollHost.addEventListener("scroll", scheduleSidebarPinUpdate, { passive: true });
  }
  if ("ResizeObserver" in window) {
    sidebarPinObserver = new ResizeObserver(() => scheduleSidebarPinUpdate());
    sidebarPinObserver.observe(mainLayoutEl);
    sidebarPinObserver.observe(coinSideStickyEl);
  }
  scheduleSidebarPinUpdate();
}

function setStepStatus(el, ok) {
  if (!el) return;
  el.textContent = ok ? "PASS" : "WAIT";
  el.classList.toggle("is-pass", Boolean(ok));
  el.classList.toggle("is-wait", !ok);
}

function setSwingBadge(kind, text) {
  if (!entrySwingBadgeEl) return;
  entrySwingBadgeEl.textContent = text;
  entrySwingBadgeEl.classList.remove("up", "down", "neutral");
  if (kind) entrySwingBadgeEl.classList.add(kind);
}

function setTradeLabels(side = "WAIT") {
  const s = String(side || "").toUpperCase();
  if (entryLabelEl) entryLabelEl.textContent = "진입 기준가";
  if (targetLabelEl) targetLabelEl.textContent = "진입 후 익절가(1차/2차)";
  if (stopLabelEl) stopLabelEl.textContent = "손절가";
  if (targetDescEl) {
    if (s === "SELL") targetDescEl.textContent = "진입가 이후 하락하여 익절을 고려하는 가격";
    else if (s === "BUY") targetDescEl.textContent = "진입가 이후 상승하여 익절을 고려하는 가격";
    else targetDescEl.textContent = "진입가 이후 목표 익절을 고려하는 가격";
  }
}

function resetAnalysisUI(note = "코인 변경: 계산 중...") {
  latestDecisionKlines = [];
  latestVolumeStates = {};
  if (buyPctEl) buyPctEl.textContent = "-";
  if (sellPctEl) sellPctEl.textContent = "-";
  if (confEl) confEl.textContent = "-";
  if (regimeEl) regimeEl.textContent = "-";
  if (positionStateEl) positionStateEl.textContent = "-";
  if (whaleStateEl) whaleStateEl.textContent = "-";
  if (volumeStateEl) volumeStateEl.textContent = "-";
  if (actionBadgeEl) {
    actionBadgeEl.textContent = "WAIT";
    actionBadgeEl.className = "action-badge wait";
  }
  if (actionTitleEl) actionTitleEl.textContent = "분석값 계산 중입니다.";
  if (actionSubtitleEl) actionSubtitleEl.textContent = "잠시 후 최신 시그널이 반영됩니다.";
  if (actionExplainEl) actionExplainEl.textContent = "-";
  if (entryTableSwingEl) entryTableSwingEl.textContent = "-";
  if (entryZoneEl) entryZoneEl.textContent = "-";
  if (targetZoneEl) targetZoneEl.textContent = "-";
  if (stopZoneEl) stopZoneEl.textContent = "-";
  setTradeLabels("WAIT");
  if (entryPlanMetricsEl) entryPlanMetricsEl.textContent = "-";
  if (entryFibLevelsEl) entryFibLevelsEl.textContent = "-";
  if (entryFibZonesEl) entryFibZonesEl.textContent = "-";
  setStepStatus(stepProbStatusEl, false);
  setStepStatus(stepFibStatusEl, false);
  setStepStatus(stepExecStatusEl, false);
  setStepStatus(stepFinalStatusEl, false);
  if (stepProbDetailEl) stepProbDetailEl.textContent = note;
  if (stepFibDetailEl) stepFibDetailEl.textContent = "-";
  if (stepExecDetailEl) stepExecDetailEl.textContent = "-";
  if (decisionFinalEl) decisionFinalEl.textContent = "-";
  if (asofEl) asofEl.textContent = note;
}

function setPassCheckUI(state, payload = {}) {
  if (!passCheckSummaryEl) return;
  passCheckSummaryEl.classList.remove("is-pending", "is-good", "is-mid", "is-low", "is-error");
  passCheckSummaryEl.classList.add(
    state === "success"
      ? payload.hitRate >= 55
        ? "is-good"
        : payload.hitRate >= 40
          ? "is-mid"
          : "is-low"
      : state === "error"
        ? "is-error"
        : "is-pending"
  );
  if (passCheckModeNoteEl) passCheckModeNoteEl.textContent = payload.modeNote || "-";
  if (passCheckBarsEl) passCheckBarsEl.textContent = payload.barsText || "-";
  if (passCheckPassesEl) passCheckPassesEl.textContent = payload.passesText || "-";
  if (passCheckHitRateEl) passCheckHitRateEl.textContent = payload.hitRateText || "-";
}

function resetPassCheckUI(note = "코인 변경: 계산 대기") {
  passCheckKey = "";
  passCheckTs = 0;
  if (passCheckTimer) {
    clearTimeout(passCheckTimer);
    passCheckTimer = null;
  }
  if (activePassCheckController) {
    try {
      activePassCheckController.abort();
    } catch (_) {}
    activePassCheckController = null;
  }
  setPassCheckUI("pending", {
    modeNote: note,
    barsText: "-",
    passesText: "-",
    hitRateText: "-",
  });
}

function getPassCheckPeriodConfig(interval, periodValue) {
  const p = String(periodValue || "3d");
  const mapByInterval = {
    "5m": { "24h": 288, "3d": 864, "7d": 2016 },
    "1h": { "24h": 24, "3d": 72, "7d": 168 },
    "4h": { "24h": 6, "3d": 18, "7d": 42 },
  };
  const bars = (mapByInterval[interval] && mapByInterval[interval][p]) || 72;
  // horizon이 길수록 과거 표본 버퍼를 크게 확보해야 0% 고정 현상을 줄일 수 있음
  const sampleBufferByPeriod = { "24h": 700, "3d": 1100, "7d": 1800 };
  const sampleBuffer = sampleBufferByPeriod[p] || 1100;
  return { period: p, horizonBars: bars, sampleBuffer };
}

function loadPassCheckCache() {
  try {
    const raw = localStorage.getItem(PASS_CHECK_CACHE_KEY);
    if (!raw) return {};
    const obj = JSON.parse(raw);
    return obj && typeof obj === "object" ? obj : {};
  } catch (_) {
    return {};
  }
}

function savePassCheckCache(cacheObj) {
  try {
    localStorage.setItem(PASS_CHECK_CACHE_KEY, JSON.stringify(cacheObj || {}));
  } catch (_) {}
}

function trimPassCheckCache(cacheObj, maxKeys = 72) {
  if (!cacheObj || typeof cacheObj !== "object") return {};
  const keys = Object.keys(cacheObj);
  if (keys.length <= maxKeys) return cacheObj;
  const dropCount = keys.length - maxKeys;
  for (let i = 0; i < dropCount; i += 1) {
    delete cacheObj[keys[i]];
  }
  return cacheObj;
}

function renderProbAsset() {
  if (!probAssetEl) return;
  const coin = COINS.find((c) => c.symbol === selectedSymbol && c.market === selectedMarket) || COINS.find((c) => c.symbol === selectedSymbol);
  if (!coin) {
    probAssetEl.textContent = selectedSymbol || "-";
    return;
  }
  probAssetEl.textContent = `${coin.name} (${coin.symbol})`;
}

const chartContainer = document.getElementById("chart");
let chart = null;
let candleSeries = null;
let volumeSeries = null;
let fibLines = [];
let avwapSeriesList = [];
let vpLines = [];
let chartResizeObserver = null;
let selectedSymbol = "BTCUSDT";
let selectedMarket = "spot";
let coinPrices = {};
let coinNews = {};
let isLoadingMain = false;
let isLoadingPrices = false;
let isLoadingNews = false;
let priceRefreshTimer = null;
let mainRefreshTimer = null;
let newsRefreshTimer = null;
let simRefreshTimer = null;
let autoRefreshTimer = null;
let fxRefreshTimer = null;
let spotListWs = null;
let futuresListWs = null;
let spotListReconnectTimer = null;
let futuresListReconnectTimer = null;
const listStreamFailureCount = { spot: 0, futures: 0 };
const listStreamDisabled = { spot: false, futures: false };
let chartWs = null;
let chartWsReconnectTimer = null;
let chartStreamKey = "";
let lastFitKey = "";
let analysisWs = null;
let analysisWsReconnectTimer = null;
let analysisStreamKey = "";
let usdtKrw = 1350;
let coinRenderTimer = null;
let mtfAnalysisCache = null;
let mtfAnalysisCacheKey = "";
let mtfFibPlanCache = null;
let mtfFibPlanCacheKey = "";
let lastFibPlan = null;
let lastOverlayFibPlan = null;
let lastSingleAnalysisInterval = "5m";
let analysisSnapshotRaw = null;
let analysisSnapshotKey = "";
let passCheckTs = 0;
let passCheckKey = "";
let activeLoadController = null;
let activeLoadSeq = 0;
let activePassCheckController = null;
let activePassCheckSeq = 0;
let passCheckTimer = null;
let autoTickBusy = false;
let autoConfigReady = false;

const PRICE_REFRESH_MS = 20000; // websocket 장애 대비 fallback
const FX_REFRESH_MS = 60000;
const NEWS_REFRESH_MS = 300000;
const SIM_REFRESH_MS = 20000;
const AUTO_REFRESH_MS = 25000;

const COINS = [
  { name: "비트코인", symbol: "BTCUSDT", market: "spot", mark: "₿" },
  { name: "이더리움", symbol: "ETHUSDT", market: "spot", mark: "◆" },
  { name: "리플", symbol: "XRPUSDT", market: "spot", mark: "X" },
  { name: "도지", symbol: "DOGEUSDT", market: "spot", mark: "Ð" },
  { name: "수이", symbol: "SUIUSDT", market: "spot", mark: "S" },
  { name: "솔라나", symbol: "SOLUSDT", market: "spot", mark: "◎" },
  { name: "크로쓰", symbol: "CROSSUSDT", market: "futures", mark: "C" },
];

const UI_STATE_KEY = "coin.ui.state.v1";
const PASS_CHECK_CACHE_KEY = "coin.passcheck.cache.v1";
const ALLOWED_ANALYSIS_INTERVALS = new Set(["5m", "1h", "4h"]);

function normalizeAnalysisInterval(v) {
  const s = String(v || "").trim();
  return ALLOWED_ANALYSIS_INTERVALS.has(s) ? s : "5m";
}

function saveUiState() {
  try {
    const analysisMode = analysisModeEl ? analysisModeEl.value : "single";
    const analysisInterval = analysisIntervalEl ? analysisIntervalEl.value : "5m";
    const state = {
      selectedSymbol,
      selectedMarket,
      analysisMode,
      analysisInterval: analysisMode === "mtf" ? "" : normalizeAnalysisInterval(analysisInterval),
      chartInterval: chartIntervalEl ? chartIntervalEl.value : "5m",
      passCheckPeriod: passCheckPeriodEl ? passCheckPeriodEl.value : "3d",
      fibToggle: fibToggleEl ? Boolean(fibToggleEl.checked) : true,
      avwapToggle: avwapToggleEl ? Boolean(avwapToggleEl.checked) : true,
      vpToggle: vpToggleEl ? Boolean(vpToggleEl.checked) : true,
      volumeToggle: volumeToggleEl ? Boolean(volumeToggleEl.checked) : true,
    };
    localStorage.setItem(UI_STATE_KEY, JSON.stringify(state));
  } catch (_) {}
}

function hasOption(el, value) {
  if (!el) return false;
  return Array.from(el.options || []).some((opt) => opt.value === value);
}

function restoreUiState() {
  try {
    const raw = localStorage.getItem(UI_STATE_KEY);
    if (!raw) return;
    const state = JSON.parse(raw);

    const symbol = String(state?.selectedSymbol || "").toUpperCase();
    const market = String(state?.selectedMarket || "").toLowerCase();
    if (COINS.some((c) => c.symbol === symbol && c.market === market)) {
      selectedSymbol = symbol;
      selectedMarket = market;
    }

    if (analysisModeEl && hasOption(analysisModeEl, state?.analysisMode)) {
      analysisModeEl.value = String(state.analysisMode);
    }
    if (analysisIntervalEl && hasOption(analysisIntervalEl, state?.analysisInterval)) {
      analysisIntervalEl.value = String(state.analysisInterval);
      if (state.analysisInterval) lastSingleAnalysisInterval = normalizeAnalysisInterval(state.analysisInterval);
    }
    if (chartIntervalEl && hasOption(chartIntervalEl, state?.chartInterval)) {
      chartIntervalEl.value = String(state.chartInterval);
    }
    if (passCheckPeriodEl && hasOption(passCheckPeriodEl, state?.passCheckPeriod)) {
      passCheckPeriodEl.value = String(state.passCheckPeriod);
    }
    if (fibToggleEl && typeof state?.fibToggle === "boolean") fibToggleEl.checked = state.fibToggle;
    if (avwapToggleEl && typeof state?.avwapToggle === "boolean") avwapToggleEl.checked = state.avwapToggle;
    if (vpToggleEl && typeof state?.vpToggle === "boolean") vpToggleEl.checked = state.vpToggle;
    if (volumeToggleEl && typeof state?.volumeToggle === "boolean") volumeToggleEl.checked = state.volumeToggle;
  } catch (_) {}
}

function setSimStatus(msg) {
  if (simStatusEl) simStatusEl.textContent = msg || "";
}

function simSymbolKey(symbol, market) {
  return String(symbol || "").toUpperCase();
}

function simParseSymbolKey(v) {
  return { symbol: String(v || "").toUpperCase(), market: "spot" };
}

function populateSimSymbolOptions() {
  if (!simSymbolEl) return;
  const allCoins = COINS;
  simSymbolEl.innerHTML = allCoins
    .map((c) => `<option value="${simSymbolKey(c.symbol)}">${c.name} (${c.symbol})${c.market === "futures" ? " · 선물" : ""}</option>`)
    .join("");
  const cur = simSymbolKey(selectedSymbol);
  const has = allCoins.some((c) => simSymbolKey(c.symbol) === cur);
  simSymbolEl.value = has ? cur : simSymbolKey(allCoins[0]?.symbol || "BTCUSDT");
}

function syncSimSymbolWithSelected() {
  if (!simSymbolEl) return;
  const key = simSymbolKey(selectedSymbol);
  const has = Array.from(simSymbolEl.options || []).some((o) => o.value === key);
  if (has) simSymbolEl.value = key;
}

function fillSimEntryWithCurrentPrice() {
  if (!simUseCurrentEl || !simUseCurrentEl.checked || !simEntryEl) return;
  const { symbol } = simParseSymbolKey(simSymbolEl ? simSymbolEl.value : simSymbolKey(selectedSymbol, selectedMarket));
  const p = Number(coinPrices[symbol]);
  if (Number.isFinite(p) && p > 0) simEntryEl.value = String(p);
}

function clampNonNegativeInput(el) {
  if (!el) return;
  const v = Number(el.value);
  if (Number.isFinite(v) && v < 0) el.value = "0";
}

function setSimRuleNote(visible) {
  if (!simRuleNoteEl) return;
  simRuleNoteEl.hidden = !visible;
  simRuleNoteEl.classList.toggle("show", Boolean(visible));
}

function showToast(msg) {
  if (!msg) return;
  if (!toastWrapEl) {
    toastWrapEl = document.createElement("div");
    toastWrapEl.className = "toast-wrap";
    document.body.appendChild(toastWrapEl);
  }
  const el = document.createElement("div");
  el.className = "toast";
  el.textContent = String(msg);
  toastWrapEl.appendChild(el);
  requestAnimationFrame(() => el.classList.add("show"));
  setTimeout(() => {
    el.classList.remove("show");
    setTimeout(() => el.remove(), 220);
  }, 1800);
}

function toggleSimForm() {
  if (!simFormEl) return;
  simFormEl.classList.toggle("is-collapsed");
  const collapsed = simFormEl.classList.contains("is-collapsed");
  if (simToggleBtnEl) simToggleBtnEl.textContent = collapsed ? "시뮬레이션 작성" : "작성 닫기";
  scheduleSidebarPinUpdate();
}

function fmtSimTs(ms) {
  const n = Number(ms);
  if (!Number.isFinite(n) || n <= 0) return "-";
  return new Date(n).toLocaleString("ko-KR", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function renderSimTrades(trades) {
  if (!simListEl) return;
  if (!Array.isArray(trades) || trades.length === 0) {
    simListEl.innerHTML = `<div class="sim-item sim-empty">현재 진행중인 시뮬레이션이 없습니다.</div>`;
    scheduleSidebarPinUpdate();
    return;
  }
  simListEl.innerHTML = trades
    .map((t) => {
      const st = String(t?.status || "OPEN").toUpperCase();
      const badgeClass =
        st === "TP" ? "tp" : st === "SL" ? "sl" : st === "FAIL" ? "fail" : st === "UNFILLED" ? "unfilled" : "open";
      const badgeText =
        st === "TP" ? "익절" : st === "SL" ? "손절" : st === "FAIL" ? "예측실패" : st === "UNFILLED" ? "미체결" : "거래중";
      const created = fmtSimTs(t?.created_ms);
      return `<div class="sim-item">
        <div class="sim-item-head">
          <div class="sim-symbol">${t.symbol}</div>
          <span class="badge ${badgeClass}">${badgeText}</span>
        </div>
        <div class="sim-prices">
          <div class="sim-price"><span>진입</span><strong>${formatPrice(Number(t.entry_price))}</strong></div>
          <div class="sim-price"><span>익절</span><strong>${formatPrice(Number(t.take_profit))}</strong></div>
          <div class="sim-price"><span>손절</span><strong>${formatPrice(Number(t.stop_loss))}</strong></div>
        </div>
        <div class="sim-time">등록 ${created}</div>
      </div>`;
    })
    .join("");
  scheduleSidebarPinUpdate();
}

function loadCachedSimTrades() {
  try {
    const raw = localStorage.getItem(SIM_TRADES_CACHE_KEY);
    if (!raw) return [];
    const arr = JSON.parse(raw);
    return Array.isArray(arr) ? arr : [];
  } catch (_) {
    return [];
  }
}

function saveCachedSimTrades(trades) {
  try {
    const arr = Array.isArray(trades) ? trades : [];
    localStorage.setItem(SIM_TRADES_CACHE_KEY, JSON.stringify(arr.slice(0, 10)));
  } catch (_) {}
}

async function loadSimTrades() {
  try {
    const data = await fetchJSON(`/api/sim/trades?limit=10&page=1&sync=0`);
    const trades = data?.trades || [];
    renderSimTrades(trades);
    saveCachedSimTrades(trades);
    setSimStatus("");
  } catch (e) {
    setSimStatus(`시뮬레이션 조회 실패: ${e.message || e}`);
  }
}

async function simAddTrade() {
  fillSimEntryWithCurrentPrice();
  const entryRaw = String(simEntryEl?.value || "").trim();
  const tpRaw = String(simTpEl?.value || "").trim();
  const slRaw = String(simSlEl?.value || "").trim();
  if (!entryRaw || !tpRaw || !slRaw) {
    setSimStatus("모든 값을 입력해주세요.");
    return;
  }
  const entry = Number(entryRaw);
  const tp = Number(tpRaw);
  const sl = Number(slRaw);
  if (![entry, tp, sl].every(Number.isFinite) || entry <= 0 || tp <= 0 || sl <= 0) {
    setSimStatus("모든 값을 입력해주세요.");
    return;
  }
  if (!(tp > entry && sl < entry)) {
    setSimRuleNote(true);
    setSimStatus("");
    return;
  }
  setSimRuleNote(false);
  const picked = simParseSymbolKey(simSymbolEl ? simSymbolEl.value : simSymbolKey(selectedSymbol, selectedMarket));
  try {
    await fetchJSON("/api/sim/trades", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        symbol: picked.symbol,
        entry,
        take_profit: tp,
        stop_loss: sl,
      }),
    });
    showToast("시뮬레이션 작성이 완료되었습니다.");
    setSimStatus("");
    if (simFormEl) simFormEl.classList.add("is-collapsed");
    if (simToggleBtnEl) simToggleBtnEl.textContent = "시뮬레이션 작성";
    await loadSimTrades();
  } catch (e) {
    setSimStatus(`기록 추가 실패: ${e.message || e}`);
  }
}

function setAutoStatus(msg) {
  if (autoStatusEl) autoStatusEl.textContent = msg || "";
}

function autoFibPlanStatusText(plan) {
  if (!plan || typeof plan !== "object") return "";
  const entry = formatPrice(Number(plan?.entry_price));
  const stop = formatPrice(Number(plan?.stop_price));
  const tp1 = formatPrice(Number(plan?.tp1_price));
  const tp2 = formatPrice(Number(plan?.tp2_price));
  if ([entry, stop, tp1, tp2].some((v) => v === "-")) return "";
  return `피보 진입 ${entry} / 손절 ${stop} / 익절 ${tp1}~${tp2}`;
}

function setAutoTickStatus(data) {
  const action = String(data?.action || "");
  const reason = String(data?.reason || "");
  const planText = autoFibPlanStatusText(data?.plan);
  if (action === "OPENED") {
    setAutoStatus("자동매매 진입 실행됨");
    return;
  }
  if (action === "NO_SIGNAL") {
    const base =
      reason === "BASIC_PASS_FAIL"
        ? "자동매매 대기: 확률 미PASS"
        : "자동매매 대기: 진입 신호 없음";
    setAutoStatus(planText ? `${base} | ${planText}` : base);
    return;
  }
  if (action === "WAIT_FIB_ENTRY") {
    const base = "자동매매 대기: 피보나치 진입가 대기";
    setAutoStatus(planText ? `${base} | ${planText}` : base);
    return;
  }
  if (action === "SKIP_DAILY_LOSS_LIMIT") {
    setAutoStatus("자동매매 중지: 일일 손실 한도 도달");
    return;
  }
  if (action === "SKIP_OPEN_LIMIT") {
    setAutoStatus("자동매매 대기: 최대 포지션 수 도달");
    return;
  }
  if (action === "SKIP_COOLDOWN") {
    setAutoStatus("자동매매 대기: 재진입 쿨다운");
    return;
  }
  if (action === "SKIP_COLLATERAL_LOW") {
    setAutoStatus("자동매매 중지: 담보금 부족");
    return;
  }
  if (action === "SKIP_COLLATERAL_ERROR") {
    setAutoStatus("자동매매 대기: 담보금 조회 실패");
    return;
  }
  if (action === "ORDER_REJECTED") {
    setAutoStatus(`실주문 실패: ${String(data?.detail || "주문 거부")}`);
    return;
  }
  if (action === "DISABLED") {
    setAutoStatus("자동매매 비활성화 상태");
    return;
  }
  setAutoStatus("");
}

function autoStatusMeta(status) {
  const st = String(status || "").toUpperCase();
  if (st === "TP") return { cls: "tp", text: "익절" };
  if (st === "SL") return { cls: "sl", text: "손절" };
  if (st === "CLOSED_FAIL") return { cls: "fail", text: "시간종료" };
  return { cls: "open", text: "보유중" };
}

function fmtSigned(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "-";
  return `${n >= 0 ? "+" : ""}${n.toFixed(2)}`;
}

function loadCachedAutoTrades() {
  try {
    const raw = localStorage.getItem(AUTO_TRADES_CACHE_KEY);
    if (!raw) return [];
    const arr = JSON.parse(raw);
    return Array.isArray(arr) ? arr : [];
  } catch (_) {
    return [];
  }
}

function saveCachedAutoTrades(rows) {
  try {
    const arr = Array.isArray(rows) ? rows : [];
    localStorage.setItem(AUTO_TRADES_CACHE_KEY, JSON.stringify(arr.slice(0, 10)));
  } catch (_) {}
}

function renderAutoTrades(records) {
  if (!autoListEl) return;
  const rows = Array.isArray(records) ? records : [];
  if (!rows.length) {
    autoListEl.innerHTML = `<div class="sim-item sim-empty">자동매매 기록이 없습니다.</div>`;
    scheduleSidebarPinUpdate();
    return;
  }
  autoListEl.innerHTML = rows
    .map((r) => {
      const meta = autoStatusMeta(r?.status);
      const pnlTxt =
        String(r?.status || "").toUpperCase() === "OPEN"
          ? "미실현"
          : `${fmtSigned(Number(r?.pnl_usdt || 0))} USDT`;
      const signalTxt = `신호 B ${Number(r?.signal_buy_pct || 0).toFixed(1)} / S ${Number(r?.signal_sell_pct || 0).toFixed(
        1
      )} / C ${(Number(r?.signal_confidence || 0) * 100).toFixed(1)}%`;
      return `<div class="sim-item">
        <div class="sim-item-head">
          <div class="sim-symbol">${r.symbol} · ${String(r.mode || "balanced") === "aggressive" ? "공격" : "기본"}</div>
          <span class="badge ${meta.cls}">${meta.text}</span>
        </div>
        <div class="sim-prices">
          <div class="sim-price"><span>진입</span><strong>${formatPrice(Number(r.entry_price))}</strong></div>
          <div class="sim-price"><span>익절</span><strong>${formatPrice(Number(r.take_profit_price))}</strong></div>
          <div class="sim-price"><span>손절</span><strong>${formatPrice(Number(r.stop_loss_price))}</strong></div>
        </div>
        <div class="sim-time">등록 ${fmtSimTs(r.opened_ms)} · 결과 ${pnlTxt}\n${signalTxt}</div>
      </div>`;
    })
    .join("");
  scheduleSidebarPinUpdate();
}

function applyAutoConfig(cfg) {
  if (!cfg || typeof cfg !== "object") return;
  if (autoEnabledEl) autoEnabledEl.checked = Boolean(cfg.enabled);
  if (autoModeEl) autoModeEl.value = String(cfg.mode || "balanced");
  if (autoIntervalEl) autoIntervalEl.value = String(cfg.interval || "5m");
  if (autoOrderSizeEl) autoOrderSizeEl.value = String(Number(cfg.order_size_usdt || 0));
  if (autoDailyLossEl) autoDailyLossEl.value = String(Number(cfg.daily_max_loss_usdt || 0));
  if (autoTpPctEl) autoTpPctEl.value = String(Number(cfg.take_profit_pct || 0));
  if (autoSlPctEl) autoSlPctEl.value = String(Number(cfg.stop_loss_pct || 0));
  if (autoCooldownMinEl) autoCooldownMinEl.value = String(Number(cfg.cooldown_min || 0));
  if (autoMaxOpenEl) autoMaxOpenEl.value = String(Number(cfg.max_open_positions || 1));
}

function collectAutoConfigPayload() {
  return {
    enabled: Boolean(autoEnabledEl?.checked),
    mode: String(autoModeEl?.value || "balanced"),
    symbol: selectedSymbol,
    market: selectedMarket,
    interval: String(autoIntervalEl?.value || "5m"),
    order_size_usdt: Number(autoOrderSizeEl?.value || 0),
    daily_max_loss_usdt: Number(autoDailyLossEl?.value || 0),
    take_profit_pct: Number(autoTpPctEl?.value || 0),
    stop_loss_pct: Number(autoSlPctEl?.value || 0),
    cooldown_min: Number(autoCooldownMinEl?.value || 0),
    max_open_positions: Number(autoMaxOpenEl?.value || 1),
  };
}

async function loadAutoConfig() {
  if (!autoEnabledEl) return;
  try {
    const data = await fetchJSON("/api/auto_trade/config");
    const cfg = data?.config || {};
    applyAutoConfig(cfg);
    autoConfigReady = true;
    setAutoStatus("");
  } catch (e) {
    autoConfigReady = false;
    setAutoStatus(`자동매매 설정 조회 실패: ${e.message || e}`);
  }
}

async function saveAutoConfig() {
  if (!autoEnabledEl) return;
  const payload = collectAutoConfigPayload();
  try {
    const data = await fetchJSON("/api/auto_trade/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const cfg = data?.config || payload;
    applyAutoConfig(cfg);
    autoConfigReady = true;
    setAutoStatus("자동매매 설정이 저장되었습니다.");
  } catch (e) {
    setAutoStatus(`설정 저장 실패: ${e.message || e}`);
  }
}

async function loadAutoTrades() {
  if (!autoListEl) return;
  try {
    const data = await fetchJSON("/api/auto_trade/records?limit=10&page=1&sync=0");
    const rows = data?.records || [];
    renderAutoTrades(rows);
    saveCachedAutoTrades(rows);
  } catch (e) {
    setAutoStatus(`자동매매 기록 조회 실패: ${e.message || e}`);
  }
}

async function runAutoTradeTick(force = false, silent = false) {
  if (!autoEnabledEl || autoTickBusy) return;
  autoTickBusy = true;
  try {
    const data = await fetchJSON("/api/auto_trade/tick", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ force: Boolean(force) }),
    });
    if (!silent) {
      setAutoTickStatus(data);
    }
    await loadAutoTrades();
  } catch (e) {
    if (!silent) setAutoStatus(`자동매매 실행 실패: ${e.message || e}`);
  } finally {
    autoTickBusy = false;
  }
}

function handleSelectCoin(symbol, market) {
  if (selectedSymbol === symbol && selectedMarket === market) return;
  selectedSymbol = symbol;
  selectedMarket = market;
  saveUiState();
  renderProbAsset();
  requestCoinListRender(true);
  renderNewsPanel();
  const reqSeq = ++coinSwitchSeq;
  if (coinSwitchTimer) clearTimeout(coinSwitchTimer);
  coinSwitchTimer = setTimeout(() => {
    if (reqSeq !== coinSwitchSeq) return;
    resetAnalysisUI("코인 변경: 계산 중...");
    resetPassCheckUI("코인 변경: 계산 중...");
    syncSimSymbolWithSelected();
    fillSimEntryWithCurrentPrice();
    loadSimTrades().catch(() => {});
    requestAnimationFrame(() => {
      if (reqSeq !== coinSwitchSeq) return;
      load().catch((e) => alert(e.message));
    });
  }, 90);
}

function coinKey(symbol, market) {
  return `${market}:${symbol}`;
}

function bindCoinItemEvents(btn, symbol, market) {
  const onSelect = () => handleSelectCoin(symbol, market);
  let touchTrack = null;
  btn.addEventListener("pointerdown", (ev) => {
    if (ev.pointerType === "mouse") return;
    if (isCoinListScrolling) return;
    lastTouchPointerTs = Date.now();
    suppressCoinClickUntil = Math.max(suppressCoinClickUntil, Date.now() + 700);
    touchTrack = {
      id: ev.pointerId,
      x: Number(ev.clientX || 0),
      y: Number(ev.clientY || 0),
      sy: Number(window.scrollY || window.pageYOffset || 0),
      ts: Date.now(),
      moved: false,
    };
  });
  btn.addEventListener("pointermove", (ev) => {
    if (!touchTrack || ev.pointerId !== touchTrack.id) return;
    const dx = Math.abs(Number(ev.clientX || 0) - touchTrack.x);
    const dy = Math.abs(Number(ev.clientY || 0) - touchTrack.y);
    if (dx > 6 || dy > 6) touchTrack.moved = true;
  });
  btn.addEventListener("pointercancel", (ev) => {
    if (touchTrack && ev.pointerId === touchTrack.id) {
      suppressCoinClickUntil = Date.now() + 700;
      touchTrack = null;
    }
  });
  btn.addEventListener("pointerup", (ev) => {
    if (ev.pointerType === "mouse") return;
    if (isCoinListScrolling) return;
    if (!touchTrack || ev.pointerId !== touchTrack.id) return;
    const elapsed = Date.now() - touchTrack.ts;
    const syNow = Number(window.scrollY || window.pageYOffset || 0);
    const pageScrolled = Math.abs(syNow - Number(touchTrack.sy || 0)) > 2;
    const recentScroll = Date.now() - lastScrollActivityTs < 180;
    const shouldSelect = !touchTrack.moved && !pageScrolled && !recentScroll && elapsed <= 600;
    touchTrack = null;
    if (!shouldSelect) {
      suppressCoinClickUntil = Date.now() + 700;
      return;
    }
    suppressCoinClickUntil = Date.now() + 700;
    onSelect();
  });
  btn.addEventListener("click", () => {
    if (isCoinListScrolling) return;
    if (Date.now() - lastTouchPointerTs < 900) return; // touch synthetic click 차단
    if (Date.now() < suppressCoinClickUntil) return;
    if (window.innerWidth <= 1100 && Date.now() - lastScrollActivityTs < 130) return;
    onSelect();
  });
}

function ensureCoinListStructure() {
  if (!coinListEl) return;
  if (coinListNodes.size === COINS.length && coinListEl.children.length === COINS.length) return;
  coinListEl.innerHTML = "";
  coinListNodes = new Map();
  for (const c of COINS) {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "coin-item";
    const left = document.createElement("span");
    left.className = "coin-left";
    const mark = document.createElement("span");
    mark.className = "coin-mark";
    mark.textContent = c.mark || "•";
    const name = document.createElement("span");
    name.className = "coin-name";
    name.textContent = c.name;
    const tag = document.createElement("span");
    tag.className = "coin-tag";
    tag.textContent = c.market === "futures" ? "선물" : "현물";
    left.appendChild(mark);
    left.appendChild(name);
    left.appendChild(tag);

    const right = document.createElement("span");
    right.className = "coin-right";
    const priceEl = document.createElement("span");
    priceEl.className = "coin-price";
    const krwEl = document.createElement("span");
    krwEl.className = "coin-price-krw";
    const newsEl = document.createElement("span");
    newsEl.className = "coin-news";
    right.appendChild(priceEl);
    right.appendChild(krwEl);
    right.appendChild(newsEl);
    btn.appendChild(left);
    btn.appendChild(right);
    bindCoinItemEvents(btn, c.symbol, c.market);
    li.appendChild(btn);
    coinListEl.appendChild(li);
    coinListNodes.set(coinKey(c.symbol, c.market), { btn, priceEl, krwEl, newsEl });
  }
}

function renderCoinList() {
  if (!coinListEl) return;
  ensureCoinListStructure();
  for (const c of COINS) {
    const nodes = coinListNodes.get(coinKey(c.symbol, c.market));
    if (!nodes) continue;
    const price = coinPrices[c.symbol];
    const news = coinNews[c.symbol] || { positive: 0, negative: 0 };
    const priceText = Number.isFinite(price) ? formatPrice(price) : "가격 불러오는 중";
    const krwText = Number.isFinite(price) ? formatKrw(price * usdtKrw) : "원화 계산 중";
    const newsText = `긍정 ${Number(news.positive || 0)} / 부정 ${Number(news.negative || 0)}`;
    nodes.btn.classList.toggle("active", c.symbol === selectedSymbol && c.market === selectedMarket);
    if (nodes.priceEl.textContent !== priceText) nodes.priceEl.textContent = priceText;
    if (nodes.krwEl.textContent !== krwText) nodes.krwEl.textContent = krwText;
    if (nodes.newsEl.textContent !== newsText) nodes.newsEl.textContent = newsText;
  }
  scheduleSidebarPinUpdate();
}

function fmtNewsTime(ms) {
  if (!Number.isFinite(Number(ms))) return "-";
  return new Date(Number(ms)).toLocaleString("ko-KR", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function renderNewsPanel() {
  if (!newsSummaryEl || !newsListEl) return;
  const coin = COINS.find((c) => c.symbol === selectedSymbol) || COINS[0];
  const data = coinNews[selectedSymbol] || { positive: 0, negative: 0, neutral: 0, items: [] };
  const pos = Number(data.positive || 0);
  const neg = Number(data.negative || 0);
  const neu = Number(data.neutral || 0);
  const items = Array.isArray(data.items) ? data.items : [];

  newsSummaryEl.textContent = `${coin.name} 뉴스 집계: 긍정 ${pos} / 부정 ${neg} / 중립 ${neu}`;
  if (!items.length) {
    newsListEl.innerHTML = `<div class="news-item"><div class="news-meta">최근 1일 기준 관련 뉴스가 없습니다.</div></div>`;
    return;
  }
  newsListEl.innerHTML = items
    .map((it) => {
      const sentiment = String(it?.sentiment || "neutral");
      const sentiClass = sentiment === "positive" ? "pos" : sentiment === "negative" ? "neg" : "neu";
      const sentiText = sentiment === "positive" ? "긍정" : sentiment === "negative" ? "부정" : "중립";
      const title = String(it?.title || "-").replace(/</g, "&lt;").replace(/>/g, "&gt;");
      const link = String(it?.link || "#").replace(/"/g, "&quot;");
      const source = String(it?.source || "-");
      const timeText = fmtNewsTime(Number(it?.published_ms));
      return `
        <article class="news-item">
          <div class="news-item-top">
            <span class="news-sentiment ${sentiClass}">${sentiText}</span>
            <span class="news-meta">${source} · ${timeText}</span>
          </div>
          <a class="news-title" href="${link}" target="_blank" rel="noopener noreferrer">${title}</a>
        </article>
      `;
    })
    .join("");
}

async function loadNewsSentiment() {
  if (isLoadingNews) return;
  isLoadingNews = true;
  try {
    const data = await fetchJSON("/api/news_sentiment");
    const symbols = data?.symbols || {};
    const next = {};
    for (const c of COINS) {
      const s = symbols?.[c.symbol] || {};
      next[c.symbol] = {
        positive: Number(s?.positive || 0),
        negative: Number(s?.negative || 0),
        neutral: Number(s?.neutral || 0),
        items: Array.isArray(s?.items) ? s.items : [],
      };
    }
    coinNews = next;
    requestCoinListRender();
    renderNewsPanel();
  } catch (_) {
  } finally {
    isLoadingNews = false;
  }
}

function setCoinPrice(symbol, price) {
  if (!Number.isFinite(price)) return;
  coinPrices[symbol] = price;
  if (coinRenderTimer) return;
  coinRenderTimer = setTimeout(() => {
    coinRenderTimer = null;
    requestCoinListRender();
  }, 250);
}

function formatPrice(v) {
  if (!Number.isFinite(v)) return "-";
  if (v >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 2 });
  if (v >= 1) return v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 });
  return v.toLocaleString(undefined, { minimumFractionDigits: 4, maximumFractionDigits: 6 });
}

function formatKrw(v) {
  if (!Number.isFinite(v)) return "₩ -";
  return `₩ ${Math.round(v).toLocaleString("ko-KR")}`;
}

function fmtZone(low, high) {
  if (!Number.isFinite(low) || !Number.isFinite(high)) return "-";
  return `${formatPrice(low)} ~ ${formatPrice(high)} USDT`;
}

function fmtOne(v) {
  if (!Number.isFinite(v)) return "-";
  return `${formatPrice(v)} USDT`;
}

function fmtZoneSorted(a, b) {
  if (!Number.isFinite(a) || !Number.isFinite(b)) return "-";
  const lo = Math.min(a, b);
  const hi = Math.max(a, b);
  return `${formatPrice(lo)} ~ ${formatPrice(hi)} USDT`;
}

function fmtTakeProfitZone(entryHi, a, b, minGapPct = 0.004) {
  if (!Number.isFinite(a) || !Number.isFinite(b)) return "-";
  let lo = Math.min(a, b);
  let hi = Math.max(a, b);
  if (Number.isFinite(entryHi) && entryHi > 0) {
    const minLo = entryHi * (1 + minGapPct);
    if (lo < minLo) lo = minLo;
    if (hi <= lo) hi = lo * (1 + minGapPct);
  }
  return `${formatPrice(lo)} ~ ${formatPrice(hi)} USDT`;
}

function zoneMid(lo, hi) {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return NaN;
  return (lo + hi) / 2;
}

function normalizeTpZone(entryHi, a, b, minGapPct) {
  if (!Number.isFinite(a) || !Number.isFinite(b)) return { lo: NaN, hi: NaN };
  let lo = Math.min(a, b);
  let hi = Math.max(a, b);
  if (Number.isFinite(entryHi) && entryHi > 0) {
    const minLo = entryHi * (1 + minGapPct);
    if (lo < minLo) lo = minLo;
    if (hi <= lo) hi = lo * (1 + minGapPct);
  }
  return { lo, hi };
}

function normalizeShortTpZone(entryLo, a, b, minGapPct) {
  if (!Number.isFinite(a) || !Number.isFinite(b)) return { lo: NaN, hi: NaN };
  let lo = Math.min(a, b);
  let hi = Math.max(a, b);
  if (Number.isFinite(entryLo) && entryLo > 0) {
    const maxHi = entryLo * (1 - minGapPct);
    if (hi > maxHi) hi = maxHi;
    if (lo >= hi) lo = hi * (1 - minGapPct);
  }
  return { lo, hi };
}

function buildTpText(t1Lo, t1Hi, t2Lo, t2Hi) {
  const line1 = Number.isFinite(t1Lo) && Number.isFinite(t1Hi) ? `1차 익절: ${formatPrice(t1Lo)} ~ ${formatPrice(t1Hi)} USDT` : "1차 익절: -";
  const line2 = Number.isFinite(t2Lo) && Number.isFinite(t2Hi) ? `2차 익절: ${formatPrice(t2Lo)} ~ ${formatPrice(t2Hi)} USDT` : "2차 익절: -";
  return `${line1}\n${line2}`;
}

function symbolProfile(symbol) {
  const s = String(symbol || "").toUpperCase();
  if (s === "BTCUSDT" || s === "ETHUSDT") return "major";
  if (s === "DOGEUSDT" || s === "SUIUSDT" || s === "SOLUSDT" || s === "XRPUSDT" || s === "CROSSUSDT") return "beta";
  return "default";
}

function decisionParamsByRegime(regime, symbol) {
  const r = String(regime || "").toUpperCase();
  let out;
  if (r === "TREND") {
    out = {
      sideStrong: 8,
      sideWeak: 4,
      confWeak: 0.2,
      confFloor: 0.05,
      probMinPct: 52,
      probMinConf: 0.3,
      probSoftConf: 0.36,
      passRegimeConf: 0.33,
      passRegimeDiff: 8,
      fibTolPct: 0.0035,
    };
  } else if (r === "HIGH_VOL") {
    out = {
      sideStrong: 12,
      sideWeak: 6,
      confWeak: 0.3,
      confFloor: 0.1,
      probMinPct: 57,
      probMinConf: 0.42,
      probSoftConf: 0.5,
      passRegimeConf: 0.45,
      passRegimeDiff: 12,
      fibTolPct: 0.006,
    };
  } else {
    out = {
      sideStrong: 10,
      sideWeak: 5,
      confWeak: 0.26,
      confFloor: 0.07,
      probMinPct: 55,
      probMinConf: 0.36,
      probSoftConf: 0.42,
      passRegimeConf: 0.38,
      passRegimeDiff: 10,
      fibTolPct: 0.004,
    };
  }
  const profile = symbolProfile(symbol);
  if (profile === "major") {
    out.probMinPct = Math.max(50, out.probMinPct - 1);
    out.probMinConf = Math.max(0.28, out.probMinConf - 0.03);
    out.probSoftConf = Math.max(0.33, out.probSoftConf - 0.03);
    out.passRegimeConf = Math.max(0.3, out.passRegimeConf - 0.03);
  } else if (profile === "beta") {
    out.sideStrong += 1;
    out.probMinConf = Math.min(0.48, out.probMinConf + 0.02);
    out.probSoftConf = Math.min(0.58, out.probSoftConf + 0.02);
    out.passRegimeConf = Math.min(0.52, out.passRegimeConf + 0.02);
  }
  return out;
}

function fmtFibZones(entryZone, resistanceZone) {
  return [`관찰 진입구간`, `- ${entryZone}`, `상단/하단 저항 구간`, `- ${resistanceZone}`].join("\n");
}

function fibPriceOf(fibPlan, ratio) {
  const v = fibPlan?.prices?.find((x) => Number(x.r) === Number(ratio))?.price;
  return Number(v);
}

async function loadFxRate() {
  try {
    const data = await fetchJSON("/api/usdt_krw");
    const rate = Number(data?.usdt_krw);
    if (Number.isFinite(rate) && rate > 0) {
      usdtKrw = rate;
      requestCoinListRender();
    }
  } catch (_) {}
}

async function loadCoinPrices() {
  if (isLoadingPrices) return;
  isLoadingPrices = true;
  try {
    const reqs = COINS.map(async (c) => {
      try {
        const data = await fetchJSON(
          `/api/klines?symbol=${encodeURIComponent(c.symbol)}&market=${encodeURIComponent(c.market)}&interval=5m&limit=50`
        );
        const rows = data?.klines || [];
        const last = rows[rows.length - 1];
        const close = Number(last?.close);
        return [c.symbol, Number.isFinite(close) ? close : null];
      } catch (_) {
        return [c.symbol, null];
      }
    });
    const result = await Promise.all(reqs);
    const next = {};
    for (const [sym, v] of result) next[sym] = v;
    coinPrices = next;
    requestCoinListRender();
  } finally {
    isLoadingPrices = false;
  }
}

function connectListStream(market, symbols) {
  if (!symbols.length) return null;
  const streamPath = symbols.map((s) => `${s.toLowerCase()}@miniTicker`).join("/");
  const base =
    market === "futures"
      ? "wss://fstream.binance.com/stream?streams="
      : "wss://stream.binance.com:9443/stream?streams=";
  const ws = new WebSocket(base + streamPath);
  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data);
      const data = msg?.data || msg;
      const sym = String(data?.s || "").toUpperCase();
      const close = Number(data?.c);
      if (sym) setCoinPrice(sym, close);
    } catch (_) {}
  };
  return ws;
}

function listStreamRef(market) {
  return market === "futures" ? futuresListWs : spotListWs;
}

function setListStreamRef(market, ws) {
  if (market === "futures") futuresListWs = ws;
  else spotListWs = ws;
}

function listReconnectTimerRef(market) {
  return market === "futures" ? futuresListReconnectTimer : spotListReconnectTimer;
}

function setListReconnectTimerRef(market, timer) {
  if (market === "futures") futuresListReconnectTimer = timer;
  else spotListReconnectTimer = timer;
}

function clearListReconnectTimer(market) {
  const t = listReconnectTimerRef(market);
  if (!t) return;
  clearTimeout(t);
  setListReconnectTimerRef(market, null);
}

function stopSingleListStream(market) {
  clearListReconnectTimer(market);
  const ws = listStreamRef(market);
  if (!ws) return;
  try {
    ws.onclose = null;
    ws.onmessage = null;
    ws.onopen = null;
    ws.onerror = null;
    ws.close();
  } catch (_) {}
  setListStreamRef(market, null);
}

function stopListTickerStreams() {
  stopSingleListStream("spot");
  stopSingleListStream("futures");
}

function scheduleSingleListReconnect(market) {
  if (document.hidden) return;
  if (listStreamDisabled[market]) return;
  if (listReconnectTimerRef(market)) return;
  const fails = Math.max(1, Number(listStreamFailureCount[market] || 1));
  const delay = Math.min(30000, 1500 * Math.pow(2, Math.max(0, fails - 1)));
  const timer = setTimeout(() => {
    setListReconnectTimerRef(market, null);
    startSingleListStream(market);
  }, delay);
  setListReconnectTimerRef(market, timer);
}

function startSingleListStream(market) {
  if (document.hidden) return;
  if (listStreamDisabled[market]) return;
  const prev = listStreamRef(market);
  if (prev && (prev.readyState === WebSocket.OPEN || prev.readyState === WebSocket.CONNECTING)) return;
  stopSingleListStream(market);
  const symbols = COINS.filter((c) => c.market === market).map((c) => c.symbol);
  if (!symbols.length) return;
  const ws = connectListStream(market, symbols);
  if (!ws) return;
  setListStreamRef(market, ws);
  ws.onopen = () => {
    listStreamFailureCount[market] = 0;
  };
  ws.onclose = () => {
    if (listStreamRef(market) !== ws) return;
    setListStreamRef(market, null);
    listStreamFailureCount[market] = Number(listStreamFailureCount[market] || 0) + 1;
    // 지역/정책 차단으로 지속 실패하는 시장은 세션 동안 비활성화
    if (listStreamFailureCount[market] >= 6) {
      listStreamDisabled[market] = true;
      return;
    }
    scheduleSingleListReconnect(market);
  };
  ws.onerror = () => {
    try {
      ws.close();
    } catch (_) {}
  };
}

function resetListStreamHealth() {
  listStreamFailureCount.spot = 0;
  listStreamFailureCount.futures = 0;
  listStreamDisabled.spot = false;
  listStreamDisabled.futures = false;
}

function startListTickerStreams() {
  if (document.hidden) return;
  startSingleListStream("spot");
  startSingleListStream("futures");
}

function hasLiveListStream() {
  const isLive = (ws) => ws && ws.readyState === WebSocket.OPEN;
  return Boolean(isLive(spotListWs) || isLive(futuresListWs));
}

function scheduleListTickerReconnect() {
  scheduleSingleListReconnect("spot");
  scheduleSingleListReconnect("futures");
}

function startListTickerStreamsLegacy__deprecated() {
  const closeWs = (ws) => {
    if (!ws) return;
    try {
      ws.onclose = null;
      ws.onmessage = null;
      ws.close();
    } catch (_) {}
  };
  closeWs(spotListWs);
  closeWs(futuresListWs);
  spotListWs = null;
  futuresListWs = null;
}

function stopChartStream() {
  if (chartWsReconnectTimer) {
    clearTimeout(chartWsReconnectTimer);
    chartWsReconnectTimer = null;
  }
  if (!chartWs) return;
  try {
    chartWs.onclose = null;
    chartWs.onmessage = null;
    chartWs.close();
  } catch (_) {}
  chartWs = null;
}

function stopAnalysisStream() {
  if (analysisWsReconnectTimer) {
    clearTimeout(analysisWsReconnectTimer);
    analysisWsReconnectTimer = null;
  }
  if (!analysisWs) return;
  try {
    analysisWs.onclose = null;
    analysisWs.onmessage = null;
    analysisWs.close();
  } catch (_) {}
  analysisWs = null;
}

function handleVisibilityRealtime() {
  if (document.hidden) {
    stopListTickerStreams();
    stopChartStream();
    stopAnalysisStream();
    return;
  }
  resetListStreamHealth();
  startListTickerStreams();
  connectChartStream();
  connectAnalysisStream();
  safeLoadPrices();
  loadNewsSentiment();
  loadFxRate();
}

function bindVisibilityRealtime() {
  document.addEventListener("visibilitychange", () => {
    handleVisibilityRealtime();
  });
  handleVisibilityRealtime();
}

function fmtTs(ms) {
  const d = new Date(ms);
  return d.toLocaleString();
}

function fmtYmd(ms) {
  const n = Number(ms || 0);
  if (!Number.isFinite(n) || n <= 0) return "-";
  const d = new Date(n);
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}.${m}.${day}`;
}

function syncAnalysisControls() {
  const mode = analysisModeEl ? analysisModeEl.value : "single";
  if (analysisIntervalEl) {
    const dashOpt = analysisIntervalEl.querySelector('option[value=""]');
    const disabled = mode === "mtf";
    if (dashOpt) {
      dashOpt.hidden = !disabled;
      dashOpt.disabled = !disabled;
    }
    if (disabled) {
      if (analysisIntervalEl.value) lastSingleAnalysisInterval = analysisIntervalEl.value;
      analysisIntervalEl.value = "";
    } else if (!analysisIntervalEl.value) {
      analysisIntervalEl.value = normalizeAnalysisInterval(lastSingleAnalysisInterval);
    } else {
      analysisIntervalEl.value = normalizeAnalysisInterval(analysisIntervalEl.value);
    }
    analysisIntervalEl.disabled = disabled;
    analysisIntervalEl.classList.toggle("is-disabled", disabled);
  }
}

function connectChartStream() {
  if (document.hidden) return;
  const interval = chartIntervalEl ? chartIntervalEl.value : "5m";
  const nextKey = `${selectedMarket}:${selectedSymbol}:${interval}`;
  if (chartWs && chartStreamKey === nextKey) return;
  chartStreamKey = nextKey;

  stopChartStream();
  const symbol = selectedSymbol.toLowerCase();
  const base =
    selectedMarket === "futures"
      ? "wss://fstream.binance.com/ws/"
      : "wss://stream.binance.com:9443/ws/";
  const url = `${base}${symbol}@kline_${interval}`;
  const ws = new WebSocket(url);
  chartWs = ws;
  ws.onmessage = (ev) => {
    if (!candleSeries) return;
    try {
      const msg = JSON.parse(ev.data);
      const k = msg?.k;
      if (!k) return;
      const t = Math.floor(Number(k.t) / 1000);
      const o = Number(k.o);
      const c = Number(k.c);
      const h = Number(k.h);
      const l = Number(k.l);
      candleSeries.update({
        time: t,
        open: o,
        high: h,
        low: l,
        close: c,
      });
      if (volumeSeries && volumeToggleEl && volumeToggleEl.checked) {
        volumeSeries.update({
          time: t,
          value: Number(k.v),
          color: c >= o ? "rgba(110,231,255,0.45)" : "rgba(255,99,132,0.45)",
        });
      }
      setCoinPrice(String(k.s).toUpperCase(), c);
    } catch (_) {}
  };
  ws.onclose = () => {
    if (chartWs !== ws) return;
    chartWs = null;
    if (document.hidden) return;
    chartWsReconnectTimer = setTimeout(() => connectChartStream(), 2000);
  };
}

function toViewAnalysis(analysisRaw, analysisMode, interval, market) {
  return analysisMode === "mtf"
    ? {
        ...analysisRaw,
        symbol: analysisRaw.symbol,
        interval: "mtf",
        asof_open_time_ms: analysisRaw?.tf?.["5m"]?.asof_open_time_ms,
        close: analysisRaw?.tf?.["5m"]?.close,
        regime: analysisRaw?.tf?.["4h"]?.regime || "-",
        reasons: analysisRaw?.tf?.["5m"]?.reasons || [],
        levels: analysisRaw?.tf?.["5m"]?.levels || { avwap: [], volume_profile: null },
        indicators: analysisRaw?.tf?.[interval]?.indicators || analysisRaw?.tf?.["5m"]?.indicators || null,
        confidence: analysisRaw.confidence,
        buy_pct: analysisRaw.buy_pct,
        sell_pct: analysisRaw.sell_pct,
        market: analysisRaw.market || market,
        explain: analysisRaw?.explain || null,
        mtf_explain: analysisRaw?.explain || null,
        tf_explain: analysisRaw?.tf?.[interval]?.explain || null,
        selected_chart_tf: interval,
      }
    : analysisRaw;
}

function clearExplainDetail() {
  if (!explainDetailEl) return;
  explainDetailEl.innerHTML = "";
}

function appendExplainRow(title, value) {
  if (!explainDetailEl) return;
  const row = document.createElement("div");
  row.className = "explain-row";
  const left = document.createElement("span");
  left.className = "explain-k";
  left.textContent = title;
  const right = document.createElement("span");
  right.className = "explain-v";
  right.textContent = value;
  row.appendChild(left);
  row.appendChild(right);
  explainDetailEl.appendChild(row);
}

function renderExplainDetail(analysis, analysisMode) {
  clearExplainDetail();
  const ex = analysis?.explain;
  if (!explainDetailEl || !ex) return;

  const pb = ex?.probability || {};
  const calcEasy = Array.isArray(ex?.calc_easy) ? ex.calc_easy : [];
  const breakdown = ex?.calc_breakdown || {};
  const breakdownRows = Array.isArray(breakdown?.rows) ? breakdown.rows : [];
  const sg = ex?.simple_guide || {};
  const conf = ex?.confidence || {};
  const decision = ex?.decision || {};
  const indicators = Array.isArray(ex?.indicators) ? ex.indicators : [];

  if (calcEasy.length) {
    appendExplainRow("계산 순서", calcEasy.join("\n"));
  }

  if (breakdownRows.length) {
    const lines = [];
    if (breakdown?.formula) lines.push(`공식: ${breakdown.formula}`);
    if (Number.isFinite(Number(breakdown?.x))) lines.push(`최종 점수 x: ${Number(breakdown.x).toFixed(4)}`);
    for (const row of breakdownRows) {
      const usedTxt =
        Array.isArray(row.used_indicators) && row.used_indicators.length
          ? `\n  └ 사용지표: ${row.used_indicators.join(", ")}`
          : "";
      const detailTxt = Array.isArray(row.detail) && row.detail.length ? `\n  └ 세부점수: ${row.detail.join(" | ")}` : "";
      lines.push(
        `[${row.label}] raw=${Number(row.raw || 0).toFixed(3)} × weight=${Number(row.weight || 0).toFixed(
          3
        )} = ${Number(row.contribution || 0).toFixed(3)}${usedTxt}${detailTxt}`
      );
    }
    appendExplainRow("부분점수 계산표", lines.join("\n"));
  }

  if (decision?.favor || conf?.label) {
    appendExplainRow(
      "현재 상태",
      `구분값: ${decision?.favor || "-"}\n신뢰도: ${(Number(conf?.value || 0) * 100).toFixed(1)}% (${conf?.label || "-"})\n의미: ${
        decision?.meaning || "-"
      }`
    );
  }

  if (pb?.buy?.label) {
    appendExplainRow(
      "롱 % 해석",
      `현재 상태: ${pb.buy.pct}% (${pb.buy.label})\n의미: ${pb.buy.meaning}\n행동 가이드: ${pb.buy.action_hint || "-"}`
    );
  }

  if (pb?.sell?.label) {
    appendExplainRow(
      "숏 % 해석",
      `현재 상태: ${pb.sell.pct}% (${pb.sell.label})\n의미: ${pb.sell.meaning}\n행동 가이드: ${pb.sell.action_hint || "-"}`
    );
  }

  if (sg?.buy_sell || sg?.confidence) {
    appendExplainRow(
      "숫자 읽는 법",
      `${sg?.buy_sell || "-"}\n기준: ${sg?.buy_sell_rule || "-"}\n${sg?.confidence || "-"}\n기준: ${sg?.confidence_rule || "-"}`
    );
  }

  if (decision?.drivers) appendExplainRow("핵심 근거", decision.drivers);

  for (const item of indicators) {
    appendExplainRow(
      `${item.name}`,
      `현재 상태: ${item.state}\n유리한 쪽: ${item.favor}\n현재 값: ${item.value}\n의미: ${item.meaning}`
    );
  }

  if (analysisMode === "mtf" && analysis?.mtf_explain?.mtf?.how_to_read) {
    appendExplainRow("MTF 읽는 법", `구분값: 멀티 타임프레임\n${analysis.mtf_explain.mtf.how_to_read}`);
    appendExplainRow("MTF 계산 기준", "4h/1h 방향이 같은 경우에만 5m 진입판단을 허용합니다.");
  }

  appendExplainRow(
    "확률-피보 정렬 규칙",
    "확률 방향과 피보 스윙 방향이 충돌하면 실행 판단은 무조건 WAIT 처리합니다."
  );

  if (lastFibPlan) {
    const methodTxt = lastFibPlan.method === "pivot" ? "피벗 기반(좌우 3봉 비교)" : "단순 최근 고저";
    const swingTxt = lastFibPlan.isUpMove ? "상승 스윙(L→H)" : "하락 스윙(H→L)";
    appendExplainRow(
      "피보 선 기준",
      `기준 데이터: 최근 ${lastFibPlan.lookback}봉\n스윙 판정: ${swingTxt}\n고점=${formatPrice(lastFibPlan.hi)}, 저점=${formatPrice(
        lastFibPlan.lo
      )}\n선정 방식: ${methodTxt}`
    );
  }
}

function mtfBiasFromAnalysis(analysis) {
  const t4 = analysis?.tf?.["4h"];
  const t1 = analysis?.tf?.["1h"];
  if (!t4 || !t1) return { aligned: true, side: null, detail: "단일 분석 모드" };
  const side4 = Number(t4.buy_pct) >= Number(t4.sell_pct) ? "BUY" : "SELL";
  const side1 = Number(t1.buy_pct) >= Number(t1.sell_pct) ? "BUY" : "SELL";
  const aligned = side4 === side1;
  return {
    aligned,
    side: aligned ? side4 : null,
    detail: `4h ${side4} / 1h ${side1}${aligned ? " (합의)" : " (충돌)"}`,
  };
}

function hasEntryReaction(side, entryLo, entryHi, klines) {
  if (!Array.isArray(klines) || klines.length < 3) return false;
  if (!Number.isFinite(entryLo) || !Number.isFinite(entryHi)) return false;
  const rows = klines.slice(-5);
  const prev = rows.slice(0, -1);
  const last = rows[rows.length - 1];
  const touched = prev.some((k) => {
    const h = Number(k?.high);
    const l = Number(k?.low);
    return Number.isFinite(h) && Number.isFinite(l) && h >= entryLo && l <= entryHi;
  });
  if (!touched) return false;
  const lastClose = Number(last?.close);
  if (!Number.isFinite(lastClose)) return false;
  if (side === "BUY") return lastClose >= entryHi;
  if (side === "SELL") return lastClose <= entryLo;
  return false;
}

function calcSignalScore({
  rawDiff,
  conf,
  regime,
  side,
  lowVolumeBlock,
  swingConflict,
  reversalReady,
  whaleScore = 0,
  flowWeight = 0.25,
}) {
  const edge = Math.abs(Number(rawDiff) || 0);
  const edgeScore = Math.max(0, Math.min(30, (edge / 20) * 30));
  const c = Number(conf);
  const confScore = Number.isFinite(c) ? Math.max(0, Math.min(25, (c / 0.6) * 25)) : 0;
  const r = String(regime || "").toUpperCase();
  const regimeScore = r === "TREND" ? 15 : r === "RANGE" ? 11 : r === "HIGH_VOL" ? 7 : 10;
  const s = String(side || "").toUpperCase();
  const sideScore = s === "BUY" || s === "SELL" ? 15 : 0;
  const fibScore = reversalReady ? 10 : swingConflict ? 0 : 4;
  const momentumScore = lowVolumeBlock ? 1 : 5;
  const whale = Math.max(-1, Math.min(1, Number(whaleScore) || 0));
  const wRaw = Number(flowWeight);
  const w = Number.isFinite(wRaw) ? Math.max(0.2, Math.min(0.3, wRaw)) : 0.25;
  let flowComponent = 0;
  if (s === "BUY") flowComponent = Math.max(0, whale) * 100;
  else if (s === "SELL") flowComponent = Math.max(0, -whale) * 100;
  const baseTotal = Math.max(0, Math.min(100, edgeScore + confScore + regimeScore + sideScore + fibScore + momentumScore));
  const baseContrib = baseTotal * (1 - w);
  const flowContrib = flowComponent * w;
  const total = baseContrib + flowContrib;
  return {
    total: Math.max(0, Math.min(100, total)),
    base_total: baseTotal,
    base_contrib: baseContrib,
    flow_contrib: flowContrib,
    flow_component: flowComponent,
    flow_weight: w,
    edge: edgeScore,
    conf: confScore,
    regime: regimeScore,
    side: sideScore,
    fib: fibScore,
    momentum: momentumScore,
    whale: flowContrib,
    flow: flowContrib,
  };
}

function renderActionSummary(analysis, fibPlan) {
  if (!actionBadgeEl || !actionTitleEl || !actionSubtitleEl) return;

  const buy = Number(analysis?.buy_pct);
  const sell = Number(analysis?.sell_pct);
  const conf = Number(analysis?.confidence);
  const regime = String(analysis?.regime || "").toUpperCase();
  const whaleObj = analysis?.whale_sentiment || {};
  const whaleScore = Number(whaleObj?.score);
  const flowScoreRaw = Number(whaleObj?.flow_score);
  const flowScore = Number.isFinite(flowScoreRaw) ? flowScoreRaw : whaleScore;
  const flowWeightRaw = Number(whaleObj?.flow_weight);
  const flowWeight = Number.isFinite(flowWeightRaw) ? Math.max(0.2, Math.min(0.3, flowWeightRaw)) : 0.25;
  const whaleLabel = String(whaleObj?.label || "중립");
  const close = Number(analysis?.close);
  const rawDiff = buy - sell;
  const hasFibSwing = Boolean(fibPlan && typeof fibPlan?.isUpMove === "boolean");
  const swingUp = hasFibSwing ? Boolean(fibPlan.isUpMove) : null;
  const swingBias =
    hasFibSwing ? (swingUp ? 6 : -6) : 0;
  const diff = rawDiff + swingBias;
  const nearTie = Math.abs(rawDiff) < 0.5;
  const rows = analysis?.explain?.calc_breakdown?.rows || [];
  const params = decisionParamsByRegime(regime, selectedSymbol);
  const atr14 = Number(analysis?.indicators?.atr14);
  const atrPctRaw = Number.isFinite(atr14) && Number.isFinite(close) && close > 0 ? atr14 / close : NaN;
  const atrPct = Number.isFinite(atrPctRaw) ? Math.min(0.05, Math.max(0.004, atrPctRaw)) : 0.01;
  const tp1GapPct = Math.min(0.022, Math.max(0.01, atrPct * 0.85));
  const tp2GapPct = Math.min(0.07, Math.max(0.022, atrPct * 2.4));
  const stopGapPct = Math.min(0.028, Math.max(0.007, atrPct * 0.9));

  let sideSignal = "WAIT";
  if (diff >= params.sideStrong || (diff >= params.sideWeak && conf >= params.confWeak)) sideSignal = "BUY";
  else if (diff <= -params.sideStrong || (diff <= -params.sideWeak && conf >= params.confWeak)) sideSignal = "SELL";
  const mtfBias = mtfBiasFromAnalysis(analysis);
  const mtfConflict = !mtfBias.aligned;
  if (mtfBias.aligned && mtfBias.side && sideSignal !== "WAIT" && sideSignal !== mtfBias.side) sideSignal = "WAIT";
  if (mtfConflict) sideSignal = "WAIT";
  let side = sideSignal;
  // 스윙 충돌은 기본적으로 대기, 단 반전 바닥 롱 조건 충족 시 예외 허용한다.
  let swingConflict = false;
  if (hasFibSwing) {
    if (swingUp && sideSignal === "SELL") swingConflict = true;
    if (!swingUp && sideSignal === "BUY") swingConflict = true;
    if (swingConflict) side = "WAIT";
  }

  // 신뢰도가 극단적으로 낮으면 방향 차이가 커도 관망으로 처리한다.
  if (conf < params.confFloor) sideSignal = "WAIT";
  if (regime === "HIGH_VOL" && conf < params.passRegimeConf && Math.abs(diff) < params.passRegimeDiff) sideSignal = "WAIT";
  if (sideSignal === "WAIT") side = "WAIT";
  let sideForPlan = side;
  setTradeLabels(sideForPlan);

  if (side === "BUY") {
    actionBadgeEl.textContent = "롱 우세";
    actionBadgeEl.className = "action-badge buy";
    if (nearTie && hasFibSwing) {
      actionTitleEl.textContent = "확률은 동률에 가깝지만 스윙 보정으로 롱 우위입니다.";
      actionSubtitleEl.textContent = `확률: 롱 ${buy.toFixed(2)}% / 숏 ${sell.toFixed(2)}%, 스윙 보정: +${Math.abs(
        swingBias
      ).toFixed(1)}%p`;
    } else {
      actionTitleEl.textContent = "지금은 롱 쪽이 유리합니다.";
      actionSubtitleEl.textContent =
        "추세/모멘텀 합의가 롱 방향입니다. 다만 고변동 구간이면 분할 진입이 안전합니다.";
    }
  } else if (side === "SELL") {
    actionBadgeEl.textContent = "숏 우세";
    actionBadgeEl.className = "action-badge sell";
    if (nearTie && hasFibSwing) {
      actionTitleEl.textContent = "확률은 동률에 가깝지만 스윙 보정으로 숏 우위입니다.";
      actionSubtitleEl.textContent = `확률: 롱 ${buy.toFixed(2)}% / 숏 ${sell.toFixed(2)}%, 스윙 보정: -${Math.abs(
        swingBias
      ).toFixed(1)}%p`;
    } else {
      actionTitleEl.textContent = "지금은 숏 쪽이 유리합니다.";
      actionSubtitleEl.textContent = "추세/모멘텀 합의가 숏 방향입니다. 변동성이 크면 분할 진입이 안전합니다.";
    }
  } else {
    actionBadgeEl.textContent = "관망";
    actionBadgeEl.className = "action-badge wait";
    actionTitleEl.textContent = mtfConflict
      ? "상위 타임프레임 방향 충돌로 관망입니다."
      : swingConflict
        ? "확률 방향과 피보 스윙이 충돌해 관망입니다."
        : "지금은 관망이 유리합니다.";
    actionSubtitleEl.textContent = mtfConflict
      ? "4h/1h 방향이 다릅니다. 상위 프레임이 정렬될 때까지 신규진입을 보류합니다."
      : swingConflict
        ? "확률은 한쪽 우위지만 피보 스윙이 반대입니다. 방향이 정렬될 때까지 대기하세요."
        : "롱/숏 우위 차이가 아직 작습니다. 우위가 커질 때까지 대기하세요.";
  }

  const top = Array.isArray(rows)
    ? [...rows].sort((a, b) => Math.abs(Number(b?.contribution || 0)) - Math.abs(Number(a?.contribution || 0))).slice(0, 2)
    : [];
  const topTxt =
    top.length > 0
      ? top.map((r) => `${r.label}(${Number(r.contribution || 0) >= 0 ? "+" : ""}${Number(r.contribution || 0).toFixed(3)})`).join(", ")
      : "주요 기여값 없음";
  const regimeTxt =
    regime === "TREND" ? "추세장" : regime === "RANGE" ? "횡보장" : regime === "HIGH_VOL" ? "고변동장" : "중립장";
  if (whaleStateEl) {
    if (Number.isFinite(flowScore)) whaleStateEl.textContent = `${whaleLabel}\n(${flowScore >= 0 ? "+" : ""}${flowScore.toFixed(2)})`;
    else whaleStateEl.textContent = whaleLabel || "중립";
  }
  let verdict = "그래서 관망이 유리합니다.";
  if (side === "BUY") verdict = "그래서 롱 우세로 판단합니다.";
  if (side === "SELL") verdict = "그래서 숏 우세로 판단합니다.";

  let entryTxt = "-";
  let targetTxt = "-";
  let stopTxt = "-";
  let fibMeaning = "";
  let readGuide = "읽는 법: 피보 기준값을 우선 참고하세요.";
  let fibLevelsTxt = "-";
  let fibZonesTxt = "-";
  let entryLo = NaN;
  let entryHi = NaN;
  let targetLo = NaN;
  let targetHi = NaN;
  let target2Lo = NaN;
  let target2Hi = NaN;
  let stopPx = NaN;
  let reactionPass = side !== "BUY";
  let reversalCandidate = false;
  let reversalReady = false;
  let reversalOverride = false;
  const lowVolumeBlock = String(latestVolumeStates?.["5m"] || latestVolumeStates?.single || "").includes("평균 이하");
  const scoreThresholdBase = 70;
  const scoreThresholdAggressive = 58;

  if (fibPlan && Number.isFinite(close)) {
    const buyP = Number(fibPlan.buyPrice);
    const sellP = Number(fibPlan.sellPrice);
    const safeClose = Number(close);
    const p0 = fibPriceOf(fibPlan, 0.0);
    const p0236 = fibPriceOf(fibPlan, 0.236);
    const p0382 = fibPriceOf(fibPlan, 0.382);
    const p05 = fibPriceOf(fibPlan, 0.5);
    const p0618 = fibPriceOf(fibPlan, 0.618);
    const p0786 = fibPriceOf(fibPlan, 0.786);
    const levelParts = [];
    if (Number.isFinite(p0236)) levelParts.push(`0.236: ${formatPrice(p0236)} USDT`);
    if (Number.isFinite(p0382)) levelParts.push(`0.382: ${formatPrice(p0382)} USDT`);
    if (Number.isFinite(p05)) levelParts.push(`0.500: ${formatPrice(p05)} USDT`);
    if (Number.isFinite(p0618)) levelParts.push(`0.618: ${formatPrice(p0618)} USDT`);
    if (Number.isFinite(p0786)) levelParts.push(`0.786: ${formatPrice(p0786)} USDT`);
    fibLevelsTxt = levelParts.length ? levelParts.join("\n") : "-";
    const swingTxt = fibPlan.isUpMove ? "상승 스윙" : "하락 스윙";
    if (entryTableSwingEl) {
      if (fibPlan.isUpMove) {
        entryTableSwingEl.textContent = "상승 추세 스윙 (L→H)";
        setSwingBadge("up", "스윙 상태: 상승 추세");
      } else if (sideForPlan === "BUY") {
        entryTableSwingEl.textContent = "하락 후 반등 시도 (역추세)";
        setSwingBadge("neutral", "스윙 상태: 하락 후 반등");
      } else {
        entryTableSwingEl.textContent = "하락 스윙 지속 (H→L)";
        setSwingBadge("down", "스윙 상태: 하락 지속");
      }
    }

    if (sideForPlan === "BUY") {
      if (fibPlan.isUpMove) {
        const entry = Number.isFinite(buyP) ? buyP : p0618;
        entryLo = Math.min(p05, p0618);
        entryHi = Math.max(p05, p0618);
        const stopFloor = entryLo * (1 - stopGapPct);
        const stop = Number.isFinite(p0786) ? Math.min(p0786, stopFloor) : stopFloor;
        const tp1 = Number.isFinite(sellP) && sellP > entryHi * (1 + tp1GapPct) ? { lo: sellP, hi: sellP } : normalizeTpZone(entryHi, p0236, p0382, tp1GapPct);
        const tp2 = normalizeTpZone(entryHi, p0, p0236, tp2GapPct);
        entryTxt = fmtOne(entry);
        targetTxt = buildTpText(tp1.lo, tp1.hi, tp2.lo, tp2.hi);
        stopTxt = fmtOne(stop);
        targetLo = tp1.lo;
        targetHi = tp1.hi;
        target2Lo = tp2.lo;
        target2Hi = tp2.hi;
        stopPx = stop;
        readGuide = "읽는 법: 상승 스윙은 되돌림(0.5~0.618)에서 진입, 상단(0.236~0.0)에서 정리합니다.";
        fibMeaning = `${swingTxt} 기준(되돌림 매수 -> 상단 정리)`;
        fibZonesTxt = fmtFibZones(fmtZoneSorted(p05, p0618), fmtZoneSorted(p0, p0236));
        if (entryTableScenarioEl) entryTableScenarioEl.textContent = "상승 추세에서 눌림 매수 후 저항 구간 분할 정리 시나리오";
      } else {
        const entry = Number.isFinite(p0236) ? p0236 : Number.isFinite(buyP) ? buyP : safeClose;
        entryLo = Math.min(p0, p0236);
        entryHi = Math.max(p0, p0236);
        const stopFloor = entryLo * (1 - stopGapPct);
        const stop = Number.isFinite(p0) ? Math.min(p0 * 0.997, stopFloor) : stopFloor;
        const tp1 = normalizeTpZone(entryHi, p0236, p0382, tp1GapPct);
        const tp2 = normalizeTpZone(entryHi, p0382, p05, tp2GapPct);
        entryTxt = fmtOne(entry);
        targetTxt = buildTpText(tp1.lo, tp1.hi, tp2.lo, tp2.hi);
        stopTxt = fmtOne(stop);
        targetLo = tp1.lo;
        targetHi = tp1.hi;
        target2Lo = tp2.lo;
        target2Hi = tp2.hi;
        stopPx = stop;
        readGuide = "읽는 법: 하락 스윙에서 매수는 역추세 시도이므로 보수적으로 짧게 대응합니다.";
        fibMeaning = `${swingTxt} 기준(저점권 반등 시도)`;
        fibZonesTxt = fmtFibZones(fmtZoneSorted(p0, p0236), fmtZoneSorted(p0236, p0382));
        if (entryTableScenarioEl) entryTableScenarioEl.textContent = "하락 이후 기술적 반등만 짧게 노리는 보수 시나리오";
      }
    } else if (sideForPlan === "SELL") {
      const entryA = Number.isFinite(p05) ? p05 : Number.isFinite(p0382) ? p0382 : Number.isFinite(p0236) ? p0236 : safeClose;
      const entryB = Number.isFinite(p0618) ? p0618 : Number.isFinite(p05) ? p05 : entryA * (1 + tp1GapPct);
      entryLo = Math.min(entryA, entryB);
      entryHi = Math.max(entryA, entryB);
      const t1Low = Number.isFinite(p0) ? p0 : entryLo * (1 - tp1GapPct * 1.2);
      const t1High = Number.isFinite(p0236) ? p0236 : entryLo * (1 - tp1GapPct);
      const tp1 = normalizeShortTpZone(entryLo, t1Low, t1High, tp1GapPct);
      const t2Low = Number.isFinite(p0) ? p0 * (1 - tp2GapPct) : entryLo * (1 - tp2GapPct * 1.5);
      const t2High = Number.isFinite(p0) ? p0 : entryLo * (1 - tp1GapPct * 1.2);
      const tp2 = normalizeShortTpZone(entryLo, t2Low, t2High, tp2GapPct);
      const stopCeil = entryHi * (1 + stopGapPct);
      const stop = Number.isFinite(p0786) ? Math.max(p0786, stopCeil) : stopCeil;
      targetLo = tp1.lo;
      targetHi = tp1.hi;
      target2Lo = tp2.lo;
      target2Hi = tp2.hi;
      stopPx = stop;
      entryTxt = fmtZoneSorted(entryLo, entryHi);
      targetTxt = buildTpText(tp1.lo, tp1.hi, tp2.lo, tp2.hi);
      stopTxt = fmtOne(stop);
      readGuide = fibPlan.isUpMove
        ? "읽는 법: 상승 스윙에서 숏은 역추세 시도이므로 보수적으로 짧게 대응합니다."
        : "읽는 법: 하락 스윙 지속 숏은 되돌림(0.5~0.618)에서 진입 후 저점권(0.236~0.0)에서 분할 정리합니다.";
      fibMeaning = fibPlan.isUpMove ? `${swingTxt} 기준(상단 저항 역추세 숏 시도)` : `${swingTxt} 기준(되돌림 숏 -> 저점 정리)`;
      fibZonesTxt = fmtFibZones(fmtZoneSorted(entryLo, entryHi), fmtZoneSorted(tp1.lo, tp1.hi));
      if (entryTableScenarioEl) {
        entryTableScenarioEl.textContent = fibPlan.isUpMove
          ? "상승 스윙에서 저항 반응을 이용한 역추세 숏 시나리오"
          : "하락 스윙 지속에서 되돌림 구간 숏 진입 후 저점 분할 정리 시나리오";
      }
    } else {
      if (fibPlan.isUpMove) {
        entryLo = Math.min(p05, p0618);
        entryHi = Math.max(p05, p0618);
        const stopFloor = entryLo * (1 - stopGapPct);
        const stop = Number.isFinite(p0786) ? Math.min(p0786, stopFloor) : stopFloor;
        const tp1 = normalizeTpZone(entryHi, p0236, p0382, tp1GapPct);
        const tp2 = normalizeTpZone(entryHi, p0, p0236, tp2GapPct);
        targetLo = tp1.lo;
        targetHi = tp1.hi;
        target2Lo = tp2.lo;
        target2Hi = tp2.hi;
        stopPx = stop;
        entryTxt = fmtZoneSorted(entryLo, entryHi);
        targetTxt = buildTpText(tp1.lo, tp1.hi, tp2.lo, tp2.hi);
        stopTxt = fmtOne(stop);
        fibMeaning = `${swingTxt} 관망(되돌림 반응 확인 단계)`;
        fibZonesTxt = fmtFibZones(fmtZoneSorted(p05, p0618), fmtZoneSorted(p0, p0236));
        if (entryTableScenarioEl) entryTableScenarioEl.textContent = "상승 추세 유지 여부를 확인하는 관망 시나리오 (진입/익절/손절 참고값 표시)";
      } else {
        entryLo = Math.min(p0, p0236);
        entryHi = Math.max(p0, p0236);
        const stopFloor = entryLo * (1 - stopGapPct);
        const stop = Number.isFinite(p0) ? Math.min(p0 * 0.997, stopFloor) : stopFloor;
        const tp1 = normalizeTpZone(entryHi, p0236, p0382, tp1GapPct);
        const tp2 = normalizeTpZone(entryHi, p0382, p05, tp2GapPct);
        targetLo = tp1.lo;
        targetHi = tp1.hi;
        target2Lo = tp2.lo;
        target2Hi = tp2.hi;
        stopPx = stop;
        entryTxt = fmtZoneSorted(entryLo, entryHi);
        targetTxt = buildTpText(tp1.lo, tp1.hi, tp2.lo, tp2.hi);
        stopTxt = fmtOne(stop);
        fibMeaning = `${swingTxt} 관망(저점권 반등 신호 확인 단계)`;
        fibZonesTxt = fmtFibZones(fmtZoneSorted(p0, p0236), fmtZoneSorted(p0236, p0382));
        if (entryTableScenarioEl) entryTableScenarioEl.textContent = "하락 추세에서 반등 신호 확인 전까지 관망하는 시나리오 (진입/익절/손절 참고값 표시)";
      }
      readGuide = "읽는 법: 관망은 구간 확인 단계이며, 확률/신뢰도 재확인 후 진입합니다.";
    }
  }

  if (fibPlan && !fibPlan.isUpMove && Number.isFinite(entryLo) && Number.isFinite(entryHi) && Number.isFinite(close)) {
    reversalCandidate = buy > sell && conf >= Math.max(params.confFloor, 0.26);
    const reversalReaction = hasEntryReaction("BUY", entryLo, entryHi, latestDecisionKlines);
    reversalReady = reversalCandidate && reversalReaction && close >= entryHi;
    if (swingConflict && reversalReady) {
      side = "BUY";
      swingConflict = false;
      reversalOverride = true;
      reactionPass = true;
      actionBadgeEl.textContent = "반전 롱 후보";
      actionBadgeEl.className = "action-badge buy";
      actionTitleEl.textContent = "하락 스윙 바닥권에서 반전 롱 조건이 확인됐습니다.";
      actionSubtitleEl.textContent = "0.0~0.236 터치 후 0.236 위 종가 회복 + 롱 우위 + 신뢰도 조건 충족";
    }
  }
  sideForPlan = side;
  setTradeLabels(sideForPlan);

  entryZoneEl.textContent = entryTxt;
  targetZoneEl.textContent = targetTxt;
  stopZoneEl.textContent = stopTxt;
  if (entryFibLevelsEl) entryFibLevelsEl.textContent = fibLevelsTxt;
  if (entryFibZonesEl) entryFibZonesEl.textContent = fibZonesTxt;
  const isHighVol = regime === "HIGH_VOL";
  const swingBiasTxt =
    hasFibSwing && Math.abs(swingBias) > 0 ? `${swingBias > 0 ? "+" : ""}${swingBias.toFixed(1)}%p` : "없음";
  const passRegime = !isHighVol || conf >= params.passRegimeConf || Math.abs(diff) >= params.passRegimeDiff;
  const fibTol = Number.isFinite(close) ? close * params.fibTolPct : 0;
  const passFib =
    Number.isFinite(entryLo) && Number.isFinite(entryHi) && Number.isFinite(close)
      ? sideForPlan === "SELL"
        ? close >= entryLo - fibTol && close <= entryHi + fibTol * 1.6
        : close >= entryLo - fibTol * 1.2 && close <= entryHi + fibTol
      : false;
  if (side === "BUY") {
    reactionPass = hasEntryReaction("BUY", entryLo, entryHi, latestDecisionKlines);
  }
  let scoreSide = side;
  if (reversalReady && scoreSide === "WAIT") scoreSide = "BUY";
  const scoreSwingConflict = swingConflict && !reversalReady;
  const signalScore = calcSignalScore({
    rawDiff,
    conf,
    regime,
    side: scoreSide,
    lowVolumeBlock,
    swingConflict: scoreSwingConflict,
    reversalReady,
    whaleScore: flowScore,
    flowWeight,
  });
  const passSignalBase = signalScore.total >= scoreThresholdBase;
  const passSignalAgg = signalScore.total >= scoreThresholdAggressive;
  const passSignal = passSignalBase || passSignalAgg;
  setStepStatus(stepProbStatusEl, passSignal);
  if (stepProbDetailEl) {
    const sideForDisplay = side;
    const sidePct = sideForDisplay === "SELL" ? sell : buy;
    const sideLabel = sideForDisplay === "SELL" ? "숏" : "롱";
    if (swingConflict && !reversalOverride) {
      stepProbDetailEl.textContent = `확률: 롱 ${buy.toFixed(2)}% / 숏 ${sell.toFixed(2)}% | 신뢰도 ${(conf * 100).toFixed(
        1
      )}%\n상태: 스윙 정방향 불일치(하락 스윙 예외 미충족)`;
    } else {
      const lines = [
        `방향: ${sideLabel} ${sidePct.toFixed(2)}% (보정 ${swingBiasTxt}) | 신뢰도 ${(conf * 100).toFixed(1)}% | 레짐: ${regimeTxt}`,
        `점수: ${signalScore.total.toFixed(1)} / 100 (기본 ${scoreThresholdBase}, 공격 ${scoreThresholdAggressive})`,
        `구성: 우위 ${signalScore.edge.toFixed(1)} · 신뢰 ${signalScore.conf.toFixed(1)} · 레짐 ${signalScore.regime.toFixed(
          1
        )} · 방향 ${signalScore.side.toFixed(1)} · 피보 ${signalScore.fib.toFixed(1)} · 모멘텀 ${signalScore.momentum.toFixed(1)} · 플로우 ${signalScore.flow.toFixed(1)}`,
        `고래/플로우: ${whaleLabel}${Number.isFinite(flowScore) ? ` (${flowScore >= 0 ? "+" : ""}${flowScore.toFixed(2)})` : ""} · 가중치 ${(flowWeight * 100).toFixed(0)}%`,
        "규칙: 공격모드는 기준점수만 완화, 안전필터(담보/일손실/손절무효)는 동일 적용",
      ];
      if (!passSignal) lines.push("미통과: 신호 점수 부족");
      if (passSignalAgg && !passSignalBase) lines.push("참고: 공격모드 기준점수(58)에서만 통과");
      if (reversalReady) lines.push("하락 스윙 예외 충족: 바닥(0.0~0.236) 터치 후 종가 회복 + 롱우위 + 신뢰도");
      stepProbDetailEl.textContent = lines.join("\n");
    }
  }
  setStepStatus(stepFibStatusEl, passFib);
  if (stepFibDetailEl) {
    stepFibDetailEl.textContent =
      Number.isFinite(entryLo) && Number.isFinite(entryHi)
        ? `현재가: ${formatPrice(close)}\n피보 진입구간: ${formatPrice(entryLo)} ~ ${formatPrice(entryHi)} USDT\n허용오차: ±${formatPrice(
            fibTol
          )}`
        : "피보 진입구간 계산 대기";
  }
  const entryMid = zoneMid(entryLo, entryHi);
  const targetMid = zoneMid(targetLo, targetHi);
  const rr =
    sideForPlan === "SELL"
      ? Number.isFinite(entryMid) && Number.isFinite(targetMid) && Number.isFinite(stopPx) && stopPx > entryMid
        ? (entryMid - targetMid) / (stopPx - entryMid)
        : NaN
      : Number.isFinite(entryMid) && Number.isFinite(targetMid) && Number.isFinite(stopPx) && entryMid > stopPx
        ? (targetMid - entryMid) / (entryMid - stopPx)
        : NaN;
  const minTpPass = Math.max(0.01, tp1GapPct * 0.55);
  const minStopPass = Math.max(0.006, stopGapPct * 0.7);
  const passPlan =
    Number.isFinite(entryMid) &&
    Number.isFinite(targetMid) &&
    Number.isFinite(stopPx) &&
    (sideForPlan === "SELL"
      ? targetMid < entryMid * (1 - minTpPass) && stopPx > entryMid * (1 + minStopPass)
      : targetMid > entryMid * (1 + minTpPass) && stopPx < entryMid * (1 - minStopPass)) &&
      (!Number.isFinite(rr) || rr >= 0.9);
  const passExecBase = passSignalBase && passRegime && passFib && passPlan && reactionPass && !lowVolumeBlock;
  const passExecAgg = passSignalAgg && passRegime && passFib && passPlan && reactionPass && !lowVolumeBlock;
  const passExec = passExecBase || passExecAgg;
  if (entryPlanMetricsEl) {
    if (Number.isFinite(entryMid) && Number.isFinite(targetMid) && Number.isFinite(stopPx) && entryMid > 0) {
      const maxTpMid = zoneMid(target2Lo, target2Hi);
      if (sideForPlan === "SELL") {
        const downPct = targetMid > 0 ? ((entryMid / targetMid) - 1) * 100 : NaN;
        const maxDownPct = Number.isFinite(maxTpMid) && maxTpMid > 0 ? ((entryMid / maxTpMid) - 1) * 100 : downPct;
        const upRiskPct = ((stopPx / entryMid) - 1) * 100;
        entryPlanMetricsEl.textContent = `예상 수익 : +${(Number.isFinite(downPct) ? downPct : 0).toFixed(2)}%\n예상 손실 : -${upRiskPct.toFixed(2)}%\nPR : ${
          Number.isFinite(rr) ? rr.toFixed(2) : "-"
        }\n최대 예상 수익 : +${(Number.isFinite(maxDownPct) ? maxDownPct : 0).toFixed(2)}% (2차)`;
      } else {
        const upPct = ((targetMid / entryMid) - 1) * 100;
        const maxUpPct = Number.isFinite(maxTpMid) ? ((maxTpMid / entryMid) - 1) * 100 : upPct;
        const downPct = ((entryMid / stopPx) - 1) * 100;
        entryPlanMetricsEl.textContent = `예상 수익 : +${upPct.toFixed(2)}%\n예상 손실 : -${downPct.toFixed(2)}%\nPR : ${
          Number.isFinite(rr) ? rr.toFixed(2) : "-"
        }\n최대 예상 수익 : +${maxUpPct.toFixed(2)}% (2차)`;
      }
    } else {
      entryPlanMetricsEl.textContent = "-";
    }
  }
  setStepStatus(stepExecStatusEl, passExec);
  if (stepExecDetailEl) {
    const tpExecTxt = String(targetTxt || "-")
      .replace("1차 익절:", "1차 익절 :")
      .replace("2차 익절:", "2차 익절 :");
    if (passExec) {
      const modeHint = passExecBase ? "기본/공격 공통 통과" : "공격모드 점수 기준 통과";
      stepExecDetailEl.textContent = `진입: 피보 진입구간(하단~상단) 터치 시 체결\n진입 기준가: ${entryTxt}\n${tpExecTxt}\n손절가: ${stopTxt}\n실행모드: ${modeHint}`;
    } else if (lowVolumeBlock) {
      stepExecDetailEl.textContent = "실행 대기\n사유: 모멘텀 약화(거래량 평균 이하)";
    } else if (!reactionPass) {
      stepExecDetailEl.textContent = reversalCandidate
        ? "실행 대기\n사유: 하락 스윙 예외 확인봉 조건 미충족"
        : "실행 대기\n사유: 진입구간 터치 후 확인봉 조건 미충족";
    } else if (!passPlan) {
      stepExecDetailEl.textContent = "실행 대기\n사유: 안전필터 미충족(RR/거리)";
    } else {
      stepExecDetailEl.textContent = "실행 대기\n사유: 신호점수 또는 피보 진입구간 조건 미충족";
    }
  }
  if (decisionFinalEl) {
    if (swingConflict && !reversalOverride) {
      if (stepFinalStatusEl) setStepStatus(stepFinalStatusEl, false);
      decisionFinalEl.textContent = "하락 스윙 예외 미충족으로 진입 보류";
    } else if (passExec && side === "BUY") {
      if (stepFinalStatusEl) setStepStatus(stepFinalStatusEl, true);
      decisionFinalEl.textContent = reversalOverride
        ? "하락 스윙 예외 충족: 반전 롱 실행 가능 구간"
        : "피보 진입구간 터치 시 분할 진입 가능 구간";
    } else if (side === "SELL" && passSignal && passRegime) {
      if (stepFinalStatusEl) setStepStatus(stepFinalStatusEl, true);
      decisionFinalEl.textContent = "숏 우세: 피보 진입구간 터치 시 분할 진입 가능 구간";
    } else if (passSignal && passRegime && !passFib) {
      if (stepFinalStatusEl) setStepStatus(stepFinalStatusEl, false);
      decisionFinalEl.textContent =
        side === "SELL"
          ? "숏 우세: 피보 진입구간(하단~상단) 도달 대기"
          : "롱 우위: 피보 진입구간(하단~상단) 도달 대기";
    } else if (passSignalAgg && !passSignalBase) {
      if (stepFinalStatusEl) setStepStatus(stepFinalStatusEl, false);
      decisionFinalEl.textContent = "공격모드 기준점수(58) 통과, 기본모드 기준점수(70)는 대기";
    } else if (side === "SELL") {
      if (stepFinalStatusEl) setStepStatus(stepFinalStatusEl, false);
      decisionFinalEl.textContent = "숏 우세: 진입 조건 재확인 후 대기";
    } else {
      if (stepFinalStatusEl) setStepStatus(stepFinalStatusEl, false);
      decisionFinalEl.textContent = "관망 후 다음 시그널 확인";
    }
  }
  if (positionStateEl) {
    if (reversalOverride) positionStateEl.textContent = "하락추세 반등 롱";
    else if (side === "BUY") positionStateEl.textContent = "롱";
    else if (side === "SELL") positionStateEl.textContent = "숏";
    else positionStateEl.textContent = "관망";
  }
  if (actionReadGuideEl) actionReadGuideEl.textContent = readGuide;
  if (entryTableSwingEl && !fibPlan) entryTableSwingEl.textContent = "-";
  if (entrySwingBadgeEl && !fibPlan) setSwingBadge("", "스윙 상태: -");
  if (entryTableScenarioEl && !fibPlan) entryTableScenarioEl.textContent = "-";
  if (entryFibLevelsEl && !fibPlan) entryFibLevelsEl.textContent = "-";
  if (entryFibZonesEl && !fibPlan) entryFibZonesEl.textContent = "-";
  if (entryPlanMetricsEl && !fibPlan) entryPlanMetricsEl.textContent = "-";
  if (stepProbStatusEl && !fibPlan) {
    stepProbStatusEl.textContent = "-";
    stepProbStatusEl.classList.remove("is-pass", "is-wait");
  }
  if (stepProbDetailEl && !fibPlan) stepProbDetailEl.textContent = "분석 데이터 대기";
  if (stepFibStatusEl && !fibPlan) {
    stepFibStatusEl.textContent = "-";
    stepFibStatusEl.classList.remove("is-pass", "is-wait");
  }
  if (stepFibDetailEl && !fibPlan) stepFibDetailEl.textContent = "피보 구간 계산 대기";
  if (stepExecStatusEl && !fibPlan) {
    stepExecStatusEl.textContent = "-";
    stepExecStatusEl.classList.remove("is-pass", "is-wait");
  }
  if (stepExecDetailEl && !fibPlan) stepExecDetailEl.textContent = "실행 판단 대기";
  if (stepFinalStatusEl && !fibPlan) {
    stepFinalStatusEl.textContent = "-";
    stepFinalStatusEl.classList.remove("is-pass", "is-wait");
  }
  if (decisionFinalEl && !fibPlan) decisionFinalEl.textContent = "데이터 대기";
  if (passCheckSummaryEl && !fibPlan) setPassCheckUI("pending");
  if (actionExplainEl) {
    const setupTxt = reversalOverride
      ? "반전 바닥 롱"
      : side === "BUY"
          ? "추세/반등 롱"
          : side === "SELL"
            ? "숏 우세"
            : "관망";
    actionExplainEl.textContent =
      `[현재 값]\n` +
      `- 롱 ${buy.toFixed(2)}% / 숏 ${sell.toFixed(2)}% (원차이 ${rawDiff >= 0 ? "+" : ""}${rawDiff.toFixed(2)}%p, 피보보정 ${
        swingBias >= 0 ? "+" : ""
      }${swingBias.toFixed(2)}%p, 반영차이 ${diff >= 0 ? "+" : ""}${diff.toFixed(2)}%p)\n` +
      `- 신뢰도 ${(conf * 100).toFixed(1)}% (지표 일치도)\n` +
      `- 레짐 ${regimeTxt} (${regime || "-"})\n` +
      `- 고래심리 ${whaleLabel}${Number.isFinite(flowScore) ? ` (${flowScore >= 0 ? "+" : ""}${flowScore.toFixed(2)})` : ""}\n` +
      `- 신호점수 ${signalScore.total.toFixed(1)}점 (기본 ${scoreThresholdBase} / 공격 ${scoreThresholdAggressive})\n` +
      `- 점수구성 우위 ${signalScore.edge.toFixed(1)} / 신뢰 ${signalScore.conf.toFixed(1)} / 레짐 ${signalScore.regime.toFixed(
        1
      )} / 방향 ${signalScore.side.toFixed(1)} / 피보 ${signalScore.fib.toFixed(1)} / 모멘텀 ${signalScore.momentum.toFixed(1)} / 플로우 ${signalScore.flow.toFixed(1)}(가중 ${(flowWeight * 100).toFixed(0)}%)\n` +
      `- 셋업 ${setupTxt}\n\n` +
      `[왜 이렇게 나왔나]\n` +
      `- 주요 반영값: ${topTxt}\n` +
      `- 피보 요약: ${fibMeaning || "-"}\n\n` +
      `[결론]\n` +
      `- ${verdict}`;
  }
}

function renderAnalysisOnly(analysis, analysisMode, market) {
  buyPctEl.textContent = `${analysis.buy_pct.toFixed(2)}%`;
  sellPctEl.textContent = `${analysis.sell_pct.toFixed(2)}%`;
  confEl.textContent = `${(analysis.confidence * 100).toFixed(1)}%`;
  regimeEl.textContent = analysis.regime || "-";
  const modeTxt = analysisMode === "mtf" ? "MTF" : "Single";
  const chartTf = chartIntervalEl ? chartIntervalEl.value : "-";
  asofEl.textContent = `${analysis.symbol} / ${analysis.market || market} / 분석:${analysis.interval} / 차트:${chartTf} / ${modeTxt} · 기준: ${fmtTs(
    analysis.asof_open_time_ms
  )} · 종가: ${analysis.close}`;
  renderExplainDetail(analysis, analysisMode);
  renderActionSummary(analysis, lastFibPlan);
}

function calcVolumeState(klines) {
  const rows = Array.isArray(klines) ? klines : [];
  if (rows.length < 22) return null;
  const vols = rows.map((k) => Number(k?.volume)).filter((v) => Number.isFinite(v) && v >= 0);
  if (vols.length < 22) return null;
  // 마지막 봉은 진행 중일 수 있어 직전 마감봉(-2)을 현재 거래량으로 사용
  const curIdx = vols.length - 2;
  const cur = vols[curIdx];
  const prev20 = vols.slice(Math.max(0, curIdx - 20), curIdx);
  const avg20 = prev20.reduce((a, b) => a + b, 0) / Math.max(1, prev20.length);
  if (!Number.isFinite(cur) || !Number.isFinite(avg20) || avg20 <= 0) return null;
  const ratio = cur / avg20;
  const state = ratio >= 1.15 ? "평균 이상" : ratio < 0.85 ? "평균 이하" : "평균수준";
  return { ratio, state };
}

function renderVolumeState(payload, analysisMode, analysisInterval) {
  if (!volumeStateEl) return;
  latestVolumeStates = {};
  if (!payload) {
    volumeStateEl.textContent = "데이터 대기";
    return;
  }
  if (analysisMode === "mtf") {
    const tfs = ["4h", "1h", "5m"];
    const lines = [];
    for (const tf of tfs) {
      const item = calcVolumeState(payload?.[tf]);
      if (!item) continue;
      lines.push(`${tf} ${item.state}`);
      latestVolumeStates[tf] = item.state;
    }
    volumeStateEl.textContent = lines.length ? lines.join("\n") : "데이터 대기";
    return;
  }
  const one = calcVolumeState(payload?.single);
  latestVolumeStates = { single: one ? one.state : "" };
  volumeStateEl.textContent = one ? `${analysisInterval} ${one.state}` : "데이터 대기";
}

function connectAnalysisStream() {
  if (document.hidden) return;
  const interval = normalizeAnalysisInterval(analysisIntervalEl ? analysisIntervalEl.value : "5m");
  const mode = analysisModeEl ? analysisModeEl.value : "single";
  const nextKey =
    mode === "mtf"
      ? `${selectedMarket}:${selectedSymbol}:mtf`
      : `${selectedMarket}:${selectedSymbol}:${interval}:single`;
  if (analysisWs && analysisStreamKey === nextKey) return;
  analysisStreamKey = nextKey;

  stopAnalysisStream();

  const wsProto = location.protocol === "https:" ? "wss" : "ws";
  const urlBase = `${wsProto}://${location.host}/ws/analysis?symbol=${encodeURIComponent(
    selectedSymbol
  )}&market=${encodeURIComponent(selectedMarket)}&mode=${encodeURIComponent(mode)}&limit=500`;
  const url = mode === "mtf" ? urlBase : `${urlBase}&interval=${encodeURIComponent(interval)}`;
  const ws = new WebSocket(url);
  analysisWs = ws;
  ws.onmessage = (ev) => {
    try {
      const raw = JSON.parse(ev.data);
      if (raw?.error) return;
      if (mode === "mtf") {
        mtfAnalysisCache = raw;
        mtfAnalysisCacheKey = `${selectedMarket}:${selectedSymbol}`;
      }
      const analysisKey =
        mode === "mtf"
          ? `${selectedMarket}:${selectedSymbol}:mtf`
          : `${selectedMarket}:${selectedSymbol}:${interval}:single`;
      analysisSnapshotRaw = raw;
      analysisSnapshotKey = analysisKey;
      const viewInterval = mode === "mtf" ? (chartIntervalEl ? chartIntervalEl.value : "5m") : interval;
      const analysis = toViewAnalysis(raw, mode, viewInterval, selectedMarket);
      renderAnalysisOnly(analysis, mode, selectedMarket);
    } catch (_) {}
  };
  ws.onclose = (ev) => {
    if (analysisWs !== ws) return;
    analysisWs = null;
    if (Number(ev?.code) === 4401) {
      window.location.href = "/static/auth.html";
      return;
    }
    if (document.hidden) return;
    analysisWsReconnectTimer = setTimeout(() => connectAnalysisStream(), 2000);
  };
}

function ensureChart() {
  if (!chartContainer) return;
  if (chart) return;
  chart = LightweightCharts.createChart(chartContainer, {
    layout: {
      background: { color: "rgba(0,0,0,0)" },
      textColor: "rgba(232,238,252,0.85)",
    },
    grid: {
      vertLines: { color: "rgba(232,238,252,0.08)" },
      horzLines: { color: "rgba(232,238,252,0.08)" },
    },
    timeScale: { timeVisible: true, secondsVisible: false },
    rightPriceScale: { borderColor: "rgba(232,238,252,0.14)" },
  });
  candleSeries = chart.addCandlestickSeries({
    upColor: "rgba(110,231,255,0.9)",
    downColor: "rgba(255,99,132,0.9)",
    borderDownColor: "rgba(255,99,132,0.9)",
    borderUpColor: "rgba(110,231,255,0.9)",
    wickDownColor: "rgba(255,99,132,0.9)",
    wickUpColor: "rgba(110,231,255,0.9)",
    priceLineVisible: true,
    lastValueVisible: true,
    priceLineColor: "rgba(255, 196, 0, 0.9)",
  });
  volumeSeries = chart.addHistogramSeries({
    priceScaleId: "volume",
    priceFormat: { type: "volume" },
    lastValueVisible: false,
    priceLineVisible: false,
  });
  chart.priceScale("volume").applyOptions({
    scaleMargins: { top: 0.78, bottom: 0.0 },
  });
}

function resizeChartToContainer() {
  if (!chart || !chartContainer) return;
  const nextW = Math.max(0, Math.floor(chartContainer.clientWidth || 0));
  const nextH = Math.max(0, Math.floor(chartContainer.clientHeight || 0));
  if (nextW <= 0 || nextH <= 0) return;
  chart.resize(nextW, nextH);
}

function initChartResizeObserver() {
  if (!chartContainer) return;
  if (chartResizeObserver) {
    try {
      chartResizeObserver.disconnect();
    } catch (_) {}
    chartResizeObserver = null;
  }
  window.addEventListener("resize", () => resizeChartToContainer(), { passive: true });
  if ("ResizeObserver" in window) {
    chartResizeObserver = new ResizeObserver(() => resizeChartToContainer());
    chartResizeObserver.observe(chartContainer);
  }
}

function clearFibLines() {
  if (!candleSeries) return;
  for (const line of fibLines) {
    try {
      candleSeries.removePriceLine(line);
    } catch (_) {}
  }
  fibLines = [];
}

function clearAvwapSeries() {
  if (!chart) return;
  for (const s of avwapSeriesList) {
    try {
      chart.removeSeries(s);
    } catch (_) {}
  }
  avwapSeriesList = [];
}

function clearVpLines() {
  if (!candleSeries) return;
  for (const line of vpLines) {
    try {
      candleSeries.removePriceLine(line);
    } catch (_) {}
  }
  vpLines = [];
}

function renderFibPrices(fibPlan) {
  if (!fibPlan || !Number.isFinite(fibPlan.buyPrice) || !Number.isFinite(fibPlan.sellPrice)) {
    return;
  }

  const p = (r) => {
    const v = fibPlan?.prices?.find((x) => Number(x.r) === Number(r))?.price;
    return Number(v);
  };
  const p0 = p(0.0);
  const p0236 = p(0.236);
  const p0382 = p(0.382);
  const p05 = p(0.5);
  const p0618 = p(0.618);
  const p0786 = p(0.786);

  // 피보 구조표 UI를 제거했으므로 여기서는 계산만 유지(다른 로직 재사용용).
}

function findPivotPoints(klines, left = 3, right = 3) {
  if (!Array.isArray(klines) || klines.length < left + right + 1) return [];
  const pivots = [];
  for (let i = left; i < klines.length - right; i++) {
    const h = Number(klines[i].high);
    const l = Number(klines[i].low);
    if (!Number.isFinite(h) || !Number.isFinite(l)) continue;

    let isHigh = true;
    let isLow = true;
    for (let j = i - left; j <= i + right; j++) {
      if (j === i) continue;
      const hj = Number(klines[j].high);
      const lj = Number(klines[j].low);
      if (!Number.isFinite(hj) || !Number.isFinite(lj)) continue;
      if (hj >= h) isHigh = false;
      if (lj <= l) isLow = false;
      if (!isHigh && !isLow) break;
    }

    if (isHigh) {
      pivots.push({ idx: i, kind: "H", price: h });
    } else if (isLow) {
      pivots.push({ idx: i, kind: "L", price: l });
    }
  }
  return pivots;
}

function pickLatestSwingFromPivots(pivots) {
  if (!Array.isArray(pivots) || pivots.length < 2) return null;

  // 마지막 유효한 반대 피벗 쌍(H-L 또는 L-H)을 최근 기준으로 선택
  for (let i = pivots.length - 1; i > 0; i--) {
    const b = pivots[i];
    const a = pivots[i - 1];
    if (!a || !b || a.kind === b.kind) continue;

    const isUpMove = a.kind === "L" && b.kind === "H";
    const lo = Math.min(a.price, b.price);
    const hi = Math.max(a.price, b.price);
    if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) continue;

    return {
      fromIdx: a.idx,
      toIdx: b.idx,
      fromKind: a.kind,
      toKind: b.kind,
      isUpMove,
      lo,
      hi,
    };
  }
  return null;
}

function computeFibPlanFromKlines(klines) {
  if (!Array.isArray(klines) || klines.length < 60) return null;

  const lookback = Math.min(200, klines.length);
  const recent = klines.slice(klines.length - lookback);

  // 권장: 피벗 기반 스윙 선정 (fallback: 단순 고저)
  const pivots = findPivotPoints(recent, 3, 3);
  const swing = pickLatestSwingFromPivots(pivots);

  let hi = NaN;
  let lo = NaN;
  let isUpMove = true;
  let method = "pivot";

  if (swing) {
    hi = swing.hi;
    lo = swing.lo;
    isUpMove = swing.isUpMove;
  } else {
    method = "simple";
    let hiIdx = -1;
    let loIdx = -1;
    hi = -Infinity;
    lo = Infinity;
    for (let i = 0; i < recent.length; i++) {
      const h = Number(recent[i].high);
      const l = Number(recent[i].low);
      if (Number.isFinite(h) && h >= hi) {
        hi = h;
        hiIdx = i;
      }
      if (Number.isFinite(l) && l <= lo) {
        lo = l;
        loIdx = i;
      }
    }
    if (!Number.isFinite(hi) || !Number.isFinite(lo) || hiIdx < 0 || loIdx < 0 || hi === lo) return null;
    isUpMove = loIdx < hiIdx;
  }

  const start = isUpMove ? lo : hi;
  const end = isUpMove ? hi : lo;
  const range = Math.abs(end - start);
  if (range <= 0) return null;

  const levels = [
    { r: 0.0, color: "rgba(232,238,252,0.25)" },
    { r: 0.236, color: "rgba(110,231,255,0.55)" },
    { r: 0.382, color: "rgba(110,231,255,0.55)" },
    { r: 0.5, color: "rgba(232,238,252,0.45)" },
    { r: 0.618, color: "rgba(255, 196, 0, 0.55)" },
    { r: 0.786, color: "rgba(255, 196, 0, 0.55)" },
    { r: 1.0, color: "rgba(232,238,252,0.25)" },
  ];

  const prices = levels.map((lv) => ({
    ...lv,
    price: isUpMove ? end - range * lv.r : end + range * lv.r,
  }));

  // 단순 규칙:
  // 상승 구간이면 0.618 되돌림을 매수 후보, 직전 고점(0.0)을 매도 후보
  // 하락 구간이면 0.618 되돌림을 매도 후보, 직전 저점(0.0)을 매수 후보
  const p618 = prices.find((x) => x.r === 0.618)?.price;
  const p0 = prices.find((x) => x.r === 0.0)?.price;
  const buyPrice = isUpMove ? p618 : p0;
  const sellPrice = isUpMove ? p0 : p618;

  return {
    lookback,
    hi,
    lo,
    isUpMove,
    method,
    prices,
    buyPrice,
    sellPrice,
  };
}

async function getMtfFibPlan(symbol, market, signal) {
  const key = `${market}:${symbol}`;
  if (mtfFibPlanCache && mtfFibPlanCacheKey === key) return mtfFibPlanCache;
  const data = await fetchJSON(
    `/api/klines?symbol=${encodeURIComponent(symbol)}&market=${encodeURIComponent(market)}&interval=5m&limit=500`,
    { signal }
  );
  latestDecisionKlines = Array.isArray(data?.klines) ? data.klines : [];
  const plan = computeFibPlanFromKlines(latestDecisionKlines);
  mtfFibPlanCache = plan;
  mtfFibPlanCacheKey = key;
  return plan;
}

async function getSingleFibPlan(symbol, market, analysisInterval, chartInterval, chartKlines, signal) {
  if (analysisInterval === chartInterval) {
    latestDecisionKlines = Array.isArray(chartKlines) ? chartKlines : [];
    return computeFibPlanFromKlines(latestDecisionKlines);
  }
  const data = await fetchJSON(
    `/api/klines?symbol=${encodeURIComponent(symbol)}&market=${encodeURIComponent(market)}&interval=${encodeURIComponent(
      analysisInterval
    )}&limit=500`,
    { signal }
  );
  latestDecisionKlines = Array.isArray(data?.klines) ? data.klines : [];
  return computeFibPlanFromKlines(latestDecisionKlines);
}

function addVpLines(analysis) {
  const vp = analysis?.levels?.volume_profile;
  if (!vp || !candleSeries) return;

  const addLine = (price, title, color, lineStyle = 0) => {
    if (!Number.isFinite(Number(price))) return;
    const line = candleSeries.createPriceLine({
      price: Number(price),
      color,
      lineWidth: title === "POC" ? 2 : 1,
      lineStyle,
      axisLabelVisible: true,
      title,
    });
    vpLines.push(line);
  };

  addLine(vp.poc, "POC", "rgba(255, 196, 0, 0.95)", 0);
  for (const p of vp.hvn || []) addLine(p, "HVN", "rgba(110,231,255,0.8)", 2);
  for (const p of vp.lvn || []) addLine(p, "LVN", "rgba(255,99,132,0.8)", 2);
}

function buildAvwapSeriesData(klines, anchorTimeMs) {
  if (!Array.isArray(klines) || klines.length === 0) return [];
  const startIdx = klines.findIndex((k) => Number(k.time) === Number(anchorTimeMs));
  if (startIdx < 0) return [];

  let cumPv = 0;
  let cumV = 0;
  const out = [];
  for (let i = startIdx; i < klines.length; i++) {
    const k = klines[i];
    const h = Number(k.high);
    const l = Number(k.low);
    const c = Number(k.close);
    const v = Number(k.volume);
    if (![h, l, c, v].every(Number.isFinite)) continue;
    const tp = (h + l + c) / 3;
    cumPv += tp * v;
    cumV += v;
    if (cumV <= 0) continue;
    out.push({
      time: Math.floor(Number(k.time) / 1000),
      value: cumPv / cumV,
    });
  }
  return out;
}

function addAvwapLines(klines, analysis) {
  if (!chart || !analysis || !analysis.levels || !Array.isArray(analysis.levels.avwap)) return;
  const colors = {
    pivot_low: "rgba(255, 196, 0, 0.95)",
    pivot_high: "rgba(110,231,255,0.95)",
  };
  for (const lv of analysis.levels.avwap) {
    const data = buildAvwapSeriesData(klines, lv.anchor_time_ms);
    if (!data.length) continue;
    const s = chart.addLineSeries({
      color: colors[lv.anchor] || "rgba(232,238,252,0.85)",
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: true,
      crosshairMarkerVisible: false,
      title: `AVWAP(${lv.anchor})`,
    });
    s.setData(data);
    avwapSeriesList.push(s);
  }
}

function addFibLinesFromPlan(fibPlan, interval) {
  if (!candleSeries || !fibPlan || !Array.isArray(fibPlan.prices)) return;
  for (const lv of fibPlan.prices) {
    const line = candleSeries.createPriceLine({
      price: lv.price,
      color: lv.color,
      lineWidth: 1,
      lineStyle: 2, // dashed
      axisLabelVisible: true,
      title: `Fib ${lv.r.toFixed(3)}`,
    });
    fibLines.push(line);
  }

}

function applyFibOverlay() {
  clearFibLines();
  if (!fibToggleEl || !fibToggleEl.checked) return;
  const chartInterval = chartIntervalEl ? chartIntervalEl.value : "5m";
  addFibLinesFromPlan(lastOverlayFibPlan, chartInterval);
}

async function fetchJSON(url, options = {}) {
  const res = await fetch(url, options);
  if (res.status === 401) {
    window.location.href = "/static/auth.html";
    throw new Error("401 unauthorized");
  }
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`${res.status} ${txt}`);
  }
  return res.json();
}

async function loadPassCheck(symbol, market, analysisMode, analysisInterval) {
  if (!passCheckSummaryEl) return;
  const interval = analysisMode === "mtf" ? "5m" : normalizeAnalysisInterval(analysisInterval);
  const periodValue = passCheckPeriodEl ? passCheckPeriodEl.value : "3d";
  const periodCfg = getPassCheckPeriodConfig(interval, periodValue);
  const horizonBars = periodCfg.horizonBars;
  const limitBars = Math.max(700, Math.min(5000, horizonBars + periodCfg.sampleBuffer));
  const key = `${symbol}:${market}:${analysisMode}:${interval}:${periodCfg.period}`;
  const now = Date.now();
  const uiCache = loadPassCheckCache();
  // 코인 전환 시 이전 코인 값이 잠깐 보이는 현상을 막기 위해,
  // 캐시 선노출 없이 DB 응답만 반영한다.
  setPassCheckUI("pending", { modeNote: "DB 통계 불러오는 중..." });
  if (key === passCheckKey && now - passCheckTs < 20000) return;
  if (activePassCheckController) {
    try {
      activePassCheckController.abort();
    } catch (_) {}
  }
  const controller = new AbortController();
  activePassCheckController = controller;
  const reqSeq = ++activePassCheckSeq;
  passCheckKey = key;
  passCheckTs = now;
  try {
    const data = await fetchJSON(
      `/api/pass_check_db?symbol=${encodeURIComponent(symbol)}&market=${encodeURIComponent(market)}&interval=${encodeURIComponent(
        interval
      )}&period=${encodeURIComponent(periodCfg.period)}`,
      { signal: controller.signal }
    );
    if (controller.signal.aborted || reqSeq !== activePassCheckSeq) return;
    if (!data) {
      setPassCheckUI("pending", {
        modeNote: "DB 누적 준비중",
        barsText: "-",
        passesText: "-",
        hitRateText: "-",
      });
      return;
    }
    const executedCnt = Number(data?.executed_count || 0);
    const hitCnt = Number(data?.tp1_hit_count || 0);
    const hitRate = Number(data?.executed_tp1_hit_rate || 0) * 100;
    const bars = Number(data?.bars || 0);
    const firstSignalMs = Number(data?.first_signal_time_ms || 0);
    const latestSignalMs = Number(data?.latest_signal_time_ms || 0);
    const updatedMs = Number(data?.updated_ms || 0);
    // 검증구간은 DB에 적재된 실제 신호 범위(최초~최신)로 표기한다.
    const endMs = latestSignalMs > 0 ? latestSignalMs : updatedMs;
    const startMs = firstSignalMs > 0 ? firstSignalMs : 0;
    const rangeText = startMs > 0 && endMs > 0 ? `${fmtYmd(startMs)} ~ ${fmtYmd(endMs)}` : bars > 0 ? `최근 ${bars}봉 (${interval})` : "-";
    const periodLabel = periodCfg.period === "24h" ? "24시간" : periodCfg.period === "7d" ? "7일" : "3일";
    const modeNote =
      analysisMode === "mtf"
        ? `MTF는 5m 기준 · 진입 후 ${periodLabel} 내 손절 없이 익절 달성만 집계`
        : `${interval} 기준 · 진입 후 ${periodLabel} 내 손절 없이 익절 달성만 집계`;
    const nextPayload = {
      modeNote,
      barsText: rangeText,
      passesText: `${executedCnt}회`,
      hitRate: hitRate,
      hitRateText: `${hitCnt}회 (${hitRate.toFixed(1)}%)`,
    };
    if (Number(data?.pass_count || 0) <= 0) {
      nextPayload.modeNote = `${modeNote} · PASS 사례 없음`;
      nextPayload.passesText = "0회";
      nextPayload.hitRate = 0;
      nextPayload.hitRateText = "0회 (0.0%)";
    }
    setPassCheckUI("success", nextPayload);
    uiCache[key] = nextPayload;
    savePassCheckCache(trimPassCheckCache(uiCache));
  } catch (e) {
    if (e?.name === "AbortError") return;
    setPassCheckUI("error", {
      modeNote: "계산 실패",
      barsText: "-",
      passesText: "-",
      hitRateText: "-",
    });
  } finally {
    if (activePassCheckController === controller) activePassCheckController = null;
  }
}

function schedulePassCheck(symbol, market, analysisMode, analysisInterval) {
  if (passCheckTimer) {
    clearTimeout(passCheckTimer);
    passCheckTimer = null;
  }
  passCheckTimer = setTimeout(() => {
    passCheckTimer = null;
    loadPassCheck(symbol, market, analysisMode, analysisInterval);
  }, 450);
}

async function load(options = {}) {
  const forceAnalysis = Boolean(options?.forceAnalysis);
  if (activeLoadController) {
    try {
      activeLoadController.abort();
    } catch (_) {}
  }
  const controller = new AbortController();
  activeLoadController = controller;
  const loadSeq = ++activeLoadSeq;
  const { signal } = controller;
  isLoadingMain = true;
  try {
    syncAnalysisControls();
    const symbol = selectedSymbol;
    const market = selectedMarket;
    const analysisInterval = normalizeAnalysisInterval(analysisIntervalEl ? analysisIntervalEl.value : "5m");
    const chartInterval = chartIntervalEl ? chartIntervalEl.value : "5m";
    const analysisMode = analysisModeEl ? analysisModeEl.value : "single";
    ensureChart();
    resizeChartToContainer();
    clearFibLines();
    clearAvwapSeries();
    clearVpLines();

    const analysisPath =
      analysisMode === "mtf"
        ? `/api/analysis_trendy_mtf?symbol=${encodeURIComponent(symbol)}&market=${encodeURIComponent(market)}&limit=500`
        : `/api/analysis_trendy?symbol=${encodeURIComponent(symbol)}&market=${encodeURIComponent(
            market
          )}&interval=${encodeURIComponent(analysisInterval)}&limit=500`;

    const klinesPromise = fetchJSON(
      `/api/klines?symbol=${encodeURIComponent(symbol)}&market=${encodeURIComponent(market)}&interval=${encodeURIComponent(
        chartInterval
      )}&limit=500`,
      { signal }
    );
    const volumeStatePromise =
      analysisMode === "mtf"
        ? Promise.all([
            fetchJSON(`/api/klines?symbol=${encodeURIComponent(symbol)}&market=${encodeURIComponent(market)}&interval=4h&limit=200`, { signal }),
            fetchJSON(`/api/klines?symbol=${encodeURIComponent(symbol)}&market=${encodeURIComponent(market)}&interval=1h&limit=200`, { signal }),
            fetchJSON(`/api/klines?symbol=${encodeURIComponent(symbol)}&market=${encodeURIComponent(market)}&interval=5m&limit=200`, { signal }),
          ]).then(([k4, k1, k5]) => ({ "4h": k4?.klines || [], "1h": k1?.klines || [], "5m": k5?.klines || [] }))
        : analysisInterval === chartInterval
          ? klinesPromise.then((k) => ({ single: k?.klines || [] }))
          : fetchJSON(
              `/api/klines?symbol=${encodeURIComponent(symbol)}&market=${encodeURIComponent(market)}&interval=${encodeURIComponent(
                analysisInterval
              )}&limit=200`,
              { signal }
            ).then((k) => ({ single: k?.klines || [] }));

    const mtfKey = `${market}:${symbol}`;
    const analysisKey =
      analysisMode === "mtf" ? `${market}:${symbol}:mtf` : `${market}:${symbol}:${analysisInterval}:single`;
    let analysisRawPromise;
    if (!forceAnalysis && analysisSnapshotRaw && analysisSnapshotKey === analysisKey) {
      analysisRawPromise = Promise.resolve(analysisSnapshotRaw);
    } else if (analysisMode === "mtf" && mtfAnalysisCache && mtfAnalysisCacheKey === mtfKey) {
      analysisRawPromise = Promise.resolve(mtfAnalysisCache);
    } else {
      analysisRawPromise = fetchJSON(analysisPath, { signal });
    }

    const [klines, analysisRaw, volumeStatePayload] = await Promise.all([klinesPromise, analysisRawPromise, volumeStatePromise]);
    if (signal.aborted || loadSeq !== activeLoadSeq) return;
    if (analysisMode === "mtf") {
      mtfAnalysisCache = analysisRaw;
      mtfAnalysisCacheKey = mtfKey;
    }
    analysisSnapshotRaw = analysisRaw;
    analysisSnapshotKey = analysisKey;

    const analysis = toViewAnalysis(analysisRaw, analysisMode, chartInterval, market);
    renderVolumeState(volumeStatePayload, analysisMode, analysisInterval);
    const overlayFibPlan = computeFibPlanFromKlines(klines.klines); // 차트 오버레이용(차트 시간봉 기준)
    lastOverlayFibPlan = overlayFibPlan;
    const decisionFibPlan =
      analysisMode === "mtf"
        ? await getMtfFibPlan(symbol, market, signal) // MTF는 5m 고정
        : await getSingleFibPlan(symbol, market, analysisInterval, chartInterval, klines.klines, signal); // 단일TF는 분석 기준
    if (signal.aborted || loadSeq !== activeLoadSeq) return;
    lastFibPlan = decisionFibPlan;

    const candles = klines.klines.map((k) => ({
      time: Math.floor(k.time / 1000),
      open: Number(k.open),
      high: Number(k.high),
      low: Number(k.low),
      close: Number(k.close),
    }));
    candleSeries.setData(candles);
    if (volumeSeries) {
      if (volumeToggleEl && volumeToggleEl.checked) {
        const volumes = klines.klines.map((k) => {
          const o = Number(k.open);
          const c = Number(k.close);
          return {
            time: Math.floor(k.time / 1000),
            value: Number(k.volume),
            color: c >= o ? "rgba(110,231,255,0.45)" : "rgba(255,99,132,0.45)",
          };
        });
        volumeSeries.setData(volumes);
      } else {
        volumeSeries.setData([]);
      }
    }
    const fitKey = `${market}:${symbol}:${chartInterval}`;
    if (lastFitKey !== fitKey) {
      chart.timeScale().fitContent();
      lastFitKey = fitKey;
    }

    renderAnalysisOnly(analysis, analysisMode, market);
    renderFibPrices(decisionFibPlan);
    renderActionSummary(analysis, decisionFibPlan);
    schedulePassCheck(symbol, market, analysisMode, analysisInterval);
    scheduleSidebarPinUpdate();

    applyFibOverlay();
    if (avwapToggleEl && avwapToggleEl.checked) {
      addAvwapLines(klines.klines, analysis);
    }
    if (vpToggleEl && vpToggleEl.checked) {
      addVpLines(analysis);
    }
    connectChartStream();
    connectAnalysisStream();
  } catch (e) {
    if (e?.name === "AbortError") return;
    throw e;
  } finally {
    if (activeLoadController === controller) activeLoadController = null;
    if (loadSeq === activeLoadSeq) isLoadingMain = false;
  }
}

async function safeLoadMain() {
  try {
    await load();
  } catch (e) {
    asofEl.textContent = "자동 갱신 오류: " + e.message;
  }
}

async function safeLoadPrices() {
  try {
    await loadCoinPrices();
  } catch (_) {}
}

function startAutoRefresh() {
  if (priceRefreshTimer) clearInterval(priceRefreshTimer);
  if (mainRefreshTimer) clearInterval(mainRefreshTimer);
  if (simRefreshTimer) clearInterval(simRefreshTimer);
  if (fxRefreshTimer) clearInterval(fxRefreshTimer);

  priceRefreshTimer = setInterval(() => {
    if (document.hidden) return;
    if (hasLiveListStream()) return;
    safeLoadPrices();
  }, PRICE_REFRESH_MS);

  // 분석/차트는 websocket으로 실시간 반영한다.
  mainRefreshTimer = null;

  fxRefreshTimer = setInterval(() => {
    if (document.hidden) return;
    loadFxRate();
  }, FX_REFRESH_MS);

  if (newsRefreshTimer) clearInterval(newsRefreshTimer);
  newsRefreshTimer = setInterval(() => {
    if (document.hidden) return;
    loadNewsSentiment();
  }, NEWS_REFRESH_MS);

  simRefreshTimer = setInterval(() => {
    if (document.hidden) return;
    loadSimTrades().catch(() => {});
  }, SIM_REFRESH_MS);
}

refreshEl.addEventListener("click", () =>
  Promise.all([loadCoinPrices(), load({ forceAnalysis: true }), loadSimTrades()]).catch((e) => alert(e.message))
);
if (headerAutoBtnEl)
  headerAutoBtnEl.addEventListener("click", () => {
    const t = Date.now();
    const screenW = Number(window.screen?.availWidth || window.innerWidth || 1280);
    const screenH = Number(window.screen?.availHeight || window.innerHeight || 900);
    const popupW = Math.max(520, Math.min(760, screenW - 40));
    const popupH = Math.max(560, Math.min(860, screenH - 70));
    const left = Math.max(0, Math.floor((screenW - popupW) / 2));
    const top = Math.max(0, Math.floor((screenH - popupH) / 2));
    const features = `width=${popupW},height=${popupH},left=${left},top=${top},resizable=yes,scrollbars=yes`;
    const pop = window.open(`/static/auto_trades_popup.html?t=${t}`, "auto_trades_popup", features);
    if (pop && !pop.closed) {
      try {
        pop.resizeTo(popupW, popupH);
        pop.moveTo(left, top);
        pop.focus();
      } catch (_) {}
    }
  });
if (analysisIntervalEl)
  analysisIntervalEl.addEventListener("change", () => {
    if (analysisIntervalEl.value) lastSingleAnalysisInterval = analysisIntervalEl.value;
    saveUiState();
    resetPassCheckUI("조건 변경: 계산 중...");
    load().catch((e) => alert(e.message));
  });
if (analysisModeEl)
  analysisModeEl.addEventListener("change", () => {
    syncAnalysisControls();
    saveUiState();
    resetPassCheckUI("조건 변경: 계산 중...");
    load().catch((e) => alert(e.message));
  });
if (chartIntervalEl)
  chartIntervalEl.addEventListener("change", () => {
    saveUiState();
    resetPassCheckUI("조건 변경: 계산 중...");
    load().catch((e) => alert(e.message));
  });
if (passCheckPeriodEl)
  passCheckPeriodEl.addEventListener("change", () => {
    saveUiState();
    resetPassCheckUI("기간 변경: 계산 중...");
    loadPassCheck(selectedSymbol, selectedMarket, analysisModeEl ? analysisModeEl.value : "single", analysisIntervalEl ? analysisIntervalEl.value : "5m").catch(
      () => {}
    );
  });
if (fibToggleEl)
  fibToggleEl.addEventListener("change", () => {
    saveUiState();
    applyFibOverlay();
  });
if (avwapToggleEl)
  avwapToggleEl.addEventListener("change", () => {
    saveUiState();
    load().catch((e) => alert(e.message));
  });
if (vpToggleEl)
  vpToggleEl.addEventListener("change", () => {
    saveUiState();
    load().catch((e) => alert(e.message));
  });
if (volumeToggleEl)
  volumeToggleEl.addEventListener("change", () => {
    saveUiState();
    load().catch((e) => alert(e.message));
  });
if (newsRefreshEl) newsRefreshEl.addEventListener("click", () => loadNewsSentiment());
if (simAddBtnEl) simAddBtnEl.addEventListener("click", () => simAddTrade());
if (simToggleBtnEl) simToggleBtnEl.addEventListener("click", () => toggleSimForm());
if (simOpenAllBtnEl)
  simOpenAllBtnEl.addEventListener("click", () => {
    window.open("/static/simulations_popup.html", "simulations_popup", "width=1020,height=780,resizable=yes,scrollbars=yes");
  });
if (simSymbolEl) simSymbolEl.addEventListener("change", () => fillSimEntryWithCurrentPrice());
if (simUseCurrentEl) simUseCurrentEl.addEventListener("change", () => fillSimEntryWithCurrentPrice());
if (simEntryEl) simEntryEl.addEventListener("input", () => { clampNonNegativeInput(simEntryEl); setSimRuleNote(false); });
if (simTpEl) simTpEl.addEventListener("input", () => { clampNonNegativeInput(simTpEl); setSimRuleNote(false); });
if (simSlEl) simSlEl.addEventListener("input", () => { clampNonNegativeInput(simSlEl); setSimRuleNote(false); });

restoreUiState();
updateDesktopModeClass();
window.addEventListener("resize", updateDesktopModeClass, { passive: true });
initSidebarPinFallback();
initCoinListInteractionGuard();
initChartResizeObserver();
renderProbAsset();
populateSimSymbolOptions();
syncSimSymbolWithSelected();
requestCoinListRender(true);
renderNewsPanel();
syncAnalysisControls();
Promise.all([loadFxRate(), loadCoinPrices(), load()])
  .catch((e) => {
    asofEl.textContent = "에러: " + e.message;
  });
renderSimTrades(loadCachedSimTrades());
loadSimTrades().catch(() => {});
setSimRuleNote(false);
startAutoRefresh();
bindVisibilityRealtime();
loadNewsSentiment();
