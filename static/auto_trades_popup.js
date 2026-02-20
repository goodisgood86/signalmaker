const filterSymbolEl = document.getElementById("filterSymbol");
const filterModeEl = document.getElementById("filterMode");
const filterStatusEl = document.getElementById("filterStatus");
const summaryEl = document.getElementById("summary");
const summarySignalStatusEl = document.getElementById("summarySignalStatus");
const summaryRefreshBtnEl = document.getElementById("summaryRefreshBtn");
const recordsBodyEl = document.getElementById("recordsBody");
const emptyStateEl = document.getElementById("emptyState");
const prevBtnEl = document.getElementById("prevBtn");
const nextBtnEl = document.getElementById("nextBtn");
const pageLabelEl = document.getElementById("pageLabel");

const cfgEnabledEl = document.getElementById("cfgEnabled");
const cfgSymbolChipsEl = document.getElementById("cfgSymbolChips");
const cfgSymbolEl = document.getElementById("cfgSymbol");
const cfgMarketEl = document.getElementById("cfgMarket");
const cfgIntervalEl = document.getElementById("cfgInterval");
const cfgModeEl = document.getElementById("cfgMode");
const cfgOrderSizeEl = document.getElementById("cfgOrderSize");
const cfgDailyLossEl = document.getElementById("cfgDailyLoss");
const cfgTpModeEl = document.getElementById("cfgTpMode");
const cfgTpPctEl = document.getElementById("cfgTpPct");
const cfgSlModeEl = document.getElementById("cfgSlMode");
const cfgSlPctEl = document.getElementById("cfgSlPct");
const cfgCooldownMinEl = document.getElementById("cfgCooldownMin");
const cfgMaxOpenEl = document.getElementById("cfgMaxOpen");
const cfgReloadBtnEl = document.getElementById("cfgReloadBtn");
const cfgSaveBtnEl = document.getElementById("cfgSaveBtn");
const cfgRunBtnEl = document.getElementById("cfgRunBtn");
const cfgTopRunBtnEl = document.getElementById("cfgTopRunBtn");
const cfgTopRunHintEl = document.getElementById("cfgTopRunHint");
const cfgRunningSymbolChipEl = document.getElementById("cfgRunningSymbolChip");
const cfgStatusEl = document.getElementById("cfgStatus");
const cfgSaveHintEl = document.getElementById("cfgSaveHint");
const configSectionEl = document.getElementById("configSection");
const cfgCollapseBtnEl = document.getElementById("cfgCollapseBtn");
const cfgUnlockInputEl = document.getElementById("cfgUnlockInput");
const cfgUnlockBtnEl = document.getElementById("cfgUnlockBtn");
const cfgLockStatusEl = document.getElementById("cfgLockStatus");
const bnApiKeyEl = document.getElementById("bnApiKey");
const bnApiSecretEl = document.getElementById("bnApiSecret");
const bnLinkBtnEl = document.getElementById("bnLinkBtn");
const bnUnlinkBtnEl = document.getElementById("bnUnlinkBtn");
const bnLinkFormEl = document.getElementById("bnLinkForm");
const bnLinkFormActionsEl = document.getElementById("bnLinkFormActions");
const bnLinkedActionsEl = document.getElementById("bnLinkedActions");
const bnLinkBadgeEl = document.getElementById("bnLinkBadge");
const bnLinkedInfoEl = document.getElementById("bnLinkedInfo");
const bnCollateralTotalEl = document.getElementById("bnCollateralTotal");
const bnCollateralSpotEl = document.getElementById("bnCollateralSpot");
const bnCollateralFuturesEl = document.getElementById("bnCollateralFutures");
const bnLinkStatusEl = document.getElementById("bnLinkStatus");
const logicOpenBtnEl = document.getElementById("logicOpenBtn");
const centerCloseBtnEl = document.getElementById("centerCloseBtn");
const logicModalEl = document.getElementById("logicModal");
const logicModalCloseBtnEl = document.getElementById("logicModalCloseBtn");

const SYMBOLS = ["ALL", "BTCUSDT", "ETHUSDT", "XRPUSDT", "DOGEUSDT", "SUIUSDT", "SOLUSDT", "CROSSUSDT"];
const TRADE_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "DOGEUSDT", "SUIUSDT", "SOLUSDT", "CROSSUSDT"];
const MODES = [
  { v: "ALL", t: "전체 모드" },
  { v: "balanced", t: "기본 모드" },
  { v: "aggressive", t: "공격 모드" },
];
const STATUSES = [
  { v: "ALL", t: "전체 상태" },
  { v: "OPEN", t: "보유중" },
  { v: "TP", t: "익절" },
  { v: "SL", t: "손절" },
  { v: "CLOSED_FAIL", t: "시간종료" },
];

let page = 1;
let hasNext = false;
let activeLoadSeq = 0;
let refreshTimer = null;
let statsRefreshTimer = null;
let collateralRefreshTimer = null;
let runtimeRefreshTimer = null;
let binanceLinkBusy = false;
let binanceLinkActive = false;
let configUnlocked = false;
let configUnlockBusy = false;
let binanceCollateralLoading = false;
let configDirty = true;
let autoRunActive = false;
let latestCollateral = null;
let collateralInsufficient = false;
let collateralGuardBusy = false;
let lastCollateralFetchMs = 0;
const COLLATERAL_MIN_INTERVAL_MS = 30000;
const STATS_REFRESH_INTERVAL_MS = 60000;
let latestOpenCount = null;
let runningSymbol = "BTCUSDT";
let latestTickStatusText = "";

function fmtPrice(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "-";
  if (n >= 1000) return n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  if (n >= 1) return n.toLocaleString("en-US", { minimumFractionDigits: 3, maximumFractionDigits: 3 });
  return n.toLocaleString("en-US", { minimumFractionDigits: 6, maximumFractionDigits: 6 });
}

function fmtSigned(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "-";
  return `${n >= 0 ? "+" : ""}${n.toFixed(2)}`;
}

function fmtUsdt(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "-";
  if (Math.abs(n) >= 1000) {
    return n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }
  if (Math.abs(n) >= 1) {
    return n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 4 });
  }
  if (Math.abs(n) >= 0.01) {
    return n.toLocaleString("en-US", { minimumFractionDigits: 4, maximumFractionDigits: 6 });
  }
  return n.toLocaleString("en-US", { minimumFractionDigits: 6, maximumFractionDigits: 8 });
}

function fmtTs(ms) {
  const n = Number(ms);
  if (!Number.isFinite(n) || n <= 0) return "-";
  return new Date(n).toLocaleString("ko-KR", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function fibPlanStatusText(plan) {
  if (!plan || typeof plan !== "object") return "";
  const entry = fmtPrice(plan?.entry_price);
  const entryLo = fmtPrice(plan?.entry_lo);
  const entryHi = fmtPrice(plan?.entry_hi);
  const stop = fmtPrice(plan?.stop_price);
  const tp1 = fmtPrice(plan?.tp1_price);
  const tp2 = fmtPrice(plan?.tp2_price);
  if ([stop, tp1, tp2].some((v) => v === "-")) return "";
  let entryText = entry;
  if (entryLo !== "-" && entryHi !== "-") entryText = `${entryLo}~${entryHi}`;
  if (entryText === "-") return "";
  return `피보 진입 ${entryText} / 손절 ${stop} / 익절 ${tp1}~${tp2}`;
}

function signalStatusText(signal) {
  if (!signal || typeof signal !== "object") return "";
  const buy = Number(signal?.buy_pct);
  const sell = Number(signal?.sell_pct);
  const conf = Number(signal?.confidence);
  const score = Number(signal?.score);
  const scoreThreshold = Number(signal?.score_threshold);
  const parts = [];
  if (Number.isFinite(buy) && Number.isFinite(sell)) parts.push(`B${buy.toFixed(1)} / S${sell.toFixed(1)}`);
  if (Number.isFinite(conf)) parts.push(`신뢰도 ${(conf * 100).toFixed(1)}%`);
  if (Number.isFinite(score) && Number.isFinite(scoreThreshold)) parts.push(`점수 ${score.toFixed(1)} / 기준 ${scoreThreshold.toFixed(0)}`);
  return parts.join(" | ");
}

function setTickStatusFromResponse(data) {
  const action = String(data?.action || "");
  const reason = String(data?.reason || "");
  const planText = fibPlanStatusText(data?.plan);
  const sigText = signalStatusText(data?.signal);
  const append = (base) => {
    const pieces = [base];
    if (planText) pieces.push(planText);
    if (sigText) pieces.push(sigText);
    return pieces.join(" | ");
  };
  if (action === "OPENED") {
    setCfgStatus("자동매매 진입 실행됨");
    return;
  }
  if (action === "NO_SIGNAL") {
    const rrVal = Number(data?.rr);
    const minRrVal = Number(data?.min_rr);
    const rrText =
      Number.isFinite(rrVal) && Number.isFinite(minRrVal) ? `RR ${rrVal.toFixed(2)} < 최소 ${minRrVal.toFixed(2)}` : "";
    const base =
      reason === "SIGNAL_SCORE_LOW"
        ? "자동매매 대기: 신호 점수 미달"
        : reason === "SPOT_SELL_BLOCKED"
          ? "자동매매 대기: 하락 우세지만 스팟 숏 금지"
        : reason === "SIGNAL_SIDE_WAIT"
          ? "자동매매 대기: 방향 우위 미확정"
          : reason === "REVERSAL_NOT_READY"
            ? "자동매매 대기: 바닥 반전 확인 미완료"
      : reason === "BASIC_PASS_FAIL"
        ? "자동매매 대기: 확률 미PASS"
        : reason === "SWING_MISMATCH"
          ? "자동매매 대기: 스윙 방향 불일치"
          : reason === "AGGRESSIVE_THRESHOLD"
            ? "자동매매 대기: 공격모드 진입 점수 미달"
          : reason === "ENTRY_ZONE_BREAK"
            ? "자동매매 대기: 진입구간 하단 이탈(재계산 대기)"
          : reason === "ENTRY_CHASE_LIMIT"
            ? "자동매매 대기: 추격 진입 제한"
            : reason === "RR_TOO_LOW"
              ? "자동매매 대기: 기대 RR 부족"
        : reason === "PLAN_INVALIDATED"
          ? "자동매매 대기: 기존 피보 시나리오 무효 (재계산 대기)"
          : "자동매매 대기: 진입 신호 없음";
    const full = append(base);
    setCfgStatus(rrText ? `${full} | ${rrText}` : full);
    return;
  }
  if (action === "WAIT_FIB_ENTRY") {
    const base = "자동매매 대기: 피보나치 진입가 대기";
    setCfgStatus(append(base));
    return;
  }
  if (action === "SKIP_DAILY_LOSS_LIMIT") {
    setCfgStatus("자동매매 중지: 일일 손실 한도 도달");
    return;
  }
  if (action === "SKIP_OPEN_LIMIT") {
    setCfgStatus("자동매매 대기: 최대 포지션 수 도달");
    return;
  }
  if (action === "SKIP_COOLDOWN") {
    setCfgStatus("자동매매 대기: 재진입 쿨다운");
    return;
  }
  if (action === "SKIP_COLLATERAL_LOW") {
    setCfgStatus("자동매매 중지: 담보금 부족");
    return;
  }
  if (action === "SKIP_COLLATERAL_ERROR") {
    setCfgStatus("자동매매 대기: 담보금 조회 실패");
    return;
  }
  if (action === "ORDER_REJECTED") {
    setCfgStatus(`실주문 실패: ${String(data?.detail || "주문 거부")}`);
    return;
  }
  if (action === "DISABLED") {
    setCfgStatus("자동매매 비활성화 상태");
    return;
  }
  setCfgStatus("");
}

function statusMeta(status) {
  const s = String(status || "").toUpperCase();
  if (s === "TP") return { cls: "b-tp", text: "익절" };
  if (s === "SL") return { cls: "b-sl", text: "손절" };
  if (s === "CLOSED_FAIL") return { cls: "b-fail", text: "시간종료" };
  return { cls: "b-open", text: "거래중" };
}

function modeText(mode) {
  const m = String(mode || "").toLowerCase();
  if (m === "aggressive") return "공격모드";
  if (m === "balanced") return "기본모드";
  return m || "-";
}

function marketBySymbol(symbol) {
  return String(symbol || "").toUpperCase() === "CROSSUSDT" ? "futures" : "spot";
}

function normalizeTradeSymbol(symbol) {
  const s = String(symbol || "").toUpperCase();
  return TRADE_SYMBOLS.includes(s) ? s : "BTCUSDT";
}

function updateRunningSymbolChip() {
  if (!cfgRunningSymbolChipEl) return;
  const symbol = normalizeTradeSymbol(runningSymbol || cfgSymbolEl?.value || "BTCUSDT");
  if (!autoRunActive) {
    cfgRunningSymbolChipEl.hidden = true;
    cfgRunningSymbolChipEl.textContent = "";
    return;
  }
  cfgRunningSymbolChipEl.hidden = false;
  cfgRunningSymbolChipEl.textContent = symbol;
}

function applySymbolSelection(symbol) {
  const target = normalizeTradeSymbol(symbol || "BTCUSDT");
  if (cfgSymbolEl) cfgSymbolEl.value = target;
  if (cfgMarketEl) cfgMarketEl.value = marketBySymbol(target);
  runningSymbol = target;
  updateRunningSymbolChip();
  if (!cfgSymbolChipsEl) return;
  const chips = cfgSymbolChipsEl.querySelectorAll(".symbol-chip");
  chips.forEach((chip) => {
    const chipSymbol = String(chip.getAttribute("data-symbol") || "").toUpperCase();
    chip.classList.toggle("active", chipSymbol === target);
  });
}

function applyExitModeUI() {
  const tpMode = String(cfgTpModeEl?.value || "auto").toLowerCase();
  const slMode = String(cfgSlModeEl?.value || "auto").toLowerCase();
  const tpManual = tpMode === "manual";
  const slManual = slMode === "manual";
  if (cfgTpPctEl) cfgTpPctEl.disabled = !tpManual;
  if (cfgSlPctEl) cfgSlPctEl.disabled = !slManual;
}

function setCfgStatus(msg) {
  const text = String(msg || "");
  if (cfgStatusEl) cfgStatusEl.textContent = text;
  latestTickStatusText = text;
  updateTopRunButton();
  updateCurrentTradeStateChip();
}

function currentTradeStateInfo(records = null) {
  let openCount = null;
  if (Number.isFinite(latestOpenCount)) openCount = Math.max(0, Math.round(Number(latestOpenCount)));
  if (openCount === null && Array.isArray(records)) {
    openCount = records.filter((r) => String(r?.status || "").toUpperCase() === "OPEN").length;
  }
  if (autoRunActive) {
    if (Number.isFinite(openCount) && openCount > 0) {
      return { text: `거래중 ${openCount}건`, cls: "live" };
    }
    return { text: "진입 대기중", cls: "wait" };
  }
  if (collateralInsufficient) return { text: "담보금 부족", cls: "warn" };
  if (Number.isFinite(openCount) && openCount > 0) {
    return { text: `거래중 ${openCount}건`, cls: "live" };
  }
  return { text: "자동매매 중지", cls: "off" };
}

function shouldShowSummarySignalStatus(info) {
  if (!info || typeof info !== "object") return false;
  return info.cls === "wait" || info.cls === "live";
}

function updateCurrentTradeStateChip(records = null) {
  const info = currentTradeStateInfo(records);
  if (summarySignalStatusEl) {
    if (shouldShowSummarySignalStatus(info)) summarySignalStatusEl.textContent = info.text;
    else summarySignalStatusEl.textContent = "";
  }
}

function setCfgLockStatus(msg) {
  if (cfgLockStatusEl) cfgLockStatusEl.textContent = msg || "";
}

function setConfigLocked(locked) {
  if (!configSectionEl) return;
  const isLocked = Boolean(locked);
  configSectionEl.classList.toggle("is-locked", isLocked);
  configUnlocked = !isLocked;
  if (cfgEnabledEl) cfgEnabledEl.disabled = isLocked;
  updateTopRunButton();
}

function isConfigUnlocked() {
  return configUnlocked;
}

function setConfigCollapsed(collapsed) {
  if (!configSectionEl) return;
  const isCollapsed = Boolean(collapsed);
  configSectionEl.classList.toggle("is-collapsed", isCollapsed);
  if (cfgCollapseBtnEl) {
    cfgCollapseBtnEl.textContent = isCollapsed ? "펼치기" : "접기";
    cfgCollapseBtnEl.setAttribute("aria-expanded", String(!isCollapsed));
  }
}

function toggleConfigCollapsed() {
  if (!configSectionEl) return;
  const isCollapsed = configSectionEl.classList.contains("is-collapsed");
  setConfigCollapsed(!isCollapsed);
}

function setLogicModalOpen(open) {
  if (!logicModalEl) return;
  const isOpen = Boolean(open);
  logicModalEl.hidden = !isOpen;
  document.body.style.overflow = isOpen ? "hidden" : "";
}

function updateTopRunButton() {
  if (!cfgTopRunBtnEl) {
    updateRunningSymbolChip();
    return;
  }
  updateSaveButtonState();
  const locked = !isConfigUnlocked();
  const blockedByCollateral = collateralInsufficient && !autoRunActive;
  let runDisabledReason = "";
  cfgTopRunBtnEl.classList.toggle("running", autoRunActive && !blockedByCollateral);
  cfgTopRunBtnEl.classList.toggle("blocked", blockedByCollateral);
  if (blockedByCollateral) {
    cfgTopRunBtnEl.textContent = "담보금 부족";
    cfgTopRunBtnEl.disabled = true;
    runDisabledReason = "담보금 부족으로 실행할 수 없습니다.";
    cfgTopRunBtnEl.classList.add("is-disabled");
    cfgTopRunBtnEl.title = runDisabledReason;
    if (cfgTopRunHintEl) cfgTopRunHintEl.textContent = latestTickStatusText || runDisabledReason;
    updateRunningSymbolChip();
    return;
  }
  if (autoRunActive) {
    cfgTopRunBtnEl.textContent = "자동매매 진행 중";
    cfgTopRunBtnEl.disabled = locked;
    if (locked) runDisabledReason = "설정 잠금 해제 후 중지할 수 있습니다.";
    cfgTopRunBtnEl.classList.toggle("is-disabled", cfgTopRunBtnEl.disabled);
    cfgTopRunBtnEl.title = runDisabledReason;
    if (cfgTopRunHintEl) cfgTopRunHintEl.textContent = latestTickStatusText || runDisabledReason;
    updateRunningSymbolChip();
    return;
  }
  cfgTopRunBtnEl.textContent = "지금 실행";
  cfgTopRunBtnEl.disabled = locked || configDirty;
  if (locked) runDisabledReason = "설정 잠금 해제 후 실행할 수 있습니다.";
  else if (configDirty) runDisabledReason = "설정 저장 완료 후 실행할 수 있습니다.";
  cfgTopRunBtnEl.classList.toggle("is-disabled", cfgTopRunBtnEl.disabled);
  cfgTopRunBtnEl.title = runDisabledReason;
  if (cfgTopRunHintEl) cfgTopRunHintEl.textContent = runDisabledReason;
  updateRunningSymbolChip();
}

function setConfigDirty(dirty) {
  configDirty = Boolean(dirty);
  updateTopRunButton();
}

function updateSaveButtonState() {
  if (!cfgSaveBtnEl) return;
  const locked = !isConfigUnlocked();
  const running = autoRunActive;
  let saveDisabledReason = "";
  cfgSaveBtnEl.disabled = locked || running;
  if (running) saveDisabledReason = "자동매매 진행 중에는 설정 저장이 비활성화됩니다.";
  else if (locked) saveDisabledReason = "설정 잠금 해제 후 저장할 수 있습니다.";
  cfgSaveBtnEl.classList.toggle("is-disabled", cfgSaveBtnEl.disabled);
  cfgSaveBtnEl.title = saveDisabledReason;
  if (cfgSaveHintEl) {
    if (saveDisabledReason) cfgSaveHintEl.textContent = saveDisabledReason;
    else cfgSaveHintEl.textContent = "";
  }
}

function selectedMarketAvailable(collateral) {
  if (!collateral || typeof collateral !== "object") return null;
  const symbol = String(cfgSymbolEl?.value || "BTCUSDT");
  const market = marketBySymbol(symbol);
  if (market === "futures") return Number(collateral?.futures?.available_usdt ?? NaN);
  return Number(collateral?.spot?.available_usdt ?? NaN);
}

function evaluateCollateralState() {
  if (!binanceLinkActive || !latestCollateral || typeof latestCollateral !== "object") {
    collateralInsufficient = false;
    updateTopRunButton();
    return;
  }
  const available = selectedMarketAvailable(latestCollateral);
  const orderSize = Number(cfgOrderSizeEl?.value || 30);
  if (!Number.isFinite(available) || !Number.isFinite(orderSize) || orderSize <= 0) {
    collateralInsufficient = false;
    updateTopRunButton();
    return;
  }
  collateralInsufficient = available + 1e-9 < orderSize;
  updateTopRunButton();
  updateCurrentTradeStateChip();
}

async function enforceCollateralGuard() {
  if (collateralGuardBusy) return;
  if (!autoRunActive || !collateralInsufficient || !isConfigUnlocked()) return;
  collateralGuardBusy = true;
  try {
    if (cfgEnabledEl) cfgEnabledEl.checked = false;
    autoRunActive = false;
    const ok = await saveConfig(true);
    if (ok) {
      setCfgStatus("담보금 부족으로 자동매매가 중단되었습니다.");
    }
  } finally {
    collateralGuardBusy = false;
    updateTopRunButton();
  }
}

function clampNonNegativeInput(el) {
  if (!el) return;
  const n = Number(el.value);
  if (Number.isFinite(n) && n < 0) el.value = "0";
}

function clampMaxOpenInput() {
  if (!cfgMaxOpenEl) return 2;
  const n = Number(cfgMaxOpenEl.value);
  if (!Number.isFinite(n)) {
    cfgMaxOpenEl.value = "2";
    return 2;
  }
  const clamped = Math.max(1, Math.min(5, Math.round(n)));
  cfgMaxOpenEl.value = String(clamped);
  return clamped;
}

async function fetchJSON(url, options) {
  const r = await fetch(url, options);
  const text = await r.text();
  if (r.status === 401) {
    // 사이트 인증 만료일 때만 auth 페이지로 이동하고,
    // 설정 잠금 비밀번호 오류(401 invalid password)는 현재 팝업에서 처리한다.
    const t = String(text || "");
    const isAuthExpired = t.includes("unauthorized") && !t.includes("invalid password");
    if (isAuthExpired) {
      window.location.href = "/static/auth.html";
      throw new Error("401 unauthorized");
    }
  }
  if (!r.ok) throw new Error(text || `HTTP ${r.status}`);
  return text ? JSON.parse(text) : null;
}

function errMessage(err) {
  const raw = String(err?.message || err || "");
  if (!raw) return "알 수 없는 오류";
  try {
    const parsed = JSON.parse(raw);
    if (typeof parsed?.detail === "string" && parsed.detail) return parsed.detail;
  } catch (_) {}
  return raw;
}

function handleLockedError(msg) {
  const m = String(msg || "");
  if (!m.includes("config is locked")) return false;
  setConfigLocked(true);
  setCfgLockStatus("설정이 잠겨 있습니다. 비밀번호를 입력해주세요.");
  return true;
}

async function loadConfigLockStatus() {
  try {
    const data = await fetchJSON("/api/auto_trade/config_lock/status");
    const enabled = Boolean(data?.enabled);
    const unlocked = !enabled || Boolean(data?.unlocked);
    setConfigLocked(!unlocked);
    await loadRuntimeStatus();
    await loadBinanceLink();
    if (!unlocked) {
      setCfgLockStatus("비밀번호를 입력 후 설정을 사용할 수 있습니다.");
      return;
    }
    setCfgLockStatus("");
    await loadConfig();
  } catch (e) {
    const msg = errMessage(e);
    setConfigLocked(true);
    setCfgLockStatus(`잠금 상태 조회 실패: ${msg}`);
  }
}

async function loadRuntimeStatus() {
  try {
    const data = await fetchJSON("/api/auto_trade/runtime");
    const runtime = data?.runtime || {};
    autoRunActive = Boolean(runtime?.enabled);
    runningSymbol = normalizeTradeSymbol(runtime?.symbol || cfgSymbolEl?.value || "BTCUSDT");
    const lastTick = runtime?.last_tick;
    if (lastTick && typeof lastTick === "object" && String(lastTick?.action || "").trim()) {
      setTickStatusFromResponse({
        action: String(lastTick?.action || ""),
        reason: String(lastTick?.reason || ""),
        detail: String(lastTick?.detail || ""),
        plan: lastTick?.plan || null,
      });
    }
    if (cfgEnabledEl) cfgEnabledEl.checked = autoRunActive;
    updateTopRunButton();
    updateCurrentTradeStateChip();
  } catch (_) {}
}

function setUnlockBusy(busy) {
  configUnlockBusy = Boolean(busy);
  if (cfgUnlockBtnEl) {
    cfgUnlockBtnEl.disabled = configUnlockBusy;
    cfgUnlockBtnEl.textContent = configUnlockBusy ? "확인중..." : "해제";
  }
  if (cfgUnlockInputEl) cfgUnlockInputEl.disabled = configUnlockBusy;
}

function applyConfig(cfg) {
  if (!cfg || typeof cfg !== "object") return;
  autoRunActive = Boolean(cfg.enabled);
  runningSymbol = normalizeTradeSymbol(cfg.symbol || cfgSymbolEl?.value || "BTCUSDT");
  if (cfgEnabledEl) cfgEnabledEl.checked = autoRunActive;
  applySymbolSelection(String(cfg.symbol || "BTCUSDT"));
  if (cfgIntervalEl) cfgIntervalEl.value = String(cfg.interval || "5m");
  if (cfgModeEl) cfgModeEl.value = String(cfg.mode || "balanced");
  if (cfgOrderSizeEl) cfgOrderSizeEl.value = String(Number(cfg.order_size_usdt || 30));
  if (cfgDailyLossEl) cfgDailyLossEl.value = String(Number(cfg.daily_max_loss_usdt || 0));
  const tpPct = Number(cfg.take_profit_pct || 0);
  if (cfgTpModeEl) cfgTpModeEl.value = tpPct > 0 ? "manual" : "auto";
  if (cfgTpPctEl) cfgTpPctEl.value = tpPct > 0 ? String(tpPct) : "";
  const slPct = Number(cfg.stop_loss_pct || 0);
  if (cfgSlModeEl) cfgSlModeEl.value = slPct > 0 ? "manual" : "auto";
  if (cfgSlPctEl) cfgSlPctEl.value = slPct > 0 ? String(slPct) : "";
  applyExitModeUI();
  if (cfgCooldownMinEl) cfgCooldownMinEl.value = "0";
  if (cfgMaxOpenEl) {
    const maxOpen = Number(cfg.max_open_positions || 2);
    cfgMaxOpenEl.value = String(Math.max(1, Math.min(5, Math.round(maxOpen))));
  }
  setConfigDirty(false);
  evaluateCollateralState();
  updateCurrentTradeStateChip();
}

function collectConfigPayload() {
  const symbol = String(cfgSymbolEl?.value || "BTCUSDT").toUpperCase();
  const market = marketBySymbol(symbol);
  const tpMode = String(cfgTpModeEl?.value || "auto").toLowerCase();
  const slMode = String(cfgSlModeEl?.value || "auto").toLowerCase();
  const tpInput = Number(cfgTpPctEl?.value || 0);
  const slInput = Number(cfgSlPctEl?.value || 0);
  const maxOpen = clampMaxOpenInput();
  const tpPct = tpMode === "manual" && Number.isFinite(tpInput) && tpInput > 0 ? tpInput : 0;
  const slPct = slMode === "manual" && Number.isFinite(slInput) && slInput > 0 ? slInput : 0;
  return {
    enabled: Boolean(cfgEnabledEl?.checked),
    symbol,
    market,
    interval: String(cfgIntervalEl?.value || "5m"),
    mode: String(cfgModeEl?.value || "balanced"),
    order_size_usdt: Number(cfgOrderSizeEl?.value || 30),
    daily_max_loss_usdt: Number(cfgDailyLossEl?.value || 0),
    take_profit_pct: tpPct,
    stop_loss_pct: slPct,
    cooldown_min: 0,
    max_open_positions: maxOpen,
  };
}

function validateConfigForSave(payload) {
  const missing = [];
  const p = payload && typeof payload === "object" ? payload : {};
  const symbol = String(p.symbol || "").toUpperCase();
  const interval = String(p.interval || "");
  const mode = String(p.mode || "");
  const orderSize = Number(p.order_size_usdt);
  const dailyLoss = Number(p.daily_max_loss_usdt);
  const maxOpen = Number(p.max_open_positions);
  const tpMode = String(cfgTpModeEl?.value || "auto").toLowerCase();
  const slMode = String(cfgSlModeEl?.value || "auto").toLowerCase();
  const tpPctInput = Number(cfgTpPctEl?.value || 0);
  const slPctInput = Number(cfgSlPctEl?.value || 0);

  if (!TRADE_SYMBOLS.includes(symbol)) missing.push("코인 선택");
  if (!["5m", "1h", "4h"].includes(interval)) missing.push("분석 타임");
  if (!["balanced", "aggressive"].includes(mode)) missing.push("투자모드");
  if (!Number.isFinite(orderSize) || orderSize < 10) missing.push("1회 거래금액 (10 이상)");
  if (!Number.isFinite(dailyLoss) || dailyLoss < 1) missing.push("일 최대 손실 (1 이상)");
  if (!Number.isFinite(maxOpen) || maxOpen < 1 || maxOpen > 5) missing.push("동시 최대 포지션 (1~5)");
  if (tpMode === "manual" && (!Number.isFinite(tpPctInput) || tpPctInput <= 0)) {
    missing.push("익절 TP (%)");
  }
  if (slMode === "manual" && (!Number.isFinite(slPctInput) || slPctInput <= 0)) {
    missing.push("손절 SL (%)");
  }
  if (!binanceLinkActive) missing.push("바이낸스 API 연동");

  if (missing.length > 0) {
    return {
      ok: false,
      message: `설정 저장 불가: 필수값 미충족 (${missing.join(", ")})`,
    };
  }
  return { ok: true, message: "" };
}

async function loadConfig() {
  if (!isConfigUnlocked()) return;
  try {
    const data = await fetchJSON("/api/auto_trade/config");
    applyConfig(data?.config || {});
    setCfgStatus("");
  } catch (e) {
    const msg = errMessage(e);
    if (handleLockedError(msg)) return;
    setCfgStatus(`설정 조회 실패: ${msg}`);
  }
}

async function saveConfig(silent = false) {
  if (!isConfigUnlocked()) return false;
  if (autoRunActive) {
    setConfigDirty(true);
    setCfgStatus("자동매매 진행 중에는 설정 저장을 할 수 없습니다.");
    updateSaveButtonState();
    return false;
  }
  try {
    const payload = collectConfigPayload();
    const validation = validateConfigForSave(payload);
    if (!validation.ok) {
      setConfigDirty(true);
      setCfgStatus(validation.message);
      return false;
    }
    const data = await fetchJSON("/api/auto_trade/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    applyConfig(data?.config || payload);
    if (!silent) setCfgStatus("설정 저장 완료");
    return true;
  } catch (e) {
    const msg = errMessage(e);
    if (handleLockedError(msg)) return false;
    setConfigDirty(true);
    if (!silent) setCfgStatus(`설정 저장 실패: ${msg}`);
    return false;
  }
}

function setBnLinkStatus(msg) {
  if (bnLinkStatusEl) bnLinkStatusEl.textContent = msg || "";
}

function setBnCollateralCards(total, spot, futures) {
  if (bnCollateralTotalEl) bnCollateralTotalEl.textContent = total || "-";
  if (bnCollateralSpotEl) bnCollateralSpotEl.textContent = spot || "-";
  if (bnCollateralFuturesEl) bnCollateralFuturesEl.textContent = futures || "-";
}

function setBnLinkBadge(linked, pending = false) {
  if (!bnLinkBadgeEl) return;
  bnLinkBadgeEl.classList.toggle("on", Boolean(linked) && !pending);
  bnLinkBadgeEl.classList.toggle("pending", Boolean(pending));
  if (pending) bnLinkBadgeEl.textContent = "연동 처리중";
  else bnLinkBadgeEl.textContent = linked ? "연동중" : "미연동";
}

function setBnLinkVisibility(linked) {
  const isLinked = Boolean(linked);
  if (bnLinkFormEl) bnLinkFormEl.hidden = isLinked;
  if (bnLinkFormActionsEl) bnLinkFormActionsEl.hidden = isLinked;
  if (bnLinkedActionsEl) bnLinkedActionsEl.hidden = !isLinked;
}

function setBnLinkBusy(busy) {
  binanceLinkBusy = Boolean(busy);
  if (bnLinkBtnEl) {
    bnLinkBtnEl.disabled = binanceLinkBusy;
    bnLinkBtnEl.textContent = binanceLinkBusy ? "연동중..." : "스팟+선물 동시 연동";
  }
  if (bnUnlinkBtnEl) bnUnlinkBtnEl.disabled = binanceLinkBusy;
  if (bnApiKeyEl) bnApiKeyEl.disabled = binanceLinkBusy;
  if (bnApiSecretEl) bnApiSecretEl.disabled = binanceLinkBusy;
  if (binanceLinkBusy) setBnLinkBadge(binanceLinkActive, true);
  else setBnLinkBadge(binanceLinkActive, false);
}

function renderBinanceLink(link) {
  const linked = Boolean(link?.linked);
  binanceLinkActive = linked;
  if (!linked) {
    latestCollateral = null;
    collateralInsufficient = false;
  }
  setBnLinkVisibility(linked);
  setBnLinkBadge(linked, false);
  if (bnLinkedInfoEl) {
    bnLinkedInfoEl.textContent = "";
  }
  if (linked) setBnCollateralCards("조회중...", "조회중...", "조회중...");
  else setBnCollateralCards("-", "-", "-");
  updateTopRunButton();
}

function renderBinanceCollateral(collateral) {
  latestCollateral = collateral && typeof collateral === "object" ? collateral : null;
  const spot = collateral?.spot || {};
  const fut = collateral?.futures || {};
  const totalBal = Number(collateral?.total_usdt ?? NaN);
  const spotBal = Number(spot?.total_usdt ?? NaN);
  const futBal = Number(fut?.total_usdt ?? NaN);
  const totalTxt = Number.isFinite(totalBal) ? `${fmtUsdt(totalBal)} USDT` : "-";
  const spotTxt = Boolean(spot?.ok) && Number.isFinite(spotBal) ? `${fmtUsdt(spotBal)} USDT` : "-";
  const futTxt = Boolean(fut?.ok) && Number.isFinite(futBal) ? `${fmtUsdt(futBal)} USDT` : "-";
  setBnCollateralCards(totalTxt, spotTxt, futTxt);
  evaluateCollateralState();
}

async function loadBinanceCollateral() {
  if (binanceCollateralLoading) return;
  const nowMs = Date.now();
  if (lastCollateralFetchMs > 0 && nowMs - lastCollateralFetchMs < COLLATERAL_MIN_INTERVAL_MS) return;
  binanceCollateralLoading = true;
  lastCollateralFetchMs = nowMs;
  try {
    const data = await fetchJSON("/api/auto_trade/binance/collateral");
    renderBinanceCollateral(data?.collateral || {});
    await enforceCollateralGuard();
    if (Boolean(data?.collateral?.stale)) {
      setBnLinkStatus("바이낸스 요청 제한으로 캐시된 담보금을 표시 중입니다.");
    } else {
      setBnLinkStatus("");
    }
  } catch (e) {
    const msg = errMessage(e);
    if (handleLockedError(msg)) return;
    if (msg.includes("binance link not found") || msg.includes("binance is not linked")) {
      latestCollateral = null;
      collateralInsufficient = false;
      setBnCollateralCards("-", "-", "-");
      updateTopRunButton();
      return;
    }
    if (msg.includes("-1003") || msg.toLowerCase().includes("rate limit")) {
      setBnLinkStatus(`요청 제한 상태입니다. 잠시 후 자동 재시도합니다. (${msg})`);
      return;
    }
    setBnCollateralCards("조회 실패", "조회 실패", "조회 실패");
    setBnLinkStatus(`담보금 조회 실패: ${msg}`);
  } finally {
    binanceCollateralLoading = false;
  }
}

async function loadBinanceLink() {
  try {
    const data = await fetchJSON("/api/auto_trade/binance/link");
    const link = data?.link || {};
    renderBinanceLink(link);
    if (Boolean(link?.linked)) {
      lastCollateralFetchMs = 0;
      await loadBinanceCollateral();
    } else {
      setBnLinkStatus("");
    }
  } catch (e) {
    const msg = errMessage(e);
    if (handleLockedError(msg)) return;
    setBnLinkStatus(`연동 상태 조회 실패: ${msg}`);
  }
}

async function linkBinance() {
  if (!isConfigUnlocked()) {
    setCfgLockStatus("먼저 비밀번호를 입력해주세요.");
    return;
  }
  const apiKey = String(bnApiKeyEl?.value || "").trim();
  const apiSecret = String(bnApiSecretEl?.value || "").trim();
  if (!apiKey || !apiSecret) {
    setBnLinkStatus("API Key / Secret를 입력해주세요.");
    return;
  }
  setBnLinkBusy(true);
  setBnLinkStatus("바이낸스 연동중...");
  try {
    const data = await fetchJSON("/api/auto_trade/binance/link", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ api_key: apiKey, api_secret: apiSecret }),
    });
    renderBinanceLink(data?.link || {});
    lastCollateralFetchMs = 0;
    await loadBinanceCollateral();
    if (bnApiKeyEl) bnApiKeyEl.value = "";
    if (bnApiSecretEl) bnApiSecretEl.value = "";
    if (Boolean(data?.verify_skipped)) {
      setBnLinkStatus("요청 제한으로 검증을 일시 건너뛰고 저장했습니다. 잠시 후 담보금 조회로 확인하세요.");
    } else {
      setBnLinkStatus("");
    }
  } catch (e) {
    const msg = errMessage(e);
    if (handleLockedError(msg)) return;
    setBnLinkStatus(`연동 실패: ${msg}`);
  } finally {
    setBnLinkBusy(false);
  }
}

async function unlinkBinance() {
  if (!isConfigUnlocked()) {
    setCfgLockStatus("먼저 비밀번호를 입력해주세요.");
    return;
  }
  setBnLinkBusy(true);
  setBnLinkStatus("연동 해제중...");
  try {
    await fetchJSON("/api/auto_trade/binance/link", { method: "DELETE" });
    renderBinanceLink({ linked: false, market: String(cfgMarketEl?.value || "spot"), api_key_masked: "" });
    setBnCollateralCards("-", "-", "-");
    if (bnApiKeyEl) bnApiKeyEl.value = "";
    if (bnApiSecretEl) bnApiSecretEl.value = "";
    setBnLinkStatus("");
  } catch (e) {
    const msg = errMessage(e);
    if (handleLockedError(msg)) return;
    setBnLinkStatus(`연동 해제 실패: ${msg}`);
  } finally {
    setBnLinkBusy(false);
  }
}

async function triggerTickOnce() {
  const data = await fetchJSON("/api/auto_trade/tick", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ force: true }),
  });
  setTickStatusFromResponse(data);
}

async function toggleTopRun() {
  if (!isConfigUnlocked()) {
    setCfgLockStatus("먼저 비밀번호를 입력해주세요.");
    return;
  }
  if (!autoRunActive && collateralInsufficient) {
    setCfgStatus("담보금 부족으로 실행할 수 없습니다.");
    updateTopRunButton();
    return;
  }
  if (autoRunActive) {
    if (cfgEnabledEl) cfgEnabledEl.checked = false;
    autoRunActive = false;
    updateTopRunButton();
    const ok = await saveConfig(true);
    if (ok) setCfgStatus("자동매매가 중지되었습니다.");
    return;
  }
  // 시작 전 저장을 먼저 수행해야 saveConfig의 "진행 중 저장 불가" 가드와 충돌하지 않는다.
  if (cfgEnabledEl) cfgEnabledEl.checked = true;
  const ok = await saveConfig(true);
  if (!ok) {
    if (cfgEnabledEl) cfgEnabledEl.checked = false;
    setCfgStatus("설정 저장 완료 후 실행할 수 있습니다.");
    updateTopRunButton();
    return;
  }
  autoRunActive = true;
  updateTopRunButton();
  try {
    await triggerTickOnce();
  } catch (e) {
    const msg = errMessage(e);
    if (handleLockedError(msg)) return;
    setCfgStatus(`실행 실패: ${msg}`);
  } finally {
    await Promise.all([load(), loadStats(true)]);
  }
}

async function runNow() {
  if (!isConfigUnlocked()) {
    setCfgLockStatus("먼저 비밀번호를 입력해주세요.");
    return;
  }
  const ok = await saveConfig(true);
  if (!ok) return;
  try {
    const data = await fetchJSON("/api/auto_trade/tick", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ force: true }),
    });
    setTickStatusFromResponse(data);
    await Promise.all([load(), loadStats(true)]);
  } catch (e) {
    const msg = errMessage(e);
    if (handleLockedError(msg)) return;
    setCfgStatus(`실행 실패: ${msg}`);
  }
}

function renderSummary(stats) {
  if (!summaryEl) return;
  const s = stats || {};
  latestOpenCount = Number.isFinite(Number(s.open)) ? Number(s.open) : latestOpenCount;
  const cards = [
    ["총 거래", Number(s.total || 0).toLocaleString("ko-KR")],
    ["손익금액", `${fmtSigned(Number(s.realized_pnl_usdt || 0))} USDT`],
    ["손익%", `${Number(s.realized_pnl_pct || 0).toFixed(2)}%`],
  ];
  summaryEl.innerHTML = cards.map((x) => `<div class="stat"><div class="stat-k">${x[0]}</div><div class="stat-v">${x[1]}</div></div>`).join("");
  updateCurrentTradeStateChip();
}

function render(records) {
  if (!recordsBodyEl) return;
  if (!Array.isArray(records) || records.length === 0) {
    recordsBodyEl.innerHTML = `<tr class="records-empty-row"><td colspan="9" class="records-num">표시할 기록이 없습니다.</td></tr>`;
    if (emptyStateEl) emptyStateEl.hidden = true;
    updateCurrentTradeStateChip([]);
    return;
  }
  if (emptyStateEl) emptyStateEl.hidden = true;

  const rows = records
    .slice()
    .sort((a, b) => Number(b?.opened_ms || 0) - Number(a?.opened_ms || 0));

  recordsBodyEl.innerHTML = rows
    .map((r) => {
      const st = statusMeta(r.status);
      const seed = Number(r?.notional_usdt || 0);
      const seedTxt = Number.isFinite(seed)
        ? `${seed.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })} USDT`
        : "-";
      const isOpen = String(r?.status || "").toUpperCase() === "OPEN";
      const pnl = Number(r?.pnl_usdt || 0);
      const profitTxt = isOpen ? "거래중" : `${fmtSigned(pnl)} USDT`;
      const profitCls = isOpen ? "" : pnl >= 0 ? "pos" : "neg";
      const buyTs = fmtTs(r.opened_ms);
      const sellRaw = Number(r?.closed_ms || 0);
      const sellTs = Number.isFinite(sellRaw) && sellRaw > 0 ? fmtTs(sellRaw) : "-";
      return `<tr>
        <td data-label="상태"><span class="badge ${st.cls}">${st.text}</span></td>
        <td data-label="코인">${String(r.symbol || "-")}</td>
        <td data-label="매매모드">${modeText(r.mode)}</td>
        <td data-label="거래시드" class="records-num">${seedTxt}</td>
        <td data-label="진입가" class="records-num">${fmtPrice(r.entry_price)}</td>
        <td data-label="익절/손절가" class="records-num">TP ${fmtPrice(r.take_profit_price)} / SL ${fmtPrice(r.stop_loss_price)}</td>
        <td data-label="수익" class="records-num records-profit ${profitCls}">${profitTxt}</td>
        <td data-label="구매일시" class="records-num">${buyTs}</td>
        <td data-label="판매일시" class="records-num">${sellTs}</td>
      </tr>`;
    })
    .join("");
  updateCurrentTradeStateChip(rows);
}

async function loadStats(sync = false) {
  const data = await fetchJSON(`/api/auto_trade/stats?sync=${sync ? "1" : "0"}`);
  renderSummary(data?.stats || {});
}

async function load() {
  const seq = ++activeLoadSeq;
  const symbol = filterSymbolEl.value;
  const mode = filterModeEl.value;
  const status = filterStatusEl.value;
  const q = new URLSearchParams({ limit: "20", page: String(page), sync: "0" });
  if (symbol && symbol !== "ALL") q.set("symbol", symbol);
  if (mode && mode !== "ALL") q.set("mode", mode);
  if (status && status !== "ALL") q.set("status", status);
  const data = await fetchJSON(`/api/auto_trade/records?${q.toString()}`);
  if (seq !== activeLoadSeq) return;
  render(data?.records || []);
  hasNext = Boolean(data?.has_next);
  pageLabelEl.textContent = `${page} 페이지`;
  prevBtnEl.disabled = page <= 1;
  nextBtnEl.disabled = !hasNext;
}

function initFilters() {
  filterSymbolEl.innerHTML = SYMBOLS.map((s) => `<option value="${s}">${s === "ALL" ? "전체 코인" : s}</option>`).join("");
  filterModeEl.innerHTML = MODES.map((s) => `<option value="${s.v}">${s.t}</option>`).join("");
  filterStatusEl.innerHTML = STATUSES.map((s) => `<option value="${s.v}">${s.t}</option>`).join("");
}

function initConfigInputs() {
  if (cfgSymbolChipsEl) {
    cfgSymbolChipsEl.querySelectorAll(".symbol-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        if (!isConfigUnlocked()) {
          setCfgLockStatus("먼저 비밀번호를 입력해주세요.");
          return;
        }
        const symbol = String(chip.getAttribute("data-symbol") || "").toUpperCase();
        applySymbolSelection(symbol);
        setConfigDirty(true);
        evaluateCollateralState();
      });
    });
  }
  if (cfgMaxOpenEl && !String(cfgMaxOpenEl.value || "").trim()) cfgMaxOpenEl.value = "2";
  clampMaxOpenInput();
  if (cfgCooldownMinEl) cfgCooldownMinEl.value = "0";
  applySymbolSelection(String(cfgSymbolEl?.value || "BTCUSDT"));
  applyExitModeUI();
}

function initConfigLock() {
  setConfigCollapsed(true);
  setConfigLocked(true);
  renderBinanceLink({ linked: false, market: String(cfgMarketEl?.value || "spot"), api_key_masked: "" });
  loadConfigLockStatus().catch(() => {});
}

async function tryUnlockConfig() {
  const inputValue = String(cfgUnlockInputEl?.value || "");
  if (!inputValue) {
    setCfgLockStatus("비밀번호를 입력해주세요.");
    return;
  }
  setUnlockBusy(true);
  try {
    await fetchJSON("/api/auto_trade/config_lock/unlock", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password: inputValue }),
    });
    setConfigLocked(false);
    setCfgLockStatus("");
    if (cfgUnlockInputEl) cfgUnlockInputEl.value = "";
    await Promise.all([loadConfig(), loadBinanceLink()]);
  } catch (e) {
    const msg = errMessage(e);
    if (msg.includes("invalid password")) setCfgLockStatus("비밀번호가 올바르지 않습니다.");
    else setCfgLockStatus(`잠금 해제 실패: ${msg}`);
    if (cfgUnlockInputEl) cfgUnlockInputEl.value = "";
    setConfigLocked(true);
  } finally {
    setUnlockBusy(false);
  }
}

function onFilterChange() {
  page = 1;
  load().catch((e) => alert(e.message || e));
}

filterSymbolEl.addEventListener("change", onFilterChange);
filterModeEl.addEventListener("change", onFilterChange);
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

if (cfgReloadBtnEl)
  cfgReloadBtnEl.addEventListener("click", () => {
    if (!isConfigUnlocked()) {
      setCfgLockStatus("먼저 비밀번호를 입력해주세요.");
      return;
    }
    loadConfig().catch(() => {});
    loadBinanceLink().catch(() => {});
  });
if (cfgSaveBtnEl) cfgSaveBtnEl.addEventListener("click", () => saveConfig(false));
if (cfgRunBtnEl) cfgRunBtnEl.addEventListener("click", () => runNow());
if (cfgTopRunBtnEl) cfgTopRunBtnEl.addEventListener("click", () => toggleTopRun());
if (cfgIntervalEl) cfgIntervalEl.addEventListener("change", () => setConfigDirty(true));
if (cfgModeEl) cfgModeEl.addEventListener("change", () => setConfigDirty(true));
if (cfgTpModeEl)
  cfgTpModeEl.addEventListener("change", () => {
    applyExitModeUI();
    setConfigDirty(true);
  });
if (cfgSlModeEl)
  cfgSlModeEl.addEventListener("change", () => {
    applyExitModeUI();
    setConfigDirty(true);
  });
if (cfgOrderSizeEl)
  cfgOrderSizeEl.addEventListener("input", () => {
    clampNonNegativeInput(cfgOrderSizeEl);
    setConfigDirty(true);
    evaluateCollateralState();
  });
if (cfgDailyLossEl)
  cfgDailyLossEl.addEventListener("input", () => {
    clampNonNegativeInput(cfgDailyLossEl);
    setConfigDirty(true);
  });
if (cfgTpPctEl)
  cfgTpPctEl.addEventListener("input", () => {
    clampNonNegativeInput(cfgTpPctEl);
    setConfigDirty(true);
  });
if (cfgSlPctEl)
  cfgSlPctEl.addEventListener("input", () => {
    clampNonNegativeInput(cfgSlPctEl);
    setConfigDirty(true);
  });
if (cfgCooldownMinEl) cfgCooldownMinEl.addEventListener("input", () => clampNonNegativeInput(cfgCooldownMinEl));
if (cfgMaxOpenEl)
  cfgMaxOpenEl.addEventListener("input", () => {
    clampMaxOpenInput();
    setConfigDirty(true);
  });
if (cfgUnlockBtnEl) cfgUnlockBtnEl.addEventListener("click", () => tryUnlockConfig());
if (cfgUnlockInputEl)
  cfgUnlockInputEl.addEventListener("keydown", (ev) => {
    if (ev.key !== "Enter") return;
    ev.preventDefault();
    tryUnlockConfig().catch(() => {});
  });
if (cfgCollapseBtnEl) cfgCollapseBtnEl.addEventListener("click", () => toggleConfigCollapsed());
if (bnLinkBtnEl) bnLinkBtnEl.addEventListener("click", () => linkBinance());
if (bnUnlinkBtnEl) bnUnlinkBtnEl.addEventListener("click", () => unlinkBinance());
if (logicOpenBtnEl) logicOpenBtnEl.addEventListener("click", () => setLogicModalOpen(true));
if (summaryRefreshBtnEl)
  summaryRefreshBtnEl.addEventListener("click", () => {
    loadStats(false).catch((e) => {
      const msg = errMessage(e);
      if (handleLockedError(msg)) return;
      setCfgStatus(`거래 통계 새로고침 실패: ${msg}`);
    });
  });
if (centerCloseBtnEl)
  centerCloseBtnEl.addEventListener("click", () => {
    try {
      window.close();
    } catch (_) {}
    if (!window.closed) window.location.href = "/";
  });
if (logicModalCloseBtnEl) logicModalCloseBtnEl.addEventListener("click", () => setLogicModalOpen(false));
if (logicModalEl) {
  logicModalEl.addEventListener("click", (ev) => {
    if (ev.target === logicModalEl) setLogicModalOpen(false);
  });
}
window.addEventListener("keydown", (ev) => {
  if (ev.key !== "Escape") return;
  if (!logicModalEl || logicModalEl.hidden) return;
  setLogicModalOpen(false);
});

initFilters();
initConfigInputs();
initConfigLock();
if (recordsBodyEl) recordsBodyEl.innerHTML = `<tr><td colspan="9" class="records-num">불러오는 중...</td></tr>`;
Promise.all([load(), loadStats()]).catch((e) => alert(e.message || e));

refreshTimer = setInterval(() => {
  if (document.hidden) return;
  load().catch(() => {});
}, 8000);
statsRefreshTimer = setInterval(() => {
  if (document.hidden) return;
  loadStats(false).catch(() => {});
}, STATS_REFRESH_INTERVAL_MS);
runtimeRefreshTimer = setInterval(() => {
  if (document.hidden) return;
  loadRuntimeStatus().catch(() => {});
}, 15000);
window.addEventListener("beforeunload", () => {
  if (refreshTimer) clearInterval(refreshTimer);
  if (statsRefreshTimer) clearInterval(statsRefreshTimer);
  if (collateralRefreshTimer) clearInterval(collateralRefreshTimer);
  if (runtimeRefreshTimer) clearInterval(runtimeRefreshTimer);
});

collateralRefreshTimer = setInterval(() => {
  if (document.hidden) return;
  if (!binanceLinkActive) return;
  loadBinanceCollateral().catch(() => {});
}, 30000);
