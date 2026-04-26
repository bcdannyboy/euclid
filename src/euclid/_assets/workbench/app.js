import katex from "./vendor/katex/katex.mjs";

const state = {
  config: null,
  analysis: null,
  activeTab: "overview",
  hasAutoOpenedAtlas: false,
  hasExplicitTabPreference: false,
  selectedHorizon: null,
  selectedChangeMetric: "return",
  selectedDeterministicOverlay: "descriptive_fit",
  hasExplicitOverlayPreference: false,
  selectedLane: "distribution",
  requestedAnalysisPath: null,
  pendingAtlasSection: null,
  actionSequence: 0,
  pendingActionId: null,
  pendingActionKind: null,
  railCollapsed: false,
  isBusy: false,
};

const HOLISTIC_EQUATION_LABEL = "Holistic equation";
const PREDICTIVE_SYMBOLIC_LAW_LABEL = "Predictive symbolic law";
const DESCRIPTIVE_RECONSTRUCTION_LABEL = "Descriptive reconstruction equation";
const DESCRIPTIVE_APPROXIMATION_LABEL =
  "Best available descriptive approximation";
const BENCHMARK_DESCRIPTIVE_FIT_LABEL = "Benchmark-local descriptive fit";
const NO_BENCHMARK_LOCAL_WINNER_LABEL = "No benchmark-local winner";
const NO_BENCHMARK_LOCAL_WINNER_RUN_COPY =
  `${NO_BENCHMARK_LOCAL_WINNER_LABEL} was selected for this run.`;
const NO_BENCHMARK_LOCAL_WINNER_PUBLICATION_COPY =
  `${NO_BENCHMARK_LOCAL_WINNER_LABEL} was selected; benchmark evidence remains local and does not produce an operator publication.`;

const tabButtons = Array.from(document.querySelectorAll(".tab-button"));
const tabPanels = {
  overview: document.querySelector("#tab-overview"),
  atlas: document.querySelector("#tab-atlas"),
  point: document.querySelector("#tab-point"),
  probabilistic: document.querySelector("#tab-probabilistic"),
  benchmark: document.querySelector("#tab-benchmark"),
  artifacts: document.querySelector("#tab-artifacts"),
};

const analysisForm = document.querySelector("#analysis-form");
const targetSelect = document.querySelector("#target-select");
const targetHint = document.querySelector("#target-hint");
const recentList = document.querySelector("#recent-list");
const recentSection = document.querySelector("#recent-analyses-section");
const shell = document.querySelector(".shell");
const hero = document.querySelector("#hero");
const analystBriefing = document.querySelector("#analyst-briefing");
const statusLine = document.querySelector("#status-line");
const apiKeyState = document.querySelector("#api-key-state");
const startDateInput = analysisForm.elements.namedItem("start_date");
const endDateInput = analysisForm.elements.namedItem("end_date");
const primaryActionButton = analysisForm.querySelector(".primary-action");
const controlRail = document.querySelector(".control-rail");
const railToggleButtons = Array.from(
  document.querySelectorAll("[data-rail-toggle], #control-rail-toggle"),
);
const defaultPrimaryActionLabel = primaryActionButton?.textContent || "Run Analysis";
const rangePresetButtons = Array.from(
  document.querySelectorAll("[data-range-preset]"),
);
const chartRegistry = new Set();

init();

async function init() {
  hydrateStateFromQuery();
  bindTabs();
  bindForm();
  bindRailToggle();
  await loadConfig();
  if (state.requestedAnalysisPath) {
    try {
      await loadAnalysisByPath(state.requestedAnalysisPath, {
        statusMessage: "Loading shared analysis…",
      });
    } catch (error) {
      clearCurrentAnalysis();
      setStatus(error.message || "Failed to load shared analysis.");
    }
  }
  render();
  syncBusyState();
}

function bindTabs() {
  const orderedButtons = tabButtons.filter((button) => button.dataset.tab && tabPanels[button.dataset.tab]);
  const activateTab = (tab, { keepAtlasJump = false } = {}) => {
    if (!tabPanels[tab]) return;
    state.activeTab = tab;
    state.hasExplicitTabPreference = true;
    if (tab !== "atlas" && !keepAtlasJump) {
      state.pendingAtlasSection = null;
    }
    renderTabs();
    syncQueryState();
  };

  for (const [index, button] of orderedButtons.entries()) {
    button.addEventListener("click", () => {
      activateTab(button.dataset.tab);
    });
    button.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
        return;
      }
      event.preventDefault();
      let nextIndex = index;
      if (event.key === "ArrowRight") {
        nextIndex = (index + 1) % orderedButtons.length;
      } else if (event.key === "ArrowLeft") {
        nextIndex = (index - 1 + orderedButtons.length) % orderedButtons.length;
      } else if (event.key === "Home") {
        nextIndex = 0;
      } else if (event.key === "End") {
        nextIndex = orderedButtons.length - 1;
      }
      const nextButton = orderedButtons[nextIndex];
      nextButton?.focus();
      if (nextButton?.dataset.tab) {
        activateTab(nextButton.dataset.tab);
      }
    });
  }
}

function bindForm() {
  targetSelect.addEventListener("change", () => {
    renderTargetHint();
  });
  bindDateRangeValidation();
  bindRangePresets();

  analysisForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (state.isBusy) return;
    const form = new FormData(analysisForm);
    const startDate = emptyToNull(form.get("start_date"));
    const endDate = emptyToNull(form.get("end_date"));
    if (!startDate || !endDate) {
      setStatus("Choose both start and end dates before running an analysis.");
      return;
    }
    if (startDate > endDate) {
      setStatus("Start date must be on or before end date.");
      return;
    }
    const payload = {
      symbol: String(form.get("symbol") || "SPY").trim(),
      target_id: String(form.get("target_id") || state.config?.default_target_id || "daily_return"),
      start_date: startDate,
      end_date: endDate,
      api_key: emptyToNull(form.get("api_key")),
      include_probabilistic: form.get("include_probabilistic") === "on",
      include_benchmark: form.get("include_benchmark") === "on",
    };
    try {
      await runOwnedAction("analyze", async (actionId) => {
        setStatus("Running live Euclid analysis…");
        const analysis = await requestJson("/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!ownsAction(actionId)) return;
        adoptAnalysis(analysis);
        syncFormFromAnalysis(analysis);
        setStatus("Analysis completed.");
        await loadConfig();
        if (!ownsAction(actionId)) return;
        render();
        scrollWorkbenchStart();
      });
    } catch (error) {
      const message = error.message || "Analysis failed.";
      setStatus(message);
      renderClearedAnalysisFailure("Analysis failed", message);
    }
  });
}

function bindRailToggle() {
  for (const button of railToggleButtons) {
    button.addEventListener("click", () => {
      state.railCollapsed = !state.railCollapsed;
      syncShellState();
      syncQueryState();
    });
  }
}

async function loadConfig() {
  try {
    const config = await requestJson("/api/config");
    state.config = config;
    const selectedTarget =
      targetSelect.value || state.analysis?.dataset?.target?.id || config.default_target_id;
    targetSelect.innerHTML = config.target_specs
      .map(
        (target) =>
          `<option value="${escapeHtml(target.id)}">${escapeHtml(target.label)}${target.recommended ? " (Recommended)" : ""}</option>`,
      )
      .join("");
    if (selectedTarget && config.target_specs.some((target) => target.id === selectedTarget)) {
      targetSelect.value = selectedTarget;
    } else if (config.default_target_id) {
      targetSelect.value = config.default_target_id;
    }
    apiKeyState.textContent = config.has_api_key_env
      ? `${config.api_key_env_var} is available`
      : `${config.api_key_env_var} is not set`;
    applyConfigDateDefaults(config);
    renderTargetHint();
    renderRecent(config.recent_analyses || []);
    syncBusyState();
  } catch (error) {
    apiKeyState.textContent = "Failed to load config";
  }
}

function renderRecent(entries) {
  const hasEntries = Array.isArray(entries) && entries.length > 0;
  recentSection?.toggleAttribute("hidden", !hasEntries);
  recentSection?.setAttribute("aria-hidden", String(!hasEntries));
  if (!hasEntries) {
    recentList.innerHTML = "";
    syncBusyState();
    return;
  }
  recentList.innerHTML = entries
    .map(
      (entry) => `
        <article class="recent-item">
          <div class="mini-kicker">${escapeHtml(entry.symbol || "Unknown")} / ${escapeHtml(humanizeKey(entry.target_id || "unknown"))}</div>
          <div class="pill-row recent-pills">
            ${entry.analysis_path === state.analysis?.analysis_path ? pill("current", "ok") : ""}
            ${pill("saved workspace", "")}
          </div>
          <div class="mono path-preview" title="${escapeHtml(entry.workspace_root || "")}">${escapeHtml(truncateMiddle(entry.workspace_root || "", 52))}</div>
          <button data-analysis-path="${escapeHtml(entry.analysis_path)}"${state.isBusy ? " disabled" : ""}>Load saved analysis</button>
        </article>
      `,
    )
    .join("");
  recentList.querySelectorAll("button[data-analysis-path]").forEach((button) => {
    button.addEventListener("click", async () => {
      const analysisPath = button.dataset.analysisPath;
      if (!analysisPath) return;
      try {
        await loadAnalysisByPath(analysisPath);
      } catch (error) {
        const message = error.message || "Failed to load saved analysis.";
        setStatus(message);
        renderClearedAnalysisFailure("Failed to load saved analysis", message);
      }
    });
  });
}

async function loadAnalysisByPath(
  analysisPath,
  { statusMessage = "Loading saved analysis…" } = {},
) {
  if (!analysisPath) return;
  await runOwnedAction("load-analysis", async (actionId) => {
    setStatus(statusMessage);
    const analysis = await requestJson(
      `/api/analysis?analysis_path=${encodeURIComponent(analysisPath)}`,
    );
    if (!ownsAction(actionId)) return;
    adoptAnalysis(analysis);
    syncFormFromAnalysis(analysis);
    setStatus(
      statusMessage === "Loading shared analysis…"
        ? "Shared analysis loaded."
        : "Saved analysis loaded.",
    );
    render();
    scrollWorkbenchStart();
  });
}

function syncFormFromAnalysis(analysis) {
  if (!analysis) return;
  const request = analysis.request || {};
  const symbolInput = analysisForm.elements.namedItem("symbol");
  const probabilisticToggle = analysisForm.elements.namedItem("include_probabilistic");
  const benchmarkToggle = analysisForm.elements.namedItem("include_benchmark");

  if (symbolInput) {
    symbolInput.value = analysis.dataset?.symbol || request.symbol || symbolInput.value;
  }
  if (analysis.dataset?.target?.id) {
    targetSelect.value = analysis.dataset.target.id;
  }
  if (startDateInput) {
    startDateInput.value = normalizeDateInput(
      request.start_date || analysis.dataset?.date_range?.start,
    );
  }
  if (endDateInput) {
    endDateInput.value = normalizeDateInput(
      request.end_date || analysis.dataset?.date_range?.end,
    );
  }
  if (probabilisticToggle) {
    probabilisticToggle.checked =
      request.include_probabilistic ?? Boolean(analysis.probabilistic);
  }
  if (benchmarkToggle) {
    benchmarkToggle.checked =
      request.include_benchmark ?? Boolean(analysis.benchmark);
  }
  validateDateRange();
  renderRangePresetState();
  renderTargetHint();
}

function bindDateRangeValidation() {
  if (startDateInput) {
    startDateInput.addEventListener("change", handleDateRangeInputChange);
  }
  if (endDateInput) {
    endDateInput.addEventListener("change", handleDateRangeInputChange);
  }
}

function bindRangePresets() {
  for (const button of rangePresetButtons) {
    button.addEventListener("click", () => {
      applyDatePreset(button.dataset.rangePreset || "");
    });
  }
}

function handleDateRangeInputChange() {
  validateDateRange();
  renderRangePresetState();
}

function validateDateRange() {
  if (!startDateInput || !endDateInput) return true;
  startDateInput.setCustomValidity("");
  endDateInput.setCustomValidity("");
  if (startDateInput.value && endDateInput.value && startDateInput.value > endDateInput.value) {
    endDateInput.setCustomValidity("End date must be on or after start date.");
    return false;
  }
  return true;
}

function applyConfigDateDefaults(config) {
  if (!startDateInput || !endDateInput) return;
  const defaultRange = config?.default_date_range;
  if (!defaultRange) return;
  if (!startDateInput.value) {
    startDateInput.value = normalizeDateInput(defaultRange.start_date);
  }
  if (!endDateInput.value) {
    endDateInput.value = normalizeDateInput(defaultRange.end_date);
  }
  validateDateRange();
  renderRangePresetState();
}

function applyDatePreset(presetId) {
  const range = presetRangeFromId(presetId);
  if (!range || !startDateInput || !endDateInput) return;
  startDateInput.value = range.start;
  endDateInput.value = range.end;
  validateDateRange();
  renderRangePresetState();
}

function presetRangeFromId(presetId) {
  const anchorDate =
    parseDateInput(endDateInput?.value) ||
    parseDateInput(state.config?.default_date_range?.end_date) ||
    todayUtc();
  const end = formatDateInput(anchorDate);

  switch (presetId) {
    case "ytd":
      return {
        start: formatDateInput(
          new Date(Date.UTC(anchorDate.getUTCFullYear(), 0, 1)),
        ),
        end,
      };
    case "1y":
      return { start: formatDateInput(shiftUtcYears(anchorDate, -1)), end };
    case "3y":
      return { start: formatDateInput(shiftUtcYears(anchorDate, -3)), end };
    case "5y":
      return { start: formatDateInput(shiftUtcYears(anchorDate, -5)), end };
    default:
      return null;
  }
}

function renderRangePresetState() {
  if (!startDateInput || !endDateInput) return;
  for (const button of rangePresetButtons) {
    const range = presetRangeFromId(button.dataset.rangePreset || "");
    const isActive = Boolean(
      range &&
        range.start === startDateInput.value &&
        range.end === endDateInput.value,
    );
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-pressed", String(isActive));
  }
}

function renderTargetHint() {
  if (!targetHint) return;
  const target = findTargetSpec(targetSelect.value);
  if (!target) {
    targetHint.textContent = "";
    return;
  }
  const recommendation = target.recommended
    ? "Recommended for analytical equation inspection."
    : "Available, but not the default analytical target.";
  const note = target.analysis_note || target.description || "";
  targetHint.textContent = `${recommendation} ${note}`.trim();
}

function hydrateStateFromQuery() {
  if (typeof window === "undefined") return;
  const params = new URLSearchParams(window.location.search);
  const requestedTab = String(params.get("tab") || "").trim();
  const requestedHorizon = Number(params.get("horizon"));
  const requestedOverlay = String(params.get("overlay") || "").trim();
  const requestedLane = String(params.get("lane") || "").trim();
  const requestedRail = String(params.get("rail") || "").trim();
  const requestedAnalysisPath = String(params.get("analysis_path") || "").trim();

  if (requestedTab && tabPanels[requestedTab]) {
    state.activeTab = requestedTab;
    state.hasExplicitTabPreference = true;
  }
  if (Number.isFinite(requestedHorizon)) {
    state.selectedHorizon = requestedHorizon;
  }
  if (requestedOverlay) {
    state.selectedDeterministicOverlay = requestedOverlay;
    state.hasExplicitOverlayPreference = true;
  }
  if (requestedLane) {
    state.selectedLane = requestedLane;
  }
  if (requestedRail === "compact") {
    state.railCollapsed = true;
  }
  if (requestedAnalysisPath) {
    state.requestedAnalysisPath = requestedAnalysisPath;
  }
}

function syncShellState() {
  controlRail?.classList.toggle("is-collapsed", state.railCollapsed);
  controlRail?.classList.toggle("compact", state.railCollapsed);
  controlRail?.setAttribute(
    "data-rail-state",
    state.railCollapsed ? "compact" : "expanded",
  );
  document.body.classList.toggle("rail-collapsed", state.railCollapsed);
  document
    .querySelector(".tab-strip")
    ?.setAttribute("role", "tablist");
  for (const button of railToggleButtons) {
    button.setAttribute("aria-expanded", String(!state.railCollapsed));
    button.setAttribute("aria-pressed", String(state.railCollapsed));
    button.setAttribute(
      "aria-label",
      state.railCollapsed ? "Expand analysis controls" : "Compact analysis controls",
    );
    button.textContent = state.railCollapsed ? "Show controls" : "Compact controls";
  }
}

function syncQueryState() {
  if (typeof window === "undefined" || !window.history?.replaceState) return;
  const params = new URLSearchParams(window.location.search);
  params.set("tab", state.activeTab);
  if (
    state.selectedHorizon !== null &&
    state.selectedHorizon !== undefined &&
    Number.isFinite(Number(state.selectedHorizon))
  ) {
    params.set("horizon", String(state.selectedHorizon));
  } else {
    params.delete("horizon");
  }
  if (state.selectedDeterministicOverlay) {
    params.set("overlay", state.selectedDeterministicOverlay);
  } else {
    params.delete("overlay");
  }
  if (state.selectedLane) {
    params.set("lane", state.selectedLane);
  } else {
    params.delete("lane");
  }
  if (state.requestedAnalysisPath) {
    params.set("analysis_path", state.requestedAnalysisPath);
  } else {
    params.delete("analysis_path");
  }
  if (state.railCollapsed) {
    params.set("rail", "compact");
  } else {
    params.delete("rail");
  }
  const nextQuery = params.toString();
  const nextUrl = `${window.location.pathname}${nextQuery ? `?${nextQuery}` : ""}${window.location.hash || ""}`;
  window.history.replaceState(null, "", nextUrl);
}

function renderAnalystBriefing() {
  if (!analystBriefing) return;
  if (!state.analysis) {
    analystBriefing.innerHTML = `
      <div class="briefing-head">
        <div>
          <p class="mini-kicker">Run summary</p>
          <h3>Core run context appears here after the first analysis loads.</h3>
        </div>
      </div>
      <div class="analysis-brief-grid briefing-grid" data-run-summary>
        ${renderBriefCard({
          title: "Publication",
          value: "No analysis loaded",
          note: "Run a saved or live workspace to separate operator publication from benchmark-local and probabilistic evidence.",
          tone: "sea",
        })}
        ${renderBriefCard({
          title: "Benchmark",
          value: "Awaiting run",
          note: "Winner, runner-up, and benchmark-local caveats will appear here once a workspace is loaded.",
          tone: "accent",
        })}
        ${renderBriefCard({
          title: "Workspace",
          value: "Choose a target",
          note: "The transformed target, date window, and active overlays are surfaced here before you dive into the tabs.",
          tone: "moss",
        })}
      </div>
    `;
    return;
  }

  const { dataset, operator_point: point, benchmark, descriptive_fit: descriptiveFit } =
    state.analysis;
  const selectedLaneEntry = selectedProbabilisticLane(state.analysis);
  const hasWinner = hasBenchmarkWinner(benchmark);
  const evidenceSummary =
    summarizeProbabilisticEvidence(state.analysis.probabilistic) ||
    "No thin-evidence warning is active for the probabilistic bundle.";

  analystBriefing.innerHTML = `
    <div class="briefing-head">
      <div>
        <p class="mini-kicker">Run summary</p>
        <h3>${escapeHtml(dataset.symbol)} / ${escapeHtml(dataset.target.label)}</h3>
      </div>
      <div class="pill-row">
        ${pill(humanizePhrase(point?.publication?.status || point?.status || "unknown"), point?.publication?.status === "abstained" ? "warn" : "ok")}
        ${pill(`h${state.selectedHorizon}`, "warn")}
        ${selectedLaneEntry ? pill(humanizeKey(selectedLaneEntry[0]), "ok") : ""}
      </div>
    </div>
    <div class="analysis-brief-grid briefing-grid" data-run-summary>
      ${renderBriefCard({
        title: "Publication",
        value: humanizePhrase(point?.publication?.status || point?.status || "unknown"),
        note:
          pointPublicationHeadlineForDisplay(
            state.analysis,
            "Operator publication state is shown separately from local benchmark selection.",
          ) ||
          "Operator publication state is shown separately from local benchmark selection.",
        tone: "sea",
      })}
      ${renderBriefCard({
        title: "Benchmark",
        value: hasWinner
          ? `Winner ${benchmark?.portfolio_selection?.winner_submitter_id || "n/a"}`
          : NO_BENCHMARK_LOCAL_WINNER_LABEL,
        note:
          descriptiveSurfaceText(
            descriptiveFit?.honesty_note,
            descriptiveFit?.semantic_audit?.headline,
          ) ||
          benchmarkSelectionExplanation(benchmark) ||
          evidenceSummary,
        tone: "accent",
      })}
      ${renderBriefCard({
        title: "Workspace",
        value: `${dataset.rows} rows · ${dataset.date_range.start.slice(0, 10)} → ${dataset.date_range.end.slice(0, 10)}`,
        note: `Target ${dataset.target.label}. Active overlay ${currentDeterministicOverlay(state.analysis)?.label || "n/a"}. Workspace rooted at ${state.analysis.workspace_root}.`,
        tone: "moss",
      })}
    </div>
    <div class="briefing-actions">
      <button type="button" class="subtle-action" data-open-tab="atlas">Open workspace</button>
      <button type="button" class="subtle-action" data-open-tab="benchmark">Inspect benchmark field</button>
      <button type="button" class="subtle-action" data-open-tab="artifacts">Inspect lineage</button>
    </div>
  `;
}

function renderBriefCard({ title, value, note, tone = "accent" }) {
  return `
    <article class="briefing-stat brief-card tone-${escapeHtml(tone)}">
      <div class="detail-label">${escapeHtml(title)}</div>
      <div class="summary-matrix-value brief-value">${escapeHtml(value)}</div>
      <div class="detail-value">${escapeHtml(note)}</div>
    </article>
  `;
}

function renderHeaderStat({ label, value, note = "" }) {
  return `
    <article class="run-stat">
      <div class="detail-label">${escapeHtml(label)}</div>
      <div class="run-stat-value">${escapeHtml(value)}</div>
      ${note ? `<div class="detail-value">${escapeHtml(note)}</div>` : ""}
    </article>
  `;
}

function analysisRunContext(analysis) {
  const dataset = analysis?.dataset || {};
  const target = dataset.target || {};
  const range = dataset.date_range || {};
  return {
    symbol: dataset.symbol || "n/a",
    target: target.label || target.id || "n/a",
    rows: dataset.rows ?? (Array.isArray(dataset.series) ? dataset.series.length : "n/a"),
    start: String(range.start || "n/a").slice(0, 10),
    end: String(range.end || "n/a").slice(0, 10),
    workspace: analysis?.workspace_root || "n/a",
    analysisPath: analysis?.analysis_path || "n/a",
  };
}

function reasonCodeText(values, fallback = "No stronger claim gaps recorded") {
  const items = Array.isArray(values)
    ? values.map(humanizeGapItem).filter(Boolean)
    : [];
  return items.length ? items.slice(0, 3).join(", ") : fallback;
}

function claimSurfaceSummary(analysis) {
  const claim = analysis?.evidence_studio?.claim_surface || {};
  const point = analysis?.operator_point || {};
  const publication = point.publication || {};
  const publicationStatus =
    claim.publication_status ||
    publication.status ||
    point.status ||
    "unknown";
  const rootClaimClass = analysis?.claim_class || "";
  const strongest = strongestEquationCard(analysis);
  const claimCeiling =
    claim.claim_ceiling ||
    claim.claim_lane ||
    rootClaimClass ||
    (publicationStatus === "abstained"
      ? "abstention"
      : strongest?.title || "candidate only");
  const publishable =
    claim.publishable ??
    analysis?.publishable ??
    ["publishable", "published"].includes(String(publicationStatus));
  const reasonCodes = [
    ...(Array.isArray(claim.abstention_reason_codes)
      ? claim.abstention_reason_codes
      : []),
    ...(Array.isArray(claim.downgrade_reason_codes)
      ? claim.downgrade_reason_codes
      : []),
    ...(Array.isArray(publication.reason_codes) ? publication.reason_codes : []),
    ...(Array.isArray(point.abstention?.reason_codes)
      ? point.abstention.reason_codes
      : []),
    ...(Array.isArray(analysis?.would_have_abstained_because)
      ? analysis.would_have_abstained_because
      : []),
    ...(Array.isArray(analysis?.gap_report) ? analysis.gap_report : []),
    ...(Array.isArray(analysis?.not_holistic_because)
      ? analysis.not_holistic_because
      : []),
  ].filter(Boolean);
  return {
    claimCeiling,
    publicationStatus,
    publishable,
    reason: reasonCodeText(
      reasonCodes,
      publishable
        ? "Publication gates recorded no blocking reason in this payload"
        : "No stronger claim recorded in this payload",
    ),
  };
}

function candidateEvidenceSummary(analysis) {
  const strongest = strongestEquationCard(analysis);
  const overlay = currentDeterministicOverlay(analysis);
  const equation = strongest?.equation || overlay?.equation || {};
  return {
    title: strongest?.title || overlay?.label || "No candidate equation retained",
    family:
      equation.family_id ||
      analysis?.descriptive_fit?.family_id ||
      analysis?.operator_point?.selected_family ||
      "n/a",
    candidate:
      analysis?.descriptive_fit?.candidate_id ||
      analysis?.descriptive_reconstruction?.candidate_id ||
      "n/a",
    source:
      analysis?.descriptive_fit?.submitter_id ||
      analysis?.descriptive_fit?.source ||
      overlay?.id ||
      "n/a",
    equationLabel:
      equation.label ||
      equation.structure_signature ||
      overlay?.label ||
      "No explicit equation renderer available",
    note:
      strongest?.honesty ||
      overlay?.honesty ||
      "Candidate identity is shown separately from the claim ceiling.",
  };
}

function residualEvidenceSummaryFor(analysis, activeOverlay) {
  const residualDiagnostics = analysis?.residual_diagnostics || {};
  const residuals = buildResidualSeries(
    analysis?.dataset?.series || [],
    activeOverlay?.series || [],
  );
  const stats = residualSummary(residuals);
  const reasonCodes = Array.isArray(residualDiagnostics.reason_codes)
    ? residualDiagnostics.reason_codes
    : [];
  return {
    status:
      residualDiagnostics.status ||
      residualDiagnostics.finite_dimensionality_status ||
      (residuals.length ? "computed from active overlay" : "not available"),
    mae: stats.mae,
    max: stats.max,
    min: stats.min,
    reason: reasonCodeText(reasonCodes, "Residual law did not raise a stronger claim"),
  };
}

function stochasticSupportSummary(analysis) {
  const selectedLaneEntry = selectedProbabilisticLane(analysis);
  const payload = selectedLaneEntry?.[1] || null;
  const residualRefs = refList(payload, "residual_history_refs");
  const stochasticRefs = refList(payload, "stochastic_model_refs");
  const downgradeReasons =
    payload?.evidence?.downgrade_reason_codes ||
    payload?.downgrade_reason_codes ||
    [];
  return {
    lane: selectedLaneEntry ? humanizeKey(selectedLaneEntry[0]) : "No lane",
    status: humanizePhrase(
      payload?.evidence?.lane_status ||
        payload?.lane_status ||
        payload?.status ||
        "unavailable",
    ),
    family: laneFamily(payload),
    refs:
      stochasticRefs.length || residualRefs.length
        ? `${stochasticRefs.length} stochastic / ${residualRefs.length} residual`
        : "No production refs",
    reason: shortList(downgradeReasons),
  };
}

function calibrationReplaySummary(analysis) {
  const selectedLaneEntry = selectedProbabilisticLane(analysis);
  const payload = selectedLaneEntry?.[1] || null;
  const replayLinks = Array.isArray(analysis?.evidence_studio?.replay_artifacts?.links)
    ? analysis.evidence_studio.replay_artifacts.links
    : [];
  return {
    replay:
      analysis?.operator_point?.replay_verification ||
      payload?.replay_verification ||
      (replayLinks.length ? `${replayLinks.length} replay links` : "n/a"),
    calibration: humanizePhrase(
      payload?.calibration?.status ||
        payload?.calibration?.gate_effect ||
        "unavailable",
    ),
    bins: String(payload ? calibrationBinCount(payload) : "n/a"),
    score: nullableNumber(
      payload?.aggregated_primary_score ??
        analysis?.operator_point?.confirmatory_primary_score,
    ),
  };
}

function buildObservationMiniSvg(analysis, activeOverlay) {
  const actual = Array.isArray(analysis?.dataset?.series) ? analysis.dataset.series : [];
  if (!actual.length) {
    return `<div class="empty-state">No ordered observations are available.</div>`;
  }
  const overlay = Array.isArray(activeOverlay?.series) ? activeOverlay.series : [];
  const width = 520;
  const height = 146;
  const padding = { top: 12, right: 18, bottom: 28, left: 38 };
  const actualValues = actual.map((point) => Number(point.observed_value)).filter(Number.isFinite);
  const overlayValues = overlay.map((point) => Number(point.fitted_value)).filter(Number.isFinite);
  const values = [...actualValues, ...overlayValues];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const xScale = (index, length) =>
    padding.left +
    (length <= 1 ? 0 : (index / (length - 1)) * (width - padding.left - padding.right));
  const yScale = (value) =>
    height -
    padding.bottom -
    ((value - min) / (max - min || 1)) * (height - padding.top - padding.bottom);
  const actualPath = polylinePath(
    actual.map((point, index) => ({
      x: index,
      y: Number(point.observed_value),
    })),
    (index) => xScale(index, actual.length),
    (value) => yScale(value),
    actual.length,
  );
  const overlayPath = overlay.length
    ? polylinePath(
        overlay.map((point, index) => ({
          x: index,
          y: Number(point.fitted_value),
        })),
        (index) => xScale(index, overlay.length),
        (value) => yScale(value),
        overlay.length,
      )
    : "";
  return `
    <svg class="observation-mini-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Ordered observation timeline">
      <line x1="${padding.left}" y1="${height - padding.bottom}" x2="${width - padding.right}" y2="${height - padding.bottom}" />
      <path d="${actualPath}" class="observed-path"></path>
      ${overlayPath ? `<path d="${overlayPath}" class="candidate-path"></path>` : ""}
      <text x="${padding.left}" y="${height - 9}">${escapeHtml(String(actual[0]?.event_time || "").slice(0, 10))}</text>
      <text x="${width - padding.right - 78}" y="${height - 9}">${escapeHtml(String(actual.at(-1)?.event_time || "").slice(0, 10))}</text>
    </svg>
  `;
}

function renderEuclidFirstScreen(analysis) {
  if (!analysis) return "";
  const context = analysisRunContext(analysis);
  const claim = claimSurfaceSummary(analysis);
  const activeOverlay = currentDeterministicOverlay(analysis);
  const candidate = candidateEvidenceSummary(analysis);
  const residual = residualEvidenceSummaryFor(analysis, activeOverlay);
  const stochastic = stochasticSupportSummary(analysis);
  const calibrationReplay = calibrationReplaySummary(analysis);
  return `
    <section class="euclid-first-screen" data-euclid-first-screen="spine" aria-label="Euclid evidence spine">
      <div class="run-context-strip">
        ${detail("Target", `${context.symbol} / ${context.target}`)}
        ${detail("Ordered rows", String(context.rows))}
        ${detail("Date range", `${context.start} to ${context.end}`)}
        ${detail("Analysis", truncateMiddle(context.analysisPath, 48))}
      </div>
      <div class="evidence-spine">
        <article class="evidence-node node-observations">
          <div class="detail-label">Ordered observations</div>
          <h3>${escapeHtml(context.symbol)} path</h3>
          ${buildObservationMiniSvg(analysis, activeOverlay)}
          <div class="detail-grid compact-details">
            ${detail("Rows", String(context.rows))}
            ${detail("Range", `${context.start} to ${context.end}`)}
          </div>
        </article>
        <article class="evidence-node node-candidate">
          <div class="detail-label">Candidate equation</div>
          <h3>${escapeHtml(candidate.title)}</h3>
          <div class="equation-copy compact-equation">
            ${renderEquationMarkup(candidate.equationLabel, {
              className: "equation-formula",
              displayMode: true,
            })}
          </div>
          <div class="detail-grid compact-details">
            ${detail("Family", candidate.family)}
            ${detail("Candidate", candidate.candidate)}
            ${detail("Source", candidate.source)}
          </div>
        </article>
        <article class="evidence-node node-evidence">
          <div class="detail-label">Residual evidence</div>
          <h3>${escapeHtml(humanizePhrase(residual.status))}</h3>
          <div class="gate-list">
            ${detail("Residual MAE", residual.mae)}
            ${detail("Residual max/min", `${residual.max} / ${residual.min}`)}
            ${detail("Stochastic support", `${stochastic.status} · ${stochastic.refs}`)}
            ${detail("Calibration", `${calibrationReplay.calibration} · ${calibrationReplay.bins} bins`)}
            ${detail("Replay", calibrationReplay.replay)}
            ${detail("Primary score", calibrationReplay.score)}
          </div>
        </article>
        <article class="evidence-node node-claim">
          <div class="detail-label">Claim ceiling</div>
          <h3>${escapeHtml(humanizePhrase(claim.claimCeiling))}</h3>
          <div class="claim-gate-strip">
            ${pill("Publication gate", claim.publishable ? "ok" : "warn")}
            ${pill(humanizePhrase(claim.publicationStatus), claim.publishable ? "ok" : "warn")}
            ${pill(`Stochastic support: ${stochastic.lane}`, "sea")}
          </div>
          <div class="gate-list">
            ${detail("Publication gate", humanizePhrase(claim.publicationStatus))}
            ${detail("Abstention / downgrade", claim.reason)}
            ${detail("Stochastic family", stochastic.family)}
            ${detail("Stochastic reason", stochastic.reason)}
          </div>
        </article>
      </div>
    </section>
  `;
}

function pointLagDays(point) {
  const label = String(point?.equation?.label || "");
  const match = label.match(/y\(t\)\s*=\s*y\(t\s*-\s*(\d+)\)/i);
  return Number(match?.[1] || 1);
}

function pointLagNarrative(point) {
  const lag = pointLagDays(point);
  return `Today's point equals the close from ${lag} trading day${lag === 1 ? "" : "s"} earlier.`;
}

function activeLaneSnapshot(selectedLaneEntry) {
  if (!selectedLaneEntry) return "n/a";
  const row = rowForSelectedHorizon(selectedLaneEntry[1]);
  const summary = normalizeLatestRow(selectedLaneEntry[0], row);
  return (
    summary.find(([label]) =>
      /center|q0\.5|upper|probability|realized/i.test(String(label || "")),
    )?.[1] ||
    summary[0]?.[1] ||
    "n/a"
  );
}

function benchmarkOutcomeDetails(benchmark) {
  const winner = benchmark?.portfolio_selection?.winner_submitter_id || NO_BENCHMARK_LOCAL_WINNER_LABEL;
  const runnerUpRaw =
    benchmark?.portfolio_selection?.selection_explanation_raw?.runner_up || null;
  const runnerUp = runnerUpRaw?.submitter_id || "n/a";
  const winnerBits = benchmark?.submitters?.find(
    (submitter) => submitter.submitter_id === winner,
  )?.selected_candidate_metrics?.total_code_bits;
  const runnerUpBits =
    runnerUpRaw?.total_code_bits ??
    benchmark?.submitters?.find(
      (submitter) => submitter.submitter_id === runnerUp,
    )?.selected_candidate_metrics?.total_code_bits;
  const margin =
    Number.isFinite(Number(winnerBits)) && Number.isFinite(Number(runnerUpBits))
      ? formatNumber(Math.abs(Number(runnerUpBits) - Number(winnerBits)))
      : "n/a";
  return {
    winner,
    runnerUp,
    margin,
    winnerBits: nullableNumber(winnerBits),
    runnerUpBits: nullableNumber(runnerUpBits),
    rule: humanizePhrase(
      benchmark?.portfolio_selection?.selection_explanation_raw?.selection_rule || "n/a",
    ),
  };
}

function overlayPlainLanguageSummary(activeOverlay, analysis) {
  if (!activeOverlay) {
    return "Choose an active overlay to inspect the equation against the observed path.";
  }
  if (activeOverlay.id === "point") {
    return pointLagNarrative(analysis?.operator_point);
  }
  return (
    activeOverlay.honesty ||
    "Inspect the displayed equation against the observed path and residual context."
  );
}

function renderWorkspaceEquationStage({ analysis, activeOverlay, selectedLaneEntry }) {
  const rowCount = analysis?.dataset?.rows || "n/a";
  return `
    <section class="panel workspace-stage" data-workspace-region="equation-stage">
      <div class="panel-head">
        <div>
          <p class="mini-kicker">Equation stage</p>
          <h3>${escapeHtml(activeOverlay?.label || "Active deterministic overlay")}</h3>
        </div>
        <div class="pill-row">
          ${pill(activeOverlay?.shortLabel || "overlay", "ok")}
          ${selectedLaneEntry ? pill(humanizeKey(selectedLaneEntry[0]), "sea") : ""}
        </div>
      </div>
      <p class="workspace-stage-summary">${escapeHtml(overlayPlainLanguageSummary(activeOverlay, analysis))}</p>
      <div class="equation-copy workspace-equation-copy">
        ${renderEquationMarkup(activeOverlay?.equation?.label || activeOverlay?.equation?.structure_signature || "No explicit renderer available.", {
          className: "equation-formula",
          displayMode: true,
        })}
        ${activeOverlay?.equation?.delta_form_label
          ? renderEquationMarkup(activeOverlay.equation.delta_form_label, {
              className: "equation-delta",
              displayMode: false,
            })
          : ""}
      </div>
      ${activeOverlay?.honesty ? banner("Interpretation", activeOverlay.honesty) : ""}
      <div class="detail-grid">
        ${detail("Target", analysis?.dataset?.target?.label || "n/a")}
        ${detail("Selected horizon", `h${state.selectedHorizon}`)}
        ${detail("Selected lane snapshot", activeLaneSnapshot(selectedLaneEntry))}
        ${detail("Rows", String(rowCount))}
      </div>
    </section>
  `;
}

function renderWorkspaceEvidenceRail({ analysis, activeOverlay, selectedLaneEntry }) {
  const benchmark = analysis?.benchmark;
  const outcome = benchmarkOutcomeDetails(benchmark);
  return `
    <section class="panel diagnostic-chart evidence-rail workspace-evidence-rail" data-workspace-region="evidence-rail">
      <div class="panel-head">
        <div>
          <p class="mini-kicker">Evidence rail</p>
          <h3>Decision guardrails</h3>
        </div>
      </div>
      ${renderSharedControls({ analysis })}
      <p>The canvas keeps the object itself central. This rail holds the active selectors, benchmark context, and thin-evidence caveats in one narrow reference surface.</p>
      <div class="detail-grid">
        ${detail("Active overlay", activeOverlay?.label || "n/a")}
        ${detail("Operator publication", humanizePhrase(analysis?.operator_point?.publication?.status || analysis?.operator_point?.status || "n/a"))}
        ${detail("Selected lane", selectedLaneEntry ? humanizeKey(selectedLaneEntry[0]) : "n/a")}
        ${detail("Benchmark winner", outcome.winner)}
        ${detail("Runner-up", outcome.runnerUp)}
        ${detail("Workspace", truncateMiddle(analysis?.workspace_root || "n/a", 44))}
      </div>
      ${selectedLaneEntry?.[1]?.evidence?.headline
        ? banner("Selected lane", selectedLaneEntry[1].evidence.headline)
        : ""}
      ${renderEvidenceStudioSummary(analysis)}
    </section>
  `;
}

function renderEvidenceStudioSummary(analysis) {
  const studio = analysis?.evidence_studio;
  if (!studio) return "";
  const claim = studio.claim_surface || {};
  const live = studio.live_evidence || {};
  const replayLinks = Array.isArray(studio.replay_artifacts?.links)
    ? studio.replay_artifacts.links
    : [];
  const engine = studio.engine_provenance?.point_lane || {};
  const abstentions = Array.isArray(claim.abstention_reason_codes)
    ? claim.abstention_reason_codes
    : [];
  const downgrades = Array.isArray(claim.downgrade_reason_codes)
    ? claim.downgrade_reason_codes
    : [];
  const boundary = live.claim_boundary || claim.live_evidence_boundary || {};
  return `
    <div class="stack-tight evidence-studio-summary">
      <div class="detail-label">Evidence Studio</div>
      <div class="pill-row">
        ${pill(humanizePhrase(claim.claim_lane || "none"), claim.publishable ? "ok" : "warn")}
        ${pill(humanizePhrase(live.status || "live evidence unavailable"), "sea")}
        ${boundary.counts_as_scientific_claim_evidence === false ? pill("non-claim live evidence", "accent") : ""}
      </div>
      <div class="detail-grid">
        ${detail("Claim ceiling", humanizePhrase(claim.claim_ceiling || "none"))}
        ${detail("Abstention", summarizeReasonCodes(abstentions))}
        ${detail("Downgrade", summarizeReasonCodes(downgrades))}
        ${detail("Replay links", String(replayLinks.length))}
        ${detail("Engine", engine.engine_id || engine.selected_family || "n/a")}
      </div>
    </div>
  `;
}

function summarizeReasonCodes(items) {
  const normalized = Array.isArray(items)
    ? items.map((item) => humanizeGapItem(item)).filter(Boolean)
    : [];
  if (!normalized.length) return "none";
  return normalized.slice(0, 3).join(", ");
}

function renderStatusMatrix(analysis) {
  if (!analysis) return "";
  const point = analysis.operator_point;
  const benchmark = analysis.benchmark;
  const descriptiveFit = analysis.descriptive_fit;
  const selectedLaneEntry = selectedProbabilisticLane(analysis);
  const hasWinner = hasBenchmarkWinner(benchmark);
  const items = [
    {
      title: "Operator lane",
      value: humanizePhrase(point?.publication?.status || point?.status || "unknown"),
      note:
        pointPublicationHeadlineForDisplay(
          analysis,
          "Point-lane publication status governs the operator publication path.",
        ) ||
        "Point-lane publication status governs the operator publication path.",
      tone: point?.publication?.status === "abstained" ? "accent" : "sea",
      tab: "point",
    },
    {
      title: BENCHMARK_DESCRIPTIVE_FIT_LABEL,
      value: descriptiveFit?.status === "completed"
        ? descriptiveFit.family_id || "available"
        : NO_BENCHMARK_LOCAL_WINNER_LABEL,
      note:
        descriptiveSurfaceText(
          descriptiveFit?.honesty_note,
          descriptiveFit?.semantic_audit?.headline,
          benchmark?.descriptive_fit_status?.headline,
        ) ||
        `${BENCHMARK_DESCRIPTIVE_FIT_LABEL} stays separate from the point lane.`,
      tone: "accent",
      tab: "atlas",
    },
    {
      title: "Probabilistic lane",
      value: selectedLaneEntry ? humanizeKey(selectedLaneEntry[0]) : "Unavailable",
      note:
        selectedLaneEntry?.[1]?.evidence?.headline ||
        summarizeProbabilisticEvidence(analysis.probabilistic) ||
        "Probabilistic evidence is summarized at the selected horizon.",
      tone: "moss",
      tab: "probabilistic",
    },
    {
      title: "Benchmark field",
      value: hasWinner
        ? benchmark?.portfolio_selection?.winner_submitter_id || "Winner selected"
        : NO_BENCHMARK_LOCAL_WINNER_LABEL,
      note:
        benchmarkSelectionExplanation(benchmark) ||
        "Benchmark-local selection remains distinct from the operator publication.",
      tone: hasWinner ? "sea" : "accent",
      tab: "benchmark",
    },
  ];

  return `
    <div class="summary-matrix status-matrix">
      ${items
        .map(
          (item) => `
            <article class="summary-matrix-cell matrix-card tone-${escapeHtml(item.tone)}">
              <div class="detail-label">${escapeHtml(item.title)}</div>
              <div class="summary-matrix-value matrix-value">${escapeHtml(item.value)}</div>
              <div class="detail-value">${escapeHtml(item.note)}</div>
              <button type="button" class="subtle-action" data-open-tab="${escapeHtml(item.tab)}">Open ${escapeHtml(item.title)}</button>
            </article>
          `,
        )
        .join("")}
    </div>
  `;
}

function render() {
  const analysisState = state.analysis ? "loaded" : "empty";
  document.body.dataset.analysisState = analysisState;
  shell?.setAttribute("data-analysis-state", analysisState);
  disposeAnalyticalCharts();
  renderRecent(state.config?.recent_analyses || []);
  renderAnalystBriefing();
  renderTabs();
  syncShellState();
  renderHero();
  renderOverview();
  renderAtlas();
  renderPoint();
  renderProbabilistic();
  renderBenchmark();
  renderArtifacts();
  bindDynamicControls();
  mountAnalyticalCharts();
  maybeScrollAtlasSection();
  syncQueryState();
}

function clearCurrentAnalysis() {
  state.analysis = null;
  state.requestedAnalysisPath = null;
}

function renderClearedAnalysisFailure(title, message) {
  clearCurrentAnalysis();
  state.activeTab = "overview";
  render();
  renderFailure(tabPanels.overview, title, message);
}

function renderTabs() {
  for (const button of tabButtons) {
    const tab = button.dataset.tab;
    const isActive = tab === state.activeTab;
    const panel = tabPanels[tab];
    const buttonId = button.id || `tab-trigger-${tab}`;
    const panelId = panel?.id || `tab-${tab}`;
    button.id = buttonId;
    button.setAttribute("role", "tab");
    button.setAttribute("aria-controls", panelId);
    button.setAttribute("aria-selected", String(isActive));
    button.setAttribute("tabindex", isActive ? "0" : "-1");
    button.classList.toggle("active", isActive);
  }
  for (const [tab, panel] of Object.entries(tabPanels)) {
    const isActive = tab === state.activeTab;
    panel.setAttribute("role", "tabpanel");
    panel.setAttribute("aria-labelledby", `tab-trigger-${tab}`);
    panel.classList.toggle("active", isActive);
    panel.toggleAttribute("hidden", !isActive);
  }
}

function adoptAnalysis(analysis) {
  const preferredHorizon = Number(state.selectedHorizon);
  const preferredChangeMetric = String(state.selectedChangeMetric || "").trim();
  const preferredOverlay = String(state.selectedDeterministicOverlay || "").trim();
  const preferredLane = String(state.selectedLane || "").trim();
  state.analysis = analysis;
  state.requestedAnalysisPath = analysis.analysis_path || state.requestedAnalysisPath;
  state.railCollapsed = true;

  const horizons = availableHorizons(analysis);
  const changeMetrics = availableChangeMetrics(analysis);
  const overlays = availableDeterministicOverlays(analysis);
  const lanes = completedProbabilisticLaneEntries(analysis);

  state.selectedHorizon = horizons.includes(preferredHorizon)
    ? preferredHorizon
    : defaultSelectedHorizon(analysis);
  state.selectedChangeMetric = changeMetrics.some(
    (metric) => metric.id === preferredChangeMetric,
  )
    ? preferredChangeMetric
    : defaultSelectedChangeMetric(analysis);
  state.selectedDeterministicOverlay = state.hasExplicitOverlayPreference &&
    overlays.some((overlay) => overlay.id === preferredOverlay)
    ? preferredOverlay
    : defaultDeterministicOverlay(analysis);
  state.selectedLane = lanes.some(([mode]) => mode === preferredLane)
    ? preferredLane
    : defaultSelectedLane(analysis);

  if (!state.hasAutoOpenedAtlas && !state.hasExplicitTabPreference) {
    state.activeTab = "atlas";
    state.hasAutoOpenedAtlas = true;
  }
}

function bindDynamicControls() {
  document.querySelectorAll("[data-horizon]").forEach((button) => {
    button.addEventListener("click", () => {
      const next = Number(button.dataset.horizon);
      if (!Number.isFinite(next)) return;
      state.selectedHorizon = next;
      render();
    });
  });
  document.querySelectorAll("[data-overlay]").forEach((button) => {
    button.addEventListener("click", () => {
      const next = String(button.dataset.overlay || "").trim();
      if (!next) return;
      state.selectedDeterministicOverlay = next;
      state.hasExplicitOverlayPreference = true;
      render();
    });
  });
  document.querySelectorAll("[data-lane-select]").forEach((button) => {
    button.addEventListener("click", () => {
      const next = String(button.dataset.laneSelect || "").trim();
      if (!next) return;
      state.selectedLane = next;
      render();
    });
  });
  document.querySelectorAll("[data-change-metric]").forEach((button) => {
    button.addEventListener("click", () => {
      const next = String(button.dataset.changeMetric || "").trim();
      if (!next) return;
      state.selectedChangeMetric = next;
      render();
    });
  });
  document.querySelectorAll("[data-open-tab]").forEach((button) => {
    button.addEventListener("click", () => {
      const next = String(button.dataset.openTab || "").trim();
      if (!next || !tabPanels[next]) return;
      state.activeTab = next;
      state.hasExplicitTabPreference = true;
      if (next !== "atlas") {
        state.pendingAtlasSection = null;
      }
      render();
    });
  });
  document.querySelectorAll("[data-atlas-jump]").forEach((button) => {
    button.addEventListener("click", () => {
      state.activeTab = "atlas";
      state.hasExplicitTabPreference = true;
      state.pendingAtlasSection = String(button.dataset.atlasJump || "").trim() || null;
      render();
    });
  });
  document.querySelectorAll("[data-copy-value]").forEach((button) => {
    button.addEventListener("click", async () => {
      const value = String(button.dataset.copyValue || "");
      if (!value) return;
      const label = button.dataset.copyLabel || "Path";
      try {
        await copyText(value);
        setStatus(`${label} copied to clipboard.`);
      } catch (error) {
        setStatus(`Unable to copy ${String(label).toLowerCase()}.`);
      }
    });
  });
}

function maybeScrollAtlasSection() {
  if (state.activeTab !== "atlas" || !state.pendingAtlasSection) return;
  const target = document.querySelector(`#${CSS.escape(state.pendingAtlasSection)}`);
  if (target && typeof target.scrollIntoView === "function") {
    target.scrollIntoView({ block: "start", behavior: "smooth" });
  }
  state.pendingAtlasSection = null;
}

function scrollWorkbenchStart() {
  const target = hero || document.querySelector(".stage");
  if (!target || typeof target.scrollIntoView !== "function") return;
  const schedule =
    typeof window !== "undefined" && typeof window.requestAnimationFrame === "function"
      ? window.requestAnimationFrame.bind(window)
      : (callback) => callback();
  schedule(() => {
    if (typeof window === "undefined" || typeof window.scrollTo !== "function") {
      target.scrollIntoView({ block: "start" });
      return;
    }
    const top = target.getBoundingClientRect().top + window.scrollY - 8;
    try {
      window.scrollTo({ top: Math.max(0, top), behavior: "auto" });
    } catch (_error) {
      target.scrollIntoView({ block: "start" });
    }
  });
}

function renderHero() {
  if (!state.analysis) {
    hero.innerHTML = `
      <div class="run-header run-header-empty" data-analysis-header>
        <div class="run-header-main">
          <div class="run-header-copy">
            <p class="hero-kicker">No analysis loaded</p>
            <h2>Choose ordered observations and run Euclid.</h2>
            <p class="hero-copy">
              The first loaded run will show observations, candidate equation, evidence,
              publication gate, and claim ceiling in one inspection surface.
            </p>
          </div>
        </div>
      </div>
    `;
    return;
  }
  const { dataset, operator_point: point, benchmark } = state.analysis;
  const selectedLaneEntry = selectedProbabilisticLane(state.analysis);
  const benchmarkOutcome = benchmarkOutcomeDetails(benchmark);
  const contextLine = `${dataset.rows} rows · ${dataset.date_range.start.slice(0, 10)} → ${dataset.date_range.end.slice(0, 10)}`;
  const summaryLine = buildHeroSummary(state.analysis);
  const claim = claimSurfaceSummary(state.analysis);

  hero.innerHTML = `
    <div class="run-header run-header-loaded" data-analysis-header>
      <div class="run-header-main">
        <div class="run-header-copy">
          <p class="hero-kicker">Active evidence workbench</p>
          <h2>${escapeHtml(dataset.symbol)} · ${escapeHtml(dataset.target.label)}</h2>
          <p class="hero-copy">${escapeHtml(summaryLine)}</p>
        </div>
        <div class="run-header-pills pill-row">
          ${pill(`Claim ceiling: ${humanizePhrase(claim.claimCeiling)}`, claim.publishable ? "ok" : "warn")}
          ${pill(humanizePhrase(point?.publication?.status || point?.status || "unknown"), point?.publication?.status === "abstained" ? "warn" : "ok")}
          ${selectedLaneEntry ? pill(humanizeKey(selectedLaneEntry[0]), "sea") : ""}
          ${pill(`h${state.selectedHorizon}`, "warn")}
        </div>
      </div>
      <div class="run-stat-grid" data-run-summary>
        ${renderHeaderStat({
          label: "Dataset",
          value: `${dataset.symbol} / ${dataset.target.label}`,
          note: contextLine,
        })}
        ${renderHeaderStat({
          label: "Operator",
          value: humanizePhrase(point?.publication?.status || point?.status || "unknown"),
          note: point?.selected_family || "No point family selected",
        })}
        ${renderHeaderStat({
          label: "Probabilistic",
          value: selectedLaneEntry ? humanizeKey(selectedLaneEntry[0]) : "Unavailable",
          note: selectedLaneEntry?.[1]?.evidence?.headline || "No probabilistic lane selected.",
        })}
        ${renderHeaderStat({
          label: "Benchmark",
          value: benchmarkOutcome.winner,
          note: benchmarkOutcome.runnerUp !== "n/a"
            ? `Runner-up ${benchmarkOutcome.runnerUp} · margin ${benchmarkOutcome.margin} code bits`
            : benchmarkSelectionExplanation(benchmark),
        })}
      </div>
      ${renderEuclidFirstScreen(state.analysis)}
    </div>
  `;
}

function availableHorizons(analysis) {
  const horizons = new Set();
  const probabilistic = analysis?.probabilistic;
  if (!probabilistic || typeof probabilistic !== "object") {
    return [1];
  }
  for (const payload of Object.values(probabilistic)) {
    if (!payload || payload.status !== "completed") continue;
    const rows = Array.isArray(payload.rows) ? payload.rows : [];
    for (const row of rows) {
      const horizon = Number(row?.horizon);
      if (Number.isFinite(horizon)) {
        horizons.add(horizon);
      }
    }
    const latestHorizon = Number(payload.latest_row?.horizon);
    if (Number.isFinite(latestHorizon)) {
      horizons.add(latestHorizon);
    }
  }
  return Array.from(horizons).sort((left, right) => left - right).length
    ? Array.from(horizons).sort((left, right) => left - right)
    : [1];
}

function defaultSelectedHorizon(analysis) {
  const horizons = availableHorizons(analysis);
  return horizons.at(-1) || 1;
}

function availableChangeMetrics(analysis) {
  const metrics = Array.isArray(analysis?.change_atlas?.metrics)
    ? analysis.change_atlas.metrics
    : [];
  return metrics.filter((metric) => metric?.id);
}

function defaultSelectedChangeMetric(analysis) {
  const metrics = availableChangeMetrics(analysis);
  return (
    metrics.find((metric) => metric.id === "return")?.id ||
    metrics[0]?.id ||
    "return"
  );
}

function hasCompletedClaim(payload, claimClass) {
  return payload?.status === "completed" && payload?.claim_class === claimClass;
}

function hasRequiredRefs(values) {
  return values.every((value) => String(value || "").trim());
}

function hasNormalizedEquationPayload(equation) {
  return Boolean(
    equation &&
      typeof equation === "object" &&
      (
        String(
          equation.label ||
            equation.structure_signature ||
            equation.delta_form_label ||
            equation.residual_label ||
            equation.coefficient_vector_label ||
            "",
        ).trim() ||
        (Array.isArray(equation.curve) && equation.curve.length)
      ),
  );
}

function hasHonestyNote(payload) {
  return Boolean(String(payload?.honesty_note || "").trim());
}

function hasLegacyHolisticMetadata(holisticEquation) {
  return ["mode", "composition_operator", "selected_probabilistic_lane"].some(
    (key) => Boolean(String(holisticEquation?.[key] || "").trim()),
  );
}

function isSyntheticViewSource(value) {
  return String(value || "").trim().startsWith("synthetic_views.");
}

function hasAuthoritativeLawState(analysis, claimClass) {
  const publicationStatus = String(
    analysis?.operator_point?.publication?.status || "",
  )
    .trim()
    .toLowerCase();
  const hasAbstentionArtifact =
    analysis?.operator_point?.abstention !== null &&
    analysis?.operator_point?.abstention !== undefined;
  const blockedCeiling = String(
    analysis?.operator_point?.abstention?.blocked_ceiling || "",
  )
    .trim()
    .toLowerCase();
  const wouldHaveAbstainedBecause = Array.isArray(
    analysis?.would_have_abstained_because,
  )
    ? analysis.would_have_abstained_because.filter((reason) =>
        String(reason || "").trim(),
      )
    : [];
  return (
    analysis?.claim_class === claimClass &&
    publicationStatus === "publishable" &&
    !hasAbstentionArtifact &&
    blockedCeiling !== "descriptive_only" &&
    blockedCeiling !== "descriptive-only" &&
    !wouldHaveAbstainedBecause.length
  );
}

function hasHolisticClaim(analysis) {
  const holisticEquation = analysis?.holistic_equation;
  const gapReport = Array.isArray(analysis?.gap_report) ? analysis.gap_report : [];
  const notHolisticBecause = Array.isArray(analysis?.not_holistic_because)
    ? analysis.not_holistic_because
    : [];
  return Boolean(
    hasCompletedClaim(holisticEquation, "holistic_equation") &&
      hasAuthoritativeLawState(analysis, "holistic_equation") &&
      hasNormalizedEquationPayload(holisticEquation?.equation) &&
      hasHonestyNote(holisticEquation) &&
      hasRequiredRefs([
        holisticEquation?.deterministic_source,
        holisticEquation?.probabilistic_source,
        holisticEquation?.validation_scope_ref,
        holisticEquation?.publication_record_ref,
      ]) &&
      !hasLegacyHolisticMetadata(holisticEquation) &&
      !isSyntheticViewSource(holisticEquation?.deterministic_source) &&
      !isSyntheticViewSource(holisticEquation?.probabilistic_source) &&
      holisticEquation?.exactness !== "sample_exact_closure" &&
      !notHolisticBecause.length &&
      !gapReport.includes("requires_exact_sample_closure") &&
      !gapReport.includes("requires_posthoc_symbolic_synthesis") &&
      !gapReport.includes("no_backend_joint_claim"),
  );
}

function hasPredictiveLaw(analysis) {
  const predictiveLaw = analysis?.predictive_law;
  return Boolean(
    hasCompletedClaim(predictiveLaw, "predictive_law") &&
      hasAuthoritativeLawState(analysis, "predictive_law") &&
      hasNormalizedEquationPayload(predictiveLaw?.equation) &&
      hasHonestyNote(predictiveLaw) &&
      hasRequiredRefs([
        predictiveLaw?.claim_card_ref,
        predictiveLaw?.scorecard_ref,
        predictiveLaw?.validation_scope_ref,
        predictiveLaw?.publication_record_ref,
      ]),
  );
}

function hasDescriptiveReconstruction(analysis) {
  const reconstruction = analysis?.descriptive_reconstruction;
  return Boolean(
    hasCompletedClaim(reconstruction, "descriptive_reconstruction") &&
      hasNormalizedEquationPayload(reconstruction?.equation) &&
      hasHonestyNote(reconstruction),
  );
}

function holisticClaimDescriptor(holisticEquation) {
  return holisticEquation?.exactness
    ? humanizePhrase(holisticEquation.exactness)
    : "Holistic claim";
}

function strongestEquationCard(analysis) {
  if (!analysis) return null;
  const holisticEquation = analysis.holistic_equation;
  const predictiveLaw = analysis.predictive_law;
  const descriptiveReconstruction = analysis.descriptive_reconstruction;
  const descriptiveFit = analysis.descriptive_fit;

  if (hasHolisticClaim(analysis)) {
    return {
      title: HOLISTIC_EQUATION_LABEL,
      kicker: "Hero output",
      equation: holisticEquation.equation,
      badges: [
        {
          label: holisticClaimDescriptor(holisticEquation),
          tone: "ok",
        },
        {
          label: claimSurfaceLabel(
            holisticEquation.deterministic_source ||
              holisticEquation.source ||
              "joint claim",
          ),
          tone: "",
        },
      ],
      honesty: holisticEquation.honesty_note || "",
      note:
        holisticEquation.equation?.coefficient_vector_label ||
        holisticEquation.equation?.residual_label ||
        "",
      tone: "accent",
      summaryRows: [
        [
          "Deterministic source",
          claimSurfaceLabel(
            holisticEquation.deterministic_source ||
              holisticEquation.source ||
              "n/a",
          ),
        ],
        [
          "Probabilistic source",
          claimSurfaceLabel(holisticEquation.probabilistic_source || "n/a"),
        ],
        ["Validation scope", holisticEquation.validation_scope_ref || "n/a"],
        ["Publication record", holisticEquation.publication_record_ref || "n/a"],
      ],
    };
  }

  if (hasPredictiveLaw(analysis)) {
    const badges = [{ label: PREDICTIVE_SYMBOLIC_LAW_LABEL, tone: "ok" }];
    if (String(analysis.operator_point?.publication?.status || "").trim()) {
      badges.push({
        label: humanizePhrase(analysis.operator_point.publication.status),
        tone: "",
      });
    }
    return {
      title: PREDICTIVE_SYMBOLIC_LAW_LABEL,
      kicker: "Hero output",
      equation: predictiveLaw.equation,
      parameters: predictiveLaw.equation?.parameter_summary || {},
      badges,
      honesty: predictiveLaw.honesty_note,
      note: predictiveLaw.equation?.delta_form_label || "",
      tone: "sea",
      summaryRows: [
        ["Claim card", predictiveLaw.claim_card_ref || "n/a"],
        ["Scorecard", predictiveLaw.scorecard_ref || "n/a"],
        ["Validation scope", predictiveLaw.validation_scope_ref || "n/a"],
        ["Publication record", predictiveLaw.publication_record_ref || "n/a"],
      ],
    };
  }

  if (hasDescriptiveReconstruction(analysis)) {
    return {
      title: DESCRIPTIVE_RECONSTRUCTION_LABEL,
      kicker: "Hero output",
      equation: descriptiveReconstruction.equation,
      parameters: descriptiveReconstruction.equation?.parameter_summary || {},
      badges: [
        {
          label: "Descriptive only",
          tone: "ok",
        },
        {
          label: humanizePhrase(
            descriptiveReconstruction.source || "workbench reconstruction",
          ),
          tone: "",
        },
      ],
      honesty: descriptiveReconstruction.honesty_note || "",
      note:
        descriptiveReconstruction.equation?.coefficient_vector_label ||
        "",
      tone: "accent",
      summaryRows: [
        ["Source", descriptiveReconstruction.source || "n/a"],
        ["Candidate", descriptiveReconstruction.candidate_id || "n/a"],
        [
          "R² vs mean",
          formatCell(
            descriptiveReconstruction.reconstruction_metrics?.r2_vs_mean_baseline,
          ),
        ],
      ],
    };
  }

  if (descriptiveFit?.status === "completed") {
    return {
      title: DESCRIPTIVE_APPROXIMATION_LABEL,
      kicker: "Hero output",
      equation: descriptiveFit.equation,
      parameters: descriptiveFit.equation?.parameter_summary || {},
      badges: [
        {
          label:
            descriptiveFit.family_id || "descriptive approximation",
          tone: "ok",
        },
        {
          label: humanizePhrase(
            descriptiveFit.semantic_audit?.classification ||
              "descriptive approximation",
          ),
          tone:
            descriptiveFit.semantic_audit?.classification === "near_persistence"
              ? "warn"
              : "",
        },
      ],
      honesty:
        descriptiveSurfaceText(
          descriptiveFit.honesty_note,
          descriptiveFit.semantic_audit?.headline,
        ) ||
        `${DESCRIPTIVE_APPROXIMATION_LABEL} is the strongest benchmark-local compact equation currently available.`,
      note:
        descriptiveFit.equation?.delta_form_label ||
        descriptiveSurfaceText(descriptiveFit.semantic_audit?.headline) ||
        "",
      tone: "accent",
      summaryRows: [
        ["Source", descriptiveFit.submitter_id || descriptiveFit.source || "n/a"],
        ["Candidate", descriptiveFit.candidate_id || "n/a"],
        ["Selection scope", descriptiveFit.selection_scope || "descriptive scope"],
      ],
    };
  }

  return null;
}

function selectedChangeMetric(analysis) {
  const metrics = availableChangeMetrics(analysis);
  return (
    metrics.find((metric) => metric.id === state.selectedChangeMetric) ||
    metrics.find((metric) => metric.id === "return") ||
    metrics[0] ||
    null
  );
}

function availableDeterministicOverlays(analysis) {
  if (!analysis) return [];
  const overlays = [];
  const holisticEquation = analysis.holistic_equation;
  const predictiveLaw = analysis.predictive_law;
  const descriptiveReconstruction = analysis.descriptive_reconstruction;
  const descriptiveFit = analysis.descriptive_fit;
  const point = analysis.operator_point;
  const probabilistic = analysis.probabilistic || {};

  if (
    hasHolisticClaim(analysis) &&
    Array.isArray(holisticEquation.equation?.curve) &&
    holisticEquation.equation.curve.length
  ) {
    overlays.push({
      id: "holistic",
      label: HOLISTIC_EQUATION_LABEL,
      shortLabel: "Holistic",
      tone: "accent",
      equation: holisticEquation.equation || {},
      series: holisticEquation.equation.curve,
      honesty:
        holisticEquation.honesty_note ||
        "Holistic equation combines the backend-backed deterministic and probabilistic claim surfaces.",
    });
  }

  if (
    hasPredictiveLaw(analysis) &&
    Array.isArray(predictiveLaw.equation?.curve) &&
    predictiveLaw.equation.curve.length
  ) {
    overlays.push({
      id: "predictive_law",
      label: PREDICTIVE_SYMBOLIC_LAW_LABEL,
      shortLabel: "Predictive",
      tone: "sea",
      equation: predictiveLaw.equation || {},
      series: predictiveLaw.equation.curve,
      honesty:
        predictiveLaw.honesty_note ||
        "Predictive symbolic law reflects the backend-backed point-lane publication path.",
    });
  }

  if (hasDescriptiveReconstruction(analysis)) {
    overlays.push({
      id: "descriptive_reconstruction",
      label: DESCRIPTIVE_RECONSTRUCTION_LABEL,
      shortLabel: "Reconstruction",
      tone: "accent",
      equation: descriptiveReconstruction.equation || {},
      series:
        descriptiveReconstruction.chart?.equation_curve ||
        descriptiveReconstruction.equation?.curve ||
        [],
      honesty:
        descriptiveReconstruction.honesty_note ||
        `${DESCRIPTIVE_RECONSTRUCTION_LABEL} is descriptive-only and stays separate from the operator publication path.`,
    });
  }

  if (descriptiveFit?.status === "completed") {
    overlays.push({
      id: "descriptive_fit",
      label: BENCHMARK_DESCRIPTIVE_FIT_LABEL,
      shortLabel: "Benchmark-local",
      tone: "accent",
      equation: descriptiveFit.equation || {},
      series:
        descriptiveFit.chart?.equation_curve ||
        descriptiveFit.equation?.curve ||
        [],
      honesty:
        descriptiveSurfaceText(
          descriptiveFit.honesty_note,
          descriptiveFit.semantic_audit?.headline,
        ) ||
        `${BENCHMARK_DESCRIPTIVE_FIT_LABEL} stays separate from the operator lane.`,
    });
  }

  if (point?.status === "completed") {
    overlays.push({
      id: "point",
      label: "Operator point path",
      shortLabel: "Point",
      tone: "sea",
      equation: point.equation || {},
      series: point.equation?.curve || [],
      honesty:
        pointPublicationHeadlineForDisplay(analysis) ||
        "Operator point path remains distinct from benchmark-local descriptive fits.",
    });
  }

  const probabilisticOverlay = pickProbabilisticMeanOverlay(probabilistic);
  if (probabilisticOverlay) {
    overlays.push(probabilisticOverlay);
  }

  return overlays.filter((overlay) => Array.isArray(overlay.series) && overlay.series.length);
}

function pickProbabilisticMeanOverlay(probabilistic) {
  const lanes = probabilistic && typeof probabilistic === "object"
    ? Object.entries(probabilistic)
    : [];
  for (const [mode, payload] of lanes) {
    if (payload?.status !== "completed") continue;
    const equationCurve = payload.equation?.curve || [];
    if (!equationCurve.length) continue;
    if (payload.selected_family !== "analytic") continue;
    return {
      id: "probabilistic_mean",
      label: "Probabilistic mean object",
      shortLabel: "Mean",
      tone: "moss",
      equation: payload.equation || {},
      series: equationCurve,
      honesty:
        payload.evidence?.headline ||
        `${humanizeKey(mode)} lane equation is shown as a probabilistic mean object, not a point publication.`,
    };
  }
  return null;
}

function defaultDeterministicOverlay(analysis) {
  const overlays = availableDeterministicOverlays(analysis);
  return overlays[0]?.id || "descriptive_fit";
}

function currentDeterministicOverlay(analysis) {
  const overlays = availableDeterministicOverlays(analysis);
  if (!overlays.length) return null;
  return (
    overlays.find((overlay) => overlay.id === state.selectedDeterministicOverlay) ||
    overlays[0]
  );
}

function completedProbabilisticLaneEntries(analysis) {
  const probabilistic = analysis?.probabilistic;
  if (!probabilistic || typeof probabilistic !== "object") return [];
  return Object.entries(probabilistic).filter(([, payload]) => payload?.status === "completed");
}

function defaultSelectedLane(analysis) {
  const lanes = completedProbabilisticLaneEntries(analysis);
  return lanes[0]?.[0] || "distribution";
}

function selectedProbabilisticLane(analysis) {
  const lanes = completedProbabilisticLaneEntries(analysis);
  return (
    lanes.find(([mode]) => mode === state.selectedLane) ||
    lanes[0] ||
    null
  );
}

function rowForSelectedHorizon(payload) {
  const rows = Array.isArray(payload?.rows) ? payload.rows : [];
  const exact = rows.find((row) => Number(row?.horizon) === Number(state.selectedHorizon));
  return exact || payload?.latest_row || rows.at(-1) || null;
}

function buildResidualSeries(actualSeries, overlaySeries) {
  const actual = Array.isArray(actualSeries) ? actualSeries : [];
  const overlay = Array.isArray(overlaySeries) ? overlaySeries : [];
  if (!actual.length || !overlay.length) return [];
  const overlayByTime = new Map(
    overlay
      .filter((point) => point && point.event_time)
      .map((point) => [String(point.event_time), Number(point.fitted_value)]),
  );
  return actual
    .map((point, index) => {
      const fitted = overlayByTime.has(point.event_time)
        ? overlayByTime.get(point.event_time)
        : Number(overlay[index]?.fitted_value);
      const observed = Number(point.observed_value);
      if (!Number.isFinite(fitted) || !Number.isFinite(observed)) return null;
      return {
        event_time: point.event_time,
        observed_value: observed - fitted,
      };
    })
    .filter(Boolean);
}

function parseEventThreshold(eventDefinition) {
  const explicitThreshold = Number(eventDefinition?.threshold);
  if (Number.isFinite(explicitThreshold)) {
    return explicitThreshold;
  }
  const match = String(eventDefinition || "").match(/(-?\d+(?:\.\d+)?)/);
  return match ? Number(match[1]) : null;
}

function isEventDefinition(value) {
  return Boolean(
    value &&
      typeof value === "object" &&
      !Array.isArray(value) &&
      ("event_id" in value ||
        "operator" in value ||
        "threshold_source" in value ||
        "threshold" in value),
  );
}

function eventOperatorSymbol(operator) {
  const symbols = {
    greater_than: ">",
    greater_than_or_equal: "≥",
    less_than: "<",
    less_than_or_equal: "≤",
    equal: "=",
    equals: "=",
    not_equal: "≠",
  };
  return symbols[String(operator || "")] || humanizePhrase(operator || "matches");
}

function inferEventSubject(eventId) {
  const normalized = String(eventId || "").toLowerCase();
  if (normalized.startsWith("target_")) return "Target";
  if (normalized.startsWith("price_")) return "Price";
  return humanizePhrase(eventId || "Target");
}

function formatEventDefinition(eventDefinition) {
  if (typeof eventDefinition === "string") {
    return eventDefinition;
  }
  if (!isEventDefinition(eventDefinition)) {
    return null;
  }
  const subject = inferEventSubject(eventDefinition.event_id);
  const operator = eventOperatorSymbol(eventDefinition.operator);
  const thresholdSource = eventDefinition.threshold_source
    ? humanizePhrase(eventDefinition.threshold_source)
    : null;
  const threshold = Number(eventDefinition.threshold);
  if (thresholdSource && Number.isFinite(threshold)) {
    return `${subject} ${operator} ${thresholdSource} (${formatNumber(threshold)})`;
  }
  if (Number.isFinite(threshold)) {
    return `${subject} ${operator} ${formatNumber(threshold)}`;
  }
  if (thresholdSource) {
    return `${subject} ${operator} ${thresholdSource}`;
  }
  if (eventDefinition.event_id) {
    return humanizePhrase(eventDefinition.event_id);
  }
  return JSON.stringify(eventDefinition);
}

function quantileValue(row, level) {
  const quantiles = Array.isArray(row?.quantiles) ? row.quantiles : [];
  const entry = quantiles.find((item) => Number(item?.level) === Number(level));
  return entry ? Number(entry.value) : null;
}

function finiteNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function firstFiniteNumber(...values) {
  for (const value of values) {
    const number = finiteNumber(value);
    if (number !== null) return number;
  }
  return null;
}

function mappedDistributionBand(row, key) {
  const interval = row?.[key];
  if (!interval || typeof interval !== "object") return null;
  const lower = firstFiniteNumber(interval.lower, interval.lower_bound);
  const upper = firstFiniteNumber(interval.upper, interval.upper_bound);
  if (lower === null || upper === null) return null;
  return { lower, upper, source: key };
}

function distributionBand(row) {
  if (!row || typeof row !== "object") return null;
  let interval =
    mappedDistributionBand(row, "configured_interval") ||
    mappedDistributionBand(row, "family_interval") ||
    mappedDistributionBand(row, "prediction_interval") ||
    mappedDistributionBand(row, "interval");
  if (!interval) {
    const lower = firstFiniteNumber(row.lower_bound, row.interval_lower, row.lower);
    const upper = firstFiniteNumber(row.upper_bound, row.interval_upper, row.upper);
    if (lower !== null && upper !== null) {
      interval = { lower, upper, source: "row_interval" };
    }
  }
  if (!interval) {
    const lower = quantileValue(row, 0.1);
    const upper = quantileValue(row, 0.9);
    if (Number.isFinite(lower) && Number.isFinite(upper)) {
      interval = { lower, upper, source: "quantiles" };
    }
  }
  if (!interval) {
    const location = finiteNumber(row.location);
    const scale = finiteNumber(row.scale);
    if (location === null || scale === null) return null;
    interval = {
      lower: location - scale,
      upper: location + scale,
      source: "location_scale",
    };
  }
  return {
    ...interval,
    center:
      firstFiniteNumber(row.location, quantileValue(row, 0.5)) ??
      interval.lower + (interval.upper - interval.lower) / 2,
  };
}

function laneFamily(payload, row = null) {
  return (
    payload?.distribution_family ||
    payload?.family ||
    payload?.family_id ||
    payload?.evidence?.family ||
    row?.distribution_family ||
    row?.family ||
    row?.family_id ||
    payload?.selected_family ||
    "n/a"
  );
}

function refLabel(value) {
  if (!value) return null;
  if (typeof value === "string") return value;
  if (typeof value === "object" && value.schema_name && value.object_id) {
    return `${value.schema_name}:${value.object_id}`;
  }
  return String(value);
}

function refList(payload, key) {
  const evidenceValues = Array.isArray(payload?.evidence?.[key])
    ? payload.evidence[key]
    : [];
  const laneValues = Array.isArray(payload?.[key]) ? payload[key] : [];
  return [...evidenceValues, ...laneValues]
    .map(refLabel)
    .filter(Boolean)
    .filter((value, index, values) => values.indexOf(value) === index);
}

function calibrationBinCount(payload) {
  const evidenceCount = Number(payload?.evidence?.calibration_bin_count);
  if (Number.isFinite(evidenceCount) && evidenceCount > 0) return evidenceCount;
  const diagnostics = Array.isArray(payload?.calibration?.diagnostics)
    ? payload.calibration.diagnostics
    : [];
  return diagnostics.reduce((total, diagnostic) => {
    const bins =
      diagnostic?.calibration_bins ||
      diagnostic?.bins ||
      diagnostic?.bin_records ||
      [];
    return total + (Array.isArray(bins) ? bins.length : 0);
  }, 0);
}

function shortList(values) {
  const list = Array.isArray(values) ? values : [];
  if (!list.length) return "none";
  return list.map((value) => humanizeKey(value)).join(", ");
}

function normalizeLatestRow(mode, row) {
  if (!row || typeof row !== "object") return [];
  if (mode === "distribution") {
    const band = distributionBand(row);
    return [
      ["Origin", String(row.origin_time || "n/a").slice(0, 10)],
      ["Horizon", `h${row.horizon ?? "n/a"}`],
      ["Family", laneFamily({}, row)],
      ["Location", formatCell(row.location)],
      band ? ["Band", `${formatCell(band.lower)} to ${formatCell(band.upper)}`] : ["Scale", formatCell(row.scale)],
      ["Realized", formatCell(row.realized_observation)],
    ];
  }
  if (mode === "quantile") {
    return [
      ["Origin", String(row.origin_time || "n/a").slice(0, 10)],
      ["Horizon", `h${row.horizon ?? "n/a"}`],
      ["Q0.1", formatCell(quantileValue(row, 0.1))],
      ["Q0.5", formatCell(quantileValue(row, 0.5))],
      ["Q0.9", formatCell(quantileValue(row, 0.9))],
    ];
  }
  if (mode === "interval") {
    return [
      ["Origin", String(row.origin_time || "n/a").slice(0, 10)],
      ["Horizon", `h${row.horizon ?? "n/a"}`],
      ["Lower", formatCell(row.lower_bound)],
      ["Upper", formatCell(row.upper_bound)],
      ["Realized", formatCell(row.realized_observation)],
    ];
  }
  if (mode === "event_probability") {
    return [
      ["Origin", String(row.origin_time || "n/a").slice(0, 10)],
      ["Horizon", `h${row.horizon ?? "n/a"}`],
      ["Probability", formatCell(row.event_probability)],
      ["Event", formatEventDefinition(row.event_definition) || "n/a"],
      ["Realized", formatCell(row.realized_event)],
    ];
  }
  return Object.entries(row).slice(0, 6).map(([key, value]) => [humanizeKey(key), formatCell(value)]);
}

function residualSummary(series) {
  if (!series.length) {
    return {
      mae: "n/a",
      max: "n/a",
      min: "n/a",
    };
  }
  const values = series.map((point) => Number(point.observed_value)).filter(Number.isFinite);
  const absolute = values.map((value) => Math.abs(value));
  return {
    mae: formatNumber(absolute.reduce((sum, value) => sum + value, 0) / absolute.length),
    max: formatNumber(Math.max(...values)),
    min: formatNumber(Math.min(...values)),
  };
}

function renderSegmentedControls({ label, items, activeValue, dataAttribute }) {
  if (!items.length) return "";
  return `
    <div class="control-group">
      <div class="detail-label">${escapeHtml(label)}</div>
      <div class="segmented-controls">
        ${items
          .map((item) => `
            <button
              type="button"
              class="segmented-button${item.value === activeValue ? " active" : ""}"
              ${dataAttribute}="${escapeHtml(String(item.value))}"
            >
              ${escapeHtml(item.label)}
            </button>
          `)
          .join("")}
      </div>
    </div>
  `;
}

function renderSharedControls({ analysis, showOverlay = true, showHorizon = true }) {
  const overlays = availableDeterministicOverlays(analysis);
  const horizons = availableHorizons(analysis);
  return `
    <div class="shared-controls">
      ${showOverlay
        ? renderSegmentedControls({
            label: "Deterministic overlay",
            items: overlays.map((overlay) => ({
              value: overlay.id,
              label: overlay.shortLabel,
            })),
            activeValue: currentDeterministicOverlay(analysis)?.id,
            dataAttribute: "data-overlay",
          })
        : ""}
      ${showHorizon
        ? renderSegmentedControls({
            label: "Probabilistic horizon",
            items: horizons.map((horizon) => ({
              value: horizon,
              label: `h${horizon}`,
            })),
            activeValue: state.selectedHorizon,
            dataAttribute: "data-horizon",
          })
        : ""}
    </div>
  `;
}

function renderEquationRibbon(analysis) {
  const entries = [];
  if (hasHolisticClaim(analysis)) {
    entries.push({
      kicker: "Holistic",
      label: HOLISTIC_EQUATION_LABEL,
      note: holisticClaimDescriptor(analysis.holistic_equation),
      tone: "accent",
      jump: "atlas-equation-stack",
    });
  }
  if (hasPredictiveLaw(analysis)) {
    entries.push({
      kicker: PREDICTIVE_SYMBOLIC_LAW_LABEL,
      label:
        analysis.predictive_law.equation?.label ||
        PREDICTIVE_SYMBOLIC_LAW_LABEL,
      note:
        analysis.predictive_law.equation?.delta_form_label ||
        analysis.predictive_law.honesty_note ||
        "",
      tone: "sea",
      jump: "atlas-equation-stack",
    });
  }
  if (hasDescriptiveReconstruction(analysis)) {
    entries.push({
      kicker: DESCRIPTIVE_RECONSTRUCTION_LABEL,
      label: analysis.descriptive_reconstruction.equation?.label || "n/a",
      note:
        analysis.descriptive_reconstruction.equation?.coefficient_vector_label ||
        analysis.descriptive_reconstruction.honesty_note ||
        "",
      tone: "accent",
      jump: "atlas-equation-stack",
    });
  }
  if (analysis.descriptive_fit?.status === "completed") {
    entries.push({
      kicker: BENCHMARK_DESCRIPTIVE_FIT_LABEL,
      label: analysis.descriptive_fit.equation?.label || "n/a",
      note:
        analysis.descriptive_fit.equation?.delta_form_label ||
        descriptiveSurfaceText(analysis.descriptive_fit.semantic_audit?.headline) ||
        "",
      tone: "accent",
      jump: "atlas-equation-stack",
    });
  }
  if (analysis.operator_point?.status === "completed") {
    entries.push({
      kicker: "Point lane",
      label: analysis.operator_point.equation?.label || "n/a",
      note: analysis.operator_point.equation?.delta_form_label || analysis.operator_point.publication?.status || "",
      tone: "sea",
      jump: "atlas-observed-path",
    });
  }
  const selectedLaneEntry = selectedProbabilisticLane(analysis);
  if (selectedLaneEntry) {
    const [mode, payload] = selectedLaneEntry;
    entries.push({
      kicker: humanizeKey(mode),
      label: payload.equation?.label || "n/a",
      note: payload.evidence?.headline || payload.equation?.delta_form_label || "",
      tone: "moss",
      jump: "atlas-uncertainty-ruler",
    });
  }
  return `
    <div class="equation-ribbon">
      ${entries
        .map((entry) => `
          <button type="button" class="equation-chip tone-${escapeHtml(entry.tone)}" data-atlas-jump="${escapeHtml(entry.jump)}">
            <span class="mini-kicker">${escapeHtml(entry.kicker)}</span>
            <strong>${escapeHtml(entry.label)}</strong>
            <span>${escapeHtml(entry.note)}</span>
          </button>
        `)
        .join("")}
    </div>
  `;
}

function renderEquationMarkup(value, { className = "", displayMode = true } = {}) {
  const normalized = String(value || "").trim();
  if (!normalized) return "";
  const candidateSource = normalized
    .replaceAll("*", " \\cdot ")
    .replaceAll(">=", " \\ge ")
    .replaceAll("<=", " \\le ");
  try {
    return `
      <div class="${escapeHtml(className)} is-katex" data-equation-renderer="katex">
        ${katex.renderToString(candidateSource, {
          displayMode,
          throwOnError: false,
          strict: "ignore",
        })}
      </div>
    `;
  } catch (error) {
    return `
      <div class="${escapeHtml(className)} is-plain" data-equation-renderer="plain">
        ${escapeHtml(normalized)}
      </div>
    `;
  }
}

function renderEquationCard({
  title,
  kicker,
  equation,
  parameters = {},
  badges = [],
  honesty = "",
  note = "",
  tone = "accent",
  actionLabel = "",
  actionValue = "",
  actionAttribute = "data-lane-select",
  selected = false,
  summaryRows = [],
  sectionClassName = "",
  dataEquationHero = "",
}) {
  const equationLabel = equation?.label || equation?.structure_signature || "No explicit renderer available.";
  const deltaForm = equation?.delta_form_label || note || "";
  return `
    <section class="panel equation-card tone-${escapeHtml(tone)}${selected ? " selected" : ""}${sectionClassName ? ` ${escapeHtml(sectionClassName)}` : ""}"${dataEquationHero ? ` data-equation-hero="${escapeHtml(dataEquationHero)}"` : ""}>
      <div class="panel-head">
        <div>
          <p class="mini-kicker">${escapeHtml(kicker)}</p>
          <h3>${escapeHtml(title)}</h3>
        </div>
        <div class="pill-row">
          ${badges.map((badge) => pill(badge.label, badge.tone || "")).join("")}
        </div>
      </div>
      <div class="equation-copy">
        ${renderEquationMarkup(equationLabel, { className: "equation-formula", displayMode: true })}
        ${deltaForm ? renderEquationMarkup(deltaForm, { className: "equation-delta", displayMode: false }) : ""}
      </div>
      ${honesty ? banner("Honesty", honesty) : ""}
      ${summaryRows.length
        ? `<div class="detail-grid">${summaryRows.map(([label, value]) => detail(label, value)).join("")}</div>`
        : ""}
      ${Object.keys(parameters || {}).length ? renderParameterSummary(parameters) : ""}
      ${actionLabel && actionValue
        ? `<button type="button" class="subtle-action" ${actionAttribute}="${escapeHtml(String(actionValue))}">${escapeHtml(actionLabel)}</button>`
        : ""}
    </section>
  `;
}

function renderOverviewEquationHero(analysis) {
  const card = strongestEquationCard(analysis);
  if (!card) {
    return "";
  }
  return renderEquationCard({
    ...card,
    actionLabel: "Open atlas",
    actionValue: "atlas",
    actionAttribute: "data-open-tab",
    sectionClassName: "overview-equation-hero",
    dataEquationHero: "overview",
  });
}

function humanizeGapItem(item) {
  const text = String(item || "").trim();
  if (!text) return "";
  if (/^[a-z0-9_-]+$/i.test(text)) {
    const phrase = text
      .replaceAll("_", " ")
      .replaceAll("-", " ")
      .replace(/\s+/g, " ")
      .trim()
      .toLowerCase();
    return phrase ? `${phrase[0].toUpperCase()}${phrase.slice(1)}` : "";
  }
  return text;
}

function renderWhyNotStrongerPanel(analysis) {
  const gapReport = Array.isArray(analysis?.gap_report)
    ? analysis.gap_report.map(humanizeGapItem).filter(Boolean)
    : [];
  const abstentionReasons = Array.isArray(analysis?.would_have_abstained_because)
    ? analysis.would_have_abstained_because.map(humanizeGapItem).filter(Boolean)
    : [];
  const holisticReasons = Array.isArray(analysis?.not_holistic_because)
    ? analysis.not_holistic_because.map(humanizeGapItem).filter(Boolean)
    : [];

  if (!gapReport.length && !abstentionReasons.length && !holisticReasons.length) {
    return "";
  }

  return `
    <section class="panel">
      <div class="panel-head">
        <div>
          <p class="mini-kicker">Claim ceiling</p>
          <h3>Why not stronger?</h3>
        </div>
      </div>
      <p>The workbench shows the strongest available claim class supported by the normalized analysis payload. Stronger wording stays withheld until these gaps are cleared.</p>
      ${renderExplanationList("Gap report", gapReport, "warn")}
      ${renderExplanationList("Operator publication gaps", abstentionReasons, "warn")}
      ${renderExplanationList("Holistic claim gaps", holisticReasons, "warn")}
    </section>
  `;
}

function renderAnalyticalCanvasPanel({ analysis, activeOverlay, selectedLaneEntry }) {
  const laneLabel = selectedLaneEntry ? humanizeKey(selectedLaneEntry[0]) : "No probabilistic lane";
  const laneHeadline =
    selectedLaneEntry?.[1]?.evidence?.headline ||
    "The selected probabilistic object stays linked to the dominant chart instead of being pushed onto a separate dashboard card.";
  const overlayLabel = activeOverlay?.equation?.label || activeOverlay?.label || "Overlay";
  const overlayColor =
    activeOverlay?.tone === "sea" ? "#2c617b" : activeOverlay?.tone === "moss" ? "#567060" : "#a45738";

  return `
    <section class="panel analytical-canvas">
      <div class="panel-head">
        <div>
          <p class="mini-kicker">Math-first surface</p>
          <h3>Analytical canvas</h3>
        </div>
        <div class="pill-row">
          ${pill(activeOverlay?.shortLabel || "overlay", "ok")}
          ${pill(laneLabel, selectedLaneEntry ? "sea" : "")}
        </div>
      </div>
      <p>Observed path, active equation, and the current uncertainty context share one dominant surface so the model can be inspected holistically instead of as scattered cards.</p>
      <div class="chart-shell chart-shell-immersive">
        <div class="chart-root chart-root-primary" data-chart-root="primary-canvas" aria-label="Primary analytical canvas"></div>
        <div class="legend">
          ${legend("Observed", "#17212b")}
          ${legend(overlayLabel, overlayColor)}
        </div>
      </div>
      ${laneHeadline ? banner("Selected uncertainty context", laneHeadline) : ""}
      <div class="detail-grid analytical-caption-grid">
        ${detail("Target transform", analysis.dataset.target.label)}
        ${detail("Deterministic overlay", activeOverlay?.label || "n/a")}
        ${detail("Probabilistic lane", laneLabel)}
        ${detail("Selected horizon", `h${state.selectedHorizon}`)}
      </div>
    </section>
  `;
}

function buildResidualChartSvg(rows) {
  if (!rows.length) return `<div class="empty-state">No residual series available for the active overlay.</div>`;
  const width = 820;
  const height = 220;
  const padding = { top: 20, right: 28, bottom: 36, left: 56 };
  const values = rows.map((point) => Number(point.observed_value)).filter(Number.isFinite);
  const min = Math.min(...values, 0);
  const max = Math.max(...values, 0);
  const yMin = min - (max - min || 1) * 0.12;
  const yMax = max + (max - min || 1) * 0.12;
  const xScale = (index) =>
    padding.left + (rows.length <= 1 ? 0 : (index / (rows.length - 1)) * (width - padding.left - padding.right));
  const yScale = (value) =>
    height - padding.bottom - ((value - yMin) / (yMax - yMin || 1)) * (height - padding.top - padding.bottom);
  const path = polylinePath(
    rows.map((point, index) => ({ ...point, x: index, y: Number(point.observed_value) })),
    (index) => xScale(index),
    (value) => yScale(value),
    rows.length,
  );
  const zeroY = yScale(0);
  return `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Residual chart">
      <line x1="${padding.left}" y1="${zeroY}" x2="${width - padding.right}" y2="${zeroY}" stroke="rgba(23,33,43,0.2)" stroke-dasharray="4 4" />
      <line x1="${padding.left}" y1="${height - padding.bottom}" x2="${width - padding.right}" y2="${height - padding.bottom}" stroke="rgba(23,33,43,0.1)" />
      <path d="${path}" fill="none" stroke="#a45738" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"></path>
      ${rows
        .filter((_, index) => index % Math.max(1, Math.ceil(rows.length / 10)) === 0 || index === rows.length - 1)
        .map(
          (point, index) => `
            <circle cx="${xScale(index)}" cy="${yScale(Number(point.observed_value))}" r="3" fill="#a45738">
              <title>${escapeHtml(`${String(point.event_time || "").slice(0, 10)} • residual ${formatNumber(point.observed_value)}`)}</title>
            </circle>
          `,
        )
        .join("")}
      <text x="${padding.left}" y="${padding.top - 4}" fill="rgba(23,33,43,0.58)" font-size="12">Residual</text>
      <text x="${padding.left}" y="${height - 12}" fill="rgba(23,33,43,0.58)" font-size="12">${escapeHtml(String(rows[0]?.event_time || "").slice(0, 10))}</text>
      <text x="${width - padding.right - 72}" y="${height - 12}" fill="rgba(23,33,43,0.58)" font-size="12">${escapeHtml(String(rows.at(-1)?.event_time || "").slice(0, 10))}</text>
    </svg>
  `;
}

function buildResidualDistributionSvg(rows) {
  if (!rows.length) {
    return `<div class="empty-state">No residual distribution is available for the active overlay.</div>`;
  }
  const values = rows
    .map((point) => Number(point.observed_value))
    .filter(Number.isFinite);
  if (!values.length) {
    return `<div class="empty-state">No residual distribution is available for the active overlay.</div>`;
  }
  const binCount = Math.min(8, Math.max(4, Math.ceil(Math.sqrt(values.length))));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const bins = Array.from({ length: binCount }, (_, index) => ({
    start: min + (span / binCount) * index,
    end: min + (span / binCount) * (index + 1),
    count: 0,
  }));
  for (const value of values) {
    const index =
      span === 0
        ? 0
        : Math.min(binCount - 1, Math.floor(((value - min) / span) * binCount));
    bins[Math.max(0, index)].count += 1;
  }
  const width = 420;
  const height = 220;
  const padding = { top: 20, right: 16, bottom: 36, left: 42 };
  const maxCount = Math.max(...bins.map((bin) => bin.count), 1);
  const xScale = (index) =>
    padding.left + (index / binCount) * (width - padding.left - padding.right);
  const yScale = (count) =>
    height - padding.bottom - (count / maxCount) * (height - padding.top - padding.bottom);
  const barWidth = (width - padding.left - padding.right) / binCount - 10;
  return `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Residual distribution">
      <line x1="${padding.left}" y1="${height - padding.bottom}" x2="${width - padding.right}" y2="${height - padding.bottom}" stroke="rgba(23,33,43,0.18)" />
      ${bins
        .map((bin, index) => {
          const x = xScale(index) + 5;
          const y = yScale(bin.count);
          const barHeight = height - padding.bottom - y;
          return `
            <rect x="${x}" y="${y}" width="${barWidth}" height="${barHeight}" rx="10" ry="10" fill="#567060">
              <title>${escapeHtml(`${formatNumber(bin.start)} to ${formatNumber(bin.end)} • ${bin.count} residuals`)}</title>
            </rect>
          `;
        })
        .join("")}
      <text x="${padding.left}" y="${height - 12}" font-size="12" fill="rgba(23,33,43,0.58)">${escapeHtml(formatNumber(min))}</text>
      <text x="${width - padding.right - 48}" y="${height - 12}" font-size="12" fill="rgba(23,33,43,0.58)">${escapeHtml(formatNumber(max))}</text>
    </svg>
  `;
}

function buildUncertaintyRulerSvg({ distributionRow, quantileRow, intervalRow, eventRow }) {
  const distributionInterval = distributionBand(distributionRow);
  const quantileLow = quantileValue(quantileRow, 0.1);
  const quantileMid = quantileValue(quantileRow, 0.5);
  const quantileHigh = quantileValue(quantileRow, 0.9);
  const eventThreshold = parseEventThreshold(eventRow?.event_definition);
  const eventLabel = formatEventDefinition(eventRow?.event_definition);
  const realized =
    Number(distributionRow?.realized_observation ?? quantileRow?.realized_observation ?? intervalRow?.realized_observation);
  const values = [
    distributionInterval?.center,
    distributionInterval?.lower,
    distributionInterval?.upper,
    quantileLow,
    quantileMid,
    quantileHigh,
    Number(intervalRow?.lower_bound),
    Number(intervalRow?.upper_bound),
    realized,
    eventThreshold,
  ].filter(Number.isFinite);
  if (!values.length) {
    return `<div class="empty-state">No aligned probabilistic rows are available for the selected horizon.</div>`;
  }
  const width = 860;
  const height = 220;
  const padding = { top: 24, right: 32, bottom: 34, left: 132 };
  const min = Math.min(...values);
  const max = Math.max(...values);
  const xScale = (value) =>
    padding.left + ((value - min) / (max - min || 1)) * (width - padding.left - padding.right);
  const rows = [
    {
      label: "Distribution",
      y: 54,
      lower: distributionInterval?.lower,
      upper: distributionInterval?.upper,
      center: distributionInterval?.center,
      color: "#2c617b",
    },
    {
      label: "Quantile",
      y: 96,
      lower: quantileLow,
      upper: quantileHigh,
      center: quantileMid,
      color: "#a45738",
    },
    {
      label: "Interval",
      y: 138,
      lower: Number(intervalRow?.lower_bound),
      upper: Number(intervalRow?.upper_bound),
      center: Number(intervalRow?.lower_bound) + ((Number(intervalRow?.upper_bound) - Number(intervalRow?.lower_bound)) / 2),
      color: "#567060",
    },
  ].filter((row) => Number.isFinite(row.lower) && Number.isFinite(row.upper));

  return `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Uncertainty ruler">
      <line x1="${padding.left}" y1="${height - padding.bottom}" x2="${width - padding.right}" y2="${height - padding.bottom}" stroke="rgba(23,33,43,0.18)" />
      ${rows
        .map((row) => `
          <text x="24" y="${row.y + 4}" font-size="12" fill="rgba(23,33,43,0.72)">${escapeHtml(row.label)}</text>
          <line x1="${xScale(row.lower)}" y1="${row.y}" x2="${xScale(row.upper)}" y2="${row.y}" stroke="${row.color}" stroke-width="8" stroke-linecap="round"></line>
          <circle cx="${xScale(row.center)}" cy="${row.y}" r="6" fill="${row.color}"></circle>
        `)
        .join("")}
      ${Number.isFinite(realized)
        ? `
          <line x1="${xScale(realized)}" y1="${padding.top}" x2="${xScale(realized)}" y2="${height - padding.bottom}" stroke="#17212b" stroke-dasharray="5 4"></line>
          <text x="${xScale(realized) + 8}" y="${padding.top + 12}" font-size="12" fill="#17212b">Realized ${escapeHtml(formatNumber(realized))}</text>
        `
        : ""}
      ${Number.isFinite(eventThreshold)
        ? `
          <line x1="${xScale(eventThreshold)}" y1="${padding.top + 18}" x2="${xScale(eventThreshold)}" y2="${height - padding.bottom - 10}" stroke="#8f433b" stroke-dasharray="2 6"></line>
          <text x="${xScale(eventThreshold) + 8}" y="${height - padding.bottom - 14}" font-size="12" fill="#8f433b">
            Threshold ${escapeHtml(formatNumber(eventThreshold))} • ${escapeHtml(formatPercent(Number(eventRow?.event_probability) || 0))}
          </text>
        `
        : eventLabel
        ? `<text x="24" y="${height - padding.bottom - 14}" font-size="12" fill="#8f433b">${escapeHtml(eventLabel)} • ${escapeHtml(formatPercent(Number(eventRow?.event_probability) || 0))}</text>`
        : ""}
      <text x="${padding.left}" y="${height - 10}" font-size="12" fill="rgba(23,33,43,0.58)">${escapeHtml(formatNumber(min))}</text>
      <text x="${width - padding.right - 48}" y="${height - 10}" font-size="12" fill="rgba(23,33,43,0.58)">${escapeHtml(formatNumber(max))}</text>
    </svg>
  `;
}

function renderOverview() {
  const panel = tabPanels.overview;
  if (!state.analysis) {
    panel.innerHTML = `<div class="panel"><p class="empty-state">Overview appears after the first analysis run.</p></div>`;
    return;
  }
  const { dataset, operator_point: point, benchmark, descriptive_fit: descriptiveFit } = state.analysis;
  const descriptiveFitStatus = benchmark?.descriptive_fit_status;
  const activeOverlay = currentDeterministicOverlay(state.analysis);
  const selectedLaneEntry = selectedProbabilisticLane(state.analysis);
  const selectedRow = selectedLaneEntry ? rowForSelectedHorizon(selectedLaneEntry[1]) : null;
  const banners = [];
  if (dataset.target?.analysis_note) {
    banners.push(banner("Target semantics", dataset.target.analysis_note));
  }
  if (descriptiveFit?.status === "completed") {
    banners.push(
      banner(
        BENCHMARK_DESCRIPTIVE_FIT_LABEL,
        descriptiveSurfaceText(
          descriptiveFit.semantic_audit?.headline,
          descriptiveFit.honesty_note,
        ) ||
          `${BENCHMARK_DESCRIPTIVE_FIT_LABEL} stays separate from the operator publication path.`,
      ),
    );
  } else if (descriptiveFitStatus?.status === "absent_no_admissible_candidate") {
    banners.push(
      banner(
        NO_BENCHMARK_LOCAL_WINNER_LABEL,
        descriptiveSurfaceText(descriptiveFitStatus.headline) ||
          "No benchmark-local descriptive fit was retained for this run.",
      ),
    );
  }
  if (point?.status === "completed") {
    banners.push(
      banner(
        "Operator lane",
        pointPublicationHeadlineForDisplay(state.analysis) ||
          formatReasonCodes(
            point.abstention?.reason_codes || point.abstention?.failure_reason_codes || [],
          ),
      ),
    );
  }
  if (benchmark?.status === "completed" && benchmark.track_summary?.abstention_mode) {
    banners.push(
      banner(
        "Benchmark status",
        `Benchmark tracks local winners under ${benchmark.track_summary.abstention_mode}; this is not an operator publication.`,
      ),
    );
  }
  const probabilisticEvidence = summarizeProbabilisticEvidence(state.analysis.probabilistic);
  if (probabilisticEvidence) {
    banners.push(banner("Probabilistic evidence", probabilisticEvidence));
  }
  panel.innerHTML = `
    ${renderOverviewEquationHero(state.analysis)}
    <div class="stack">
      <section class="panel overview-panel">
        <div class="panel-head">
          <div>
            <p class="mini-kicker">Run framing</p>
            <h3>Claim, evidence, and next investigations</h3>
          </div>
          <div class="pill-row">
            ${pill(activeOverlay?.shortLabel || "no overlay", "ok")}
            ${pill(`h${state.selectedHorizon}`, "warn")}
          </div>
        </div>
        <p>The overview starts from the strongest available mathematical object, then shows the operator publication path and follow-up views needed to judge whether the result is descriptive, publishable, or still evidence-limited.</p>
        ${renderStatusMatrix(state.analysis)}
        ${renderEquationRibbon(state.analysis)}
        ${renderWhyNotStrongerPanel(state.analysis)}
        <div class="atlas-jump-grid">
          <button type="button" class="subtle-action" data-atlas-jump="atlas-run-framing">Open workspace framing</button>
          <button type="button" class="subtle-action" data-atlas-jump="atlas-workspace-stage">Open equation stage</button>
          <button type="button" class="subtle-action" data-atlas-jump="atlas-observed-path">Open analytical canvas</button>
          <button type="button" class="subtle-action" data-atlas-jump="atlas-uncertainty-ruler">Open uncertainty ruler</button>
          <button type="button" class="subtle-action" data-atlas-jump="atlas-residuals">Open residual diagnostics</button>
        </div>
      </section>
      ${banners.join("")}
      <div class="split-columns analytical-layout">
        ${renderAnalyticalCanvasPanel({
          analysis: state.analysis,
          activeOverlay,
          selectedLaneEntry,
        })}
        <section class="panel diagnostic-chart evidence-rail">
          <div class="panel-head">
            <div>
              <p class="mini-kicker">Evidence rail</p>
              <h3>Decision guardrails</h3>
            </div>
          </div>
          ${renderSharedControls({ analysis: state.analysis })}
          <p>The analytical canvas carries the object itself. This rail keeps the decision constraints, benchmark-local context, and current selectors visible without taking over the stage.</p>
          <div class="detail-grid">
            ${detail("Active overlay", activeOverlay?.label || "n/a")}
            ${detail("Selected horizon", `h${state.selectedHorizon}`)}
            ${detail("Selected lane", selectedLaneEntry ? humanizeKey(selectedLaneEntry[0]) : "n/a")}
            ${detail("Lane snapshot", selectedRow ? normalizeLatestRow(selectedLaneEntry[0], selectedRow)[2]?.[1] || "n/a" : "n/a")}
            ${detail("Benchmark-local winner", benchmark?.portfolio_selection?.winner_submitter_id || "n/a")}
            ${detail("Workspace", state.analysis.workspace_root)}
          </div>
          ${selectedLaneEntry?.[1]?.evidence?.headline
            ? banner("Selected lane", selectedLaneEntry[1].evidence.headline)
            : ""}
        </section>
      </div>
      ${renderExplanationPanel("overview")}
    </div>
  `;
}

function renderAtlas() {
  const panel = tabPanels.atlas;
  if (!state.analysis) {
    panel.innerHTML = `<div class="panel"><p class="empty-state">Evidence appears after the first analysis run.</p></div>`;
    return;
  }
  const {
    dataset,
    operator_point: point,
    benchmark,
    descriptive_fit: descriptiveFit,
    descriptive_reconstruction: descriptiveReconstruction,
  } = state.analysis;
  const holisticEquation = state.analysis.holistic_equation;
  const probabilisticLanes = completedProbabilisticLaneEntries(state.analysis);
  const activeOverlay = currentDeterministicOverlay(state.analysis);
  const selectedLaneEntry = selectedProbabilisticLane(state.analysis);
  const residuals = buildResidualSeries(dataset.series, activeOverlay?.series || []);
  const residualStats = residualSummary(residuals);
  const distributionRow = rowForSelectedHorizon(state.analysis.probabilistic?.distribution);
  const quantileRow = rowForSelectedHorizon(state.analysis.probabilistic?.quantile);
  const intervalRow = rowForSelectedHorizon(state.analysis.probabilistic?.interval);
  const eventRow = rowForSelectedHorizon(state.analysis.probabilistic?.event_probability);

  const equationCards = [];
  if (hasHolisticClaim(state.analysis)) {
    equationCards.push(
      renderEquationCard({
        title: HOLISTIC_EQUATION_LABEL,
        kicker: "Composed equation",
        equation: holisticEquation.equation,
        badges: [
          {
            label: holisticClaimDescriptor(holisticEquation),
            tone: "ok",
          },
          {
            label: claimSurfaceLabel(
              holisticEquation.deterministic_source ||
                holisticEquation.source ||
                "descriptive_fit",
            ),
            tone: "",
          },
        ],
        honesty: holisticEquation.honesty_note || "",
        note:
          holisticEquation.equation?.coefficient_vector_label ||
          holisticEquation.equation?.residual_label ||
          "",
        tone: "accent",
        summaryRows: [
          [
            "Deterministic source",
            claimSurfaceLabel(holisticEquation.deterministic_source || "n/a"),
          ],
          [
            "Probabilistic source",
            claimSurfaceLabel(holisticEquation.probabilistic_source || "n/a"),
          ],
          ["Validation scope", holisticEquation.validation_scope_ref || "n/a"],
          ["Publication record", holisticEquation.publication_record_ref || "n/a"],
          ["Rows", String(holisticEquation.row_count || dataset.rows || "n/a")],
        ],
      }),
    );
  }
  if (hasDescriptiveReconstruction(state.analysis)) {
    equationCards.push(
      renderEquationCard({
        title: DESCRIPTIVE_RECONSTRUCTION_LABEL,
        kicker: "Descriptive only",
        equation: descriptiveReconstruction.equation,
        parameters: descriptiveReconstruction.equation?.parameter_summary || {},
        badges: [
          { label: "Descriptive only", tone: "ok" },
          {
            label: humanizePhrase(descriptiveReconstruction.source || "reconstruction"),
            tone: "",
          },
        ],
        honesty: descriptiveReconstruction.honesty_note || "",
        note:
          descriptiveReconstruction.equation?.coefficient_vector_label || "",
        tone: "accent",
        summaryRows: [
          ["Source", descriptiveReconstruction.source || "n/a"],
          ["Candidate", descriptiveReconstruction.candidate_id || "n/a"],
          [
            "R² vs mean",
            formatCell(
              descriptiveReconstruction.reconstruction_metrics?.r2_vs_mean_baseline,
            ),
          ],
          [
            "Normalized MAE",
            formatCell(
              descriptiveReconstruction.reconstruction_metrics?.normalized_mae,
            ),
          ],
        ],
      }),
    );
  }
  if (descriptiveFit?.status === "completed") {
    equationCards.push(
      renderEquationCard({
        title: BENCHMARK_DESCRIPTIVE_FIT_LABEL,
        kicker: "Benchmark-local",
        equation: descriptiveFit.equation,
        parameters: descriptiveFit.equation?.parameter_summary || {},
        badges: [
          { label: descriptiveFit.family_id || "unknown", tone: "ok" },
          {
            label: humanizePhrase(descriptiveFit.semantic_audit?.classification || "descriptive candidate"),
            tone: descriptiveFit.semantic_audit?.classification === "near_persistence" ? "warn" : "",
          },
        ],
        honesty:
          descriptiveSurfaceText(
            descriptiveFit.honesty_note,
            descriptiveFit.semantic_audit?.headline,
          ) || "",
        note: descriptiveSurfaceText(descriptiveFit.semantic_audit?.headline) || "",
        tone: "accent",
        summaryRows: [
          ["Source", descriptiveFit.submitter_id || descriptiveFit.source || "n/a"],
          ["Candidate", descriptiveFit.candidate_id || "n/a"],
          ["Naive vs fit MAE", formatImprovement(descriptiveFit.semantic_audit?.relative_improvement_vs_naive_last_value)],
          ["Suggested rerun", descriptiveFit.semantic_audit?.recommended_target_label || "n/a"],
        ],
      }),
    );
  }
  if (point?.status === "completed") {
    equationCards.push(
      renderEquationCard({
        title: "Operator point lane",
        kicker: "Point lane",
        equation: point.equation,
        parameters: point.equation?.parameter_summary || {},
        badges: [
          { label: humanizePhrase(point.publication?.status || point.status || "unknown"), tone: point.publication?.status === "abstained" ? "warn" : "ok" },
          { label: point.selected_family || "unknown", tone: "ok" },
        ],
        honesty: pointPublicationHeadlineForDisplay(state.analysis) || "",
        tone: "sea",
        summaryRows: [
          ["Replay", point.replay_verification || "n/a"],
          ["Score", nullableNumber(point.confirmatory_primary_score)],
          ["Comparison", point.comparison?.comparison_class_status || "n/a"],
          ["Search scope", point.search_scope?.scope_kind || "n/a"],
        ],
      }),
    );
  }
  for (const [mode, payload] of probabilisticLanes) {
    equationCards.push(
      renderEquationCard({
        title: `${humanizeKey(mode)} object`,
        kicker: humanizeKey(mode),
        equation: payload.equation,
        parameters: payload.equation?.parameter_summary || {},
        badges: [
          { label: humanizePhrase(payload.evidence?.strength || "unknown evidence"), tone: toneForEvidence(payload.evidence?.strength) },
          { label: payload.calibration?.status || "unknown", tone: payload.calibration?.passed ? "ok" : "warn" },
        ],
        honesty: payload.evidence?.headline || payload.search_scope?.headline || "",
        note: payload.equation?.delta_form_label || "",
        tone: mode === "distribution" ? "sea" : mode === "quantile" ? "accent" : mode === "interval" ? "moss" : "danger",
        actionLabel: mode === state.selectedLane ? "Selected lane" : "Make active lane",
        actionValue: mode,
        selected: mode === state.selectedLane,
        summaryRows: normalizeLatestRow(mode, rowForSelectedHorizon(payload)),
      }),
    );
  }

  panel.innerHTML = `
    <div class="stack atlas-stack workspace-stack">
      <section id="atlas-run-framing" class="panel atlas-major workspace-framing">
        <div class="panel-head">
          <div>
            <p class="mini-kicker">Evidence workspace</p>
            <h3>Analytical workspace for equation, path, residuals, and uncertainty</h3>
          </div>
          <div class="pill-row">
            ${pill(point?.publication?.status || point?.status || "unknown", point?.publication?.status === "abstained" ? "warn" : "ok")}
            ${pill(`h${state.selectedHorizon}`, "warn")}
            ${selectedLaneEntry ? pill(humanizeKey(selectedLaneEntry[0]), "sea") : ""}
          </div>
        </div>
        <p>Read the run as ordered observations, a candidate equation, residual evidence, stochastic support, and the claim ceiling those gates permit.</p>
        ${renderSharedControls({ analysis: state.analysis })}
        <div class="detail-grid">
          ${detail("Symbol", dataset.symbol)}
          ${detail("Range", `${dataset.date_range.start.slice(0, 10)} → ${dataset.date_range.end.slice(0, 10)}`)}
          ${detail("Operator publication", humanizePhrase(point?.publication?.status || point?.status || "unknown"))}
          ${detail("Benchmark winner", benchmark?.portfolio_selection?.winner_submitter_id || "n/a")}
          ${detail("Probabilistic lanes", String(probabilisticLanes.length))}
          ${detail("Active overlay", activeOverlay?.label || "n/a")}
        </div>
      </section>
      <div class="split-columns analytical-layout workspace-stage-grid workspace-core-grid">
        <div class="stack-tight workspace-primary-column">
          <div id="atlas-workspace-stage">
            ${renderWorkspaceEquationStage({
              analysis: state.analysis,
              activeOverlay,
              selectedLaneEntry,
            })}
          </div>
          <section id="atlas-observed-path" class="panel atlas-major workspace-canvas" data-workspace-region="analysis-canvas">
            <div class="panel-head">
          <div>
            <p class="mini-kicker">Observation timeline</p>
            <h3>Observed path versus active deterministic overlay</h3>
          </div>
            </div>
            <p>${escapeHtml(activeOverlay?.honesty || "Active overlay tracks the observed path; it does not explain all daily wiggles.")}</p>
            <div class="chart-shell">
              <div class="chart-root chart-root-primary" data-chart-root="primary-canvas" aria-label="Primary analytical canvas"></div>
              <div class="legend">
                ${legend("Observed", "#17212b")}
                ${legend(activeOverlay?.equation?.label || activeOverlay?.label || "Overlay", activeOverlay?.tone === "sea" ? "#2c617b" : activeOverlay?.tone === "moss" ? "#567060" : "#a45738")}
              </div>
            </div>
            <div class="detail-grid analytical-caption-grid">
              ${detail("Target transform", state.analysis.dataset.target.label)}
              ${detail("Deterministic overlay", activeOverlay?.label || "n/a")}
              ${detail("Selected lane", selectedLaneEntry ? humanizeKey(selectedLaneEntry[0]) : "n/a")}
              ${detail("Lane snapshot", activeLaneSnapshot(selectedLaneEntry))}
            </div>
          </section>
        </div>
        ${renderWorkspaceEvidenceRail({
          analysis: state.analysis,
          activeOverlay,
          selectedLaneEntry,
        })}
      </div>
      <section id="atlas-uncertainty-ruler" class="panel atlas-major">
        <div class="panel-head">
          <div>
            <p class="mini-kicker">Uncertainty ruler</p>
            <h3>Aligned latest-origin objects on a common price axis</h3>
          </div>
          <div class="pill-row">
            ${pill(`Active horizon h${state.selectedHorizon}`, "warn")}
            ${selectedLaneEntry ? pill(humanizeKey(selectedLaneEntry[0]), "ok") : ""}
          </div>
        </div>
        <p>Distribution, quantile, interval, and event-probability objects share one axis so you can compare center, spread, and realized outcomes without re-reading each lane card.</p>
        ${renderSharedControls({ analysis: state.analysis, showOverlay: false })}
        <div class="chart-shell">
          <div class="chart-root chart-root-uncertainty" data-chart-root="uncertainty-ruler" aria-label="Aligned uncertainty ruler"></div>
        </div>
        <div class="detail-grid">
          ${normalizeLatestRow("distribution", distributionRow).map(([label, value]) => detail(`Distribution ${label}`, value)).join("")}
          ${normalizeLatestRow("quantile", quantileRow).slice(2).map(([label, value]) => detail(`Quantile ${label}`, value)).join("")}
          ${normalizeLatestRow("interval", intervalRow).slice(2).map(([label, value]) => detail(`Interval ${label}`, value)).join("")}
          ${eventRow ? detail("Event threshold", formatEventDefinition(eventRow.event_definition)) : ""}
        </div>
      </section>
      <div class="split-columns workspace-support-grid">
        <section id="atlas-residuals" class="panel atlas-major">
          <div class="panel-head">
            <div>
              <p class="mini-kicker">Residual diagnostics</p>
              <h3>What the active deterministic overlay does not explain</h3>
            </div>
          </div>
          <p>Residuals are computed as observed minus fitted on the active deterministic overlay. Use this panel to separate level tracking from the misses that remain.</p>
          <div class="detail-grid">
            ${detail("Residual basis", (activeOverlay?.label || "n/a").toLowerCase())}
            ${detail("Mean absolute residual", residualStats.mae)}
            ${detail("Max residual", residualStats.max)}
            ${detail("Min residual", residualStats.min)}
          </div>
          <div class="chart-shell">
            ${buildResidualChartSvg(residuals)}
          </div>
        </section>
        ${renderExplanationPanel("atlas")}
      </div>
      <section id="atlas-equation-stack" class="stack atlas-anchor">
        <div class="panel">
          <div class="panel-head">
            <div>
              <p class="mini-kicker">Supporting objects</p>
              <h3>Deterministic, benchmark-local, and probabilistic equations</h3>
            </div>
          </div>
          <p>Each supporting card keeps the exact equation, delta form, and honesty framing available for drilldown after you inspect the main workspace.</p>
        </div>
        <div class="equation-grid">
          ${equationCards.join("")}
        </div>
      </section>
    </div>
  `;
}

function renderPoint() {
  const panel = tabPanels.point;
  const point = state.analysis?.operator_point;
  const descriptiveFit = state.analysis?.descriptive_fit;
  if (!state.analysis || !point) {
    panel.innerHTML = `<div class="panel"><p class="empty-state">Forecast path results appear here.</p></div>`;
    return;
  }
  if (point.status !== "completed") {
    renderFailure(panel, "Point lane failed", point.error?.message || "Unknown error.");
    return;
  }
  const activeOverlay = currentDeterministicOverlay(state.analysis);
  const residuals = buildResidualSeries(
    state.analysis.dataset.series,
    activeOverlay?.series || [],
  );
  const residualStats = residualSummary(residuals);
  const scorecardStatus = point.scorecard?.descriptive_status || point.scorecard?.status || "unknown";
  const predictionRows = point.prediction_rows || [];
  const lagDays = pointLagDays(point);
  panel.innerHTML = `
    <div class="stack">
      <section class="panel">
        <div class="panel-head">
          <div>
            <p class="mini-kicker">Forecast state</p>
            <h3>Operator equation, Point lane state, and residual basis</h3>
          </div>
        </div>
        ${renderSharedControls({ analysis: state.analysis, showHorizon: false })}
        <div class="detail-grid">
          ${detail("Residual basis", (activeOverlay?.label || "n/a").toLowerCase())}
          ${detail("Publication", humanizePhrase(point.publication?.status || point.status || "unknown"))}
          ${detail("Confirmatory score", nullableNumber(point.confirmatory_primary_score))}
          ${detail("Comparison status", point.comparison?.comparison_class_status || "n/a")}
          ${detail("Mean absolute residual", residualStats.mae)}
          ${detail("Scorecard", scorecardStatus)}
        </div>
      </section>
      <section class="panel point-lag-panel" data-point-story="lag-explanation">
        <div class="panel-head">
          <div>
            <p class="mini-kicker">Repeat-rule story</p>
            <h3>${escapeHtml(pointLagNarrative(point))}</h3>
          </div>
          <div class="pill-row">
            ${pill(`Lag ${lagDays}d`, "sea")}
            ${pill(humanizePhrase(point.publication?.status || point.status || "unknown"), point.publication?.status === "abstained" ? "warn" : "ok")}
          </div>
        </div>
        <p>The point lane is easiest to read as a copied-value rule: mark the source value at <code>t-${lagDays}</code>, compare it against the realized value at <code>t</code>, and read the gap as <code>observed(t) - y(t-${lagDays})</code>.</p>
        <div class="detail-grid">
          ${detail("Equation", point.equation?.label || "n/a")}
          ${detail("Prediction rows", String(predictionRows.length))}
          ${detail("Residual MAE", residualStats.mae)}
          ${detail("Comparison status", point.comparison?.comparison_class_status || "n/a")}
        </div>
      </section>
      ${renderExplanationPanel("point")}
      <div class="split-columns">
        ${renderEquationCard({
          title: "Operator point equation",
          kicker: "Point lane",
          equation: point.equation,
          parameters: point.equation?.parameter_summary || {},
          badges: [
            { label: humanizePhrase(point.publication?.status || point.status || "unknown"), tone: point.publication?.status === "abstained" ? "warn" : "ok" },
            { label: point.selected_family || "unknown", tone: "ok" },
          ],
          honesty: pointPublicationHeadlineForDisplay(state.analysis) || "",
          tone: "sea",
          summaryRows: [
            ["Replay", point.replay_verification || "n/a"],
            ["Comparison", point.comparison?.comparison_class_status || "n/a"],
            ["Search scope", point.search_scope?.scope_kind || "n/a"],
            ["Prediction rows", String(predictionRows.length)],
          ],
        })}
        <section class="panel diagnostic-chart">
          <div class="panel-head">
            <div>
              <p class="mini-kicker">Path comparison</p>
              <h3>Observed versus copied ${lagDays}-day point path</h3>
            </div>
          </div>
          <p>The chart compares the realized series to the point-lane replay so you can see where the repeat rule tracks the market and where it misses.</p>
          <div class="chart-shell">
            ${buildLineChartSvg({
              actualSeries: state.analysis.dataset.series,
              overlaySeries: point.equation.curve || [],
              yLabel: state.analysis.dataset.target.y_axis_label,
            })}
            <div class="legend">
              ${legend("Observed", "#17212b")}
              ${legend(point.equation.label || "Point path", "#2c617b")}
            </div>
          </div>
          ${point.abstention ? banner("Abstention", formatReasonCodes(point.abstention.reason_codes || [])) : ""}
        </section>
      </div>
      <div class="split-columns">
        <section class="panel">
          <div class="panel-head">
            <div>
              <p class="mini-kicker">Residual view</p>
              <h3>Active overlay residuals</h3>
            </div>
          </div>
          <div class="chart-shell">${buildResidualChartSvg(residuals)}</div>
        </section>
        <section class="panel">
          <div class="panel-head">
            <div>
              <p class="mini-kicker">Residual distribution</p>
              <h3>Error concentration</h3>
            </div>
          </div>
          <p>Use the residual histogram with the time-series view to distinguish one-off misses from persistent structural error.</p>
          <div class="chart-shell">${buildResidualDistributionSvg(residuals)}</div>
        </section>
      </div>
      <section class="panel">
        <div class="panel-head">
          <div>
            <p class="mini-kicker">Latest forecast rows</p>
            <h3>Sealed evaluation rows</h3>
          </div>
        </div>
        ${tableFromObjects(predictionRows.slice(-6))}
      </section>
      ${descriptiveFit?.status === "completed"
        ? renderEquationCard({
            title: BENCHMARK_DESCRIPTIVE_FIT_LABEL,
            kicker: "Benchmark-local",
            equation: descriptiveFit.equation,
            parameters: descriptiveFit.equation?.parameter_summary || {},
            badges: [
              { label: descriptiveFit.family_id || "unknown", tone: "ok" },
              {
                label: humanizePhrase(descriptiveFit.semantic_audit?.classification || "descriptive candidate"),
                tone: descriptiveFit.semantic_audit?.classification === "near_persistence" ? "warn" : "",
              },
            ],
            honesty:
              descriptiveSurfaceText(
                descriptiveFit.honesty_note,
                descriptiveFit.semantic_audit?.headline,
              ) || "",
            tone: "accent",
            summaryRows: [
              ["Source", descriptiveFit.submitter_id || descriptiveFit.source || "n/a"],
              ["Candidate", descriptiveFit.candidate_id || "n/a"],
              ["Suggested rerun", descriptiveFit.semantic_audit?.recommended_target_label || "n/a"],
              ["Why rerun", descriptiveFit.semantic_audit?.recommended_target_reason || "n/a"],
            ],
          })
        : ""}
    </div>
  `;
}

function renderProbabilistic() {
  const panel = tabPanels.probabilistic;
  const probabilistic = state.analysis?.probabilistic;
  if (!state.analysis || !probabilistic) {
    panel.innerHTML = `<div class="panel"><p class="empty-state">Calibration and probabilistic lanes appear here when enabled.</p></div>`;
    return;
  }
  const selectedLaneEntry = selectedProbabilisticLane(state.analysis);
  const selectedPayload = selectedLaneEntry?.[1] || null;
  panel.innerHTML = `
    <div class="stack">
      <section class="panel">
        <div class="panel-head">
          <div>
            <p class="mini-kicker">Calibration</p>
            <h3>Probabilistic lane objects under the shared horizon selector</h3>
          </div>
          <div class="pill-row">
            ${pill(`Active horizon h${state.selectedHorizon}`, "warn")}
            ${selectedLaneEntry ? pill(humanizeKey(selectedLaneEntry[0]), "ok") : ""}
          </div>
        </div>
        ${renderSharedControls({ analysis: state.analysis, showOverlay: false })}
      </section>
      <section class="panel probabilistic-summary-panel">
        <div class="panel-head">
          <div>
            <p class="mini-kicker">Claim discipline</p>
            <h3>Read calibration through sample, bins, and gate effect</h3>
          </div>
          <div class="pill-row">
            ${selectedLaneEntry ? pill(humanizeKey(selectedLaneEntry[0]), "sea") : ""}
            ${selectedLaneEntry?.[1]?.calibration?.status ? pill(selectedLaneEntry[1].calibration.status, selectedLaneEntry[1].calibration?.passed ? "ok" : "warn") : ""}
          </div>
        </div>
        <p>${escapeHtml(selectedPayload?.evidence?.headline || summarizeProbabilisticEvidence(state.analysis.probabilistic) || "Probabilistic evidence will appear here when a lane is selected.")}</p>
        <div class="detail-grid">
          ${detail("Family", laneFamily(selectedPayload))}
          ${detail("Lane status", humanizePhrase(selectedPayload?.evidence?.lane_status || selectedPayload?.lane_status || selectedPayload?.status || "n/a"))}
          ${detail("Sample size", String(selectedPayload?.evidence?.sample_size ?? "n/a"))}
          ${detail("Origins", String(selectedPayload?.evidence?.origin_count ?? "n/a"))}
          ${detail("Horizons", String(selectedPayload?.evidence?.horizon_count ?? "n/a"))}
          ${detail("Calibration effect", humanizePhrase(selectedPayload?.calibration?.gate_effect || "n/a"))}
          ${detail("Calibration bins", String(selectedPayload ? calibrationBinCount(selectedPayload) : "n/a"))}
          ${detail("Downgrade reasons", shortList(selectedPayload?.evidence?.downgrade_reason_codes || selectedPayload?.downgrade_reason_codes))}
        </div>
      </section>
      ${renderChangeAtlas(state.analysis)}
      ${renderExplanationPanel("probabilistic")}
      <div class="lane-grid">
        ${Object.entries(probabilistic)
          .map(([mode, payload]) => renderProbabilisticLane(mode, payload))
          .join("")}
      </div>
    </div>
  `;
}

function renderProbabilisticLane(mode, payload) {
  if (payload.status !== "completed") {
    return `
      <section class="panel">
        <div class="panel-head">
          <div>
            <p class="mini-kicker">${escapeHtml(humanizeKey(mode))}</p>
            <h3>Lane failed</h3>
          </div>
          ${pill("failed", "fail")}
        </div>
        <div class="failure">${escapeHtml(payload.error?.message || "Unknown error.")}</div>
      </section>
    `;
  }
  const diagnostics = payload.calibration?.diagnostics || [];
  const chart = renderProbabilisticChart(mode, payload.chart || {});
  const evidenceTone = toneForEvidence(payload.evidence?.strength);
  const selectedRow = rowForSelectedHorizon(payload);
  const family = laneFamily(payload, selectedRow);
  const residualRefs = refList(payload, "residual_history_refs");
  const stochasticRefs = refList(payload, "stochastic_model_refs");
  const downgradeReasons =
    payload.evidence?.downgrade_reason_codes || payload.downgrade_reason_codes || [];
  const isSelected = state.selectedLane === mode;
  return `
    <section class="panel probabilistic-lane${isSelected ? " selected-lane" : ""}">
      <div class="panel-head">
        <div>
          <p class="mini-kicker">${escapeHtml(humanizeKey(mode))}</p>
          <h3>${escapeHtml(payload.selected_family)}</h3>
        </div>
        <div class="pill-row">
          ${pill(humanizePhrase(payload.evidence?.strength || "unknown evidence"), evidenceTone)}
          ${pill(payload.calibration?.status || "unknown", payload.calibration?.passed ? "ok" : "warn")}
        </div>
      </div>
      <div class="equation-copy">
        ${renderEquationMarkup(payload.equation.label || payload.equation.structure_signature || "No explicit renderer available.", {
          className: "equation-formula",
          displayMode: true,
        })}
        ${payload.equation?.delta_form_label
          ? renderEquationMarkup(payload.equation.delta_form_label, {
              className: "equation-delta",
              displayMode: false,
            })
          : ""}
      </div>
      ${payload.evidence?.headline ? banner("Evidence", payload.evidence.headline) : ""}
      ${payload.search_scope?.headline ? banner("Search scope", payload.search_scope.headline) : ""}
      ${chart}
      <div class="detail-grid">
        ${detail("Active horizon", `h${state.selectedHorizon}`)}
        ${detail("Family", family)}
        ${detail("Lane Status", humanizePhrase(payload.evidence?.lane_status || payload.lane_status || payload.status || "n/a"))}
        ${detail("Replay", payload.replay_verification)}
        ${detail("Primary Score", nullableNumber(payload.aggregated_primary_score))}
        ${detail("Sample Size", String(payload.evidence?.sample_size ?? "n/a"))}
        ${detail("Origins / Horizons", `${payload.evidence?.origin_count ?? "n/a"} / ${payload.evidence?.horizon_count ?? "n/a"}`)}
        ${detail("Calibration Gate", humanizePhrase(payload.calibration?.gate_effect || "n/a"))}
        ${detail("Calibration Bins", String(calibrationBinCount(payload)))}
        ${detail("Stochastic Model Refs", stochasticRefs.length ? stochasticRefs.map((value) => truncateMiddle(value, 72)).join(", ") : "none")}
        ${detail("Residual History Refs", residualRefs.length ? residualRefs.map((value) => truncateMiddle(value, 72)).join(", ") : "none")}
        ${detail("Downgrade Reasons", shortList(downgradeReasons))}
        ${detail("Diagnostics", formatDiagnosticSummary(diagnostics))}
      </div>
      <div class="detail-grid">
        ${normalizeLatestRow(mode, selectedRow).map(([label, value]) => detail(label, value)).join("")}
      </div>
      <button type="button" class="subtle-action" data-lane-select="${escapeHtml(mode)}">
        ${isSelected ? "Selected lane" : "Make active lane"}
      </button>
    </section>
  `;
}

function renderChangeAtlas(analysis) {
  const atlas = analysis?.change_atlas;
  const metric = selectedChangeMetric(analysis);
  if (!atlas || !metric) {
    return "";
  }
  const selectedLaneEntry = selectedProbabilisticLane(analysis);
  const laneKind = selectedLaneEntry?.[0] || "distribution";
  const historical =
    atlas.historical?.[metric.id]?.[String(state.selectedHorizon)] || null;
  const forecast =
    atlas.forecast?.lanes?.[laneKind]?.[metric.id]?.[String(state.selectedHorizon)] ||
    null;

  return `
    <section class="panel">
      <div class="panel-head">
        <div>
          <p class="mini-kicker">Change atlas</p>
          <h3>Historical empirical distribution versus current forecast distribution</h3>
        </div>
        <div class="pill-row">
          ${pill(`h${state.selectedHorizon}`, "warn")}
          ${pill(metric.label || humanizeKey(metric.id), "ok")}
          ${selectedLaneEntry ? pill(humanizeKey(laneKind), "sea") : ""}
        </div>
      </div>
      ${renderSegmentedControls({
        label: "Change metric",
        items: availableChangeMetrics(analysis).map((item) => ({
          value: item.id,
          label: item.short_label || item.label || humanizeKey(item.id),
        })),
        activeValue: metric.id,
        dataAttribute: "data-change-metric",
      })}
      <p>${escapeHtml(atlas.headline || "Inspect the selected probabilistic lane in change space.")}</p>
      <div class="lane-grid">
        <section class="panel probabilistic-lane tone-moss">
          <div class="panel-head">
            <div>
              <p class="mini-kicker">Historical</p>
              <h3>Historical empirical distribution</h3>
            </div>
            ${historical ? pill(`${historical.sample_size || 0} samples`, "ok") : pill("Unavailable", "warn")}
          </div>
          ${historical
            ? `<div class="chart-shell">${buildHistogramSvg(historical.histogram || [], metric.id)}</div>`
            : `<p class="empty-state">Historical change distribution is unavailable at the selected horizon.</p>`}
          <div class="detail-grid">
            ${historical
              ? [
                  detail("Sample size", String(historical.sample_size ?? "n/a")),
                  detail("Latest", formatChangeMetricValue(metric.id, historical.latest_value)),
                  detail("Mean", formatChangeMetricValue(metric.id, historical.mean)),
                  detail("Stdev", formatChangeMetricValue(metric.id, historical.stdev)),
                  detail("Q0.1", formatChangeMetricValue(metric.id, quantileFromSummary(historical, 0.1))),
                  detail("Q0.9", formatChangeMetricValue(metric.id, quantileFromSummary(historical, 0.9))),
                ].join("")
              : ""}
          </div>
        </section>
        <section class="panel probabilistic-lane tone-sea">
          <div class="panel-head">
            <div>
              <p class="mini-kicker">Forecast</p>
              <h3>Forecast distribution</h3>
            </div>
            ${selectedLaneEntry ? pill(humanizeKey(laneKind), "sea") : ""}
          </div>
          ${forecast
            ? `<div class="chart-shell">${buildForecastChangeSvg(forecast, laneKind, metric.id)}</div>`
            : `<p class="empty-state">No projected forecast distribution is available for the selected lane and horizon.</p>`}
          <div class="detail-grid">
            ${forecast
              ? renderForecastChangeDetails(forecast, laneKind, metric.id)
              : ""}
          </div>
        </section>
      </div>
    </section>
  `;
}

function renderBenchmark() {
  const panel = tabPanels.benchmark;
  const benchmark = state.analysis?.benchmark;
  if (!state.analysis || !benchmark) {
    panel.innerHTML = `<div class="panel"><p class="empty-state">Search and codelength comparison appear here when enabled.</p></div>`;
    return;
  }
  if (benchmark.status !== "completed") {
    renderFailure(panel, "Benchmark failed", benchmark.error?.message || "Unknown error.");
    return;
  }
  const hasWinner = hasBenchmarkWinner(benchmark);
  const outcome = benchmarkOutcomeDetails(benchmark);
  const fieldRows = buildBenchmarkFieldRows(benchmark);
  const selectionField = `
    <section class="panel frontier-chart">
      <div class="panel-head">
          <div>
            <p class="mini-kicker">Search field</p>
            <h3>Backend finalists on the codelength axis</h3>
          </div>
      </div>
      <p>${escapeHtml(
        benchmarkSelectionExplanation(benchmark) ||
          "Backend finalists are plotted against total code bits so no-winner states remain visually explicit.",
      )}</p>
      <div class="chart-shell">
        ${buildBenchmarkSelectionFieldSvg(
          fieldRows,
          benchmark.portfolio_selection?.winner_submitter_id || "",
        )}
      </div>
    </section>
  `;
  const chart = barChartPanel({
    title: "Finalist total code bits",
    description:
      benchmarkSelectionExplanation(benchmark) ||
      "Benchmark-local finalist comparison.",
    series: benchmark.chart?.total_code_bits || [],
  });
  panel.innerHTML = `
    <div class="stack">
      <section class="panel benchmark-outcome-strip" data-benchmark-region="outcome-strip">
        <div class="panel-head">
          <div>
            <p class="mini-kicker">Search outcome</p>
            <h3>Winner, runner-up, codelength, and selection rule</h3>
          </div>
        </div>
        <p>The benchmark is local evidence for comparison, not a replacement for the operator publication path. Start here before reading the deeper evidence panels.</p>
        <div class="detail-grid">
          ${detail("Winner", outcome.winner)}
          ${detail("Runner-up", outcome.runnerUp)}
          ${detail("Winner bits", outcome.winnerBits)}
          ${detail("Runner-up bits", outcome.runnerUpBits)}
          ${detail("Margin", outcome.margin === "n/a" ? "n/a" : `${outcome.margin} code bits`)}
          ${detail("Selection rule", outcome.rule)}
        </div>
      </section>
      ${renderExplanationPanel("benchmark")}
      ${banner(
        "Benchmark honesty",
        hasWinner
          ? `Winner ${benchmark.portfolio_selection?.winner_submitter_id} is a benchmark-local selection, not an operator publication.`
          : NO_BENCHMARK_LOCAL_WINNER_PUBLICATION_COPY,
      )}
      <div class="split-columns">
        ${selectionField}
        <section class="panel">
          <div class="panel-head">
            <div>
              <p class="mini-kicker">Comparison framing</p>
              <h3>Operator lane vs benchmark-local outcome</h3>
            </div>
          </div>
          <div class="detail-grid">
            ${detail("Operator point lane", state.analysis.operator_point?.selected_family || "n/a")}
            ${detail("Operator publication", humanizePhrase(state.analysis.operator_point?.publication?.status || "n/a"))}
            ${detail("Benchmark winner", hasWinner ? benchmark.portfolio_selection?.winner_submitter_id : NO_BENCHMARK_LOCAL_WINNER_LABEL)}
            ${detail("Winner candidate", hasWinner ? benchmark.portfolio_selection?.winner_candidate_id || "n/a" : "No admissible finalist selected")}
            ${detail("Runner-up", benchmark.portfolio_selection?.selection_explanation_raw?.runner_up?.submitter_id || "n/a")}
            ${detail("Selection rule", humanizePhrase(benchmark.portfolio_selection?.selection_explanation_raw?.selection_rule || "n/a"))}
          </div>
          ${!hasWinner
            ? banner(
                "No-winner state",
                benchmarkSelectionExplanation(benchmark) ||
                  "No admissible finalist was selected.",
              )
            : ""}
        </section>
      </div>
      <div class="split-columns">
        ${state.analysis.descriptive_fit?.status === "completed"
          ? renderEquationCard({
              title: BENCHMARK_DESCRIPTIVE_FIT_LABEL,
              kicker: "Benchmark-local",
              equation: state.analysis.descriptive_fit.equation,
              parameters: state.analysis.descriptive_fit.equation?.parameter_summary || {},
              badges: [
                { label: state.analysis.descriptive_fit.family_id || "unknown", tone: "ok" },
                {
                  label: humanizePhrase(state.analysis.descriptive_fit.semantic_audit?.classification || "descriptive candidate"),
                  tone: state.analysis.descriptive_fit.semantic_audit?.classification === "near_persistence" ? "warn" : "",
                },
              ],
              honesty:
                descriptiveSurfaceText(
                  state.analysis.descriptive_fit.honesty_note,
                  state.analysis.descriptive_fit.semantic_audit?.headline,
                ) || "",
              tone: "accent",
              summaryRows: [
                ["Source", state.analysis.descriptive_fit.submitter_id || state.analysis.descriptive_fit.source || "n/a"],
                ["Candidate", state.analysis.descriptive_fit.candidate_id || "n/a"],
                ["Naive vs fit MAE", formatImprovement(state.analysis.descriptive_fit.semantic_audit?.relative_improvement_vs_naive_last_value)],
                ["Suggested rerun", state.analysis.descriptive_fit.semantic_audit?.recommended_target_label || "n/a"],
              ],
            })
          : `<section class="panel"><p class="empty-state">${NO_BENCHMARK_LOCAL_WINNER_RUN_COPY}</p></section>`}
        ${chart}
      </div>
      <div class="split-columns">
        <section class="panel diagnostic-chart">
          <div class="panel-head">
            <div>
              <p class="mini-kicker">Portfolio selection</p>
              <h3>Decision trace</h3>
            </div>
          </div>
          ${renderDecisionTrace(benchmark.portfolio_selection?.decision_trace || [], { hasWinner })}
        </section>
        <section class="panel">
          <div class="panel-head">
            <div>
              <p class="mini-kicker">Submitter ledgers</p>
              <h3>Finalist summaries</h3>
            </div>
          </div>
          ${tableFromObjects(buildBenchmarkFinalistRows(benchmark.submitters || []))}
        </section>
      </div>
    </div>
  `;
}

function renderArtifacts() {
  const panel = tabPanels.artifacts;
  if (!state.analysis) {
    panel.innerHTML = `<div class="panel"><p class="empty-state">Artifact paths appear after an analysis run.</p></div>`;
    return;
  }
  const groups = [
    {
      kicker: "Workspace",
      title: "Run root and normalized payload",
      note: "The workbench saves both the rendered payload and the workspace root so the same run can be reloaded or audited later.",
      items: [
        ["Workspace", state.analysis.workspace_root, "Saved run root for this workbench analysis."],
        ["Saved analysis", state.analysis.analysis_path, "Normalized payload rendered by the UI."],
      ],
    },
    {
      kicker: "Dataset lineage",
      title: "Observed series and transformed target",
      note: "These files separate the raw market history from the transformed target Euclid actually modeled.",
      items: [
        ["Dataset", state.analysis.dataset.dataset_csv, "Target-transformed dataset Euclid actually fit."],
        ["Raw history", state.analysis.dataset.raw_history_json, "Underlying market history before the target transform."],
      ],
    },
    {
      kicker: "Operator lane",
      title: "Point-lane manifests and replay roots",
      note: "This lineage preserves the operator-side search scope, manifest, and replayable outputs used for the point lane.",
      items: [
        ["Point manifest", state.analysis.operator_point?.manifest_path, "Operator point-lane manifest, including search scope."],
        ["Point output", state.analysis.operator_point?.output_root, "Replayable point-lane run artifacts."],
      ],
    },
    {
      kicker: "Benchmark lane",
      title: "Benchmark-local selection provenance",
      note: "Benchmark artifacts remain local evidence for finalist screening and reporting, not operator publication.",
      items: [
        ["Benchmark manifest", state.analysis.benchmark?.manifest_path, "Benchmark task manifest used for local finalist selection."],
        ["Benchmark report", state.analysis.benchmark?.report_path, "Rendered markdown report for predictive generalization."],
      ],
    },
  ];
  panel.innerHTML = `
    <div class="stack">
      <section class="panel artifact-role-summary" data-artifact-region="role-summary">
        <div class="panel-head">
          <div>
            <p class="mini-kicker">Role summary</p>
            <h3>What each saved artifact does before you drill into paths</h3>
          </div>
        </div>
        <p>Read the artifact roles first, then expand into exact paths only when you need to replay, audit, or copy a specific file location.</p>
        <div class="artifact-cards artifact-role-cards">
          <article class="artifact-card">
            <div class="detail-label">Inputs</div>
            <div class="detail-value">Raw market history and the transformed dataset that Euclid actually modeled.</div>
          </article>
          <article class="artifact-card">
            <div class="detail-label">Operator outputs</div>
            <div class="detail-value">Point-lane manifests and replay roots used for operator publication decisions.</div>
          </article>
          <article class="artifact-card">
            <div class="detail-label">Benchmark evidence</div>
            <div class="detail-value">Benchmark-local manifests and reports used to compare finalist candidates.</div>
          </article>
          <article class="artifact-card">
            <div class="detail-label">Reload surface</div>
            <div class="detail-value">Workspace root and normalized analysis payload for reopening the same run later.</div>
          </article>
        </div>
      </section>
      ${renderExplanationPanel("artifacts")}
      <section class="panel">
        <div class="panel-head">
          <div>
            <p class="mini-kicker">Lineage map</p>
            <h3>Saved outputs by analytical role</h3>
          </div>
        </div>
        <p>Artifacts are grouped by role so you can move from target construction to operator replay to benchmark-local reporting without reading one long path ledger.</p>
        <div class="lineage-grid">
          ${groups
            .map(
              (group) => `
                <section class="lineage-card">
                  <div class="panel-head">
                    <div>
                      <p class="mini-kicker">${escapeHtml(group.kicker)}</p>
                      <h3>${escapeHtml(group.title)}</h3>
                    </div>
                  </div>
                  <p>${escapeHtml(group.note)}</p>
                  <div class="artifact-list">
                    ${group.items
                      .filter(([, value]) => Boolean(value))
                      .map(
                        ([label, value, note]) => `
                          <div class="artifact-row artifact-card">
                            <div class="detail-label">${escapeHtml(label)}</div>
                            <div class="detail-value">${escapeHtml(note)}</div>
                            <div class="mono artifact-path path-full" title="${escapeHtml(String(value))}">${escapeHtml(String(value))}</div>
                            <button
                              type="button"
                              class="subtle-action"
                              data-copy-value="${escapeHtml(String(value))}"
                              data-copy-label="${escapeHtml(label)}"
                            >
                              Copy path
                            </button>
                          </div>
                        `,
                      )
                      .join("")}
                  </div>
                </section>
              `,
            )
            .join("")}
        </div>
      </section>
    </div>
  `;
}

function lineChartPanel({ title, description, actualSeries, overlaySeries, overlayLabel, yLabel, compact = false }) {
  const chart = buildLineChartSvg({
    actualSeries,
    overlaySeries,
    yLabel,
  });
  return `
    <section class="panel${compact ? "" : ""}">
      <div class="panel-head">
        <div>
          <p class="mini-kicker">${escapeHtml(yLabel)}</p>
          <h3>${escapeHtml(title)}</h3>
        </div>
      </div>
      <p>${escapeHtml(description || "")}</p>
      <div class="chart-shell">
        ${chart}
        <div class="legend">
          ${legend("Observed", "#17212b")}
          ${legend(overlayLabel || "Fitted", "#a45738")}
        </div>
      </div>
    </section>
  `;
}

function renderDescriptiveFitPanel(descriptiveFit, { title, yLabel }) {
  const audit = descriptiveFit.semantic_audit || {};
  return `
    <section class="panel">
      <div class="panel-head">
        <div>
          <p class="mini-kicker">${BENCHMARK_DESCRIPTIVE_FIT_LABEL}</p>
          <h3>${escapeHtml(title)}</h3>
        </div>
        <div class="pill-row">
          ${pill(descriptiveFit.family_id || "unknown", "ok")}
          ${audit.classification ? pill(humanizePhrase(audit.classification), audit.classification === "near_persistence" ? "warn" : "") : ""}
        </div>
      </div>
      <p>${escapeHtml(descriptiveSurfaceText(descriptiveFit.honesty_note) || "")}</p>
      ${audit.headline ? banner("Semantic audit", descriptiveSurfaceText(audit.headline)) : ""}
      <div class="detail-grid">
        ${detail("Source", descriptiveFit.submitter_id || descriptiveFit.source || "n/a")}
        ${detail("Candidate", descriptiveFit.candidate_id || "n/a")}
        ${detail("Family", descriptiveFit.family_id || "n/a")}
        ${detail("Total code bits", nullableNumber(descriptiveFit.metrics?.total_code_bits))}
        ${detail("Naive vs fit MAE", formatImprovement(audit.relative_improvement_vs_naive_last_value))}
        ${detail("Delta form", audit.delta_form_label || "n/a")}
        ${detail("Suggested rerun target", audit.recommended_target_label || "n/a")}
        ${detail("Why rerun", audit.recommended_target_reason || "n/a")}
      </div>
      ${renderEquationMarkup(
        descriptiveFit.equation?.label || descriptiveFit.equation?.structure_signature || "No explicit equation renderer available.",
        { className: "equation-formula", displayMode: true },
      )}
      <div class="chart-shell">
        ${buildLineChartSvg({
          actualSeries: state.analysis.dataset.series,
          overlaySeries: descriptiveFit.chart?.equation_curve || [],
          yLabel,
        })}
        <div class="legend">
          ${legend("Observed", "#17212b")}
          ${legend(
            descriptiveFit.equation?.label || BENCHMARK_DESCRIPTIVE_FIT_LABEL,
            "#a45738",
          )}
        </div>
      </div>
      ${renderParameterSummary(descriptiveFit.equation?.parameter_summary || {})}
    </section>
  `;
}

function renderExplanationPanel(pageKey) {
  const bundle = state.analysis?.llm_explanations;
  const fallback = buildBuiltInGuide(pageKey);
  const page = bundle?.status === "completed" ? bundle.pages?.[pageKey] : null;
  const guide = buildExplanationGuide(pageKey, page, fallback);
  if (!guide) return "";
  const narrative = mergeGuideText([guide.narrative], 460);
  const isBuiltIn = guide.source === "built-in";
  const statusLabel = bundle?.status ? humanizePhrase(bundle.status) : "built in";
  return `
    <section class="panel explainer-panel${isBuiltIn ? " builtin-explainer" : ""}">
      <div class="panel-head">
        <div>
          <p class="mini-kicker">Interpretation guide</p>
          <h3>${escapeHtml(guide.headline || (isBuiltIn ? "Built-in guide" : "Explain this page"))}</h3>
        </div>
        <div class="pill-row">
          ${pill(isBuiltIn ? "built-in guide" : "llm guide", "ok")}
          ${!isBuiltIn && bundle?.model ? pill(bundle.model, "") : ""}
          ${isBuiltIn && bundle?.status && bundle.status !== "completed" ? pill(statusLabel, bundle.status === "failed" ? "warn" : "") : ""}
        </div>
      </div>
      <p>${escapeHtml(guide.summary || "")}</p>
      ${narrative ? `<div class="guide-inline">${escapeHtml(narrative)}</div>` : ""}
      ${isBuiltIn && bundle?.message ? banner("LLM note", bundle.message) : ""}
      ${renderExplanationList("Key takeaways", guide.key_takeaways || guide.bullets || [])}
      ${renderExplanationList("Cautions", guide.cautions || [], "warn")}
      ${renderExplanationTerms(guide.terms || [])}
    </section>
  `;
}

function buildExplanationGuide(pageKey, page, fallback) {
  if (!page && !fallback) return null;
  return {
    source: page ? "llm" : "built-in",
    headline: firstPresentText(page?.headline, fallback?.headline, page ? "Explain this page" : "Built-in guide"),
    summary: mergeGuideText(
      [page?.summary, fallback?.summary],
      340,
    ),
    narrative: mergeGuideText(
      [page?.narrative, fallback?.narrative],
      460,
    ),
    key_takeaways: mergeGuideList(
      page?.key_takeaways || page?.bullets || [],
      fallback?.key_takeaways || fallback?.bullets || [],
      5,
    ),
    cautions: mergeGuideList(page?.cautions || [], fallback?.cautions || [], 4),
    terms: mergeGuideTerms(page?.terms || [], fallback?.terms || [], 4),
  };
}

function buildBuiltInGuide(pageKey) {
  const analysis = state.analysis;
  if (!analysis) return null;
  const point = analysis.operator_point;
  const benchmark = analysis.benchmark;
  const descriptiveFit = analysis.descriptive_fit;
  const activeOverlay = currentDeterministicOverlay(analysis);
  const selectedLaneEntry = selectedProbabilisticLane(analysis);
  const targetLabel = analysis.dataset?.target?.label || "target transform";
  const pointStatus = point?.publication?.status || point?.status || "unknown";
  const selectedLaneLabel = selectedLaneEntry ? humanizeKey(selectedLaneEntry[0]) : "no probabilistic lane";
  const selectedOverlayLabel = activeOverlay?.label || "active deterministic overlay";
  const commonTerms = [
    {
      term: "benchmark-local",
      meaning:
        "A result chosen only inside the benchmark comparison lane; it is not the same thing as an operator publication.",
    },
    {
      term: "point lane",
      meaning:
        "The operator-facing deterministic lane used for publication gates, abstentions, and replay verification.",
    },
    {
      term: "target transform",
      meaning:
        `The transformed series Euclid actually models. In this run the active target is ${targetLabel}.`,
    },
  ];

  if (pageKey === "overview") {
    return {
      headline: "Built-in guide",
      summary:
        pointStatus === "abstained"
          ? "Read this run as an operator abstention first: benchmark-local and probabilistic evidence add context, but they do not upgrade the run into a publishable claim."
          : "Read this run from the operator publication outward: benchmark-local and probabilistic evidence explain the claim, but they do not replace the point-lane decision.",
      narrative:
        mergeGuideText(
          [
            pointPublicationHeadlineForDisplay(analysis),
            benchmarkSelectionExplanation(benchmark),
            summarizeProbabilisticEvidence(analysis.probabilistic),
            "Use this page to understand which result class actually survived and which panels only add supporting evidence.",
          ],
          420,
        ),
      key_takeaways: [
        `The operator lane is currently ${humanizePhrase(point?.publication?.status || point?.status || "unknown")}.`,
        descriptiveSurfaceText(
          descriptiveFit?.honesty_note,
          descriptiveFit?.semantic_audit?.headline,
        ) || `${BENCHMARK_DESCRIPTIVE_FIT_LABEL} stays separate from the operator publication path.`,
        summarizeProbabilisticEvidence(analysis.probabilistic) ||
          "Use the active horizon and lane selectors to compare probabilistic objects on the same footing.",
      ],
      cautions: [
        "A benchmark-local winner can still coexist with an operator abstention.",
        "Thin probabilistic evidence means a passed gate may rest on smoke-sized calibration.",
      ],
      terms: commonTerms,
    };
  }

  if (pageKey === "atlas") {
    return {
      headline: "Built-in guide",
      summary:
        "Use the workspace to explain how the active deterministic overlay, selected probabilistic lane, and residual diagnostics fit together before you decide whether a smooth curve actually supports a strong claim.",
      narrative:
        mergeGuideText(
          [
            `${selectedOverlayLabel} is the active deterministic object and ${selectedLaneLabel} is the selected uncertainty view at h${state.selectedHorizon}.`,
            "That combination tells you what the model is claiming on the canvas, what uncertainty object sits beside it, and what the residuals still leave unexplained.",
          ],
          420,
        ),
      key_takeaways: [
        "The equation stack surfaces deterministic and probabilistic objects together.",
        "Residuals show what the active deterministic overlay fails to explain.",
        "The uncertainty ruler aligns different probabilistic summaries on one selected horizon.",
      ],
      cautions: [
        "A clean-looking deterministic curve can still be benchmark-local or non-publishable.",
      ],
      terms: commonTerms,
    };
  }

  if (pageKey === "point") {
    return {
      headline: "Built-in guide",
      summary:
        "The point page tells you whether Euclid actually had a publishable deterministic claim and why that claim survived or failed the publication gates.",
      narrative:
        mergeGuideText(
          [
            pointPublicationHeadlineForDisplay(analysis),
            `The selected family on this run is ${point?.selected_family || "n/a"}.`,
            "Read the residual basis and sealed forecast rows as evidence for how the candidate behaved under evaluation, not as a free-form story about market direction.",
          ],
          420,
        ),
      key_takeaways: [
        `Current point-lane family: ${point?.selected_family || "n/a"}.`,
        "Residual charts help you see whether the active deterministic overlay leaves structured error behind.",
        "Prediction rows are sealed evaluation outputs, not freeform what-if forecasts.",
      ],
      cautions: [
        "A replay-verified candidate can still fail publication gates and become an abstention.",
      ],
      terms: commonTerms,
    };
  }

  if (pageKey === "probabilistic") {
    return {
      headline: "Built-in guide",
      summary:
        "The probabilistic page explains uncertainty quality, not just uncertainty shape. Read calibration beside sample size and origin count before trusting any lane.",
      narrative:
        mergeGuideText(
          [
            selectedLaneEntry?.[1]?.evidence?.headline,
            `${selectedLaneLabel} is currently active at h${state.selectedHorizon}.`,
            "Switch lanes to compare what changes in center, spread, threshold, or coverage while keeping the evaluation horizon fixed.",
          ],
          420,
        ),
      key_takeaways: [
        `Active horizon: h${state.selectedHorizon}.`,
        selectedLaneEntry
          ? `${humanizeKey(selectedLaneEntry[0])} is the active lane for cross-page comparisons.`
          : "No active probabilistic lane is available.",
        "Calibration status and sample size matter as much as the plotted interval or probability.",
      ],
      cautions: [
        "Different probabilistic objects are not interchangeable; each summarizes uncertainty differently.",
        "A pass on a tiny calibration sample is not strong probabilistic evidence.",
      ],
      terms: commonTerms,
    };
  }

  if (pageKey === "benchmark") {
    return {
      headline: "Built-in guide",
      summary:
        "The benchmark page explains whether any local finalist actually won the comparison field, and why that local result still does or does not matter for operator publication.",
      narrative:
        mergeGuideText(
          [
            benchmarkSelectionExplanation(benchmark),
            `The operator lane remains ${humanizePhrase(pointStatus)}.`,
            "Use the selection field and decision trace together to understand whether a local winner exists at all and whether it changes the analytical reading of the run.",
          ],
          420,
        ),
      key_takeaways: [
        hasBenchmarkWinner(benchmark)
          ? `Benchmark-local winner: ${benchmark?.portfolio_selection?.winner_submitter_id || "n/a"}.`
          : NO_BENCHMARK_LOCAL_WINNER_RUN_COPY,
        "Code-length comparisons show local finalist efficiency, not publication validity.",
        "Decision traces explain how the benchmark field narrowed or failed to narrow the finalists.",
      ],
      cautions: [
        "Do not treat the benchmark-local winner as the same thing as the operator publication path.",
      ],
      terms: commonTerms,
    };
  }

  if (pageKey === "artifacts") {
    return {
      headline: "Built-in guide",
      summary:
        "The artifacts page explains provenance: which files produced this run and where to go when you need to audit, reload, or replay the analysis.",
      narrative:
        mergeGuideText(
          [
            `The current workspace root is ${analysis.workspace_root}.`,
            "Use these grouped paths to move from raw inputs to transformed targets to operator and benchmark outputs without losing the causal chain behind the analysis.",
          ],
          420,
        ),
      key_takeaways: [
        "Dataset lineage distinguishes raw market history from the transformed target actually modeled.",
        "Operator and benchmark outputs are grouped by analytical role instead of one long path ledger.",
        "Copy-path actions are there to move quickly into deeper audit work.",
      ],
      cautions: [
        "Filesystem paths are provenance, not interpretation; pair them with the analytical tabs before drawing conclusions.",
      ],
      terms: commonTerms,
    };
  }

  return null;
}

function renderExplanationList(title, items, tone = "") {
  const normalized = Array.isArray(items)
    ? items.map((item) => String(item || "").trim()).filter(Boolean)
    : [];
  if (!normalized.length) return "";
  return `
    <div class="stack-tight">
      <div class="detail-label">${escapeHtml(title)}</div>
      <ul class="plain-bullets ${tone ? `plain-bullets-${tone}` : ""}">
        ${normalized.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
      </ul>
    </div>
  `;
}

function renderExplanationTerms(items) {
  const normalized = Array.isArray(items)
    ? items
        .filter((item) => item && typeof item === "object")
        .map((item) => ({
          term: String(item.term || "").trim(),
          meaning: String(item.meaning || "").trim(),
        }))
        .filter((item) => item.term && item.meaning)
    : [];
  if (!normalized.length) return "";
  return `
    <div class="term-grid">
      ${normalized
        .map(
          (item) => `
            <div class="term-card">
              <div class="detail-label">${escapeHtml(item.term)}</div>
              <div class="detail-value">${escapeHtml(item.meaning)}</div>
            </div>
          `,
        )
        .join("")}
    </div>
  `;
}

function barChartPanel({ title, description, series }) {
  return `
    <section class="panel">
      <div class="panel-head">
        <div>
          <p class="mini-kicker">Benchmark</p>
          <h3>${escapeHtml(title)}</h3>
        </div>
      </div>
      <p>${escapeHtml(description || "")}</p>
      <div class="chart-shell">${buildBarChartSvg(series || [])}</div>
    </section>
  `;
}

function renderProbabilisticChart(mode, chart) {
  if (mode === "distribution" || mode === "interval") {
    return `<div class="chart-shell">${buildBandChartSvg(chart.forecast_bands || [])}</div>`;
  }
  if (mode === "event_probability") {
    return `<div class="chart-shell">${buildProbabilityChartSvg(chart.forecast_probabilities || [])}</div>`;
  }
  if (mode === "quantile") {
    return `<div class="chart-shell">${buildQuantileChartSvg(chart.forecast_quantiles || [])}</div>`;
  }
  return "";
}

function buildHistogramSvg(bins, metricId) {
  const histogram = Array.isArray(bins) ? bins : [];
  const usable = histogram.filter((bin) => Number(bin?.count) > 0);
  if (!usable.length) {
    return `<div class="empty-state">No histogram data available.</div>`;
  }
  const width = 820;
  const height = 240;
  const padding = { top: 20, right: 16, bottom: 40, left: 44 };
  const maxCount = Math.max(...usable.map((bin) => Number(bin.count) || 0), 1);
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;
  const barWidth = innerWidth / usable.length;
  return `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Historical ${escapeHtml(metricId)} histogram">
      <rect x="0" y="0" width="${width}" height="${height}" fill="transparent"></rect>
      ${usable
        .map((bin, index) => {
          const count = Number(bin.count) || 0;
          const x = padding.left + index * barWidth + 4;
          const heightValue = (count / maxCount) * innerHeight;
          const y = padding.top + innerHeight - heightValue;
          return `
            <rect x="${x}" y="${y}" width="${Math.max(8, barWidth - 8)}" height="${heightValue}" rx="8" fill="rgba(86, 112, 96, 0.78)">
              <title>${escapeHtml(`${formatChangeMetricValue(metricId, bin.lower)} to ${formatChangeMetricValue(metricId, bin.upper)} • ${count} samples`)}</title>
            </rect>
          `;
        })
        .join("")}
      <line x1="${padding.left}" y1="${padding.top + innerHeight}" x2="${width - padding.right}" y2="${padding.top + innerHeight}" stroke="rgba(23,33,43,0.18)" />
      <text x="${padding.left}" y="${height - 12}" fill="rgba(23,33,43,0.58)" font-size="12">${escapeHtml(formatChangeMetricValue(metricId, usable[0].lower))}</text>
      <text x="${width - padding.right - 72}" y="${height - 12}" fill="rgba(23,33,43,0.58)" font-size="12">${escapeHtml(formatChangeMetricValue(metricId, usable.at(-1).upper))}</text>
    </svg>
  `;
}

function buildForecastChangeSvg(summary, laneKind, metricId) {
  if (laneKind === "event_probability") {
    return buildProbabilitySnapshotSvg(summary);
  }
  const marks = [];
  if (summary.lower != null) marks.push(Number(summary.lower));
  if (summary.center != null) marks.push(Number(summary.center));
  if (summary.upper != null) marks.push(Number(summary.upper));
  if (summary.realized != null) marks.push(Number(summary.realized));
  if (Array.isArray(summary.quantiles)) {
    for (const quantile of summary.quantiles) {
      if (Number.isFinite(Number(quantile?.value))) {
        marks.push(Number(quantile.value));
      }
    }
  }
  if (!marks.length) {
    return `<div class="empty-state">No projected forecast snapshot is available.</div>`;
  }
  const width = 820;
  const height = 180;
  const padding = { left: 48, right: 24, top: 28, bottom: 36 };
  const minValue = Math.min(...marks, 0);
  const maxValue = Math.max(...marks, 0);
  const domainMin = minValue - (maxValue - minValue || 1) * 0.12;
  const domainMax = maxValue + (maxValue - minValue || 1) * 0.12;
  const xScale = (value) =>
    padding.left +
    ((value - domainMin) / (domainMax - domainMin || 1)) *
      (width - padding.left - padding.right);
  const baselineY = height / 2;
  const zeroX = xScale(0);
  const quantiles = Array.isArray(summary.quantiles) ? summary.quantiles : [];
  return `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Projected ${escapeHtml(metricId)} forecast snapshot">
      <rect x="0" y="0" width="${width}" height="${height}" fill="transparent"></rect>
      <line x1="${padding.left}" y1="${baselineY}" x2="${width - padding.right}" y2="${baselineY}" stroke="rgba(23,33,43,0.18)" />
      <line x1="${zeroX}" y1="${padding.top}" x2="${zeroX}" y2="${height - padding.bottom}" stroke="rgba(164,87,56,0.28)" stroke-dasharray="5 5" />
      ${summary.lower != null && summary.upper != null
        ? `<line x1="${xScale(Number(summary.lower))}" y1="${baselineY}" x2="${xScale(Number(summary.upper))}" y2="${baselineY}" stroke="#2c617b" stroke-width="10" stroke-linecap="round" opacity="0.35" />`
        : ""}
      ${quantiles
        .map(
          (quantile) => `
            <circle cx="${xScale(Number(quantile.value))}" cy="${baselineY}" r="${Number(quantile.level) === 0.5 ? 7 : 5}" fill="${Number(quantile.level) === 0.5 ? "#2c617b" : "#567060"}">
              <title>${escapeHtml(`Q${quantile.level}: ${formatChangeMetricValue(metricId, quantile.value)}`)}</title>
            </circle>
          `,
        )
        .join("")}
      ${summary.center != null
        ? `<circle cx="${xScale(Number(summary.center))}" cy="${baselineY}" r="7" fill="#2c617b"><title>${escapeHtml(`Center ${formatChangeMetricValue(metricId, summary.center)}`)}</title></circle>`
        : ""}
      ${summary.realized != null
        ? `<circle cx="${xScale(Number(summary.realized))}" cy="${baselineY}" r="6" fill="#17212b"><title>${escapeHtml(`Realized ${formatChangeMetricValue(metricId, summary.realized)}`)}</title></circle>`
        : ""}
      <text x="${padding.left}" y="${height - 12}" fill="rgba(23,33,43,0.58)" font-size="12">${escapeHtml(formatChangeMetricValue(metricId, domainMin))}</text>
      <text x="${width - padding.right - 84}" y="${height - 12}" fill="rgba(23,33,43,0.58)" font-size="12">${escapeHtml(formatChangeMetricValue(metricId, domainMax))}</text>
    </svg>
  `;
}

function buildProbabilitySnapshotSvg(summary) {
  const probability = Number(summary?.probability);
  if (!Number.isFinite(probability)) {
    return `<div class="empty-state">No projected event probability is available.</div>`;
  }
  const width = 820;
  const height = 160;
  const fillWidth = Math.max(0, Math.min(1, probability)) * 680;
  return `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Projected event probability">
      <rect x="52" y="60" width="680" height="20" rx="10" fill="rgba(23,33,43,0.1)"></rect>
      <rect x="52" y="60" width="${fillWidth}" height="20" rx="10" fill="#2c617b"></rect>
      <text x="52" y="42" fill="rgba(23,33,43,0.72)" font-size="13">${escapeHtml(formatEventDefinition(summary.event_definition) || "Event probability")}</text>
      <text x="744" y="76" fill="rgba(23,33,43,0.72)" font-size="13">${escapeHtml(formatPercent(probability))}</text>
    </svg>
  `;
}

function renderForecastChangeDetails(summary, laneKind, metricId) {
  const rows = [
    detail("Origin", String(summary.origin_time || "n/a").slice(0, 10)),
    detail("Horizon", `h${summary.horizon ?? "n/a"}`),
    detail("Origin close", formatNumber(summary.origin_close)),
  ];
  if (laneKind === "event_probability") {
    rows.push(detail("Probability", formatPercent(summary.probability)));
    rows.push(detail("Realized event", formatCell(summary.realized_event)));
    rows.push(
      detail(
        "Event",
        formatEventDefinition(summary.event_definition) || formatCell(summary.event_definition),
      ),
    );
    return rows.join("");
  }
  if (summary.lower != null) {
    rows.push(detail("Lower", formatChangeMetricValue(metricId, summary.lower)));
  }
  if (summary.center != null) {
    rows.push(detail("Center", formatChangeMetricValue(metricId, summary.center)));
  }
  if (summary.upper != null) {
    rows.push(detail("Upper", formatChangeMetricValue(metricId, summary.upper)));
  }
  if (summary.realized != null) {
    rows.push(detail("Realized", formatChangeMetricValue(metricId, summary.realized)));
  }
  if (Array.isArray(summary.quantiles)) {
    rows.push(detail("Q0.1", formatChangeMetricValue(metricId, quantileValueFromList(summary.quantiles, 0.1))));
    rows.push(detail("Q0.5", formatChangeMetricValue(metricId, quantileValueFromList(summary.quantiles, 0.5))));
    rows.push(detail("Q0.9", formatChangeMetricValue(metricId, quantileValueFromList(summary.quantiles, 0.9))));
  }
  return rows.join("");
}

function buildLineChartSvg({ actualSeries, overlaySeries, yLabel }) {
  const seriesA = (actualSeries || []).map((point, index) => ({
    x: index,
    y: Number(point.observed_value),
    label: point.event_time,
  }));
  const seriesB = (overlaySeries || []).map((point, index) => ({
    x: index,
    y: Number(point.fitted_value),
    label: point.event_time,
  }));
  const values = [...seriesA, ...seriesB].map((point) => point.y).filter(Number.isFinite);
  if (!values.length) return `<div class="empty-state">No chart data available.</div>`;
  const width = 820;
  const height = 340;
  const padding = { top: 28, right: 28, bottom: 36, left: 56 };
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const yMin = minValue - (maxValue - minValue || 1) * 0.08;
  const yMax = maxValue + (maxValue - minValue || 1) * 0.08;
  const xScale = (index, count) =>
    padding.left +
    (count <= 1 ? 0 : (index / (count - 1)) * (width - padding.left - padding.right));
  const yScale = (value) =>
    height - padding.bottom - ((value - yMin) / (yMax - yMin || 1)) * (height - padding.top - padding.bottom);
  const actualPath = polylinePath(seriesA, xScale, yScale, seriesA.length);
  const overlayPath = polylinePath(seriesB, xScale, yScale, Math.max(seriesA.length, seriesB.length));
  const yTicks = Array.from({ length: 5 }, (_, index) => {
    const ratio = index / 4;
    const value = yMax - (yMax - yMin) * ratio;
    return {
      value,
      y: yScale(value),
    };
  });
  const actualStep = Math.max(1, Math.ceil(seriesA.length / 10));
  const overlayStep = Math.max(1, Math.ceil(seriesB.length / 10));
  return `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(yLabel)} chart">
      <rect x="0" y="0" width="${width}" height="${height}" fill="transparent"></rect>
      ${yTicks
        .map(
          (tick) => `
            <line x1="${padding.left}" y1="${tick.y}" x2="${width - padding.right}" y2="${tick.y}" stroke="rgba(23,33,43,0.08)" />
            <text x="10" y="${tick.y + 4}" fill="rgba(23,33,43,0.48)" font-size="12">${escapeHtml(formatNumber(tick.value))}</text>
          `,
        )
        .join("")}
      <line x1="${padding.left}" y1="${height - padding.bottom}" x2="${width - padding.right}" y2="${height - padding.bottom}" stroke="rgba(23,33,43,0.18)" />
      <line x1="${padding.left}" y1="${padding.top}" x2="${padding.left}" y2="${height - padding.bottom}" stroke="rgba(23,33,43,0.18)" />
      <text x="${padding.left}" y="${padding.top - 8}" fill="rgba(23,33,43,0.58)" font-size="12">${escapeHtml(yLabel)}</text>
      <text x="${padding.left}" y="${height - 12}" fill="rgba(23,33,43,0.58)" font-size="12">${escapeHtml(seriesA[0]?.label?.slice(0, 10) || "")}</text>
      <text x="${width - padding.right - 72}" y="${height - 12}" fill="rgba(23,33,43,0.58)" font-size="12">${escapeHtml(seriesA.at(-1)?.label?.slice(0, 10) || "")}</text>
      <path d="${actualPath}" fill="none" stroke="#17212b" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"></path>
      ${seriesA
        .filter((_, index) => index % actualStep === 0 || index === seriesA.length - 1)
        .map(
          (point) => `
            <circle cx="${xScale(point.x, seriesA.length)}" cy="${yScale(point.y)}" r="3.5" fill="#17212b">
              <title>${escapeHtml(`${String(point.label || "").slice(0, 10)} • observed ${formatNumber(point.y)}`)}</title>
            </circle>
          `,
        )
        .join("")}
      ${
        overlayPath
          ? `
            <path d="${overlayPath}" fill="none" stroke="#a45738" stroke-width="2.5" stroke-dasharray="6 4" stroke-linejoin="round" stroke-linecap="round"></path>
            ${seriesB
              .filter((_, index) => index % overlayStep === 0 || index === seriesB.length - 1)
              .map(
                (point) => `
                  <circle cx="${xScale(point.x, Math.max(seriesA.length, seriesB.length))}" cy="${yScale(point.y)}" r="3" fill="#a45738">
                    <title>${escapeHtml(`${String(point.label || "").slice(0, 10)} • fitted ${formatNumber(point.y)}`)}</title>
                  </circle>
                `,
              )
              .join("")}
          `
          : ""
      }
    </svg>
  `;
}

function buildBarChartSvg(series) {
  if (!series.length) return `<div class="empty-state">No benchmark chart data available.</div>`;
  const width = 820;
  const rowHeight = 40;
  const height = 40 + series.length * rowHeight;
  const maxValue = Math.max(...series.map((item) => Number(item.total_code_bits) || 0), 1);
  return `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Benchmark code bits comparison">
      ${series
        .map((item, index) => {
          const y = 24 + index * rowHeight;
          const value = Number(item.total_code_bits) || 0;
          const barWidth = (value / maxValue) * 420;
          return `
            <text x="20" y="${y + 14}" font-size="12" fill="rgba(23,33,43,0.72)">${escapeHtml(item.submitter_id || "unknown")}</text>
            <rect x="260" y="${y}" rx="10" ry="10" width="${barWidth}" height="16" fill="#2c617b">
              <title>${escapeHtml(`${item.submitter_id || "unknown"} • ${formatNumber(value)} total code bits`)}</title>
            </rect>
            <text x="${260 + barWidth + 12}" y="${y + 13}" font-size="12" fill="rgba(23,33,43,0.72)">${formatNumber(value)}</text>
          `;
        })
        .join("")}
    </svg>
  `;
}

function buildBenchmarkFieldRows(benchmark) {
  return (benchmark?.submitters || []).map((submitter, index) => ({
    rank: index + 1,
    submitter_id: submitter.submitter_id || `submitter_${index + 1}`,
    candidate_id: submitter.selected_candidate_id || null,
    total_code_bits: Number(submitter.selected_candidate_metrics?.total_code_bits),
    families: (submitter.backend_families || []).join(", ") || "n/a",
  }));
}

function buildBenchmarkSelectionFieldSvg(rows, winnerSubmitterId) {
  if (!rows.length) {
    return `<div class="empty-state">No benchmark submitter rows are available.</div>`;
  }
  const finiteRows = rows.filter((row) => Number.isFinite(row.total_code_bits));
  if (!finiteRows.length) {
    return `<div class="empty-state">No admissible finalist was selected, so the benchmark field has no finite code-length points to plot.</div>`;
  }
  const width = 860;
  const height = 88 + rows.length * 56;
  const padding = { top: 28, right: 40, bottom: 40, left: 180 };
  const min = Math.min(...finiteRows.map((row) => row.total_code_bits));
  const max = Math.max(...finiteRows.map((row) => row.total_code_bits));
  const xScale = (value) =>
    padding.left + ((value - min) / (max - min || 1)) * (width - padding.left - padding.right);
  return `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Benchmark selection field">
      <line x1="${padding.left}" y1="${height - padding.bottom}" x2="${width - padding.right}" y2="${height - padding.bottom}" stroke="rgba(23,33,43,0.18)" />
      <text x="${padding.left}" y="${height - 12}" font-size="12" fill="rgba(23,33,43,0.58)">${escapeHtml(formatNumber(min))}</text>
      <text x="${width - padding.right - 48}" y="${height - 12}" font-size="12" fill="rgba(23,33,43,0.58)">${escapeHtml(formatNumber(max))}</text>
      ${rows
        .map((row, index) => {
          const y = padding.top + index * 56;
          if (!Number.isFinite(row.total_code_bits)) {
            return `
              <text x="24" y="${y + 6}" font-size="12" fill="rgba(23,33,43,0.72)">${escapeHtml(row.submitter_id)}</text>
              <text x="${padding.left}" y="${y + 6}" font-size="12" fill="#8f433b">No finalist</text>
              <line x1="${padding.left}" y1="${y + 18}" x2="${width - padding.right}" y2="${y + 18}" stroke="rgba(143,67,59,0.14)" stroke-dasharray="3 6" />
            `;
          }
          const x = xScale(row.total_code_bits);
          const isWinner = row.submitter_id === winnerSubmitterId;
          return `
            <text x="24" y="${y + 6}" font-size="12" fill="rgba(23,33,43,0.72)">${escapeHtml(row.submitter_id)}</text>
            <text x="24" y="${y + 22}" font-size="11" fill="rgba(23,33,43,0.52)">${escapeHtml(row.families)}</text>
            <line x1="${padding.left}" y1="${y + 18}" x2="${x}" y2="${y + 18}" stroke="${isWinner ? "#a45738" : "rgba(44,97,123,0.28)"}" stroke-width="3" />
            <circle cx="${x}" cy="${y + 18}" r="${isWinner ? 8 : 6}" fill="${isWinner ? "#a45738" : "#2c617b"}">
              <title>${escapeHtml(`${row.submitter_id} • ${formatNumber(row.total_code_bits)} total code bits${isWinner ? " • winner" : ""}`)}</title>
            </circle>
            <text x="${x + 12}" y="${y + 22}" font-size="12" fill="rgba(23,33,43,0.72)">${escapeHtml(formatNumber(row.total_code_bits))}</text>
          `;
        })
        .join("")}
    </svg>
  `;
}

function buildBandChartSvg(rows) {
  if (!rows.length) return `<div class="empty-state">No probabilistic range data available.</div>`;
  if (rows.length < 2) {
    return `<div class="empty-state">Only ${rows.length} forecast row is available. Need at least 2 rows to plot a range history.</div>`;
  }
  const width = 820;
  const height = 220;
  const padding = { top: 20, right: 24, bottom: 32, left: 56 };
  const lows = rows.map((row) => Number(row.lower));
  const highs = rows.map((row) => Number(row.upper));
  const actuals = rows.map((row) => Number(row.realized_observation));
  const values = [...lows, ...highs, ...actuals].filter(Number.isFinite);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const xScale = (index) =>
    padding.left + (rows.length <= 1 ? 0 : (index / (rows.length - 1)) * (width - padding.left - padding.right));
  const yScale = (value) =>
    height - padding.bottom - ((value - min) / (max - min || 1)) * (height - padding.top - padding.bottom);
  return `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Forecast range chart">
      <line x1="${padding.left}" y1="${height - padding.bottom}" x2="${width - padding.right}" y2="${height - padding.bottom}" stroke="rgba(23,33,43,0.18)" />
      ${rows
        .map((row, index) => {
          const x = xScale(index);
          return `
            <line x1="${x}" y1="${yScale(Number(row.lower))}" x2="${x}" y2="${yScale(Number(row.upper))}" stroke="#567060" stroke-width="6" stroke-linecap="round"></line>
            <circle cx="${x}" cy="${yScale(Number(row.realized_observation))}" r="4" fill="#a45738"></circle>
          `;
        })
        .join("")}
    </svg>
  `;
}

function buildQuantileChartSvg(rows) {
  if (!rows.length) return `<div class="empty-state">No quantile data available.</div>`;
  const normalized = rows.map((row) => {
    const quantiles = new Map((row.quantiles || []).map((entry) => [String(entry.level), Number(entry.value)]));
    return {
      low: quantiles.get("0.1"),
      mid: quantiles.get("0.5"),
      high: quantiles.get("0.9"),
      actual: Number(row.realized_observation),
    };
  });
  return buildBandChartSvg(
    normalized.map((row, index) => ({
      lower: row.low,
      upper: row.high,
      realized_observation: row.actual,
      index,
    })),
  );
}

function buildProbabilityChartSvg(rows) {
  if (!rows.length) return `<div class="empty-state">No event-probability data available.</div>`;
  const width = 820;
  const rowHeight = 36;
  const height = 30 + rowHeight * rows.length;
  return `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Event probability chart">
      ${rows
        .map((row, index) => {
          const probability = Number(row.event_probability) || 0;
          const y = 18 + index * rowHeight;
          return `
            <text x="20" y="${y + 13}" font-size="12" fill="rgba(23,33,43,0.72)">h${index + 1}</text>
            <rect x="80" y="${y}" rx="10" ry="10" width="${probability * 520}" height="16" fill="#a45738"></rect>
            <text x="${80 + probability * 520 + 12}" y="${y + 13}" font-size="12" fill="rgba(23,33,43,0.72)">${formatPercent(probability)}</text>
          `;
        })
        .join("")}
    </svg>
  `;
}

function polylinePath(points, xScale, yScale, count) {
  if (!points.length) return "";
  return points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${xScale(point.x, count)} ${yScale(point.y)}`)
    .join(" ");
}

function rollingWindowForTarget(targetId) {
  if (targetId === "daily_return" || targetId === "log_return") return 20;
  return 30;
}

function buildRollingMeanSeries(actualSeries, windowSize) {
  const points = Array.isArray(actualSeries) ? actualSeries : [];
  if (!points.length) return [];
  const window = Math.max(2, Math.min(points.length, Number(windowSize) || 20));
  const series = [];
  let runningSum = 0;
  for (let index = 0; index < points.length; index += 1) {
    runningSum += Number(points[index].observed_value) || 0;
    if (index >= window) {
      runningSum -= Number(points[index - window].observed_value) || 0;
    }
    const divisor = Math.min(index + 1, window);
    series.push({
      event_time: points[index].event_time,
      fitted_value: runningSum / divisor,
    });
  }
  return series;
}

function buildHeroSummary(analysis) {
  const strongestCard = strongestEquationCard(analysis);
  if (strongestCard) {
    if (strongestCard.honesty) return strongestCard.honesty;
    if (strongestCard.title === HOLISTIC_EQUATION_LABEL) {
      return "Holistic equation cleared the strict joint evidence gate.";
    }
    if (strongestCard.title === PREDICTIVE_SYMBOLIC_LAW_LABEL) {
      return "Predictive symbolic law cleared the point-lane publication gate.";
    }
    if (strongestCard.title === DESCRIPTIVE_RECONSTRUCTION_LABEL) {
      return "Descriptive reconstruction is shown because no publishable law survived and the workbench generated a descriptive-only explicit-time path fit.";
    }
    if (strongestCard.title === DESCRIPTIVE_APPROXIMATION_LABEL) {
      return "Best available descriptive approximation is shown because no stronger claim cleared publication.";
    }
  }

  const point = analysis.operator_point;
  const descriptiveReconstruction = analysis.descriptive_reconstruction;
  const descriptiveFit = analysis.descriptive_fit;
  const benchmark = analysis.benchmark;
  const clauses = [];
  if (point?.publication?.status === "abstained") {
    clauses.push("operator abstained after point-lane publication checks");
  } else if (point?.selected_family) {
    clauses.push(`${humanizePhrase(point.selected_family)} selected in the point lane`);
  }
  if (descriptiveFit?.semantic_audit?.classification === "near_persistence") {
    clauses.push(
      "benchmark-local descriptive fit is near persistence on this raw close target",
    );
  } else if (descriptiveReconstruction?.status === "completed") {
    clauses.push(
      "workbench descriptive reconstruction is shown separately as a descriptive-only path fit",
    );
  } else if (descriptiveFit?.status === "completed") {
    clauses.push(
      "benchmark-local descriptive fit is shown separately as benchmark-local evidence",
    );
  } else if (benchmark?.descriptive_fit_status?.status === "absent_no_admissible_candidate") {
    clauses.push(
      "no benchmark-local winner was retained for this target",
    );
  }
  if (benchmark?.portfolio_selection?.winner_submitter_id) {
    clauses.push(`benchmark-local winner ${benchmark.portfolio_selection.winner_submitter_id}`);
  }
  return clauses.join(" • ") || point?.error?.message || "Point lane failed.";
}

function summarizeProbabilisticEvidence(probabilistic) {
  if (!probabilistic || typeof probabilistic !== "object") return "";
  const thin = Object.entries(probabilistic).filter(
    ([, payload]) => payload?.status === "completed" && payload?.evidence?.strength === "thin",
  );
  if (!thin.length) return "";
  return `${thin.length} lane${thin.length === 1 ? "" : "s"} only have smoke-sized calibration evidence. Passed gates here do not imply strong probabilistic proof.`;
}

function renderParameterSummary(params) {
  const entries = Object.entries(params || {});
  if (!entries.length) {
    return "";
  }
  return `
    <div class="parameter-strip">
      ${entries
        .map(
          ([key, value]) => `
            <div class="parameter-chip">
              <span class="detail-label">${escapeHtml(humanizeKey(key))}</span>
              <strong>${escapeHtml(formatCell(value))}</strong>
            </div>
          `,
        )
        .join("")}
    </div>
  `;
}

function renderDecisionTrace(trace, { hasWinner = true } = {}) {
  if (!trace?.length) {
    return `<p class="empty-state">${hasWinner ? "No decision trace available." : "No admissible finalist was selected."}</p>`;
  }
  return `
    <div class="trace-list">
      ${trace
        .map(
          (entry) => `
            <div class="trace-step">
              <div class="detail-label">${escapeHtml(humanizeKey(entry.step || "step"))}</div>
              <div class="detail-value">${escapeHtml(summarizeTraceEntry(entry))}</div>
            </div>
          `,
        )
        .join("")}
    </div>
  `;
}

function summarizeTraceEntry(entry) {
  if (!entry || typeof entry !== "object") return String(entry || "n/a");
  if (entry.step === "collect_submitter_finalists") {
    const finalists = Object.entries(entry.family_finalists || {})
      .map(([submitterId, candidateId]) => `${submitterId}: ${candidateId}`)
      .join(" • ");
    return finalists || "Collected family finalists.";
  }
  if (entry.step === "rank_submitter_finalists") {
    return `Ordered candidate ids: ${(entry.ordered_candidate_ids || []).join(", ") || "n/a"}`;
  }
  if (entry.step === "select_portfolio_winner") {
    if (!entry.selected_backend_family && !entry.selected_candidate_id) {
      return "No admissible finalist was selected.";
    }
    return `Selected ${entry.selected_backend_family || "unknown"} / ${entry.selected_candidate_id || "unknown"}`;
  }
  return JSON.stringify(entry);
}

function buildBenchmarkFinalistRows(submitters) {
  return (submitters || []).map((submitter) => ({
    submitter: submitter.submitter_id || "unknown",
    candidate: submitter.selected_candidate_id || "none",
    families: (submitter.backend_families || []).join(", ") || "n/a",
    total_code_bits: submitter.selected_candidate_metrics?.total_code_bits ?? null,
  }));
}

function compactNarrative(text, limit = 280) {
  const normalized = String(text || "").trim();
  if (!normalized) return "";
  const sentences = normalized.match(/[^.!?]+[.!?]*/g)?.map((sentence) => sentence.trim()).filter(Boolean) || [normalized];
  const compact = sentences.slice(0, 2).join(" ").trim();
  if (compact.length <= limit) return compact;
  return `${compact.slice(0, limit - 1).trim()}…`;
}

function mergeGuideText(parts, limit = 360) {
  const seen = new Set();
  const merged = [];
  for (const part of parts) {
    const normalized = String(part || "").replace(/\s+/g, " ").trim();
    if (!normalized) continue;
    const key = normalized.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    merged.push(normalized);
  }
  if (!merged.length) return "";
  const joined = merged.join(" ");
  if (joined.length <= limit) return joined;
  return `${joined.slice(0, limit - 1).trim()}…`;
}

function mergeGuideList(primary, secondary, limit = 4) {
  const merged = [];
  const seen = new Set();
  for (const item of [...(primary || []), ...(secondary || [])]) {
    const normalized = String(item || "").replace(/\s+/g, " ").trim();
    if (!normalized) continue;
    const key = normalized.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    merged.push(normalized);
    if (merged.length >= limit) break;
  }
  return merged;
}

function mergeGuideTerms(primary, secondary, limit = 4) {
  const merged = [];
  const seen = new Set();
  for (const item of [...(primary || []), ...(secondary || [])]) {
    if (!item || typeof item !== "object") continue;
    const term = String(item.term || "").trim();
    const meaning = String(item.meaning || "").trim();
    if (!term || !meaning) continue;
    const key = term.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    merged.push({ term, meaning });
    if (merged.length >= limit) break;
  }
  return merged;
}

function firstPresentText(...values) {
  for (const value of values) {
    const normalized = String(value || "").trim();
    if (normalized) return normalized;
  }
  return "";
}

function disposeAnalyticalCharts() {
  for (const chart of chartRegistry) {
    try {
      chart.dispose?.();
    } catch (error) {
      // Ignore dispose errors from torn-down DOM nodes during rerenders.
    }
  }
  chartRegistry.clear();
}

function mountAnalyticalCharts() {
  const primaryRoots = Array.from(
    document.querySelectorAll('[data-chart-root="primary-canvas"]'),
  ).filter((root) => root.closest(".tab-panel")?.classList.contains("active"));
  for (const root of primaryRoots) {
    mountPrimaryCanvasChart(root);
  }
  const uncertaintyRoot = Array.from(
    document.querySelectorAll('[data-chart-root="uncertainty-ruler"]'),
  ).find((root) => root.closest(".tab-panel")?.classList.contains("active"));
  if (uncertaintyRoot) {
    mountUncertaintyRulerChart(uncertaintyRoot);
  }
}

function mountPrimaryCanvasChart(root) {
  if (!state.analysis || !root) return;
  const activeOverlay = currentDeterministicOverlay(state.analysis);
  const overlaySeries =
    activeOverlay?.series ||
    buildRollingMeanSeries(
      state.analysis.dataset.series,
      rollingWindowForTarget(state.analysis.dataset.target?.id),
    );
  const fallback = buildLineChartSvg({
    actualSeries: state.analysis.dataset.series,
    overlaySeries,
    yLabel: state.analysis.dataset.target.y_axis_label,
  });
  const echartsApi = globalThis.echarts;
  root.innerHTML = "";
  if (!echartsApi?.init) {
    root.innerHTML = fallback;
    return;
  }
  const chart = echartsApi.init(root, null, { renderer: "svg" });
  const categories = state.analysis.dataset.series.map((point) =>
    String(point.event_time || "").slice(0, 10),
  );
  const observedData = state.analysis.dataset.series.map((point) =>
    Number(point.observed_value),
  );
  const overlayData = overlaySeries.map((point) => Number(point.fitted_value));
  const overlayColor =
    activeOverlay?.tone === "sea" ? "#2c617b" : activeOverlay?.tone === "moss" ? "#567060" : "#a45738";
  chart.setOption({
    animation: false,
    grid: { left: 56, right: 20, top: 26, bottom: 48 },
    tooltip: { trigger: "axis" },
    legend: { show: false },
    xAxis: {
      type: "category",
      boundaryGap: false,
      data: categories,
      axisLabel: { color: "rgba(23,33,43,0.62)" },
      axisLine: { lineStyle: { color: "rgba(23,33,43,0.18)" } },
    },
    yAxis: {
      type: "value",
      axisLabel: { color: "rgba(23,33,43,0.62)" },
      splitLine: { lineStyle: { color: "rgba(23,33,43,0.08)" } },
    },
    dataZoom: [
      { type: "inside", zoomLock: false },
      { type: "slider", height: 18, bottom: 10, brushSelect: false },
    ],
    series: [
      {
        name: "Observed",
        type: "line",
        smooth: false,
        showSymbol: false,
        data: observedData,
        lineStyle: { width: 3, color: "#17212b" },
      },
      {
        name: activeOverlay?.label || "Overlay",
        type: "line",
        smooth: false,
        showSymbol: false,
        data: overlayData,
        lineStyle: { width: 2.5, type: "dashed", color: overlayColor },
      },
    ],
  });
  chartRegistry.add(chart);
}

function mountUncertaintyRulerChart(root) {
  if (!state.analysis || !root) return;
  const distributionRow = rowForSelectedHorizon(state.analysis.probabilistic?.distribution);
  const quantileRow = rowForSelectedHorizon(state.analysis.probabilistic?.quantile);
  const intervalRow = rowForSelectedHorizon(state.analysis.probabilistic?.interval);
  const eventRow = rowForSelectedHorizon(state.analysis.probabilistic?.event_probability);
  const realized =
    Number(distributionRow?.realized_observation ?? quantileRow?.realized_observation ?? intervalRow?.realized_observation);
  const eventThreshold = parseEventThreshold(eventRow?.event_definition);
  const eventLabel = formatEventDefinition(eventRow?.event_definition);
  const fallback = buildUncertaintyRulerSvg({
    distributionRow,
    quantileRow,
    intervalRow,
    eventRow,
  });
  const echartsApi = globalThis.echarts;
  root.innerHTML = "";
  if (!echartsApi?.init) {
    root.innerHTML = fallback;
    return;
  }

  const rows = [
    {
      index: 0,
      label: "Distribution",
      lower: Number(distributionRow?.location) - Number(distributionRow?.scale),
      upper: Number(distributionRow?.location) + Number(distributionRow?.scale),
      center: Number(distributionRow?.location),
      color: "#2c617b",
    },
    {
      index: 1,
      label: "Quantile",
      lower: quantileValue(quantileRow, 0.1),
      upper: quantileValue(quantileRow, 0.9),
      center: quantileValue(quantileRow, 0.5),
      color: "#a45738",
    },
    {
      index: 2,
      label: "Interval",
      lower: Number(intervalRow?.lower_bound),
      upper: Number(intervalRow?.upper_bound),
      center:
        Number(intervalRow?.lower_bound) +
        (Number(intervalRow?.upper_bound) - Number(intervalRow?.lower_bound)) / 2,
      color: "#567060",
    },
  ].filter(
    (row) =>
      Number.isFinite(row.lower) &&
      Number.isFinite(row.upper) &&
      Number.isFinite(row.center),
  );
  if (!rows.length) {
    root.innerHTML = fallback;
    return;
  }

  const values = rows.flatMap((row) => [row.lower, row.upper, row.center]);
  if (Number.isFinite(realized)) {
    values.push(realized);
  }
  if (Number.isFinite(eventThreshold)) {
    values.push(eventThreshold);
  }
  const annotationLines = [];
  if (Number.isFinite(realized)) {
    annotationLines.push({
      xAxis: realized,
      label: {
        formatter: `Realized ${formatNumber(realized)}`,
        color: "#17212b",
      },
      lineStyle: {
        color: "#17212b",
        type: "dashed",
        width: 1.5,
      },
    });
  }
  if (Number.isFinite(eventThreshold)) {
    annotationLines.push({
      xAxis: eventThreshold,
      label: {
        formatter: `Threshold ${formatNumber(eventThreshold)} • ${formatPercent(Number(eventRow?.event_probability) || 0)}`,
        color: "#8f433b",
      },
      lineStyle: {
        color: "#8f433b",
        type: "dotted",
        width: 1.5,
      },
    });
  }
  const chart = echartsApi.init(root, null, { renderer: "svg" });
  chart.setOption({
    animation: false,
    grid: { left: 100, right: 20, top: 18, bottom: 36 },
    tooltip: {
      trigger: "item",
      formatter: (params) => {
        const row = rows[params.dataIndex] || rows[params.value?.[0]];
        if (!row) return "";
        return `${row.label}<br/>low: ${formatNumber(row.lower)}<br/>center: ${formatNumber(row.center)}<br/>high: ${formatNumber(row.upper)}`;
      },
    },
    xAxis: {
      type: "value",
      min: Math.min(...values),
      max: Math.max(...values),
      axisLabel: { color: "rgba(23,33,43,0.62)" },
      splitLine: { lineStyle: { color: "rgba(23,33,43,0.08)" } },
    },
    yAxis: {
      type: "category",
      inverse: true,
      data: rows.map((row) => row.label),
      axisLabel: { color: "rgba(23,33,43,0.72)" },
      axisTick: { show: false },
      axisLine: { show: false },
    },
    series: [
      {
        type: "custom",
        data: rows.map((row) => [
          row.index,
          row.lower,
          row.upper,
          row.center,
          row.color,
        ]),
        renderItem(params, api) {
          const categoryIndex = api.value(0);
          const lowPoint = api.coord([api.value(1), categoryIndex]);
          const highPoint = api.coord([api.value(2), categoryIndex]);
          const centerPoint = api.coord([api.value(3), categoryIndex]);
          const color = api.value(4);
          return {
            type: "group",
            children: [
              {
                type: "line",
                shape: {
                  x1: lowPoint[0],
                  y1: lowPoint[1],
                  x2: highPoint[0],
                  y2: highPoint[1],
                },
                style: {
                  stroke: color,
                  lineWidth: 8,
                  lineCap: "round",
                },
              },
              {
                type: "circle",
                shape: {
                  cx: centerPoint[0],
                  cy: centerPoint[1],
                  r: 6,
                },
                style: {
                  fill: color,
                },
              },
            ],
          };
        },
      },
      {
        type: "scatter",
        data: rows.map((row) => [row.center, row.label]),
        symbolSize: 0,
        silent: true,
        tooltip: { show: false },
        markLine: annotationLines.length
          ? {
              symbol: "none",
              silent: true,
              data: annotationLines,
            }
          : undefined,
      },
    ],
    graphic:
      !Number.isFinite(eventThreshold) && eventLabel
        ? [
            {
              type: "text",
              left: 24,
              bottom: 12,
              style: {
                text: `${eventLabel} • ${formatPercent(Number(eventRow?.event_probability) || 0)}`,
                fill: "#8f433b",
                fontSize: 12,
              },
            },
          ]
        : [],
  });
  chartRegistry.add(chart);
}

function hasBenchmarkWinner(benchmark) {
  return Boolean(benchmark?.portfolio_selection?.winner_submitter_id);
}

function benchmarkSelectionExplanation(benchmark) {
  const explanation = benchmark?.portfolio_selection?.selection_explanation;
  if (typeof explanation === "string" && explanation.trim()) {
    const trimmed = explanation.trim();
    if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
      try {
        const parsed = JSON.parse(trimmed);
        return benchmarkSelectionExplanation({
          portfolio_selection: {
            ...benchmark?.portfolio_selection,
            selection_explanation: null,
            selection_explanation_raw: parsed,
          },
        });
      } catch (error) {
        return trimmed;
      }
    }
    return trimmed;
  }
  const raw = benchmark?.portfolio_selection?.selection_explanation_raw;
  if (!raw || typeof raw !== "object") {
    return hasBenchmarkWinner(benchmark)
      ? `Winner ${benchmark?.portfolio_selection?.winner_submitter_id || "unknown"} was selected.`
      : "No admissible finalist was selected.";
  }
  const clauses = [];
  if (raw.selection_rule) {
    clauses.push(humanizePhrase(raw.selection_rule));
  }
  if (raw.runner_up?.submitter_id) {
    clauses.push(`runner-up ${raw.runner_up.submitter_id}`);
  }
  if (hasBenchmarkWinner(benchmark)) {
    clauses.unshift(
      `Winner ${benchmark?.portfolio_selection?.winner_submitter_id || "unknown"} selected`,
    );
  } else {
    clauses.unshift("No admissible finalist was selected");
  }
  return clauses.join(" • ");
}

function renderFailure(container, title, message) {
  container.innerHTML = renderFailureBox(title, message);
}

function renderFailureBox(title, message) {
  return `
    <div class="panel">
      <div class="panel-head">
        <div>
          <p class="mini-kicker">Failure</p>
          <h3>${escapeHtml(title)}</h3>
        </div>
      </div>
      <div class="failure">${escapeHtml(message)}</div>
    </div>
  `;
}

function banner(title, message) {
  return `
    <div class="banner">
      <strong>${escapeHtml(title)}.</strong> ${escapeHtml(message)}
    </div>
  `;
}

function metric(label, value) {
  return `
    <div class="metric">
      <div class="mini-kicker">${escapeHtml(label)}</div>
      <div class="metric-value">${escapeHtml(value)}</div>
    </div>
  `;
}

function detail(label, value) {
  return `
    <div class="detail-row">
      <div class="detail-label">${escapeHtml(label)}</div>
      <div class="detail-value">${escapeHtml(String(value ?? "n/a"))}</div>
    </div>
  `;
}

function humanizePhrase(value) {
  if (!value) return "n/a";
  return String(value).replaceAll("_", " ");
}

function sanitizeNarrativeText(value) {
  const text = String(value || "").trim();
  if (!text) return "";
  return text
    .replace(/\bHolistic law\b/gi, HOLISTIC_EQUATION_LABEL)
    .replace(/\bPredictive claim\b/gi, PREDICTIVE_SYMBOLIC_LAW_LABEL)
    .replace(/\bTop-line law\b/gi, "Top-line claim")
    .replace(/\btop-line law\b/gi, "top-line claim")
    .replace(/\bactive deterministic law\b/gi, "active deterministic overlay")
    .replace(/\boperator claim\b/gi, "operator publication");
}

function sanitizeDescriptiveSurfaceText(value) {
  const text = sanitizeNarrativeText(value);
  if (!text) return "";
  return text
    .replace(/\bHolistic equation\b/gi, "joint claim")
    .replace(/\bPredictive symbolic law\b/gi, "predictive claim")
    .replace(
      /\bdeterministic and probabilistic laws together\b/gi,
      "deterministic and probabilistic objects together",
    );
}

function descriptiveSurfaceText(...values) {
  for (const value of values) {
    const text = sanitizeDescriptiveSurfaceText(value);
    if (text) return text;
  }
  return "";
}

function isDescriptiveOnlyAnalysis(analysis) {
  return Boolean(analysis) && !hasHolisticClaim(analysis) && !hasPredictiveLaw(analysis);
}

function pointPublicationHeadlineForDisplay(analysis, fallback = "") {
  const point = analysis?.operator_point;
  const text = point?.publication?.headline || point?.search_scope?.headline || fallback;
  if (!text) return "";
  return isDescriptiveOnlyAnalysis(analysis)
    ? sanitizeDescriptiveSurfaceText(text)
    : text;
}

function claimSurfaceLabel(value) {
  const normalized = String(value || "").trim();
  if (!normalized) return "n/a";
  if (normalized === "holistic_equation") return HOLISTIC_EQUATION_LABEL;
  if (normalized === "predictive_law") return PREDICTIVE_SYMBOLIC_LAW_LABEL;
  if (normalized === "descriptive_reconstruction") {
    return DESCRIPTIVE_RECONSTRUCTION_LABEL;
  }
  if (normalized === "descriptive_fit") return BENCHMARK_DESCRIPTIVE_FIT_LABEL;
  return humanizePhrase(normalized);
}

function findTargetSpec(targetId) {
  return state.config?.target_specs?.find((target) => target.id === targetId) || null;
}

function formatDiagnosticSummary(diagnostics) {
  const items = Array.isArray(diagnostics) ? diagnostics : diagnostics ? [diagnostics] : [];
  if (!items.length) return "none";
  const first = items[0];
  if (!first || typeof first !== "object") return String(first);
  const diagnosticId = first.diagnostic_id ? humanizePhrase(first.diagnostic_id) : null;
  const summary = Object.entries(first)
    .filter(([key, value]) => key !== "diagnostic_id" && value !== null && value !== undefined)
    .slice(0, 4)
    .map(([key, value]) => `${humanizeKey(key)}: ${formatCell(value)}`);
  return [diagnosticId, ...summary].filter(Boolean).join(" • ");
}

function pill(label, tone = "") {
  return `<span class="pill ${tone}">${escapeHtml(String(label || "unknown"))}</span>`;
}

function legend(label, color) {
  return `<div class="legend-item"><span class="swatch" style="background:${color}"></span><span>${escapeHtml(label)}</span></div>`;
}

function tableFromObjects(rows) {
  if (!rows?.length) return `<p class="empty-state">No rows available.</p>`;
  const keys = Object.keys(rows[0]);
  return `
    <div class="table-wrap">
      <table>
        <thead>
          <tr>${keys.map((key) => `<th>${escapeHtml(humanizeKey(key))}</th>`).join("")}</tr>
        </thead>
        <tbody>
          ${rows
            .map(
              (row) => `
                <tr>
                  ${keys
                    .map((key) => `<td>${escapeHtml(formatCell(row[key]))}</td>`)
                    .join("")}
                </tr>
              `,
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

function formatCell(value) {
  if (typeof value === "number") return formatNumber(value);
  if (isEventDefinition(value)) {
    return formatEventDefinition(value) || JSON.stringify(value, null, 2);
  }
  if (Array.isArray(value) || typeof value === "object") {
    return JSON.stringify(value, null, 2);
  }
  if (typeof value === "string" && value.includes("T") && value.includes("Z")) {
    return formatTimestamp(value);
  }
  return value == null ? "n/a" : String(value);
}

function quantileFromSummary(summary, level) {
  const quantiles = Array.isArray(summary?.quantiles) ? summary.quantiles : [];
  const match = quantiles.find((item) => Number(item?.level) === Number(level));
  return match ? Number(match.value) : null;
}

function quantileValueFromList(quantiles, level) {
  const list = Array.isArray(quantiles) ? quantiles : [];
  const match = list.find((item) => Number(item?.level) === Number(level));
  return match ? Number(match.value) : null;
}

function formatReasonCodes(codes) {
  if (!codes?.length) return "No explicit reason codes provided.";
  return `Reason codes: ${codes.join(", ")}.`;
}

function formatImprovement(value) {
  if (value == null || Number.isNaN(Number(value))) return "n/a";
  const percent = Number(value) * 100;
  return `${percent >= 0 ? "+" : ""}${percent.toFixed(1)}% vs naive y(t)=y(t-1)`;
}

function formatNumber(value) {
  if (value == null || Number.isNaN(Number(value))) return "n/a";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 6 });
}

function nullableNumber(value) {
  return value == null ? "n/a" : formatNumber(value);
}

function formatPercent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatChangeMetricValue(metricId, value) {
  if (value == null || Number.isNaN(Number(value))) return "n/a";
  if (metricId === "return" || metricId === "log_return") {
    return `${(Number(value) * 100).toFixed(2)}%`;
  }
  return formatNumber(value);
}

function formatTimestamp(value) {
  return String(value).replace("T", " ").replace("Z", "Z");
}

function normalizeDateInput(value) {
  if (!value) return "";
  return String(value).slice(0, 10);
}

function parseDateInput(value) {
  const normalized = normalizeDateInput(value);
  if (!normalized) return null;
  const [year, month, day] = normalized.split("-").map(Number);
  if (!year || !month || !day) return null;
  return new Date(Date.UTC(year, month - 1, day));
}

function formatDateInput(value) {
  const year = String(value.getUTCFullYear());
  const month = String(value.getUTCMonth() + 1).padStart(2, "0");
  const day = String(value.getUTCDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function shiftUtcYears(value, offsetYears) {
  const shifted = new Date(
    Date.UTC(
      value.getUTCFullYear() + offsetYears,
      value.getUTCMonth(),
      value.getUTCDate(),
    ),
  );
  if (shifted.getUTCMonth() === value.getUTCMonth()) {
    return shifted;
  }
  return new Date(
    Date.UTC(value.getUTCFullYear() + offsetYears, value.getUTCMonth() + 1, 0),
  );
}

function todayUtc() {
  const now = new Date();
  return new Date(Date.UTC(now.getFullYear(), now.getMonth(), now.getDate()));
}

function toneForEvidence(strength) {
  if (strength === "thin") return "warn";
  if (strength === "limited") return "";
  if (strength === "substantial") return "ok";
  return "";
}

function emptyToNull(value) {
  const text = String(value || "").trim();
  return text ? text : null;
}

async function copyText(value) {
  if (navigator?.clipboard?.writeText) {
    await navigator.clipboard.writeText(value);
    return;
  }
  const helper = document.createElement("textarea");
  helper.value = value;
  helper.setAttribute("readonly", "readonly");
  helper.style.position = "absolute";
  helper.style.left = "-9999px";
  document.body.append(helper);
  helper.select();
  document.execCommand("copy");
  helper.remove();
}

function ownsAction(actionId) {
  return state.pendingActionId === actionId;
}

async function runOwnedAction(kind, runner) {
  const actionId = ++state.actionSequence;
  state.pendingActionId = actionId;
  state.pendingActionKind = kind;
  state.isBusy = true;
  syncBusyState();
  try {
    return await runner(actionId);
  } catch (error) {
    if (!ownsAction(actionId)) return null;
    throw error;
  } finally {
    if (ownsAction(actionId)) {
      state.pendingActionId = null;
      state.pendingActionKind = null;
      state.isBusy = false;
      syncBusyState();
    }
  }
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload?.error?.message || `Request failed with ${response.status}`);
  }
  return payload;
}

function setStatus(message) {
  statusLine.textContent = message;
}

function syncBusyState() {
  if (primaryActionButton) {
    primaryActionButton.disabled = state.isBusy;
    primaryActionButton.textContent =
      state.pendingActionKind === "analyze"
        ? "Running Analysis…"
        : state.pendingActionKind === "load-analysis"
        ? "Loading Analysis…"
        : defaultPrimaryActionLabel;
  }
  analysisForm.setAttribute("aria-busy", String(state.isBusy));
  recentList.setAttribute("aria-busy", String(state.isBusy));
  recentList.querySelectorAll("button[data-analysis-path]").forEach((button) => {
    button.disabled = state.isBusy;
  });
}

function humanizeKey(value) {
  return String(value || "")
    .replaceAll("_", " ")
    .replaceAll("-", " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function truncateMiddle(value, limit = 64) {
  const text = String(value || "");
  if (text.length <= limit) return text;
  const head = Math.ceil((limit - 3) / 2);
  const tail = Math.floor((limit - 3) / 2);
  return `${text.slice(0, head)}...${text.slice(-tail)}`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

export const __test__ = {
  state,
  adoptAnalysis,
  render,
};
