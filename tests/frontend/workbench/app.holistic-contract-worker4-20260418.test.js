import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { afterEach, describe, expect, test, vi } from "vitest";

const TEST_DIR = dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = resolve(TEST_DIR, "../../..");
const ASSET_DIR = resolve(ROOT_DIR, "src/euclid/_assets/workbench");
const FIXTURE_DIR = resolve(TEST_DIR, "fixtures");
const APP_MODULE_URL = pathToFileURL(resolve(ASSET_DIR, "app.js")).href;
const originalFetch = globalThis.fetch;
const DESCRIPTIVE_SELECTION_SCOPE = "shared_planning_cir_only";
const DESCRIPTIVE_SELECTION_RULE =
  "min_total_code_bits_then_max_description_gain_then_min_structure_code_bits_then_min_canonical_byte_length_then_candidate_id";

const [shellHtml, shellCss, spyFixture, gldFixture] = await Promise.all([
  readFile(resolve(ASSET_DIR, "index.html"), "utf8"),
  readFile(resolve(ASSET_DIR, "app.css"), "utf8"),
  readJsonFixture(
    "analysis-holistic-contract-worker4-spy-daily-return-20260418.json",
  ),
  readJsonFixture(
    "analysis-holistic-contract-worker4-gld-price-close-20260418.json",
  ),
]);

afterEach(() => {
  vi.restoreAllMocks();
  if (originalFetch) {
    globalThis.fetch = originalFetch;
  } else {
    delete globalThis.fetch;
  }
  resetDocument();
});

describe("workbench holistic contract regressions worker4 2026-04-18", () => {
  test("keeps the SPY exact-closure abstention fixture out of the hero slot", async () => {
    const analysis = clone(spyFixture);

    await mountSavedAnalysis(analysis);

    await waitFor(() => {
      expect(textContent("#tab-overview")).toContain("No benchmark-local winner");
    });

    expect(
      document.querySelector('#tab-overview [data-equation-hero="overview"]'),
    ).toBeNull();
    expect(textContent("#tab-overview")).toMatch(
      /operator (?:lane abstained|abstention)/i,
    );
    expect(textContent("#tab-overview")).not.toContain("sample exact closure");
    expect(textContent("#tab-overview")).not.toContain("Holistic equation");
    expect(textContent("#tab-overview")).not.toContain("Holistic law");
    expect(textContent("#tab-overview")).not.toMatch(/\blaw\b/i);
  });

  test("does not let stale synthetic holistic composition metadata outrank the normalized non-law taxonomy", async () => {
    const analysis = clone(spyFixture);
    analysis.claim_class = "holistic_equation";
    analysis.publishable = true;
    analysis.gap_report = [];
    analysis.not_holistic_because = [];
    analysis.would_have_abstained_because = [];
    analysis.operator_point = {
      ...analysis.operator_point,
      result_mode: "candidate_publication",
      publication: {
        status: "publishable",
        headline:
          "Saved analysis still carried a completed holistic payload with authoritative refs.",
        reason_codes: [],
      },
    };
    delete analysis.operator_point.abstention;
    analysis.holistic_equation = {
      status: "completed",
      claim_class: "holistic_equation",
      exactness: "joint_validation_scope_claim",
      deterministic_source: "operator_point",
      probabilistic_source: "distribution",
      validation_scope_ref: "validation_scope:spy_joint_claim",
      publication_record_ref: "publication_record:spy_joint_claim",
      mode: "composition_stochastic",
      composition_operator: "additive_residual",
      selected_probabilistic_lane: "distribution",
      equation: {
        label: "y(t) = compact_drift + epsilon_bridge",
        curve: analysis.operator_point.chart.equation_curve,
      },
      honesty_note:
        "Stale saved analysis carried a fully populated holistic payload that must stay diagnostic-only when legacy synthetic composition metadata survives.",
    };

    await mountSavedAnalysis(analysis);

    await waitFor(() => {
      expect(textContent("#tab-overview")).toContain("No benchmark-local winner");
    });

    expect(
      document.querySelector('#tab-overview [data-equation-hero="overview"]'),
    ).toBeNull();
    expect(textContent("#tab-overview")).not.toContain("epsilon_bridge");
    expect(textContent("#tab-overview")).not.toContain("Holistic equation");
    expect(textContent("#tab-overview")).not.toContain("Holistic law");
    expect(textContent("#tab-overview")).not.toMatch(/\blaw\b/i);
  });

  test("keeps the GLD abstention fixture descriptive instead of promoting a stochastic holistic claim", async () => {
    const analysis = clone(gldFixture);

    await mountSavedAnalysis(analysis);

    await waitFor(() => {
      expect(textContent("#tab-overview")).toContain(
        "Best available descriptive approximation",
      );
    });

    const overviewHero = document.querySelector(
      '#tab-overview [data-equation-hero="overview"]',
    );

    expect(overviewHero).not.toBeNull();
    expect(textContent("#hero")).toContain(analysis.descriptive_fit.honesty_note);
    expect(textContent("#hero")).not.toContain("operator abstained");
    expect(overviewHero.textContent || "").toContain("-0.160657");
    expect(overviewHero.textContent || "").toContain("1.00169");
    expect(overviewHero.textContent || "").not.toContain("166.48");
    expect(overviewHero.textContent || "").not.toContain("0.218356");
    expect(textContent("#tab-overview")).toContain(
      "Raw close descriptive fit is effectively a persistence equation with a tiny intercept.",
    );
    expect(textContent("#tab-overview")).not.toContain("Holistic equation");
    expect(textContent("#tab-overview")).not.toMatch(/\blaw\b/i);
    expect(overviewHero.textContent || "").not.toMatch(
      /(?:\\varepsilon|varepsilon|ε|stochastic)/i,
    );
  });

  test("does not leave the descriptive lane empty when a legacy fallback is available", async () => {
    const analysis = clone(gldFixture);
    delete analysis.holistic_equation;
    analysis.descriptive_fit = {
      status: "completed",
      source: "legacy_operator_point_fallback",
      submitter_id: "legacy_operator_point_fallback",
      submitter_class: "legacy_compatibility_projection",
      candidate_id: null,
      family_id: analysis.operator_point.selected_family,
      metrics: {},
      honesty_note:
        "Legacy saved analysis predates descriptive-bank payloads. Projecting the operator point equation as a descriptive-only fallback for display; not eligible for top-line publication.",
      semantic_audit: {
        classification: "level_fit",
        headline:
          "Raw close descriptive fit should be read as a level model before it is read as market structure.",
        summary:
          "Level targets often preserve persistence and drift. Inspect the naive last-value baseline and the delta form before inferring semantics.",
        delta_form_label: null,
        fit_mae: 0.5089040000000011,
        naive_last_value_mae: 1.110000000000004,
        relative_improvement_vs_naive_last_value: 0.5415279279279286,
        recommended_target_id: "daily_return",
        recommended_target_label: "Daily Return",
        recommended_target_reason:
          "Better default for interpretable descriptive equations because it makes level persistence explicit instead of hiding it inside raw prices.",
      },
      equation: {
        ...analysis.operator_point.equation,
        family_id: analysis.operator_point.selected_family,
      },
      chart: {
        ...analysis.operator_point.chart,
      },
      selection_scope: DESCRIPTIVE_SELECTION_SCOPE,
      selection_rule: DESCRIPTIVE_SELECTION_RULE,
      law_eligible: false,
      law_rejection_reason_codes: ["legacy_compatibility_projection"],
      claim_class: "descriptive_fit",
      is_law_claim: false,
    };
    analysis.claim_class = "descriptive_fit";
    analysis.holistic_equation = null;

    await mountSavedAnalysis(analysis);

    await waitFor(() => {
      expect(textContent("#tab-overview")).toContain(
        "Best available descriptive approximation",
      );
    });

    expect(textContent("#tab-overview")).not.toContain(
      "No benchmark-local winner",
    );
    expect(textContent("#tab-overview")).toContain(
      analysis.operator_point.equation.label,
    );
    expect(textContent("#tab-overview")).not.toMatch(/\blaw\b/i);
    expect(textContent("#tab-overview")).toContain(
      analysis.descriptive_fit.honesty_note,
    );
  });
});

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function buildConfig({ recentAnalyses = [] } = {}) {
  return {
    default_target_id: "daily_return",
    target_specs: [
      {
        id: "daily_return",
        label: "Daily Return",
        description: "Predict the close-to-close fractional return for each day.",
        analysis_note: "Recommended for analytical equation inspection.",
        recommended: true,
      },
      {
        id: "price_close",
        label: "Price Close",
        description: "Predict the raw close level for each trading day.",
        analysis_note:
          "Useful for descriptive-fit inspection but more persistence-like.",
        recommended: false,
      },
    ],
    has_api_key_env: false,
    api_key_env_var: "FMP_API_KEY",
    default_date_range: {
      start_date: "2025-01-08",
      end_date: "2025-02-14",
    },
    recent_analyses: recentAnalyses,
  };
}

function buildRecentEntry(analysis) {
  return {
    symbol: analysis.dataset.symbol,
    target_id: analysis.dataset.target.id,
    workspace_root: analysis.workspace_root,
    analysis_path: analysis.analysis_path,
  };
}

async function mountSavedAnalysis(analysis) {
  await mountWorkbench({
    route(url) {
      if (url.pathname === "/api/config") {
        return jsonResponse(
          buildConfig({
            recentAnalyses: [buildRecentEntry(analysis)],
          }),
        );
      }
      if (url.pathname === "/api/analysis") {
        return jsonResponse(analysis);
      }
      throw new Error(`Unhandled request: ${url.pathname}`);
    },
  });

  document
    .querySelector('button[data-analysis-path]')
    .dispatchEvent(new MouseEvent("click", { bubbles: true }));
}

function jsonResponse(payload, { ok = true, status = 200 } = {}) {
  return Promise.resolve({
    ok,
    status,
    json: async () => payload,
  });
}

async function mountWorkbench({ route }) {
  document.open();
  document.write(shellHtml);
  document.close();

  const styleTag = document.createElement("style");
  styleTag.textContent = shellCss;
  document.head.append(styleTag);

  const fetchMock = vi.fn((input, init) => {
    const rawUrl = typeof input === "string" ? input : input.url;
    return route(new URL(rawUrl, "http://localhost"), init);
  });
  vi.stubGlobal("fetch", fetchMock);

  await import(
    `${APP_MODULE_URL}?case=${Date.now()}-${Math.random().toString(16).slice(2)}`
  );

  await waitFor(() => {
    expect(document.querySelector("#target-select").options.length).toBeGreaterThan(
      0,
    );
  });

  return { fetchMock };
}

async function readJsonFixture(name) {
  const raw = await readFile(resolve(FIXTURE_DIR, name), "utf8");
  return JSON.parse(raw);
}

function resetDocument() {
  if (typeof window !== "undefined" && window.history?.replaceState) {
    window.history.replaceState({}, "", "/");
  }
  document.open();
  document.write("<!doctype html><html><head></head><body></body></html>");
  document.close();
}

function textContent(selector) {
  return (
    document.querySelector(selector)?.textContent?.replace(/\s+/g, " ").trim() || ""
  );
}

async function waitFor(assertion, { timeoutMs = 1000 } = {}) {
  const deadline = Date.now() + timeoutMs;
  let lastError = null;

  while (Date.now() < deadline) {
    try {
      return assertion();
    } catch (error) {
      lastError = error;
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }

  throw lastError || new Error("Timed out waiting for assertion.");
}
