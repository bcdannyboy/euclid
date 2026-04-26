import { beforeEach, describe, expect, it, vi } from "vitest";
import { readFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";

const INDEX_PATH =
  "/Users/danielbloom/Desktop/euclid/src/euclid/_assets/workbench/index.html";
const APP_PATH =
  "/Users/danielbloom/Desktop/euclid/src/euclid/_assets/workbench/app.js";
const ANALYSIS_PATH =
  "/Users/danielbloom/Desktop/euclid/tests/frontend/workbench/fixtures/analysis-saved.json";

const CONFIG_FIXTURE = {
  default_target_id: "price_close",
  api_key_env_var: "FMP_API_KEY",
  has_api_key_env: true,
  recent_analyses: [],
  target_specs: [
    {
      id: "price_close",
      label: "Price Close",
      recommended: true,
      description: "Observed close level.",
    },
  ],
};

async function setupWorkbench() {
  const [html, analysisText] = await Promise.all([
    readFile(INDEX_PATH, "utf8"),
    readFile(ANALYSIS_PATH, "utf8"),
  ]);
  document.documentElement.innerHTML = html;
  const analysis = JSON.parse(analysisText);
  globalThis.fetch = vi.fn(async () => ({
    ok: true,
    json: async () => CONFIG_FIXTURE,
    text: async () => JSON.stringify(CONFIG_FIXTURE),
  }));
  globalThis.echarts = undefined;
  const moduleUrl = `${pathToFileURL(APP_PATH).href}?t=${Date.now()}-${Math.random()}`;
  const module = await import(moduleUrl);
  await Promise.resolve();
  return { analysis, module };
}

describe("workbench workspace redesign", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    document.documentElement.innerHTML = "<html><head></head><body></body></html>";
  });

  it("renders a canonical analysis header and workspace navigation for a loaded run", async () => {
    const { analysis, module } = await setupWorkbench();

    module.__test__.adoptAnalysis(analysis);
    module.__test__.render();

    expect(document.querySelector("[data-analysis-header]")).toBeTruthy();
    expect(document.querySelector("[data-run-summary]")).toBeTruthy();
    expect(
      Array.from(document.querySelectorAll(".tab-button"), (button) =>
        button.textContent.trim(),
      ),
    ).toEqual([
      "Overview",
      "Evidence",
      "Forecast",
      "Calibration",
      "Search",
      "Artifacts",
    ]);
    expect(document.body.textContent).toContain(analysis.dataset.symbol);
    expect(document.body.textContent).toContain(analysis.dataset.target.label);
  });

  it("renders a Euclid first-screen evidence spine before deeper views", async () => {
    const { analysis, module } = await setupWorkbench();

    module.__test__.adoptAnalysis(analysis);
    module.__test__.render();

    const firstScreen = document.querySelector('[data-euclid-first-screen="spine"]');
    expect(firstScreen).toBeTruthy();
    expect(firstScreen.textContent).toContain("Ordered observations");
    expect(firstScreen.textContent).toContain("Candidate equation");
    expect(firstScreen.textContent).toContain("Residual evidence");
    expect(firstScreen.textContent).toContain("Stochastic support");
    expect(firstScreen.textContent).toContain("Publication gate");
    expect(firstScreen.textContent).toContain("Claim ceiling");
    expect(firstScreen.textContent).toContain("Replay");
    expect(firstScreen.textContent).toContain("Calibration");
    expect(firstScreen.textContent).toContain(analysis.dataset.symbol);
  });

  it("renders a linked evidence workspace, forecast explanation, and decision-first evidence regions", async () => {
    const { analysis, module } = await setupWorkbench();

    module.__test__.adoptAnalysis(analysis);
    module.__test__.render();

    expect(document.querySelector('[data-workspace-region="equation-stage"]')).toBeTruthy();
    expect(document.querySelector('[data-workspace-region="analysis-canvas"]')).toBeTruthy();
    expect(document.querySelector('[data-workspace-region="evidence-rail"]')).toBeTruthy();
    expect(document.querySelector('[data-point-story="lag-explanation"]')).toBeTruthy();
    expect(document.querySelector('[data-benchmark-region="outcome-strip"]')).toBeTruthy();
    expect(document.querySelector('[data-artifact-region="role-summary"]')).toBeTruthy();
  });

  it("renders the evidence studio claim, replay, provenance, and live boundary", async () => {
    const { analysis, module } = await setupWorkbench();
    analysis.evidence_studio = {
      claim_surface: {
        claim_lane: "descriptive",
        claim_ceiling: "descriptive_structure",
        publication_status: "abstained",
        publishable: false,
        abstention_reason_codes: ["robustness_failed"],
        downgrade_reason_codes: ["operator_not_publishable"],
        live_evidence_boundary: {
          counts_as_scientific_claim_evidence: false,
        },
      },
      live_evidence: {
        status: "passed",
        claim_boundary: {
          counts_as_scientific_claim_evidence: false,
        },
      },
      replay_artifacts: {
        links: [
          {
            section: "operator_point",
            role: "manifest_path",
            value: "/tmp/workbench-run/operator-point.yaml",
          },
        ],
      },
      engine_provenance: {
        point_lane: {
          engine_id: "bounded_symbolic_search",
        },
      },
      diagnostics: {},
    };

    module.__test__.adoptAnalysis(analysis);
    module.__test__.render();

    const evidenceText = document.querySelector(
      '[data-workspace-region="evidence-rail"]',
    ).textContent;
    expect(evidenceText).toContain("Evidence Studio");
    expect(evidenceText).toContain("descriptive");
    expect(evidenceText).toContain("non-claim live evidence");
    expect(evidenceText).toContain("bounded_symbolic_search");
    expect(evidenceText).toContain("Replay links");
  });
});
