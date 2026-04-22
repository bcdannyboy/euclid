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

const [shellHtml, shellCss, savedAnalysisFixture, noWinnerFixture] =
  await Promise.all([
    readFile(resolve(ASSET_DIR, "index.html"), "utf8"),
    readFile(resolve(ASSET_DIR, "app.css"), "utf8"),
    readJsonFixture("analysis-saved.json"),
    readJsonFixture("analysis-no-winner.json"),
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

describe("workbench packaged asset harness", () => {
  test("renders non-blank empty states on first paint after config load", async () => {
    await mountWorkbench({
      route(url) {
        if (url.pathname === "/api/config") {
          return jsonResponse(buildConfig());
        }
        throw new Error(`Unhandled request: ${url.pathname}`);
      },
    });

    expect(textContent("#hero")).toContain("No analysis loaded");
    expect(textContent("#tab-overview")).toContain(
      "Overview appears after the first analysis run.",
    );
    expect(textContent("#tab-atlas")).toContain(
      "Atlas appears after the first analysis run.",
    );
    expect(textContent("#tab-point")).toContain(
      "Point lane results appear here.",
    );
    expect(textContent("#tab-probabilistic")).toContain(
      "Probabilistic lanes appear here when enabled.",
    );
    expect(textContent("#tab-benchmark")).toContain(
      "Benchmark comparison appears here when enabled.",
    );
    expect(textContent("#tab-artifacts")).toContain(
      "Artifact paths appear after an analysis run.",
    );
  });

  test("auto-opens Atlas on first analysis load and preserves manual tab selection on reload", async () => {
    const analysis = clone(savedAnalysisFixture);

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

    await waitFor(() => {
      expect(
        document
          .querySelector('.tab-button[data-tab="atlas"]')
          ?.classList.contains("active"),
      ).toBe(true);
    });

    expect(textContent("#tab-atlas")).toContain("Equation Atlas");

    document
      .querySelector('.tab-button[data-tab="point"]')
      .dispatchEvent(new MouseEvent("click", { bubbles: true }));

    await waitFor(() => {
      expect(
        document
          .querySelector('.tab-button[data-tab="point"]')
          ?.classList.contains("active"),
      ).toBe(true);
    });

    document
      .querySelector('button[data-analysis-path]')
      .dispatchEvent(new MouseEvent("click", { bubbles: true }));

    await waitFor(() => {
      expect(
        document
          .querySelector('.tab-button[data-tab="point"]')
          ?.classList.contains("active"),
      ).toBe(true);
    });

    expect(
      document
        .querySelector('.tab-button[data-tab="atlas"]')
        ?.classList.contains("active"),
    ).toBe(false);
  });

  test("shares horizon and deterministic overlay state across Atlas and lane tabs", async () => {
    const analysis = buildAtlasFixture();

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

    await waitFor(() => {
      expect(textContent("#tab-atlas")).toContain("Equation Atlas");
    });

    document
      .querySelector('#tab-atlas [data-horizon="2"]')
      .dispatchEvent(new MouseEvent("click", { bubbles: true }));

    await waitFor(() => {
      expect(textContent("#tab-probabilistic")).toContain("Active horizon h2");
    });

    expect(textContent("#tab-atlas")).toContain("505.3");
    expect(textContent("#tab-probabilistic")).toContain("505.3");

    document
      .querySelector('#tab-atlas [data-overlay="point"]')
      .dispatchEvent(new MouseEvent("click", { bubbles: true }));

    await waitFor(() => {
      expect(textContent("#tab-point")).toContain(
        "Residual basis operator point path",
      );
    });

    expect(textContent("#tab-atlas")).toContain(
      "Replay verified a candidate, but publication gates failed.",
    );
  });

  test("guards the live-run and saved-analysis flow against stale action races", async () => {
    const savedAnalysis = clone(savedAnalysisFixture);
    const liveAnalysis = buildLiveAnalyzeFixture(savedAnalysis);
    const analyzeResponse = deferred();

    await mountWorkbench({
      route(url) {
        if (url.pathname === "/api/config") {
          return jsonResponse(
            buildConfig({
              recentAnalyses: [buildRecentEntry(savedAnalysis)],
            }),
          );
        }
        if (url.pathname === "/api/analyze") {
          return analyzeResponse.promise;
        }
        if (url.pathname === "/api/analysis") {
          expect(url.searchParams.get("analysis_path")).toBe(
            savedAnalysis.analysis_path,
          );
          return jsonResponse(savedAnalysis);
        }
        throw new Error(`Unhandled request: ${url.pathname}`);
      },
    });

    document.querySelector('input[name="symbol"]').value =
      liveAnalysis.dataset.symbol;
    document
      .querySelector("#analysis-form")
      .dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));

    await waitFor(() => {
      expect(textContent("#status-line")).toContain("Running live Euclid analysis");
    });

    const loadButton = document.querySelector('button[data-analysis-path]');
    expect(loadButton).not.toBeNull();

    if (loadButton.disabled) {
      analyzeResponse.resolve(jsonResponse(liveAnalysis));

      await waitFor(() => {
        expect(textContent("#hero")).toContain(liveAnalysis.dataset.symbol);
      });

      document
        .querySelector('button[data-analysis-path]')
        .dispatchEvent(new MouseEvent("click", { bubbles: true }));

      await waitFor(() => {
        expect(textContent("#hero")).toContain(savedAnalysis.dataset.symbol);
        expect(textContent("#tab-artifacts")).toContain(savedAnalysis.analysis_path);
      });
    } else {
      loadButton.dispatchEvent(new MouseEvent("click", { bubbles: true }));

      await waitFor(() => {
        expect(textContent("#hero")).toContain(savedAnalysis.dataset.symbol);
        expect(textContent("#tab-artifacts")).toContain(savedAnalysis.analysis_path);
      });

      analyzeResponse.resolve(jsonResponse(liveAnalysis));
      await settle();
    }

    expect(textContent("#hero")).toContain(savedAnalysis.dataset.symbol);
    expect(textContent("#hero")).not.toContain(liveAnalysis.dataset.symbol);
    expect(textContent("#tab-artifacts")).toContain(savedAnalysis.analysis_path);
    expect(textContent("#tab-artifacts")).not.toContain(liveAnalysis.analysis_path);
  });

  test("clears the current analysis after a rejected analyze request following a successful load", async () => {
    const savedAnalysis = clone(savedAnalysisFixture);
    const analyzeErrorMessage = "Live analyze failed after the saved run was already on screen.";

    await mountWorkbench({
      route(url) {
        if (url.pathname === "/api/config") {
          return jsonResponse(
            buildConfig({
              recentAnalyses: [buildRecentEntry(savedAnalysis)],
            }),
          );
        }
        if (url.pathname === "/api/analysis") {
          return jsonResponse(savedAnalysis);
        }
        if (url.pathname === "/api/analyze") {
          return jsonResponse(
            { error: { message: analyzeErrorMessage } },
            { ok: false, status: 500 },
          );
        }
        throw new Error(`Unhandled request: ${url.pathname}`);
      },
    });

    document
      .querySelector('button[data-analysis-path]')
      .dispatchEvent(new MouseEvent("click", { bubbles: true }));

    await waitFor(() => {
      expect(textContent("#tab-artifacts")).toContain(savedAnalysis.analysis_path);
    });

    expect(
      document
        .querySelector('.tab-button[data-tab="atlas"]')
        ?.classList.contains("active"),
    ).toBe(true);

    document.querySelector('input[name="symbol"]').value = "QQQ";
    document
      .querySelector("#analysis-form")
      .dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));

    await waitFor(() => {
      expect(textContent("#status-line")).toContain(analyzeErrorMessage);
    });

    expect(textContent("#hero")).toContain("No analysis loaded");
    expect(textContent("#hero")).not.toContain(savedAnalysis.dataset.symbol);
    expect(
      document
        .querySelector('.tab-button[data-tab="overview"]')
        ?.classList.contains("active"),
    ).toBe(true);
    expect(document.querySelector("#tab-overview")?.hasAttribute("hidden")).toBe(false);
    expect(textContent("#tab-overview")).toContain("Analysis failed");
    expect(textContent("#tab-overview")).toContain(analyzeErrorMessage);
    expect(textContent("#tab-artifacts")).not.toContain(savedAnalysis.analysis_path);
  });

  test("clears the current analysis after a rejected saved-analysis load following a successful load", async () => {
    const savedAnalysis = clone(savedAnalysisFixture);
    const rejectedAnalysis = clone(savedAnalysisFixture);
    const loadErrorMessage = "Saved analysis reload failed after the prior run had already loaded.";

    rejectedAnalysis.analysis_path =
      "/tmp/euclid/saved/workspaces/2026-04-18T171500Z-rejected-load/analysis.json";
    rejectedAnalysis.workspace_root =
      "/tmp/euclid/saved/workspaces/2026-04-18T171500Z-rejected-load";
    rejectedAnalysis.dataset.symbol = "QQQ";

    await mountWorkbench({
      route(url) {
        if (url.pathname === "/api/config") {
          return jsonResponse(
            buildConfig({
              recentAnalyses: [
                buildRecentEntry(savedAnalysis),
                buildRecentEntry(rejectedAnalysis),
              ],
            }),
          );
        }
        if (url.pathname === "/api/analysis") {
          if (url.searchParams.get("analysis_path") === rejectedAnalysis.analysis_path) {
            return jsonResponse(
              { error: { message: loadErrorMessage } },
              { ok: false, status: 500 },
            );
          }
          return jsonResponse(savedAnalysis);
        }
        throw new Error(`Unhandled request: ${url.pathname}`);
      },
    });

    document
      .querySelector('button[data-analysis-path]')
      .dispatchEvent(new MouseEvent("click", { bubbles: true }));

    await waitFor(() => {
      expect(textContent("#tab-artifacts")).toContain(savedAnalysis.analysis_path);
    });

    expect(
      document
        .querySelector('.tab-button[data-tab="atlas"]')
        ?.classList.contains("active"),
    ).toBe(true);

    document
      .querySelectorAll('button[data-analysis-path]')[1]
      .dispatchEvent(new MouseEvent("click", { bubbles: true }));

    await waitFor(() => {
      expect(textContent("#status-line")).toContain(loadErrorMessage);
    });

    expect(textContent("#hero")).toContain("No analysis loaded");
    expect(textContent("#hero")).not.toContain(savedAnalysis.dataset.symbol);
    expect(textContent("#hero")).not.toContain(rejectedAnalysis.dataset.symbol);
    expect(
      document
        .querySelector('.tab-button[data-tab="overview"]')
        ?.classList.contains("active"),
    ).toBe(true);
    expect(document.querySelector("#tab-overview")?.hasAttribute("hidden")).toBe(false);
    expect(textContent("#tab-overview")).toContain("Failed to load saved analysis");
    expect(textContent("#tab-overview")).toContain(loadErrorMessage);
    expect(textContent("#tab-artifacts")).not.toContain(savedAnalysis.analysis_path);
    expect(textContent("#tab-artifacts")).not.toContain(rejectedAnalysis.analysis_path);
  });

  test("renders completed benchmark no-winner states honestly", async () => {
    const analysis = clone(noWinnerFixture);

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

    await waitFor(() => {
      expect(textContent("#tab-benchmark")).toContain("Portfolio selection");
    });

    document
      .querySelector('.tab-button[data-tab="overview"]')
      .dispatchEvent(new MouseEvent("click", { bubbles: true }));

    await waitFor(() => {
      expect(
        document
          .querySelector('.tab-button[data-tab="overview"]')
          ?.classList.contains("active"),
      ).toBe(true);
    });

    expect(textContent("#tab-overview")).toContain("No benchmark-local winner");

    document
      .querySelector('.tab-button[data-tab="benchmark"]')
      .dispatchEvent(new MouseEvent("click", { bubbles: true }));

    await waitFor(() => {
      expect(
        document
          .querySelector('.tab-button[data-tab="benchmark"]')
          ?.classList.contains("active"),
      ).toBe(true);
    });

    const benchmarkText = textContent("#tab-benchmark");
    expect(benchmarkText).toContain("No benchmark-local winner");
    expect(benchmarkText).toContain("No admissible finalist was selected");
    expect(benchmarkText).toContain("n/a");
    expect(benchmarkText).not.toContain("Winner unknown");
    expect(benchmarkText).not.toContain("Selected unknown / unknown");
  });

  test("keeps long recent-analysis previews and artifact paths layout-safe", async () => {
    const analysis = clone(savedAnalysisFixture);

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

    const recentPreview = document.querySelector("#recent-list .recent-item .mono");
    expect(recentPreview.textContent).toContain("...");
    expect(recentPreview.textContent).not.toBe(analysis.workspace_root);
    expect(recentPreview.getAttribute("title")).toBe(analysis.workspace_root);

    document
      .querySelector('button[data-analysis-path]')
      .dispatchEvent(new MouseEvent("click", { bubbles: true }));

    await waitFor(() => {
      expect(
        document.querySelectorAll("#tab-artifacts .artifact-path").length,
      ).toBeGreaterThan(0);
    });

    for (const node of document.querySelectorAll("#tab-artifacts .artifact-path")) {
      const styles = getComputedStyle(node);
      const wrapsAnywhere =
        styles.getPropertyValue("overflow-wrap") === "anywhere";
      const breaksWords =
        styles.getPropertyValue("word-break") === "break-word";
      expect(wrapsAnywhere || breaksWords).toBe(true);
    }
  });

  test("renders richer explainer narrative and cautions", async () => {
    const analysis = clone(savedAnalysisFixture);
    analysis.llm_explanations = {
      status: "completed",
      model: "gpt-5.4",
      pages: {
        overview: {
          headline: "Holistic overview",
          summary: "Short summary first.",
          narrative:
            "Longer narrative tying the observed path, abstention, benchmark-local fit, and thin probabilistic evidence together.",
          key_takeaways: [
            "Observed series rose materially.",
            "Descriptive fit is near persistence.",
          ],
          cautions: [
            "Benchmark-local fit is not an operator publication.",
          ],
          terms: [
            {
              term: "benchmark-local",
              meaning: "Chosen only inside the benchmark comparison lane.",
            },
          ],
        },
      },
    };

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

    await waitFor(() => {
      expect(textContent("#tab-overview")).toContain("Holistic overview");
    });

    const overviewText = textContent("#tab-overview");
    expect(overviewText).toContain(
      "Longer narrative tying the observed path",
    );
    expect(overviewText).toContain(
      "Read this run from the operator publication outward",
    );
    expect(overviewText).toContain("Observed series rose materially.");
    expect(overviewText).toContain(
      "Benchmark-local fit is not an operator publication.",
    );
  });

  test("falls back to explanatory guides when a completed llm bundle omits a page", async () => {
    const analysis = clone(savedAnalysisFixture);
    analysis.llm_explanations = {
      status: "completed",
      model: "gpt-5.4",
      pages: {
        overview: {
          headline: "Holistic overview",
          summary: "Short summary first.",
        },
      },
    };

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

    document
      .querySelector('.tab-button[data-tab="point"]')
      .dispatchEvent(new MouseEvent("click", { bubbles: true }));

    await waitFor(() => {
      expect(textContent("#tab-point")).toContain("Interpretation guide");
    });

    expect(textContent("#tab-point")).toContain(
      "publishable deterministic claim",
    );
  });

  test("auto-loads and synchronizes shareable URL state for analysis, tab, and selectors", async () => {
    const analysis = buildAtlasFixture();
    const analysisPath = analysis.analysis_path;

    window.history.replaceState(
      {},
      "",
      `/?analysis_path=${encodeURIComponent(analysisPath)}&tab=probabilistic&horizon=2&overlay=point&lane=quantile&rail=compact`,
    );

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
          expect(url.searchParams.get("analysis_path")).toBe(analysisPath);
          return jsonResponse(analysis);
        }
        throw new Error(`Unhandled request: ${url.pathname}`);
      },
    });

    await waitFor(() => {
      expect(textContent("#hero")).toContain(analysis.dataset.symbol);
    });

    expect(
      document
        .querySelector('.tab-button[data-tab="probabilistic"]')
        ?.classList.contains("active"),
    ).toBe(true);
    expect(textContent("#tab-probabilistic")).toContain("Active horizon h2");
    expect(
      document.querySelector(".control-rail")?.classList.contains("is-collapsed"),
    ).toBe(true);

    const initialUrl = new URL(window.location.href);
    expect(initialUrl.searchParams.get("analysis_path")).toBe(analysisPath);
    expect(initialUrl.searchParams.get("tab")).toBe("probabilistic");
    expect(initialUrl.searchParams.get("horizon")).toBe("2");
    expect(initialUrl.searchParams.get("overlay")).toBe("point");
    expect(initialUrl.searchParams.get("lane")).toBe("quantile");
    expect(initialUrl.searchParams.get("rail")).toBe("compact");

    document
      .querySelector('.tab-button[data-tab="atlas"]')
      .dispatchEvent(new MouseEvent("click", { bubbles: true }));
    document
      .querySelector('#tab-atlas [data-horizon="1"]')
      .dispatchEvent(new MouseEvent("click", { bubbles: true }));

    await waitFor(() => {
      const nextUrl = new URL(window.location.href);
      expect(nextUrl.searchParams.get("tab")).toBe("atlas");
      expect(nextUrl.searchParams.get("horizon")).toBe("1");
    });
  });

  test("renders analyst briefing shell and accessible tab semantics after load", async () => {
    const analysis = clone(savedAnalysisFixture);

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

    await waitFor(() => {
      expect(textContent("#tab-atlas")).toContain("Equation Atlas");
    });

    const tablist = document.querySelector('.tab-strip[role="tablist"]');
    const activeTab = document.querySelector('.tab-button[data-tab="atlas"]');
    const activePanel = document.querySelector("#tab-atlas");

    expect(tablist).not.toBeNull();
    expect(activeTab?.getAttribute("role")).toBe("tab");
    expect(activeTab?.getAttribute("aria-selected")).toBe("true");
    expect(activeTab?.getAttribute("tabindex")).toBe("0");
    expect(activePanel?.getAttribute("role")).toBe("tabpanel");
    expect(activePanel?.hasAttribute("hidden")).toBe(false);
    expect(document.querySelector("#status-line")?.getAttribute("aria-live")).toBe(
      "polite",
    );
    expect(textContent("#analyst-briefing")).toContain("Evidence");
    expect(textContent("#analyst-briefing")).toContain("Method");
  });

  test("re-centers the loaded workbench around an analytical canvas and canonical equation rendering", async () => {
    const analysis = buildAtlasFixture();

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

    await waitFor(() => {
      expect(textContent("#tab-overview")).toContain("Analytical canvas");
    });

    expect(textContent("#tab-overview")).toContain("Interpretation guide");
    expect(
      document.querySelector('#tab-overview [data-chart-root="primary-canvas"]'),
    ).not.toBeNull();
    expect(
      document.querySelector('#tab-atlas [data-chart-root="uncertainty-ruler"]'),
    ).not.toBeNull();
    expect(
      document.querySelectorAll('[data-equation-renderer="katex"]').length,
    ).toBeGreaterThan(0);
  });

  test("keeps synthetic exact-closure payloads out of the atlas equation stack", async () => {
    const analysis = buildAtlasFixture();
    analysis.claim_class = "descriptive_fit";
    analysis.holistic_equation = {
      status: "completed",
      claim_class: "descriptive_fit",
      source: "descriptive_fit",
      exactness: "sample_exact_closure",
      honesty_note:
        "Holistic equation adds an exact residual-closure term over the observed sample.",
      equation: {
        label:
          String.raw`y(t) = \left(2.75 + 0.994\cdot y(t-1)\right) + \sum_{i=0}^{3} c_i \prod_{j \ne i}\frac{\tau(t)-j}{i-j},\quad \tau(t_i)=i,\quad c=\left[0, 0.439, 0.542, 1.161\right]`,
        curve: analysis.dataset.series.map((point) => ({
          event_time: point.event_time,
          fitted_value: point.observed_value,
        })),
      },
    };

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

    await waitFor(() => {
      expect(textContent("#tab-atlas")).toContain("Equation Atlas");
    });

    const atlasText = textContent("#tab-atlas");
    expect(atlasText).not.toContain("Holistic equation");
    expect(atlasText).not.toContain(
      "Holistic equation adds an exact residual-closure term over the observed sample.",
    );
    expect(atlasText).not.toContain("sample exact closure");
    expect(atlasText).toContain("Benchmark-local descriptive fit");
  });

  test("keeps synthetic holistic equations out of the overview hero output and explains the remaining gaps", async () => {
    const analysis = buildAtlasFixture();
    analysis.claim_class = "descriptive_fit";
    analysis.gap_report = [
      "operator_not_publishable",
      "no_backend_joint_claim",
      "probabilistic_evidence_thin",
    ];
    analysis.would_have_abstained_because = [
      "confirmatory_holdout_failed",
      "predictive_scope_not_confirmed",
    ];
    analysis.not_holistic_because = [
      "Joint claim requires an explicit backend-backed holistic publication path.",
    ];
    analysis.holistic_equation = {
      status: "completed",
      claim_class: "descriptive_fit",
      source: "descriptive_fit",
      exactness: "sample_exact_closure",
      composition_operator: "additive_residual",
      selected_probabilistic_lane: "distribution",
      honesty_note:
        "Holistic equation adds an exact residual-closure term over the observed sample.",
      equation: {
        label:
          String.raw`y(t) = \left(2.75 + 0.994\cdot y(t-1)\right) + \sum_{i=0}^{9} c_i \prod_{j \ne i}\frac{\tau(t)-j}{i-j}`,
        delta_form_label:
          String.raw`c_i = y_i - \hat{y}_i^{\mathrm{compact}}`,
      },
    };

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

    await waitFor(() => {
      expect(textContent("#tab-overview")).toContain("Best available descriptive approximation");
    });

    const heroCard = document.querySelector('#tab-overview [data-equation-hero="overview"]');
    expect(heroCard).not.toBeNull();
    const overviewText = textContent("#tab-overview");
    expect(overviewText).toContain(
      analysis.descriptive_fit.honesty_note,
    );
    expect(overviewText).toContain("Why not stronger?");
    expect(overviewText).toContain("Operator not publishable");
    expect(overviewText).toContain("No backend joint claim");
    expect(overviewText).toContain("Probabilistic evidence thin");
    expect(overviewText).not.toContain("Holistic equation adds an exact residual-closure term over the observed sample.");
    expect(overviewText).not.toContain("sample exact closure");
    expect(heroCard.querySelector('[data-equation-renderer="katex"]')).not.toBeNull();
  });

  test("allows long backend-backed holistic equations to scroll instead of clipping inside equation cards", async () => {
    const analysis = buildAtlasFixture();
    analysis.operator_point.publication = {
      status: "publishable",
      headline: "Operator point publication remains backend-backed for the holistic law path.",
    };
    analysis.claim_class = "holistic_equation";
    analysis.holistic_equation = {
      status: "completed",
      claim_class: "holistic_equation",
      deterministic_source: "predictive_law",
      probabilistic_source: "distribution",
      validation_scope_ref: "artifacts/validation-scope.json",
      publication_record_ref: "artifacts/publication-record.json",
      honesty_note:
        "Holistic equation stays scrollable even when the backend-backed label is very long.",
      equation: {
        label:
          String.raw`y(t) = \left(1 + 0.9\cdot y(t-1)\right) + \left(-0.25 + 0.1\cdot y(t-1)\right) + \sum_{i=0}^{24} c_i \prod_{j \ne i}\frac{\tau(t)-j}{i-j} + \varepsilon_t`,
        curve: analysis.descriptive_fit.chart.equation_curve,
      },
    };

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

    await waitFor(() => {
      expect(textContent("#tab-atlas")).toContain("Holistic equation");
    });

    const formula = document.querySelector(
      '#tab-atlas .equation-card .equation-formula.is-katex',
    );
    expect(formula).not.toBeNull();
    expect(getComputedStyle(formula).overflowX).toBe("auto");
  });

  test("uses a backend-backed holistic equation as the default overlay and preserves law labeling", async () => {
    const analysis = buildAtlasFixture();
    analysis.operator_point.publication = {
      status: "publishable",
      headline: "Operator point publication remains backend-backed for the holistic law path.",
    };
    analysis.claim_class = "holistic_equation";
    analysis.holistic_equation = {
      status: "completed",
      claim_class: "holistic_equation",
      deterministic_source: "predictive_law",
      probabilistic_source: "distribution",
      validation_scope_ref: "artifacts/validation-scope.json",
      publication_record_ref: "artifacts/publication-record.json",
      honesty_note:
        "Holistic equation renders the backend-backed deterministic and probabilistic sources together.",
      equation: {
        label:
          String.raw`y(t) = \left(1 + 0.9\cdot y(t-1)\right) + \left(-0.25 + 0.1\cdot y(t-1)\right) + \varepsilon_t`,
        delta_form_label:
          String.raw`\varepsilon_t \sim \mathcal{N}(0, \sigma_t^2)`,
        curve: analysis.descriptive_fit.chart.equation_curve,
      },
    };

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

    await waitFor(() => {
      expect(textContent("#tab-atlas")).toContain("Holistic equation");
    });

    const atlasText = textContent("#tab-atlas");
    expect(atlasText).toContain("Predictive symbolic law");
    expect(atlasText).toContain("Distribution");
    expect(atlasText).toContain(
      "Holistic equation renders the backend-backed deterministic and probabilistic sources together.",
    );

    document
      .querySelector('.tab-button[data-tab="point"]')
      .dispatchEvent(new MouseEvent("click", { bubbles: true }));

    await waitFor(() => {
      expect(textContent("#tab-point")).toContain("holistic equation");
    });
  });

  test("uses a neutral holistic claim label when a valid holistic payload only carries the minimal normalized contract fields", async () => {
    const analysis = buildAtlasFixture();
    analysis.operator_point.publication = {
      status: "publishable",
      headline: "Operator point publication remains backend-backed for the holistic law path.",
    };
    analysis.claim_class = "holistic_equation";
    analysis.holistic_equation = {
      status: "completed",
      claim_class: "holistic_equation",
      deterministic_source: "predictive_law",
      probabilistic_source: "distribution",
      validation_scope_ref: "artifacts/validation-scope.json",
      publication_record_ref: "artifacts/publication-record.json",
      honesty_note:
        "Holistic equation stays neutral when the backend returns only the normalized holistic payload.",
      equation: {
        curve: analysis.descriptive_fit.chart.equation_curve,
      },
    };

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

    await waitFor(() => {
      expect(textContent("#tab-overview")).toContain("Holistic equation");
    });

    const overviewText = textContent("#tab-overview");
    expect(overviewText).toContain("Holistic claim");
    expect(overviewText).not.toContain("sample exact closure");

    const atlasText = textContent("#tab-atlas");
    expect(atlasText).toContain("Holistic claim");
    expect(atlasText).not.toContain("sample exact closure");
  });

  test("rejects a synthetic holistic payload when legacy composition metadata is still attached", async () => {
    const analysis = buildAtlasFixture();
    analysis.operator_point.publication = {
      status: "publishable",
      headline: "Operator point publication remains backend-backed for the holistic law path.",
    };
    analysis.claim_class = "holistic_equation";
    analysis.holistic_equation = {
      status: "completed",
      claim_class: "holistic_equation",
      deterministic_source: "predictive_law",
      probabilistic_source: "distribution",
      validation_scope_ref: "artifacts/validation-scope.json",
      publication_record_ref: "artifacts/publication-record.json",
      mode: "composition_stochastic",
      composition_operator: "additive_residual",
      selected_probabilistic_lane: "distribution",
      honesty_note:
        "Legacy synthetic holistic metadata must not promote the payload as a valid holistic law object.",
      equation: {
        label:
          String.raw`y(t) = \left(1 + 0.9\cdot y(t-1)\right) + \left(-0.25 + 0.1\cdot y(t-1)\right) + \varepsilon_t`,
        curve: analysis.descriptive_fit.chart.equation_curve,
      },
    };

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

    await waitFor(() => {
      expect(textContent("#tab-overview")).toContain(
        "Best available descriptive approximation",
      );
    });

    const overviewText = textContent("#tab-overview");
    expect(overviewText).not.toContain("Holistic equation");
    expect(overviewText).not.toContain(
      "Legacy synthetic holistic metadata must not promote the payload as a valid holistic law object.",
    );

    const atlasText = textContent("#tab-atlas");
    expect(atlasText).not.toContain("Holistic equation");
    expect(atlasText).toContain("Benchmark-local descriptive fit");
  });

  test("downgrades a stale top-level holistic claim when the payload is synthetic and non-backed", async () => {
    const analysis = buildAtlasFixture();
    analysis.claim_class = "descriptive_fit";
    analysis.holistic_equation = {
      status: "completed",
      claim_class: "holistic_equation",
      source: "descriptive_fit",
      exactness: "sample_exact_closure",
      composition_operator: "additive_residual",
      selected_probabilistic_lane: "distribution",
      honesty_note:
        "Stale synthetic holistic payload should not outrank the descriptive fallback.",
      equation: {
        label:
          String.raw`y(t) = \left(2.75 + 0.994\cdot y(t-1)\right) + \sum_{i=0}^{9} c_i \prod_{j \ne i}\frac{\tau(t)-j}{i-j}`,
        delta_form_label:
          String.raw`c_i = y_i - \hat{y}_i^{\mathrm{compact}}`,
        curve: analysis.descriptive_fit.chart.equation_curve,
      },
    };

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

    await waitFor(() => {
      expect(textContent("#tab-overview")).toContain(
        "Best available descriptive approximation",
      );
    });

    const overviewText = textContent("#tab-overview");
    expect(overviewText).not.toContain("Holistic equation");
    expect(overviewText).not.toContain("sample exact closure");

    const atlasText = textContent("#tab-atlas");
    expect(atlasText).not.toContain("Holistic equation");
    expect(atlasText).not.toContain("sample exact closure");
    expect(atlasText).toContain("Benchmark-local descriptive fit");
  });

  test("prefers predictive symbolic law when a stale holistic payload is present but the normalized top-level taxonomy is predictive", async () => {
    const analysis = buildAtlasFixture();
    analysis.operator_point.publication = {
      status: "publishable",
      headline: "Operator point publication remains backend-backed for the predictive-within-scope path.",
    };
    analysis.claim_class = "predictive_law";
    analysis.gap_report = ["no_backend_joint_claim"];
    analysis.not_holistic_because = [
      "No validated deterministic-plus-probabilistic joint claim was published.",
    ];
    analysis.predictive_law = {
      status: "completed",
      claim_class: "predictive_law",
      claim_card_ref: "artifacts/claim-card.json",
      scorecard_ref: "artifacts/scorecard.json",
      validation_scope_ref: "artifacts/validation-scope.json",
      publication_record_ref: "artifacts/publication-record.json",
      honesty_note:
        "Predictive symbolic law remains the strongest backend-backed claim when the saved holistic top line is stale.",
      equation: {
        label: String.raw`y(t) = 1.8 + 0.92\cdot y(t-1)`,
        delta_form_label: String.raw`\Delta y(t) = 1.8 - 0.08\cdot y(t-1)`,
        curve: analysis.operator_point.equation.curve,
      },
    };
    analysis.holistic_equation = {
      status: "completed",
      claim_class: "holistic_equation",
      source: "predictive_law",
      deterministic_source: "predictive_law",
      probabilistic_source: "distribution",
      validation_scope_ref: "artifacts/validation-scope.json",
      publication_record_ref: "artifacts/publication-record.json",
      composition_operator: "additive_residual",
      selected_probabilistic_lane: "distribution",
      honesty_note:
        "Stale saved holistic payload should not outrank the surviving predictive-within-scope claim.",
      equation: {
        label:
          String.raw`y(t) = \left(1 + 0.9\cdot y(t-1)\right) + \left(-0.25 + 0.1\cdot y(t-1)\right) + \varepsilon_t`,
        curve: analysis.descriptive_fit.chart.equation_curve,
      },
    };

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

    await waitFor(() => {
      expect(textContent("#tab-overview")).toContain(
        "Predictive symbolic law",
      );
    });

    const overviewText = textContent("#tab-overview");
    expect(overviewText).toContain(
      "Predictive symbolic law remains the strongest backend-backed claim when the saved holistic top line is stale.",
    );
    expect(overviewText).not.toContain("Holistic equation");
    expect(overviewText).not.toContain(
      "Best available descriptive approximation",
    );
    expect(overviewText).toContain("Why not stronger?");
    expect(overviewText).toContain("No backend joint claim");

    document
      .querySelector('.tab-button[data-tab="point"]')
      .dispatchEvent(new MouseEvent("click", { bubbles: true }));

    await waitFor(() => {
      expect(textContent("#tab-point")).toContain(
        "Residual basis predictive symbolic law",
      );
    });
  });

  test("rejects stale holistic promotion when gap semantics require post-hoc symbolic synthesis", async () => {
    const analysis = buildAtlasFixture();
    analysis.claim_class = "descriptive_fit";
    analysis.gap_report = ["requires_posthoc_symbolic_synthesis"];
    analysis.not_holistic_because = [
      "Holistic claim would require post-hoc symbolic synthesis.",
    ];
    analysis.holistic_equation = {
      status: "completed",
      claim_class: "holistic_equation",
      source: "descriptive_fit",
      deterministic_source: "predictive_law",
      probabilistic_source: "distribution",
      validation_scope_ref: "artifacts/validation-scope.json",
      publication_record_ref: "artifacts/publication-record.json",
      composition_operator: "additive_residual",
      selected_probabilistic_lane: "distribution",
      honesty_note:
        "Post-hoc symbolic synthesis should not promote a stale holistic claim.",
      equation: {
        label:
          String.raw`y(t) = \left(2.75 + 0.994\cdot y(t-1)\right) + \phi_{\mathrm{posthoc}}(t)`,
        curve: analysis.descriptive_fit.chart.equation_curve,
      },
    };

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

    await waitFor(() => {
      expect(textContent("#tab-overview")).toContain(
        "Best available descriptive approximation",
      );
    });

    const overviewText = textContent("#tab-overview");
    expect(overviewText).not.toContain("Holistic equation");
    expect(overviewText).toContain("Why not stronger?");
    expect(overviewText).toContain("Requires posthoc symbolic synthesis");

    const atlasText = textContent("#tab-atlas");
    expect(atlasText).not.toContain("Holistic equation");
  });

  test("rejects stale holistic promotion when the payload only points at a synthetic symbolic view", async () => {
    const analysis = buildAtlasFixture();
    analysis.claim_class = "descriptive_fit";
    analysis.not_holistic_because = [
      "Holistic claim would require post-hoc symbolic synthesis.",
    ];
    analysis.holistic_equation = {
      status: "completed",
      claim_class: "holistic_equation",
      source: "descriptive_fit",
      mode: "composition_stochastic",
      deterministic_source: "predictive_law",
      probabilistic_source: "synthetic_views.symbolic_synthesis",
      validation_scope_ref: "artifacts/validation-scope.json",
      publication_record_ref: "artifacts/publication-record.json",
      composition_operator: "additive_residual",
      selected_probabilistic_lane: "distribution",
      honesty_note:
        "Synthetic symbolic views must stay diagnostic-only even when a stale holistic payload was saved.",
      equation: {
        label:
          String.raw`y(t) = \left(2.75 + 0.994\cdot y(t-1)\right) + \phi_{\mathrm{posthoc}}(t)`,
        delta_form_label: String.raw`\varepsilon_t \sim \phi_{\mathrm{synthetic}}`,
        curve: analysis.descriptive_fit.chart.equation_curve,
      },
    };

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

    await waitFor(() => {
      expect(textContent("#tab-overview")).toContain(
        "Best available descriptive approximation",
      );
    });

    const overviewText = textContent("#tab-overview");
    expect(overviewText).not.toContain("Holistic equation");
    expect(overviewText).toContain(
      "Holistic claim would require post-hoc symbolic synthesis.",
    );

    const atlasText = textContent("#tab-atlas");
    expect(atlasText).not.toContain("Holistic equation");
    expect(atlasText).toContain("Benchmark-local descriptive fit");
  });

  test("keeps stale holistic payloads out of the atlas equation stack when the claim stays descriptive", async () => {
    const analysis = buildAtlasFixture();
    analysis.claim_class = "descriptive_fit";
    analysis.holistic_equation = {
      status: "completed",
      claim_class: "descriptive_fit",
      source: "descriptive_fit",
      exactness: "sample_exact_closure",
      composition_operator: "additive_residual",
      selected_probabilistic_lane: "distribution",
      honesty_note:
        "Legacy holistic payload should not render as a top-line holistic card.",
      equation: {
        label:
          String.raw`y(t) = \left(2.75 + 0.994\cdot y(t-1)\right) + \sum_{i=0}^{9} c_i \prod_{j \ne i}\frac{\tau(t)-j}{i-j}`,
        delta_form_label:
          String.raw`c_i = y_i - \hat{y}_i^{\mathrm{compact}}`,
        curve: analysis.descriptive_fit.chart.equation_curve,
      },
    };

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

    await waitFor(() => {
      expect(textContent("#tab-atlas")).toContain("Equation Atlas");
    });

    const atlasText = textContent("#tab-atlas");
    expect(atlasText).not.toContain("Holistic equation");
    expect(atlasText).not.toContain(
      "Legacy holistic payload should not render as a top-line holistic card.",
    );
    expect(atlasText).toContain("Benchmark-local descriptive fit");
  });

  test("sanitizes banned claim labels on descriptive-only surfaces", async () => {
    const analysis = buildAtlasFixture();
    analysis.claim_class = "descriptive_fit";
    analysis.descriptive_fit.honesty_note =
      "Holistic equation stayed below the predictive symbolic law path.";
    analysis.descriptive_fit.semantic_audit.headline =
      "The equation stack keeps deterministic and probabilistic laws together.";
    analysis.operator_point.publication.headline =
      "The active deterministic law stays below the operator claim.";

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

    await waitFor(() => {
      expect(textContent("#tab-overview")).toContain(
        "Best available descriptive approximation",
      );
    });

    const overviewText = textContent("#tab-overview");
    expect(overviewText).toContain(
      "joint claim stayed below the predictive claim path.",
    );
    expect(overviewText).not.toContain(
      "Holistic equation stayed below the predictive symbolic law path.",
    );

    const atlasText = textContent("#tab-atlas");
    expect(atlasText).toContain("Benchmark-local descriptive fit");
    expect(atlasText).toContain(
      "deterministic and probabilistic objects together",
    );
    expect(atlasText).not.toContain(
      "deterministic and probabilistic laws together",
    );

    document
      .querySelector('.tab-button[data-tab="point"]')
      .dispatchEvent(new MouseEvent("click", { bubbles: true }));

    await waitFor(() => {
      expect(textContent("#tab-point")).toContain("Point lane state");
    });

    const pointText = textContent("#tab-point");
    expect(pointText).toContain("active deterministic overlay");
    expect(pointText).toContain("operator publication");
    expect(pointText).not.toContain("active deterministic law");
    expect(pointText).not.toContain("operator claim");
  });

  test("prefers the predictive symbolic law when no holistic claim passed", async () => {
    const analysis = buildAtlasFixture();
    analysis.operator_point.publication = {
      status: "publishable",
      headline: "Operator point publication remains backend-backed for the predictive-within-scope path.",
    };
    analysis.claim_class = "predictive_law";
    analysis.gap_report = [
      "no_backend_joint_claim",
      "probabilistic_evidence_thin",
    ];
    analysis.not_holistic_because = [
      "No validated deterministic-plus-probabilistic joint claim was published.",
    ];
    analysis.predictive_law = {
      status: "completed",
      claim_class: "predictive_law",
      claim_card_ref: "artifacts/claim-card.json",
      scorecard_ref: "artifacts/scorecard.json",
      validation_scope_ref: "artifacts/validation-scope.json",
      publication_record_ref: "artifacts/publication-record.json",
      honesty_note:
        "Predictive symbolic law reflects the publishable point-lane claim inside the declared validation scope.",
      equation: {
        delta_form_label: String.raw`\Delta y(t) = 1.8 - 0.08\cdot y(t-1)`,
        curve: analysis.operator_point.equation.curve,
      },
    };

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

    await waitFor(() => {
      expect(textContent("#tab-overview")).toContain("Predictive symbolic law");
    });

    const overviewText = textContent("#tab-overview");
    expect(overviewText).toContain(
      "Predictive symbolic law reflects the publishable point-lane claim inside the declared validation scope.",
    );
    expect(textContent("#hero")).toContain(
      "Predictive symbolic law reflects the publishable point-lane claim inside the declared validation scope.",
    );
    expect(textContent("#hero")).not.toContain("operator abstained");
    expect(overviewText).toContain("Why not stronger?");
    expect(overviewText).toContain("No backend joint claim");
    expect(overviewText).toContain("Probabilistic evidence thin");
    expect(overviewText).not.toContain("Best available descriptive approximation");
  });

  test("rejects law promotion when a retained abstention artifact survives under a publishable status", async () => {
    const analysis = buildAtlasFixture();
    analysis.operator_point.publication = {
      status: "publishable",
      headline: "Stale normalized status survived even though an abstention artifact was retained.",
    };
    analysis.operator_point.abstention = {};
    analysis.claim_class = "predictive_law";
    analysis.predictive_law = {
      status: "completed",
      claim_class: "predictive_law",
      claim_card_ref: "artifacts/claim-card.json",
      scorecard_ref: "artifacts/scorecard.json",
      validation_scope_ref: "artifacts/validation-scope.json",
      publication_record_ref: "artifacts/publication-record.json",
      honesty_note:
        "Predictive symbolic law must not render when the operator publication still retains an abstention artifact.",
      equation: {
        delta_form_label: String.raw`\Delta y(t) = 1.8 - 0.08\cdot y(t-1)`,
        curve: analysis.operator_point.equation.curve,
      },
    };

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

    await waitFor(() => {
      expect(textContent("#tab-overview")).toContain(
        "Best available descriptive approximation",
      );
    });

    const overviewText = textContent("#tab-overview");
    expect(overviewText).not.toContain("Predictive symbolic law");
    expect(overviewText).not.toContain(
      "Predictive symbolic law must not render when the operator publication still retains an abstention artifact.",
    );
  });

  test("rejects law promotion when the operator publication stays candidate-only", async () => {
    const analysis = buildAtlasFixture();
    analysis.operator_point.publication = {
      status: "candidate_only",
      headline: "Operator point path stopped at candidate-only publication.",
    };
    analysis.operator_point.abstention = {
      blocked_ceiling: "descriptive_only",
      reason_codes: ["candidate_only_publication"],
    };
    analysis.would_have_abstained_because = ["candidate_only_publication"];
    analysis.claim_class = "holistic_equation";
    analysis.predictive_law = {
      status: "completed",
      claim_class: "predictive_law",
      claim_card_ref: "artifacts/claim-card.json",
      scorecard_ref: "artifacts/scorecard.json",
      validation_scope_ref: "artifacts/validation-scope.json",
      publication_record_ref: "artifacts/publication-record.json",
      honesty_note:
        "Predictive symbolic law should not render when the operator publication never cleared candidate-only status.",
      equation: {
        label: String.raw`y(t) = 1.8 + 0.92\cdot y(t-1)`,
        delta_form_label: String.raw`\Delta y(t) = 1.8 - 0.08\cdot y(t-1)`,
        curve: analysis.operator_point.equation.curve,
      },
    };
    analysis.holistic_equation = {
      status: "completed",
      claim_class: "holistic_equation",
      source: "predictive_law",
      deterministic_source: "predictive_law",
      probabilistic_source: "distribution",
      validation_scope_ref: "artifacts/validation-scope.json",
      publication_record_ref: "artifacts/publication-record.json",
      composition_operator: "additive_residual",
      selected_probabilistic_lane: "distribution",
      honesty_note:
        "Holistic equation should not render when the operator publication never cleared candidate-only status.",
      equation: {
        label:
          String.raw`y(t) = \left(1 + 0.9\cdot y(t-1)\right) + \left(-0.25 + 0.1\cdot y(t-1)\right) + \varepsilon_t`,
        curve: analysis.descriptive_fit.chart.equation_curve,
      },
    };

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

    await waitFor(() => {
      expect(textContent("#tab-overview")).toContain(
        "Best available descriptive approximation",
      );
    });

    const overviewText = textContent("#tab-overview");
    expect(overviewText).not.toContain("Holistic equation");
    expect(overviewText).not.toContain("Predictive symbolic law");
  });

  test("rejects incomplete law payloads and falls back to the descriptive fit", async () => {
    const analysis = buildAtlasFixture();
    analysis.operator_point.publication = {
      status: "publishable",
      headline: "Operator point publication remains backend-backed for the law path under completeness checks.",
    };
    analysis.claim_class = "holistic_equation";
    analysis.holistic_equation = {
      status: "completed",
      claim_class: "holistic_equation",
      deterministic_source: "predictive_law",
      probabilistic_source: "distribution",
      validation_scope_ref: "artifacts/validation-scope.json",
      publication_record_ref: "artifacts/publication-record.json",
      equation: {},
    };
    analysis.predictive_law = {
      status: "completed",
      claim_class: "predictive_law",
      claim_card_ref: "artifacts/claim-card.json",
      scorecard_ref: "artifacts/scorecard.json",
      validation_scope_ref: "artifacts/validation-scope.json",
      publication_record_ref: "artifacts/publication-record.json",
      equation: {},
    };

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

    await waitFor(() => {
      expect(textContent("#tab-overview")).toContain(
        "Best available descriptive approximation",
      );
    });

    const overviewText = textContent("#tab-overview");
    expect(overviewText).not.toContain("Holistic equation");
    expect(overviewText).not.toContain("Predictive symbolic law");
    expect(textContent("#tab-atlas")).toContain(
      "Active overlay Benchmark-local descriptive fit",
    );

    document
      .querySelector('.tab-button[data-tab="point"]')
      .dispatchEvent(new MouseEvent("click", { bubbles: true }));

    await waitFor(() => {
      expect(textContent("#tab-point")).toContain(
        "Residual basis benchmark-local descriptive fit",
      );
    });
  });

  test("renders structured event definitions as readable probabilistic labels", async () => {
    const analysis = buildAtlasFixture();
    const structuredEvent = {
      event_id: "target_ge_origin_target",
      operator: "greater_than_or_equal",
      threshold: 503.6,
      threshold_source: "origin_target",
    };
    analysis.probabilistic.event_probability.latest_row.event_definition =
      structuredEvent;
    analysis.probabilistic.event_probability.rows =
      analysis.probabilistic.event_probability.rows.map((row) => ({
        ...row,
        event_definition: structuredEvent,
      }));
    analysis.probabilistic.event_probability.chart.forecast_probabilities =
      analysis.probabilistic.event_probability.chart.forecast_probabilities.map(
        (row) => ({
          ...row,
          event_definition: structuredEvent,
        }),
      );

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

    await waitFor(() => {
      expect(textContent("#tab-atlas")).toContain("Equation Atlas");
    });

    expect(textContent("#tab-atlas")).toContain(
      "Target ≥ origin target (503.6)",
    );
    expect(textContent("#tab-atlas")).not.toContain("[object Object]");
  });

  test("renders change-atlas historical and forecast summaries for the selected lane", async () => {
    const analysis = buildAtlasFixture();

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

    document
      .querySelector('.tab-button[data-tab="probabilistic"]')
      .dispatchEvent(new MouseEvent("click", { bubbles: true }));

    await waitFor(() => {
      expect(textContent("#tab-probabilistic")).toContain("Change atlas");
    });

    const probabilisticText = textContent("#tab-probabilistic");
    expect(probabilisticText).toContain("Historical empirical distribution");
    expect(probabilisticText).toContain("Forecast distribution");
    expect(probabilisticText).toContain(
      "Forecast lanes are shown as changes from the origin close.",
    );
    expect(probabilisticText).toContain("Origin close");
  });

  test("replaces unavailable llm guides with a built-in explainer fallback", async () => {
    const analysis = clone(savedAnalysisFixture);
    analysis.llm_explanations = {
      status: "unavailable",
      message: "OPENAI_API_KEY is not configured.",
    };

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

    await waitFor(() => {
      expect(textContent("#tab-overview")).toContain("Built-in guide");
    });

    const overviewText = textContent("#tab-overview");
    expect(overviewText).toContain("benchmark-local");
    expect(overviewText).toContain("point lane");
    expect(overviewText).toContain("OPENAI_API_KEY is not configured.");
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

function buildLiveAnalyzeFixture(savedAnalysis) {
  const liveAnalysis = clone(savedAnalysis);
  liveAnalysis.workspace_root =
    "/tmp/euclid/live/workspaces/2026-04-16T165500Z-spy-daily-return-live-overwrite-attempt";
  liveAnalysis.analysis_path = `${liveAnalysis.workspace_root}/analysis.live.json`;
  liveAnalysis.dataset.symbol = "SPY";
  liveAnalysis.dataset.rows = 5;
  liveAnalysis.dataset.stats.latest_value = 0.0041;
  liveAnalysis.dataset.dataset_csv = `${liveAnalysis.workspace_root}/datasets/spy-live-daily-return.csv`;
  liveAnalysis.dataset.raw_history_json = `${liveAnalysis.workspace_root}/datasets/spy-live-history.json`;
  liveAnalysis.operator_point.selected_family = "seasonal_naive";
  liveAnalysis.operator_point.publication = {
    status: "publishable",
    headline: "Live analyze response that should lose ownership.",
  };
  liveAnalysis.operator_point.manifest_path = `${liveAnalysis.workspace_root}/runs/operator/manifest.live.yaml`;
  liveAnalysis.operator_point.output_root = `${liveAnalysis.workspace_root}/runs/operator/output/live`;
  liveAnalysis.benchmark.manifest_path = `${liveAnalysis.workspace_root}/runs/benchmark/manifest.live.yaml`;
  liveAnalysis.benchmark.report_path = `${liveAnalysis.workspace_root}/reports/live-benchmark.md`;
  liveAnalysis.benchmark.portfolio_selection.winner_submitter_id =
    "seasonal_backend";
  liveAnalysis.benchmark.portfolio_selection.winner_candidate_id =
    "seasonal_naive_h1";
  liveAnalysis.benchmark.portfolio_selection.selection_explanation =
    "Seasonal backend wins the live rerun.";
  liveAnalysis.descriptive_fit.submitter_id = "seasonal_backend";
  liveAnalysis.descriptive_fit.candidate_id = "seasonal_naive_h1";
  liveAnalysis.descriptive_fit.equation.label = "y(t) = y(t-5)";
  return liveAnalysis;
}

function buildAtlasFixture() {
  const analysis = clone(savedAnalysisFixture);
  analysis.operator_point.publication = {
    status: "abstained",
    headline: "Replay verified a candidate, but publication gates failed.",
  };
  analysis.operator_point.equation.delta_form_label =
    "Δy(t) = 2.75 - 0.006*y(t-1)";
  analysis.descriptive_fit.equation.delta_form_label =
    "Δy(t) = 2.75 - 0.006*y(t-1)";
  analysis.probabilistic = {
    distribution: {
      status: "completed",
      selected_family: "analytic",
      replay_verification: "verified",
      aggregated_primary_score: 0.56,
      evidence: {
        strength: "thin",
        sample_size: 2,
        origin_count: 1,
        horizon_count: 2,
        headline: "Distribution evidence is thin.",
      },
      calibration: {
        status: "passed",
        passed: true,
        gate_effect: "publishable",
        diagnostics: [],
      },
      equation: {
        candidate_id: "analytic_lag1_affine",
        family_id: "analytic",
        label: "y(t) = 2.5 + 0.995*y(t-1)",
        delta_form_label: "Δy(t) = 2.5 - 0.005*y(t-1)",
        parameter_summary: {
          intercept: 2.5,
          lag_coefficient: 0.995,
        },
      },
      latest_row: {
        origin_time: "2025-01-13T00:00:00Z",
        available_at: "2025-01-15T21:00:00Z",
        horizon: 2,
        location: 505.3,
        scale: 4.5,
        realized_observation: 503.6,
      },
      rows: [
        {
          origin_time: "2025-01-13T00:00:00Z",
          available_at: "2025-01-14T21:00:00Z",
          horizon: 1,
          location: 504.2,
          scale: 3.1,
          realized_observation: 503.1,
        },
        {
          origin_time: "2025-01-13T00:00:00Z",
          available_at: "2025-01-15T21:00:00Z",
          horizon: 2,
          location: 505.3,
          scale: 4.5,
          realized_observation: 503.6,
        },
      ],
      chart: {
        forecast_bands: [
          {
            origin_time: "2025-01-13T00:00:00Z",
            available_at: "2025-01-14T21:00:00Z",
            center: 504.2,
            lower: 501.1,
            upper: 507.3,
            realized_observation: 503.1,
          },
          {
            origin_time: "2025-01-13T00:00:00Z",
            available_at: "2025-01-15T21:00:00Z",
            center: 505.3,
            lower: 500.8,
            upper: 509.8,
            realized_observation: 503.6,
          },
        ],
      },
    },
    quantile: {
      status: "completed",
      selected_family: "analytic",
      replay_verification: "verified",
      evidence: {
        strength: "thin",
        sample_size: 2,
        origin_count: 1,
        horizon_count: 2,
        headline: "Quantile evidence is thin.",
      },
      calibration: {
        status: "passed",
        passed: true,
        gate_effect: "publishable",
        diagnostics: [],
      },
      equation: {
        candidate_id: "analytic_lag1_affine",
        family_id: "analytic",
        label: "Q_p = μ + σΦ^-1(p)",
      },
      latest_row: {
        origin_time: "2025-01-13T00:00:00Z",
        available_at: "2025-01-15T21:00:00Z",
        horizon: 2,
        quantiles: [
          { level: 0.1, value: 499.9 },
          { level: 0.5, value: 505.3 },
          { level: 0.9, value: 510.7 },
        ],
        realized_observation: 503.6,
      },
      rows: [
        {
          origin_time: "2025-01-13T00:00:00Z",
          available_at: "2025-01-14T21:00:00Z",
          horizon: 1,
          quantiles: [
            { level: 0.1, value: 500.1 },
            { level: 0.5, value: 504.2 },
            { level: 0.9, value: 508.3 },
          ],
          realized_observation: 503.1,
        },
        {
          origin_time: "2025-01-13T00:00:00Z",
          available_at: "2025-01-15T21:00:00Z",
          horizon: 2,
          quantiles: [
            { level: 0.1, value: 499.9 },
            { level: 0.5, value: 505.3 },
            { level: 0.9, value: 510.7 },
          ],
          realized_observation: 503.6,
        },
      ],
      chart: {
        forecast_quantiles: [
          {
            origin_time: "2025-01-13T00:00:00Z",
            available_at: "2025-01-14T21:00:00Z",
            horizon: 1,
            quantiles: [
              { level: 0.1, value: 500.1 },
              { level: 0.5, value: 504.2 },
              { level: 0.9, value: 508.3 },
            ],
            realized_observation: 503.1,
          },
          {
            origin_time: "2025-01-13T00:00:00Z",
            available_at: "2025-01-15T21:00:00Z",
            horizon: 2,
            quantiles: [
              { level: 0.1, value: 499.9 },
              { level: 0.5, value: 505.3 },
              { level: 0.9, value: 510.7 },
            ],
            realized_observation: 503.6,
          },
        ],
      },
    },
    interval: {
      status: "completed",
      selected_family: "recursive",
      replay_verification: "verified",
      evidence: {
        strength: "thin",
        sample_size: 2,
        origin_count: 1,
        horizon_count: 2,
        headline: "Interval evidence is thin.",
      },
      calibration: {
        status: "passed",
        passed: true,
        gate_effect: "publishable",
        diagnostics: [],
      },
      equation: {
        candidate_id: "recursive_level_smoother",
        family_id: "recursive",
        label: "level(t) = 0.95*x(t-1) + 0.05*level(t-1)",
      },
      latest_row: {
        origin_time: "2025-01-13T00:00:00Z",
        available_at: "2025-01-15T21:00:00Z",
        horizon: 2,
        lower_bound: 500.4,
        upper_bound: 508.0,
        realized_observation: 503.6,
      },
      rows: [
        {
          origin_time: "2025-01-13T00:00:00Z",
          available_at: "2025-01-14T21:00:00Z",
          horizon: 1,
          lower_bound: 500.9,
          upper_bound: 507.5,
          realized_observation: 503.1,
        },
        {
          origin_time: "2025-01-13T00:00:00Z",
          available_at: "2025-01-15T21:00:00Z",
          horizon: 2,
          lower_bound: 500.4,
          upper_bound: 508.0,
          realized_observation: 503.6,
        },
      ],
      chart: {
        forecast_bands: [
          {
            origin_time: "2025-01-13T00:00:00Z",
            available_at: "2025-01-14T21:00:00Z",
            lower: 500.9,
            upper: 507.5,
            realized_observation: 503.1,
          },
          {
            origin_time: "2025-01-13T00:00:00Z",
            available_at: "2025-01-15T21:00:00Z",
            lower: 500.4,
            upper: 508.0,
            realized_observation: 503.6,
          },
        ],
      },
    },
    event_probability: {
      status: "completed",
      selected_family: "algorithmic",
      replay_verification: "verified",
      evidence: {
        strength: "thin",
        sample_size: 2,
        origin_count: 1,
        horizon_count: 2,
        headline: "Event-probability evidence is thin.",
      },
      calibration: {
        status: "passed",
        passed: true,
        gate_effect: "publishable",
        diagnostics: [],
      },
      equation: {
        candidate_id: "algorithmic_last_observation",
        family_id: "algorithmic",
        label: "y(t) = y(t-1)",
      },
      latest_row: {
        origin_time: "2025-01-13T00:00:00Z",
        available_at: "2025-01-15T21:00:00Z",
        horizon: 2,
        event_probability: 0.5,
        realized_event: 0,
        event_definition: "Y >= 503.6",
      },
      rows: [
        {
          origin_time: "2025-01-13T00:00:00Z",
          available_at: "2025-01-14T21:00:00Z",
          horizon: 1,
          event_probability: 0.56,
          realized_event: 1,
          event_definition: "Y >= 503.6",
        },
        {
          origin_time: "2025-01-13T00:00:00Z",
          available_at: "2025-01-15T21:00:00Z",
          horizon: 2,
          event_probability: 0.5,
          realized_event: 0,
          event_definition: "Y >= 503.6",
        },
      ],
      chart: {
        forecast_probabilities: [
          {
            origin_time: "2025-01-13T00:00:00Z",
            available_at: "2025-01-14T21:00:00Z",
            event_probability: 0.56,
            realized_event: 1,
            event_definition: "Y >= 503.6",
          },
          {
            origin_time: "2025-01-13T00:00:00Z",
            available_at: "2025-01-15T21:00:00Z",
            event_probability: 0.5,
            realized_event: 0,
            event_definition: "Y >= 503.6",
          },
        ],
      },
    },
  };
  analysis.change_atlas = {
    status: "completed",
    horizons: [1, 2],
    headline:
      "Forecast lanes are shown as changes from the origin close.",
    metrics: [
      {
        id: "delta",
        label: "Delta",
        short_label: "Delta",
      },
      {
        id: "return",
        label: "Return",
        short_label: "Return",
      },
      {
        id: "log_return",
        label: "Log Return",
        short_label: "Log Return",
      },
    ],
    historical: {
      delta: {
        1: {
          horizon: 1,
          sample_size: 3,
          latest_value: -1.4,
          mean: 0.5,
          stdev: 2.1,
          quantiles: [
            { level: 0.1, value: -2.4 },
            { level: 0.5, value: 0.3 },
            { level: 0.9, value: 3.1 },
          ],
          histogram: [
            { lower: -3.0, upper: -1.0, count: 1 },
            { lower: -1.0, upper: 1.0, count: 1 },
            { lower: 1.0, upper: 3.5, count: 1 },
          ],
        },
        2: {
          horizon: 2,
          sample_size: 2,
          latest_value: -0.8,
          mean: 1.2,
          stdev: 1.8,
          quantiles: [
            { level: 0.1, value: -0.8 },
            { level: 0.5, value: 1.2 },
            { level: 0.9, value: 3.2 },
          ],
          histogram: [
            { lower: -1.0, upper: 0.5, count: 1 },
            { lower: 0.5, upper: 3.5, count: 1 },
          ],
        },
      },
      return: {
        1: {
          horizon: 1,
          sample_size: 3,
          latest_value: -0.0028,
          mean: 0.001,
          stdev: 0.004,
          quantiles: [
            { level: 0.1, value: -0.0048 },
            { level: 0.5, value: 0.0006 },
            { level: 0.9, value: 0.0061 },
          ],
          histogram: [
            { lower: -0.006, upper: -0.001, count: 1 },
            { lower: -0.001, upper: 0.002, count: 1 },
            { lower: 0.002, upper: 0.007, count: 1 },
          ],
        },
        2: {
          horizon: 2,
          sample_size: 2,
          latest_value: -0.0016,
          mean: 0.0022,
          stdev: 0.0035,
          quantiles: [
            { level: 0.1, value: -0.0016 },
            { level: 0.5, value: 0.0022 },
            { level: 0.9, value: 0.006 },
          ],
          histogram: [
            { lower: -0.002, upper: 0.001, count: 1 },
            { lower: 0.001, upper: 0.0065, count: 1 },
          ],
        },
      },
      log_return: {
        1: {
          horizon: 1,
          sample_size: 3,
          latest_value: -0.0028,
          mean: 0.001,
          stdev: 0.004,
          quantiles: [
            { level: 0.1, value: -0.0048 },
            { level: 0.5, value: 0.0006 },
            { level: 0.9, value: 0.0061 },
          ],
          histogram: [
            { lower: -0.006, upper: -0.001, count: 1 },
            { lower: -0.001, upper: 0.002, count: 1 },
            { lower: 0.002, upper: 0.007, count: 1 },
          ],
        },
        2: {
          horizon: 2,
          sample_size: 2,
          latest_value: -0.0016,
          mean: 0.0022,
          stdev: 0.0035,
          quantiles: [
            { level: 0.1, value: -0.0016 },
            { level: 0.5, value: 0.0022 },
            { level: 0.9, value: 0.006 },
          ],
          histogram: [
            { lower: -0.002, upper: 0.001, count: 1 },
            { lower: 0.001, upper: 0.0065, count: 1 },
          ],
        },
      },
    },
    forecast: {
      support: "price_projection",
      lanes: {
        distribution: {
          delta: {
            1: {
              horizon: 1,
              origin_time: "2025-01-13T00:00:00Z",
              origin_close: 503.0,
              center: 1.2,
              lower: -1.9,
              upper: 4.3,
              realized: 0.1,
            },
            2: {
              horizon: 2,
              origin_time: "2025-01-13T00:00:00Z",
              origin_close: 503.0,
              center: 2.3,
              lower: -2.2,
              upper: 6.8,
              realized: 0.6,
            },
          },
          return: {
            1: {
              horizon: 1,
              origin_time: "2025-01-13T00:00:00Z",
              origin_close: 503.0,
              center: 0.0023856858846918487,
              lower: -0.003777335984095428,
              upper: 0.008548707753479125,
              realized: 0.00019880715705765412,
            },
            2: {
              horizon: 2,
              origin_time: "2025-01-13T00:00:00Z",
              origin_close: 503.0,
              center: 0.0045725646123260435,
              lower: -0.00437375745526839,
              upper: 0.013518886679920477,
              realized: 0.0011928429423459245,
            },
          },
          log_return: {
            1: {
              horizon: 1,
              origin_time: "2025-01-13T00:00:00Z",
              origin_close: 503.0,
              center: 0.0023828447034153024,
              lower: -0.0037844899048188816,
              upper: 0.00851238908469167,
              realized: 0.00019878740215655006,
            },
            2: {
              horizon: 2,
              origin_time: "2025-01-13T00:00:00Z",
              origin_close: 503.0,
              center: 0.0045621418183938725,
              lower: -0.004383350947726132,
              upper: 0.013428360303874084,
              realized: 0.0011921317868573215,
            },
          },
        },
      },
    },
  };
  return analysis;
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });
  return { promise, resolve, reject };
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

async function settle(turns = 8) {
  for (let index = 0; index < turns; index += 1) {
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
}

function textContent(selector) {
  return document.querySelector(selector)?.textContent?.replace(/\s+/g, " ").trim() || "";
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
