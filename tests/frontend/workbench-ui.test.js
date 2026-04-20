import { beforeEach, describe, expect, it, vi } from "vitest";
import { readFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";

const INDEX_PATH =
  "/Users/danielbloom/Desktop/euclid/src/euclid/_assets/workbench/index.html";
const APP_PATH =
  "/Users/danielbloom/Desktop/euclid/src/euclid/_assets/workbench/app.js";
const ANALYSIS_PATH =
  "/Users/danielbloom/Desktop/euclid/build/workbench/20260420T010656Z-spy-price-close/analysis.json";

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
    ).toContain("Workspace");
    expect(document.body.textContent).toContain("SPY");
    expect(document.body.textContent).toContain("Price Close");
  });

  it("renders a linked analytical workspace, point explanation, and decision-first evidence regions", async () => {
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
});
