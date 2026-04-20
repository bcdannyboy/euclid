from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_notebook_demo_executes_against_the_sample_demo_assets(
    tmp_path: Path,
) -> None:
    notebook_path = PROJECT_ROOT / "output/jupyter-notebook/prototype-demo.ipynb"

    notebook_json = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook_json["cells"]
    code = "\n".join(
        "".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"
    )

    assert cells[0]["cell_type"] == "markdown"
    assert "prototype demo" in "".join(cells[0]["source"]).lower()
    assert "run_demo_probabilistic_evaluation" in code
    assert "inspect_demo_probabilistic_prediction" in code
    assert "inspect_demo_calibration" in code
    assert "publish_demo_run_to_catalog" in code
    assert "load_demo_publication_catalog" in code
    assert "inspect_demo_catalog_entry" in code
    assert "probabilistic-distribution-demo.yaml" in code
    assert "probabilistic-interval-demo.yaml" in code
    assert "probabilistic-quantile-demo.yaml" in code
    assert "probabilistic-event-probability-demo.yaml" in code

    cells[2]["source"] = (
        "from __future__ import annotations\n\n"
        "from pathlib import Path\n"
        "import sys\n\n"
        f"PROJECT_ROOT = Path({str(PROJECT_ROOT)!r})\n"
        "SRC_ROOT = PROJECT_ROOT / 'src'\n"
        "if str(SRC_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(SRC_ROOT))\n\n"
        "from euclid import (\n"
        "    inspect_demo_catalog_entry,\n"
        "    inspect_demo_calibration,\n"
        "    inspect_demo_probabilistic_prediction,\n"
        "    load_demo_publication_catalog,\n"
        "    publish_demo_run_to_catalog,\n"
        "    run_demo,\n"
        "    run_demo_probabilistic_evaluation,\n"
        ")\n\n"
        "MANIFEST_DIR = PROJECT_ROOT / 'fixtures/runtime/phase06'\n"
        "RETAINED_MANIFEST = PROJECT_ROOT / 'fixtures/runtime/prototype-demo.yaml'\n"
        f"OUTPUT_ROOT = Path({str(tmp_path / 'notebook-demo')!r})\n"
        "{\n"
        "    'manifest_dir': str(MANIFEST_DIR),\n"
        "    'retained_manifest': str(RETAINED_MANIFEST),\n"
        "    'output_root': str(OUTPUT_ROOT),\n"
        "}\n"
    )

    namespace: dict[str, object] = {"__name__": "__main__"}
    for cell in cells:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), namespace)

    cases = namespace["probabilistic_cases"]

    assert set(cases) == {
        "distribution",
        "interval",
        "quantile",
        "event_probability",
    }
    assert cases["distribution"]["prediction"].forecast_object_type == "distribution"
    assert cases["interval"]["prediction"].forecast_object_type == "interval"
    assert cases["quantile"]["prediction"].forecast_object_type == "quantile"
    assert (
        cases["event_probability"]["prediction"].forecast_object_type
        == "event_probability"
    )
    assert all(case["calibration"].diagnostics for case in cases.values())
    published_catalog = namespace["published_catalog"]
    published_entry = namespace["published_entry"]
    published_inspection = namespace["published_inspection"]

    assert published_catalog.entry_count == 1
    assert published_entry.publication_mode == "abstention_only_publication"
    assert published_inspection.entry.publication_id == published_entry.publication_id
    assert published_inspection.replay_bundle.replay_verification_status == "verified"
