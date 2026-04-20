from __future__ import annotations

from euclid.workbench.server import build_app_shell


def test_build_app_shell_includes_core_workbench_sections() -> None:
    html = build_app_shell()

    assert "Euclid Market Workbench" in html
    assert "No analysis loaded" in html
    assert "Overview" in html
    assert "Atlas" in html
    assert "Point" in html
    assert "Probabilistic" in html
    assert "Benchmark" in html
    assert "Artifacts" in html
    assert "analysis-form" in html
    assert "tab-overview" in html
    assert "tab-atlas" in html
    assert "app.js" in html
    assert "/vendor/katex/katex.min.css" in html
    assert ">Bars<" not in html
    assert "Quick range" in html
