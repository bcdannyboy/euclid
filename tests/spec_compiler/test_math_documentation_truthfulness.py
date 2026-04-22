from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MATH_DOC = REPO_ROOT / "docs" / "math.md"
REFERENCE_INDEX = REPO_ROOT / "docs" / "reference" / "README.md"
TRUTHFULNESS_DOCS = (
    REPO_ROOT / "docs" / "testing-truthfulness.md",
    REPO_ROOT / "docs" / "reference" / "testing-truthfulness.md",
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_math_doc_matches_retained_stochastic_support_path() -> None:
    text = _text(MATH_DOC)

    assert "`src/euclid/modules/probabilistic_evaluation.py`" in text
    assert "`src/euclid/stochastic/process_models.py`" in text
    assert "`family_id=\"gaussian\"`" in text
    assert "`horizon_scale_law=\"sqrt_horizon\"`" in text
    assert "s_h=s_0\\sqrt{h}" in text

    stale_fragments = (
        "Scale uses family-specific growth",
        "Analytic: $s_h=s_0\\sqrt{h}$",
        "Recursive: $s_h=s_0(1+0.15(h-1))$",
        "Spectral: $s_h=s_0(1+(A/10)h)$",
        "Algorithmic: $s_h=s_0(1+0.2(h-1))$",
    )
    for fragment in stale_fragments:
        assert fragment not in text


def test_math_doc_states_rewrite_invariance_and_claim_boundaries() -> None:
    text = _text(MATH_DOC)

    for required in (
        "## 12) Rewrite and equality-saturation behavior",
        "rewrite_evidence_only_not_claim",
        "unsafe_rewrite_rejected",
        "## 13) Invariance and transport gates",
        "residual_spread",
        "max_parameter_drift",
        "min_support_jaccard",
        "## 14) Claim promotion boundaries",
        "External-engine outputs, rewrite traces, live API success, and benchmark files",
    ):
        assert required in text


def test_math_doc_uses_github_safe_math_markdown() -> None:
    text = _text(MATH_DOC)
    lines = text.splitlines()
    inside_display_math = False
    block_start = 0
    display_blocks = 0

    for line_number, line in enumerate(lines, start=1):
        if "$$" in line:
            assert line == "$$", (
                f"GitHub display math fences must be top-level lines at line "
                f"{line_number}: {line!r}"
            )
            inside_display_math = not inside_display_math
            if inside_display_math:
                display_blocks += 1
                block_start = line_number
                assert line_number == 1 or not lines[line_number - 2].strip(), (
                    f"GitHub display math opening fence at line {line_number} "
                    "must be preceded by a blank line"
                )
            else:
                assert line_number == len(lines) or not lines[line_number].strip(), (
                    f"GitHub display math closing fence at line {line_number} "
                    "must be followed by a blank line"
                )
                block_start = 0
            continue
        if not inside_display_math:
            assert r"\mathbf{1}_" not in line, (
                "Indicator functions with subscripts must stay in display math "
                f"so GitHub does not parse underscores as emphasis at line {line_number}"
            )
            continue
        assert line == line.lstrip(), (
            f"GitHub may nest indented display math inside a Markdown block at "
            f"line {line_number} (block starts at line {block_start}): {line!r}"
        )
        assert not line.lstrip().startswith(("-", "*", "+", ">", "#")), (
            f"GitHub may parse markdown inside display math at line {line_number} "
            f"(block starts at line {block_start}): {line!r}"
        )

    assert not inside_display_math, "Unclosed display math block in docs/math.md"
    assert display_blocks >= 1
    assert r"\mathbb{1}\{" not in text


def test_reference_index_routes_to_math_document() -> None:
    text = _text(REFERENCE_INDEX)

    assert "`../math.md`" in text
    assert "stochastic support, rewrite, invariance, and calibration" in text


def test_workbench_truthfulness_docs_reference_existing_tests() -> None:
    pattern = re.compile(r"`([^`]*tests/[^`]+?\.py)`")
    for doc_path in TRUTHFULNESS_DOCS:
        missing = []
        for match in pattern.finditer(_text(doc_path)):
            relative_path = match.group(1)
            if "*" in relative_path:
                continue
            if not (REPO_ROOT / relative_path).is_file():
                missing.append(relative_path)
        assert not missing, (
            f"{doc_path.relative_to(REPO_ROOT).as_posix()} references missing tests: "
            + ", ".join(missing)
        )
