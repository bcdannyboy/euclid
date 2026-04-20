from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
MATH_SOURCE_MAP_PATH = REPO_ROOT / "schemas/core/math-source-map.yaml"
SOURCE_MAP_PATH = REPO_ROOT / "schemas/core/source-map.yaml"

LIVE_REFERENCE_DOCS = {
    "docs/reference/system.md",
    "docs/reference/modeling-pipeline.md",
    "docs/reference/search-core.md",
    "docs/reference/contracts-manifests.md",
}
EXPECTED_SOURCE_DOCUMENTS = {
    "docs/reference/modeling-pipeline.md",
    "docs/reference/search-core.md",
}
ALLOWED_DECISIONS = {"promote_directly", "synthesize", "replace_boundary_only"}
ALLOWED_POSTURES = {
    "affirmative_core",
    "contract_core",
    "mixed_contract_with_boundary",
    "boundary_only",
}
LEGACY_DOC_PREFIXES = ("docs/canonical/", "docs/math/", "docs/module-specs/")


def _load_math_source_map() -> dict:
    return yaml.safe_load(MATH_SOURCE_MAP_PATH.read_text(encoding="utf-8"))


def _load_source_map() -> dict:
    return yaml.safe_load(SOURCE_MAP_PATH.read_text(encoding="utf-8"))


def _assert_existing_repo_path(path_text: str) -> None:
    path = REPO_ROOT / path_text
    assert path.exists(), f"missing referenced repo path: {path_text}"


def test_math_source_map_exists_and_covers_requested_source_set() -> None:
    payload = _load_math_source_map()

    assert payload["map_id"] == "canonical_math_source_map"
    recorded_sources = {entry["path"] for entry in payload["source_documents"]}
    assert recorded_sources == EXPECTED_SOURCE_DOCUMENTS

    for entry in payload["source_documents"]:
        _assert_existing_repo_path(entry["path"])
        assert not entry["path"].startswith(LEGACY_DOC_PREFIXES)
        assert entry["posture"] in ALLOWED_POSTURES
        assert entry["strongest_for"], f"{entry['path']} must map to at least one canonical object"

    source_map = _load_source_map()
    assert source_map["reference_workspace"]["docs_root"] == "docs/reference"

    entries = {entry["source"]: set(entry["canonical_targets"]) for entry in source_map["entries"]}
    assert LIVE_REFERENCE_DOCS <= entries["README.md"]
    assert entries["src/euclid/modules"] == {"docs/reference/modeling-pipeline.md"}
    assert entries["src/euclid/search"] == {"docs/reference/search-core.md"}
    assert entries["src/euclid/math"] == {"docs/reference/search-core.md"}
    assert entries["src/euclid/contracts"] == {"docs/reference/contracts-manifests.md"}
    assert entries["src/euclid/manifests"] == {"docs/reference/contracts-manifests.md"}


def test_math_source_map_uses_closed_decisions_and_valid_repo_references() -> None:
    payload = _load_math_source_map()
    source_map = _load_source_map()
    schema_roots = tuple(source_map["reference_workspace"]["schema_roots"])

    assert {item["id"] for item in payload["decision_vocab"]} == ALLOWED_DECISIONS
    assert {item["id"] for item in payload["document_posture_vocab"]} == ALLOWED_POSTURES

    seen_object_ids = set()
    seen_decisions = set()

    for entry in payload["supplemental_support_refs"]:
        _assert_existing_repo_path(entry["path"])
        assert not entry["path"].startswith(LEGACY_DOC_PREFIXES)
        assert entry["path"].startswith("schemas/core/")
        assert entry["purpose"]

    canonical_doc_targets = set()

    for obj in payload["canonical_objects"]:
        object_id = obj["object_id"]
        assert object_id not in seen_object_ids, f"duplicate canonical object id: {object_id}"
        seen_object_ids.add(object_id)

        decision = obj["decision"]
        assert decision in ALLOWED_DECISIONS
        seen_decisions.add(decision)

        assert obj["canonical_targets"], f"{object_id} must declare canonical targets"
        for target in obj["canonical_targets"]:
            assert not target.startswith(LEGACY_DOC_PREFIXES)
            assert target.startswith(("docs/reference/", "schemas/"))
            if target.startswith("docs/reference/"):
                _assert_existing_repo_path(target)
                canonical_doc_targets.add(target)
            else:
                assert any(target.startswith(f"{schema_root}/") for schema_root in schema_roots), (
                    f"{object_id} must target current schema roots: {target}"
                )

        strongest = obj["strongest_precursor"]
        _assert_existing_repo_path(strongest["path"])
        assert not strongest["path"].startswith(LEGACY_DOC_PREFIXES)
        assert strongest["path"].startswith(("docs/reference/", "schemas/core/"))
        assert strongest["rationale"]

        for support in obj.get("supporting_precursors", []):
            _assert_existing_repo_path(support["path"])
            assert not support["path"].startswith(LEGACY_DOC_PREFIXES)
            assert support["path"].startswith(("docs/reference/", "schemas/core/"))
            assert support["rationale"]

        assert "rewrite_requirements" in obj, f"{object_id} must make rewrite requirements explicit"

    assert seen_decisions == ALLOWED_DECISIONS
    assert canonical_doc_targets == EXPECTED_SOURCE_DOCUMENTS


def test_math_source_map_links_every_source_document_to_a_canonical_object() -> None:
    payload = _load_math_source_map()
    object_ids = {obj["object_id"] for obj in payload["canonical_objects"]}

    linked_object_ids = set()
    seen_source_docs = set()
    for entry in payload["source_documents"]:
        seen_source_docs.add(entry["path"])
        linked_object_ids.update(entry["strongest_for"])

    assert seen_source_docs == EXPECTED_SOURCE_DOCUMENTS
    assert linked_object_ids <= object_ids
    assert linked_object_ids == object_ids, "every canonical object should be claimed by the source inventory"
