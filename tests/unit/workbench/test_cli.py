from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from euclid.cli import app

RUNNER = CliRunner()


def test_workbench_serve_command_delegates_to_server(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_serve_workbench(
        *,
        host: str,
        port: int,
        output_root: Path,
        project_root: Path | None,
        api_key_env_var: str,
        open_browser: bool,
    ) -> None:
        captured.update(
            {
                "host": host,
                "port": port,
                "output_root": output_root,
                "project_root": project_root,
                "api_key_env_var": api_key_env_var,
                "open_browser": open_browser,
            }
        )

    monkeypatch.setattr("euclid.cli.workbench.serve_workbench", _fake_serve_workbench)

    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            "8899",
            "--output-root",
            str(tmp_path / "workbench"),
            "--api-key-env-var",
            "EUCLID_FMP_KEY",
            "--no-open-browser",
        ],
    )

    assert result.exit_code == 0
    assert "Euclid workbench listening on http://127.0.0.1:8899" in result.stdout
    assert captured == {
        "host": "127.0.0.1",
        "port": 8899,
        "output_root": tmp_path / "workbench",
        "project_root": None,
        "api_key_env_var": "EUCLID_FMP_KEY",
        "open_browser": False,
    }
