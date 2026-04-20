from __future__ import annotations

import json
import mimetypes
import os
import webbrowser
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlparse

from euclid.workbench.service import (
    create_workbench_analysis,
    default_workbench_date_range,
    list_recent_analyses,
    normalize_analysis_payload,
    ordered_target_specs,
)
from euclid.workbench.explainer import (
    DEFAULT_OPENAI_API_KEY_ENV_VAR,
    ensure_cached_workbench_explanations,
    workbench_explainer_model,
)

_CLIENT_DISCONNECT_ERRORS = (
    BrokenPipeError,
    ConnectionAbortedError,
    ConnectionResetError,
)


def build_app_shell() -> str:
    return _read_asset_text("index.html")


def serve_workbench(
    *,
    host: str,
    port: int,
    output_root: Path,
    project_root: Path | None,
    api_key_env_var: str,
    open_browser: bool,
    openai_api_key_env_var: str = DEFAULT_OPENAI_API_KEY_ENV_VAR,
) -> None:
    server = create_workbench_server(
        host=host,
        port=port,
        output_root=output_root,
        project_root=project_root,
        api_key_env_var=api_key_env_var,
        openai_api_key_env_var=openai_api_key_env_var,
    )
    url = f"http://{host}:{port}"
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - manual shutdown path
        pass
    finally:
        server.server_close()


def create_workbench_server(
    *,
    host: str,
    port: int,
    output_root: Path,
    project_root: Path | None,
    api_key_env_var: str,
    openai_api_key_env_var: str = DEFAULT_OPENAI_API_KEY_ENV_VAR,
) -> ThreadingHTTPServer:
    resolved_output_root = output_root.resolve()
    resolved_output_root.mkdir(parents=True, exist_ok=True)
    resolved_project_root = (
        project_root.resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[3]
    )
    handler = partial(
        WorkbenchRequestHandler,
        output_root=resolved_output_root,
        project_root=resolved_project_root,
        api_key_env_var=api_key_env_var,
        openai_api_key_env_var=openai_api_key_env_var,
    )
    return ThreadingHTTPServer((host, port), handler)


class WorkbenchRequestHandler(BaseHTTPRequestHandler):
    server_version = "EuclidWorkbench/1.0"

    def __init__(
        self,
        *args: Any,
        output_root: Path,
        project_root: Path,
        api_key_env_var: str,
        openai_api_key_env_var: str,
        **kwargs: Any,
    ) -> None:
        self.output_root = output_root
        self.project_root = project_root
        self.api_key_env_var = api_key_env_var
        self.openai_api_key_env_var = openai_api_key_env_var
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send_asset("index.html", "text/html; charset=utf-8")
            return
        if parsed.path == "/app.css":
            self._send_asset("app.css", "text/css; charset=utf-8")
            return
        if parsed.path == "/app.js":
            self._send_asset("app.js", "text/javascript; charset=utf-8")
            return
        if parsed.path.startswith("/vendor/"):
            asset_name = parsed.path.removeprefix("/")
            if ".." in Path(asset_name).parts:
                self._send_json(
                    {
                        "error": {
                            "type": "BadRequest",
                            "message": "invalid asset path",
                        }
                    },
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            try:
                self._send_asset(asset_name, _asset_content_type(asset_name))
            except FileNotFoundError:
                self._send_json(
                    {
                        "error": {
                            "type": "NotFound",
                            "message": f"unknown path {parsed.path}",
                        }
                    },
                    status=HTTPStatus.NOT_FOUND,
                )
            return
        if parsed.path == "/api/config":
            target_specs = ordered_target_specs()
            self._send_json(
                {
                    "target_specs": target_specs,
                    "default_target_id": target_specs[0]["id"] if target_specs else None,
                    "default_date_range": default_workbench_date_range(),
                    "recent_analyses": list_recent_analyses(
                        output_root=self.output_root,
                        limit=8,
                    ),
                    "api_key_env_var": self.api_key_env_var,
                    "has_api_key_env": bool(
                        os.environ.get(self.api_key_env_var, "").strip()
                    ),
                    "llm_explanations": {
                        "api_key_env_var": self.openai_api_key_env_var,
                        "has_api_key_env": bool(
                            os.environ.get(
                                self.openai_api_key_env_var, ""
                            ).strip()
                        ),
                        "model": workbench_explainer_model(),
                    },
                }
            )
            return
        if parsed.path == "/api/analysis":
            params = parse_qs(parsed.query)
            analysis_path = params.get("analysis_path", [None])[0]
            if analysis_path is None:
                self._send_json(
                    {
                        "error": {
                            "type": "BadRequest",
                            "message": "analysis_path is required",
                        }
                    },
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            try:
                payload = self._load_saved_analysis(analysis_path)
            except FileNotFoundError:
                self._send_json(
                    {"error": {"type": "NotFound", "message": "analysis not found"}},
                    status=HTTPStatus.NOT_FOUND,
                )
                return
            except ValueError as exc:
                self._send_json(
                    {"error": {"type": "BadRequest", "message": str(exc)}},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            self._send_json(payload)
            return
        self._send_json(
            {
                "error": {
                    "type": "NotFound",
                    "message": f"unknown path {parsed.path}",
                }
            },
            status=HTTPStatus.NOT_FOUND,
        )

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/analyze":
            self._send_json(
                {
                    "error": {
                        "type": "NotFound",
                        "message": f"unknown path {parsed.path}",
                    }
                },
                status=HTTPStatus.NOT_FOUND,
            )
            return
        try:
            payload = self._read_json_body()
            api_key = str(payload.get("api_key") or "").strip() or os.environ.get(
                self.api_key_env_var, ""
            ).strip()
            if not api_key:
                raise ValueError(
                    f"Provide an API key in the form or export {self.api_key_env_var}."
                )
            analysis = create_workbench_analysis(
                symbol=str(payload.get("symbol") or "SPY").strip().upper(),
                api_key=api_key,
                target_id=str(
                    payload.get("target_id")
                    or (ordered_target_specs()[0]["id"] if ordered_target_specs() else "")
                ),
                output_root=self.output_root,
                project_root=self.project_root,
                start_date=_string_or_none(payload.get("start_date")),
                end_date=_string_or_none(payload.get("end_date")),
                benchmark_workers=max(1, int(payload.get("benchmark_workers") or 1)),
                include_probabilistic=bool(payload.get("include_probabilistic", True)),
                include_benchmark=bool(payload.get("include_benchmark", True)),
            )
            analysis = normalize_analysis_payload(analysis)
            analysis = self._attach_cached_workbench_explanations(analysis)
        except ValueError as exc:
            self._send_json(
                {"error": {"type": "BadRequest", "message": str(exc)}},
                status=HTTPStatus.BAD_REQUEST,
            )
            return
        except Exception as exc:  # pragma: no cover - live/manual path
            self._send_json(
                {"error": {"type": exc.__class__.__name__, "message": str(exc)}},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return
        self._send_json(analysis)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        if not raw_body:
            return {}
        payload = json.loads(raw_body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _send_asset(self, asset_name: str, content_type: str) -> None:
        payload = _read_asset_bytes(asset_name)
        self._write_response(
            payload,
            status=HTTPStatus.OK,
            content_type=content_type,
        )

    def _send_json(self, payload: Any, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, indent=2).encode("utf-8")
        self._write_response(
            encoded,
            status=status,
            content_type="application/json; charset=utf-8",
        )

    def _write_response(
        self,
        payload: bytes,
        *,
        status: HTTPStatus,
        content_type: str,
    ) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except _CLIENT_DISCONNECT_ERRORS:
            self.close_connection = True

    def _load_saved_analysis(self, analysis_path: str) -> dict[str, Any]:
        resolved_path = Path(analysis_path).expanduser().resolve()
        if not _is_within_root(root=self.output_root, candidate=resolved_path):
            raise ValueError(
                "analysis_path must point inside the configured output root"
            )
        if not resolved_path.is_file():
            raise FileNotFoundError(resolved_path)
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("saved analysis payload must be a JSON object")
        payload.setdefault("analysis_path", str(resolved_path))
        payload = normalize_analysis_payload(payload)
        return self._attach_cached_workbench_explanations(payload)

    def _attach_cached_workbench_explanations(
        self,
        analysis: Mapping[str, Any],
    ) -> dict[str, Any]:
        analysis_path = _string_or_none(analysis.get("analysis_path"))
        return ensure_cached_workbench_explanations(
            analysis,
            api_key=os.environ.get(self.openai_api_key_env_var, "").strip() or None,
            analysis_path=(
                Path(analysis_path).expanduser()
                if analysis_path
                else None
            ),
            model=workbench_explainer_model(),
        )


def _read_asset_text(asset_name: str) -> str:
    asset_path = resources.files("euclid").joinpath(
        "_assets",
        "workbench",
        asset_name,
    )
    return asset_path.read_text(encoding="utf-8")


def _read_asset_bytes(asset_name: str) -> bytes:
    asset_path = resources.files("euclid").joinpath(
        "_assets",
        "workbench",
        asset_name,
    )
    return asset_path.read_bytes()


def _is_within_root(*, root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _asset_content_type(asset_name: str) -> str:
    suffix = Path(asset_name).suffix.lower()
    explicit = {
        ".css": "text/css; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".mjs": "text/javascript; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".woff2": "font/woff2",
        ".woff": "font/woff",
        ".ttf": "font/ttf",
    }
    if suffix in explicit:
        return explicit[suffix]
    guessed, _ = mimetypes.guess_type(asset_name)
    return guessed or "application/octet-stream"


__all__ = [
    "build_app_shell",
    "create_workbench_server",
    "serve_workbench",
]
