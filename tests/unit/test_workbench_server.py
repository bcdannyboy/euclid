from __future__ import annotations

from http import HTTPStatus

import pytest

from euclid.workbench.server import WorkbenchRequestHandler


class _BrokenPipeWriter:
    def write(self, _: bytes) -> None:
        raise BrokenPipeError("client disconnected during body write")


class _StubWorkbenchHandler:
    _write_response = WorkbenchRequestHandler._write_response

    def __init__(self, *, fail_stage: str) -> None:
        self.fail_stage = fail_stage
        self.statuses: list[HTTPStatus] = []
        self.headers: list[tuple[str, str]] = []
        self.wfile = _BrokenPipeWriter()

    def send_response(self, status: HTTPStatus) -> None:
        self.statuses.append(status)

    def send_header(self, name: str, value: str) -> None:
        self.headers.append((name, value))

    def end_headers(self) -> None:
        if self.fail_stage == "headers":
            raise BrokenPipeError("client disconnected during header flush")


@pytest.mark.parametrize("fail_stage", ["headers", "body"])
def test_send_json_ignores_expected_client_disconnects(fail_stage: str) -> None:
    handler = _StubWorkbenchHandler(fail_stage=fail_stage)

    WorkbenchRequestHandler._send_json(handler, {"status": "ok"})

    assert handler.statuses == [HTTPStatus.OK]
    assert ("Content-Type", "application/json; charset=utf-8") in handler.headers
