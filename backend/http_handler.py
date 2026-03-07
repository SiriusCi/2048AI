"""HTTP request handling for static files and game APIs."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .service import GameService


class GameRequestHandler(SimpleHTTPRequestHandler):
    """Serve static frontend files and JSON API for game control."""

    service = GameService()
    root_dir = Path(__file__).resolve().parent.parent

    @classmethod
    def configure(cls, *, root_dir: Path | None = None, service: GameService | None = None) -> None:
        if root_dir is not None:
            cls.root_dir = root_dir
        if service is not None:
            cls.service = service

    def __init__(self, *args: Any, directory: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, directory=directory or str(self.root_dir), **kwargs)

    def do_GET(self) -> None:  # noqa: N802 (required by BaseHTTPRequestHandler)
        path = urlparse(self.path).path
        if path.startswith("/api/"):
            self._handle_api_get(path)
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802 (required by BaseHTTPRequestHandler)
        path = urlparse(self.path).path
        if not path.startswith("/api/"):
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        self._handle_api_post(path)

    def _handle_api_get(self, path: str) -> None:
        if path == "/api/state":
            self._send_json(self.service.state())
            return
        self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def _handle_api_post(self, path: str) -> None:
        if path == "/api/restart":
            self._send_json(self.service.restart())
            return

        if path == "/api/keep-playing":
            self._send_json(self.service.keep_playing())
            return

        if path == "/api/move":
            body = self._read_json()
            if body is None:
                self._send_json({"error": "Invalid JSON body"}, HTTPStatus.BAD_REQUEST)
                return
            try:
                direction = int(body.get("direction"))
            except (TypeError, ValueError):
                self._send_json(
                    {"error": "Field 'direction' must be an integer in [0, 1, 2, 3]"},
                    HTTPStatus.BAD_REQUEST,
                )
                return

            if direction not in (0, 1, 2, 3):
                self._send_json(
                    {"error": "Field 'direction' must be one of 0, 1, 2, 3"},
                    HTTPStatus.BAD_REQUEST,
                )
                return

            try:
                self._send_json(self.service.move(direction))
            except ValueError as error:
                self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return

        self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def _read_json(self) -> dict[str, Any] | None:
        try:
            raw_length = self.headers.get("Content-Length", "0")
            length = int(raw_length)
            payload = self.rfile.read(length) if length > 0 else b"{}"
            decoded = payload.decode("utf-8")
            loaded = json.loads(decoded)
            if isinstance(loaded, dict):
                return loaded
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        return None

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
