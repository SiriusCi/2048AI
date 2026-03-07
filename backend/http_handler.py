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
        if path == "/api/train/status":
            self._send_json(self.service.training_status())
            return
        self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def _handle_api_post(self, path: str) -> None:
        if path == "/api/train/start":
            body = self._read_json()
            if body is None:
                self._send_json({"error": "Invalid JSON body"}, HTTPStatus.BAD_REQUEST)
                return

            episodes = body.get("episodes", 100)
            workers = body.get("workers", 1)
            seed = body.get("seed", None)
            max_steps = body.get("maxSteps", None)
            terminate_on_win = body.get("terminateOnWin", True)

            try:
                episodes = int(episodes)
                workers = int(workers)
            except (TypeError, ValueError):
                self._send_json(
                    {"error": "Fields 'episodes' and 'workers' must be integers"},
                    HTTPStatus.BAD_REQUEST,
                )
                return

            if seed is not None:
                try:
                    seed = int(seed)
                except (TypeError, ValueError):
                    self._send_json({"error": "Field 'seed' must be an integer or null"}, HTTPStatus.BAD_REQUEST)
                    return

            if max_steps is not None:
                try:
                    max_steps = int(max_steps)
                except (TypeError, ValueError):
                    self._send_json(
                        {"error": "Field 'maxSteps' must be an integer or null"},
                        HTTPStatus.BAD_REQUEST,
                    )
                    return
                if max_steps <= 0:
                    self._send_json({"error": "Field 'maxSteps' must be > 0"}, HTTPStatus.BAD_REQUEST)
                    return

            if not isinstance(terminate_on_win, bool):
                self._send_json(
                    {"error": "Field 'terminateOnWin' must be true or false"},
                    HTTPStatus.BAD_REQUEST,
                )
                return

            if episodes <= 0:
                self._send_json({"error": "Field 'episodes' must be > 0"}, HTTPStatus.BAD_REQUEST)
                return
            if workers < 0:
                self._send_json({"error": "Field 'workers' must be >= 0"}, HTTPStatus.BAD_REQUEST)
                return

            try:
                status = self.service.training_start(
                    episodes=episodes,
                    workers=workers,
                    seed=seed,
                    max_steps=max_steps,
                    terminate_on_win=terminate_on_win,
                )
            except RuntimeError as error:
                self._send_json({"error": str(error)}, HTTPStatus.CONFLICT)
                return
            except ValueError as error:
                self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                return

            self._send_json(status, HTTPStatus.ACCEPTED)
            return

        if path == "/api/train/stop":
            self._send_json(self.service.training_stop())
            return

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
