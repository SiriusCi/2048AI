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
        if path == "/api/replay/list":
            self._send_json({"replays": self.service.replay_list()})
            return
        if path == "/api/replay/status":
            self._send_json(self.service.replay_status())
            return
        self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def _handle_api_post(self, path: str) -> None:
        if path == "/api/train/start":
            body = self._read_json()
            if body is None:
                self._send_json({"error": "Invalid JSON body"}, HTTPStatus.BAD_REQUEST)
                return

            defaults = self.service.training_start_defaults()
            episodes = body.get("episodes", defaults.get("episodes"))
            workers = body.get("workers", defaults.get("workers"))
            seed = body.get("seed", defaults.get("seed"))
            max_steps = body.get("maxSteps", defaults.get("maxSteps"))
            terminate_on_win = body.get("terminateOnWin", defaults.get("terminateOnWin"))
            sync_with_frontend = body.get("syncWithFrontend", defaults.get("syncWithFrontend"))
            tensorboard_log_dir = body.get("tensorboardLogDir", defaults.get("tensorboardLogDir"))
            tensorboard_run_name = body.get("tensorboardRunName", defaults.get("tensorboardRunName"))
            checkpoint_every_episodes = body.get(
                "checkpointEveryEpisodes",
                defaults.get("checkpointEveryEpisodes"),
            )
            checkpoint_dir = body.get("checkpointDir", defaults.get("checkpointDir"))
            checkpoint_prefix = body.get("checkpointPrefix", defaults.get("checkpointPrefix"))
            load_model_path = body.get("loadModelPath", defaults.get("loadModelPath"))
            play_only = body.get("playOnly", defaults.get("playOnly"))

            try:
                episodes = int(episodes)
                workers = int(workers)
                checkpoint_every_episodes = int(checkpoint_every_episodes)
            except (TypeError, ValueError):
                self._send_json(
                    {"error": "Fields 'episodes', 'workers' and 'checkpointEveryEpisodes' must be integers"},
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
            if not isinstance(sync_with_frontend, bool):
                self._send_json(
                    {"error": "Field 'syncWithFrontend' must be true or false"},
                    HTTPStatus.BAD_REQUEST,
                )
                return

            if tensorboard_log_dir is not None and not isinstance(tensorboard_log_dir, str):
                self._send_json(
                    {"error": "Field 'tensorboardLogDir' must be a string or null"},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            if tensorboard_run_name is not None and not isinstance(tensorboard_run_name, str):
                self._send_json(
                    {"error": "Field 'tensorboardRunName' must be a string or null"},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            if checkpoint_dir is not None and not isinstance(checkpoint_dir, str):
                self._send_json(
                    {"error": "Field 'checkpointDir' must be a string or null"},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            if not isinstance(checkpoint_prefix, str) or checkpoint_prefix.strip() == "":
                self._send_json(
                    {"error": "Field 'checkpointPrefix' must be a non-empty string"},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            if load_model_path is not None and not isinstance(load_model_path, str):
                self._send_json(
                    {"error": "Field 'loadModelPath' must be a string or null"},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            if not isinstance(play_only, bool):
                self._send_json(
                    {"error": "Field 'playOnly' must be true or false"},
                    HTTPStatus.BAD_REQUEST,
                )
                return

            if episodes <= 0:
                self._send_json({"error": "Field 'episodes' must be > 0"}, HTTPStatus.BAD_REQUEST)
                return
            if workers < 0:
                self._send_json({"error": "Field 'workers' must be >= 0"}, HTTPStatus.BAD_REQUEST)
                return
            if checkpoint_every_episodes < 0:
                self._send_json({"error": "Field 'checkpointEveryEpisodes' must be >= 0"}, HTTPStatus.BAD_REQUEST)
                return

            try:
                status = self.service.training_start(
                    episodes=episodes,
                    workers=workers,
                    seed=seed,
                    max_steps=max_steps,
                    terminate_on_win=terminate_on_win,
                    sync_with_frontend=sync_with_frontend,
                    tensorboard_log_dir=tensorboard_log_dir,
                    tensorboard_run_name=tensorboard_run_name,
                    checkpoint_every_episodes=checkpoint_every_episodes,
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_prefix=checkpoint_prefix,
                    load_model_path=load_model_path,
                    play_only=play_only,
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

        if path == "/api/train/step-done":
            body = self._read_json()
            if body is None:
                self._send_json({"error": "Invalid JSON body"}, HTTPStatus.BAD_REQUEST)
                return

            frame_id = body.get("frameId")
            try:
                frame_id = int(frame_id)
            except (TypeError, ValueError):
                self._send_json({"error": "Field 'frameId' must be an integer"}, HTTPStatus.BAD_REQUEST)
                return

            try:
                self._send_json(self.service.training_step_done(frame_id))
            except ValueError as error:
                self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
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

        if path == "/api/replay/load":
            body = self._read_json()
            if body is None:
                self._send_json({"error": "Invalid JSON body"}, HTTPStatus.BAD_REQUEST)
                return
            filename = body.get("filename")
            if not filename or not isinstance(filename, str):
                self._send_json({"error": "Field 'filename' is required"}, HTTPStatus.BAD_REQUEST)
                return
            try:
                self._send_json(self.service.replay_load(filename))
            except (FileNotFoundError, OSError) as error:
                self._send_json({"error": str(error)}, HTTPStatus.NOT_FOUND)
            return

        if path == "/api/replay/step":
            self._send_json(self.service.replay_step())
            return

        if path == "/api/replay/stop":
            self._send_json(self.service.replay_stop())
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
