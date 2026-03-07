"""Thread-safe service wrappers for gameplay and RL training operations."""

from __future__ import annotations

import copy
import os
import threading
import time
from typing import Any

from .game import Game2048
from .rl import ReinforceCnnConfig, ReinforceCnnTrainer


class TrainingManager:
    """Manage asynchronous RL training jobs for frontend monitoring."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._status = self._new_status()
        self._latest_model_state: dict[str, Any] | None = None

    def _new_status(self) -> dict[str, Any]:
        return {
            "running": False,
            "stopRequested": False,
            "stopped": False,
            "error": None,
            "algorithm": "reinforce",
            "network": "cnn-3layer-no-padding",
            "encoding": "onehot-16x4x4",
            "requestedEpisodes": 0,
            "completedEpisodes": 0,
            "workers": 1,
            "seed": None,
            "maxSteps": None,
            "terminateOnWin": True,
            "averageScore": 0.0,
            "maxTileSeen": 0,
            "latestState": None,
            "lastEpisode": None,
            "recentEpisodes": [],
            "loss": None,
            "entropy": None,
            "globalStep": 0,
            "startedAt": None,
            "finishedAt": None,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._status)

    def start(
        self,
        *,
        episodes: int,
        workers: int,
        seed: int | None,
        max_steps: int | None,
        terminate_on_win: bool,
    ) -> dict[str, Any]:
        if episodes <= 0:
            raise ValueError("episodes must be greater than 0")
        if workers < 0:
            raise ValueError("workers must be >= 0")

        resolved_workers = (os.cpu_count() or 1) if workers == 0 else workers
        if resolved_workers != 1:
            raise ValueError("CNN policy-gradient training currently supports workers=1")

        with self._lock:
            if self._status["running"]:
                raise RuntimeError("Training is already running")

            self._stop_event = threading.Event()
            self._status = self._new_status()
            self._status.update(
                {
                    "running": True,
                    "requestedEpisodes": episodes,
                    "workers": resolved_workers,
                    "seed": seed,
                    "maxSteps": max_steps,
                    "terminateOnWin": terminate_on_win,
                    "startedAt": time.time(),
                }
            )

            self._thread = threading.Thread(
                target=self._run_job,
                args=(episodes, seed, max_steps, terminate_on_win),
                daemon=True,
            )
            self._thread.start()
            return copy.deepcopy(self._status)

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if not self._status["running"]:
                return copy.deepcopy(self._status)
            self._status["stopRequested"] = True
            self._stop_event.set()
            return copy.deepcopy(self._status)

    def _on_episode_end(
        self,
        episode: int,
        result: dict[str, Any],
        metrics: dict[str, Any],
    ) -> None:
        episode_summary = {
            "episode": episode,
            "score": int(result.get("score", 0)),
            "maxTile": int(result.get("maxTile", 0)),
            "steps": int(result.get("steps", 0)),
            "won": bool(result.get("won", False)),
            "over": bool(result.get("over", False)),
            "terminated": bool(result.get("terminated", False)),
            "truncated": bool(result.get("truncated", False)),
        }

        with self._lock:
            completed = int(self._status["completedEpisodes"]) + 1
            self._status["completedEpisodes"] = completed
            self._status["averageScore"] = float(metrics.get("averageScore", self._status["averageScore"]))
            self._status["maxTileSeen"] = max(
                int(self._status["maxTileSeen"]),
                int(result.get("maxTile", 0)),
            )
            self._status["lastEpisode"] = episode_summary
            self._status["latestState"] = {
                "state": result.get("state"),
                "rawBoard": result.get("rawBoard"),
                "score": int(result.get("score", 0)),
                "maxTile": int(result.get("maxTile", 0)),
                "steps": int(result.get("steps", 0)),
            }
            self._status["recentEpisodes"].append(episode_summary)
            if len(self._status["recentEpisodes"]) > 20:
                self._status["recentEpisodes"] = self._status["recentEpisodes"][-20:]
            self._status["loss"] = metrics.get("loss")
            self._status["entropy"] = metrics.get("entropy")
            self._status["globalStep"] = int(metrics.get("globalStep", self._status["globalStep"]))

    def _run_job(
        self,
        episodes: int,
        seed: int | None,
        max_steps: int | None,
        terminate_on_win: bool,
    ) -> None:
        error_message: str | None = None
        try:
            trainer = ReinforceCnnTrainer(ReinforceCnnConfig(), seed=seed)
            summary = trainer.train(
                episodes=episodes,
                max_steps=max_steps,
                terminate_on_win=terminate_on_win,
                stop_event=self._stop_event,
                on_episode_end=self._on_episode_end,
            )
            self._latest_model_state = summary.get("modelStateDict")
            with self._lock:
                self._status["globalStep"] = int(summary.get("globalStep", self._status["globalStep"]))
                self._status["loss"] = summary.get("lastLoss", self._status["loss"])
                self._status["entropy"] = summary.get("lastEntropy", self._status["entropy"])
                self._status["averageScore"] = float(summary.get("averageScore", self._status["averageScore"]))
                self._status["maxTileSeen"] = max(
                    int(self._status["maxTileSeen"]),
                    int(summary.get("maxTileSeen", 0)),
                )
        except Exception as error:  # noqa: BLE001
            error_message = str(error)
        finally:
            with self._lock:
                completed = int(self._status["completedEpisodes"])
                self._status["running"] = False
                self._status["finishedAt"] = time.time()
                self._status["stopped"] = bool(self._status["stopRequested"]) or completed < episodes
                self._status["error"] = error_message


class GameService:
    """Thread-safe wrapper around a single game instance."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._game = Game2048()
        self._training = TrainingManager()

    def state(self) -> dict[str, Any]:
        with self._lock:
            return self._game.serialize_state()

    def restart(self) -> dict[str, Any]:
        with self._lock:
            self._game.reset()
            return self._game.serialize_state()

    def keep_playing(self) -> dict[str, Any]:
        with self._lock:
            self._game.keep_going()
            return self._game.serialize_state()

    def move(self, direction: int) -> dict[str, Any]:
        with self._lock:
            moved = self._game.move(direction)
            state = self._game.serialize_state()
            state["moved"] = moved
            return state

    def training_status(self) -> dict[str, Any]:
        return self._training.status()

    def training_start(
        self,
        *,
        episodes: int,
        workers: int,
        seed: int | None,
        max_steps: int | None,
        terminate_on_win: bool,
    ) -> dict[str, Any]:
        return self._training.start(
            episodes=episodes,
            workers=workers,
            seed=seed,
            max_steps=max_steps,
            terminate_on_win=terminate_on_win,
        )

    def training_stop(self) -> dict[str, Any]:
        return self._training.stop()
