"""Thread-safe service wrappers for gameplay and RL training operations."""

from __future__ import annotations

import copy
import os
import threading
import time
from typing import Any

from .game import Game2048
from .rl import DQNConfig, DQNTrainer


class TrainingManager:
    """Manage asynchronous RL training jobs for frontend monitoring."""

    POST_ACK_DELAY_SEC = 0.1
    DEFAULT_TENSORBOARD_LOG_DIR = os.path.join("runs", "2048")
    DEFAULT_CHECKPOINT_DIR = os.path.join("models", "2048")

    def __init__(
        self,
        *,
        rl_config: DQNConfig | None = None,
        post_ack_delay_sec: float | None = None,
        default_tensorboard_log_dir: str | None = DEFAULT_TENSORBOARD_LOG_DIR,
        default_checkpoint_dir: str | None = DEFAULT_CHECKPOINT_DIR,
    ) -> None:
        self._lock = threading.Lock()
        self._sync_cv = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._rl_config = rl_config or DQNConfig()
        self._post_ack_delay_sec = (
            self.POST_ACK_DELAY_SEC if post_ack_delay_sec is None else float(post_ack_delay_sec)
        )
        if self._post_ack_delay_sec < 0:
            raise ValueError("post_ack_delay_sec must be >= 0")
        self._default_tensorboard_log_dir = default_tensorboard_log_dir
        self._default_checkpoint_dir = default_checkpoint_dir
        self._status = self._new_status()
        self._latest_model_state: dict[str, Any] | None = None

    def _new_status(self) -> dict[str, Any]:
        return {
            "running": False,
            "stopRequested": False,
            "stopped": False,
            "error": None,
            "algorithm": "dqn",
            "network": "dueling-cnn-3layer",
            "encoding": "onehot-16x4x4",
            "playOnly": False,
            "requestedEpisodes": 0,
            "completedEpisodes": 0,
            "currentEpisode": 0,
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
            "epsilon": None,
            "globalStep": 0,
            "syncWithFrontend": False,
            "latestFrameId": 0,
            "ackedFrameId": 0,
            "awaitingAck": False,
            "coolingDown": False,
            "postAckDelaySec": self._post_ack_delay_sec,
            "tensorboardEnabled": False,
            "tensorboardLogDir": self._default_tensorboard_log_dir,
            "tensorboardRunName": None,
            "tensorboardRunDir": None,
            "tensorboardWarning": None,
            "checkpointEveryEpisodes": 0,
            "checkpointDir": self._default_checkpoint_dir,
            "checkpointPrefix": "dqn_cnn",
            "checkpointsSaved": 0,
            "latestCheckpointPath": None,
            "loadModelPath": None,
            "loadedModelPath": None,
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
        sync_with_frontend: bool = False,
        tensorboard_log_dir: str | None = None,
        tensorboard_run_name: str | None = None,
        checkpoint_every_episodes: int = 0,
        checkpoint_dir: str | None = None,
        checkpoint_prefix: str = "dqn_cnn",
        load_model_path: str | None = None,
        play_only: bool = False,
    ) -> dict[str, Any]:
        if episodes <= 0:
            raise ValueError("episodes must be greater than 0")
        if workers < 0:
            raise ValueError("workers must be >= 0")
        if checkpoint_every_episodes < 0:
            raise ValueError("checkpoint_every_episodes must be >= 0")

        resolved_workers = (os.cpu_count() or 1) if workers == 0 else workers
        if resolved_workers != 1:
            raise ValueError("CNN policy-gradient training currently supports workers=1")

        resolved_tensorboard_log_dir = (
            self._default_tensorboard_log_dir if tensorboard_log_dir is None else tensorboard_log_dir
        )
        resolved_checkpoint_dir = self._default_checkpoint_dir if checkpoint_dir is None else checkpoint_dir

        with self._sync_cv:
            if self._status["running"]:
                raise RuntimeError("Training is already running")

            self._stop_event = threading.Event()
            self._status = self._new_status()
            self._status.update(
                {
                    "running": True,
                    "algorithm": "dqn-play" if play_only else "dqn",
                    "playOnly": play_only,
                    "requestedEpisodes": episodes,
                    "workers": resolved_workers,
                    "seed": seed,
                    "maxSteps": max_steps,
                    "terminateOnWin": terminate_on_win,
                    "syncWithFrontend": bool(sync_with_frontend),
                    "tensorboardLogDir": resolved_tensorboard_log_dir,
                    "tensorboardRunName": tensorboard_run_name,
                    "checkpointEveryEpisodes": checkpoint_every_episodes,
                    "checkpointDir": resolved_checkpoint_dir,
                    "checkpointPrefix": checkpoint_prefix,
                    "loadModelPath": load_model_path,
                    "startedAt": time.time(),
                }
            )

            self._thread = threading.Thread(
                target=self._run_job,
                args=(
                    episodes,
                    seed,
                    max_steps,
                    terminate_on_win,
                    resolved_tensorboard_log_dir,
                    tensorboard_run_name,
                    checkpoint_every_episodes,
                    resolved_checkpoint_dir,
                    checkpoint_prefix,
                    load_model_path,
                    play_only,
                ),
                daemon=True,
            )
            self._thread.start()
            return copy.deepcopy(self._status)

    def stop(self) -> dict[str, Any]:
        with self._sync_cv:
            if not self._status["running"]:
                return copy.deepcopy(self._status)
            self._status["stopRequested"] = True
            self._stop_event.set()
            self._sync_cv.notify_all()
            return copy.deepcopy(self._status)

    def step_done(self, frame_id: int) -> dict[str, Any]:
        if frame_id < 0:
            raise ValueError("frameId must be >= 0")

        with self._sync_cv:
            current_acked = int(self._status["ackedFrameId"])
            latest = int(self._status["latestFrameId"])
            effective_frame = min(frame_id, latest)
            if effective_frame > current_acked:
                self._status["ackedFrameId"] = effective_frame

            self._status["awaitingAck"] = bool(
                self._status["running"] and int(self._status["ackedFrameId"]) < latest
            )
            self._sync_cv.notify_all()
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

        average_score = 0.0
        checkpoints_saved = 0
        latest_checkpoint_path: str | None = None

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
            self._status["epsilon"] = metrics.get("epsilon")
            self._status["globalStep"] = int(metrics.get("globalStep", self._status["globalStep"]))
            self._status["currentEpisode"] = int(episode)
            if metrics.get("checkpointPath") is not None:
                self._status["latestCheckpointPath"] = metrics.get("checkpointPath")
            if metrics.get("checkpointsSaved") is not None:
                self._status["checkpointsSaved"] = int(metrics.get("checkpointsSaved", 0))

            average_score = float(self._status["averageScore"])
            checkpoints_saved = int(self._status["checkpointsSaved"])
            latest_checkpoint_path = self._status["latestCheckpointPath"]

        checkpoint_text = latest_checkpoint_path if latest_checkpoint_path else "-"
        print(
            "Episode {episode}: score={score}, maxTile={maxTile}, steps={steps}, won={won}, "
            "avgScore={average:.2f}, checkpointsSaved={checkpoints}, latestCheckpoint={checkpoint}".format(
                episode=episode_summary["episode"],
                score=episode_summary["score"],
                maxTile=episode_summary["maxTile"],
                steps=episode_summary["steps"],
                won=episode_summary["won"],
                average=average_score,
                checkpoints=checkpoints_saved,
                checkpoint=checkpoint_text,
            ),
            flush=True,
        )

    def _on_step(
        self,
        episode: int,
        obs: dict[str, Any],
        metrics: dict[str, Any],
    ) -> None:
        with self._sync_cv:
            frame_id = int(self._status["latestFrameId"]) + 1
            sync_with_frontend = bool(self._status.get("syncWithFrontend", False))
            self._status["currentEpisode"] = int(episode)
            self._status["globalStep"] = int(metrics.get("globalStep", self._status["globalStep"]))
            self._status["latestState"] = {
                "state": obs.get("state"),
                "rawBoard": obs.get("rawBoard"),
                "score": int(obs.get("score", 0)),
                "maxTile": int(obs.get("maxTile", 0)),
                "steps": int(obs.get("steps", 0)),
                "action": obs.get("action"),
                "animationGrid": obs.get("animationGrid"),
            }
            self._status["latestFrameId"] = frame_id
            if not sync_with_frontend:
                # Pure backend mode: do not block on frontend ACK.
                self._status["ackedFrameId"] = frame_id
                self._status["awaitingAck"] = False
                self._status["coolingDown"] = False
                self._sync_cv.notify_all()
                return

            self._status["awaitingAck"] = True
            self._sync_cv.notify_all()

            while (
                not self._stop_event.is_set()
                and self._status["running"]
                and int(self._status["ackedFrameId"]) < frame_id
            ):
                self._sync_cv.wait(timeout=0.5)

            self._status["awaitingAck"] = False
            self._status["coolingDown"] = True
            self._sync_cv.notify_all()

            deadline = time.time() + self._post_ack_delay_sec
            while not self._stop_event.is_set() and self._status["running"]:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                self._sync_cv.wait(timeout=min(0.2, remaining))

            self._status["coolingDown"] = False
            self._sync_cv.notify_all()

    def _run_job(
        self,
        episodes: int,
        seed: int | None,
        max_steps: int | None,
        terminate_on_win: bool,
        tensorboard_log_dir: str | None,
        tensorboard_run_name: str | None,
        checkpoint_every_episodes: int,
        checkpoint_dir: str | None,
        checkpoint_prefix: str,
        load_model_path: str | None,
        play_only: bool,
    ) -> None:
        error_message: str | None = None
        try:
            trainer = DQNTrainer(self._rl_config, seed=seed)
            summary = trainer.train(
                episodes=episodes,
                max_steps=max_steps,
                terminate_on_win=terminate_on_win,
                stop_event=self._stop_event,
                on_step=self._on_step,
                on_episode_end=self._on_episode_end,
                tensorboard_log_dir=tensorboard_log_dir,
                tensorboard_run_name=tensorboard_run_name,
                checkpoint_every_episodes=checkpoint_every_episodes,
                checkpoint_dir=checkpoint_dir,
                checkpoint_prefix=checkpoint_prefix,
                load_model_path=load_model_path,
                play_only=play_only,
            )
            self._latest_model_state = summary.get("modelStateDict")
            with self._lock:
                self._status["globalStep"] = int(summary.get("globalStep", self._status["globalStep"]))
                self._status["loss"] = summary.get("lastLoss", self._status["loss"])
                self._status["epsilon"] = summary.get("lastEpsilon", self._status["epsilon"])
                self._status["averageScore"] = float(summary.get("averageScore", self._status["averageScore"]))
                self._status["maxTileSeen"] = max(
                    int(self._status["maxTileSeen"]),
                    int(summary.get("maxTileSeen", 0)),
                )
                self._status["tensorboardEnabled"] = bool(summary.get("tensorboardEnabled", False))
                self._status["tensorboardRunDir"] = summary.get("tensorboardRunDir")
                self._status["tensorboardWarning"] = summary.get("tensorboardWarning")
                self._status["playOnly"] = bool(summary.get("playOnly", self._status["playOnly"]))
                self._status["loadedModelPath"] = summary.get("loadedModelPath")
                self._status["checkpointEveryEpisodes"] = int(
                    summary.get("checkpointEveryEpisodes", self._status["checkpointEveryEpisodes"])
                )
                self._status["checkpointDir"] = summary.get("checkpointDir", self._status["checkpointDir"])
                self._status["checkpointsSaved"] = int(summary.get("checkpointsSaved", self._status["checkpointsSaved"]))
                self._status["latestCheckpointPath"] = summary.get("latestCheckpointPath")
        except Exception as error:  # noqa: BLE001
            error_message = str(error)
        finally:
            with self._sync_cv:
                completed = int(self._status["completedEpisodes"])
                self._status["running"] = False
                self._status["finishedAt"] = time.time()
                self._status["stopped"] = bool(self._status["stopRequested"]) or completed < episodes
                self._status["error"] = error_message
                self._status["awaitingAck"] = False
                self._status["coolingDown"] = False
                self._sync_cv.notify_all()


class GameService:
    """Thread-safe wrapper around a single game instance."""

    DEFAULT_TRAINING_START_DEFAULTS: dict[str, Any] = {
        "episodes": 100,
        "workers": 1,
        "seed": None,
        "maxSteps": None,
        "terminateOnWin": True,
        "syncWithFrontend": False,
        "tensorboardLogDir": TrainingManager.DEFAULT_TENSORBOARD_LOG_DIR,
        "tensorboardRunName": None,
        "checkpointEveryEpisodes": 0,
        "checkpointDir": TrainingManager.DEFAULT_CHECKPOINT_DIR,
        "checkpointPrefix": "dqn_cnn",
        "loadModelPath": None,
        "playOnly": False,
    }

    def __init__(
        self,
        *,
        training_defaults: dict[str, Any] | None = None,
        rl_config: DQNConfig | None = None,
        post_ack_delay_sec: float | None = None,
        default_tensorboard_log_dir: str | None = None,
        default_checkpoint_dir: str | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._game = Game2048()
        self._training_defaults = copy.deepcopy(self.DEFAULT_TRAINING_START_DEFAULTS)
        if training_defaults is not None:
            self._training_defaults.update(training_defaults)
        self._training = TrainingManager(
            rl_config=rl_config,
            post_ack_delay_sec=post_ack_delay_sec,
            default_tensorboard_log_dir=default_tensorboard_log_dir,
            default_checkpoint_dir=default_checkpoint_dir,
        )

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
            animation_grid = self._game.consume_last_animation_grid()
            if animation_grid is not None:
                state["animationGrid"] = animation_grid
            state["moved"] = moved
            return state

    def training_status(self) -> dict[str, Any]:
        status = self._training.status()
        status["trainingDefaults"] = self.training_start_defaults()
        return status

    def training_start_defaults(self) -> dict[str, Any]:
        return copy.deepcopy(self._training_defaults)

    def training_start(
        self,
        *,
        episodes: int,
        workers: int,
        seed: int | None,
        max_steps: int | None,
        terminate_on_win: bool,
        sync_with_frontend: bool = False,
        tensorboard_log_dir: str | None = None,
        tensorboard_run_name: str | None = None,
        checkpoint_every_episodes: int = 0,
        checkpoint_dir: str | None = None,
        checkpoint_prefix: str = "dqn_cnn",
        load_model_path: str | None = None,
        play_only: bool = False,
    ) -> dict[str, Any]:
        return self._training.start(
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

    def training_stop(self) -> dict[str, Any]:
        return self._training.stop()

    def training_step_done(self, frame_id: int) -> dict[str, Any]:
        return self._training.step_done(frame_id)
