"""Thread-safe service wrappers for game operations."""

from __future__ import annotations

import threading
from typing import Any

from .game import Game2048


class GameService:
    """Thread-safe wrapper around a single game instance."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._game = Game2048()

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
