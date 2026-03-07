"""Core 2048 gameplay logic."""

from __future__ import annotations

import copy
import random
from typing import Any


class Game2048:
    """A Python port of the original 2048 game logic."""

    DIRECTIONS = {
        0: (0, -1),  # Up
        1: (1, 0),   # Right
        2: (0, 1),   # Down
        3: (-1, 0),  # Left
    }

    def __init__(self, size: int = 4, start_tiles: int = 2, seed: int | None = None) -> None:
        self.size = size
        self.start_tiles = start_tiles
        self.best_score = 0
        self._rng = random.Random()
        self.set_seed(seed)
        self.reset()

    def set_seed(self, seed: int | None) -> None:
        self._rng.seed(seed)

    def reset(self) -> None:
        self.grid = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.tile_ids = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.score = 0
        self.over = False
        self.won = False
        self.keep_playing = False
        self._next_tile_id = 1
        self._last_animation_grid: dict[str, Any] | None = None
        self._add_start_tiles()

    def keep_going(self) -> None:
        self.keep_playing = True

    def is_game_terminated(self) -> bool:
        return self.over or (self.won and not self.keep_playing)

    def board(self) -> list[list[int]]:
        return [[self.grid[x][y] for x in range(self.size)] for y in range(self.size)]

    def max_tile(self) -> int:
        return max((value for column in self.grid for value in column), default=0)

    def _add_start_tiles(self) -> None:
        for _ in range(self.start_tiles):
            self._add_random_tile()

    def _new_tile_id(self) -> int:
        tile_id = self._next_tile_id
        self._next_tile_id += 1
        return tile_id

    def _available_cells(self) -> list[tuple[int, int]]:
        cells: list[tuple[int, int]] = []
        for x in range(self.size):
            for y in range(self.size):
                if self.grid[x][y] == 0:
                    cells.append((x, y))
        return cells

    def _cells_available(self) -> bool:
        return bool(self._available_cells())

    def _within_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size

    def _cell_content(self, x: int, y: int) -> int:
        if not self._within_bounds(x, y):
            return 0
        return self.grid[x][y]

    def _add_random_tile(self) -> dict[str, Any] | None:
        cells = self._available_cells()
        if not cells:
            return None
        x, y = self._rng.choice(cells)
        value = 2 if self._rng.random() < 0.9 else 4
        tile_id = self._new_tile_id()
        self.grid[x][y] = value
        self.tile_ids[x][y] = tile_id
        return {"id": tile_id, "value": value, "position": {"x": x, "y": y}}

    def _build_traversals(self, vector: tuple[int, int]) -> tuple[list[int], list[int]]:
        xs = list(range(self.size))
        ys = list(range(self.size))
        if vector[0] == 1:
            xs.reverse()
        if vector[1] == 1:
            ys.reverse()
        return xs, ys

    def _find_farthest_position(
        self, x: int, y: int, vector: tuple[int, int]
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        previous_x, previous_y = x, y
        while True:
            next_x = previous_x + vector[0]
            next_y = previous_y + vector[1]
            if not self._within_bounds(next_x, next_y):
                return (previous_x, previous_y), (next_x, next_y)
            if self._cell_content(next_x, next_y) != 0:
                return (previous_x, previous_y), (next_x, next_y)
            previous_x, previous_y = next_x, next_y

    def _tile_matches_available(self) -> bool:
        for x in range(self.size):
            for y in range(self.size):
                value = self.grid[x][y]
                if value == 0:
                    continue
                for dx, dy in self.DIRECTIONS.values():
                    nx, ny = x + dx, y + dy
                    if self._within_bounds(nx, ny) and self.grid[nx][ny] == value:
                        return True
        return False

    def moves_available(self) -> bool:
        return self._cells_available() or self._tile_matches_available()

    def _snapshot(self) -> dict[str, Any]:
        return {
            "grid": [column[:] for column in self.grid],
            "tile_ids": [column[:] for column in self.tile_ids],
            "score": self.score,
            "over": self.over,
            "won": self.won,
            "keep_playing": self.keep_playing,
            "best_score": self.best_score,
            "rng_state": self._rng.getstate(),
            "next_tile_id": self._next_tile_id,
            "last_animation_grid": copy.deepcopy(self._last_animation_grid),
        }

    def _restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.grid = [column[:] for column in snapshot["grid"]]
        self.tile_ids = [column[:] for column in snapshot["tile_ids"]]
        self.score = snapshot["score"]
        self.over = snapshot["over"]
        self.won = snapshot["won"]
        self.keep_playing = snapshot["keep_playing"]
        self.best_score = snapshot["best_score"]
        self._rng.setstate(snapshot["rng_state"])
        self._next_tile_id = snapshot["next_tile_id"]
        self._last_animation_grid = copy.deepcopy(snapshot["last_animation_grid"])

    def _empty_animation_cells(self) -> list[list[dict[str, Any] | None]]:
        return [[None for _ in range(self.size)] for _ in range(self.size)]

    def _build_static_animation_grid(self) -> dict[str, Any]:
        cells = self._empty_animation_cells()
        for x in range(self.size):
            for y in range(self.size):
                value = self.grid[x][y]
                if value == 0:
                    continue
                cells[x][y] = {
                    "position": {"x": x, "y": y},
                    "value": value,
                    "previousPosition": {"x": x, "y": y},
                }
        return {"size": self.size, "cells": cells}

    def current_animation_grid(self) -> dict[str, Any]:
        return self._build_static_animation_grid()

    def consume_last_animation_grid(self) -> dict[str, Any] | None:
        if self._last_animation_grid is None:
            return None
        snapshot = copy.deepcopy(self._last_animation_grid)
        self._last_animation_grid = None
        return snapshot

    def can_move(self, direction: int) -> bool:
        snapshot = self._snapshot()
        try:
            return self.move(direction)
        finally:
            self._restore_snapshot(snapshot)

    def legal_moves(self) -> list[int]:
        return [direction for direction in self.DIRECTIONS if self.can_move(direction)]

    def move(self, direction: int) -> bool:
        if direction not in self.DIRECTIONS:
            raise ValueError("Direction must be 0, 1, 2 or 3")

        if self.is_game_terminated():
            self._last_animation_grid = self._build_static_animation_grid()
            return False

        vector = self.DIRECTIONS[direction]
        traversals_x, traversals_y = self._build_traversals(vector)
        moved = False
        merged_positions: set[tuple[int, int]] = set()
        origins: dict[int, tuple[int, int]] = {}
        start_values: dict[int, int] = {}
        current_positions: dict[int, tuple[int, int]] = {}
        merge_records: dict[int, dict[str, Any]] = {}

        for x in range(self.size):
            for y in range(self.size):
                tile_id = self.tile_ids[x][y]
                value = self.grid[x][y]
                if tile_id == 0 or value == 0:
                    continue
                origins[tile_id] = (x, y)
                start_values[tile_id] = value
                current_positions[tile_id] = (x, y)

        for x in traversals_x:
            for y in traversals_y:
                value = self.grid[x][y]
                if value == 0:
                    continue
                tile_id = self.tile_ids[x][y]

                (fx, fy), (nx, ny) = self._find_farthest_position(x, y, vector)

                can_merge = (
                    self._within_bounds(nx, ny)
                    and self.grid[nx][ny] == value
                    and (nx, ny) not in merged_positions
                )

                if can_merge:
                    target_tile_id = self.tile_ids[nx][ny]
                    self.grid[x][y] = 0
                    self.tile_ids[x][y] = 0
                    self.grid[nx][ny] = value * 2
                    merged_tile_id = self._new_tile_id()
                    self.tile_ids[nx][ny] = merged_tile_id
                    merged_positions.add((nx, ny))
                    self.score += value * 2
                    if value * 2 == 2048:
                        self.won = True
                    moved = True
                    current_positions.pop(tile_id, None)
                    current_positions.pop(target_tile_id, None)
                    current_positions[merged_tile_id] = (nx, ny)
                    merge_records[merged_tile_id] = {
                        "fromIds": [tile_id, target_tile_id],
                        "to": {"x": nx, "y": ny},
                        "value": value * 2,
                    }
                elif (fx, fy) != (x, y):
                    self.grid[x][y] = 0
                    self.tile_ids[x][y] = 0
                    self.grid[fx][fy] = value
                    self.tile_ids[fx][fy] = tile_id
                    current_positions[tile_id] = (fx, fy)
                    moved = True

        spawned_tile = None
        if moved:
            spawned_tile = self._add_random_tile()
            if spawned_tile is not None:
                pos = spawned_tile["position"]
                current_positions[spawned_tile["id"]] = (pos["x"], pos["y"])
            if not self.moves_available():
                self.over = True
            if self.score > self.best_score:
                self.best_score = self.score

        animation_cells = self._empty_animation_cells()
        for tile_id, (x, y) in current_positions.items():
            value = self.grid[x][y]
            if value == 0:
                continue

            if tile_id in merge_records:
                merge_record = merge_records[tile_id]
                merged_from_tiles: list[dict[str, Any]] = []
                for source_id in merge_record["fromIds"]:
                    source_origin = origins.get(source_id, (x, y))
                    source_value = start_values.get(source_id, value // 2)
                    source_tile = {
                        "position": {"x": x, "y": y},
                        "value": source_value,
                        "previousPosition": {"x": source_origin[0], "y": source_origin[1]},
                    }
                    merged_from_tiles.append(source_tile)

                animation_cells[x][y] = {
                    "position": {"x": x, "y": y},
                    "value": value,
                    "mergedFrom": merged_from_tiles,
                }
                continue

            tile_state: dict[str, Any] = {
                "position": {"x": x, "y": y},
                "value": value,
                "previousPosition": {"x": x, "y": y},
            }
            origin = origins.get(tile_id)
            if origin is not None and origin != (x, y):
                tile_state["previousPosition"] = {"x": origin[0], "y": origin[1]}
            animation_cells[x][y] = tile_state

        if spawned_tile is not None:
            pos = spawned_tile["position"]
            x = pos["x"]
            y = pos["y"]
            # Spawn tiles should appear as "new" with no previousPosition.
            animation_cells[x][y] = {
                "position": {"x": x, "y": y},
                "value": self.grid[x][y],
                "isNew": True,
            }

        self._last_animation_grid = {"size": self.size, "cells": animation_cells}
        return moved

    def _serialize_grid(self) -> dict[str, Any]:
        cells: list[list[dict[str, Any] | None]] = []
        for x in range(self.size):
            column: list[dict[str, Any] | None] = []
            for y in range(self.size):
                value = self.grid[x][y]
                if value == 0:
                    column.append(None)
                else:
                    column.append(
                        {
                            "position": {"x": x, "y": y},
                            "value": value,
                        }
                    )
            cells.append(column)
        return {"size": self.size, "cells": cells}

    def serialize_state(self) -> dict[str, Any]:
        return {
            "grid": self._serialize_grid(),
            "score": self.score,
            "over": self.over,
            "won": self.won,
            "keepPlaying": self.keep_playing,
            "bestScore": self.best_score,
            "terminated": self.is_game_terminated(),
        }
