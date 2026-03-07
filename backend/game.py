"""Core 2048 gameplay logic."""

from __future__ import annotations

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

    def __init__(self, size: int = 4, start_tiles: int = 2) -> None:
        self.size = size
        self.start_tiles = start_tiles
        self.best_score = 0
        self._rng = random.Random()
        self.reset()

    def reset(self) -> None:
        self.grid = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.score = 0
        self.over = False
        self.won = False
        self.keep_playing = False
        self._add_start_tiles()

    def keep_going(self) -> None:
        self.keep_playing = True

    def is_game_terminated(self) -> bool:
        return self.over or (self.won and not self.keep_playing)

    def _add_start_tiles(self) -> None:
        for _ in range(self.start_tiles):
            self._add_random_tile()

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

    def _add_random_tile(self) -> None:
        cells = self._available_cells()
        if not cells:
            return
        x, y = self._rng.choice(cells)
        self.grid[x][y] = 2 if self._rng.random() < 0.9 else 4

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

    def move(self, direction: int) -> bool:
        if direction not in self.DIRECTIONS:
            raise ValueError("Direction must be 0, 1, 2 or 3")

        if self.is_game_terminated():
            return False

        vector = self.DIRECTIONS[direction]
        traversals_x, traversals_y = self._build_traversals(vector)
        moved = False
        merged_positions: set[tuple[int, int]] = set()

        for x in traversals_x:
            for y in traversals_y:
                value = self.grid[x][y]
                if value == 0:
                    continue

                (fx, fy), (nx, ny) = self._find_farthest_position(x, y, vector)

                can_merge = (
                    self._within_bounds(nx, ny)
                    and self.grid[nx][ny] == value
                    and (nx, ny) not in merged_positions
                )

                if can_merge:
                    self.grid[x][y] = 0
                    self.grid[nx][ny] = value * 2
                    merged_positions.add((nx, ny))
                    self.score += value * 2
                    if value * 2 == 2048:
                        self.won = True
                    moved = True
                elif (fx, fy) != (x, y):
                    self.grid[x][y] = 0
                    self.grid[fx][fy] = value
                    moved = True

        if moved:
            self._add_random_tile()
            if not self.moves_available():
                self.over = True
            if self.score > self.best_score:
                self.best_score = self.score

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
