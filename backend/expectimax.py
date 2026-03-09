"""Expectimax AI for 2048 — standalone script.

Usage:
    python -m backend.expectimax [--depth 5] [--episodes 10] [--no-terminate-on-win]

The algorithm treats player moves as MAX nodes and random tile spawns as
CHANCE nodes.  A hand-crafted evaluation function rewards:
  - empty cells         (survival)
  - monotonic rows/cols (structure)
  - smoothness          (adjacent-tile similarity)
  - merge potential

No training required — pure search + heuristics.

**Performance**: uses a 64-bit bitboard (4 bits per cell, log2 values) with
precomputed row-move / row-score / row-heuristic lookup tables so each board
move is 4 table lookups and evaluation is 8 table lookups (rows + cols).
Transposition table caches subtree values.  Depth 3–4 runs < 50 ms/move.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random as _random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

# ===================================================================
# Bitboard helpers
# ===================================================================
# Board = 64-bit int.  Cell (row, col) uses bits [row*16 + col*4 .. +3].
# Cell value = log2(tile): 0=empty, 1="2", 2="4", … 15="32768".
# A "row" is 16 bits (4 nibbles).

ROW_MASK = 0xFFFF
NIB_MASK = 0xF


def _pack_board(rows: list[list[int]]) -> int:
    b = 0
    for r in range(4):
        for c in range(4):
            v = rows[r][c]
            nib = int(math.log2(v)) if v > 0 else 0
            b |= nib << (r * 16 + c * 4)
    return b


def _unpack_board(b: int) -> list[list[int]]:
    out: list[list[int]] = []
    for r in range(4):
        row: list[int] = []
        for c in range(4):
            nib = (b >> (r * 16 + c * 4)) & NIB_MASK
            row.append((1 << nib) if nib else 0)
        out.append(row)
    return out


def _get_cell(b: int, r: int, c: int) -> int:
    return (b >> (r * 16 + c * 4)) & NIB_MASK


def _set_cell(b: int, r: int, c: int, v: int) -> int:
    shift = r * 16 + c * 4
    return (b & ~(NIB_MASK << shift)) | ((v & NIB_MASK) << shift)


def _get_row(b: int, r: int) -> int:
    return (b >> (r * 16)) & ROW_MASK


def _set_row(b: int, r: int, row16: int) -> int:
    shift = r * 16
    return (b & ~(ROW_MASK << shift)) | ((row16 & ROW_MASK) << shift)


def _reverse_row(row16: int) -> int:
    return (
        ((row16 >> 12) & NIB_MASK)
        | (((row16 >> 8) & NIB_MASK) << 4)
        | (((row16 >> 4) & NIB_MASK) << 8)
        | ((row16 & NIB_MASK) << 12)
    )


def _transpose(b: int) -> int:
    """Transpose 4×4 nibble matrix (unrolled for speed)."""
    r0 = b & ROW_MASK
    r1 = (b >> 16) & ROW_MASK
    r2 = (b >> 32) & ROW_MASK
    r3 = (b >> 48) & ROW_MASK

    t0 = ((r0) & 0xF) | (((r1) & 0xF) << 4) | (((r2) & 0xF) << 8) | (((r3) & 0xF) << 12)
    t1 = (((r0 >> 4) & 0xF)) | (((r1 >> 4) & 0xF) << 4) | (((r2 >> 4) & 0xF) << 8) | (((r3 >> 4) & 0xF) << 12)
    t2 = (((r0 >> 8) & 0xF)) | (((r1 >> 8) & 0xF) << 4) | (((r2 >> 8) & 0xF) << 8) | (((r3 >> 8) & 0xF) << 12)
    t3 = (((r0 >> 12) & 0xF)) | (((r1 >> 12) & 0xF) << 4) | (((r2 >> 12) & 0xF) << 8) | (((r3 >> 12) & 0xF) << 12)

    return t0 | (t1 << 16) | (t2 << 32) | (t3 << 48)


# ===================================================================
# Precomputed move tables  (built once at import time)
# ===================================================================

_ROW_LEFT:        list[int] = [0] * 65536
_ROW_RIGHT:       list[int] = [0] * 65536
_ROW_SCORE_LEFT:  list[int] = [0] * 65536
_ROW_SCORE_RIGHT: list[int] = [0] * 65536


def _slide_left(line: list[int]) -> tuple[list[int], int]:
    tiles = [v for v in line if v != 0]
    merged: list[int] = []
    score = 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            new_val = tiles[i] + 1
            merged.append(new_val)
            score += 1 << new_val
            i += 2
        else:
            merged.append(tiles[i])
            i += 1
    while len(merged) < 4:
        merged.append(0)
    return merged, score


def _nibs_to_row(c: list[int]) -> int:
    return c[0] | (c[1] << 4) | (c[2] << 8) | (c[3] << 12)


def _init_move_tables() -> None:
    for row16 in range(65536):
        c0 = row16 & NIB_MASK
        c1 = (row16 >> 4) & NIB_MASK
        c2 = (row16 >> 8) & NIB_MASK
        c3 = (row16 >> 12) & NIB_MASK

        # Left
        left_result, left_score = _slide_left([c0, c1, c2, c3])
        _ROW_LEFT[row16] = _nibs_to_row(left_result)
        _ROW_SCORE_LEFT[row16] = left_score

        # Right = reverse → slide left → reverse
        right_result, right_score = _slide_left([c3, c2, c1, c0])
        right_result.reverse()
        _ROW_RIGHT[row16] = _nibs_to_row(right_result)
        _ROW_SCORE_RIGHT[row16] = right_score


_init_move_tables()


# ===================================================================
# Board moves  (0=Up, 1=Right, 2=Down, 3=Left)
# ===================================================================

def _move_left(b: int) -> tuple[int, int]:
    n = 0; s = 0
    r0 = b & ROW_MASK;            n |= _ROW_LEFT[r0];              s += _ROW_SCORE_LEFT[r0]
    r1 = (b >> 16) & ROW_MASK;    n |= _ROW_LEFT[r1] << 16;       s += _ROW_SCORE_LEFT[r1]
    r2 = (b >> 32) & ROW_MASK;    n |= _ROW_LEFT[r2] << 32;       s += _ROW_SCORE_LEFT[r2]
    r3 = (b >> 48) & ROW_MASK;    n |= _ROW_LEFT[r3] << 48;       s += _ROW_SCORE_LEFT[r3]
    return n, s


def _move_right(b: int) -> tuple[int, int]:
    n = 0; s = 0
    r0 = b & ROW_MASK;            n |= _ROW_RIGHT[r0];             s += _ROW_SCORE_RIGHT[r0]
    r1 = (b >> 16) & ROW_MASK;    n |= _ROW_RIGHT[r1] << 16;      s += _ROW_SCORE_RIGHT[r1]
    r2 = (b >> 32) & ROW_MASK;    n |= _ROW_RIGHT[r2] << 32;      s += _ROW_SCORE_RIGHT[r2]
    r3 = (b >> 48) & ROW_MASK;    n |= _ROW_RIGHT[r3] << 48;      s += _ROW_SCORE_RIGHT[r3]
    return n, s


def _move_up(b: int) -> tuple[int, int]:
    t = _transpose(b)
    n, s = _move_left(t)
    return _transpose(n), s


def _move_down(b: int) -> tuple[int, int]:
    t = _transpose(b)
    n, s = _move_right(t)
    return _transpose(n), s


_MOVE_FN = [_move_up, _move_right, _move_down, _move_left]


def do_move(b: int, d: int) -> tuple[int, int] | None:
    n, s = _MOVE_FN[d](b)
    return None if n == b else (n, s)


# ===================================================================
# Board queries
# ===================================================================

def _empty_cells(b: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(4):
        rr = (b >> (r * 16)) & ROW_MASK
        if not (rr & 0x000F): cells.append((r, 0))
        if not (rr & 0x00F0): cells.append((r, 1))
        if not (rr & 0x0F00): cells.append((r, 2))
        if not (rr & 0xF000): cells.append((r, 3))
    return cells


def _count_empty(b: int) -> int:
    n = 0
    tmp = b
    for _ in range(16):
        if not (tmp & NIB_MASK):
            n += 1
        tmp >>= 4
    return n


def _max_tile(b: int) -> int:
    m = 0
    tmp = b
    for _ in range(16):
        nib = tmp & NIB_MASK
        if nib > m:
            m = nib
        tmp >>= 4
    return (1 << m) if m else 0


# ===================================================================
# Fully table-driven evaluation  (8 lookups per board)
# ===================================================================

_HEUR: list[float] = [0.0] * 65536


def _init_heur_table() -> None:
    W_EMPTY  = 270.0
    W_MONO   = 47.0
    W_SMOOTH = 10.0
    W_MERGE  = 20.0
    W_EDGE   = 5.0
    W_MAX    = 5.0

    for row16 in range(65536):
        c0 = row16 & NIB_MASK
        c1 = (row16 >> 4) & NIB_MASK
        c2 = (row16 >> 8) & NIB_MASK
        c3 = (row16 >> 12) & NIB_MASK
        c = (c0, c1, c2, c3)

        score = 0.0

        # Empty cells
        for v in c:
            if v == 0:
                score += W_EMPTY

        # Monotonicity (best of left-to-right increasing or decreasing)
        inc = dec = 0.0
        for i in range(3):
            a, b = c[i], c[i + 1]
            if a > b:
                dec += b - a  # negative contribution
            elif b > a:
                inc += a - b
        score += W_MONO * max(inc, dec)

        # Smoothness (penalty for neighbour value differences)
        for i in range(3):
            if c[i] and c[i + 1]:
                score -= W_SMOOTH * abs(c[i] - c[i + 1])

        # Merge potential (reward adjacent equal tiles)
        for i in range(3):
            if c[i] != 0 and c[i] == c[i + 1]:
                score += W_MERGE * c[i]

        # Edge bonus (reward large tiles at row edges)
        score += W_EDGE * (c0 + c3)

        # Max tile bonus
        score += W_MAX * max(c)

        _HEUR[row16] = score


_init_heur_table()


def evaluate(b: int) -> float:
    t = _transpose(b)
    return (
        _HEUR[b & ROW_MASK]
        + _HEUR[(b >> 16) & ROW_MASK]
        + _HEUR[(b >> 32) & ROW_MASK]
        + _HEUR[(b >> 48) & ROW_MASK]
        + _HEUR[t & ROW_MASK]
        + _HEUR[(t >> 16) & ROW_MASK]
        + _HEUR[(t >> 32) & ROW_MASK]
        + _HEUR[(t >> 48) & ROW_MASK]
    )


# ===================================================================
# Expectimax search  (with transposition table)
# ===================================================================

_tt: dict[int, float] = {}   # transposition table: (board ^ depth_tag) → value
_TT_LIMIT = 800_000


def _chance_node(b: int, depth: int) -> float:
    empties = _empty_cells(b)
    if not empties:
        return evaluate(b)

    total = 0.0
    n = len(empties)
    for r, c in empties:
        shift = r * 16 + c * 4
        b2 = b | (1 << shift)   # place tile "2" (nib=1)
        b4 = b | (2 << shift)   # place tile "4" (nib=2)
        total += 0.9 * _max_node(b2, depth - 1) + 0.1 * _max_node(b4, depth - 1)
    return total / n


def _max_node(b: int, depth: int) -> float:
    if depth <= 0:
        return evaluate(b)

    key = b ^ (depth * 0x1234567890ABCDEF)
    cached = _tt.get(key)
    if cached is not None:
        return cached

    best = -1e18
    any_legal = False
    for d in range(4):
        result = do_move(b, d)
        if result is None:
            continue
        any_legal = True
        val = _chance_node(result[0], depth)
        if val > best:
            best = val

    val = best if any_legal else evaluate(b)
    if len(_tt) < _TT_LIMIT:
        _tt[key] = val
    return val


def best_move(b: int, depth: int = 4) -> int:
    best_val = -1e18
    best_dir = 0
    for d in range(4):
        result = do_move(b, d)
        if result is None:
            continue
        val = _chance_node(result[0], depth)
        if val > best_val:
            best_val = val
            best_dir = d
    return best_dir


# ===================================================================
# Self-contained game loop
# ===================================================================

def _spawn_tile(b: int, rng: _random.Random) -> int:
    empties = _empty_cells(b)
    if not empties:
        return b
    r, c = rng.choice(empties)
    nib = 1 if rng.random() < 0.9 else 2
    return _set_cell(b, r, c, nib)


DIRECTION_NAMES = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}


def _render_board(b: int, score: int, step: int) -> str:
    board = _unpack_board(b)
    mt = _max_tile(b)
    width = max(5, len(str(mt)) + 1)
    header = f"  Score: {score}   Step: {step}   MaxTile: {mt}"
    lines = [header, "  " + "-" * (width * 4 + 5)]
    for row in board:
        cells = " ".join(f"{v:>{width}}" if v else f"{'·':>{width}}" for v in row)
        lines.append("  " + cells)
    lines.append("")
    return "\n".join(lines)


def play_one_game(
    depth: int = 4,
    seed: int | None = None,
    verbose: bool = True,
    terminate_on_win: bool = False,
) -> dict[str, Any]:
    global _tt
    _tt = {}

    rng = _random.Random(seed)
    b = 0
    b = _spawn_tile(b, rng)
    b = _spawn_tile(b, rng)
    score = 0
    step = 0
    won = False

    # Record initial board and every action for replay
    initial_board = _unpack_board(b)
    actions: list[int] = []

    while True:
        if verbose:
            sys.stdout.write(_render_board(b, score, step))
            sys.stdout.flush()

        if do_move(b, 0) is None and do_move(b, 1) is None and \
           do_move(b, 2) is None and do_move(b, 3) is None:
            break

        mt = _max_tile(b)
        if terminate_on_win and mt >= 2048 and not won:
            won = True
            break

        # Adaptive depth: use higher depth when board is crowded (more dangerous)
        empties = _count_empty(b)
        if empties >= 8:
            d = max(1, depth - 2)
        elif empties >= 5:
            d = max(2, depth - 1)
        else:
            d = depth

        t0 = time.perf_counter()
        direction = best_move(b, d)
        dt = time.perf_counter() - t0

        result = do_move(b, direction)
        if result is None:
            break

        actions.append(direction)
        b, gained = result
        score += gained
        b = _spawn_tile(b, rng)
        step += 1

        if _max_tile(b) >= 2048:
            won = True

        if verbose:
            print(f"  -> {DIRECTION_NAMES[direction]}  (+{gained})  d={d}  [{dt*1000:.0f} ms]")

    if verbose:
        sys.stdout.write(_render_board(b, score, step))
        tag = "WON!" if won else "GAME OVER"
        mt = _max_tile(b)
        print(f"  *** {tag} ***  Score={score}  MaxTile={mt}  Steps={step}\n")

    return {
        "score": score,
        "maxTile": _max_tile(b),
        "steps": step,
        "won": won,
        "seed": seed,
        "depth": depth,
        "actions": actions,
        "initialBoard": initial_board,
    }


# ===================================================================
# CLI
# ===================================================================

# ===================================================================
# Replay file I/O
# ===================================================================

DEFAULT_REPLAY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "replays")


def save_replay(
    result: dict[str, Any],
    replay_dir: str | Path = DEFAULT_REPLAY_DIR,
    filename: str | None = None,
) -> str:
    """Save a game result (with actions) to a JSON replay file. Returns the path."""
    replay_dir = Path(replay_dir)
    replay_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if filename is None:
        score = result.get("score", 0)
        mt = result.get("maxTile", 0)
        filename = f"expectimax_{ts}_s{score}_t{mt}.json"

    record = {
        "algorithm": "expectimax",
        "depth": result.get("depth"),
        "seed": result.get("seed"),
        "actions": result.get("actions", []),
        "score": result.get("score", 0),
        "maxTile": result.get("maxTile", 0),
        "steps": result.get("steps", 0),
        "won": result.get("won", False),
        "timestamp": ts,
    }

    path = replay_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
    return str(path)


def load_replay(path: str | Path) -> dict[str, Any]:
    """Load a replay JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_replays(replay_dir: str | Path = DEFAULT_REPLAY_DIR) -> list[dict[str, Any]]:
    """List available replay files with metadata."""
    replay_dir = Path(replay_dir)
    if not replay_dir.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for p in sorted(replay_dir.glob("*.json"), reverse=True):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            items.append({
                "filename": p.name,
                "algorithm": data.get("algorithm", "unknown"),
                "score": data.get("score", 0),
                "maxTile": data.get("maxTile", 0),
                "steps": data.get("steps", 0),
                "won": data.get("won", False),
                "depth": data.get("depth"),
                "seed": data.get("seed"),
                "timestamp": data.get("timestamp", ""),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return items


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Expectimax AI for 2048")
    p.add_argument("--depth", type=int, default=4, help="Search depth (default 4)")
    p.add_argument("--episodes", type=int, default=1, help="Number of games to play")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--quiet", action="store_true", help="Suppress per-step output")
    p.add_argument("--record", action="store_true", help="Save replay JSON to replays/ dir")
    p.add_argument("--replay-dir", type=str, default=DEFAULT_REPLAY_DIR,
                    help="Directory for replay files (default: replays/)")
    p.add_argument("--no-terminate-on-win", dest="terminate_on_win",
                    action="store_false", default=False,
                    help="Continue playing past 2048 (default behaviour)")
    p.add_argument("--terminate-on-win", dest="terminate_on_win",
                    action="store_true", help="Stop when 2048 tile is reached")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    scores: list[int] = []
    max_tiles: list[int] = []

    for ep in range(1, args.episodes + 1):
        ep_seed = (args.seed + ep - 1) if args.seed is not None else None
        verbose = not args.quiet
        if verbose:
            print(f"{'='*50}")
            print(f"  Episode {ep}/{args.episodes}  (depth={args.depth})")
            print(f"{'='*50}")

        result = play_one_game(
            depth=args.depth,
            seed=ep_seed,
            verbose=verbose,
            terminate_on_win=args.terminate_on_win,
        )
        scores.append(result["score"])
        max_tiles.append(result["maxTile"])

        if args.record:
            path = save_replay(result, replay_dir=args.replay_dir)
            print(f"  Replay saved: {path}")

        if not verbose:
            print(
                f"Episode {ep}: score={result['score']}, "
                f"maxTile={result['maxTile']}, steps={result['steps']}, "
                f"won={result['won']}"
            )

    if args.episodes > 1:
        avg = sum(scores) / len(scores)
        print(f"\n{'='*50}")
        print(f"  {args.episodes} episodes  |  depth={args.depth}")
        print(f"  avgScore = {avg:.0f}")
        print(f"  maxTile  = {max(max_tiles)}")
        print(f"  scores   = {scores}")
        print(f"{'='*50}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
