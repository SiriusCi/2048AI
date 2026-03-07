"""Pure-Python headless mode for running 2048 without a browser."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
from typing import Any, Sequence

from .game import Game2048


class Headless2048Env:
    """A minimal environment interface for headless training/simulation."""

    ACTIONS = (0, 1, 2, 3)

    def __init__(
        self,
        *,
        size: int = 4,
        start_tiles: int = 2,
        seed: int | None = None,
        max_steps: int | None = None,
        terminate_on_win: bool = True,
    ) -> None:
        self.game = Game2048(size=size, start_tiles=start_tiles, seed=seed)
        self.max_steps = max_steps
        self.terminate_on_win = terminate_on_win
        self.steps = 0

        if not self.terminate_on_win:
            self.game.keep_going()

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.game.set_seed(seed)

        self.game.reset()
        self.steps = 0

        if not self.terminate_on_win:
            self.game.keep_going()

        return self.observation()

    def legal_actions(self) -> list[int]:
        return list(self.ACTIONS)

    def movable_actions(self) -> list[int]:
        return self.game.legal_moves()

    @staticmethod
    def _encode_log2(board: list[list[int]]) -> list[list[int]]:
        encoded: list[list[int]] = []
        for row in board:
            encoded_row = []
            for value in row:
                if value <= 0:
                    encoded_row.append(0)
                else:
                    encoded_row.append(int(math.log2(value)))
            encoded.append(encoded_row)
        return encoded

    def observation(self) -> dict[str, Any]:
        raw_board = self.game.board()
        encoded_state = self._encode_log2(raw_board)
        return {
            "state": encoded_state,
            "board": encoded_state,
            "rawBoard": raw_board,
            "score": self.game.score,
            "bestScore": self.game.best_score,
            "won": self.game.won,
            "over": self.game.over,
            "steps": self.steps,
            "maxTile": self.game.max_tile(),
            "legalActions": self.legal_actions(),
            "movableActions": self.movable_actions(),
        }

    def step(self, action: int) -> tuple[dict[str, Any], int, bool, bool, dict[str, Any]]:
        if action not in self.ACTIONS:
            raise ValueError("Action must be one of 0, 1, 2, 3")

        previous_score = self.game.score
        moved = self.game.move(action)
        animation_grid = self.game.consume_last_animation_grid()
        if animation_grid is None:
            animation_grid = self.game.current_animation_grid()
        self.steps += 1

        if self.game.won and not self.terminate_on_win and not self.game.keep_playing:
            self.game.keep_going()

        reward = self.game.score - previous_score
        terminated = self.game.over or (self.game.won and self.terminate_on_win)
        truncated = self.max_steps is not None and self.steps >= self.max_steps

        observation = self.observation()
        info = {
            "moved": moved,
            "action": action,
            "score": self.game.score,
            "steps": self.steps,
            "animationGrid": animation_grid,
            "legalActions": observation["legalActions"],
            "movableActions": observation["movableActions"],
        }
        return observation, reward, terminated, truncated, info

    def render(self) -> str:
        board = self.game.board()
        width = max(4, len(str(self.game.max_tile())))
        rows = [
            " ".join(f"{value:>{width}d}" if value else f"{'.':>{width}s}" for value in row)
            for row in board
        ]
        return "\n".join(rows)


def run_random_episode(
    env: Headless2048Env,
    rng: random.Random,
    seed: int | None = None,
) -> dict[str, Any]:
    env.reset(seed=seed)
    terminated = False
    truncated = False
    total_reward = 0

    while not (terminated or truncated):
        action = rng.choice(env.legal_actions())
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

    result = env.observation()
    result["terminated"] = terminated
    result["truncated"] = truncated
    result["totalReward"] = total_reward
    return result


def _run_random_episode_task(
    *,
    episode: int,
    env_seed: int | None,
    action_seed: int | None,
    max_steps: int | None,
    terminate_on_win: bool,
) -> tuple[int, dict[str, Any]]:
    env = Headless2048Env(
        max_steps=max_steps,
        seed=env_seed,
        terminate_on_win=terminate_on_win,
    )
    rng = random.Random(action_seed)
    result = run_random_episode(env, rng, seed=env_seed)
    return episode, result


def _episode_seed(base_seed: int | None, episode: int) -> int | None:
    if base_seed is None:
        return None
    return base_seed + episode - 1


def _action_seed(base_seed: int | None, episode: int) -> int | None:
    if base_seed is None:
        return None
    return (base_seed * 1000003 + episode * 9176) & 0xFFFFFFFF


def run_random_episodes(
    *,
    episodes: int,
    base_seed: int | None,
    max_steps: int | None,
    terminate_on_win: bool,
    workers: int,
) -> list[tuple[int, dict[str, Any]]]:
    if workers <= 1 or episodes == 1:
        env = Headless2048Env(
            max_steps=max_steps,
            seed=base_seed,
            terminate_on_win=terminate_on_win,
        )
        results: list[tuple[int, dict[str, Any]]] = []
        for episode in range(1, episodes + 1):
            env_seed = _episode_seed(base_seed, episode)
            action_seed = _action_seed(base_seed, episode)
            rng = random.Random(action_seed)
            result = run_random_episode(env, rng, seed=env_seed)
            results.append((episode, result))
        return results

    tasks = [
        {
            "episode": episode,
            "env_seed": _episode_seed(base_seed, episode),
            "action_seed": _action_seed(base_seed, episode),
            "max_steps": max_steps,
            "terminate_on_win": terminate_on_win,
        }
        for episode in range(1, episodes + 1)
    ]

    unordered_results: list[tuple[int, dict[str, Any]]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_random_episode_task, **task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            unordered_results.append(future.result())

    unordered_results.sort(key=lambda item: item[0])
    return unordered_results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pure-Python headless 2048 episodes.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max steps per episode")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, use 0 for CPU count)",
    )
    parser.set_defaults(terminate_on_win=True)
    parser.add_argument(
        "--terminate-on-win",
        dest="terminate_on_win",
        action="store_true",
        help="Terminate episodes immediately when a 2048 tile appears (default)",
    )
    parser.add_argument(
        "--no-terminate-on-win",
        dest="terminate_on_win",
        action="store_false",
        help="Continue after reaching 2048",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print episode summaries as JSON lines",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.episodes <= 0:
        raise ValueError("--episodes must be greater than 0")
    if args.workers < 0:
        raise ValueError("--workers must be >= 0")
    if args.workers == 0:
        workers = os.cpu_count() or 1
    else:
        workers = args.workers

    episodes_results = run_random_episodes(
        episodes=args.episodes,
        base_seed=args.seed,
        max_steps=args.max_steps,
        terminate_on_win=args.terminate_on_win,
        workers=workers,
    )
    score_sum = 0
    max_tile_seen = 0

    for episode, result in episodes_results:
        score_sum += result["score"]
        max_tile_seen = max(max_tile_seen, result["maxTile"])

        summary = {
            "episode": episode,
            "score": result["score"],
            "maxTile": result["maxTile"],
            "steps": result["steps"],
            "won": result["won"],
            "over": result["over"],
            "terminated": result["terminated"],
            "truncated": result["truncated"],
        }

        if args.json:
            print(json.dumps(summary, ensure_ascii=True))
        else:
            print(
                "Episode {episode}: score={score}, maxTile={maxTile}, steps={steps}, won={won}".format(
                    **summary
                )
            )

    average_score = score_sum / args.episodes
    footer = {
        "episodes": args.episodes,
        "workers": workers,
        "averageScore": average_score,
        "maxTileSeen": max_tile_seen,
    }

    if args.json:
        print(json.dumps(footer, ensure_ascii=True))
    else:
        print(
            "Summary: episodes={episodes}, workers={workers}, averageScore={averageScore:.2f}, maxTileSeen={maxTileSeen}".format(
                **footer
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
