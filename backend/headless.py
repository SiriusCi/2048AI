"""Pure-Python headless mode for running 2048 without a browser."""

from __future__ import annotations

import argparse
import json
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
        return self.game.legal_moves()

    def observation(self) -> dict[str, Any]:
        return {
            "board": self.game.board(),
            "score": self.game.score,
            "bestScore": self.game.best_score,
            "won": self.game.won,
            "over": self.game.over,
            "steps": self.steps,
            "maxTile": self.game.max_tile(),
            "legalActions": self.legal_actions(),
        }

    def step(self, action: int) -> tuple[dict[str, Any], int, bool, bool, dict[str, Any]]:
        if action not in self.ACTIONS:
            raise ValueError("Action must be one of 0, 1, 2, 3")

        previous_score = self.game.score
        moved = self.game.move(action)
        self.steps += 1

        if self.game.won and not self.terminate_on_win and not self.game.keep_playing:
            self.game.keep_going()

        reward = self.game.score - previous_score
        terminated = self.game.over or (self.game.won and self.terminate_on_win)
        truncated = self.max_steps is not None and self.steps >= self.max_steps

        observation = self.observation()
        info = {
            "moved": moved,
            "score": self.game.score,
            "steps": self.steps,
            "legalActions": observation["legalActions"],
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
        legal_actions = env.legal_actions()
        if not legal_actions:
            break

        action = rng.choice(legal_actions)
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

    result = env.observation()
    result["terminated"] = terminated
    result["truncated"] = truncated
    result["totalReward"] = total_reward
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pure-Python headless 2048 episodes.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max steps per episode")
    parser.add_argument(
        "--terminate-on-win",
        action="store_true",
        help="Terminate episodes immediately when a 2048 tile appears",
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

    env = Headless2048Env(
        max_steps=args.max_steps,
        seed=args.seed,
        terminate_on_win=args.terminate_on_win,
    )
    rng = random.Random(args.seed)

    score_sum = 0
    max_tile_seen = 0

    for episode in range(1, args.episodes + 1):
        episode_seed = None if args.seed is None else args.seed + episode - 1
        result = run_random_episode(env, rng, seed=episode_seed)
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
        "averageScore": average_score,
        "maxTileSeen": max_tile_seen,
    }

    if args.json:
        print(json.dumps(footer, ensure_ascii=True))
    else:
        print(
            "Summary: episodes={episodes}, averageScore={averageScore:.2f}, maxTileSeen={maxTileSeen}".format(
                **footer
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
