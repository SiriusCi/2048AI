#!/usr/bin/env python3
"""CLI entrypoint for headless RL training of the 2048 agent.

Usage examples:
  # Train 100 episodes with default config
  python train.py

  # Train 500 episodes with checkpointing every 50
  python train.py --episodes 500 --checkpoint-every 50

  # Train with custom learning rate and seed
  python train.py --episodes 1000 --lr 0.001 --seed 42

  # Resume training from a checkpoint
  python train.py --episodes 500 --load-model models/2048/reinforce_cnn_ep000100.pt

  # Play-only mode (no gradient updates, just inference)
  python train.py --episodes 10 --load-model models/2048/reinforce_cnn_ep000100.pt --play-only
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

from backend.config import load_runtime_config
from backend.rl import DQNConfig, DQNTrainer


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 2048 RL agent (DQN + CNN) from the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="YAML config path (default: config.yaml)",
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Number of training episodes (overrides config)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Max steps per episode (overrides config)",
    )
    parser.add_argument(
        "--no-terminate-on-win", action="store_true",
        help="Continue playing after reaching 2048",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--gamma", type=float, default=None,
        help="Discount factor (overrides config)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Replay batch size (overrides config)",
    )
    parser.add_argument(
        "--epsilon-start", type=float, default=None,
        help="Starting epsilon for exploration (overrides config)",
    )
    parser.add_argument(
        "--epsilon-end", type=float, default=None,
        help="Final epsilon for exploration (overrides config)",
    )
    parser.add_argument(
        "--epsilon-decay-steps", type=int, default=None,
        help="Steps over which epsilon decays (overrides config)",
    )
    parser.add_argument(
        "--invalid-penalty", type=float, default=None,
        help="Invalid action penalty (overrides config)",
    )
    parser.add_argument(
        "--merge-bonus-scale", type=float, default=None,
        help="Merge value bonus scale (overrides config)",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=None,
        help="Save checkpoint every N episodes (overrides config)",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Checkpoint directory (overrides config)",
    )
    parser.add_argument(
        "--checkpoint-prefix", type=str, default=None,
        help="Checkpoint file prefix (overrides config)",
    )
    parser.add_argument(
        "--tensorboard-dir", type=str, default=None,
        help="TensorBoard log directory (overrides config)",
    )
    parser.add_argument(
        "--tensorboard-run", type=str, default=None,
        help="TensorBoard run name (overrides config)",
    )
    parser.add_argument(
        "--no-tensorboard", action="store_true",
        help="Disable TensorBoard logging",
    )
    parser.add_argument(
        "--load-model", type=str, default=None,
        help="Path to a .pt checkpoint to resume from",
    )
    parser.add_argument(
        "--play-only", action="store_true",
        help="Run inference only (no gradient updates)",
    )
    parser.add_argument(
        "--log-every", type=int, default=1,
        help="Print episode info every N episodes (default: 1)",
    )
    parser.add_argument(
        "--num_envs", type=int, default=8,
        help="",
    )
    return parser.parse_args(argv)


def _make_episode_callback(log_every: int = 1):
    def _on_episode_end(
        episode: int,
        result: dict[str, Any],
        metrics: dict[str, Any],
    ) -> None:
        checkpoint_path = metrics.get("checkpointPath")
        should_log = (episode % log_every == 0) or (episode == 1) or checkpoint_path
        if not should_log:
            return

        score = int(result.get("score", 0))
        max_tile = int(result.get("maxTile", 0))
        steps = int(result.get("steps", 0))
        won = bool(result.get("won", False))
        avg_score = metrics.get("averageScore", 0.0)
        loss = metrics.get("loss")
        epsilon = metrics.get("epsilon")
        global_step = metrics.get("globalStep", 0)
        replay_size = metrics.get("replaySize", 0)

        loss_str = f"{loss:.4f}" if loss is not None else "N/A"
        epsilon_str = f"{epsilon:.4f}" if epsilon is not None else "N/A"
        checkpoint_str = f" | ckpt={checkpoint_path}" if checkpoint_path else ""

        print(
            f"Episode {episode}: score={score}, maxTile={max_tile}, steps={steps}, "
            f"won={won}, avgScore={avg_score:.2f}, loss={loss_str}, "
            f"eps={epsilon_str}, replay={replay_size}, globalStep={global_step}{checkpoint_str}",
            flush=True,
        )
    return _on_episode_end


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    # Load config
    runtime = load_runtime_config(args.config)
    td = runtime["trainingDefaults"]
    rl_raw = runtime["rl"]

    # Resolve parameters (CLI overrides config)
    episodes = args.episodes if args.episodes is not None else int(td["episodes"])
    seed = args.seed if args.seed is not None else td.get("seed")
    max_steps = args.max_steps if args.max_steps is not None else td.get("maxSteps")
    terminate_on_win = not args.no_terminate_on_win and bool(td.get("terminateOnWin", True))

    lr = args.lr if args.lr is not None else float(rl_raw["learningRate"])
    gamma = args.gamma if args.gamma is not None else float(rl_raw["gamma"])
    batch_size = args.batch_size if args.batch_size is not None else int(rl_raw["batchSize"])
    replay_capacity = int(rl_raw["replayCapacity"])
    min_replay_size = int(rl_raw["minReplaySize"])
    target_update_freq = int(rl_raw["targetUpdateFreq"])
    train_freq = int(rl_raw["trainFreq"])
    num_envs = args.num_envs if args.num_envs is not None else int(rl_raw["numEnvs"])
    max_grad_norm = float(rl_raw["maxGradNorm"])
    epsilon_start = args.epsilon_start if args.epsilon_start is not None else float(rl_raw["epsilonStart"])
    epsilon_end = args.epsilon_end if args.epsilon_end is not None else float(rl_raw["epsilonEnd"])
    epsilon_decay_steps = args.epsilon_decay_steps if args.epsilon_decay_steps is not None else int(rl_raw["epsilonDecaySteps"])
    invalid_penalty = args.invalid_penalty if args.invalid_penalty is not None else float(rl_raw["invalidActionPenalty"])
    merge_bonus_scale = args.merge_bonus_scale if args.merge_bonus_scale is not None else float(rl_raw["mergeValueBonusScale"])

    checkpoint_every = args.checkpoint_every if args.checkpoint_every is not None else int(td.get("checkpointEveryEpisodes", 0))
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir is not None else td.get("checkpointDir")
    checkpoint_prefix = args.checkpoint_prefix if args.checkpoint_prefix is not None else str(td.get("checkpointPrefix", "dqn_cnn"))

    tensorboard_dir = None if args.no_tensorboard else (args.tensorboard_dir if args.tensorboard_dir is not None else td.get("tensorboardLogDir"))
    tensorboard_run = args.tensorboard_run if args.tensorboard_run is not None else td.get("tensorboardRunName")

    load_model = args.load_model if args.load_model is not None else td.get("loadModelPath")
    play_only = args.play_only or bool(td.get("playOnly", False))

    # Build RL config
    rl_config = DQNConfig(
        max_exponent=int(rl_raw["maxExponent"]),
        gamma=gamma,
        learning_rate=lr,
        batch_size=batch_size,
        replay_capacity=replay_capacity,
        min_replay_size=min_replay_size,
        target_update_freq=target_update_freq,
        train_freq=train_freq,
        num_envs=num_envs,
        max_grad_norm=max_grad_norm,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
        invalid_action_penalty=invalid_penalty,
        merge_value_bonus_scale=merge_bonus_scale,
    )

    # Print training config summary
    print("=" * 60)
    print("2048 RL Training (DQN + CNN)")
    print("=" * 60)
    print(f"  Config file:          {runtime['configPath']}")
    print(f"  Episodes:             {episodes}")
    print(f"  Seed:                 {seed}")
    print(f"  Max steps/episode:    {max_steps or 'unlimited'}")
    print(f"  Terminate on win:     {terminate_on_win}")
    print(f"  Play only:            {play_only}")
    print(f"  Learning rate:        {lr}")
    print(f"  Gamma:                {gamma}")
    print(f"  Batch size:           {batch_size}")
    print(f"  Replay capacity:      {replay_capacity}")
    print(f"  Min replay size:      {min_replay_size}")
    print(f"  Target update freq:   {target_update_freq}")
    print(f"  Train freq:           {train_freq}")
    print(f"  Num envs:             {num_envs}")
    print(f"  Max grad norm:        {max_grad_norm}")
    print(f"  Epsilon:              {epsilon_start} -> {epsilon_end} over {epsilon_decay_steps} steps")
    print(f"  Invalid penalty:      {invalid_penalty}")
    print(f"  Merge bonus scale:    {merge_bonus_scale}")
    print(f"  Checkpoint every:     {checkpoint_every} episodes")
    print(f"  Checkpoint dir:       {checkpoint_dir or 'N/A'}")
    print(f"  TensorBoard dir:      {tensorboard_dir or 'disabled'}")
    print(f"  Load model:           {load_model or 'N/A'}")
    print("=" * 60)

    # Setup stop event for graceful Ctrl+C
    stop_event = threading.Event()

    def _signal_handler(sig: int, frame: Any) -> None:
        print("\nCtrl+C received, stopping after current episode...", flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)

    # Create trainer and run
    trainer = DQNTrainer(rl_config, seed=seed)

    start_time = time.time()
    summary = trainer.train(
        episodes=episodes,
        max_steps=max_steps,
        terminate_on_win=terminate_on_win,
        stop_event=stop_event,
        on_episode_end=_make_episode_callback(args.log_every),
        tensorboard_log_dir=tensorboard_dir,
        tensorboard_run_name=tensorboard_run,
        checkpoint_every_episodes=checkpoint_every,
        checkpoint_dir=checkpoint_dir,
        checkpoint_prefix=checkpoint_prefix,
        load_model_path=load_model,
        play_only=play_only,
    )
    elapsed = time.time() - start_time

    # Print final summary
    print()
    print("=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"  Completed episodes:   {summary['completedEpisodes']}")
    print(f"  Average score:        {summary['averageScore']:.2f}")
    print(f"  Max tile seen:        {summary['maxTileSeen']}")
    print(f"  Global steps:         {summary['globalStep']}")
    print(f"  Last loss:            {summary['lastLoss']}")
    print(f"  Last epsilon:         {summary['lastEpsilon']}")
    print(f"  Elapsed time:         {elapsed:.1f}s")
    if summary.get("tensorboardRunDir"):
        print(f"  TensorBoard dir:      {summary['tensorboardRunDir']}")
    if summary.get("tensorboardWarning"):
        print(f"  TensorBoard warning:  {summary['tensorboardWarning']}")
    if summary.get("checkpointsSaved", 0) > 0:
        print(f"  Checkpoints saved:    {summary['checkpointsSaved']}")
        print(f"  Latest checkpoint:    {summary['latestCheckpointPath']}")
    if summary.get("loadedModelPath"):
        print(f"  Loaded model from:    {summary['loadedModelPath']}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
