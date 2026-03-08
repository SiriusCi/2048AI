"""RL training components for 2048 using CNN + one-hot encoding."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.distributions import Categorical

from .headless import Headless2048Env

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional runtime dependency
    SummaryWriter = None  # type: ignore[assignment]


@dataclass
class ReinforceCnnConfig:
    """Configuration for a simple CNN policy-gradient trainer."""

    max_exponent: int = 15
    gamma: float = 0.99
    learning_rate: float = 3e-4
    entropy_coef: float = 1e-3
    invalid_action_penalty: float = -1.0
    merge_value_bonus_scale: float = 1.0


class OneHotCnnPolicyNet(nn.Module):
    """Shallow CNN policy net for 4x4 board with no padding."""

    def __init__(self, in_channels: int, num_actions: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class ReinforceCnnTrainer:
    """REINFORCE trainer with one-hot encoding and shallow CNN policy."""

    def __init__(self, config: ReinforceCnnConfig, *, seed: int | None = None) -> None:
        self.config = config
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)

        self.device = torch.device("cpu")
        self.channels = self.config.max_exponent + 1
        self.policy_net = OneHotCnnPolicyNet(self.channels).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)

        self.global_step = 0
        self.last_loss: float | None = None
        self.last_entropy: float | None = None

    def _load_model_weights(self, model_path: str) -> str:
        checkpoint_path = Path(model_path).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model file not found: {checkpoint_path}")

        payload = torch.load(str(checkpoint_path), map_location=self.device)
        if isinstance(payload, dict) and "modelStateDict" in payload:
            state_dict = payload["modelStateDict"]
            loaded_global_step = payload.get("globalStep")
            if loaded_global_step is not None:
                try:
                    self.global_step = int(loaded_global_step)
                except (TypeError, ValueError):
                    pass
            optimizer_state = payload.get("optimizerStateDict")
            if optimizer_state is not None:
                try:
                    self.optimizer.load_state_dict(optimizer_state)
                except Exception:  # noqa: BLE001 - best-effort restore
                    pass
        elif isinstance(payload, dict):
            state_dict = payload
        else:
            raise ValueError("Unsupported model file format.")

        self.policy_net.load_state_dict(state_dict, strict=True)
        return str(checkpoint_path)

    def _save_checkpoint(
        self,
        *,
        checkpoint_path: Path,
        episode: int,
        average_score: float,
        max_tile_seen: int,
    ) -> str:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "algorithm": "reinforce",
            "network": "cnn-3layer-no-padding",
            "encoding": "onehot-16x4x4",
            "episode": int(episode),
            "globalStep": int(self.global_step),
            "averageScore": float(average_score),
            "maxTileSeen": int(max_tile_seen),
            "config": {
                "maxExponent": self.config.max_exponent,
                "gamma": self.config.gamma,
                "learningRate": self.config.learning_rate,
                "entropyCoef": self.config.entropy_coef,
                "invalidActionPenalty": self.config.invalid_action_penalty,
                "mergeValueBonusScale": self.config.merge_value_bonus_scale,
            },
            "modelStateDict": self.policy_net.state_dict(),
            "optimizerStateDict": self.optimizer.state_dict(),
            "savedAt": time.time(),
        }
        torch.save(payload, str(checkpoint_path))
        return str(checkpoint_path.resolve())

    def _build_tensorboard_writer(
        self,
        *,
        tensorboard_log_dir: str | None,
        tensorboard_run_name: str | None,
    ) -> tuple[Any | None, str | None, str | None]:
        if not tensorboard_log_dir:
            return None, None, None

        if SummaryWriter is None:
            return None, None, "TensorBoard writer unavailable: install tensorboard package."

        base_dir = Path(tensorboard_log_dir).expanduser().resolve()
        run_name = tensorboard_run_name or time.strftime("reinforce_%Y%m%d_%H%M%S")
        run_dir = base_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            writer = SummaryWriter(log_dir=str(run_dir))
        except Exception as error:  # noqa: BLE001
            return None, None, f"Failed to initialize TensorBoard writer: {error}"
        return writer, str(run_dir), None

    def _encode_state(self, state: list[list[int]]) -> torch.Tensor:
        # Input state is log2 board with empty as 0. We one-hot into 16x4x4.
        encoded = torch.zeros((self.channels, 4, 4), dtype=torch.float32)
        for row in range(4):
            for col in range(4):
                exponent = int(state[row][col])
                if exponent < 0:
                    exponent = 0
                if exponent > self.config.max_exponent:
                    exponent = self.config.max_exponent
                encoded[exponent, row, col] = 1.0
        return encoded

    def _discounted_returns(self, rewards: list[float]) -> torch.Tensor:
        returns = []
        running = 0.0
        for reward in reversed(rewards):
            running = float(reward) + self.config.gamma * running
            returns.append(running)
        returns.reverse()
        values = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if values.numel() > 1:
            mean = values.mean()
            std = values.std(unbiased=False)
            if std.item() > 1e-6:
                values = (values - mean) / (std + 1e-8)
        return values

    def train(
        self,
        *,
        episodes: int,
        max_steps: int | None,
        terminate_on_win: bool,
        stop_event: Any,
        on_step: Callable[[int, dict[str, Any], dict[str, Any]], None] | None = None,
        on_episode_end: Callable[[int, dict[str, Any], dict[str, Any]], None] | None = None,
        tensorboard_log_dir: str | None = None,
        tensorboard_run_name: str | None = None,
        checkpoint_every_episodes: int = 0,
        checkpoint_dir: str | None = None,
        checkpoint_prefix: str = "reinforce_cnn",
        load_model_path: str | None = None,
        play_only: bool = False,
    ) -> dict[str, Any]:
        score_sum = 0.0
        max_tile_seen = 0
        completed = 0
        checkpoints_saved = 0
        latest_checkpoint_path: str | None = None
        loaded_model_path: str | None = None
        writer, tensorboard_run_dir, tensorboard_warning = self._build_tensorboard_writer(
            tensorboard_log_dir=tensorboard_log_dir,
            tensorboard_run_name=tensorboard_run_name,
        )

        if checkpoint_every_episodes < 0:
            raise ValueError("checkpoint_every_episodes must be >= 0")

        resolved_checkpoint_dir: str | None
        if checkpoint_every_episodes > 0:
            base_dir = checkpoint_dir or str(Path("models") / "2048")
            resolved_checkpoint_dir = str(Path(base_dir).expanduser().resolve())
        else:
            resolved_checkpoint_dir = None

        if load_model_path is not None:
            loaded_model_path = self._load_model_weights(load_model_path)

        if writer is not None:
            config_payload = {
                "algorithm": "reinforce",
                "network": "cnn-3layer-no-padding",
                "encoding": "onehot-16x4x4",
                "maxExponent": self.config.max_exponent,
                "gamma": self.config.gamma,
                "learningRate": self.config.learning_rate,
                "entropyCoef": self.config.entropy_coef,
                "invalidActionPenalty": self.config.invalid_action_penalty,
                "mergeValueBonusScale": self.config.merge_value_bonus_scale,
                "episodes": episodes,
                "maxSteps": max_steps,
                "terminateOnWin": terminate_on_win,
                "seed": self.seed,
                "playOnly": play_only,
                "checkpointEveryEpisodes": checkpoint_every_episodes,
                "checkpointDir": resolved_checkpoint_dir,
                "checkpointPrefix": checkpoint_prefix,
                "loadModelPath": loaded_model_path,
            }
            writer.add_text("run/config", json.dumps(config_payload, ensure_ascii=True, sort_keys=True), 0)
            writer.add_scalar("run/episodes", float(episodes), 0)
            writer.add_scalar("run/max_steps", float(max_steps) if max_steps is not None else -1.0, 0)
            writer.add_scalar("run/seed", float(self.seed) if self.seed is not None else -1.0, 0)
            writer.add_scalar("run/play_only", 1.0 if play_only else 0.0, 0)
            writer.add_scalar("run/checkpoint_every_episodes", float(checkpoint_every_episodes), 0)
            if loaded_model_path is not None:
                writer.add_text("run/loaded_model_path", loaded_model_path, 0)

        try:
            for episode in range(1, episodes + 1):
                if stop_event.is_set():
                    break

                env_seed = None if self.seed is None else self.seed + episode - 1
                env = Headless2048Env(
                    seed=env_seed,
                    max_steps=max_steps,
                    terminate_on_win=terminate_on_win,
                    invalid_action_penalty=self.config.invalid_action_penalty,
                    merge_value_bonus_scale=self.config.merge_value_bonus_scale,
                )
                obs = env.reset(seed=env_seed)
                terminated = False
                truncated = False
                total_reward = 0.0

                log_probs: list[torch.Tensor] = []
                entropies: list[torch.Tensor] = []
                rewards: list[float] = []
                actions: list[int] = []
                moved_count = 0

                while not (terminated or truncated):
                    if stop_event.is_set():
                        break

                    state_tensor = self._encode_state(obs["state"]).unsqueeze(0).to(self.device)
                    logits = self.policy_net(state_tensor)
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    action_index = int(action.item())

                    next_obs, reward, terminated, truncated, step_info = env.step(action_index)
                    log_probs.append(dist.log_prob(action))
                    entropy_step = dist.entropy()
                    entropies.append(entropy_step)
                    rewards.append(float(reward))
                    actions.append(action_index)

                    moved = bool(step_info.get("moved", False))
                    if moved:
                        moved_count += 1

                    self.global_step += 1
                    total_reward += float(reward)
                    obs = next_obs

                    if writer is not None:
                        probs = dist.probs.squeeze(0).detach().cpu()
                        movable_actions = step_info.get("movableActions", [])
                        raw_board = obs.get("rawBoard", [])
                        empty_cells = sum(1 for row in raw_board for value in row if int(value) == 0)
                        invalid_action_penalty = float(step_info.get("invalidActionPenalty", 0.0))
                        merge_bonus = float(step_info.get("mergeBonus", 0.0))
                        merge_value_bonus = float(step_info.get("mergeValueBonus", 0.0))
                        merge_count = float(step_info.get("mergeCount", 0.0))
                        merge_value_log2_counted_sum = float(step_info.get("mergeValueLog2CountedSum", 0.0))
                        score_delta = float(step_info.get("scoreDelta", reward))
                        empty_cells_reduced = float(step_info.get("emptyCellsReduced", 0.0))
                        writer.add_scalar("train/step_reward", float(reward), self.global_step)
                        writer.add_scalar("reward/score_delta", score_delta, self.global_step)
                        writer.add_scalar("reward/invalid_action_penalty", invalid_action_penalty, self.global_step)
                        writer.add_scalar("reward/merge_bonus", merge_bonus, self.global_step)
                        writer.add_scalar("reward/merge_value_bonus", merge_value_bonus, self.global_step)
                        writer.add_scalar("reward/merge_count", merge_count, self.global_step)
                        writer.add_scalar(
                            "reward/merge_value_log2_counted_sum",
                            merge_value_log2_counted_sum,
                            self.global_step,
                        )
                        writer.add_scalar("reward/empty_cells_reduced", empty_cells_reduced, self.global_step)
                        writer.add_scalar("train/episode_return_running", total_reward, self.global_step)
                        writer.add_scalar("train/score", float(obs.get("score", 0)), self.global_step)
                        writer.add_scalar("train/max_tile", float(obs.get("maxTile", 0)), self.global_step)
                        writer.add_scalar("train/action_selected", float(action_index), self.global_step)
                        writer.add_scalar("train/moved", 1.0 if moved else 0.0, self.global_step)
                        writer.add_scalar("train/movable_actions_count", float(len(movable_actions)), self.global_step)
                        writer.add_scalar("train/empty_cells", float(empty_cells), self.global_step)
                        writer.add_scalar(
                            "train/policy_entropy_step",
                            float(entropy_step.detach().cpu().item()),
                            self.global_step,
                        )
                        for action_id in range(4):
                            writer.add_scalar(
                                f"policy/action_prob_{action_id}",
                                float(probs[action_id].item()),
                                self.global_step,
                            )

                    if on_step is not None:
                        step_obs = dict(obs)
                        step_obs["action"] = int(step_info.get("action", action_index))
                        step_obs["animationGrid"] = step_info.get("animationGrid")
                        on_step(
                            episode,
                            step_obs,
                            {
                                "globalStep": self.global_step,
                                "totalReward": total_reward,
                            },
                        )

                if stop_event.is_set():
                    break

                grad_norm = 0.0
                policy_loss_value = 0.0
                return_mean = 0.0
                return_std = 0.0

                entropy_tensor = torch.stack(entropies).mean()
                self.last_entropy = float(entropy_tensor.detach().cpu().item())

                if play_only:
                    self.last_loss = None
                else:
                    returns = self._discounted_returns(rewards)
                    log_probs_tensor = torch.stack(log_probs)

                    policy_loss = -(log_probs_tensor * returns).sum()
                    loss = policy_loss - self.config.entropy_coef * entropy_tensor

                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()

                    grad_norm_sq = 0.0
                    for parameter in self.policy_net.parameters():
                        if parameter.grad is None:
                            continue
                        param_norm = float(parameter.grad.detach().data.norm(2).item())
                        grad_norm_sq += param_norm * param_norm
                    grad_norm = grad_norm_sq ** 0.5

                    self.optimizer.step()

                    self.last_loss = float(loss.detach().cpu().item())
                    policy_loss_value = float(policy_loss.detach().cpu().item())
                    return_mean = float(returns.mean().detach().cpu().item()) if returns.numel() > 0 else 0.0
                    return_std = (
                        float(returns.std(unbiased=False).detach().cpu().item()) if returns.numel() > 1 else 0.0
                    )

                score_sum += float(obs["score"])
                max_tile_seen = max(max_tile_seen, int(obs["maxTile"]))
                completed += 1

                episode_result = dict(obs)
                episode_result["terminated"] = bool(terminated)
                episode_result["truncated"] = bool(truncated)
                episode_result["totalReward"] = total_reward

                metrics = {
                    "entropy": self.last_entropy,
                    "loss": self.last_loss,
                    "globalStep": self.global_step,
                    "averageScore": (score_sum / completed) if completed > 0 else 0.0,
                }
                if checkpoint_every_episodes > 0 and resolved_checkpoint_dir is not None:
                    if episode % checkpoint_every_episodes == 0:
                        checkpoint_name = f"{checkpoint_prefix}_ep{episode:06d}.pt"
                        checkpoint_path = Path(resolved_checkpoint_dir) / checkpoint_name
                        latest_checkpoint_path = self._save_checkpoint(
                            checkpoint_path=checkpoint_path,
                            episode=episode,
                            average_score=metrics["averageScore"],
                            max_tile_seen=max_tile_seen,
                        )
                        checkpoints_saved += 1
                        metrics["checkpointPath"] = latest_checkpoint_path
                        metrics["checkpointsSaved"] = checkpoints_saved

                if on_episode_end is not None:
                    on_episode_end(episode, episode_result, metrics)

                if writer is not None:
                    writer.add_scalar("episode/score", float(obs.get("score", 0)), episode)
                    writer.add_scalar("episode/max_tile", float(obs.get("maxTile", 0)), episode)
                    writer.add_scalar("episode/steps", float(obs.get("steps", 0)), episode)
                    writer.add_scalar("episode/total_reward", float(total_reward), episode)
                    writer.add_scalar("episode/won", 1.0 if bool(obs.get("won", False)) else 0.0, episode)
                    writer.add_scalar("episode/terminated", 1.0 if bool(terminated) else 0.0, episode)
                    writer.add_scalar("episode/truncated", 1.0 if bool(truncated) else 0.0, episode)
                    writer.add_scalar("episode/move_success_rate", moved_count / max(1, len(actions)), episode)
                    writer.add_scalar("optimizer/loss", float(self.last_loss or 0.0), self.global_step)
                    writer.add_scalar("optimizer/policy_loss", policy_loss_value, self.global_step)
                    writer.add_scalar("optimizer/entropy", self.last_entropy, self.global_step)
                    writer.add_scalar("optimizer/grad_norm", grad_norm, self.global_step)
                    writer.add_scalar("optimizer/returns_mean", return_mean, self.global_step)
                    writer.add_scalar("optimizer/returns_std", return_std, self.global_step)
                    writer.add_scalar("optimizer/learning_rate", self.config.learning_rate, self.global_step)
                    writer.add_scalar("run/checkpoints_saved", float(checkpoints_saved), episode)
                    if actions:
                        writer.add_histogram(
                            "episode/actions",
                            torch.tensor(actions, dtype=torch.float32),
                            episode,
                        )
                    writer.flush()
        finally:
            if writer is not None:
                writer.flush()
                writer.close()

        return {
            "completedEpisodes": completed,
            "averageScore": (score_sum / completed) if completed > 0 else 0.0,
            "maxTileSeen": max_tile_seen,
            "globalStep": self.global_step,
            "lastLoss": self.last_loss,
            "lastEntropy": self.last_entropy,
            "modelStateDict": self.policy_net.state_dict(),
            "tensorboardEnabled": bool(writer is not None),
            "tensorboardRunDir": tensorboard_run_dir,
            "tensorboardWarning": tensorboard_warning,
            "playOnly": bool(play_only),
            "loadedModelPath": loaded_model_path,
            "checkpointEveryEpisodes": int(checkpoint_every_episodes),
            "checkpointDir": resolved_checkpoint_dir,
            "checkpointsSaved": int(checkpoints_saved),
            "latestCheckpointPath": latest_checkpoint_path,
        }
