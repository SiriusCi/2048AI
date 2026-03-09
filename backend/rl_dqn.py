"""RL training components for 2048 using DQN + CNN + one-hot encoding."""

from __future__ import annotations

import json
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, NamedTuple

import torch
from torch import nn

from .headless import Headless2048Env

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional runtime dependency
    SummaryWriter = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DQNConfig:
    """Configuration for the DQN trainer."""

    max_exponent: int = 15
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 128
    replay_capacity: int = 50_000
    min_replay_size: int = 1000
    target_update_freq: int = 500
    max_grad_norm: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 50_000
    invalid_action_penalty: float = 0.0
    merge_value_bonus_scale: float = 1.0


# ---------------------------------------------------------------------------
# Q-Network (Dueling architecture)
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Dueling Q-network with CNN backbone for 4x4 board."""

    def __init__(self, in_channels: int, num_actions: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool
    next_movable: list[int]


class ReplayBuffer:
    """Fixed-size circular replay buffer."""

    def __init__(self, capacity: int) -> None:
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self._buffer, batch_size)

    def __len__(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# DQN Trainer
# ---------------------------------------------------------------------------

class DQNTrainer:
    """DQN trainer with one-hot CNN, target network, and experience replay."""

    def __init__(self, config: DQNConfig, *, seed: int | None = None) -> None:
        self.config = config
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.channels = self.config.max_exponent + 1

        self.q_net = QNetwork(self.channels).to(self.device)
        self.target_net = QNetwork(self.channels).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.config.learning_rate)
        self.replay = ReplayBuffer(self.config.replay_capacity)

        self.global_step = 0
        self.last_loss: float | None = None
        self.last_epsilon: float = self.config.epsilon_start

    def _epsilon(self) -> float:
        frac = min(1.0, self.global_step / max(1, self.config.epsilon_decay_steps))
        return self.config.epsilon_start + frac * (self.config.epsilon_end - self.config.epsilon_start)

    def _encode_state(self, state: list[list[int]]) -> torch.Tensor:
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

    def _select_action(self, state_tensor: torch.Tensor, movable: list[int], epsilon: float) -> int:
        if not movable:
            return 0
        if random.random() < epsilon:
            return random.choice(movable)
        with torch.no_grad():
            q_values = self.q_net(state_tensor.unsqueeze(0).to(self.device)).squeeze(0)
            mask = torch.full_like(q_values, float("-inf"))
            for a in movable:
                mask[a] = 0.0
            return int((q_values + mask).argmax().item())

    def _learn(self) -> float:
        if len(self.replay) < self.config.min_replay_size:
            return 0.0

        batch = self.replay.sample(self.config.batch_size)
        states = torch.stack([t.state for t in batch]).to(self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.stack([t.next_state for t in batch]).to(self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_online = self.q_net(next_states)
            next_q_target = self.target_net(next_states)
            for i, t in enumerate(batch):
                if t.done:
                    continue
                m = torch.full((4,), float("-inf"), device=self.device)
                for a in t.next_movable:
                    m[a] = 0.0
                next_q_online[i] = next_q_online[i] + m
            best_actions = next_q_online.argmax(dim=1)
            next_q = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            target = rewards + self.config.gamma * next_q * (1.0 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        return float(loss.item())

    def _load_model_weights(self, model_path: str) -> str:
        checkpoint_path = Path(model_path).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model file not found: {checkpoint_path}")
        payload = torch.load(str(checkpoint_path), map_location=self.device)
        if isinstance(payload, dict) and "modelStateDict" in payload:
            state_dict = payload["modelStateDict"]
            gs = payload.get("globalStep")
            if gs is not None:
                try:
                    self.global_step = int(gs)
                except (TypeError, ValueError):
                    pass
            opt_state = payload.get("optimizerStateDict")
            if opt_state is not None:
                try:
                    self.optimizer.load_state_dict(opt_state)
                except Exception:
                    pass
            tgt_state = payload.get("targetNetStateDict")
            if tgt_state is not None:
                try:
                    self.target_net.load_state_dict(tgt_state)
                except Exception:
                    pass
        elif isinstance(payload, dict):
            state_dict = payload
        else:
            raise ValueError("Unsupported model file format.")
        self.q_net.load_state_dict(state_dict, strict=True)
        return str(checkpoint_path)

    def _save_checkpoint(self, *, checkpoint_path: Path, episode: int,
                         average_score: float, max_tile_seen: int) -> str:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "algorithm": "dqn",
            "network": "dueling-cnn-3layer",
            "encoding": "onehot-16x4x4",
            "episode": int(episode),
            "globalStep": int(self.global_step),
            "averageScore": float(average_score),
            "maxTileSeen": int(max_tile_seen),
            "config": {
                "maxExponent": self.config.max_exponent,
                "gamma": self.config.gamma,
                "learningRate": self.config.learning_rate,
                "batchSize": self.config.batch_size,
                "replayCapacity": self.config.replay_capacity,
                "minReplaySize": self.config.min_replay_size,
                "targetUpdateFreq": self.config.target_update_freq,
                "maxGradNorm": self.config.max_grad_norm,
                "epsilonStart": self.config.epsilon_start,
                "epsilonEnd": self.config.epsilon_end,
                "epsilonDecaySteps": self.config.epsilon_decay_steps,
                "invalidActionPenalty": self.config.invalid_action_penalty,
                "mergeValueBonusScale": self.config.merge_value_bonus_scale,
            },
            "modelStateDict": self.q_net.state_dict(),
            "targetNetStateDict": self.target_net.state_dict(),
            "optimizerStateDict": self.optimizer.state_dict(),
            "savedAt": time.time(),
        }
        torch.save(payload, str(checkpoint_path))
        return str(checkpoint_path.resolve())

    def _build_tensorboard_writer(self, *, tensorboard_log_dir: str | None,
                                   tensorboard_run_name: str | None):
        if not tensorboard_log_dir:
            return None, None, None
        if SummaryWriter is None:
            return None, None, "TensorBoard writer unavailable: install tensorboard package."
        base_dir = Path(tensorboard_log_dir).expanduser().resolve()
        run_name = tensorboard_run_name or time.strftime("dqn_%Y%m%d_%H%M%S")
        run_dir = base_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            writer = SummaryWriter(log_dir=str(run_dir))
        except Exception as error:
            return None, None, f"Failed to initialize TensorBoard writer: {error}"
        return writer, str(run_dir), None

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
        checkpoint_prefix: str = "dqn_cnn",
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
                "algorithm": "dqn", "network": "dueling-cnn-3layer",
                "encoding": "onehot-16x4x4",
                "gamma": self.config.gamma, "learningRate": self.config.learning_rate,
                "batchSize": self.config.batch_size,
                "replayCapacity": self.config.replay_capacity,
                "targetUpdateFreq": self.config.target_update_freq,
                "epsilonStart": self.config.epsilon_start,
                "epsilonEnd": self.config.epsilon_end,
                "epsilonDecaySteps": self.config.epsilon_decay_steps,
                "episodes": episodes, "seed": self.seed,
            }
            writer.add_text("run/config", json.dumps(config_payload, ensure_ascii=True, sort_keys=True), 0)

        try:
            for episode in range(1, episodes + 1):
                if stop_event.is_set():
                    break

                env_seed = None if self.seed is None else self.seed + episode - 1
                env = Headless2048Env(
                    seed=env_seed, max_steps=max_steps,
                    terminate_on_win=terminate_on_win,
                    invalid_action_penalty=self.config.invalid_action_penalty,
                    merge_value_bonus_scale=self.config.merge_value_bonus_scale,
                )
                obs = env.reset(seed=env_seed)
                terminated = False
                truncated = False
                total_reward = 0.0
                actions: list[int] = []
                moved_count = 0
                ep_losses: list[float] = []

                state_tensor = self._encode_state(obs["state"])

                while not (terminated or truncated):
                    if stop_event.is_set():
                        break

                    movable = obs.get("movableActions", [0, 1, 2, 3])
                    epsilon = 0.0 if play_only else self._epsilon()
                    self.last_epsilon = epsilon
                    action_index = self._select_action(state_tensor, movable, epsilon)

                    next_obs, reward, terminated, truncated, step_info = env.step(action_index)
                    next_state_tensor = self._encode_state(next_obs["state"])
                    next_movable = next_obs.get("movableActions", [0, 1, 2, 3])

                    if not play_only:
                        done = terminated or truncated
                        self.replay.push(Transition(
                            state=state_tensor,
                            action=action_index,
                            reward=float(reward),
                            next_state=next_state_tensor,
                            done=done,
                            next_movable=next_movable,
                        ))
                        loss_val = self._learn()
                        if loss_val > 0:
                            ep_losses.append(loss_val)

                        # Target network sync
                        if self.global_step % self.config.target_update_freq == 0:
                            self.target_net.load_state_dict(self.q_net.state_dict())

                    actions.append(action_index)
                    moved = bool(step_info.get("moved", False))
                    if moved:
                        moved_count += 1
                    self.global_step += 1
                    total_reward += float(reward)

                    if writer is not None:
                        writer.add_scalar("train/step_reward", float(reward), self.global_step)
                        writer.add_scalar("train/epsilon", epsilon, self.global_step)
                        if ep_losses:
                            writer.add_scalar("train/step_loss", ep_losses[-1], self.global_step)

                    if on_step is not None:
                        step_obs = dict(next_obs)
                        step_obs["action"] = action_index
                        step_obs["animationGrid"] = step_info.get("animationGrid")
                        on_step(episode, step_obs, {
                            "globalStep": self.global_step,
                            "totalReward": total_reward,
                        })

                    state_tensor = next_state_tensor
                    obs = next_obs

                if stop_event.is_set():
                    break

                avg_loss = sum(ep_losses) / len(ep_losses) if ep_losses else 0.0
                self.last_loss = avg_loss if ep_losses else None

                score_sum += float(obs["score"])
                max_tile_seen = max(max_tile_seen, int(obs["maxTile"]))
                completed += 1

                episode_result = dict(obs)
                episode_result["terminated"] = bool(terminated)
                episode_result["truncated"] = bool(truncated)
                episode_result["totalReward"] = total_reward

                metrics: dict[str, Any] = {
                    "epsilon": self.last_epsilon,
                    "loss": self.last_loss,
                    "globalStep": self.global_step,
                    "averageScore": (score_sum / completed) if completed > 0 else 0.0,
                    "replaySize": len(self.replay),
                }

                if checkpoint_every_episodes > 0 and resolved_checkpoint_dir is not None:
                    if episode % checkpoint_every_episodes == 0:
                        cp_name = f"{checkpoint_prefix}_ep{episode:06d}.pt"
                        cp_path = Path(resolved_checkpoint_dir) / cp_name
                        latest_checkpoint_path = self._save_checkpoint(
                            checkpoint_path=cp_path, episode=episode,
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
                    writer.add_scalar("episode/epsilon", self.last_epsilon, episode)
                    writer.add_scalar("episode/avg_loss", avg_loss, episode)
                    writer.add_scalar("episode/replay_size", float(len(self.replay)), episode)
                    writer.add_scalar("episode/move_success_rate",
                                      moved_count / max(1, len(actions)), episode)
                    writer.add_scalar("run/checkpoints_saved", float(checkpoints_saved), episode)
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
            "lastEpsilon": self.last_epsilon,
            "modelStateDict": self.q_net.state_dict(),
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
