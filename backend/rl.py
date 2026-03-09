"""RL training components for 2048 using DQN + CNN + one-hot encoding."""

from __future__ import annotations

import json
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, NamedTuple

import numpy as np
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
    learning_rate: float = 5e-5
    batch_size: int = 256
    replay_capacity: int = 500_000
    min_replay_size: int = 5000
    target_update_freq: int = 2000
    train_freq: int = 1
    gradient_steps: int = 4
    num_envs: int = 8
    max_grad_norm: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 100_000
    invalid_action_penalty: float = 0.0
    merge_value_bonus_scale: float = 0.0
    reward_log_scale: bool = True
    empty_cell_reward_scale: float = 0.25
    n_step: int = 3


# ---------------------------------------------------------------------------
# Q-Network (Dueling architecture)
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Dueling Q-network with CNN backbone for 4x4 board."""

    def __init__(self, in_channels: int, num_actions: int = 4) -> None:
        super().__init__()
        # Conv path 1: 2x2 kernels (local patterns)
        self.conv2x2 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
        )  # -> (256, 2, 2)
        # Conv path 2: 1x1 + 3x3 (broader context)
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )  # -> (256, 2, 2)
        # Merge: 256*4 + 256*4 = 2048
        feat_dim = 256 * 4 + 256 * 4
        self.value_stream = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.conv2x2(x).flatten(1)
        f2 = self.conv3x3(x).flatten(1)
        features = torch.cat([f1, f2], dim=1)
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


class PrioritizedReplayBuffer:
    """Proportional prioritized experience replay (sum-tree)."""

    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self._buffer: list[Transition | None] = [None] * capacity
        self._priorities = np.zeros(capacity, dtype=np.float64)
        self._pos = 0
        self._size = 0
        self._max_priority = 1.0

    def push(self, transition: Transition) -> None:
        self._buffer[self._pos] = transition
        self._priorities[self._pos] = self._max_priority ** self.alpha
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4
               ) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        priorities = self._priorities[:self._size]
        probs = priorities / priorities.sum()
        indices = np.random.choice(self._size, size=batch_size, p=probs, replace=True)
        samples = [self._buffer[i] for i in indices]  # type: ignore[misc]

        # Importance-sampling weights
        weights = (self._size * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        return samples, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        priorities = (np.abs(td_errors) + 1e-6) ** self.alpha
        for idx, p in zip(indices, priorities):
            self._priorities[idx] = p
            self._max_priority = max(self._max_priority, float(p) ** (1.0 / self.alpha))

    def __len__(self) -> int:
        return self._size


class NStepBuffer:
    """Accumulates transitions and produces n-step return transitions."""

    def __init__(self, n: int, gamma: float) -> None:
        self.n = n
        self.gamma = gamma
        self._buf: deque[tuple[torch.Tensor, int, float, torch.Tensor, bool, list[int]]] = deque()

    def push(self, state: torch.Tensor, action: int, reward: float,
             next_state: torch.Tensor, done: bool,
             next_movable: list[int]) -> list[Transition]:
        """Add a step, return any ready n-step transitions."""
        self._buf.append((state, action, reward, next_state, done, next_movable))
        if done:
            return self._flush()
        if len(self._buf) >= self.n:
            return self._pop_one()
        return []

    def _nstep_return(self, entries: list | deque) -> float:
        R = 0.0
        for e in reversed(entries):
            R = e[2] + self.gamma * R
        return R

    def _pop_one(self) -> list[Transition]:
        entries = list(self._buf)[:self.n]
        R = self._nstep_return(entries)
        t = Transition(
            state=entries[0][0], action=entries[0][1], reward=R,
            next_state=entries[-1][3], done=False,
            next_movable=entries[-1][5],
        )
        self._buf.popleft()
        return [t]

    def _flush(self) -> list[Transition]:
        """Flush all remaining entries on episode end."""
        entries = list(self._buf)
        transitions: list[Transition] = []
        for i in range(len(entries)):
            remaining = entries[i:]
            R = self._nstep_return(remaining)
            transitions.append(Transition(
                state=remaining[0][0], action=remaining[0][1], reward=R,
                next_state=entries[-1][3], done=True,
                next_movable=entries[-1][5],
            ))
        self._buf.clear()
        return transitions


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
        self.replay = PrioritizedReplayBuffer(self.config.replay_capacity)

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

    def _select_actions_batch(
        self,
        state_batch: torch.Tensor,
        movable_list: list[list[int]],
        epsilon: float,
    ) -> list[int]:
        """Select actions for a batch of states using epsilon-greedy with masking."""
        n = state_batch.shape[0]
        actions: list[int] = []

        # Get Q-values for entire batch in one forward pass
        with torch.no_grad():
            q_all = self.q_net(state_batch.to(self.device))  # (N, 4)

        for i in range(n):
            movable = movable_list[i]
            if not movable:
                actions.append(0)
                continue
            if random.random() < epsilon:
                actions.append(random.choice(movable))
            else:
                q = q_all[i]
                mask = torch.full_like(q, float("-inf"))
                for a in movable:
                    mask[a] = 0.0
                actions.append(int((q + mask).argmax().item()))
        return actions

    def _per_beta(self) -> float:
        """Anneal PER importance-sampling beta from 0.4 to 1.0 over training."""
        frac = min(1.0, self.global_step / max(1, self.config.epsilon_decay_steps * 5))
        return 0.4 + 0.6 * frac

    def _learn(self) -> float:
        if len(self.replay) < self.config.min_replay_size:
            return 0.0

        beta = self._per_beta()
        batch, indices, is_weights = self.replay.sample(self.config.batch_size, beta=beta)
        states = torch.stack([t.state for t in batch]).to(self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.stack([t.next_state for t in batch]).to(self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)
        weights = torch.tensor(is_weights, dtype=torch.float32, device=self.device)

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
            gamma_n = self.config.gamma ** self.config.n_step
            target = rewards + gamma_n * next_q * (1.0 - dones)

        td_errors = (q_values - target).detach().cpu().numpy()
        self.replay.update_priorities(indices, td_errors)

        elementwise_loss = nn.functional.smooth_l1_loss(q_values, target, reduction="none")
        loss = (elementwise_loss * weights).mean()
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

    def _make_env(self, env_seed: int | None, max_steps: int | None,
                   terminate_on_win: bool) -> tuple[Headless2048Env, dict[str, Any], torch.Tensor]:
        """Create an env, reset it, return (env, obs, state_tensor)."""
        env = Headless2048Env(
            seed=env_seed, max_steps=max_steps,
            terminate_on_win=terminate_on_win,
            invalid_action_penalty=self.config.invalid_action_penalty,
            merge_value_bonus_scale=self.config.merge_value_bonus_scale,
            reward_log_scale=self.config.reward_log_scale,
            empty_cell_reward_scale=self.config.empty_cell_reward_scale,
        )
        obs = env.reset(seed=env_seed)
        return env, obs, self._encode_state(obs["state"])

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
        recent_scores: deque[float] = deque(maxlen=100)
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
                "numEnvs": self.config.num_envs,
                "epsilonStart": self.config.epsilon_start,
                "epsilonEnd": self.config.epsilon_end,
                "epsilonDecaySteps": self.config.epsilon_decay_steps,
                "episodes": episodes, "seed": self.seed,
            }
            writer.add_text("run/config", json.dumps(config_payload, ensure_ascii=True, sort_keys=True), 0)

        num_envs = self.config.num_envs
        batch_iter = 0
        target_sync_iters = max(1, self.config.target_update_freq // num_envs)
        # Track next seed offset for creating new envs
        next_seed_offset = num_envs

        # Per-env tracking
        envs: list[Headless2048Env] = []
        obs_list: list[dict[str, Any]] = []
        state_tensors: list[torch.Tensor] = []
        ep_rewards: list[float] = [0.0] * num_envs
        ep_losses_all: list[list[float]] = [[] for _ in range(num_envs)]
        n_step_bufs: list[NStepBuffer] = [
            NStepBuffer(self.config.n_step, self.config.gamma) for _ in range(num_envs)
        ]

        # Initialize all environments
        for i in range(num_envs):
            env_seed = None if self.seed is None else self.seed + i
            env, obs, st = self._make_env(env_seed, max_steps, terminate_on_win)
            envs.append(env)
            obs_list.append(obs)
            state_tensors.append(st)

        try:
            while completed < episodes:
                if stop_event.is_set():
                    break

                epsilon = 0.0 if play_only else self._epsilon()
                self.last_epsilon = epsilon

                # Batch forward pass: select actions for all envs at once
                state_batch = torch.stack(state_tensors)
                movable_list = [obs.get("movableActions", [0, 1, 2, 3]) for obs in obs_list]
                actions = self._select_actions_batch(state_batch, movable_list, epsilon)

                # Step all envs
                for i in range(num_envs):
                    if stop_event.is_set():
                        break

                    action_index = actions[i]
                    next_obs, reward, terminated, truncated, step_info = envs[i].step(action_index)
                    next_st = self._encode_state(next_obs["state"])
                    next_movable = next_obs.get("movableActions", [0, 1, 2, 3])

                    if not play_only:
                        done = terminated or truncated
                        nstep_transitions = n_step_bufs[i].push(
                            state=state_tensors[i], action=action_index,
                            reward=float(reward), next_state=next_st,
                            done=done, next_movable=next_movable,
                        )
                        for t in nstep_transitions:
                            self.replay.push(t)

                    ep_rewards[i] += float(reward)
                    self.global_step += 1

                    if on_step is not None:
                        step_obs = dict(next_obs)
                        step_obs["action"] = action_index
                        step_obs["animationGrid"] = step_info.get("animationGrid")
                        on_step(completed + 1, step_obs, {
                            "globalStep": self.global_step,
                            "totalReward": ep_rewards[i],
                        })

                    # Episode finished — auto-reset this env
                    if terminated or truncated:
                        completed += 1
                        recent_scores.append(float(next_obs["score"]))
                        max_tile_seen = max(max_tile_seen, int(next_obs["maxTile"]))

                        avg_loss_ep = (sum(ep_losses_all[i]) / len(ep_losses_all[i])
                                       if ep_losses_all[i] else 0.0)
                        self.last_loss = avg_loss_ep if ep_losses_all[i] else None

                        episode_result = dict(next_obs)
                        episode_result["terminated"] = bool(terminated)
                        episode_result["truncated"] = bool(truncated)
                        episode_result["totalReward"] = ep_rewards[i]

                        metrics: dict[str, Any] = {
                            "epsilon": self.last_epsilon,
                            "loss": self.last_loss,
                            "globalStep": self.global_step,
                            "averageScore": (sum(recent_scores) / len(recent_scores)) if recent_scores else 0.0,
                            "replaySize": len(self.replay),
                        }

                        if checkpoint_every_episodes > 0 and resolved_checkpoint_dir is not None:
                            if completed % checkpoint_every_episodes == 0:
                                cp_name = f"{checkpoint_prefix}_ep{completed:06d}.pt"
                                cp_path = Path(resolved_checkpoint_dir) / cp_name
                                latest_checkpoint_path = self._save_checkpoint(
                                    checkpoint_path=cp_path, episode=completed,
                                    average_score=metrics["averageScore"],
                                    max_tile_seen=max_tile_seen,
                                )
                                checkpoints_saved += 1
                                metrics["checkpointPath"] = latest_checkpoint_path
                                metrics["checkpointsSaved"] = checkpoints_saved

                        if on_episode_end is not None:
                            on_episode_end(completed, episode_result, metrics)

                        if writer is not None:
                            writer.add_scalar("episode/score", float(next_obs.get("score", 0)), completed)
                            writer.add_scalar("episode/max_tile", float(next_obs.get("maxTile", 0)), completed)
                            writer.add_scalar("episode/steps", float(next_obs.get("steps", 0)), completed)
                            writer.add_scalar("episode/total_reward", ep_rewards[i], completed)
                            writer.add_scalar("episode/epsilon", self.last_epsilon, completed)
                            writer.add_scalar("episode/avg_loss", avg_loss_ep, completed)
                            writer.add_scalar("episode/replay_size", float(len(self.replay)), completed)

                        # Reset this env slot
                        env_seed = None if self.seed is None else self.seed + next_seed_offset
                        next_seed_offset += 1
                        envs[i], obs_list[i], state_tensors[i] = self._make_env(
                            env_seed, max_steps, terminate_on_win)
                        ep_rewards[i] = 0.0
                        ep_losses_all[i] = []
                        n_step_bufs[i] = NStepBuffer(self.config.n_step, self.config.gamma)

                        if completed >= episodes:
                            break
                    else:
                        state_tensors[i] = next_st
                        obs_list[i] = next_obs

                # Learn & target sync (based on batch iterations, not global_step)
                batch_iter += 1

                if not play_only and batch_iter % self.config.train_freq == 0:
                    for _gs in range(self.config.gradient_steps):
                        loss_val = self._learn()
                        if loss_val > 0:
                            for ll in ep_losses_all:
                                ll.append(loss_val)

                if not play_only and batch_iter % target_sync_iters == 0:
                    self.target_net.load_state_dict(self.q_net.state_dict())

                if writer is not None and batch_iter % max(1, 500 // num_envs) == 0:
                    writer.flush()
        finally:
            if writer is not None:
                writer.flush()
                writer.close()

        return {
            "completedEpisodes": completed,
            "averageScore": (sum(recent_scores) / len(recent_scores)) if recent_scores else 0.0,
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
