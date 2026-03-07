"""RL training components for 2048 using CNN + one-hot encoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn
from torch.distributions import Categorical

from .headless import Headless2048Env


@dataclass
class ReinforceCnnConfig:
    """Configuration for a simple CNN policy-gradient trainer."""

    max_exponent: int = 15
    gamma: float = 0.99
    learning_rate: float = 3e-4
    entropy_coef: float = 1e-3


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
        on_episode_end: Callable[[int, dict[str, Any], dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        score_sum = 0.0
        max_tile_seen = 0
        completed = 0

        for episode in range(1, episodes + 1):
            if stop_event.is_set():
                break

            env_seed = None if self.seed is None else self.seed + episode - 1
            env = Headless2048Env(
                seed=env_seed,
                max_steps=max_steps,
                terminate_on_win=terminate_on_win,
            )
            obs = env.reset(seed=env_seed)
            terminated = False
            truncated = False
            total_reward = 0.0

            log_probs: list[torch.Tensor] = []
            entropies: list[torch.Tensor] = []
            rewards: list[float] = []

            while not (terminated or truncated):
                if stop_event.is_set():
                    break

                state_tensor = self._encode_state(obs["state"]).unsqueeze(0).to(self.device)
                logits = self.policy_net(state_tensor)
                dist = Categorical(logits=logits)
                action = dist.sample()

                next_obs, reward, terminated, truncated, _ = env.step(int(action.item()))
                log_probs.append(dist.log_prob(action))
                entropies.append(dist.entropy())
                rewards.append(float(reward))

                self.global_step += 1
                total_reward += float(reward)
                obs = next_obs

            if stop_event.is_set():
                break

            returns = self._discounted_returns(rewards)
            log_probs_tensor = torch.stack(log_probs)
            entropy_tensor = torch.stack(entropies).mean()

            policy_loss = -(log_probs_tensor * returns).sum()
            loss = policy_loss - self.config.entropy_coef * entropy_tensor

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            self.last_loss = float(loss.detach().cpu().item())
            self.last_entropy = float(entropy_tensor.detach().cpu().item())

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
            if on_episode_end is not None:
                on_episode_end(episode, episode_result, metrics)

        return {
            "completedEpisodes": completed,
            "averageScore": (score_sum / completed) if completed > 0 else 0.0,
            "maxTileSeen": max_tile_seen,
            "globalStep": self.global_step,
            "lastLoss": self.last_loss,
            "lastEntropy": self.last_entropy,
            "modelStateDict": self.policy_net.state_dict(),
        }
