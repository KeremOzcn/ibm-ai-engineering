"""Compare online-target and lagged-target DQN at matched environment-step budgets."""

from __future__ import annotations

import copy
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from projects.common import file_sha256, runtime, source_identity, write_json


class QNetwork(nn.Sequential):
    """A small function approximator; no pre-trained policy is loaded."""

    def __init__(self):
        super().__init__(
            nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2)
        )


def bellman_target(rewards, terminated, next_values, gamma=0.99):
    """Time-limit truncations bootstrap; genuine terminal states do not."""
    return rewards + gamma * (~terminated).float() * next_values


def evaluate_policy(model, seeds: list[int], *, random_policy=False) -> list[int]:
    """Use a separate environment and frozen seeds, with exploration switched off."""
    import gymnasium as gym

    env = gym.make("CartPole-v1", max_episode_steps=500)
    scores = []
    model.eval()
    for seed in seeds:
        state, _ = env.reset(seed=seed)
        env.action_space.seed(seed)
        total = 0
        while True:
            with torch.no_grad():
                action = (
                    env.action_space.sample()
                    if random_policy
                    else int(model(torch.tensor(state)).argmax())
                )
            state, reward, terminated, truncated, _ = env.step(action)
            total += int(reward)
            if terminated or truncated:
                break
        scores.append(total)
    env.close()
    return scores


def train(
    seed: int, output: Path, *, lagged_target: bool, total_steps: int, validation_seeds: list[int]
) -> dict:
    """Hold step budget, reward, replay capacity and exploration schedule constant."""
    import gymnasium as gym

    torch.manual_seed(seed)
    rng = random.Random(seed)
    replay_rng = random.Random(seed + 1)
    model = QNetwork()
    target = copy.deepcopy(model).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    replay = deque(maxlen=10000)
    env = gym.make("CartPole-v1", max_episode_steps=500)
    state, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    episode_return, returns, history, best = 0, [], [], -1.0
    path = output / "best.pt"
    output.mkdir(parents=True, exist_ok=True)
    for step in range(1, total_steps + 1):
        epsilon = max(0.05, 1.0 - 0.95 * step / max(1, int(total_steps * 0.7)))
        if rng.random() < epsilon:
            action = rng.randrange(2)
        else:
            with torch.no_grad():
                action = int(model(torch.tensor(state)).argmax())
        next_state, reward, terminated, truncated, _ = env.step(action)
        replay.append((state.copy(), action, reward, next_state.copy(), terminated))
        episode_return += int(reward)
        state = next_state
        if terminated or truncated:
            returns.append(episode_return)
            episode_return = 0
            state, _ = env.reset()
        if len(replay) >= 256 and step % 4 == 0:
            batch = replay_rng.sample(list(replay), 64)
            states = torch.tensor(np.stack([r[0] for r in batch]))
            actions = torch.tensor([r[1] for r in batch])
            rewards = torch.tensor([r[2] for r in batch], dtype=torch.float32)
            following = torch.tensor(np.stack([r[3] for r in batch]))
            terminal = torch.tensor([r[4] for r in batch])
            with torch.no_grad():
                next_values = (target if lagged_target else model)(following).max(1).values
                expected = bellman_target(rewards, terminal, next_values)
            loss = nn.functional.smooth_l1_loss(
                model(states).gather(1, actions[:, None]).squeeze(1), expected
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            if lagged_target and step % 500 == 0:
                target.load_state_dict(model.state_dict())
        if step % max(1, total_steps // 5) == 0 or step == total_steps:
            validation = evaluate_policy(model, validation_seeds)
            mean = float(np.mean(validation))
            history.append({"environment_steps": step, "validation_mean_return": mean})
            if mean > best:
                best = mean
                torch.save(model.state_dict(), path)
    env.close()
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    fresh = QNetwork()
    fresh.load_state_dict(torch.load(path, weights_only=True))
    fresh.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            model(torch.zeros(3, 4)), fresh(torch.zeros(3, 4)), rtol=0, atol=0
        )
    return {
        "model": model,
        "history": history,
        "training_episode_returns": returns,
        "checkpoint_sha256": file_sha256(path),
        "exact_reload": True,
        "environment_steps": total_steps,
    }


def run(output: Path, *, full: bool = False) -> dict:
    """Use repeated independent training seeds and reserved validation/test reset seeds."""
    torch.set_num_threads(1)
    started = time.perf_counter()
    train_seeds = [11, 22, 33] if full else [11]
    validation_seeds = list(range(1000, 1005))
    test_seeds = list(range(2000, 2020)) if full else list(range(2000, 2005))
    total_steps = 20000 if full else 1000
    manifest = {
        "train_seeds": train_seeds,
        "validation_seeds": validation_seeds,
        "test_seeds": test_seeds,
        "episode_limit": 500,
    }
    write_json(output / "split-manifest.json", manifest)
    result = {}
    for name, lagged in [("online_target", False), ("lagged_target", True)]:
        runs = []
        for seed in train_seeds:
            run_result = train(
                seed,
                output / name / str(seed),
                lagged_target=lagged,
                total_steps=total_steps,
                validation_seeds=validation_seeds,
            )
            scores = evaluate_policy(run_result.pop("model"), test_seeds)
            runs.append(
                {
                    "seed": seed,
                    **run_result,
                    "test_returns": scores,
                    "test_mean_return": float(np.mean(scores)),
                }
            )
        means = [r["test_mean_return"] for r in runs]
        result[name] = {
            "runs": runs,
            "mean_of_seed_means": float(np.mean(means)),
            "std_of_seed_means": float(np.std(means, ddof=1)) if len(means) > 1 else None,
        }
    report = {
        "project": "cartpole",
        "profile": "three-seed-step-budget" if full else "bounded-pilot",
        "models": result,
        "random_baseline_returns": evaluate_policy(QNetwork(), test_seeds, random_policy=True),
        "manifest": manifest,
        "training_config": {
            "replay_capacity": 10000,
            "batch_size": 64,
            "warmup_steps": 256,
            "gradient_interval": 4,
            "target_sync_steps": 500,
            "gamma": 0.99,
            "learning_rate": 0.001,
        },
        "split_manifest_sha256": file_sha256(output / "split-manifest.json"),
        "limitations": "CartPole is a simple simulated control task. Three seeds and twenty test episodes per seed do not establish general RL robustness; no real-world control claim.",
        "seconds": time.perf_counter() - started,
        "runtime": runtime(["torch", "numpy", "gymnasium"]),
        "source": source_identity(
            globals().get("__file__"), globals().get("PROJECT_SOURCE_SHA256")
        ),
    }
    write_json(output / "metrics.json", report)
    return report
