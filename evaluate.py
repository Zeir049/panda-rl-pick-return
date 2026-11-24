"""Headless evaluation of the saved models (no GUI; deterministic policies unless --stochastic).

    python evaluate.py                                  # both models, 50 episodes each
    python evaluate.py --only dqn --dqn checkpoints/dqn_panda_discrete_v1_500k.zip
    python evaluate.py --episodes 5                     # quick check

The DQN env draws its goals from an RNG seeded at construction (seed=0), so every
run evaluates the same 50 goals. The PPO env has a fixed start state.
"""
import argparse

import numpy as np
from stable_baselines3 import DQN, PPO

from panda_discrete_rl_env import PandaDiscreteReachEnv
from panda_pick_return_env import PandaPickReturnEnv


def _mean(xs):
    return f"{np.mean(xs):.1f}" if len(xs) else "-"


def eval_dqn(path, episodes, deterministic=True):
    env = PandaDiscreteReachEnv(render_mode="none", seed=0)
    model = DQN.load(path)
    success = collided = 0
    lengths, success_lengths, final_dists = [], [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, k = False, 0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, term, trunc, info = env.step(int(action))
            k += 1
            done = term or trunc
        success += info["success"]
        collided += info["collided"]
        lengths.append(k)
        final_dists.append(info["distance"])
        if info["success"]:
            success_lengths.append(k)
    env.close()
    timeouts = episodes - success - collided
    mode = "deterministic" if deterministic else "stochastic"
    print(f"[DQN reach] {path}, {episodes} episodes, {mode}: "
          f"success {success}/{episodes} ({100 * success / episodes:.0f} %), "
          f"collision {collided}/{episodes}, timeout {timeouts}/{episodes}, "
          f"mean steps {_mean(lengths)} (successful episodes: {_mean(success_lengths)}), "
          f"median final distance {100 * np.median(final_dists):.1f} cm")


def eval_ppo(path, episodes, deterministic=True):
    env = PandaPickReturnEnv(render_mode="none", seed=0)
    model = PPO.load(path)
    counts = {"grasped": 0, "lifted": 0, "collided": 0, "success": 0}
    lengths, success_lengths, home_errs = [], [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, k = False, 0
        seen = {key: False for key in counts}
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, term, trunc, info = env.step(action)
            k += 1
            for key in seen:
                seen[key] |= bool(info[key])
            done = term or trunc
        for key in counts:
            counts[key] += seen[key]
        lengths.append(k)
        home_errs.append(info["home_err"])
        if info["success"]:
            success_lengths.append(k)
    env.close()
    summary = ", ".join(f"{k} {v}/{episodes}" for k, v in counts.items())
    mode = "deterministic" if deterministic else "stochastic"
    print(f"[PPO pick-return] {path}, {episodes} episodes, {mode} (ever during episode): {summary}, "
          f"mean steps {_mean(lengths)} (successful episodes: {_mean(success_lengths)}), "
          f"mean final home error {np.mean(home_errs):.2f} rad")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--only", choices=["dqn", "ppo", "both"], default="both")
    parser.add_argument("--dqn", default="dqn_panda_discrete.zip")
    parser.add_argument("--ppo", default="ppo_panda_pick_return.zip")
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of taking the mode")
    args = parser.parse_args()
    if args.only in ("dqn", "both"):
        eval_dqn(args.dqn, args.episodes, not args.stochastic)
    if args.only in ("ppo", "both"):
        eval_ppo(args.ppo, args.episodes, not args.stochastic)
