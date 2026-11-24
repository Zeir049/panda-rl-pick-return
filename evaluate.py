"""Headless evaluation of the saved models (no GUI).

    python evaluate.py               # 50 episodes per model
    python evaluate.py --episodes 5  # quick check
"""
import argparse

import numpy as np
from stable_baselines3 import DQN, PPO

from panda_discrete_rl_env import PandaDiscreteReachEnv
from panda_pick_return_env import PandaPickReturnEnv


def eval_dqn(path, episodes):
    env = PandaDiscreteReachEnv(render_mode="none", seed=0)
    model = DQN.load(path)
    success = collided = 0
    lengths, returns = [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, ret, k = False, 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(action))
            ret += r
            k += 1
            done = term or trunc
        success += info["success"]
        collided += info["collided"]
        lengths.append(k)
        returns.append(ret)
    env.close()
    print(f"[DQN reach] {episodes} episodes: success {success}/{episodes}, "
          f"collision {collided}/{episodes}, mean length {np.mean(lengths):.1f}, "
          f"mean return {np.mean(returns):.2f}")


def eval_ppo(path, episodes):
    env = PandaPickReturnEnv(render_mode="none", seed=0)
    model = PPO.load(path)
    counts = {"grasped": 0, "lifted": 0, "collided": 0, "success": 0}
    lengths, returns = [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, ret, k = False, 0.0, 0
        seen = {key: False for key in counts}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ret += r
            k += 1
            for key in seen:
                seen[key] |= bool(info[key])
            done = term or trunc
        for key in counts:
            counts[key] += seen[key]
        lengths.append(k)
        returns.append(ret)
    env.close()
    summary = ", ".join(f"{k} {v}/{episodes}" for k, v in counts.items())
    print(f"[PPO pick-return] {episodes} episodes: {summary}, "
          f"mean length {np.mean(lengths):.1f}, mean return {np.mean(returns):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--dqn", default="dqn_panda_discrete.zip")
    parser.add_argument("--ppo", default="ppo_panda_pick_return.zip")
    args = parser.parse_args()
    eval_dqn(args.dqn, args.episodes)
    eval_ppo(args.ppo, args.episodes)
