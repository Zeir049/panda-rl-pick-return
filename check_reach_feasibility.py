"""Can the discrete reaching task be solved at all? A greedy scripted policy, no learning.

At every step it picks the action that moves the IK target closest to the goal,
on the same 50 goals evaluate.py uses (env RNG seeded with 0).

    python check_reach_feasibility.py                    # end-effector = panda_grasptarget (current env)
    python check_reach_feasibility.py --link panda_hand  # the end-effector link used before the fix
"""
import argparse

import numpy as np

from panda_discrete_rl_env import PandaDiscreteReachEnv, find_link_index


def main(link_name, episodes):
    env = PandaDiscreteReachEnv(render_mode="none", seed=0)
    env.eef_link = find_link_index(env.robot, link_name, pb=env._p)  # used for IK and for measuring
    moves = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]) * env.step_size
    success = collided = 0
    final_dists, lags = [], []
    for _ in range(episodes):
        env.reset()
        done = False
        while not done:
            cand = [np.linalg.norm(env._clamp_to_bounds(env.desired_pos + m) - env.goal_pos) for m in moves]
            _, _, term, trunc, info = env.step(int(np.argmin(cand)))
            done = term or trunc
        success += info["success"]
        collided += info["collided"]
        final_dists.append(info["distance"])
        lags.append(np.linalg.norm(env.desired_pos - env._eef_pos()))
    env.close()
    print(f"[greedy scripted reach] link {link_name}, {episodes} episodes: "
          f"success {success}/{episodes}, collision {collided}/{episodes}, "
          f"median final distance {100 * np.median(final_dists):.1f} cm, "
          f"median |IK target - measured end-effector| at episode end {100 * np.median(lags):.1f} cm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--link", default="panda_grasptarget")
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    main(args.link, args.episodes)
