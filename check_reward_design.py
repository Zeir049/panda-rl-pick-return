"""Does the reward make returning home the best behaviour? Compare scripted policies, no learning.

    python check_reward_design.py

For each reward version, three hand-written policies run one full episode each:
  return : grasp -> lift -> go back to the home pose (the intended behaviour)
  hold   : grasp -> lift -> keep holding the cube in the air until the step limit
  idle   : do nothing
The intended behaviour should have the highest episode return.
"""
import numpy as np

from panda_pick_return_env import PandaPickReturnEnv

YAW = np.pi / 2  # gripper yaw for which the scripted grasp works (see check_grasp_feasibility.py)


def run(reward_version, behaviour):
    env = PandaPickReturnEnv(render_mode="none", reward_version=reward_version)
    env.reset()
    P = env._p
    orn = P.getQuaternionFromEuler([np.pi, 0, YAW])
    obj = env._obj_pos()
    total, info, done = 0.0, {}, False

    def step(action):
        nonlocal total, info, done
        _, r, term, trunc, info = env.step(action)
        total += r
        done = term or trunc

    def move(target, grip, steps):
        for _ in range(steps):
            if done:
                return
            q = P.calculateInverseKinematics(env.robot, env.eef_link, list(target), orn, maxNumIterations=200)
            a = np.clip((np.array(q[:7]) - env.target_q) / env.joint_delta_scale, -1, 1)
            step(np.r_[a, grip])

    if behaviour != "idle":
        move(obj + [0, 0, 0.10], +1, 30)   # above the cube
        move(obj, +1, 30)                  # descend
        move(obj, -1, 3)                   # close -> grasp
        move(obj + [0, 0, 0.15], -1, 30)   # lift
    err_after_lift = info.get("home_err", float("nan"))
    while not done:
        if behaviour == "return":
            a = np.clip((env.home_q - env.target_q) / env.joint_delta_scale, -1, 1)
            step(np.r_[a, -1])
        else:  # hold position (or idle): zero joint deltas, keep the gripper closed
            step(np.r_[np.zeros(7), -1])
    env.close()
    info["home_err_after_lift"] = err_after_lift
    return total, info


if __name__ == "__main__":
    for version in ("v1", "v2"):
        for behaviour in ("return", "hold", "idle"):
            total, info = run(version, behaviour)
            print(f"reward {version} | {behaviour:6s}: episode return {total:8.1f} | "
                  f"grasped {info['grasped']}, lifted {info['lifted']}, success {info['success']}, "
                  f"steps {info['step']}, home error after the lift phase {info['home_err_after_lift']:.2f} rad")
