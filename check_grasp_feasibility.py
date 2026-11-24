"""Can the pick-and-return task be solved at all? A hand-written IK controller, no learning.

Moves the fingertip point (panda_grasptarget, the TCP) above the cube, down onto it,
closes the gripper, lifts and returns to the home pose, using only the env's own
action interface (joint-target deltas + gripper command).

    python check_grasp_feasibility.py

If this scripted controller cannot trigger the env's grasp condition, an RL policy cannot either.
"""
import numpy as np
import pybullet

from panda_pick_return_env import PandaPickReturnEnv


def run(yaw):
    env = PandaPickReturnEnv(render_mode="none")
    env.reset()
    P = getattr(env, "_p", pybullet)  # v0 used the module-level client
    tcp = next(i for i in range(P.getNumJoints(env.robot))
               if P.getJointInfo(env.robot, i)[12] == b"panda_grasptarget")
    orn = P.getQuaternionFromEuler([np.pi, 0, yaw])  # top-down, rotated about z
    obj = env._obj_pos()
    info = {}

    def move(target, grip, steps):
        nonlocal info
        for _ in range(steps):
            q = P.calculateInverseKinematics(env.robot, tcp, list(target), orn, maxNumIterations=200)
            a = np.clip((np.array(q[:7]) - env.target_q) / env.joint_delta_scale, -1, 1)
            _, _, term, _, info = env.step(np.r_[a, grip])
            if term:
                return True
        return False

    move(obj + [0, 0, 0.10], +1, 30)                 # above the cube, gripper open
    move(obj, +1, 30)                                # descend
    tcp_err = np.linalg.norm(np.array(P.getLinkState(env.robot, tcp)[0]) - env._obj_pos())
    env_dist = info["dist"]                          # distance the env's grasp check uses
    move(obj, -1, 3)                                 # close
    grasped = info["grasped"]
    move(obj + [0, 0, 0.15], -1, 30)                 # lift
    lifted = info["lifted"]
    for _ in range(60):                              # back to the home pose
        a = np.clip((env.home_q - env.target_q) / env.joint_delta_scale, -1, 1)
        _, _, term, _, info = env.step(np.r_[a, -1])
        if term:
            break
    print(f"yaw {np.degrees(yaw):3.0f} deg: TCP-cube {100 * tcp_err:4.1f} cm, "
          f"env grasp distance {100 * env_dist:4.1f} cm (threshold {100 * env.grasp_dist:.1f}) | "
          f"grasped {grasped}, lifted {lifted}, success {info['success']} at step {info['step']}")
    env.close()


def link_offsets():
    env = PandaPickReturnEnv(render_mode="none")
    env.reset()
    P = getattr(env, "_p", pybullet)
    idx = {P.getJointInfo(env.robot, i)[12]: i for i in range(P.getNumJoints(env.robot))}
    hand = P.getLinkState(env.robot, idx[b"panda_hand"], computeForwardKinematics=True)
    tcp = P.getLinkState(env.robot, idx[b"panda_grasptarget"], computeForwardKinematics=True)
    d = lambda a, b: 100 * np.linalg.norm(np.subtract(a, b))
    print(f"panda_hand getLinkState()[0] (centre of mass): {d(hand[0], hand[4]):.1f} cm from its link frame, "
          f"{d(hand[0], tcp[4]):.1f} cm from the TCP (panda_grasptarget frame); "
          f"panda_grasptarget COM vs frame: {d(tcp[0], tcp[4]):.1f} cm")
    env.close()


if __name__ == "__main__":
    link_offsets()
    for yaw in (0.0, np.pi / 4, np.pi / 2):
        run(yaw)
