# panda-rl-pick-return

**Obstacle-aware reaching and pick-and-return with a simulated Franka Panda (PyBullet).**
An early prototype: IK target following with spherical safety margins around obstacles, and two custom Gymnasium environments trained with Stable-Baselines3 — continuous grasp–lift–return (PPO) and discrete obstacle-aware reaching (DQN).

All object, goal and obstacle positions are read directly from the simulator. That shortcut is what my follow-up project, [panda-vision-grasp](https://github.com/Zeir049/panda-vision-grasp), removes: there the object is localized from a single RGB-D view.

> Status: the environments and the IK demos run, but **neither saved policy solves its task** (see [Results](#results)). This repo is kept as a record of the prototype, not as a working RL solution.

## Scene

All scripts build the same world: a table, a Panda with a fixed base on the table top (home pose `q = [0, -0.6, 0, -2.0, 0, 1.6, 0.8]`), and three static boxes on the table as obstacles. Physics runs at 240 Hz with 200 solver iterations.

## 1. IK target following with obstacle safety spheres

`panda_obstacle_avoid.py`, `panda_random_target.py` (and `panda_follow_target.py` without obstacles)

```
desired target (blue) ──► push out of safety spheres ──► clamp to workspace ──► safe target (red) ──► PyBullet IK ──► joint position control
```

* Each box is approximated by a sphere centered on the box with radius `‖half_extents‖ + 0.08 m`.
* If the desired end-effector target lies inside a sphere, it is projected radially onto the sphere surface (`push_out_of_obstacles`). The IK solver then follows this safe target with a fixed top-down orientation.
* Only the **end-effector target** is constrained. The arm links are not checked, and there is no path planning: if the target moves across a sphere the safe target jumps to the other side. In `panda_obstacle_avoid.py` the on-screen `near_obstacle` flag (any robot link within 2 cm of a box) shows when the arm still gets close.

## 2. `PandaPickReturnEnv` — grasp, lift, return (PPO)

`panda_pick_return_env.py`. Goal: grasp a green 6 cm cube placed at a fixed position next to an obstacle, lift it, and return to the home pose while holding it.

| | |
|---|---|
| Observation (26) | joint angles (7), joint velocities (7), gripper opening (1), hand position (3), object position (3), hand − object (3), grasped / lifted flags (2) |
| Action (8, continuous in [−1, 1]) | joint-target deltas × 0.06 rad (7, clamped to URDF limits), gripper open/close × 0.02 m (1); 20 sim steps per action |
| Grasp | **Simulated**: when the `panda_hand` frame is within 4.5 cm of the object center and the gripper command is < 0.012, the object is attached to the hand with a fixed constraint (no friction grasp) |
| Reward | −distance(hand, object) − 0.01 per step − 0.001‖a‖²; +1.5 while grasped, +2.5 while lifted (> 10 cm above the table); +0.3 / (1 + ‖q − q_home‖) while grasped and lifted; −3 on contact with an obstacle; +12 on success |
| Termination | success = grasped ∧ lifted ∧ ‖q − q_home‖ < 0.25 rad; truncation after 250 steps |

Trained with PPO (`n_steps=1024`, `batch_size=256`, `lr=3e-4`, 300k steps, single env).

## 3. `PandaDiscreteReachEnv` — obstacle-aware reaching (DQN)

`panda_discrete_rl_env.py`. Goal: move the end-effector to a random goal point while avoiding the obstacles.

| | |
|---|---|
| Observation (21) | hand position (3), goal (3), current IK target (3), hand − goal (3), obstacle centers (3 × 3) |
| Action (6, discrete) | move the IK target by ±2 cm along x, y or z (clamped to the workspace); the arm follows with IK, 20 sim steps per action |
| Goal sampling | uniform in the workspace, rejected if inside an obstacle safety sphere |
| Reward | −distance(hand, goal) − 0.01 per step; +10 on success (< 3.5 cm); −5 and episode end on contact with an obstacle |
| Termination | success or collision; truncation after 200 steps |

Trained with DQN (`n_envs=4`, buffer 100k, `lr=1e-3`, `batch_size=256`, 50k steps); TensorBoard log in `tb_logs/DQN_1`.

## Results

`python evaluate.py --episodes 50` (headless, deterministic policies):

| Model | Episodes | Success | Other |
|---|---|---|---|
| `dqn_panda_discrete.zip` (reach) | 50 | **0 / 50** | 0 collisions, all episodes run to the 200-step limit, mean return −67.5 |
| `ppo_panda_pick_return.zip` (pick-return) | 50 | **0 / 50** | never grasps; mean return −33.2. The start state is fixed, so all 50 episodes are identical |

From `tb_logs/DQN_1` (training, with ε-greedy exploration): mean episode return rose from −65.3 to −56.4 and mean episode length fell from 200 to 127 over 50k steps. Because of the shared-world issue below, I do not read much into these curves. There is no training log for the PPO model.

## Known issues (found when cleaning up this repo)

1. **Parallel DQN envs share one simulation.** The environments call PyBullet without a `physicsClientId`, so every instance uses client 0, and creating a new instance resets the previous one's world. With `n_envs=4`, the four envs took turns driving the same robot during training. Fix: store the client id and pass `physicsClientId=self.cid` to every call, or train with `n_envs=1`.
2. **The grasp condition is probably very hard to reach.** It measures distance from the `panda_hand` frame, which sits about 10 cm above the fingertips, to the cube center. In a quick IK test (top-down and side approaches) the hand stopped 11–14 cm from the cube because the fingers hit the cube and table first. This likely explains why PPO never grasps. Measuring from the `panda_grasptarget` link (index 11) would be the natural fix.
3. The cube's fixed start position overlaps obstacle box 1 by 1 cm in y, so it is pushed out slightly when the simulation starts.
4. The IK demos use link 11 (`panda_grasptarget`) as end-effector, while the RL environments use link 8 (`panda_hand`).

## Limitations / next steps

* **Ground-truth state.** Object, goal and obstacle positions come straight from the simulator. [panda-vision-grasp](https://github.com/Zeir049/panda-vision-grasp) replaces this with RGB-D perception and a friction-only grasp.
* Obstacle handling is a sphere approximation on the end-effector target only, not collision-free motion planning for the whole arm.
* Fixed scene: one object position and three fixed obstacles, so even a working policy would not show generalization.

## Run

```bash
pip install -r requirements.txt

python smoke_test.py --direct                 # install check without a window (omit --direct for the GUI)
python panda_follow_target.py                 # GUI: WASD / R,F move the target, O orbit, Q quit
python panda_obstacle_avoid.py                # GUI: I,J,K,L / R,F move the target, H toggles avoidance
python panda_random_target.py                 # GUI: random targets with avoidance, Ctrl+C to quit

python panda_discrete_rl_env.py train --timesteps 50000      # DQN -> dqn_panda_discrete.zip, tb_logs/
python panda_discrete_rl_env.py play                         # GUI rollout of the saved model
python panda_pick_return_env.py train --timesteps 300000     # PPO -> ppo_panda_pick_return.zip
python panda_pick_return_env.py play                         # GUI rollout of the saved model

python evaluate.py --episodes 50              # headless evaluation (table above)
tensorboard --logdir tb_logs
```

Tested with Python 3.12, PyBullet 3.2.7, Gymnasium 1.2.3, Stable-Baselines3 2.7.1, PyTorch 2.10 (CPU) on Windows 11.

## Files

| File | Content |
|---|---|
| `panda_follow_target.py` | Interactive IK target following (keyboard), no obstacles |
| `panda_obstacle_avoid.py` | Interactive IK target following with obstacle safety spheres |
| `panda_random_target.py` | Same, with randomly moving targets |
| `panda_pick_return_env.py` | `PandaPickReturnEnv` + PPO train / play |
| `panda_discrete_rl_env.py` | `PandaDiscreteReachEnv` + DQN train / play |
| `evaluate.py` | Headless evaluation of both saved models |
| `smoke_test.py` | PyBullet install check |
| `ppo_panda_pick_return.zip`, `dqn_panda_discrete.zip` | Saved SB3 models (180 KB / 120 KB) |
| `tb_logs/DQN_1` | TensorBoard log of the DQN run |
