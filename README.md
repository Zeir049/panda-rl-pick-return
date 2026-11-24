# panda-rl-pick-return

**Obstacle-aware reaching and pick-and-return with a simulated Franka Panda (PyBullet).**
An early prototype: IK target following with spherical safety margins around obstacles, and two custom Gymnasium environments trained with Stable-Baselines3 — discrete obstacle-aware reaching (DQN) and continuous grasp–lift–return (PPO).

All object, goal and obstacle positions are read directly from the simulator. That shortcut is what my follow-up project, [panda-vision-grasp](https://github.com/Zeir049/panda-vision-grasp), removes: there the object is localized from a single RGB-D view.

> **Status.** The first versions of both policies solved 0 / 50 episodes. I traced this to three bugs, fixed them and retrained ([What changed and why](#what-changed-and-why)).
> **Reaching (DQN)** now succeeds in 7 / 50 episodes with the greedy policy and 30 / 50 with ε = 0.05.
> **Pick-and-return (PPO)** now grasps and lifts the cube in 25 / 50 stochastic episodes but **never completes the return**, so it is still unsolved.

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

## 2. `PandaDiscreteReachEnv` — obstacle-aware reaching (DQN)

`panda_discrete_rl_env.py`. Goal: move the end-effector to a random goal point while avoiding the obstacles.

| | |
|---|---|
| End-effector | `panda_grasptarget`, the point between the fingertips (TCP); used for IK and for all distances |
| Observation (21) | TCP position (3), goal (3), current IK target (3), TCP − goal (3), obstacle centers (3 × 3) |
| Action (6, discrete) | move the IK target by ±2 cm along x, y or z (clamped to the workspace); the arm follows with IK, 20 sim steps per action |
| Goal sampling | uniform in the workspace, rejected if inside an obstacle safety sphere |
| Reward | −distance(TCP, goal) − 0.01 per step; +10 on success (< 3.5 cm); −5 and episode end on contact with an obstacle |
| Termination | success or collision; truncation after 200 steps |

DQN with `n_envs=4` (each env has its own physics client), buffer 100k, `lr=1e-3`, `batch_size=256`, ε 1 → 0.05 over the first 20 % of training, 500k steps.

## 3. `PandaPickReturnEnv` — grasp, lift, return (PPO)

`panda_pick_return_env.py`. Goal: grasp a green 6 cm cube placed at a fixed position next to an obstacle, lift it, and return to the home pose while holding it.

| | |
|---|---|
| End-effector | `panda_grasptarget` (TCP), used for the observation, the reward and the grasp check |
| Observation (26) | joint angles (7), joint velocities (7), gripper opening (1), TCP position (3), object position (3), TCP − object (3), grasped / lifted flags (2) |
| Action (8, continuous in [−1, 1]) | joint-target deltas × 0.06 rad (7, clamped to URDF limits), gripper open/close × 0.02 m (1); 20 sim steps per action |
| Grasp | **Simulated**: when the TCP is within 4.5 cm of the cube center and the gripper command is < 0.012, the cube is attached at the TCP with a fixed constraint (no friction grasp) |
| Reward | −distance(TCP, object) − 0.01 per step − 0.001‖a‖²; +1.5 per step while grasped, +2.5 per step while lifted (> 10 cm above the table); +0.3 / (1 + ‖q − q_home‖) while grasped and lifted; −3 on contact with an obstacle; +12 on success |
| Termination | success = grasped ∧ lifted ∧ ‖q − q_home‖ < 0.25 rad; truncation after 250 steps |

PPO, single env, `n_steps=1024`, `batch_size=256`, `lr=3e-4`, 300k steps.

## What changed and why

Version names: **v0** = the original code ([commit f377879](https://github.com/Zeir049/panda-rl-pick-return/commit/f377879)); DQN **v1** = fix 1 only; DQN **v2** and PPO **v1** = the current code.

### Fix 1 — the four DQN environments were secretly one simulation

* **Bug.** Every PyBullet call used the module-level API without a `physicsClientId`, so every call went to client 0. When `make_vec_env(n_envs=4)` created the second environment, its `resetSimulation()` wiped the first one's world and rebuilt it in the same client. The four "parallel" environments took turns commanding **one** robot, so each one's transitions were partly caused by the others' actions.
* **Symptom.** Training looked like it was improving: in `tb_logs/dqn_v0` mean episode length fell from 200 to 127. Yet the saved policy scored 0 / 50 when evaluated alone. You can confirm the bug by creating two envs and calling `getNumBodies()` on each client: the second client is empty.
* **Fix.** Each env now owns a `pybullet_utils.bullet_client.BulletClient` and makes every call through it (`self._p.…`); the helper functions take the client as an argument.

### Fix 2 — "end-effector" was the hand's centre of mass, not the fingertips

* **Bug.** Both environments used link `panda_hand` and read its position with `getLinkState(...)[0]`. That value is the link's **centre of mass**, which is 4.0 cm from the link frame and 6.5 cm from the point between the fingertips (`panda_grasptarget`, the TCP).
  * PPO: when the TCP is exactly on the cube center, the measured "hand–cube" distance is still about 6 cm, above the 4.5 cm grasp threshold. The grasp could never trigger.
  * DQN: the 3.5 cm success radius was measured from that point, while the arm was steered with IK on `panda_hand`.
* **How I checked that a fix was needed, before retraining.** I wrote scripted controllers that use no learning (`check_grasp_feasibility.py`, `check_reach_feasibility.py`). If a hand-written controller cannot solve the task, RL cannot either.

  | Scripted check | old end-effector (`panda_hand` COM) | TCP (`panda_grasptarget`) |
  |---|---|---|
  | Top-down grasp-lift-return, yaw 90° | TCP 0.7 cm from cube, env distance 6.1 cm → **no grasp** | env distance 0.7 cm → grasp, lift, **success at step 112** |
  | Greedy reaching, same 50 goals | **0 / 50**, median final distance 6.6 cm | **31 / 50**, median final distance 3.4 cm |

  At yaw 0° and 45° the scripted grasp fails in both versions because the TCP stops 9.9–12.6 cm short of the cube. I did not investigate further; the cube sits right next to an obstacle (see Known issues). The link offsets above are printed by `check_grasp_feasibility.py`.
* **Fix.** Both environments use `panda_grasptarget` for IK, observations, rewards and the grasp check. In that link the centre of mass and the frame coincide. The grasp constraint now attaches the cube at the TCP.

### Results before / after

All numbers come from `python evaluate.py` over 50 episodes (files in `results/`). The DQN env draws goals from an RNG seeded with 0, so every row uses the same 50 goals. These goals are **not held out**: the training envs use the same seed. "Stochastic" means ε = 0.05 for DQN and sampling from the Gaussian policy for PPO.

**Reaching (DQN)**

| Version | Steps | Greedy success | ε = 0.05 success | Collisions (greedy) | Median final distance (greedy) |
|---|---|---|---|---|---|
| v0: shared simulation, hand COM | 50k | 0 / 50 | – | 0 / 50 | 29.5 cm |
| v1: own client per env, hand COM | 50k | 0 / 50 | – | 0 / 50 | 10.8 cm |
| v1: own client per env, hand COM | 500k | 0 / 50 | – | 2 / 50 | 7.0 cm |
| **v2: own client per env, TCP** | 500k | **7 / 50** | **30 / 50** | 0 / 50 | 6.0 cm (ε: 3.4 cm) |
| *scripted greedy controller (v2 env)* | – | *31 / 50* | – | *1 / 50* | *3.4 cm* |

Successful greedy episodes take 19.6 steps on average. The 43 failures are all timeouts, not collisions. A likely reason: the action set has no "stay" action, so near the goal a deterministic policy keeps stepping back and forth on the 2 cm grid. A little randomness breaks the loop (ε = 0.05 gives 30 / 50), and the scripted controller is limited the same way (31 / 50). I have not tested this explanation further.

**Pick-and-return (PPO)**

| Version | Mode | Grasped | Lifted | Returned home (success) | Touched obstacle |
|---|---|---|---|---|---|
| v0: hand COM | deterministic | 0 / 50 | 0 / 50 | 0 / 50 | 0 / 50 |
| v1: TCP | deterministic | 0 / 50 | 0 / 50 | 0 / 50 | 50 / 50 |
| v1: TCP | stochastic | **25 / 50** | **25 / 50** | **0 / 50** | 42 / 50 |

The PPO start state is fixed, so the 50 deterministic episodes are identical.

After fix 2 the policy learned to grasp and lift (training mean return −203 → +252, `tb_logs/ppo_v1_300k`), but it never returns home. A likely cause, **not yet tested**, is the reward design: holding the lifted cube pays about +4 per step until the 250-step limit, while returning home pays +12 once and **ends the episode**. Stopping early forfeits the per-step bonus, so the reward never favours returning. I made one bounded training attempt and did not tune further. The next change would be to make the holding bonus small relative to the success bonus, or to pay it only as progress toward home. I have not checked why the deterministic (mean) action never grasps.

## Known issues

1. The cube's fixed start position overlaps obstacle box 1 by 1 cm in y, so the cube is pushed out slightly when the simulation starts and ends up touching that box, which constrains how the gripper can approach it.
2. The IK demos (`panda_follow_target.py`, `panda_obstacle_avoid.py`) use link 11 (`panda_grasptarget`) as end-effector; `panda_random_target.py` looks up `panda_hand` (link 8). This does not affect the RL results.
3. All four DQN training envs use seed 0, so they sample the same sequence of goals.

## Limitations / next steps

* **Ground-truth state.** Object, goal and obstacle positions come straight from the simulator. [panda-vision-grasp](https://github.com/Zeir049/panda-vision-grasp) replaces this with RGB-D perception and a friction-only grasp.
* Obstacle handling is a sphere approximation on the end-effector target only, not collision-free motion planning for the whole arm.
* Fixed scene: one cube position and three fixed obstacles, and the evaluation goals are not held out, so these results say nothing about generalization.
* PPO pick-and-return is unsolved (see above).

## Run

```bash
pip install -r requirements.txt

python smoke_test.py --direct                 # install check without a window (omit --direct for the GUI)
python panda_follow_target.py                 # GUI: WASD / R,F move the target, O orbit, Q quit
python panda_obstacle_avoid.py                # GUI: I,J,K,L / R,F move the target, H toggles avoidance
python panda_random_target.py                 # GUI: random targets with avoidance, Ctrl+C to quit

python panda_discrete_rl_env.py train --timesteps 500000     # DQN -> dqn_panda_discrete.zip, tb_logs/
python panda_discrete_rl_env.py play                         # GUI rollout of the saved model
python panda_pick_return_env.py train --timesteps 300000     # PPO -> ppo_panda_pick_return.zip, tb_logs/
python panda_pick_return_env.py play                         # GUI rollout of the saved model

python evaluate.py                            # both current models, deterministic
python evaluate.py --stochastic               # ε = 0.05 (DQN) / sampled actions (PPO)
python check_grasp_feasibility.py             # scripted grasp-lift-return
python check_reach_feasibility.py [--link panda_hand]   # scripted greedy reaching
tensorboard --logdir tb_logs
```

The v0 and DQN v1 checkpoints only match the environment code of their own version. To re-evaluate them, check out that version first (v0: commit f377879; DQN v1: the commit "Fix shared PyBullet client …"). Their result files were produced that way; the model paths inside them were updated after the files moved to `checkpoints/`.

Tested with Python 3.12, PyBullet 3.2.7, Gymnasium 1.2.3, Stable-Baselines3 2.7.1, PyTorch 2.10 (CPU) on Windows 11.

## Files

| File | Content |
|---|---|
| `panda_follow_target.py` | Interactive IK target following (keyboard), no obstacles |
| `panda_obstacle_avoid.py` | Interactive IK target following with obstacle safety spheres |
| `panda_random_target.py` | Same, with randomly moving targets |
| `panda_discrete_rl_env.py` | `PandaDiscreteReachEnv` + DQN train / play |
| `panda_pick_return_env.py` | `PandaPickReturnEnv` + PPO train / play |
| `evaluate.py` | Headless evaluation of saved models |
| `check_grasp_feasibility.py`, `check_reach_feasibility.py` | Scripted (non-learning) controllers that test whether each task is solvable |
| `smoke_test.py` | PyBullet install check |
| `dqn_panda_discrete.zip`, `ppo_panda_pick_return.zip` | Current models (DQN v2, PPO v1) |
| `checkpoints/` | Earlier models: `*_v0`, DQN `v1_50k`, `v1_500k` |
| `results/` | Evaluation and feasibility outputs quoted above |
| `tb_logs/` | TensorBoard logs: `dqn_v0`, `dqn_v1_50k`, `dqn_v1_500k`, `dqn_v2_500k`, `ppo_v1_300k` (no log exists for PPO v0) |
