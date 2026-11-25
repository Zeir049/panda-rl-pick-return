# panda-rl-pick-return

**Obstacle-aware reaching and pick-and-return with a simulated Franka Panda (PyBullet).**
An early prototype: IK target following with spherical safety margins around obstacles, and two custom Gymnasium environments trained with Stable-Baselines3 — discrete obstacle-aware reaching (DQN) and continuous grasp–lift–return (PPO).

All object, goal and obstacle positions are read directly from the simulator. That shortcut is what my follow-up project, [panda-vision-grasp](https://github.com/Zeir049/panda-vision-grasp), removes: there the object is localized from a single RGB-D view.

> **Status.** The first versions of both policies solved 0 / 50 episodes. I traced this to two simulation bugs and a reward-design flaw, fixed them and retrained ([What changed and why](#what-changed-and-why)).
> **Reaching (DQN)** succeeds on 50 held-out goals in 10 / 50 episodes with the greedy policy and 29 / 50 with ε = 0.05; a scripted controller reaches 35 / 50.
> **Pick-and-return (PPO)** grasps and lifts the cube in 35 / 50 stochastic episodes but **never returns home (0 / 50)**, so it is still unsolved.

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
| Reward (v2, default) | −distance(TCP, object) − 0.01 per step − 0.001‖a‖²; **+5 once** when the grasp happens, **+5 once** on the first lift (> 10 cm above the table); while grasped and lifted, +2 × (decrease of ‖q − q_home‖) since the last step, i.e. pay only for progress toward home; −3 per step in contact with an obstacle; **+50** on success. The old reward is kept as `reward_version="v1"` (see fix 3) |
| Termination | success = grasped ∧ lifted ∧ ‖q − q_home‖ < 0.25 rad; truncation after 250 steps |

PPO, single env, `n_steps=1024`, `batch_size=256`, `lr=3e-4`, 300k steps.

## What changed and why

Version names: **v0** = the original code ([commit f377879](https://github.com/Zeir049/panda-rl-pick-return/commit/f377879)); DQN **v1** = fix 1 only; DQN **v2** = fixes 1 + 2 (current); PPO **v1** = fixes 1 + 2; PPO **v2** = fixes 1 + 2 + 3 (current).

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

All numbers come from `python evaluate.py` over 50 episodes (files in `results/`). The DQN env draws goals from a seeded RNG. Training used seed 0, so the seed-0 rows below are **not held out**; the held-out row uses seed 1000, which was never used in training. "Stochastic" means ε = 0.05 for DQN and sampling from the Gaussian policy for PPO. Stochastic evaluation is not seeded, so a re-run can differ by an episode or two.

**Reaching (DQN)**

| Version | Steps | Greedy success | ε = 0.05 success | Collisions (greedy) | Median final distance (greedy) |
|---|---|---|---|---|---|
| v0: shared simulation, hand COM | 50k | 0 / 50 | – | 0 / 50 | 29.5 cm |
| v1: own client per env, hand COM | 50k | 0 / 50 | – | 0 / 50 | 10.8 cm |
| v1: own client per env, hand COM | 500k | 0 / 50 | – | 2 / 50 | 7.0 cm |
| **v2: own client per env, TCP** | 500k | **7 / 50** | **30 / 50** | 0 / 50 | 6.0 cm (ε: 3.4 cm) |
| *scripted greedy controller (v2 env)* | – | *31 / 50* | – | *1 / 50* | *3.4 cm* |
| **v2, held-out goals (seed 1000)** | 500k | **10 / 50** | **29 / 50** | 0 / 50 | 5.5 cm (ε: 3.5 cm) |
| *scripted greedy controller, seed 1000* | – | *35 / 50* | – | *1 / 50* | *3.3 cm* |

On held-out goals the policy does about as well as on the training seed (greedy 10 vs 7, ε 29 vs 30), and stays below the scripted controller (35 / 50), so the seed-0 result was not memorization of those 50 goals. Successful greedy episodes take 19.6 steps on average (seed 0). The 43 seed-0 failures are all timeouts, not collisions. A likely reason: the action set has no "stay" action, so near the goal a deterministic policy keeps stepping back and forth on the 2 cm grid. A little randomness breaks the loop (ε = 0.05 gives 30 / 50), and the scripted controller is limited the same way (31 / 50). I have not tested this explanation further.

**Pick-and-return (PPO)** — results for PPO v2 are under fix 3 below.

| Version | Mode | Grasped | Lifted | Returned home (success) | Touched obstacle |
|---|---|---|---|---|---|
| v0: hand COM | deterministic | 0 / 50 | 0 / 50 | 0 / 50 | 0 / 50 |
| v1: TCP | deterministic | 0 / 50 | 0 / 50 | 0 / 50 | 50 / 50 |
| v1: TCP | stochastic | **25 / 50** | **25 / 50** | **0 / 50** | 42 / 50 |

The PPO start state is fixed, so the 50 deterministic episodes are identical.

After fix 2 the policy learned to grasp and lift (training mean return −203 → +252, `tb_logs/ppo_v1_300k`), but it never returned home.

### Fix 3 — the PPO reward paid more for holding the cube than for finishing

* **Bug.** The v1 reward paid +1.5 per step while grasped and +2.5 per step while lifted, but only +12 once for success, and success **ends the episode**. Holding the cube in the air until the 250-step limit was therefore worth much more than returning home.
* **Symptom.** The v1 policy grasped and lifted the cube, then kept holding it: on average it stayed lifted for 126.7 steps, but never moved toward home (`results/eval_ppo_v1_post_lift_stats.txt`).
* **Check before retraining.** `check_reward_design.py` scores three scripted behaviours under each reward (`results/reward_design_check.txt`):

  | Scripted behaviour | v1 reward | v2 reward |
  |---|---|---|
  | grasp → lift → **return home** (success at step 112) | 114.8 | **−20.3** (best) |
  | grasp → lift → **hold** until step 250 | **666.2** (best) | −75.3 |
  | idle | −149.4 | −149.4 |

* **Fix (reward v2).** Grasp and lift are each rewarded **once** (+5), not per step. While the cube is held, the reward pays only for **progress toward home**: +2 × the decrease in ‖q − q_home‖, which sums to zero if the arm just holds still. Success pays +50. Everything else (reaching term, step and action penalties, −3 obstacle contact) is unchanged. The old reward remains available as `reward_version="v1"`.
* **Result (one 300k-step run, same hyper-parameters, `tb_logs/ppo_v2_300k`, `results/eval_ppo_v2.txt`).**

  | Version | Mode | Grasped | Lifted | Returned home (success) | Touched obstacle | Steps spent lifted |
  |---|---|---|---|---|---|---|
  | v1 reward | stochastic | 25 / 50 | 25 / 50 | 0 / 50 | 42 / 50 | 126.7 |
  | **v2 reward** | stochastic | **35 / 50** | **35 / 50** | **0 / 50** | **13 / 50** | 19.1 |
  | v2 reward | deterministic | 0 / 50 | 0 / 50 | 0 / 50 | 0 / 50 | – |

  The fix removed the incentive to hold: the policy now stays lifted for 19 steps instead of 127. It grasps more often and touches the obstacle far less. **It still never returns home, so PPO pick-and-return remains unsolved.**
* **Why it still fails (diagnosis, not yet fixed).** The learned policy reaches the cube in an awkward arm posture: on average 4.36 rad from the home pose in joint space at the moment of lift. The scripted grasp is 1.95 rad from home. While lifted, the policy gets no closer than 4.28 rad (success needs < 0.25 rad), and after about 19 steps the cube drops below the lift height. So the return phase was never learned: it is a long joint-space move the policy never discovered in 300k steps.
  * Possible next steps, not tried: start some episodes with the cube already in hand (a curriculum for the return phase), add ‖q − q_home‖ to the observation, or penalise posture during the grasp.
  * The deterministic (mean) action still never grasps; I have not investigated why.
  * I stopped after one bounded run (41 min). I did not tune on the evaluation episodes; the PPO start state is fixed, so they are the same as the training start state.

## Known issues

1. The cube's fixed start position overlaps obstacle box 1 by 1 cm in y, so the cube is pushed out slightly when the simulation starts and ends up touching that box, which constrains how the gripper can approach it.
2. The IK demos (`panda_follow_target.py`, `panda_obstacle_avoid.py`) use link 11 (`panda_grasptarget`) as end-effector; `panda_random_target.py` looks up `panda_hand` (link 8). This does not affect the RL results.
3. All four DQN training envs use seed 0, so they sample the same sequence of goals.

## Limitations / next steps

* **Ground-truth state.** Object, goal and obstacle positions come straight from the simulator. [panda-vision-grasp](https://github.com/Zeir049/panda-vision-grasp) replaces this with RGB-D perception and a friction-only grasp.
* Obstacle handling is a sphere approximation on the end-effector target only, not collision-free motion planning for the whole arm.
* Fixed scene: one cube position and three fixed obstacles. DQN is also evaluated on held-out goals, but the obstacles never move, and the PPO start state is the same in training and evaluation, so these results say little about generalization.
* PPO pick-and-return is unsolved (see fix 3).

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
python evaluate.py --only dqn --seed 1000     # held-out DQN goals
python check_grasp_feasibility.py             # scripted grasp-lift-return
python check_reach_feasibility.py [--link panda_hand] [--seed 1000]   # scripted greedy reaching
python check_reward_design.py                 # scripted return vs hold under reward v1 / v2
tensorboard --logdir tb_logs
```

The v0 and DQN v1 checkpoints only match the environment code of their own version (`checkpoints/ppo_panda_pick_return_v1.zip` still runs in the current env, since only the reward changed). To re-evaluate them, check out that version first (v0: commit f377879; DQN v1: the commit "Fix shared PyBullet client …"). Their result files were produced that way; the model paths inside them were updated after the files moved to `checkpoints/`.

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
| `check_reward_design.py` | Scripted return vs hold vs idle, scored under reward v1 and v2 |
| `smoke_test.py` | PyBullet install check |
| `dqn_panda_discrete.zip`, `ppo_panda_pick_return.zip` | Current models (DQN v2, PPO v2) |
| `checkpoints/` | Earlier models: `*_v0`, DQN `v1_50k`, `v1_500k`, PPO `v1` |
| `results/` | Evaluation and feasibility outputs quoted above |
| `tb_logs/` | TensorBoard logs: `dqn_v0`, `dqn_v1_50k`, `dqn_v1_500k`, `dqn_v2_500k`, `ppo_v1_300k`, `ppo_v2_300k` (no log exists for PPO v0) |
