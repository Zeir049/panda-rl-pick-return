import os
import time
import numpy as np

import pybullet as p
from pybullet_utils import bullet_client as bc
import pybullet_data

import gymnasium as gym
from gymnasium import spaces


def find_link_index(body_id: int, name_contains: str, pb=p) -> int:
    n = pb.getNumJoints(body_id)
    for i in range(n):
        info = pb.getJointInfo(body_id, i)
        link_name = info[12].decode("utf-8")
        if name_contains in link_name:
            return i
    return n - 1


def get_body_top_z(body_id: int, pb=p) -> float:
    top_z = -1e9
    aabb_min, aabb_max = pb.getAABB(body_id, -1)
    top_z = max(top_z, aabb_max[2])
    for link in range(pb.getNumJoints(body_id)):
        aabb_min, aabb_max = pb.getAABB(body_id, link)
        top_z = max(top_z, aabb_max[2])
    return top_z


def make_box(half_extents, pos, rgba=(0.7, 0.7, 0.7, 1.0), pb=p):
    col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=half_extents)
    vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba)
    body = pb.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos,
    )
    return body


class PandaDiscreteReachEnv(gym.Env):
    """
    - 행동(action): 6개 (±x, ±y, ±z) 로 end-effector 목표점(desired_pos)을 한 칸씩 이동
    - 로봇은 IK로 desired_pos를 따라감
    - 목표(goal_pos)에 도달하면 성공
    - 장애물(박스) 접촉하면 실패(패널티)
    """

    metadata = {"render_modes": ["human", "none"], "render_fps": 60}

    def __init__(
        self,
        render_mode="none",
        step_size=0.02,
        max_steps=200,
        substeps=20,
        success_dist=0.035,
        collision_dist=0.0,
        seed=0,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.step_size = float(step_size)
        self.max_steps = int(max_steps)
        self.substeps = int(substeps)
        self.success_dist = float(success_dist)
        self.collision_dist = float(collision_dist)  # 0이면 접촉만 체크
        self.rng = np.random.default_rng(seed)

        # 6방향 action
        self.action_space = spaces.Discrete(6)

        # 관측(Observation)
        # eef_pos(3) + goal_pos(3) + desired_pos(3) + (eef-goal)(3) + obstacle_centers(9) = 21
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(21,),
            dtype=np.float32,
        )

        self.cid = None
        self.robot = None
        self.table_id = None
        self.obstacle_ids = []
        self.goal_id = None
        self.desired_id = None

        self.arm_joints = list(range(7))
        self.eef_link = None
        self.joint_damping = None

        self.table_top_z = None
        self.bounds = None  # (x_min, x_max, y_min, y_max, z_min, z_max)
        self.obstacles = []  # list of {"center": np.array(3), "radius": float}

        self.step_count = 0
        self.goal_pos = np.zeros(3, dtype=float)
        self.desired_pos = np.zeros(3, dtype=float)

        self._connect_and_build()

    def _connect_and_build(self):
        if self.render_mode == "human":
            self._p = bc.BulletClient(connection_mode=p.GUI)
            self.cid = self._p._client
            self._p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=55,
                cameraPitch=-35,
                cameraTargetPosition=[0.55, 0.0, 0.75],
            )
        else:
            self._p = bc.BulletClient(connection_mode=p.DIRECT)
            self.cid = self._p._client

        if self.cid < 0:
            raise RuntimeError("PyBullet connect 실패")

        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.resetSimulation()
        self._p.setGravity(0, 0, -9.81)
        self._p.setTimeStep(1 / 240)
        self._p.setRealTimeSimulation(0)
        self._p.setPhysicsEngineParameter(numSolverIterations=200, fixedTimeStep=1 / 240)

        # 기본 월드
        self._p.loadURDF("plane.urdf")
        self.table_id = self._p.loadURDF("table/table.urdf", [0.5, 0.0, 0.0], useFixedBase=True)
        self.table_top_z = get_body_top_z(self.table_id, pb=self._p)

        # 로봇을 테이블 위에 설치
        robot_base_pos = [0.35, 0.0, self.table_top_z + 0.001]
        self.robot = self._p.loadURDF("franka_panda/panda.urdf", robot_base_pos, useFixedBase=True)

        # v2: IK 목표 링크 겸 측정점을 panda_grasptarget(TCP, 손가락 끝 사이)으로 변경.
        # v0/v1에서는 panda_hand를 썼는데, getLinkState()[0]은 링크 프레임이 아니라 질량중심(COM)이라
        # IK가 맞춘 점과 측정한 점이 수 cm 어긋나 success 반경(3.5 cm)에 들어갈 수 없었음.
        self.eef_link = find_link_index(self.robot, "panda_grasptarget", pb=self._p)
        self.joint_damping = [0.06] * self._p.getNumJoints(self.robot)

        # 장애물 3개 (테이블 위)
        self.obstacle_ids.clear()
        self.obstacles.clear()

        box_specs = [
            ([0.06, 0.06, 0.06], [0.62, 0.18, self.table_top_z + 0.06], (0.8, 0.8, 0.2, 1)),
            ([0.10, 0.04, 0.05], [0.72, -0.12, self.table_top_z + 0.05], (0.2, 0.8, 0.8, 1)),
            ([0.05, 0.10, 0.04], [0.52, -0.22, self.table_top_z + 0.04], (0.9, 0.4, 0.4, 1)),
        ]
        margin = 0.08  # 회피/샘플링 여유

        for he, pos, color in box_specs:
            oid = make_box(he, pos, rgba=color, pb=self._p)
            self.obstacle_ids.append(oid)
            he_v = np.array(he, dtype=float)
            pos_v = np.array(pos, dtype=float)
            radius = float(np.linalg.norm(he_v) + margin)
            self.obstacles.append({"center": pos_v, "radius": radius})

        # 목표/디자이어드 구체(시각화)
        r = 0.025
        col_s = self._p.createCollisionShape(self._p.GEOM_SPHERE, radius=r)
        goal_vis = self._p.createVisualShape(self._p.GEOM_SPHERE, radius=r, rgbaColor=[0.2, 1.0, 0.2, 1])     # 초록(Goal)
        des_vis  = self._p.createVisualShape(self._p.GEOM_SPHERE, radius=r, rgbaColor=[0.2, 0.6, 1.0, 1])     # 파랑(Desired)

        self.goal_id = self._p.createMultiBody(0, col_s, goal_vis, [0.7, 0.0, self.table_top_z + 0.25])
        self.desired_id = self._p.createMultiBody(0, col_s, des_vis, [0.6, 0.0, self.table_top_z + 0.25])

        # 작업공간 bounds (테이블 위)
        x_min, x_max = 0.35, 0.90
        y_min, y_max = -0.35, 0.35
        z_min, z_max = self.table_top_z + 0.12, self.table_top_z + 0.60
        self.bounds = (x_min, x_max, y_min, y_max, z_min, z_max)

    def _eef_pos(self) -> np.ndarray:
        ls = self._p.getLinkState(self.robot, self.eef_link, computeForwardKinematics=True)
        return np.array(ls[0], dtype=float)

    def _clamp_to_bounds(self, pos: np.ndarray) -> np.ndarray:
        x_min, x_max, y_min, y_max, z_min, z_max = self.bounds
        pos2 = pos.copy()
        pos2[0] = np.clip(pos2[0], x_min, x_max)
        pos2[1] = np.clip(pos2[1], y_min, y_max)
        pos2[2] = np.clip(pos2[2], z_min, z_max)
        return pos2

    def _sample_pos_away_from_obstacles(self) -> np.ndarray:
        x_min, x_max, y_min, y_max, z_min, z_max = self.bounds
        for _ in range(200):
            cand = np.array([
                self.rng.uniform(x_min, x_max),
                self.rng.uniform(y_min, y_max),
                self.rng.uniform(z_min, z_max),
            ], dtype=float)
            ok = True
            for ob in self.obstacles:
                if np.linalg.norm(cand - ob["center"]) < ob["radius"]:
                    ok = False
                    break
            if ok:
                return cand
        # 실패하면 그냥 리턴(드물게)
        return np.array([0.65, 0.0, self.table_top_z + 0.25], dtype=float)

    def _collision_with_obstacles(self) -> bool:
        # 접촉(contact) 기준
        for oid in self.obstacle_ids:
            if self.collision_dist <= 0.0:
                pts = self._p.getContactPoints(bodyA=self.robot, bodyB=oid)
                if pts:
                    return True
            else:
                pts = self._p.getClosestPoints(bodyA=self.robot, bodyB=oid, distance=self.collision_dist)
                if pts:
                    return True
        return False

    def _obs(self) -> np.ndarray:
        eef = self._eef_pos()
        goal = self.goal_pos
        des = self.desired_pos
        delta = eef - goal
        obs = np.concatenate([
            eef, goal, des, delta,
            self.obstacles[0]["center"],
            self.obstacles[1]["center"],
            self.obstacles[2]["center"],
        ], axis=0).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # 홈 포즈로 리셋
        home_q = [0.0, -0.6, 0.0, -2.0, 0.0, 1.6, 0.8]
        for j, qv in zip(self.arm_joints, home_q):
            self._p.resetJointState(self.robot, j, qv)

        # 한 번 시뮬 진행해서 FK 안정화
        for _ in range(10):
            self._p.stepSimulation()

        # goal 샘플링(장애물 근처 피함)
        self.goal_pos = self._sample_pos_away_from_obstacles()
        self._p.resetBasePositionAndOrientation(self.goal_id, self.goal_pos.tolist(), [0, 0, 0, 1])

        # desired는 현재 eef 근처에서 시작
        eef = self._eef_pos()
        self.desired_pos = self._clamp_to_bounds(eef.copy())
        self._p.resetBasePositionAndOrientation(self.desired_id, self.desired_pos.tolist(), [0, 0, 0, 1])

        info = {"distance": float(np.linalg.norm(self._eef_pos() - self.goal_pos))}
        return self._obs(), info

    def step(self, action):
        self.step_count += 1

        # 6방향 이동
        dx = np.array([self.step_size, 0.0, 0.0])
        dy = np.array([0.0, self.step_size, 0.0])
        dz = np.array([0.0, 0.0, self.step_size])

        if action == 0:      # +x
            self.desired_pos += dx
        elif action == 1:    # -x
            self.desired_pos -= dx
        elif action == 2:    # +y
            self.desired_pos += dy
        elif action == 3:    # -y
            self.desired_pos -= dy
        elif action == 4:    # +z
            self.desired_pos += dz
        elif action == 5:    # -z
            self.desired_pos -= dz

        self.desired_pos = self._clamp_to_bounds(self.desired_pos)
        self._p.resetBasePositionAndOrientation(self.desired_id, self.desired_pos.tolist(), [0, 0, 0, 1])

        # IK로 desired_pos로 이동
        target_orn = self._p.getQuaternionFromEuler([np.pi, 0, 0])
        q = self._p.calculateInverseKinematics(
            self.robot,
            self.eef_link,
            self.desired_pos.tolist(),
            target_orn,
            jointDamping=self.joint_damping,
            maxNumIterations=200,
            residualThreshold=1e-4,
        )

        for j in self.arm_joints:
            self._p.setJointMotorControl2(
                self.robot,
                j,
                self._p.POSITION_CONTROL,
                targetPosition=q[j],
                force=35,
                maxVelocity=0.9,
            )

        # 실제 이동이 따라오도록 substeps만큼 시뮬
        for _ in range(self.substeps):
            self._p.stepSimulation()
            if self.render_mode == "human":
                time.sleep(1 / 240)

        eef = self._eef_pos()
        dist = float(np.linalg.norm(eef - self.goal_pos))

        collided = self._collision_with_obstacles()
        success = dist < self.success_dist

        # reward 설계(기본형)
        reward = -dist  # 가까울수록 덜 마이너스
        reward -= 0.01  # step penalty

        terminated = False
        truncated = False

        if success:
            reward += 10.0
            terminated = True

        if collided:
            reward -= 5.0
            terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        info = {
            "distance": dist,
            "success": success,
            "collided": collided,
            "step": self.step_count,
        }

        return self._obs(), float(reward), terminated, truncated, info

    def close(self):
        try:
            if self.cid is not None:
                self._p.disconnect()
        except Exception:
            pass
        self.cid = None


def train_dqn(model_path="dqn_panda_discrete.zip", total_timesteps=50_000):
    from stable_baselines3 import DQN
    from stable_baselines3.common.env_util import make_vec_env

    # 학습은 DIRECT(빠름)
    def env_fn():
        return PandaDiscreteReachEnv(render_mode="none", seed=0)

    # v1: 각 env가 자기 BulletClient(physics client)를 가지므로 4개 env가 서로 독립된 시뮬레이션을 돌린다.
    # (v0에서는 모든 인스턴스가 client 0을 공유해 하나의 로봇/월드를 번갈아 조작했음)
    vec_env = make_vec_env(env_fn, n_envs=4)

    model = DQN(
        "MlpPolicy",
        vec_env,
        verbose=1,
        buffer_size=100_000,
        learning_rate=1e-3,
        batch_size=256,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        tensorboard_log="./tb_logs",
    )

    tb_name = os.path.splitext(os.path.basename(model_path))[0]
    model.learn(total_timesteps=total_timesteps, tb_log_name=tb_name)
    model.save(model_path)
    vec_env.close()
    print("Saved:", model_path)


def play(model_path="dqn_panda_discrete.zip"):
    from stable_baselines3 import DQN

    env = PandaDiscreteReachEnv(render_mode="human", seed=0)

    if os.path.exists(model_path):
        model = DQN.load(model_path)
        print("Loaded:", model_path)
    else:
        model = None
        print("No model found. Random actions.")

    obs, info = env.reset()
    while True:
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(int(action))

        if terminated or truncated:
            obs, info = env.reset()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "play"], help="train: 학습 / play: GUI로 보기")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--model", type=str, default="dqn_panda_discrete.zip")
    args = parser.parse_args()

    if args.mode == "train":
        train_dqn(model_path=args.model, total_timesteps=args.timesteps)
    else:
        play(model_path=args.model)