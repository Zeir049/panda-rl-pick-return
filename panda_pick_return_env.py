import os
import time
import argparse
import numpy as np

import pybullet as p
from pybullet_utils import bullet_client as bc
import pybullet_data

import gymnasium as gym
from gymnasium import spaces


def find_joint_index(body_id: int, name_contains: str, pb=p):
    for i in range(pb.getNumJoints(body_id)):
        info = pb.getJointInfo(body_id, i)
        jname = info[1].decode("utf-8")
        if name_contains in jname:
            return i
    return None


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
        basePosition=pos
    )
    return body


class PandaPickReturnEnv(gym.Env):
    """
    목표:
    1) 장애물 옆(고정)에 놓인 초록 물체를 집고
    2) 들어올린 뒤
    3) 홈 자세로 복귀 (물체를 들고 있으면 성공)

    - 파란 공(waypoint) 없음
    - Action: 관절 7개 delta + 그리퍼 1개(open/close)
    - grasp는 안정적으로 하기 위해 조건 만족 시 object를 end-effector에 fixed constraint로 "부착"
    """

    metadata = {"render_modes": ["human", "none"], "render_fps": 60}

    def __init__(
        self,
        render_mode="none",
        max_steps=250,
        substeps=20,
        joint_delta_scale=0.06,      # action(±1)이 관절각에 주는 변화량(rad)
        gripper_speed=0.02,          # 그리퍼 목표 이동량(m) per step
        success_joint_tol=0.25,      # 홈 복귀 판정(관절 거리)
        grasp_dist=0.045,            # 이 거리 이내 + 그리퍼 닫힘이면 grasp 허용
        lift_height=0.10,            # 상판 대비 이만큼 들면 lift 성공
        seed=0,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.substeps = int(substeps)
        self.joint_delta_scale = float(joint_delta_scale)
        self.gripper_speed = float(gripper_speed)
        self.success_joint_tol = float(success_joint_tol)
        self.grasp_dist = float(grasp_dist)
        self.lift_height = float(lift_height)
        self.rng = np.random.default_rng(seed)

        # action: 7 arm joints + 1 gripper cmd
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

        # obs: q(7) + dq(7) + gripper_open(1) + eef_pos(3) + obj_pos(3) + (eef-obj)(3) + flags(2: grasped, lifted) = 26
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32)

        # pybullet handles
        self.cid = None
        self.table_id = None
        self.robot = None
        self.obj_id = None
        self.obstacle_ids = []

        # indices
        self.arm_joints = list(range(7))
        self.eef_link = None
        self.finger_j1 = None
        self.finger_j2 = None

        # state
        self.step_count = 0
        self.table_top_z = None
        self.home_q = np.array([0.0, -0.6, 0.0, -2.0, 0.0, 1.6, 0.8], dtype=float)
        self.target_q = self.home_q.copy()
        self.gripper_open = 0.04  # 0~0.04 정도가 안전(판다)
        self.grasped = False
        self.lifted = False
        self.attach_cid = None

        self._build()

    def _build(self):
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

        self._p.loadURDF("plane.urdf")
        self.table_id = self._p.loadURDF("table/table.urdf", [0.5, 0.0, 0.0], useFixedBase=True)
        self.table_top_z = get_body_top_z(self.table_id, pb=self._p)

        # 로봇을 테이블 위에 올림
        robot_base_pos = [0.35, 0.0, self.table_top_z + 0.001]
        self.robot = self._p.loadURDF("franka_panda/panda.urdf", robot_base_pos, useFixedBase=True)

        # v1: end-effector 기준점을 panda_hand(손목, 손가락 끝보다 ~10 cm 위)에서
        # panda_grasptarget(두 손가락 끝 사이, TCP)으로 변경. 관측/보상/grasp 판정 모두 이 점을 사용.
        # v0에서는 panda_hand 기준이라 손가락이 물체/테이블에 먼저 닿아 grasp 조건(4.5 cm)을 만족할 수 없었음.
        self.eef_link = find_link_index(self.robot, "panda_grasptarget", pb=self._p)

        # finger joint 찾기(이름 기반)
        self.finger_j1 = find_joint_index(self.robot, "finger_joint1", pb=self._p)
        self.finger_j2 = find_joint_index(self.robot, "finger_joint2", pb=self._p)
        if self.finger_j1 is None or self.finger_j2 is None:
            # 일부 버전에서 이름이 다를 수 있어 fallback (보통 9,10 근처)
            # 그래도 최소한 학습이 돌게 마지막 두 개 prismatic을 찾는 방식
            candidates = []
            for i in range(self._p.getNumJoints(self.robot)):
                info = self._p.getJointInfo(self.robot, i)
                jtype = info[2]
                if jtype == self._p.JOINT_PRISMATIC:
                    candidates.append(i)
            if len(candidates) >= 2:
                self.finger_j1, self.finger_j2 = candidates[-2], candidates[-1]

        # 장애물 3개 (물체 근처로 배치)
        self.obstacle_ids.clear()
        box1_he = [0.06, 0.06, 0.06]
        box2_he = [0.10, 0.04, 0.05]
        box3_he = [0.05, 0.10, 0.04]

        box1_pos = [0.62, 0.18, self.table_top_z + box1_he[2]]
        box2_pos = [0.72, -0.12, self.table_top_z + box2_he[2]]
        box3_pos = [0.52, -0.22, self.table_top_z + box3_he[2]]

        self.obstacle_ids.append(make_box(box1_he, box1_pos, rgba=(0.8, 0.8, 0.2, 1), pb=self._p))
        self.obstacle_ids.append(make_box(box2_he, box2_pos, rgba=(0.2, 0.8, 0.8, 1), pb=self._p))
        self.obstacle_ids.append(make_box(box3_he, box3_pos, rgba=(0.9, 0.4, 0.4, 1), pb=self._p))

        # ✅ 초록 물체(집을 대상) : 장애물(box1) 근처에 "고정 위치"로 배치
        # 박스1 옆으로 아주 가깝게 둠(너 요청)
        obj_size = 0.03
        obj_col = self._p.createCollisionShape(self._p.GEOM_BOX, halfExtents=[obj_size, obj_size, obj_size])
        obj_vis = self._p.createVisualShape(self._p.GEOM_BOX, halfExtents=[obj_size, obj_size, obj_size], rgbaColor=[0.2, 1.0, 0.2, 1])

        # box1_pos 근처 고정(약간 옆/앞)
        obj_pos = [0.62, 0.10, self.table_top_z + obj_size]  # 장애물 쪽에 붙여둠
        self.obj_id = self._p.createMultiBody(
            baseMass=0.05,  # 물리적으로 들어올릴 수 있게
            baseCollisionShapeIndex=obj_col,
            baseVisualShapeIndex=obj_vis,
            basePosition=obj_pos
        )
        self._p.changeDynamics(self.obj_id, -1, lateralFriction=1.2, spinningFriction=0.01, rollingFriction=0.01)

    def _eef_pos(self):
        ls = self._p.getLinkState(self.robot, self.eef_link, computeForwardKinematics=True)
        return np.array(ls[0], dtype=float)

    def _obj_pos(self):
        pos, orn = self._p.getBasePositionAndOrientation(self.obj_id)
        return np.array(pos, dtype=float)

    def _get_joint_state(self):
        q = []
        dq = []
        for j in self.arm_joints:
            js = self._p.getJointState(self.robot, j)
            q.append(js[0])
            dq.append(js[1])
        return np.array(q, dtype=float), np.array(dq, dtype=float)

    def _set_arm_targets(self, q_target):
        for i, j in enumerate(self.arm_joints):
            self._p.setJointMotorControl2(
                self.robot, j,
                self._p.POSITION_CONTROL,
                targetPosition=float(q_target[i]),
                force=50,
                maxVelocity=1.2,
            )

    def _set_gripper(self, open_width):
        # 판다 finger는 각 finger가 절반씩 움직이는 형태라 보통 0~0.04 범위를 줌
        w = float(np.clip(open_width, 0.0, 0.04))
        if self.finger_j1 is not None:
            self._p.setJointMotorControl2(self.robot, self.finger_j1, self._p.POSITION_CONTROL, targetPosition=w, force=30)
        if self.finger_j2 is not None:
            self._p.setJointMotorControl2(self.robot, self.finger_j2, self._p.POSITION_CONTROL, targetPosition=w, force=30)
        self.gripper_open = w

    def _robot_collided(self) -> bool:
        for oid in self.obstacle_ids:
            if self._p.getContactPoints(bodyA=self.robot, bodyB=oid):
                return True
        # 테이블이랑도 과격한 접촉하면 패널티 주고 싶으면 여기 추가 가능
        return False

    def _try_grasp_attach(self):
        # grasp 조건: eef가 obj 근처 + 그리퍼 충분히 닫힘
        if self.grasped:
            return

        eef = self._eef_pos()
        obj = self._obj_pos()
        dist = float(np.linalg.norm(eef - obj))

        if dist < self.grasp_dist and self.gripper_open < 0.012:
            # 물체를 eef에 fixed constraint로 부착(프로토타입에서 안정적)
            # eef가 이제 TCP(손가락 끝 사이)이므로 물체 중심을 TCP에 그대로 부착
            self.attach_cid = self._p.createConstraint(
                parentBodyUniqueId=self.robot,
                parentLinkIndex=self.eef_link,
                childBodyUniqueId=self.obj_id,
                childLinkIndex=-1,
                jointType=self._p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0.0, 0.0, 0.0],
                childFramePosition=[0.0, 0.0, 0.0],
            )
            self.grasped = True

    def _check_lifted(self):
        if not self.grasped:
            self.lifted = False
            return
        obj = self._obj_pos()
        self.lifted = (obj[2] > self.table_top_z + self.lift_height)

    def _obs(self):
        q, dq = self._get_joint_state()
        eef = self._eef_pos()
        obj = self._obj_pos()
        delta = eef - obj
        flags = np.array([1.0 if self.grasped else 0.0,
                          1.0 if self.lifted else 0.0], dtype=float)
        obs = np.concatenate([q, dq, [self.gripper_open], eef, obj, delta, flags], axis=0)
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # constraint 제거
        if self.attach_cid is not None:
            try:
                self._p.removeConstraint(self.attach_cid)
            except Exception:
                pass
        self.attach_cid = None
        self.grasped = False
        self.lifted = False

        # 로봇 홈 리셋
        self.target_q = self.home_q.copy()
        for i, j in enumerate(self.arm_joints):
            self._p.resetJointState(self.robot, j, float(self.home_q[i]))

        # 그리퍼 open
        self._set_gripper(0.04)

        # 물체 위치는 "고정" 유지 (원하면 미세 랜덤도 가능)
        obj_size = 0.03
        fixed_obj_pos = [0.62, 0.10, self.table_top_z + obj_size]
        self._p.resetBasePositionAndOrientation(self.obj_id, fixed_obj_pos, [0, 0, 0, 1])
        self._p.resetBaseVelocity(self.obj_id, [0, 0, 0], [0, 0, 0])

        # 안정화 스텝
        for _ in range(10):
            self._p.stepSimulation()

        info = {}
        return self._obs(), info

    def step(self, action):
        self.step_count += 1
        action = np.array(action, dtype=float).clip(-1, 1)

        # 관절 타겟 업데이트(7개)
        dq_cmd = action[:7] * self.joint_delta_scale
        self.target_q = self.target_q + dq_cmd

        # 관절 제한 clamp (URDF joint limit 참고)
        for i, j in enumerate(self.arm_joints):
            info = self._p.getJointInfo(self.robot, j)
            lo = info[8]
            hi = info[9]
            # hi/lo가 유효하지 않은 경우가 있어 fallback
            if lo < hi and abs(lo) < 1e10 and abs(hi) < 1e10:
                self.target_q[i] = float(np.clip(self.target_q[i], lo, hi))

        # 그리퍼 제어(1개): +면 open, -면 close
        grip_cmd = float(action[7])
        self._set_gripper(self.gripper_open + grip_cmd * self.gripper_speed)

        # 모터 적용
        self._set_arm_targets(self.target_q)

        # 시뮬 진행
        for _ in range(self.substeps):
            self._p.stepSimulation()
            if self.render_mode == "human":
                time.sleep(1 / 240)

        # grasp 시도 및 lift 상태 갱신
        self._try_grasp_attach()
        self._check_lifted()

        # 보상 설계
        eef = self._eef_pos()
        obj = self._obj_pos()
        dist = float(np.linalg.norm(eef - obj))

        # 기본: 물체에 가까워지기
        reward = -dist
        reward -= 0.01  # step penalty
        reward -= 0.001 * float(np.sum(np.square(action)))  # action penalty

        # grasp / lift 보상
        if self.grasped:
            reward += 1.5
        if self.lifted:
            reward += 2.5

        # 홈 복귀 보상(물체 들고 있어야 의미 있음)
        q, _ = self._get_joint_state()
        home_err = float(np.linalg.norm(q - self.home_q))
        if self.grasped and self.lifted:
            reward += 0.3 * (1.0 / (1.0 + home_err))  # 홈에 가까울수록 +

        # 충돌 패널티
        collided = self._robot_collided()
        if collided:
            reward -= 3.0

        # 성공 조건: grasped + lifted + home pose 근처
        success = (self.grasped and self.lifted and home_err < self.success_joint_tol)

        terminated = False
        truncated = False

        if success:
            reward += 12.0
            terminated = True

        # 충돌이 심각하면 종료하고 싶다면 여기서 terminate 해도 됨
        # if collided:
        #     terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        info = {
            "dist": dist,
            "home_err": home_err,
            "grasped": self.grasped,
            "lifted": self.lifted,
            "collided": collided,
            "success": success,
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


def train(model_path="ppo_panda_pick_return.zip", timesteps=300_000):
    from stable_baselines3 import PPO

    # Windows 안정: n_envs=1로 고정
    env = PandaPickReturnEnv(render_mode="none", seed=0)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        learning_rate=3e-4,
        ent_coef=0.0,
        tensorboard_log="./tb_logs",
        clip_range=0.2,
    )

    tb_name = os.path.splitext(os.path.basename(model_path))[0]
    model.learn(total_timesteps=timesteps, tb_log_name=tb_name)
    model.save(model_path)
    env.close()
    print("Saved:", model_path)


def play(model_path="ppo_panda_pick_return.zip"):
    from stable_baselines3 import PPO

    env = PandaPickReturnEnv(render_mode="human", seed=0)

    if os.path.exists(model_path):
        model = PPO.load(model_path)
        print("Loaded:", model_path)
    else:
        model = None
        print("No model found -> random actions")

    obs, _ = env.reset()
    while True:
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "play"])
    parser.add_argument("--timesteps", type=int, default=300000)
    parser.add_argument("--model", type=str, default="ppo_panda_pick_return.zip")
    args = parser.parse_args()

    if args.mode == "train":
        train(model_path=args.model, timesteps=args.timesteps)
    else:
        play(model_path=args.model)