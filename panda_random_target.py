import time
import numpy as np
import pybullet as p
import pybullet_data


def get_body_top_z(body_id: int) -> float:
    top_z = -1e9
    aabb_min, aabb_max = p.getAABB(body_id, -1)
    top_z = max(top_z, aabb_max[2])
    for link in range(p.getNumJoints(body_id)):
        aabb_min, aabb_max = p.getAABB(body_id, link)
        top_z = max(top_z, aabb_max[2])
    return top_z


def find_link_index(body_id: int, name_contains: str) -> int:
    n = p.getNumJoints(body_id)
    for i in range(n):
        info = p.getJointInfo(body_id, i)
        link_name = info[12].decode("utf-8")
        if name_contains in link_name:
            return i
    return n - 1


def make_box(half_extents, pos, rgba=(0.7, 0.7, 0.7, 1.0)):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba)
    body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=pos)
    return body


def clamp(v, lo, hi):
    return float(np.clip(v, lo, hi))


def push_out_of_obstacles(pos, obstacles):
    """pos를 각 장애물의 안전반경 밖으로 투영(간단 회피)"""
    p2 = pos.copy()
    for ob in obstacles:
        c = ob["center"]
        r = ob["radius"]
        d = p2 - c
        dist = float(np.linalg.norm(d))
        if dist < 1e-8:
            d = np.array([1.0, 0.0, 0.0], dtype=float)
            dist = 1.0
        if dist < r:
            p2 = c + (d / dist) * r
    return p2


def main():
    cid = p.connect(p.GUI)
    if cid < 0:
        raise RuntimeError("PyBullet GUI 연결 실패")

    # 혹시 W 같은 기본 단축키로 와이어프레임이 켜져도 항상 꺼두기
    p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()

    dt = 1 / 240
    p.setTimeStep(dt)
    p.setRealTimeSimulation(0)
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(numSolverIterations=200, fixedTimeStep=dt)

    # 카메라
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=55,
        cameraPitch=-35,
        cameraTargetPosition=[0.55, 0.0, 0.75],
    )

    p.loadURDF("plane.urdf")

    # 테이블
    table_id = p.loadURDF("table/table.urdf", [0.5, 0.0, 0.0], useFixedBase=True)
    table_top_z = get_body_top_z(table_id)

    # 로봇(테이블 위에 올림)
    robot_base_pos = [0.35, 0.0, table_top_z + 0.001]
    robot = p.loadURDF("franka_panda/panda.urdf", robot_base_pos, useFixedBase=True)

    arm_joints = list(range(7))
    eef_link = find_link_index(robot, "panda_hand")  # 안정적으로 찾기
    print("EEF link index =", eef_link)

    # 홈 포즈
    home_q = [0.0, -0.6, 0.0, -2.0, 0.0, 1.6, 0.8]
    for j, qv in zip(arm_joints, home_q):
        p.resetJointState(robot, j, qv)

    joint_damping = [0.06] * p.getNumJoints(robot)
    target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])

    # ===== 목표점 2개: 원하는 목표(파란) / 안전 목표(빨간)
    r = 0.025
    col_s = p.createCollisionShape(p.GEOM_SPHERE, radius=r)
    vis_des = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=[0.2, 0.6, 1.0, 1])  # 파란
    vis_safe = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=[1.0, 0.2, 0.2, 1])  # 빨간

    desired_pos = np.array([0.60, 0.00, table_top_z + 0.25], dtype=float)
    safe_pos = desired_pos.copy()

    desired_id = p.createMultiBody(0, col_s, vis_des, desired_pos.tolist())
    safe_id = p.createMultiBody(0, col_s, vis_safe, safe_pos.tolist())

    # ===== 장애물(박스) 3개
    box1_he = [0.06, 0.06, 0.06]
    box2_he = [0.10, 0.04, 0.05]
    box3_he = [0.05, 0.10, 0.04]

    box1_pos = [0.62, 0.18, table_top_z + box1_he[2]]
    box2_pos = [0.72, -0.12, table_top_z + box2_he[2]]
    box3_pos = [0.52, -0.22, table_top_z + box3_he[2]]

    make_box(box1_he, box1_pos, rgba=(0.8, 0.8, 0.2, 1))
    make_box(box2_he, box2_pos, rgba=(0.2, 0.8, 0.8, 1))
    make_box(box3_he, box3_pos, rgba=(0.9, 0.4, 0.4, 1))

    # 장애물 “안전 반경”(박스를 구로 근사) + 마진
    margin = 0.08
    obstacles = []
    for he, pos in [(np.array(box1_he), np.array(box1_pos)),
                    (np.array(box2_he), np.array(box2_pos)),
                    (np.array(box3_he), np.array(box3_pos))]:
        radius = float(np.linalg.norm(he) + margin)
        obstacles.append({"center": pos.astype(float), "radius": radius})

    # ===== 목표 이동 범위 제한(테이블 아래로 못 내려가게)
    x_min, x_max = 0.35, 0.90
    y_min, y_max = -0.35, 0.35
    z_min, z_max = table_top_z + 0.12, table_top_z + 0.60

    # ===== 랜덤 이동 파라미터
    rng = np.random.default_rng()
    avoid_on = True

    # 일정 시간마다 새 랜덤 목표점(goal_pos)을 뽑고, desired_pos가 그쪽으로 부드럽게 이동
    goal_pos = desired_pos.copy()
    change_every_sec = 1.2        # 몇 초마다 새 목표 뽑을지
    move_speed = 0.20             # m/s (빠르면 0.12~0.18로 낮추기)
    next_change_t = time.time() + change_every_sec

    ui_id = -1

    try:
        while True:
            if not p.isConnected():
                return

            # 와이어프레임 강제 OFF(혹시 모를 토글 방지)
            p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)

            now = time.time()

            # 새 랜덤 목표점 샘플링
            if now >= next_change_t:
                goal_pos = np.array([
                    rng.uniform(x_min, x_max),
                    rng.uniform(y_min, y_max),
                    rng.uniform(z_min, z_max),
                ], dtype=float)
                next_change_t = now + change_every_sec

            # desired_pos를 goal_pos 쪽으로 "속도 제한"하며 이동(부드러운 랜덤 이동)
            d = goal_pos - desired_pos
            dist = float(np.linalg.norm(d))
            max_step = move_speed * dt

            if dist > 1e-6:
                step = min(dist, max_step)
                desired_pos += (d / dist) * step

            # 범위 제한
            desired_pos[0] = clamp(desired_pos[0], x_min, x_max)
            desired_pos[1] = clamp(desired_pos[1], y_min, y_max)
            desired_pos[2] = clamp(desired_pos[2], z_min, z_max)

            # 원하는 목표(파란)
            p.resetBasePositionAndOrientation(desired_id, desired_pos.tolist(), [0, 0, 0, 1])

            # 회피 적용한 안전 목표(빨간)
            if avoid_on:
                safe_pos = push_out_of_obstacles(desired_pos, obstacles)
            else:
                safe_pos = desired_pos.copy()

            safe_pos[0] = clamp(safe_pos[0], x_min, x_max)
            safe_pos[1] = clamp(safe_pos[1], y_min, y_max)
            safe_pos[2] = clamp(safe_pos[2], z_min, z_max)

            p.resetBasePositionAndOrientation(safe_id, safe_pos.tolist(), [0, 0, 0, 1])

            # IK는 안전 목표를 따라감
            q = p.calculateInverseKinematics(
                robot, eef_link, safe_pos.tolist(), target_orn,
                jointDamping=joint_damping,
                maxNumIterations=200,
                residualThreshold=1e-4,
            )

            for j in arm_joints:
                p.setJointMotorControl2(
                    robot, j,
                    p.POSITION_CONTROL,
                    targetPosition=q[j],
                    force=35,
                    maxVelocity=0.9,
                )

            p.stepSimulation()

            # 화면 표시
            text = (
                "Random target mode (no manual control)\n"
                f"desired(파란)={desired_pos.round(3).tolist()}  safe(빨간)={safe_pos.round(3).tolist()}\n"
                f"move_speed={move_speed:.2f} m/s  change_every={change_every_sec:.1f}s  avoid={avoid_on}"
            )
            ui_id = p.addUserDebugText(
                text, [0.02, 0.02, 1.0],
                textColorRGB=[1, 1, 1],
                textSize=1.2,
                replaceItemUniqueId=ui_id
            )

            time.sleep(dt)

    except KeyboardInterrupt:
        # Ctrl+C로 종료
        pass
    finally:
        p.disconnect()


if __name__ == "__main__":
    main()