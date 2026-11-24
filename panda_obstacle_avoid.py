import time
import numpy as np
import pybullet as p
import pybullet_data


def get_body_top_z(body_id: int) -> float:
    """바디 전체 링크의 AABB를 훑어서 가장 높은 z(top)를 찾음"""
    top_z = -1e9
    aabb_min, aabb_max = p.getAABB(body_id, -1)  # base
    top_z = max(top_z, aabb_max[2])
    for link in range(p.getNumJoints(body_id)):
        aabb_min, aabb_max = p.getAABB(body_id, link)
        top_z = max(top_z, aabb_max[2])
    return top_z


def is_down(events, key):
    return key in events and (events[key] & p.KEY_IS_DOWN)


def triggered(events, key):
    return key in events and (events[key] & p.KEY_WAS_TRIGGERED)


def make_box(half_extents, pos, rgba=(0.7, 0.7, 0.7, 1.0)):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba)
    body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=pos)
    return body


def clamp(v, lo, hi):
    return float(np.clip(v, lo, hi))


def push_out_of_obstacles(pos, obstacles):
    """
    pos: (3,) target position
    obstacles: list of dict {center, radius}
    -> pos를 각 장애물 안전반경 밖으로 밀어냄
    """
    p2 = pos.copy()
    for ob in obstacles:
        c = ob["center"]
        r = ob["radius"]
        d = p2 - c
        dist = float(np.linalg.norm(d))
        if dist < 1e-8:
            # 정 중앙이면 임의 방향으로 밀어냄
            d = np.array([1.0, 0.0, 0.0], dtype=float)
            dist = 1.0
        if dist < r:
            # 안전 반경 밖으로 투영
            p2 = c + (d / dist) * r
    return p2


def main():
    cid = p.connect(p.GUI)
    if cid < 0:
        raise RuntimeError("PyBullet GUI 연결 실패")

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1 / 240)
    p.setRealTimeSimulation(0)
    p.setPhysicsEngineParameter(numSolverIterations=200, fixedTimeStep=1 / 240)

    # 시점 고정
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=55,
        cameraPitch=-35,
        cameraTargetPosition=[0.55, 0.0, 0.75],
    )

    p.loadURDF("plane.urdf")
    table_id = p.loadURDF("table/table.urdf", [0.5, 0.0, 0.0], useFixedBase=True)
    table_top_z = get_body_top_z(table_id)

    # 로봇을 테이블 위에 설치
    robot_base_pos = [0.35, 0.0, table_top_z + 0.001]
    robot = p.loadURDF("franka_panda/panda.urdf", robot_base_pos, useFixedBase=True)

    arm_joints = list(range(7))
    eef_link = 11

    # 홈 포즈
    home_q = [0.0, -0.6, 0.0, -2.0, 0.0, 1.6, 0.8]
    for j, qv in zip(arm_joints, home_q):
        p.resetJointState(robot, j, qv)

    joint_damping = [0.06] * p.getNumJoints(robot)
    target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])

    # ===== 목표점 2개: "원하는 목표(파란)" vs "안전 목표(빨간)"
    r = 0.025
    col_s = p.createCollisionShape(p.GEOM_SPHERE, radius=r)

    vis_des = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=[0.2, 0.6, 1.0, 1])  # 파란(원하는 목표)
    vis_safe = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=[1.0, 0.2, 0.2, 1])  # 빨간(실제 목표)

    desired_pos = np.array([0.60, 0.00, table_top_z + 0.25], dtype=float)
    safe_pos = desired_pos.copy()

    desired_id = p.createMultiBody(0, col_s, vis_des, desired_pos.tolist())
    safe_id = p.createMultiBody(0, col_s, vis_safe, safe_pos.tolist())

    # ===== 장애물 배치(테이블 위)
    # 박스 크기(half extents)
    box1_he = [0.06, 0.06, 0.06]
    box2_he = [0.10, 0.04, 0.05]
    box3_he = [0.05, 0.10, 0.04]

    # 박스 위치(z는 상판 위로)
    box1_pos = [0.62, 0.18, table_top_z + box1_he[2]]
    box2_pos = [0.72, -0.12, table_top_z + box2_he[2]]
    box3_pos = [0.52, -0.22, table_top_z + box3_he[2]]

    ob1 = make_box(box1_he, box1_pos, rgba=(0.8, 0.8, 0.2, 1))
    ob2 = make_box(box2_he, box2_pos, rgba=(0.2, 0.8, 0.8, 1))
    ob3 = make_box(box3_he, box3_pos, rgba=(0.9, 0.4, 0.4, 1))

    # 장애물 “안전 반경”(박스를 구로 근사) + 여유 마진
    margin = 0.08  # 이 값을 키우면 더 멀리 회피함
    obstacles = []
    for he, pos in [(np.array(box1_he), np.array(box1_pos)),
                    (np.array(box2_he), np.array(box2_pos)),
                    (np.array(box3_he), np.array(box3_pos))]:
        radius = float(np.linalg.norm(he) + margin)
        obstacles.append({"center": pos.astype(float), "radius": radius})

    obstacle_bodies = [ob1, ob2, ob3]

    # ===== 조작 파라미터
    move_step = 0.003
    paused = False
    auto_orbit = False
    avoid_on = True

    # 목표 움직임 범위 제한(테이블 아래로 못 내려가게)
    x_min, x_max = 0.35, 0.90
    y_min, y_max = -0.35, 0.35
    z_min, z_max = table_top_z + 0.12, table_top_z + 0.60

    ui_id = -1
    omega = 0.6

    # ===== 키 설정 (PyBullet 기본 단축키 충돌 피하려고 IJKL 권장)
    # I/K: x, J/L: y, R/F: z
    while True:
        if not p.isConnected():
            return

        events = p.getKeyboardEvents()

        if triggered(events, ord("q")) or triggered(events, ord("Q")):
            print("Quit")
            break

        if triggered(events, ord(" ")):  # Space
            paused = not paused

        if triggered(events, ord("o")) or triggered(events, ord("O")):
            auto_orbit = not auto_orbit

        if triggered(events, ord("h")) or triggered(events, ord("H")):
            avoid_on = not avoid_on

        if triggered(events, ord("+")) or triggered(events, ord("=")):
            move_step = min(move_step + 0.001, 0.02)
        if triggered(events, ord("-")) or triggered(events, ord("_")):
            move_step = max(move_step - 0.001, 0.001)

        if triggered(events, ord("c")) or triggered(events, ord("C")):
            desired_pos[:] = [0.60, 0.00, table_top_z + 0.25]

        if not paused:
            # 수동 조작
            if is_down(events, ord("i")) or is_down(events, ord("I")):
                desired_pos[0] += move_step
            if is_down(events, ord("k")) or is_down(events, ord("K")):
                desired_pos[0] -= move_step
            if is_down(events, ord("j")) or is_down(events, ord("J")):
                desired_pos[1] += move_step
            if is_down(events, ord("l")) or is_down(events, ord("L")):
                desired_pos[1] -= move_step
            if is_down(events, ord("r")) or is_down(events, ord("R")):
                desired_pos[2] += move_step
            if is_down(events, ord("f")) or is_down(events, ord("F")):
                desired_pos[2] -= move_step

            # 자동 원운동
            if auto_orbit:
                t = time.time()
                desired_pos[0] = 0.60 + 0.10 * np.cos(omega * t)
                desired_pos[1] = 0.00 + 0.10 * np.sin(omega * t)
                desired_pos[2] = table_top_z + 0.25

            # 범위 제한
            desired_pos[0] = clamp(desired_pos[0], x_min, x_max)
            desired_pos[1] = clamp(desired_pos[1], y_min, y_max)
            desired_pos[2] = clamp(desired_pos[2], z_min, z_max)

            # 원하는 목표(파란 구) 업데이트
            p.resetBasePositionAndOrientation(desired_id, desired_pos.tolist(), [0, 0, 0, 1])

            # ===== 회피 로직: 원하는 목표를 장애물 반경 밖으로 밀어낸 "안전 목표" 계산
            if avoid_on:
                safe_pos = push_out_of_obstacles(desired_pos, obstacles)
            else:
                safe_pos = desired_pos.copy()

            # 안전 목표도 범위 제한
            safe_pos[0] = clamp(safe_pos[0], x_min, x_max)
            safe_pos[1] = clamp(safe_pos[1], y_min, y_max)
            safe_pos[2] = clamp(safe_pos[2], z_min, z_max)

            # 안전 목표(빨간 구) 업데이트
            p.resetBasePositionAndOrientation(safe_id, safe_pos.tolist(), [0, 0, 0, 1])

            # IK는 안전 목표를 따라감
            q = p.calculateInverseKinematics(
                robot,
                eef_link,
                safe_pos.tolist(),
                target_orn,
                jointDamping=joint_damping,
                maxNumIterations=200,
                residualThreshold=1e-4,
            )

            for j in arm_joints:
                p.setJointMotorControl2(
                    robot,
                    j,
                    p.POSITION_CONTROL,
                    targetPosition=q[j],
                    force=35,
                    maxVelocity=0.9,
                )

            p.stepSimulation()

        # 로봇-장애물 근접/접촉 감지(경고용)
        near = False
        for ob in obstacle_bodies:
            pts = p.getClosestPoints(bodyA=robot, bodyB=ob, distance=0.02)
            if pts:
                near = True
                break

        text = (
            "IJKL: XY 이동 | R/F: Z 이동 | Space: Pause | O: AutoOrbit | C: Center | +/-: Speed | Q: Quit\n"
            f"H: Avoid On/Off (현재: {avoid_on})\n"
            f"desired(파란)={desired_pos.round(3).tolist()}  safe(빨간)={safe_pos.round(3).tolist()}  "
            f"step={move_step:.3f}  paused={paused}  near_obstacle={near}"
        )
        ui_id = p.addUserDebugText(
            text,
            [0.02, 0.02, 1.0],
            textColorRGB=[1, 1, 1],
            textSize=1.2,
            replaceItemUniqueId=ui_id,
        )

        time.sleep(1 / 240)

    p.disconnect()


if __name__ == "__main__":
    main()