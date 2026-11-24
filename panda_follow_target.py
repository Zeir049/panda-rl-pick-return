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

def main():
    cid = p.connect(p.GUI)
    if cid < 0:
        raise RuntimeError("PyBullet GUI 연결 실패")

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1/240)
    p.setRealTimeSimulation(0)
    p.setPhysicsEngineParameter(numSolverIterations=200, fixedTimeStep=1/240)

    # 카메라 고정
    p.resetDebugVisualizerCamera(
        cameraDistance=1.4,
        cameraYaw=55,
        cameraPitch=-35,
        cameraTargetPosition=[0.5, 0.0, 0.75]
    )

    p.loadURDF("plane.urdf")

    # 테이블 고정
    table_id = p.loadURDF("table/table.urdf", [0.5, 0.0, 0.0], useFixedBase=True)
    table_top_z = get_body_top_z(table_id)

    # 로봇을 테이블 위에 설치(중요!)
    robot_base_pos = [0.35, 0.0, table_top_z + 0.001]
    robot = p.loadURDF("franka_panda/panda.urdf", robot_base_pos, useFixedBase=True)

    arm_joints = list(range(7))
    eef_link = 11  # panda_hand가 보통 11

    # 안정적인 홈 포즈
    home_q = [0.0, -0.6, 0.0, -2.0, 0.0, 1.6, 0.8]
    for j, qv in zip(arm_joints, home_q):
        p.resetJointState(robot, j, qv)

    # 목표(빨간 구체) 생성
    sphere_radius = 0.03
    target_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=sphere_radius)
    target_visual = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=sphere_radius,
        rgbaColor=[1, 0, 0, 1]
    )

    target_pos = np.array([0.55, 0.0, table_top_z + 0.25], dtype=float)
    target_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=target_collision,
        baseVisualShapeIndex=target_visual,
        basePosition=target_pos.tolist()
    )

    target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
    joint_damping = [0.06] * p.getNumJoints(robot)

    # 조작 파라미터
    move_step = 0.003  # 한 프레임당 이동량(빠르면 줄이기)
    paused = False
    auto_orbit = False
    omega = 0.6  # 자동 원운동 속도(토글용)

    # 목표 위치 제한(테이블 아래로 안 가게)
    x_min, x_max = 0.35, 0.85
    y_min, y_max = -0.35, 0.35
    z_min, z_max = table_top_z + 0.12, table_top_z + 0.60

    # UI 텍스트 업데이트용 핸들
    ui_id = -1

    while True:
        if not p.isConnected():
            return

        events = p.getKeyboardEvents()

        # 종료
        if triggered(events, ord('q')) or triggered(events, ord('Q')):
            print("Quit")
            break

        # 일시정지
        if triggered(events, ord(' ')):  # Space
            paused = not paused

        # 자동 원운동 토글
        if triggered(events, ord('o')) or triggered(events, ord('O')):
            auto_orbit = not auto_orbit

        # 속도 조절
        if triggered(events, ord('+')) or triggered(events, ord('=')):
            move_step = min(move_step + 0.001, 0.02)
        if triggered(events, ord('-')) or triggered(events, ord('_')):
            move_step = max(move_step - 0.001, 0.001)

        # 목표 리셋(센터)
        if triggered(events, ord('c')) or triggered(events, ord('C')):
            target_pos[:] = [0.55, 0.0, table_top_z + 0.25]

        if not paused:
            # 수동 조작 (WASD + RF)
            # W/S: x 전후, A/D: y 좌우, R/F: z 상하
            if is_down(events, ord('w')) or is_down(events, ord('W')):
                target_pos[0] += move_step
            if is_down(events, ord('s')) or is_down(events, ord('S')):
                target_pos[0] -= move_step
            if is_down(events, ord('a')) or is_down(events, ord('A')):
                target_pos[1] += move_step
            if is_down(events, ord('d')) or is_down(events, ord('D')):
                target_pos[1] -= move_step
            if is_down(events, ord('r')) or is_down(events, ord('R')):
                target_pos[2] += move_step
            if is_down(events, ord('f')) or is_down(events, ord('F')):
                target_pos[2] -= move_step

            # 자동 원운동(토글 켜졌을 때)
            if auto_orbit:
                t = time.time()
                target_pos[0] = 0.55 + 0.08 * np.cos(omega * t)
                target_pos[1] = 0.00 + 0.08 * np.sin(omega * t)
                target_pos[2] = table_top_z + 0.25

            # 범위 제한(테이블 아래/밖으로 못 가게)
            target_pos[0] = float(np.clip(target_pos[0], x_min, x_max))
            target_pos[1] = float(np.clip(target_pos[1], y_min, y_max))
            target_pos[2] = float(np.clip(target_pos[2], z_min, z_max))

            # 목표 위치 업데이트
            p.resetBasePositionAndOrientation(target_id, target_pos.tolist(), [0, 0, 0, 1])

            # IK
            q = p.calculateInverseKinematics(
                robot, eef_link, target_pos.tolist(), target_orn,
                jointDamping=joint_damping,
                maxNumIterations=200,
                residualThreshold=1e-4
            )

            # 관절 제어(너무 빠르면 maxVelocity/force 낮추기)
            for j in arm_joints:
                p.setJointMotorControl2(
                    robot, j,
                    p.POSITION_CONTROL,
                    targetPosition=q[j],
                    force=35,
                    maxVelocity=0.9
                )

            p.stepSimulation()

        # 화면에 상태 표시(좌상단)
        text = (
            f"WASD: XY 이동 | R/F: Z 이동 | Space: Pause | O: AutoOrbit | C: Center | +/-: Speed | Q: Quit\n"
            f"target={target_pos.round(3).tolist()}  step={move_step:.3f}  paused={paused}  auto={auto_orbit}"
        )
        ui_id = p.addUserDebugText(
            text, [0.02, 0.02, 1.0],
            textColorRGB=[1, 1, 1],
            textSize=1.2,
            replaceItemUniqueId=ui_id
        )

        time.sleep(1/240)

    p.disconnect()

if __name__ == "__main__":
    main()