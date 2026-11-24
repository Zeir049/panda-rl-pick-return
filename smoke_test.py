"""PyBullet 설치 확인용: plane + table + Panda를 불러와 10초(2400 step) 시뮬레이션.

    python smoke_test.py            # GUI 창
    python smoke_test.py --direct   # 창 없이 (headless)
"""
import sys
import time
import pybullet as p
import pybullet_data

headless = "--direct" in sys.argv

p.connect(p.DIRECT if headless else p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

p.loadURDF("plane.urdf")
p.loadURDF("table/table.urdf", [0.5, 0, 0])
robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

for _ in range(2400):
    p.stepSimulation()
    if not headless:
        time.sleep(1/240)

print(f"OK: Panda loaded with {p.getNumJoints(robot)} joints, 2400 steps simulated")
p.disconnect()
