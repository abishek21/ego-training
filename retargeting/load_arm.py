"""
Phase 1a: load the SO-101 URDF in PyBullet and inspect its kinematics.
---------------------------------------------------------------------
Verifies the arm model loads (all meshes resolve) and prints every joint's
name, type, axis, limits, and the link chain. Run this FIRST — before any IK —
so we trust the model. Headless by default (DIRECT); pass --gui to visualize.

SO-101 joints (from URDF): 1 base-pan, 2 shoulder-lift, 3 elbow-flex,
4 wrist-flex, 5 wrist-roll, 6 gripper. Wrist has pitch(4)+roll(5) but NO
independent yaw -> we can match position + approach + roll, not full 3-DoF
wrist orientation (documented limitation).

Usage:
    venv_retarget/bin/python retargeting/load_arm.py            # headless summary
    venv_retarget/bin/python retargeting/load_arm.py --gui      # 3D viewer
"""

import argparse
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

URDF = Path(__file__).resolve().parent / "so101" / "so101.urdf"
JOINT_TYPES = {p.JOINT_REVOLUTE: "revolute", p.JOINT_PRISMATIC: "prismatic",
               p.JOINT_FIXED: "fixed", p.JOINT_PLANAR: "planar",
               p.JOINT_SPHERICAL: "spherical"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()

    mode = p.GUI if args.gui else p.DIRECT
    cid = p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # useFixedBase: the arm is bolted to a table, base doesn't fall.
    robot = p.loadURDF(str(URDF), useFixedBase=True,
                       flags=p.URDF_USE_INERTIA_FROM_FILE)

    n = p.getNumJoints(robot)
    print("=" * 66)
    print(f"SO-101 loaded OK: {n} joints")
    print("=" * 66)
    movable = []
    for j in range(n):
        info = p.getJointInfo(robot, j)
        jname = info[1].decode()
        jtype = JOINT_TYPES.get(info[2], str(info[2]))
        lower, upper = info[8], info[9]
        axis = info[13]
        child = info[12].decode()
        parent_idx = info[16]
        tag = ""
        if info[2] != p.JOINT_FIXED:
            movable.append((j, jname))
            tag = f"  limits=[{lower:+.3f},{upper:+.3f}] rad " \
                  f"([{np.degrees(lower):+.0f},{np.degrees(upper):+.0f}] deg)"
        print(f"  [{j}] joint '{jname}'  type={jtype}  child='{child}'{tag}")

    print("-" * 66)
    print(f"MOVABLE joints ({len(movable)}): "
          f"{[f'{j}:{name}' for j, name in movable]}")

    # end-effector candidate: the 'gripperframe' or 'gripper' link.
    link_names = {}
    for j in range(n):
        link_names[p.getJointInfo(robot, j)[12].decode()] = j
    for cand in ("gripperframe", "gripper", "moving_jaw_so101_v1"):
        if cand in link_names:
            li = link_names[cand]
            st = p.getLinkState(robot, li)
            print(f"EE candidate link '{cand}' (idx {li}) world pos at zero pose: "
                  f"{np.round(st[0], 4)}")

    if args.gui:
        print("\nGUI open — inspect the arm. Ctrl-C to exit.")
        import time
        try:
            while True:
                p.stepSimulation()
                time.sleep(1 / 120)
        except KeyboardInterrupt:
            pass
    p.disconnect()


if __name__ == "__main__":
    main()
