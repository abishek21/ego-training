"""
cam_previews.py — render the SO-101 at several camera angles as a labeled grid.
------------------------------------------------------------------------------
GUI camera picking crashes on this macOS/PyBullet build, so instead we render
candidate views OFFLINE (TinyRenderer, no window) into one contact-sheet PNG.
Look at it, pick the angle number, and pass its printed cam args to
overlay_filled.py.

Usage:
    venv_retarget/bin/python retargeting/cam_previews.py --out retargeting/cam_grid.png
"""

import argparse
from pathlib import Path

import numpy as np
import cv2
import pybullet as p
import pybullet_data

URDF = Path(__file__).resolve().parent / "so101" / "so101.urdf"

# candidate (name, yaw, pitch, dist, target)
VIEWS = [
    ("1 front",        90,  -20, 0.9, [0.15, 0.0, 0.15]),
    ("2 front-high",   90,  -45, 0.9, [0.15, 0.0, 0.15]),
    ("3 right-side",    0,  -20, 0.9, [0.15, 0.0, 0.15]),
    ("4 left-side",   180,  -20, 0.9, [0.15, 0.0, 0.15]),
    ("5 3/4 left",     50,  -30, 0.9, [0.10, -0.05, 0.15]),
    ("6 3/4 right",   130,  -30, 0.9, [0.10, 0.05, 0.15]),
    ("7 top-down",     90,  -80, 1.0, [0.15, 0.0, 0.10]),
    ("8 low-front",    90,  -10, 0.8, [0.15, 0.0, 0.15]),
    ("9 3/4 close",    60,  -35, 0.7, [0.15, 0.0, 0.15]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="retargeting/cam_grid.png")
    ap.add_argument("--tile", type=int, default=420)
    args = ap.parse_args()

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    try:
        p.loadURDF("plane.urdf")
    except Exception:
        pass
    robot = p.loadURDF(str(URDF), useFixedBase=True)
    # a mild non-zero pose so the arm isn't perfectly straight (easier to read)
    mov = [j for j in range(p.getNumJoints(robot))
           if p.getJointInfo(robot, j)[2] != p.JOINT_FIXED]
    for j, a in zip(mov, [0.3, -0.5, 0.6, -0.3, 0.2, 0.5]):
        p.resetJointState(robot, j, a)

    S = args.tile
    proj = p.computeProjectionMatrixFOV(fov=55, aspect=1.0, nearVal=0.02, farVal=3.0)
    tiles = []
    for name, yaw, pitch, dist, tgt in VIEWS:
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=tgt, distance=dist, yaw=yaw, pitch=pitch,
            roll=0, upAxisIndex=2)
        img = p.getCameraImage(S, S, view, proj, renderer=p.ER_TINY_RENDERER)
        rgb = np.reshape(img[2], (S, S, 4))[:, :, :3].astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        label = f"{name}  yaw={yaw} pitch={pitch} d={dist} t={tgt}"
        cv2.rectangle(bgr, (0, 0), (S, 44), (0, 0, 0), -1)
        cv2.putText(bgr, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 180), 2)
        cv2.putText(bgr, f"yaw {yaw} pit {pitch} d {dist}", (10, S - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        tiles.append(bgr)

    # 3x3 grid
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, 9, 3)]
    grid = np.vstack(rows)
    cv2.imwrite(args.out, grid)
    p.disconnect()
    print(f"Wrote {args.out}")
    print("Pick a number; its cam args:")
    for name, yaw, pitch, dist, tgt in VIEWS:
        print(f"  {name}: --cam-yaw {yaw} --cam-pitch {pitch} --cam-dist {dist} "
              f"--cam-target {tgt[0]} {tgt[1]} {tgt[2]}")


if __name__ == "__main__":
    main()
