"""
dump_view.py — capture the EXACT camera view matrix for a given GUI camera state.
---------------------------------------------------------------------------------
PyBullet's interactive GUI camera and the offline
computeViewMatrixFromYawPitchRoll() can disagree (convention quirk), so a view
that looks right in the picker renders wrong offline. This script sidesteps that:
it opens a GUI, sets the debug camera to the EXACT yaw/pitch/dist/target you
picked, reads back the ACTUAL 16-float view matrix PyBullet uses, and saves it.
The renderer then uses that matrix directly -> guaranteed identical framing.

Usage:
    venv_retarget/bin/python retargeting/dump_view.py \
        --yaw -89.4 --pitch -67.9 --dist 1.04 --target 0.145 -0.099 -0.119 \
        --out retargeting/view_matrix.json
"""

import argparse
import json
import time
from pathlib import Path

import pybullet as p
import pybullet_data

URDF = Path(__file__).resolve().parent / "so101" / "so101.urdf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaw", type=float, required=True)
    ap.add_argument("--pitch", type=float, required=True)
    ap.add_argument("--dist", type=float, required=True)
    ap.add_argument("--target", type=float, nargs=3, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    try:
        p.loadURDF("plane.urdf")
    except Exception:
        pass
    p.loadURDF(str(URDF), useFixedBase=True)
    p.resetDebugVisualizerCamera(cameraDistance=args.dist, cameraYaw=args.yaw,
                                 cameraPitch=args.pitch,
                                 cameraTargetPosition=args.target)
    # let the GUI apply the camera, then read the ACTUAL matrices it uses
    for _ in range(10):
        p.stepSimulation()
        time.sleep(0.02)
    info = p.getDebugVisualizerCamera()
    view = list(info[2])          # 16 floats, column-major
    proj = list(info[3])
    Path(args.out).write_text(json.dumps({
        "yaw": args.yaw, "pitch": args.pitch, "dist": args.dist,
        "target": args.target, "view_matrix": view, "proj_matrix": proj,
    }, indent=2))
    print(f"Captured exact view matrix -> {args.out}")
    p.disconnect()


if __name__ == "__main__":
    main()
