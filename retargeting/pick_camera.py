"""
pick_camera.py — interactively choose the render camera for the SO-101 scene.
----------------------------------------------------------------------------
Opens the arm + floor + world-axis triad in the PyBullet GUI. Rotate/zoom/pan
to the view you want (drag = orbit, scroll = zoom, ctrl/cmd+drag = pan). It
prints the LIVE camera parameters (yaw, pitch, distance, target) ~twice a
second. When the view looks right, copy the printed line — those are the
--cam-yaw / --cam-pitch / --cam-dist / --cam-target values to pass to
overlay_filled.py (or pose_ik.py) so the render matches this view exactly.

Usage:
    venv_retarget/bin/python retargeting/pick_camera.py
"""

import time
from pathlib import Path

import pybullet as p
import pybullet_data

URDF = Path(__file__).resolve().parent / "so101" / "so101.urdf"


def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    try:
        p.loadURDF("plane.urdf")
    except Exception:
        pass
    p.loadURDF(str(URDF), useFixedBase=True)

    # world axis triad: X red (FORWARD), Y green (LEFT), Z blue (UP)
    o = [0, 0, 0]
    p.addUserDebugLine(o, [0.25, 0, 0], [1, 0, 0], lineWidth=3)
    p.addUserDebugLine(o, [0, 0.25, 0], [0, 1, 0], lineWidth=3)
    p.addUserDebugLine(o, [0, 0, 0.25], [0, 0, 1], lineWidth=3)
    p.addUserDebugText("X fwd", [0.27, 0, 0], [1, 0, 0])
    p.addUserDebugText("Y left", [0, 0.27, 0], [0, 1, 0])
    p.addUserDebugText("Z up", [0, 0, 0.27], [0, 0.4, 1])

    print("=" * 70)
    print("PICK CAMERA — orbit/zoom/pan with the TRACKPAD to the view you want.")
    print("  drag=orbit   scroll=zoom   Cmd/Ctrl+drag=pan")
    print("The live camera params print below (~2x/sec). Copy the line you like.")
    print("Ctrl-C to exit (prints FINAL camera).")
    print("=" * 70)

    # a sensible starting view
    p.resetDebugVisualizerCamera(cameraDistance=0.9, cameraYaw=50,
                                 cameraPitch=-30,
                                 cameraTargetPosition=[0.0, -0.1, 0.2])

    # PASSIVE: only READ the camera (no sliders / no per-frame reset). The
    # slider+reset path was auto-closing the window on this macOS/Metal build.
    last = 0.0
    try:
        while True:
            p.stepSimulation()
            now = time.time()
            if now - last > 0.5:
                last = now
                try:
                    info = p.getDebugVisualizerCamera()
                except p.error:
                    break   # window closed
                yaw, pitch, dist, tgt = info[8], info[9], info[10], info[11]
                print(f"  --cam-yaw {yaw:.1f} --cam-pitch {pitch:.1f} "
                      f"--cam-dist {dist:.2f} "
                      f"--cam-target {tgt[0]:.3f} {tgt[1]:.3f} {tgt[2]:.3f}",
                      flush=True)
            time.sleep(1 / 120)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            info = p.getDebugVisualizerCamera()
            print("\nFINAL camera:")
            print(f"  --cam-yaw {info[8]:.1f} --cam-pitch {info[9]:.1f} "
                  f"--cam-dist {info[10]:.2f} "
                  f"--cam-target {info[11][0]:.3f} {info[11][1]:.3f} {info[11][2]:.3f}",
                  flush=True)
        except Exception:
            pass
        p.disconnect()


if __name__ == "__main__":
    main()
