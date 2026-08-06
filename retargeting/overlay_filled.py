"""
overlay_filled.py — side-by-side SO-101 (from FILLED wrist trajectory) | real video.
------------------------------------------------------------------------------------
Renders a smooth, slow-mo MP4:
    LEFT  = SO-101 driven by wrist_traj_filled.json (gap-filled + smoothed wrist
            position via IK, gripper joint from gripper_openness)
    RIGHT = the synced source video (real hand)

Purpose: visually confirm the gap-filling removed the jumpiness, and see which
frames were measured vs inferred (the HUD color-codes the `source` tag).

SMOOTH + SLOW, done right: we OFFLINE-render every frame (no realtime lag), and
write the MP4 at a LOWER output fps than the source. Every source frame is
present (=> smooth), the clip just plays slower (=> slow-mo). This is better than
frame-dropping (jerky) or realtime sleep (laggy on the sim).

Reuses IK + workspace mapping from pose_ik.py so the robot behaves identically
to the position-IK demo, just fed the filled trajectory + gripper.

Usage:
    venv_retarget/bin/python retargeting/overlay_filled.py \
        --traj retargeting/wrist_traj_filled.json \
        --video "stereo_data/Stereo Video with IMU (shared)/camera_left_2min.mp4" \
        --out retargeting/filled_overlay.mp4 --out-fps 12

HONEST NOTES:
  - Filled/lowconf frames are INFERRED (see trajectory_filter.py [T1]); the HUD
    marks them so you can tell inference from measurement.
  - Gripper openness in a gap is interpolated (no fingers observed there); a
    grasp hidden entirely inside a gap won't show (contact-event fusion = TODO).
  - Workspace mapping / axis correspondence are placeholders (pose_ik [B1][B2]).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import cv2
import pybullet as p
import pybullet_data

# reuse the vetted helpers from the position-IK module
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose_ik import (EE_LINK, DEFAULT_AXIS_MAP, WorkspaceMapper,
                     robot_reachable_box, solve_ik, URDF)

GRIP_JOINT_NAME = "6"
J6_CLOSED, J6_OPEN = 0.0, 1.5        # from gripper_map (direction confirmed)
SRC_COLOR = {"measured": (0, 255, 0), "filled": (0, 210, 255),
             "lowconf": (0, 120, 255), "outlier_rejected": (0, 0, 255)}


def load_traj(path, use_lowconf):
    d = json.load(open(path))
    frames = d["frames"]
    recs = []
    for f in frames:
        if f["source"] == "lowconf" and not use_lowconf:
            continue
        recs.append((f["frame_idx"], f["timestamp_us"],
                     np.asarray(f["position"], float),
                     f.get("gripper_openness"), f["source"]))
    return recs, d["metadata"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-fps", type=float, default=12.0,
                    help="playback fps; lower = slower (still every frame = smooth)")
    ap.add_argument("--box-frac", type=float, default=0.55)
    ap.add_argument("--fwd", type=float, default=0.20)
    ap.add_argument("--up", type=float, default=0.05)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--cam-dist", type=float, default=0.8)
    ap.add_argument("--cam-yaw", type=float, default=-89.4)
    ap.add_argument("--cam-pitch", type=float, default=-67.9)
    # angle from user's pick; target re-aimed at the arm's WORKING region center
    # (not the rest-pose base the picker showed) so the moving arm stays framed.
    ap.add_argument("--cam-target", type=float, nargs=3, default=[0.13, -0.17, 0.27])
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--drop-lowconf", action="store_true",
                    help="skip lowconf frames entirely (default: render, marked)")
    args = ap.parse_args()

    recs, meta = load_traj(args.traj, use_lowconf=not args.drop_lowconf)
    recs = recs[::args.stride]
    human_pos = np.array([r[2] for r in recs])

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot = p.loadURDF(str(URDF), useFixedBase=True)
    mov = [j for j in range(p.getNumJoints(robot))
           if p.getJointInfo(robot, j)[2] != p.JOINT_FIXED]
    arm_joints = [j for j in mov if p.getJointInfo(robot, j)[1].decode() != GRIP_JOINT_NAME]
    grip_joint = next(j for j in mov if p.getJointInfo(robot, j)[1].decode() == GRIP_JOINT_NAME)
    lower = [p.getJointInfo(robot, j)[8] for j in arm_joints]
    upper = [p.getJointInfo(robot, j)[9] for j in arm_joints]
    ranges = [u - l for l, u in zip(lower, upper)]

    bmin, bmax, bcenter = robot_reachable_box(robot, mov)
    mapper = WorkspaceMapper(human_pos, bmin, bmax, bcenter, args.box_frac,
                             DEFAULT_AXIS_MAP, box_offset=(args.fwd, 0.0, args.up))

    W = H = 640
    try:
        p.loadURDF("plane.urdf")
    except Exception:
        pass
    vs = p.createVisualShape(p.GEOM_SPHERE, radius=0.012, rgbaColor=[0, 1, 0, 0.8])
    marker = p.createMultiBody(baseVisualShapeIndex=vs, basePosition=[0, 0, 0])
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=args.cam_target, distance=args.cam_dist,
        yaw=args.cam_yaw, pitch=args.cam_pitch, roll=0, upAxisIndex=2)
    proj = p.computeProjectionMatrixFOV(fov=55, aspect=1.0, nearVal=0.02, farVal=3.0)

    cap = cv2.VideoCapture(args.video); vpos = -1
    writer = None
    for codec in ("avc1", "mp4v"):
        w = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*codec),
                            args.out_fps, (W * 2, H))
        if w.isOpened():
            writer = w; break
        w.release()

    seed = [0.0] * len(arm_joints)
    errors = []
    print(f"Rendering {len(recs)} frames @ out_fps={args.out_fps} -> {args.out}")
    for k, (fidx, ts, hp, grip, src) in enumerate(recs):
        target = mapper.map(hp)
        q, achieved = solve_ik(robot, arm_joints, lower, upper, ranges, target, seed)
        seed = q
        err = float(np.linalg.norm(achieved - target)); errors.append(err)
        # gripper joint from openness (fall back to open if missing)
        g = 0.5 if grip is None else float(np.clip(grip, 0, 1))
        j6 = J6_CLOSED + g * (J6_OPEN - J6_CLOSED)
        p.resetJointState(robot, grip_joint, j6)
        p.resetBasePositionAndOrientation(marker, target.tolist(), [0, 0, 0, 1])

        img = p.getCameraImage(W, H, view, proj, renderer=p.ER_TINY_RENDERER)
        rob = np.reshape(img[2], (H, W, 4))[:, :, :3].astype(np.uint8)
        rob = cv2.cvtColor(rob, cv2.COLOR_RGB2BGR)
        col = SRC_COLOR.get(src, (200, 200, 200))
        qd = [np.degrees(x) for x in q]
        lines = [(f"SO-101 (retargeted, FILLED)  frame {fidx}", (0, 255, 180)),
                 (f"source: {src.upper()}", col),
                 (f"pos err {err*100:4.1f} cm  gripper {g:.2f}", (0, 255, 180)),
                 (f"J1-5 [{qd[0]:+.0f} {qd[1]:+.0f} {qd[2]:+.0f} {qd[3]:+.0f} {qd[4]:+.0f}]", (0, 255, 180)),
                 ("X=fwd(red) Y=left(grn) Z=up(blu)", (180, 180, 180))]
        y = 24
        for ln, c in lines:
            cv2.putText(rob, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
            y += 24

        # synced real video (sequential exact decode)
        srcimg = np.zeros((H, W, 3), np.uint8)
        while vpos < fidx:
            if not cap.grab():
                break
            vpos += 1
        if vpos == fidx:
            ok, vf = cap.retrieve()
            if ok:
                srcimg = cv2.resize(vf, (W, H))
        cv2.putText(srcimg, f"SOURCE (real hand)  frame {fidx}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        writer.write(np.hstack([rob, srcimg]))
        if (k + 1) % 200 == 0:
            print(f"  {k+1}/{len(recs)}")

    writer.release(); cap.release(); p.disconnect()
    errors = np.array(errors)
    print(f"Done -> {args.out}")
    print(f"  median pos err {np.median(errors)*100:.1f}cm, "
          f"{(errors<=args.tol).mean()*100:.0f}% within {args.tol*100:.0f}cm")


if __name__ == "__main__":
    main()
