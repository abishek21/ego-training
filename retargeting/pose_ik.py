"""
Phase 1c: static / trajectory POSITION IK — human wrist -> SO-101 joints.
------------------------------------------------------------------------
Takes the human wrist POSITION (from hands_3d_pose.json), maps it into the
SO-101's reachable workspace, and solves inverse kinematics (PyBullet) for the
5 arm joints so the tool point (gripperframe) reaches it. Gripper (J6) is driven
from hand openness (Phase 1b). Renders a live HUD of metrics as it moves.

This is POSITION-ONLY for the first increment (orientation added later): it
isolates and validates the workspace mapping before the trickier orientation /
tool-frame convention.

=========================== GAPS & ASSUMPTIONS (READ) ===========================
[B1] WORKSPACE MAPPING IS A PLACEHOLDER, NOT HAND-EYE CALIBRATION. We normalize
     the human wrist's observed per-axis range into a shrunk sub-box of the
     robot's reachable workspace. This reproduces the SHAPE of the motion, not
     absolute human->robot poses. Production would use delta actions (no
     mapping) or object-relative or a real hand-eye/marker calibration.
[B2] AXIS CORRESPONDENCE IS AN ASSUMPTION. Camera frame (x=right, y=down,
     z=forward) != robot base frame (x=forward, y=lateral, z=up). The default
     --axis-map sends human-vertical->robot-vertical etc.; VERIFY in the GUI
     (does the arm go up when the hand goes up?). Configurable.
[B3] POSITION-ONLY. Wrist ORIENTATION is ignored in this increment. The arm
     reaches the right point but the tool orientation is whatever IK returns.
     Orientation (pitch/roll; NO yaw on SO-101) is the next increment.
[B4] IK MAY BE APPROXIMATE / FAIL near reach limits. We run FK on the solution
     and report the true position error; frames with error above --tol are
     flagged (not silently accepted).
[B5] OPEN-LOOP REPLAY. This drives stored targets directly — it is a
     VISUALIZATION, not a closed-loop policy. Drift/anchoring caveats apply only
     to this replay, not to training-time delta labels.
================================================================================

Usage:
    venv_retarget/bin/python retargeting/pose_ik.py --hands stereo_3d/hands_3d_pose.json --gui
    venv_retarget/bin/python retargeting/pose_ik.py --hands ... --hand Right --box-frac 0.55
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

URDF = Path(__file__).resolve().parent / "so101" / "so101.urdf"
EE_LINK = 6            # 'gripperframe'
GRIP_JOINT_NAME = "6"

# Default axis map: for each ROBOT axis, (human source axis, sign).
# Camera: x=right, y=down(+), z=forward(more negative = farther).
# Robot : x=forward, y=lateral, z=up.
#   robot x (forward) <- -human z  (more-negative-forward -> larger forward)
#   robot y (lateral) <-  human x  (right)
#   robot z (up)      <- -human y  (human +y is down -> negate for up)
DEFAULT_AXIS_MAP = {"x": ("z", -1.0), "y": ("x", +1.0), "z": ("y", -1.0)}
AXIS_IDX = {"x": 0, "y": 1, "z": 2}


def load_wrist_positions(frames, hand):
    recs = []
    for fr in frames:
        for h in fr["hands"]:
            if hand and h.get("handedness") != hand:
                continue
            pos = np.asarray(h["wrist_pose_cam"]["position"], dtype=np.float64)
            recs.append((fr["frame_idx"], fr.get("timestamp_us"), pos))
    return recs


def robot_reachable_box(robot, mov, n=20000, seed=0):
    lims = [(p.getJointInfo(robot, j)[8], p.getJointInfo(robot, j)[9]) for j in mov]
    rng = np.random.default_rng(seed)
    pts = np.empty((n, 3))
    for i in range(n):
        for j, (lo, hi) in zip(mov, lims):
            p.resetJointState(robot, j, rng.uniform(lo, hi))
        pts[i] = p.getLinkState(robot, EE_LINK)[0]
    for j in mov:
        p.resetJointState(robot, j, 0.0)
    return pts.min(0), pts.max(0), pts.mean(0)


class WorkspaceMapper:
    """[B1][B2] map human wrist positions -> robot target box (placeholder)."""
    def __init__(self, human_pos, box_min, box_max, box_center, box_frac, axis_map):
        self.h_min = human_pos.min(0)
        self.h_max = human_pos.max(0)
        self.h_rng = np.maximum(self.h_max - self.h_min, 1e-6)
        # shrink reachable box around its center to stay off the limits
        half = (box_max - box_min) / 2 * box_frac
        self.t_min = box_center - half
        self.t_max = box_center + half
        self.axis_map = axis_map

    def map(self, hp):
        out = np.zeros(3)
        for raxis, (haxis, sign) in self.axis_map.items():
            hi = AXIS_IDX[haxis]; ri = AXIS_IDX[raxis]
            norm = (hp[hi] - self.h_min[hi]) / self.h_rng[hi]   # [0,1]
            if sign < 0:
                norm = 1.0 - norm
            out[ri] = self.t_min[ri] + norm * (self.t_max[ri] - self.t_min[ri])
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", required=True)
    ap.add_argument("--hand", default="Right")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--box-frac", type=float, default=0.55,
                    help="[B1] fraction of reachable box to use (smaller=safer)")
    ap.add_argument("--tol", type=float, default=0.02, help="[B4] pos error flag (m)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    data = json.load(open(args.hands))
    recs = load_wrist_positions(data["frames"], args.hand)[::args.stride]
    human_pos = np.array([r[2] for r in recs])

    p.connect(p.GUI if args.gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot = p.loadURDF(str(URDF), useFixedBase=True)
    mov = [j for j in range(p.getNumJoints(robot))
           if p.getJointInfo(robot, j)[2] != p.JOINT_FIXED]
    arm_joints = [j for j in mov if p.getJointInfo(robot, j)[1].decode() != GRIP_JOINT_NAME]
    lower = [p.getJointInfo(robot, j)[8] for j in arm_joints]
    upper = [p.getJointInfo(robot, j)[9] for j in arm_joints]
    ranges = [u - l for l, u in zip(lower, upper)]

    bmin, bmax, bcenter = robot_reachable_box(robot, mov)
    mapper = WorkspaceMapper(human_pos, bmin, bmax, bcenter, args.box_frac, DEFAULT_AXIS_MAP)

    print("=" * 70)
    print("PHASE 1c — POSITION IK (human wrist -> SO-101 J1..J5)")
    print("=" * 70)
    print(f"  hand: {args.hand}  frames: {len(recs)}  box_frac: {args.box_frac}")
    print(f"  target box min {mapper.t_min.round(3)} max {mapper.t_max.round(3)}")
    print("  ⚠️  [B1] workspace mapping = placeholder (not hand-eye calib)")
    print("  ⚠️  [B2] axis correspondence assumed — verify in GUI")
    print("  ⚠️  [B3] POSITION-ONLY (orientation ignored this increment)")
    print("  ⚠️  [B5] open-loop replay = visualization, not a policy")

    errors = []
    seed = [0.0] * len(arm_joints)
    hud = None
    marker = None
    for k, (fidx, ts, hp) in enumerate(recs):
        target = mapper.map(hp)
        # IK (position only). rest poses/seed = previous solution for continuity.
        sol = p.calculateInverseKinematics(
            robot, EE_LINK, targetPosition=target.tolist(),
            lowerLimits=lower, upperLimits=upper, jointRanges=ranges,
            restPoses=seed, maxNumIterations=200, residualThreshold=1e-4)
        q = list(sol[:len(arm_joints)])
        seed = q
        for j, qi in zip(arm_joints, q):
            p.resetJointState(robot, j, qi)
        achieved = np.array(p.getLinkState(robot, EE_LINK)[0])
        err = float(np.linalg.norm(achieved - target))
        errors.append(err)

        if args.gui:
            # target marker (green sphere) + achieved (line)
            if marker is None:
                vs = p.createVisualShape(p.GEOM_SPHERE, radius=0.012, rgbaColor=[0, 1, 0, 0.7])
                marker = p.createMultiBody(baseVisualShapeIndex=vs, basePosition=target.tolist())
            else:
                p.resetBasePositionAndOrientation(marker, target.tolist(), [0, 0, 0, 1])
            qd = [np.degrees(x) for x in q]
            txt = (f"frame {fidx}\n"
                   f"target  [{target[0]:+.3f} {target[1]:+.3f} {target[2]:+.3f}]\n"
                   f"reached [{achieved[0]:+.3f} {achieved[1]:+.3f} {achieved[2]:+.3f}]\n"
                   f"pos err {err*100:5.1f} cm  {'OK' if err<=args.tol else 'HIGH'}\n"
                   f"J1..J5 deg [{qd[0]:+.0f} {qd[1]:+.0f} {qd[2]:+.0f} {qd[3]:+.0f} {qd[4]:+.0f}]")
            hud = p.addUserDebugText(txt, [0, 0, 0.6], textColorRGB=[1, 1, 1],
                                     textSize=1.3, replaceItemUniqueId=hud or -1)
            p.stepSimulation()
            time.sleep(1.0 / args.fps)

    errors = np.array(errors)
    print("-" * 70)
    print(f"  position error (cm): median {np.median(errors)*100:.1f}  "
          f"p95 {np.percentile(errors,95)*100:.1f}  max {errors.max()*100:.1f}")
    print(f"  frames within tol ({args.tol*100:.0f}cm): "
          f"{(errors<=args.tol).mean()*100:.0f}%")
    if args.gui:
        print("  GUI open — Ctrl-C to exit."); 
        try:
            while True:
                p.stepSimulation(); time.sleep(1/120)
        except KeyboardInterrupt:
            pass
    p.disconnect()


if __name__ == "__main__":
    main()
