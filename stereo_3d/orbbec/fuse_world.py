"""
Stage 3d: fuse ORB-SLAM3 camera poses -> lift hands into the WORLD frame.
-------------------------------------------------------------------------
Takes the stereo VO trajectory (per-frame camera pose T_world_cam) and the
camera-frame hand data, and produces hands in the WORLD frame (fixed to the
first camera pose).

    f_orbbec_stereo.txt  +  hands_3d_pose.json  -->  hands_3d_world.json

WORLD FRAME (be precise): origin = the FIRST tracked camera pose (ORB-SLAM3
convention), axes gravity-unaware (stereo-only, no IMU). It is a consistent
LOCAL world frame, NOT geographic. Units meters.

TRANSFORM: TUM trajectory rows are T_world_cam (camera pose in world):
    p_world = R_wc * p_cam + t_wc
    R_world_wrist = R_wc * R_cam_wrist

TIMESTAMP JOIN (important): the trajectory writes ns timestamps as FLOAT
(~1.78e18), which exceeds float64's exact-int range (2^53). So we match each
trajectory row to a frame by NEAREST timestamp (frames are 33 ms apart; float
error is ~sub-microsecond, so the match is unambiguous) and assert the residual
is tiny. frame_map.csv maps ts_ns -> ORIGINAL video frame_idx, which is how we
line poses up with hands_3d frames.

HONESTY: this fuses; it does NOT prove accuracy. The trajectory is bounded and
metric but low-parallax (head moved ~18cm/~20deg). The definitive check is a
STATIC scene point staying fixed in world coords — which needs a known static
object (Stage 4) or saved SLAM map points. Here we validate what we can:
join integrity, coverage, and world-vs-camera motion sanity.

Usage:
    python fuse_world.py \
        --traj ../slam_out/f_orbbec_stereo.txt \
        --frame-map ../slam_data/frame_map.csv \
        --hands ../hands_3d_pose.json \
        --out ../hands_3d_world.json
"""

import argparse
import csv
import json

import numpy as np


def quat_to_R(qx, qy, qz, qw):
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
    ])


def R_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr+1.0)*2; qw=0.25*S
        qx=(R[2,1]-R[1,2])/S; qy=(R[0,2]-R[2,0])/S; qz=(R[1,0]-R[0,1])/S
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S=np.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2; qw=(R[2,1]-R[1,2])/S
        qx=0.25*S; qy=(R[0,1]+R[1,0])/S; qz=(R[0,2]+R[2,0])/S
    elif R[1,1] > R[2,2]:
        S=np.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2; qw=(R[0,2]-R[2,0])/S
        qx=(R[0,1]+R[1,0])/S; qy=0.25*S; qz=(R[1,2]+R[2,1])/S
    else:
        S=np.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2; qw=(R[1,0]-R[0,1])/S
        qx=(R[0,2]+R[2,0])/S; qy=(R[1,2]+R[2,1])/S; qz=0.25*S
    q=np.array([qx,qy,qz,qw]); return q/np.linalg.norm(q)


def load_frame_map(path):
    ts, idx = [], []
    with open(path) as f:
        r = csv.reader(f); next(r)
        for row in r:
            if row:
                ts.append(int(row[0])); idx.append(int(row[1]))
    return np.asarray(ts, dtype=np.int64), np.asarray(idx, dtype=np.int64)


def build_frame_poses(traj_path, fm_ts, fm_idx):
    """Match trajectory rows to frame indices by NEAREST timestamp.

    Returns {frame_idx: (R_wc(3x3), t_wc(3,))} and residual stats.
    """
    d = np.loadtxt(traj_path)
    tts = d[:, 0]                      # float ns (precision-limited)
    order = np.argsort(fm_ts)
    fm_ts_s, fm_idx_s = fm_ts[order], fm_idx[order]

    poses = {}
    residuals = []
    dup = 0
    for row in d:
        t = row[0]
        j = np.searchsorted(fm_ts_s, t)
        cand = [k for k in (j-1, j) if 0 <= k < len(fm_ts_s)]
        best = min(cand, key=lambda k: abs(fm_ts_s[k]-t))
        residuals.append(abs(fm_ts_s[best]-t))
        fidx = int(fm_idx_s[best])
        if fidx in poses:
            dup += 1
        R = quat_to_R(row[4], row[5], row[6], row[7])
        poses[fidx] = (R, row[1:4].astype(np.float64))
    residuals = np.asarray(residuals)
    return poses, residuals, dup


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj", required=True)
    p.add_argument("--frame-map", required=True)
    p.add_argument("--hands", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    fm_ts, fm_idx = load_frame_map(args.frame_map)
    poses, residuals, dup = build_frame_poses(args.traj, fm_ts, fm_idx)

    print("=" * 56)
    print("STAGE 3d — LIFT HANDS TO WORLD FRAME")
    print("=" * 56)
    print(f"  trajectory poses: {len(poses)}  ts-match residual "
          f"median={np.median(residuals):.0f}ns max={residuals.max():.0f}ns")
    if residuals.max() > 1_000_000:  # 1 ms
        print("  ⚠️  large ts-match residual (>1ms) — join may be misaligned!")
    if dup:
        print(f"  ⚠️  {dup} trajectory rows mapped to an already-used frame")

    with open(args.hands) as f:
        data = json.load(f)

    n_frames_lifted = 0
    n_frames_no_pose = 0
    n_hands = 0
    for fr in data["frames"]:
        fidx = fr["frame_idx"]
        pose = poses.get(fidx)
        if pose is None:
            fr["world_pose_available"] = False
            n_frames_no_pose += 1
            continue
        R_wc, t_wc = pose
        fr["world_pose_available"] = True
        fr["T_world_cam"] = {
            "R": R_wc.round(8).tolist(),
            "t": t_wc.round(6).tolist(),
        }
        n_frames_lifted += 1
        for h in fr["hands"]:
            kp = np.asarray(h["keypoints_3d"], dtype=np.float64)   # (21,3) cam
            kp_world = (R_wc @ kp.T).T + t_wc
            h["keypoints_3d_world"] = kp_world.round(5).tolist()
            # Lift wrist 6DoF if present (Stage 2 output).
            if "wrist_pose_cam" in h:
                Rcw = np.asarray(h["wrist_pose_cam"]["rotation_matrix"])
                pos_c = np.asarray(h["wrist_pose_cam"]["position"])
                R_world_wrist = R_wc @ Rcw
                pos_w = R_wc @ pos_c + t_wc
                h["wrist_pose_world"] = {
                    "position": pos_w.round(5).tolist(),
                    "quaternion": R_to_quat(R_world_wrist).round(6).tolist(),
                    "rotation_matrix": R_world_wrist.round(6).tolist(),
                }
            n_hands += 1

    meta = data["metadata"]
    meta["stage3_world"] = {
        "applied": True,
        "trajectory": str(args.traj),
        "method": "ORB-SLAM3 stereo-only (no IMU) VO",
        "world_frame": "origin = first tracked camera pose; local, gravity-unaware; meters",
        "frames_lifted": n_frames_lifted,
        "frames_without_pose": n_frames_no_pose,
        "hands_lifted": n_hands,
        "ts_match_residual_ns_median": float(np.median(residuals)),
        "ts_match_residual_ns_max": float(residuals.max()),
        "caveat": ("Bounded, metric, but low-parallax (head ~18cm/~20deg). "
                   "Not validated against a static ground-truth point; do that "
                   "with a known static object (Stage 4)."),
    }
    meta["notes"] = meta.get("notes", "") + " | Stage3d world-frame lift (fuse_world.py, stereo-only VO)."

    with open(args.out, "w") as f:
        json.dump(data, f)

    print(f"  frames lifted to world: {n_frames_lifted}")
    print(f"  frames WITHOUT pose:    {n_frames_no_pose}")
    print(f"  hands lifted:           {n_hands}")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
