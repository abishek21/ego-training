"""
Trajectory filter — gap-fill + smooth the wearer's wrist trajectory.
--------------------------------------------------------------------
The per-frame hand detector misses frames (occlusion, single-camera, low
confidence), so the raw wrist trajectory has HOLES. A robot needs a continuous,
smooth trajectory. This fills short gaps and smooths the whole path.

METHOD (matches the recorded output's metadata):
  - Constant-velocity KALMAN FILTER on 3D position (state = [pos(3), vel(3)]).
    When a frame has a measurement, we update; when it doesn't, we PREDICT from
    the last velocity — this is the gap-fill.
  - RTS SMOOTHER (backward pass) so fills are consistent with BOTH past and
    future, and jitter is removed.
  - Chi-square OUTLIER GATING: a measurement whose innovation is too large
    (> gate_chi2) is rejected (treated as missing) — kills triangulation jumps.
  - Orientation: SLERP between nearest measured quaternions (rotation can't go
    through a linear KF). Gripper openness: linear interpolation.

SOURCE TAGGING (honesty — every frame says where it came from):
  measured          : an accepted detection.
  filled            : predicted through a SHORT gap (<= max_gap frames). Usable.
  lowconf           : predicted through a LONG gap (> max_gap). Inferred, weak —
                      down-weight or drop for training.
  outlier_rejected  : a detection rejected by the chi-square gate, then predicted.
  + per-frame `confidence` in [0,1] from the smoothed position covariance.

=========================== GAPS & ASSUMPTIONS (READ) ===========================
[T1] FILLED / LOWCONF FRAMES ARE INFERRED, NOT MEASURED. The constant-velocity
     model assumes smooth motion; it will be WRONG across a fast direction change
     hidden inside a gap. `source` + `confidence` expose this — do not treat
     filled poses as ground truth.
[T2] LONG GAPS (> max_gap) are emitted as `lowconf`, not dropped, so the frame
     index stays continuous. A consumer SHOULD filter these for training.
[T3] q_accel / r_meas are tuning knobs (process vs measurement trust). Defaults
     match the recorded run; re-tune per recording if motion characteristics
     differ. Higher q_accel = trust the model less = follow measurements more.
================================================================================

Usage:
    venv/bin/python retargeting/trajectory_filter.py \
        --hands stereo_3d/hands_3d_pose.json --hand Right \
        --out retargeting/wrist_traj_filled.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP = 0, 4, 8, 9


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def openness(kp):
    """Scale-invariant hand openness (thumb-index / hand size)."""
    hs = np.linalg.norm(kp[MIDDLE_MCP] - kp[WRIST])
    if hs < 1e-6:
        return np.nan
    return float(np.linalg.norm(kp[THUMB_TIP] - kp[INDEX_TIP]) / hs)


def slerp(q0, q1, t):
    """Spherical linear interpolation between unit quaternions (xyzw)."""
    q0 = np.asarray(q0, float); q1 = np.asarray(q1, float)
    q0 /= np.linalg.norm(q0); q1 /= np.linalg.norm(q1)
    dot = np.dot(q0, q1)
    if dot < 0.0:            # take the shorter arc
        q1 = -q1; dot = -dot
    if dot > 0.9995:        # nearly identical -> linear + renorm
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta0 = np.arccos(np.clip(dot, -1, 1))
    s0 = np.sin((1 - t) * theta0) / np.sin(theta0)
    s1 = np.sin(t * theta0) / np.sin(theta0)
    q = s0 * q0 + s1 * q1
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------------------
# constant-velocity Kalman + RTS smoother (3D position)
# ---------------------------------------------------------------------------
def cv_kalman_rts(times_s, meas, q_accel, r_meas, gate_chi2):
    """times_s: (N,) seconds. meas: (N,3) or NaN rows for missing.
    Returns smoothed positions (N,3), smoothed pos-covariance trace (N,),
    and a boolean (N,) 'rejected' for chi-square-gated measurements.
    """
    N = len(times_s)
    xdim = 6                                  # [px,py,pz, vx,vy,vz]
    I3 = np.eye(3)
    H = np.hstack([I3, np.zeros((3, 3))])     # measure position
    R = r_meas * I3

    xs_pred = np.zeros((N, xdim)); Ps_pred = np.zeros((N, xdim, xdim))
    xs_filt = np.zeros((N, xdim)); Ps_filt = np.zeros((N, xdim, xdim))
    Fs = np.zeros((N, xdim, xdim))
    rejected = np.zeros(N, bool)

    # init from first available measurement
    first = next((i for i in range(N) if not np.isnan(meas[i, 0])), 0)
    x = np.zeros(xdim); x[:3] = np.nan_to_num(meas[first])
    P = np.eye(xdim) * 1.0

    for k in range(N):
        dt = (times_s[k] - times_s[k - 1]) if k > 0 else 0.0
        F = np.eye(xdim)
        F[:3, 3:] = I3 * dt
        Fs[k] = F
        # constant-acceleration-driven process noise (white-accel CV model)
        q = q_accel ** 2
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        Q = q * np.block([
            [dt4 / 4 * I3, dt3 / 2 * I3],
            [dt3 / 2 * I3, dt2 * I3],
        ])
        # predict
        x = F @ x
        P = F @ P @ F.T + Q
        xs_pred[k] = x; Ps_pred[k] = P

        z = meas[k]
        if not np.isnan(z[0]):
            y = z - H @ x                      # innovation
            S = H @ P @ H.T + R
            md2 = float(y @ np.linalg.solve(S, y))   # squared Mahalanobis dist
            if md2 > gate_chi2:                # OUTLIER -> skip update
                rejected[k] = True
            else:
                K = P @ H.T @ np.linalg.inv(S)
                x = x + K @ y
                P = (np.eye(xdim) - K @ H) @ P
        xs_filt[k] = x; Ps_filt[k] = P

    # RTS backward smoother
    xs_smooth = xs_filt.copy(); Ps_smooth = Ps_filt.copy()
    for k in range(N - 2, -1, -1):
        F = Fs[k + 1]
        Pp = Ps_pred[k + 1]
        C = Ps_filt[k] @ F.T @ np.linalg.inv(Pp)
        xs_smooth[k] = xs_filt[k] + C @ (xs_smooth[k + 1] - xs_pred[k + 1])
        Ps_smooth[k] = Ps_filt[k] + C @ (Ps_smooth[k + 1] - Pp) @ C.T

    pos = xs_smooth[:, :3]
    trace_pos = np.array([np.trace(Ps_smooth[k][:3, :3]) for k in range(N)])
    return pos, trace_pos, rejected


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", required=True)
    ap.add_argument("--hand", default="Right")
    ap.add_argument("--out", required=True)
    ap.add_argument("--q-accel", type=float, default=30.0, help="[T3] process accel std")
    ap.add_argument("--r-meas", type=float, default=1e-4, help="[T3] meas noise var")
    ap.add_argument("--gate-chi2", type=float, default=16.0, help="outlier gate")
    ap.add_argument("--max-gap", type=int, default=12, help="frames; longer=lowconf")
    args = ap.parse_args()

    data = json.load(open(args.hands))
    frames = data["frames"]
    fps = data["metadata"].get("fps", 30.0)

    # collect the chosen hand per frame_idx
    by_idx = {}
    for fr in frames:
        for h in fr["hands"]:
            if h.get("handedness") == args.hand:
                by_idx[fr["frame_idx"]] = (fr.get("timestamp_us"), h)
                break

    if not by_idx:
        raise SystemExit(f"No '{args.hand}' hand found.")
    lo, hi = min(by_idx), max(by_idx)         # measured span
    idxs = list(range(lo, hi + 1))
    N = len(idxs)

    # build measurement arrays (NaN where missing)
    meas = np.full((N, 3), np.nan)
    quats = [None] * N
    grips = np.full(N, np.nan)
    ts = np.full(N, -1, dtype=np.int64)
    measured_mask = np.zeros(N, bool)
    for k, fidx in enumerate(idxs):
        if fidx in by_idx:
            tus, h = by_idx[fidx]
            ts[k] = tus if tus is not None else -1
            meas[k] = h["wrist_pose_cam"]["position"]
            quats[k] = h["wrist_pose_cam"]["quaternion"]
            grips[k] = openness(np.asarray(h["keypoints_3d"]))
            measured_mask[k] = True

    # timestamps in seconds (fall back to fps if any missing)
    if (ts < 0).any():
        times_s = np.arange(N) / fps
    else:
        times_s = (ts - ts[0]) / 1e6

    pos, trace_pos, rejected = cv_kalman_rts(
        times_s, meas, args.q_accel, args.r_meas, args.gate_chi2)

    # a measurement rejected by the gate is no longer "measured"
    measured_mask = measured_mask & ~rejected

    # Classify each non-measured frame by the length of the CONSECUTIVE gap run
    # it belongs to: if the whole hole is <= max_gap it's 'filled' (short,
    # trustworthy); if the hole is longer, the ENTIRE run is 'lowconf' (the
    # constant-velocity assumption is unreliable over long occlusions).
    gap_run_len = np.zeros(N, dtype=int)
    k = 0
    while k < N:
        if measured_mask[k]:
            k += 1
            continue
        j = k
        while j < N and not measured_mask[j]:
            j += 1
        gap_run_len[k:j] = j - k        # length of this consecutive hole
        k = j

    # orientation fill (SLERP) + gripper fill (linear interp)
    meas_idx = np.where(measured_mask)[0]
    quat_out = [None] * N
    grip_out = np.full(N, np.nan)
    for k in range(N):
        if measured_mask[k]:
            quat_out[k] = list(np.asarray(quats[k], float) /
                               np.linalg.norm(quats[k]))
            grip_out[k] = grips[k]
        else:
            left = meas_idx[meas_idx < k]; right = meas_idx[meas_idx > k]
            if len(left) and len(right):
                a, b = left[-1], right[0]
                t = (k - a) / (b - a)
                quat_out[k] = list(slerp(quats[a], quats[b], t))
                grip_out[k] = grips[a] + t * (grips[b] - grips[a])
            elif len(left):
                quat_out[k] = list(np.asarray(quats[left[-1]], float))
                grip_out[k] = grips[left[-1]]
            elif len(right):
                quat_out[k] = list(np.asarray(quats[right[0]], float))
                grip_out[k] = grips[right[0]]

    # confidence from smoothed position covariance (bounded, monotone in trace)
    conf = 1.0 / (1.0 + trace_pos / (args.r_meas * 100))
    conf = np.clip(conf, 0.0, 0.999)

    # assemble output + source tags
    out_frames = []
    counts = {"measured": 0, "filled": 0, "lowconf": 0, "outlier_rejected": 0}
    for k, fidx in enumerate(idxs):
        if rejected[k]:
            src = "outlier_rejected"
        elif measured_mask[k]:
            src = "measured"
        elif gap_run_len[k] <= args.max_gap:
            src = "filled"
        else:
            src = "lowconf"
        counts[src] += 1
        out_frames.append({
            "frame_idx": int(fidx),
            "timestamp_us": int(ts[k]) if ts[k] >= 0 else None,
            "position": [round(float(v), 5) for v in pos[k]],
            "quaternion": [round(float(v), 6) for v in quat_out[k]],
            "gripper_openness": None if np.isnan(grip_out[k]) else round(float(grip_out[k]), 4),
            "source": src,
            "confidence": round(float(conf[k]), 4),
        })

    out = {
        "metadata": {
            "source_hands": args.hands, "hand": args.hand,
            "fps": round(fps, 4),
            "method": "constant-velocity Kalman + RTS smoother; SLERP orientation; interp gripper",
            "params": {"q_accel": args.q_accel, "r_meas": args.r_meas,
                       "gate_chi2": args.gate_chi2, "max_gap": args.max_gap},
            "counts": {"span_frames": N, **counts},
            "notes": "[T1] filled/lowconf frames are inferred, not measured — see 'source'/'confidence'.",
        },
        "frames": out_frames,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("=" * 64)
    print("TRAJECTORY FILTER (CV Kalman + RTS smoother)")
    print("=" * 64)
    print(f"  hand={args.hand}  span_frames={N} (idx {lo}..{hi})")
    for k, v in counts.items():
        print(f"    {k:16s}: {v}")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
