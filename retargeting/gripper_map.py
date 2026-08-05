"""
Phase 1b: map human hand openness -> SO-101 gripper joint (J6).
--------------------------------------------------------------
Pure, testable mapping from the 21 hand keypoints to a single parallel-gripper
scalar, then to the SO-101 gripper joint angle. Optionally animates J6 in
PyBullet from the recorded hand so you can see "hand opens -> gripper opens".

DESIGN (kept production-honest, NOT oversimplified):
  openness metric (scale-invariant):
      g_raw = ||thumb_tip(4) - index_tip(8)|| / ||wrist(0) - middle_MCP(9)||
  The denominator normalizes by hand size so the metric is invariant to how far
  the hand is from the camera. (Reference-recommended form.)

  calibration g_closed / g_open:
      robust dataset percentiles (default p5 / p95) — NOT min/max, because
      triangulation outliers reach ~13x the median. Configurable / overridable.

  normalize -> g in [0,1] (0 = fully closed, 1 = fully open), clipped.

  g -> J6 radians: linear between two calibrated joint values.

=========================== GAPS & ASSUMPTIONS (READ) ===========================
[A1] J6 OPEN/CLOSED DIRECTION IS UNVERIFIED. URDF limit for joint '6' is
     [-0.1745, +1.7453] rad, but which extreme is "jaw open" vs "jaw closed" is
     NOT stated in the URDF and NOT yet checked against hardware/GUI. Defaults
     below are a REASONED GUESS (open near upper limit). MUST be confirmed in
     the GUI or against the real arm before production. Flip with --invert-j6.
[A2] CALIBRATION ASSUMES THE CLIP SPANS THE FULL OPEN/CLOSED RANGE. p5/p95 of
     THIS recording define closed/open. If the wearer never fully opened/closed
     their hand, the mapping is compressed. Production fix: calibrate from an
     explicit open-hand and closed-hand reference gesture, or from grasp/contact
     phases, not just global percentiles.
[A3] PURE GEOMETRY, NO CONTACT OVERRIDE HERE. The reference recommends contact
     events override finger geometry for close/open TIMING (fingers can look
     open while gripping). That fusion is Phase 2; this module is the geometric
     signal only. Emitted `gripper_geom` should be treated as pre-contact-fusion.
[A4] OUTLIER/NOISE: a few frames have implausible g_raw (bad triangulation).
     We clip to the calibrated range; we do NOT yet temporally smooth here
     (smoothing belongs in the Phase-2 trajectory filter, before IK).
[A5] THUMB/INDEX are the noisiest triangulated points (small, occluded). A
     multi-finger openness (thumb vs index+middle+ring) is more robust; provided
     as --metric multi. Default is the simple thumb-index form for transparency.
================================================================================

Usage:
    venv_retarget/bin/python retargeting/gripper_map.py --hands stereo_3d/hands_3d_pose.json
    venv_retarget/bin/python retargeting/gripper_map.py --hands ... --hand Right --gui
    venv_retarget/bin/python retargeting/gripper_map.py --hands ... --out gripper_signal.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

# MediaPipe/MANO keypoint indices
WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP = 0, 4, 8, 9
INDEX_MCP, RING_TIP = 5, 16

# SO-101 joint '6' (gripper) URDF limits (rad).
J6_LOWER, J6_UPPER = -0.174533, 1.745329
# [A1] REASONED GUESS: hand fully OPEN -> jaw near upper; CLOSED -> near lower.
J6_OPEN_DEFAULT = 1.5          # ~86 deg, inside upper limit (margin)
J6_CLOSED_DEFAULT = 0.0        # ~0 deg (jaw nearly shut), inside lower margin


def openness_raw(kp, metric="thumb_index"):
    """Scale-invariant hand openness from keypoints (kp: (21,3))."""
    hand_size = np.linalg.norm(kp[MIDDLE_MCP] - kp[WRIST])
    if hand_size < 1e-6:
        return np.nan
    if metric == "thumb_index":
        return float(np.linalg.norm(kp[THUMB_TIP] - kp[INDEX_TIP]) / hand_size)
    elif metric == "multi":
        # [A5] average thumb-to-(index,middle,ring) tips -> more robust
        tips = [INDEX_TIP, 12, RING_TIP]
        d = np.mean([np.linalg.norm(kp[THUMB_TIP] - kp[t]) for t in tips])
        return float(d / hand_size)
    raise ValueError(metric)


def calibrate(g_all, lo_pct=5, hi_pct=95, override=None):
    """Robust closed/open calibration from dataset percentiles."""
    if override is not None:
        return override  # (g_closed, g_open)
    g = np.asarray([x for x in g_all if np.isfinite(x)])
    g_closed = float(np.percentile(g, lo_pct))
    g_open = float(np.percentile(g, hi_pct))
    return g_closed, g_open


def normalize(g_raw, g_closed, g_open):
    """-> [0,1], 0=closed 1=open, clipped."""
    if not np.isfinite(g_raw) or g_open - g_closed < 1e-9:
        return np.nan
    return float(np.clip((g_raw - g_closed) / (g_open - g_closed), 0.0, 1.0))


def gripper_to_j6(g_norm, j6_closed=J6_CLOSED_DEFAULT, j6_open=J6_OPEN_DEFAULT,
                  invert=False):
    """Map normalized openness [0,1] to J6 radians (clamped to URDF limits)."""
    if invert:
        j6_closed, j6_open = j6_open, j6_closed
    val = j6_closed + g_norm * (j6_open - j6_closed)
    return float(np.clip(val, J6_LOWER, J6_UPPER))


def build_signal(frames, hand_filter, metric):
    """Return list of (frame_idx, timestamp_us, g_raw) for the chosen hand."""
    out = []
    for fr in frames:
        for h in fr["hands"]:
            if hand_filter and h.get("handedness") != hand_filter:
                continue
            kp = np.asarray(h["keypoints_3d"], dtype=np.float64)
            out.append((fr["frame_idx"], fr.get("timestamp_us"),
                        openness_raw(kp, metric), h.get("handedness")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", required=True)
    ap.add_argument("--hand", default=None, help="Left/Right/None(all)")
    ap.add_argument("--metric", choices=["thumb_index", "multi"], default="thumb_index")
    ap.add_argument("--lo-pct", type=float, default=5)
    ap.add_argument("--hi-pct", type=float, default=95)
    ap.add_argument("--g-closed", type=float, default=None, help="override calib")
    ap.add_argument("--g-open", type=float, default=None, help="override calib")
    ap.add_argument("--j6-closed", type=float, default=J6_CLOSED_DEFAULT)
    ap.add_argument("--j6-open", type=float, default=J6_OPEN_DEFAULT)
    ap.add_argument("--invert-j6", action="store_true", help="[A1] flip open/closed")
    ap.add_argument("--out", default=None, help="write per-frame gripper JSON")
    ap.add_argument("--gui", action="store_true", help="animate J6 in PyBullet")
    args = ap.parse_args()

    data = json.load(open(args.hands))
    frames = data["frames"]
    sig = build_signal(frames, args.hand, args.metric)
    g_all = [s[2] for s in sig]

    override = None
    if args.g_closed is not None and args.g_open is not None:
        override = (args.g_closed, args.g_open)
    g_closed, g_open = calibrate(g_all, args.lo_pct, args.hi_pct, override)

    finite = np.asarray([g for g in g_all if np.isfinite(g)])
    print("=" * 68)
    print("PHASE 1b — GRIPPER MAP (human openness -> SO-101 J6)")
    print("=" * 68)
    print(f"  hand filter: {args.hand or 'ALL'}   metric: {args.metric}")
    print(f"  samples: {len(finite)}")
    print(f"  g_raw   min={finite.min():.2f} p5={np.percentile(finite,5):.2f} "
          f"median={np.median(finite):.2f} p95={np.percentile(finite,95):.2f} "
          f"max={finite.max():.2f}")
    print(f"  CALIBRATION  g_closed={g_closed:.3f}  g_open={g_open:.3f}"
          f"{'  (OVERRIDE)' if override else f'  (p{args.lo_pct:.0f}/p{args.hi_pct:.0f})'}")
    print(f"  J6 map: closed={args.j6_closed:.3f}rad open={args.j6_open:.3f}rad"
          f"{'  [INVERTED]' if args.invert_j6 else ''}")
    print("  ⚠️  [A1] J6 open/closed DIRECTION unverified — confirm in --gui / hardware.")
    print("  ⚠️  [A2] calibration assumes clip spans full open/close range.")
    print("  ⚠️  [A3] geometric only — contact-event override is Phase 2.")

    # build per-frame normalized + J6
    records = []
    for fidx, ts, g_raw, handed in sig:
        gn = normalize(g_raw, g_closed, g_open)
        j6 = gripper_to_j6(gn, args.j6_closed, args.j6_open, args.invert_j6) \
            if np.isfinite(gn) else None
        records.append({"frame_idx": fidx, "timestamp_us": ts, "handedness": handed,
                        "gripper_geom": None if np.isnan(gn) else round(gn, 4),
                        "j6_rad": None if j6 is None else round(j6, 5)})

    if args.out:
        outp = {
            "metadata": {
                "source": args.hands, "hand": args.hand, "metric": args.metric,
                "calibration": {"g_closed": g_closed, "g_open": g_open,
                                "lo_pct": args.lo_pct, "hi_pct": args.hi_pct,
                                "override": override is not None},
                "j6_map": {"closed_rad": args.j6_closed, "open_rad": args.j6_open,
                           "inverted": args.invert_j6,
                           "limits_rad": [J6_LOWER, J6_UPPER]},
                "assumptions": {
                    "A1_j6_direction_unverified": True,
                    "A2_calib_assumes_full_range": True,
                    "A3_no_contact_override_geometric_only": True,
                },
                "field_notes": "gripper_geom in [0,1] (0=closed,1=open); j6_rad = SO-101 gripper joint target.",
            },
            "frames": records,
        }
        Path(args.out).write_text(json.dumps(outp, indent=2))
        print(f"  -> wrote {args.out} ({len(records)} frames)")

    if args.gui:
        animate(records, args.hand)


def animate(records, hand):
    """Drive SO-101 J6 in PyBullet from the gripper signal (visual check of A1)."""
    import time
    import pybullet as p
    import pybullet_data
    urdf = Path(__file__).resolve().parent / "so101" / "so101.urdf"
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot = p.loadURDF(str(urdf), useFixedBase=True)
    # find joint '6' index
    j6_idx = None
    for j in range(p.getNumJoints(robot)):
        if p.getJointInfo(robot, j)[1].decode() == "6":
            j6_idx = j
    assert j6_idx is not None
    print(f"\nGUI: driving J6 (idx {j6_idx}) from hand '{hand or 'ALL'}'. "
          "Watch jaw vs hand openness to VERIFY [A1]. Ctrl-C to exit.")
    vals = [r["j6_rad"] for r in records if r["j6_rad"] is not None]
    i = 0
    try:
        while True:
            v = vals[i % len(vals)]
            p.setJointMotorControl2(robot, j6_idx, p.POSITION_CONTROL, targetPosition=v)
            p.stepSimulation()
            time.sleep(1 / 30)      # ~playback at 30fps
            i += 1
    except KeyboardInterrupt:
        p.disconnect()


if __name__ == "__main__":
    main()
