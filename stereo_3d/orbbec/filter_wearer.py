"""
Post-process hands_3d.json -> keep only the WEARER's hands.
-----------------------------------------------------------
DERIVED, tunable step deliberately kept OUT of the core detection script
(process_clip.py). Workflow stays:

    video -> keypoints -> hands_3d.json   (raw, source of truth, never filtered)
                            |
                            +--> filter_wearer.py --> hands_3d_wearer.json (view)

Why separate: wearer-selection thresholds are judgement calls we want to
re-tune WITHOUT re-running WiLoR on the GPU pod. The raw JSON is preserved;
this writes a filtered projection of it and records the exact params used.

Coordinate frame (from hands_3d.json metadata): LEFT camera, meters.
IMPORTANT: this camera looks down -z (points in front have NEGATIVE z, e.g.
wrist z ~ -0.22). So all "forward" math uses abs(z).

Cues (in priority order):
  1. lateral    : the wearer's own hands stay near the optical axis. A second
                  person's hand off to the side has large |x| (absolute) and
                  large |x|/|z| (scale-robust angular offset). THIS is the cue
                  that actually removes the observed intruder (x ~ +0.46 m,
                  wearer |x| < 0.2 m).
  2. distance   : wearer's hands are within arm's reach (secondary; the
                  intruder at ~0.55 m is NOT separable by distance alone).
  3. span QA    : reject implausible triangulated hand spans (noise). Risky —
                  the wearer's near/occluded hand legitimately shows ~0.09 m
                  spans on this clip, so keep this LOOSE or use --smooth-span.
  4. closest-N  : a wearer has <=2 hands -> keep the closest, 1 per handedness.

Honest caveats:
  - Per-frame + stateless by default. This is a geometric gate, not a tracker.
    A brief mislabel can still flicker. `--smooth-span` adds a light temporal
    median on the SPAN VALUE used for QA only (never touches keypoints).
  - Handedness on ego fisheye is unreliable; closest-N fallback covers label
    collapse.
  - Thresholds default to THIS clip's geometry. Re-validate on new recordings;
    the printed reject_reasons breakdown is there to tune from data.

Usage:
    python filter_wearer.py --in ../hands_3d.json --out ../hands_3d_wearer.json
    python filter_wearer.py --in ../hands_3d.json --out ../hands_3d_wearer.json \
        --max-abs-x 0.30 --max-dist 0.85 --min-span 0.06 --smooth-span
"""

import argparse
import json
from collections import defaultdict, deque

import numpy as np


def wearer_gate(hand, cfg):
    """Return (keep: bool, reason: str) for a single hand record."""
    kp = np.asarray(hand["keypoints_3d"], dtype=np.float64)
    x, y, z = kp[0]                        # wrist, LEFT-camera frame (meters)
    dist = float(hand["wrist_distance_m"])
    span = float(hand.get("_span_gate", hand["hand_span_m"]))

    # 1. lateral (the discriminating cue) -------------------------------
    # Ego geometry is ASYMMETRIC: the wearer's own two hands live in
    # x ~ [negative .. +0.20] m (just left/right of their centerline, directly
    # below the camera). The intruder is a 2nd person off to the RIGHT, so
    # their hands sit at large POSITIVE x (~+0.39..+0.61 m). Hence a ONE-SIDED
    # cut on the +x side, NOT a symmetric |x| gate (which would wrongly cull a
    # wearer hand reaching to the left). We do NOT gate on vertical |y|/|z|:
    # the wearer's hands are directly below the camera (large -y) and that is
    # expected, not a rejection signal.
    if x > cfg["max_x_right"]:
        return False, "lateral_right"
    if x < cfg["min_x_left"]:
        return False, "lateral_left"
    # 2. height (the cue x alone misses) --------------------------------
    # The wearer's arms enter from the BOTTOM of the ego frame, so their hands
    # are always LOW: y <= ~-0.25 m. An intruder across the table has hands
    # HIGHER in view (y >= ~-0.14 m). When an intruder hand drifts toward the
    # center (x < max_x_right) the x-cut misses it, but this height gate
    # catches it. Threshold sits in the empirical gap (wearer -0.25 .. -0.32,
    # intruder -0.08 .. -0.14). NOTE: assumes a downward-looking ego cam and a
    # seated table task; if the wearer raises hands to face height this would
    # wrongly drop them — revisit for non-table activities.
    if y > cfg["max_y"]:
        return False, "too_high"
    # 2. distance -------------------------------------------------------
    if not (cfg["min_dist"] < dist < cfg["max_dist"]):
        return False, "distance"
    # 3. span QA --------------------------------------------------------
    if not (cfg["min_span"] < span < cfg["max_span"]):
        return False, "span"
    return True, "ok"


def select_wearer_hands(hands, cfg):
    """From gated candidates keep the wearer's hands: closest, <=N, 1/handed."""
    hands = sorted(hands, key=lambda h: h["wrist_distance_m"])
    kept, seen = [], set()
    for h in hands:
        hd = h.get("handedness", "?")
        if hd not in seen:
            kept.append(h)
            seen.add(hd)
        if len(kept) >= cfg["max_hands"]:
            break
    if not kept:                          # handedness collapsed -> closest-N
        kept = hands[: cfg["max_hands"]]
    return kept


def apply_span_smoothing(frames, win):
    """Temporal median of span per handedness -> stored in _span_gate.

    Only the QA gate uses this smoothed value; the emitted hand_span_m and
    keypoints are left untouched. Stateless-friendly: uses a trailing window.
    """
    hist = defaultdict(lambda: deque(maxlen=win))
    for fr in frames:
        for h in fr["hands"]:
            hd = h.get("handedness", "?")
            hist[hd].append(float(h["hand_span_m"]))
            h["_span_gate"] = float(np.median(hist[hd]))


def main(in_path, out_path, cfg, smooth_span):
    with open(in_path) as f:
        data = json.load(f)
    frames = data["frames"]

    if smooth_span:
        apply_span_smoothing(frames, cfg["smooth_win"])

    counts = defaultdict(int)
    kept_frames, kept_hands = 0, 0
    for fr in frames:
        gated = []
        for h in fr["hands"]:
            keep, reason = wearer_gate(h, cfg)
            counts[reason] += 1
            if keep:
                gated.append(h)
        wearer = select_wearer_hands(gated, cfg)
        for h in wearer:
            h.pop("_span_gate", None)     # strip internal scratch field
        for h in fr["hands"]:
            h.pop("_span_gate", None)
        fr["hands"] = wearer
        if wearer:
            kept_frames += 1
            kept_hands += len(wearer)

    meta = data["metadata"]
    meta["frames_with_hands"] = kept_frames
    meta["total_hand_instances"] = kept_hands
    meta["wearer_filter"] = {
        "applied": True,
        "source_json": str(in_path),
        "params": cfg,
        "smooth_span": smooth_span,
        "reject_reasons": dict(counts),
    }
    meta["notes"] = meta.get("notes", "") + " | Wearer-filtered (post-process, filter_wearer.py)."

    with open(out_path, "w") as f:
        json.dump(data, f)

    total = sum(counts.values())
    print("=" * 56)
    print("WEARER FILTER COMPLETE")
    print("=" * 56)
    print(f"  in : {in_path}")
    print(f"  out: {out_path}")
    print(f"  candidate hands seen : {total}")
    for r in ("ok", "lateral_right", "lateral_left", "too_high", "distance", "span"):
        if counts[r]:
            print(f"    {r:14s}: {counts[r]:5d} ({counts[r]/total*100:4.1f}%)")
    print(f"  frames with wearer hands: {kept_frames}/{len(frames)}")
    print(f"  wearer hand instances   : {kept_hands}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--min-dist", type=float, default=0.10)
    p.add_argument("--max-dist", type=float, default=0.85)
    p.add_argument("--min-span", type=float, default=0.06)
    p.add_argument("--max-span", type=float, default=0.25)
    p.add_argument("--max-x-right", type=float, default=0.30,    # meters (primary cue)
                   help="drop hands with wrist x greater than this (intruder side)")
    p.add_argument("--min-x-left", type=float, default=-0.45,    # meters (loose)
                   help="drop hands with wrist x less than this (far-left, rare)")
    p.add_argument("--max-y", type=float, default=-0.20,         # meters (height gate)
                   help="drop hands with wrist y ABOVE this (too high = intruder "
                        "across table; wearer hands are low, y<=-0.25)")
    p.add_argument("--max-hands", type=int, default=2)
    p.add_argument("--smooth-span", action="store_true")
    p.add_argument("--smooth-win", type=int, default=5)
    args = p.parse_args()

    cfg = {
        "min_dist": args.min_dist, "max_dist": args.max_dist,
        "min_span": args.min_span, "max_span": args.max_span,
        "max_x_right": args.max_x_right, "min_x_left": args.min_x_left,
        "max_y": args.max_y,
        "max_hands": args.max_hands,
        "smooth_win": args.smooth_win,
    }
    main(args.in_path, args.out_path, cfg, args.smooth_span)
