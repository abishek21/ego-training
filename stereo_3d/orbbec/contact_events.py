"""
Stage 4b: hand-object CONTACT EVENTS from 3D hands + SAM2 masks.
----------------------------------------------------------------
For each frame, reproject the wearer's 3D hand keypoints to 2D (KB fisheye) and
measure fingertip proximity to each object mask (package_bag, clips_bag). Contact
= fingertips within `--touch-px` of / inside the mask. Grasp/release EVENTS are
the rising/falling edges of that per-frame contact signal, debounced to remove
flicker.

Renders a validation video: left = frame + masks + hand skeleton; right = a
status panel with live contact state per (hand, object) + a scrolling event log.
You eyeball whether grasp/release fire at the right moments, and we tune --touch-px.

HONEST CAVEATS:
- Contact is INFERRED from 2D proximity (no touch sensor). 2D can't disambiguate
  depth: a hand passing IN FRONT of an object still projects onto its mask. For
  this ego manipulation view (hand & object at similar depth) it's reasonable; a
  3D-depth gate can be added later using the hand's known distance.
- Masks are LEFT-camera only; objects have no 3D yet (that's the stretch goal).

Usage:
    python contact_events.py \
        --hands ../hands_3d_wearer.json \
        --masks ../../sam2_pipeline/masks_left_full/masks \
        --data "/path/Stereo Video with IMU (shared)" \
        --out-json ../contact_events.json \
        --out-video ../contact_overlay.mp4 \
        --touch-px 25 --seconds 0   # 0 = full clip; set e.g. 20 for a quick test
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from calibration import load_stereo_calibration
from stereo3d import HAND_CONNECTIONS

FINGERTIPS = [4, 8, 12, 16, 20]           # thumb..pinky tips
OBJ_COLORS = {"obj_1": (0, 100, 255), "obj_2": (255, 150, 0)}   # BGR
OBJ_NAMES = {"obj_1": "package_bag", "obj_2": "clips_bag"}
HAND_COLORS = {"Left": (0, 255, 0), "Right": (255, 200, 0)}


def project_kb(pts3d, cam):
    obj = np.asarray(pts3d, np.float64).reshape(-1, 1, 3)
    px, _ = cv2.fisheye.projectPoints(obj, np.zeros(3), np.zeros(3), cam.K, cam.D)
    return px.reshape(-1, 2)


def load_mask(masks_dir, frame_idx):
    p = masks_dir / f"{frame_idx:05d}.npz"
    if not p.exists():
        return {}
    d = np.load(p)
    return {k: d[k] for k in d.keys()}


def contact_distance(pts_px, mask):
    """Min pixel distance from any fingertip to the mask (0 if inside)."""
    H, W = mask.shape
    # distance transform of the BACKGROUND -> distance to nearest mask pixel
    inv = (mask == 0).astype(np.uint8)
    dt = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    dmin = 1e9
    inside = False
    for (x, y) in pts_px:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H:
            if mask[yi, xi] > 0:
                inside = True
                dmin = 0.0
            else:
                dmin = min(dmin, float(dt[yi, xi]))
    return dmin, inside


def debounce(signal, min_on=3, min_off=3):
    """Clean a boolean per-frame signal: require min consecutive frames to flip."""
    out = signal.copy()
    n = len(signal)
    i = 0
    state = signal[0]
    run_start = 0
    for i in range(1, n + 1):
        if i == n or signal[i] != state:
            run_len = i - run_start
            need = min_on if state else min_off
            if run_len < need and run_start > 0:
                out[run_start:i] = out[run_start - 1]   # absorb short run
            if i < n:
                state = signal[i]; run_start = i
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hands", required=True)
    p.add_argument("--masks", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-video", required=True)
    p.add_argument("--touch-px", type=float, default=25.0,
                   help="fingertip within this many px of mask => contact")
    p.add_argument("--min-on", type=int, default=3, help="debounce: min grasp frames")
    p.add_argument("--min-off", type=int, default=3, help="debounce: min release frames")
    p.add_argument("--seconds", type=float, default=0.0, help="0=full clip")
    args = p.parse_args()

    data = Path(args.data)
    masks_dir = Path(args.masks)
    stereo = load_stereo_calibration(next(data.glob("*calibration_camera.yaml")))
    left_mp4 = data / "camera_left_2min.mp4"

    with open(args.hands) as f:
        hd = json.load(f)
    frames = hd["frames"]

    cap = cv2.VideoCapture(str(left_mp4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = len(frames) if args.seconds <= 0 else min(len(frames), int(args.seconds * fps))

    # ---- PASS 1: per-frame raw contact signal per (handedness, object) ----
    # key = (handedness, obj_key) -> list of (frame_idx, contact_bool, dist)
    raw = {}
    hands_by_frame = {fr["frame_idx"]: fr for fr in frames}
    print(f"Computing contact for {n} frames (touch<{args.touch_px}px)...")
    for idx in range(n):
        fr = hands_by_frame.get(idx)
        masks = load_mask(masks_dir, idx)
        if not fr:
            continue
        for h in fr["hands"]:
            hd_lbl = h["handedness"]
            tips3d = np.asarray(h["keypoints_3d"])[FINGERTIPS]
            tips_px = project_kb(tips3d, stereo.left)
            for okey, mask in masks.items():
                dist, inside = contact_distance(tips_px, mask)
                contact = inside or dist <= args.touch_px
                raw.setdefault((hd_lbl, okey), {})[idx] = (contact, dist)

    # ---- extract debounced events ----
    events = []
    per_pair_signal = {}
    for (hd_lbl, okey), fr_map in raw.items():
        idxs = sorted(fr_map.keys())
        sig = np.array([fr_map[i][0] for i in idxs], dtype=bool)
        sig = debounce(sig, args.min_on, args.min_off)
        per_pair_signal[(hd_lbl, okey)] = dict(zip(idxs, sig))
        prev = False
        for k, i in enumerate(idxs):
            cur = sig[k]
            if cur and not prev:
                events.append({"type": "grasp", "frame_idx": int(i),
                               "timestamp_us": hands_by_frame[i]["timestamp_us"],
                               "hand": hd_lbl, "object": OBJ_NAMES.get(okey, okey)})
            elif not cur and prev:
                events.append({"type": "release", "frame_idx": int(i),
                               "timestamp_us": hands_by_frame[i]["timestamp_us"],
                               "hand": hd_lbl, "object": OBJ_NAMES.get(okey, okey)})
            prev = cur
    events.sort(key=lambda e: e["frame_idx"])

    # ---- write events JSON ----
    out = {
        "metadata": {
            "source_hands": str(args.hands),
            "source_masks": str(masks_dir),
            "touch_px": args.touch_px,
            "debounce": {"min_on": args.min_on, "min_off": args.min_off},
            "objects": OBJ_NAMES,
            "method": "2D fingertip-to-mask proximity (KB reprojection); inferred, not sensed",
            "caveat": "2D proximity ignores depth; hand in front of object may register contact.",
            "n_events": len(events),
        },
        "events": events,
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  -> {len(events)} events -> {args.out_json}")

    # ---- PASS 2: render validation overlay ----
    PANEL = 460
    writer = None
    for codec in ("avc1", "mp4v"):
        w = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*codec),
                            fps, (W + PANEL, H))
        if w.isOpened():
            writer = w; break
        w.release()

    log = []
    print("Rendering overlay...")
    for idx in range(n):
        ok, img = cap.read()
        if not ok:
            break
        fr = hands_by_frame.get(idx)
        masks = load_mask(masks_dir, idx)
        # draw masks tinted
        for okey, mask in masks.items():
            color = OBJ_COLORS.get(okey, (200, 200, 200))
            overlay = img.copy()
            overlay[mask > 0] = color
            img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
        # draw hands
        if fr:
            for h in fr["hands"]:
                kp = project_kb(np.asarray(h["keypoints_3d"]), stereo.left)
                col = HAND_COLORS.get(h["handedness"], (255, 255, 255))
                for a, b in HAND_CONNECTIONS:
                    cv2.line(img, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)), col, 2)
                for t in FINGERTIPS:
                    cv2.circle(img, tuple(kp[t].astype(int)), 5, col, -1)

        # side panel
        panel = np.zeros((H, PANEL, 3), np.uint8)
        y = 40
        cv2.putText(panel, f"Frame {idx}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 200), 2)
        y += 50
        cv2.putText(panel, "CONTACT STATE", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 40
        for (hd_lbl, okey), sigmap in sorted(per_pair_signal.items()):
            on = sigmap.get(idx, False)
            oname = OBJ_NAMES.get(okey, okey)
            color = (0, 255, 0) if on else (90, 90, 90)
            txt = f"{hd_lbl:5s} - {oname:12s} {'CONTACT' if on else '.'}"
            cv2.putText(panel, txt, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            y += 32
        # event log (recent)
        for e in events:
            if e["frame_idx"] == idx:
                log.append(f"f{idx} {e['type'].upper()} {e['hand']}/{e['object']}")
        y += 20
        cv2.putText(panel, "EVENTS", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 34
        for line in log[-14:]:
            c = (0, 220, 0) if "GRASP" in line else (0, 160, 255)
            cv2.putText(panel, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
            y += 26

        writer.write(np.hstack([img, panel]))
    cap.release(); writer.release()
    print(f"  -> {args.out_video}")


if __name__ == "__main__":
    main()
