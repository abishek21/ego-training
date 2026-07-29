"""
Render hands_3d.json (or a filtered view) back onto the LEFT video.
------------------------------------------------------------------
The JSON stores only 3D points (left-camera frame, meters). To draw them we
REPROJECT to pixels with the KB fisheye model (cv2.fisheye.projectPoints),
using the left camera's K, D. Points are already in the left-camera frame, so
rvec = tvec = 0.

Two modes:
  --json ONE         : draw all hands in that JSON (green).
  --compare RAW FILT : draw hands kept by FILT in green, and hands present in
                       RAW but DROPPED by FILT in red. Lets you visually
                       confirm the intruder (2nd-person hand) was removed.

Hands are matched between RAW and FILT by wrist 3D proximity (< 1 cm), which
is exact here since the filter only drops/keeps, never edits, keypoints.

Usage:
    # confirm intruder removal on first 5 seconds
    python render_overlay.py \
        --data "/path/Stereo Video with IMU (shared)" \
        --compare ../hands_3d.json ../hands_3d_wearer.json \
        --out ../test_filter_5s.mp4 --seconds 5
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from calibration import load_stereo_calibration
from stereo3d import HAND_CONNECTIONS

GREEN = (0, 255, 0)
RED = (0, 0, 255)
# Handedness colors (single-json mode): Left=green, Right=cyan.
HAND_COLORS = {"Left": (0, 255, 0), "Right": (255, 200, 0)}


def load_timestamps(pts_path):
    """Read *_camera_left_pts.csv -> list[int] microsecond timestamps."""
    ts = []
    with open(pts_path) as f:
        reader = csv.reader(f)
        next(reader)  # header: timestamp_us
        for row in reader:
            if row:
                ts.append(int(row[0]))
    return ts


def project_kb(pts3d, cam):
    """(N,3) left-camera-frame meters -> (N,2) pixels via KB fisheye."""
    obj = np.asarray(pts3d, dtype=np.float64).reshape(-1, 1, 3)
    rvec = np.zeros(3); tvec = np.zeros(3)
    px, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, cam.K, cam.D)
    return px.reshape(-1, 2)


def draw_hand(img, pts_px, color, thick=2, r=4):
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, tuple(pts_px[a].astype(int)),
                 tuple(pts_px[b].astype(int)), color, thick)
    for p in pts_px:
        cv2.circle(img, tuple(p.astype(int)), r, color, -1)


def load_frames(path):
    with open(path) as f:
        return json.load(f)["frames"]


def wrist_key(hand):
    w = hand["keypoints_3d"][0]
    return (round(w[0], 4), round(w[1], 4), round(w[2], 4))


def main(data_dir, out_path, json_one, compare, seconds, max_frames):
    data = Path(data_dir)
    cam_yaml = next(data.glob("*calibration_camera.yaml"))
    left_mp4 = data / "camera_left_2min.mp4"
    left_pts_csv = next(data.glob("*camera_left_pts.csv"))
    stereo = load_stereo_calibration(cam_yaml)
    left_ts = load_timestamps(left_pts_csv)   # ground-truth microsecond pts

    if compare:
        raw_frames = load_frames(compare[0])
        filt_frames = load_frames(compare[1])
        # index filtered kept-hands by wrist key per frame_idx
        kept = {fr["frame_idx"]: {wrist_key(h) for h in fr["hands"]}
                for fr in filt_frames}
        frames = raw_frames
    else:
        frames = load_frames(json_one)
        kept = None

    cap = cv2.VideoCapture(str(left_mp4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n = int(round(seconds * fps)) if seconds else len(frames)
    if max_frames:
        n = min(n, max_frames)
    n = min(n, len(frames))

    writer = None
    for codec in ("avc1", "mp4v"):
        w = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*codec),
                            fps, (W, H))
        if w.isOpened():
            writer = w
            break
        w.release()

    by_idx = {fr["frame_idx"]: fr for fr in frames}
    n_kept = n_dropped = 0
    ts_mismatches = 0
    print(f"Rendering {n} frames @ {fps:.6f}fps -> {out_path}")
    for idx in range(n):
        ok, img = cap.read()
        if not ok:
            break
        fr = by_idx.get(idx)

        # ── Data-integrity: verify the index chain via the µs timestamp ──
        # video frame idx -> pts CSV row idx -> JSON frame's timestamp_us must
        # all agree. If the decoder ever drops/dups a frame this fails loudly.
        ts_csv = left_ts[idx] if idx < len(left_ts) else -1
        ts_json = fr.get("timestamp_us", -1) if fr else -1
        if fr is not None and ts_json != ts_csv:
            ts_mismatches += 1
            if ts_mismatches <= 5:
                print(f"  ⚠️  ts mismatch @frame {idx}: "
                      f"json={ts_json} csv={ts_csv}")

        if fr:
            for h in fr["hands"]:
                pts3d = np.asarray(h["keypoints_3d"])
                # Prefer the detector's RAW 2D pixels if the JSON carries them
                # (self-contained, pixel-exact). Fall back to reprojecting the
                # 3D via the KB model for older JSONs that store only 3D.
                if "keypoints_2d_left" in h:
                    px = np.asarray(h["keypoints_2d_left"], dtype=np.float64)
                    src = "2d"
                else:
                    px = project_kb(pts3d, stereo.left)
                    src = "reproj"
                if kept is not None:
                    is_kept = wrist_key(h) in kept.get(idx, set())
                    color = GREEN if is_kept else RED
                    n_kept += is_kept
                    n_dropped += (not is_kept)
                    label = f"{h['wrist_distance_m']*100:.0f}cm x={pts3d[0][0]:+.2f}"
                else:
                    # single-json mode: color + label by handedness so we can
                    # verify Left/Right are maintained correctly.
                    handed = h.get("handedness", "?")
                    color = HAND_COLORS.get(handed, (200, 200, 200))
                    label = f"{handed} x={pts3d[0][0]:+.2f} {h['wrist_distance_m']*100:.0f}cm"
                draw_hand(img, px, color)
                cv2.putText(img, label, tuple(px[0].astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
        if kept is not None:
            tag = "GREEN=wearer  RED=dropped"
        else:
            tag = "GREEN=Left  CYAN=Right"
        cv2.putText(img, f"Frame {idx}  {tag}", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 200), 2)
        # Burn the EXACT original microsecond timestamp into the frame.
        cv2.putText(img, f"pts_us={ts_csv}", (15, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
        writer.write(img)

    cap.release(); writer.release()
    print("Done.")
    print(f"  timestamp checks: {n} frames, mismatches={ts_mismatches} "
          f"({'OK — index chain intact' if ts_mismatches == 0 else 'DRIFT DETECTED'})")
    if kept is not None:
        print(f"  kept (green)   hand-draws: {n_kept}")
        print(f"  dropped (red)  hand-draws: {n_dropped}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--json", dest="json_one")
    p.add_argument("--compare", nargs=2, metavar=("RAW", "FILTERED"))
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args()
    assert args.json_one or args.compare, "provide --json or --compare"
    main(args.data, args.out, args.json_one, args.compare,
         args.seconds, args.max_frames)
