"""
Step 3: Process the full stereo clip -> 3D hand keypoints (the deliverable).
---------------------------------------------------------------------------
Runs stereo detection + triangulation on every synchronized (left,right)
frame, preserving each frame's EXACT original timestamp, and produces:

  hands_3d.json   - per-frame 3D hand keypoints (meters, left-camera frame)
                    + wrist distance, hand span, reprojection error, timestamp
  overlay_left.mp4 - left video with detected keypoints drawn (verification)

Data-integrity guarantees (see .github/copilot-instructions.md):
  - FPS uses the exact reported value (never rounded).
  - Every output frame carries its exact original timestamp_us from *_pts.csv.
  - The .mp4 is a 2-min clip; the pts CSV covers the full recording — we map
    video frame i -> pts row i.

Usage:
    python process_clip.py \
        --data "/path/Stereo Video with IMU (shared)" \
        --output ../out_orbbec \
        [--max-frames 300]
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from calibration import load_stereo_calibration
from stereo3d import StereoHandDetector, lift_frame_to_3d, HAND_CONNECTIONS


def load_timestamps(pts_path: Path) -> list[int]:
    ts = []
    with open(pts_path) as f:
        reader = csv.reader(f)
        next(reader)  # header: timestamp_us
        for row in reader:
            if row:
                ts.append(int(row[0]))
    return ts


def draw_hand(img, pts_px, color):
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, tuple(pts_px[a].astype(int)),
                 tuple(pts_px[b].astype(int)), color, 2)
    for p in pts_px:
        cv2.circle(img, tuple(p.astype(int)), 4, color, -1)


def main(data_dir, out_dir, max_frames, detector_kind="mediapipe"):
    data = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cam_yaml = next(data.glob("*calibration_camera.yaml"))
    left_mp4 = data / "camera_left_2min.mp4"
    right_mp4 = data / "camera_right_2min.mp4"
    left_pts_csv = next(data.glob("*camera_left_pts.csv"))

    stereo = load_stereo_calibration(cam_yaml)
    left_ts = load_timestamps(left_pts_csv)

    capL = cv2.VideoCapture(str(left_mp4))
    capR = cv2.VideoCapture(str(right_mp4))
    fps = capL.get(cv2.CAP_PROP_FPS)             # EXACT reported fps
    W = int(capL.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(capL.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nL = int(capL.get(cv2.CAP_PROP_FRAME_COUNT))
    nR = int(capR.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(nL, nR)
    if max_frames:
        n_frames = min(n_frames, max_frames)

    print(f"📹 Orbbec Ego stereo | {W}x{H} @ {fps:.4f}fps")
    print(f"   Frames to process: {n_frames} (L={nL} R={nR})")
    print(f"   Baseline: {stereo.baseline_m*1000:.2f} mm")
    print(f"   Timestamps available: {len(left_ts)}")
    print()

    # Overlay writer (left camera, avc1 for compatibility)
    writer = None
    for codec in ("avc1", "mp4v"):
        w = cv2.VideoWriter(str(out_dir / "overlay_left.mp4"),
                            cv2.VideoWriter_fourcc(*codec), fps, (W, H))
        if w.isOpened():
            writer = w
            print(f"   Overlay codec: {codec}")
            break
        w.release()

    # Select the 2D hand detector backend.
    if detector_kind == "wilor":
        from wilor_detector import WiLoRDetector
        detector = WiLoRDetector()
    else:
        detector = StereoHandDetector(max_hands=4)

    colors = [(0, 255, 0), (0, 200, 255), (255, 100, 0), (255, 0, 255)]
    records = []
    frames_with_hands = 0
    total_hands = 0

    print("🚀 Processing...")
    for idx in range(n_frames):
        okL, left = capL.read()
        okR, right = capR.read()
        if not (okL and okR):
            break

        hands3d = lift_frame_to_3d(left, right, detector, stereo, wearer_only=True)

        # Build JSON record with EXACT timestamp
        ts = left_ts[idx] if idx < len(left_ts) else -1
        frame_rec = {
            "frame_idx": idx,
            "timestamp_us": ts,
            "hands": [],
        }
        for h in hands3d:
            frame_rec["hands"].append({
                "handedness": h.handedness,
                "keypoints_3d": h.keypoints_3d.round(5).tolist(),
                "wrist_distance_m": round(h.distance_m, 4),
                "hand_span_m": round(h.hand_span_m(), 4),
                "reproj_error": round(h.reproj_error_px, 5),
            })
        records.append(frame_rec)

        if hands3d:
            frames_with_hands += 1
            total_hands += len(hands3d)

        # Draw overlay on left frame
        for i, h in enumerate(hands3d):
            draw_hand(left, h.left_2d, colors[i % len(colors)])
            wrist = tuple(h.left_2d[0].astype(int))
            cv2.putText(left, f"{h.distance_m*100:.0f}cm", wrist,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(left, f"Frame {idx} | hands: {len(hands3d)}",
                    (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 200), 2)
        writer.write(left)

        if (idx + 1) % 60 == 0:
            pct = (idx + 1) / n_frames * 100
            print(f"   [{pct:5.1f}%] frame {idx+1}/{n_frames} — hands this frame: {len(hands3d)}")

    capL.release(); capR.release(); writer.release()
    detector.close()

    # Save the deliverable JSON
    output = {
        "metadata": {
            "device": "Orbbec Ego",
            "baseline_mm": round(stereo.baseline_m * 1000, 3),
            "fps": fps,
            "frame_width": W,
            "frame_height": H,
            "coordinate_frame": "left_camera",
            "units": "meters",
            "distortion_model": "KB",
            "num_frames": len(records),
            "frames_with_hands": frames_with_hands,
            "total_hand_instances": total_hands,
            "notes": "3D hand keypoints via stereo triangulation of undistorted "
                     "MediaPipe points. Wearer-only (distance-filtered).",
        },
        "frames": records,
    }
    json_path = out_dir / "hands_3d.json"
    with open(json_path, "w") as f:
        json.dump(output, f)

    size_mb = json_path.stat().st_size / 1e6
    print()
    print("=" * 60)
    print("✅ CLIP PROCESSING COMPLETE")
    print("=" * 60)
    print(f"   Deliverable: {json_path} ({size_mb:.1f} MB)")
    print(f"   Overlay:     {out_dir / 'overlay_left.mp4'}")
    print(f"   Frames:      {len(records)}")
    print(f"   With hands:  {frames_with_hands} ({frames_with_hands/len(records)*100:.0f}%)")
    print(f"   Hand instances: {total_hands}")

    # Quick stats on 3D quality
    spans = [hh["hand_span_m"] for r in records for hh in r["hands"]]
    dists = [hh["wrist_distance_m"] for r in records for hh in r["hands"]]
    reprojs = [hh["reproj_error"] for r in records for hh in r["hands"]]
    if spans:
        print(f"   Hand span:  mean={np.mean(spans)*100:.1f}cm  std={np.std(spans)*100:.1f}cm")
        print(f"   Distance:   mean={np.mean(dists)*100:.1f}cm  range=[{min(dists)*100:.0f},{max(dists)*100:.0f}]cm")
        print(f"   Reproj err: mean={np.mean(reprojs):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="../out_orbbec")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--detector", default="mediapipe",
                        choices=["mediapipe", "wilor"],
                        help="2D hand detector backend")
    args = parser.parse_args()
    main(args.data, args.output, args.max_frames, args.detector)
