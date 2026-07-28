"""
GPU test: WiLoR detection rate + single-frame 3D validation.
-------------------------------------------------------------
Run this on the GPU pod AFTER installing WiLoR-mini. It:
  1. Reports WiLoR detection rate over N frames (both cameras) —
     compare against MediaPipe's ~38% baseline.
  2. Triangulates one frame to 3D and prints metric coords + hand span
     (sanity: adult hand span ~18cm, wrist distance within arm's reach).

Usage:
    python test_wilor_gpu.py \
        --data "/path/Stereo Video with IMU (shared)" \
        --frames 300 --viz-frame 75
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from calibration import load_stereo_calibration
from stereo3d import (undistort_points_kb, triangulate, reprojection_error,
                      match_hands, HAND_CONNECTIONS)
from wilor_detector import WiLoRDetector


def detection_rate(data: Path, detector, n_frames: int):
    capL = cv2.VideoCapture(str(data / "camera_left_2min.mp4"))
    capR = cv2.VideoCapture(str(data / "camera_right_2min.mp4"))
    both = only_one = neither = left_any = 0
    total = 0
    for _ in range(n_frames):
        okL, L = capL.read(); okR, R = capR.read()
        if not (okL and okR):
            break
        total += 1
        nl = len(detector.detect(L))
        nr = len(detector.detect(R))
        if nl > 0:
            left_any += 1
        if nl > 0 and nr > 0:
            both += 1
        elif nl > 0 or nr > 0:
            only_one += 1
        else:
            neither += 1
    capL.release(); capR.release()
    print("\n=== WiLoR DETECTION RATE ===")
    print(f"  Frames tested:               {total}")
    print(f"  Any left detection:          {left_any} ({left_any/total*100:.0f}%)  "
          f"[MediaPipe baseline was 38%]")
    print(f"  BOTH cameras (usable 3D):    {both} ({both/total*100:.0f}%)  "
          f"[MediaPipe baseline was 30%]")
    print(f"  Only ONE camera:             {only_one} ({only_one/total*100:.0f}%)")
    print(f"  Neither:                     {neither} ({neither/total*100:.0f}%)")


def validate_frame_3d(data: Path, detector, stereo, frame_idx: int):
    def read(vid, idx):
        cap = cv2.VideoCapture(str(vid))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read(); cap.release()
        return f if ok else None

    L = read(data / "camera_left_2min.mp4", frame_idx)
    R = read(data / "camera_right_2min.mp4", frame_idx)
    lh = detector.detect(L)
    rh = detector.detect(R)
    print(f"\n=== SINGLE-FRAME 3D (frame {frame_idx}) ===")
    print(f"  left hands: {len(lh)}  right hands: {len(rh)}")
    if not lh or not rh:
        print("  (no stereo-visible hand this frame)")
        return

    matches = match_hands(lh, rh, stereo.left, stereo.right)
    for li, ri in matches:
        ln = undistort_points_kb(lh[li].keypoints_px, stereo.left)
        rn = undistort_points_kb(rh[ri].keypoints_px, stereo.right)
        p3d = triangulate(ln, rn, stereo)
        reproj = reprojection_error(p3d, ln, rn, stereo)
        dist = float(np.linalg.norm(p3d[0]))
        span = float(np.linalg.norm(p3d[12] - p3d[0]))
        print(f"  [{lh[li].handedness}] wrist_dist={dist*100:.1f}cm  "
              f"span={span*100:.1f}cm  reproj={reproj:.4f}")

        # Save viz
        for a, b in HAND_CONNECTIONS:
            cv2.line(L, tuple(lh[li].keypoints_px[a].astype(int)),
                     tuple(lh[li].keypoints_px[b].astype(int)), (0, 255, 0), 2)
            cv2.line(R, tuple(rh[ri].keypoints_px[a].astype(int)),
                     tuple(rh[ri].keypoints_px[b].astype(int)), (0, 255, 0), 2)
    combined = np.hstack([L, R])
    combined = cv2.resize(combined, None, fx=1600 / combined.shape[1],
                          fy=1600 / combined.shape[1])
    out = Path(__file__).parent / f"_wilor_frame3d_{frame_idx}.jpg"
    cv2.imwrite(str(out), combined)
    print(f"  saved: {out}")


def main(data_dir, n_frames, viz_frame):
    data = Path(data_dir)
    stereo = load_stereo_calibration(next(data.glob("*calibration_camera.yaml")))
    print(f"📐 Baseline: {stereo.baseline_m*1000:.2f} mm")

    detector = WiLoRDetector()
    validate_frame_3d(data, detector, stereo, viz_frame)
    detection_rate(data, detector, n_frames)
    detector.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--viz-frame", type=int, default=75)
    args = parser.parse_args()
    main(args.data, args.frames, args.viz_frame)
