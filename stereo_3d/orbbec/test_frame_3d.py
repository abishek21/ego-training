"""
Single-frame stereo 3D test — SEE the 3D hand keypoints.
---------------------------------------------------------
Loads one synchronized (left,right) frame, lifts hands to 3D, prints the
metric coordinates + QA metrics, and saves a side-by-side visualization
with detected 2D keypoints (so you can verify detection + matching).

Usage:
    python test_frame_3d.py \
        --data "/path/Stereo Video with IMU (shared)" \
        --frame 75
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from calibration import load_stereo_calibration
from stereo3d import (StereoHandDetector, lift_frame_to_3d, HAND_CONNECTIONS)


def read_frame(video_path: Path, idx: int):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {idx} from {video_path}")
    return frame


def draw_hand(img, pts_px, color=(0, 255, 0)):
    for a, b in HAND_CONNECTIONS:
        pa = tuple(pts_px[a].astype(int))
        pb = tuple(pts_px[b].astype(int))
        cv2.line(img, pa, pb, color, 2)
    for p in pts_px:
        cv2.circle(img, tuple(p.astype(int)), 4, color, -1)
    return img


def main(data_dir, frame_idx):
    data = Path(data_dir)
    cam_yaml = next(data.glob("*calibration_camera.yaml"))
    left_mp4 = next(data.glob("camera_left_2min.mp4").__iter__()) \
        if False else data / "camera_left_2min.mp4"
    right_mp4 = data / "camera_right_2min.mp4"

    stereo = load_stereo_calibration(cam_yaml)
    print(f"📐 Baseline: {stereo.baseline_m*1000:.2f} mm")

    left = read_frame(left_mp4, frame_idx)
    right = read_frame(right_mp4, frame_idx)
    print(f"🖼️  Frame {frame_idx}: left {left.shape[1]}x{left.shape[0]}")

    detector = StereoHandDetector(max_hands=4)
    hands3d = lift_frame_to_3d(left, right, detector, stereo, wearer_only=True)
    detector.close()

    print()
    print("=" * 60)
    print(f"RESULT: {len(hands3d)} wearer hand(s) triangulated")
    print("=" * 60)

    colors = [(0, 255, 0), (0, 200, 255), (255, 100, 0), (255, 0, 255)]
    for i, h in enumerate(hands3d):
        print(f"\n Hand {i} [{h.handedness}]")
        print(f"   wrist distance:   {h.distance_m*100:.1f} cm from camera")
        print(f"   hand span:        {h.hand_span_m()*100:.1f} cm (wrist->mid tip)")
        print(f"   reproj error:     {h.reproj_error_px:.4f} (normalized)")
        wrist = h.keypoints_3d[0]
        idx_tip = h.keypoints_3d[8]
        print(f"   wrist 3D (m):     [{wrist[0]:+.3f}, {wrist[1]:+.3f}, {wrist[2]:+.3f}]")
        print(f"   index tip 3D (m): [{idx_tip[0]:+.3f}, {idx_tip[1]:+.3f}, {idx_tip[2]:+.3f}]")

        # Draw on both images
        c = colors[i % len(colors)]
        draw_hand(left, h.left_2d, c)
        draw_hand(right, h.right_2d, c)

    # Side-by-side visualization
    combined = np.hstack([left, right])
    scale = 1600 / combined.shape[1]
    combined = cv2.resize(combined, None, fx=scale, fy=scale)
    cv2.putText(combined, f"Frame {frame_idx}: LEFT | RIGHT  ({len(hands3d)} hands 3D)",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out_path = Path(__file__).parent / f"_frame3d_{frame_idx}.jpg"
    cv2.imwrite(str(out_path), combined)
    print(f"\n💾 Saved visualization: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--frame", type=int, default=75)
    args = parser.parse_args()
    main(args.data, args.frame)
