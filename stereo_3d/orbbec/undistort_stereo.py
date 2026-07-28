"""
Step 1: Undistort Orbbec Ego fisheye stereo video.
----------------------------------------------------
Reads left + right fisheye (Kannala-Brandt "KB") videos, undistorts each
frame using the provided camera calibration, and writes undistorted videos
+ a per-frame timestamp manifest so NOTHING drifts vs the IMU.

Critical timestamp handling:
  - The .mp4 is a 2-min clip (~3592 frames)
  - The *_pts.csv covers the FULL recording (~31890 rows)
  - We map video frame i -> pts row i (the first N rows), preserving the
    exact original timestamp_us for every output frame.

Outputs (in --output dir):
  left_undistorted.mp4
  right_undistorted.mp4
  frames_manifest.csv   (frame_idx, left_ts_us, right_ts_us)
  undistort_params.json (K, D, new_K used — for reproducible 2D<->3D later)

Usage:
    python undistort_stereo.py \
        --data "/path/Stereo Video with IMU (shared)" \
        --output ../data_orbbec_undistorted \
        --max-frames 300      # optional, for a quick test
"""

import argparse
import json
import csv
import re
from pathlib import Path

import cv2
import numpy as np
import yaml


# ── Calibration parsing ─────────────────────────────────────────────

def load_camera_calib(calib_path: Path):
    """Parse the Orbbec camera calibration YAML → per-camera K, D (KB fisheye)."""
    with open(calib_path) as f:
        # The file has a comment header; yaml.safe_load handles it.
        calib = yaml.safe_load(f)

    cams = {}
    for cam in calib["cameras"]:
        intr = cam["intrinsics"]
        dist = cam["distortion"]
        K = np.array([
            [intr["fx"], 0.0,        intr["cx"]],
            [0.0,        intr["fy"], intr["cy"]],
            [0.0,        0.0,        1.0],
        ], dtype=np.float64)
        # Kannala-Brandt fisheye uses 4 coefficients: k1..k4
        D = np.array([
            dist["k1"], dist["k2"], dist["k3"], dist["k4"]
        ], dtype=np.float64)
        cams[cam["name"]] = {
            "id": cam["id"],
            "K": K,
            "D": D,
            "width": cam["image_width"],
            "height": cam["image_height"],
            "model": cam["distortion_model"],
        }
    return cams


def load_timestamps(pts_path: Path):
    """Read the *_pts.csv (single column timestamp_us) → list of ints."""
    ts = []
    with open(pts_path) as f:
        reader = csv.reader(f)
        header = next(reader)  # timestamp_us
        for row in reader:
            if row:
                ts.append(int(row[0]))
    return ts


# ── Undistortion ────────────────────────────────────────────────────

def build_undistort_maps(K, D, size, balance=0.5, fov_scale=1.0):
    """
    Precompute fisheye undistortion remap maps.

    balance   : 0 = crop tight (zoom in, may stretch)
                1 = keep full FOV (black borders, less stretch)
    fov_scale : >1 zooms OUT (fits more content, reduces elongation)
    """
    W, H = size
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (W, H), np.eye(3), balance=balance, fov_scale=fov_scale
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (W, H), cv2.CV_16SC2
    )
    return map1, map2, new_K


def process_side(video_path, pts, cam, out_video_path, map1, map2,
                 fps, max_frames=None):
    """Undistort every frame of one camera; return per-frame timestamps used."""
    cap = cv2.VideoCapture(str(video_path))
    W, H = cam["width"], cam["height"]

    fourcc = None
    writer = None
    for codec in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        w = cv2.VideoWriter(str(out_video_path), fourcc, fps, (W, H))
        if w.isOpened():
            writer = w
            break
        w.release()

    used_ts = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and idx >= max_frames:
            break

        undist = cv2.remap(frame, map1, map2,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT)
        writer.write(undist)

        # Preserve the EXACT original timestamp for this frame
        used_ts.append(pts[idx] if idx < len(pts) else -1)
        idx += 1

        if idx % 60 == 0:
            print(f"      frame {idx}")

    cap.release()
    writer.release()
    return used_ts


def main(data_dir, out_dir, max_frames, balance, fov_scale):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Locate files (names have the Ego_<serial>_<ts> prefix)
    def find(pattern):
        matches = list(data_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No file matching {pattern} in {data_dir}")
        return matches[0]

    calib_cam = find("*calibration_camera.yaml")
    left_mp4 = find("camera_left_2min.mp4")
    right_mp4 = find("camera_right_2min.mp4")
    left_pts = find("*camera_left_pts.csv")
    right_pts = find("*camera_right_pts.csv")

    print(f"📂 Calibration: {calib_cam.name}")
    cams = load_camera_calib(calib_cam)
    print(f"   Cameras: {list(cams.keys())}")

    # Map video files to calibration entries (IR_L = left, IR_R = right)
    cam_left = cams.get("IR_L") or list(cams.values())[0]
    cam_right = cams.get("IR_R") or list(cams.values())[1]
    print(f"   Left  model={cam_left['model']}  fx={cam_left['K'][0,0]:.1f}")
    print(f"   Right model={cam_right['model']} fx={cam_right['K'][0,0]:.1f}")

    # Timestamps (full recording; we use the first N to match the mp4)
    left_ts = load_timestamps(left_pts)
    right_ts = load_timestamps(right_pts)
    print(f"   Timestamps: left={len(left_ts)} right={len(right_ts)}")

    # Video fps (use the actual reported value to avoid drift)
    capL = cv2.VideoCapture(str(left_mp4))
    fps = capL.get(cv2.CAP_PROP_FPS)
    nL = int(capL.get(cv2.CAP_PROP_FRAME_COUNT))
    capL.release()
    print(f"   Video fps={fps:.4f}, left frames={nL}")

    size = (cam_left["width"], cam_left["height"])

    # Build undistort maps per camera
    print(f"\n🔧 Building undistortion maps (KB fisheye, balance={balance}, fov_scale={fov_scale})...")
    mapL1, mapL2, newK_L = build_undistort_maps(cam_left["K"], cam_left["D"], size, balance, fov_scale)
    mapR1, mapR2, newK_R = build_undistort_maps(cam_right["K"], cam_right["D"], size, balance, fov_scale)

    # Process both cameras
    print("\n🎬 Undistorting LEFT...")
    used_ts_L = process_side(left_mp4, left_ts, cam_left,
                             out_dir / "left_undistorted.mp4",
                             mapL1, mapL2, fps, max_frames)
    print("🎬 Undistorting RIGHT...")
    used_ts_R = process_side(right_mp4, right_ts, cam_right,
                             out_dir / "right_undistorted.mp4",
                             mapR1, mapR2, fps, max_frames)

    # Write frame manifest (exact timestamps preserved)
    n = min(len(used_ts_L), len(used_ts_R))
    manifest = out_dir / "frames_manifest.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "left_ts_us", "right_ts_us", "lr_dt_us"])
        for i in range(n):
            lt, rt = used_ts_L[i], used_ts_R[i]
            w.writerow([i, lt, rt, rt - lt])

    # Save undistortion params for reproducible 2D<->3D later
    params = {
        "fps": fps,
        "image_size": {"width": size[0], "height": size[1]},
        "left": {
            "K": cam_left["K"].tolist(),
            "D": cam_left["D"].tolist(),
            "new_K": newK_L.tolist(),
            "model": cam_left["model"],
        },
        "right": {
            "K": cam_right["K"].tolist(),
            "D": cam_right["D"].tolist(),
            "new_K": newK_R.tolist(),
            "model": cam_right["model"],
        },
    }
    with open(out_dir / "undistort_params.json", "w") as f:
        json.dump(params, f, indent=2)

    print()
    print("=" * 60)
    print("✅ UNDISTORTION COMPLETE")
    print("=" * 60)
    print(f"   Output dir:  {out_dir}")
    print(f"   Frames:      {n} (fps={fps:.4f})")
    print(f"   left_undistorted.mp4 / right_undistorted.mp4")
    print(f"   frames_manifest.csv  (exact timestamps preserved)")
    print(f"   undistort_params.json (K/D/new_K for 3D lifting)")
    # Sanity: L-R timestamp offset
    if n:
        dts = [abs(used_ts_R[i] - used_ts_L[i]) for i in range(n)]
        print(f"   L-R timestamp offset: mean={np.mean(dts):.0f}us max={np.max(dts):.0f}us")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Provider data folder")
    parser.add_argument("--output", default="../data_orbbec_undistorted")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--balance", type=float, default=0.5,
                        help="0=tight crop (may stretch), 1=full FOV (black borders)")
    parser.add_argument("--fov-scale", type=float, default=1.0,
                        help=">1 zooms out (reduces elongation)")
    args = parser.parse_args()
    main(args.data, args.output, args.max_frames, args.balance, args.fov_scale)
