"""
Overlay — Render SAM 2 masks onto the video.
---------------------------------------------
Reads masks (from segment.py) and draws colored semi-transparent
overlays on each frame, producing the final segmented video.

Usage:
    python overlay.py --video ego_press.mp4 --masks masks_output/ --output segmented.mp4
"""

import cv2
import json
import argparse
import numpy as np
from pathlib import Path


# Colors per object (BGR)
OBJ_COLORS = {
    1: (0, 0, 255),      # Red   — iron
    2: (0, 255, 0),      # Green — sheet
    3: (255, 200, 0),    # Cyan  — cloth
    4: (0, 255, 255),    # Yellow
    5: (255, 0, 255),    # Magenta
}

ALPHA = 0.5  # Mask transparency


def overlay(video_path, masks_dir, output_path):
    masks_dir = Path(masks_dir)

    # Load summary
    with open(masks_dir / "summary.json") as f:
        summary = json.load(f)

    obj_labels = {int(k): v for k, v in summary["objects"].items()}
    fps = summary["fps"]
    frame_w = summary["frame_width"]
    frame_h = summary["frame_height"]
    masks_path = masks_dir / "masks"

    cap = cv2.VideoCapture(video_path)

    # Compatible codec
    writer = None
    for codec in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        w = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
        if w.isOpened():
            writer = w
            print(f"Using codec: {codec}")
            break
        w.release()

    print("🎨 Rendering mask overlays...")

    # Only render up to the last frame that has a mask (skip the rest)
    mask_files = sorted(masks_path.glob("*.npz"))
    last_mask_idx = int(mask_files[-1].stem) if mask_files else -1
    print(f"   Masks available up to frame {last_mask_idx}; rendering {last_mask_idx + 1} frames")

    frame_idx = 0
    while frame_idx <= last_mask_idx:
        ret, frame = cap.read()
        if not ret:
            break

        mask_file = masks_path / f"{frame_idx:05d}.npz"
        if mask_file.exists():
            data = np.load(mask_file)
            for key in data.files:
                obj_id = int(key.replace("obj_", ""))
                mask = data[key].astype(bool)
                color = OBJ_COLORS.get(obj_id, (255, 255, 255))

                # Blend color only on masked pixels (in-place, no full-frame alloc)
                frame[mask] = (frame[mask] * (1 - ALPHA) + np.array(color) * ALPHA).astype(np.uint8)

                # Draw contour outline
                mask_u8 = mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, color, 3)

                # Label at centroid
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    M = cv2.moments(largest)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        label = obj_labels.get(obj_id, f"obj{obj_id}")
                        cv2.putText(frame, label, (cx, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                        cv2.putText(frame, label, (cx, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

        writer.write(frame)
        frame_idx += 1

        if frame_idx % 30 == 0:
            print(f"   Frame {frame_idx}/{last_mask_idx + 1}")

    cap.release()
    writer.release()

    print(f"\n✅ Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--output", default="segmented.mp4")
    args = parser.parse_args()

    overlay(args.video, args.masks, args.output)
