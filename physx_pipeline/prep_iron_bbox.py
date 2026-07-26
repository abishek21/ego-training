"""
Prepare iron input as a BOUNDING-BOX crop (not tight mask).
------------------------------------------------------------
Uses the SAM2 mask only to find the iron's bounding box, then crops
the full rectangular region. This captures the COMPLETE iron even
where the hand occludes parts of it (tight masking would leave holes).

Two outputs:
  - RGB crop (bbox region, background+hand included)  → for PhysX remove_bg=True
  - RGBA crop (bbox with mask alpha)                  → for PhysX remove_bg=False

Usage:
    python prep_iron_bbox.py --video /path/ego_press.mp4 \
        --masks ../sam2_pipeline/masks_full --frame 950 --output iron_bbox.png
"""

import argparse
import cv2
import numpy as np
from pathlib import Path


def prep(video_path, masks_dir, frame_idx, output_path, pad=60):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"❌ Could not read frame {frame_idx}")
        return
    h, w = frame.shape[:2]

    mask_file = Path(masks_dir) / "masks" / f"{frame_idx:05d}.npz"
    if not mask_file.exists():
        print(f"❌ No mask at {mask_file}")
        return
    data = np.load(mask_file)
    mask = data[data.files[0]].astype(bool)

    # Bounding box from mask
    ys, xs = np.where(mask)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)

    crop = frame[y1:y2, x1:x2]
    print(f"📷 Frame {frame_idx}: bbox crop {crop.shape[1]}x{crop.shape[0]}")

    stem = Path(output_path).with_suffix("")

    # Output 1: plain RGB bbox crop (full iron, includes hand/bg)
    rgb_path = f"{stem}_rgb.png"
    cv2.imwrite(rgb_path, crop)
    print(f"   ✅ RGB bbox crop: {rgb_path}  (use with PhysX --remove_bg True)")

    # Output 2: RGBA where alpha = mask within the bbox (tight, has holes)
    crop_mask = mask[y1:y2, x1:x2]
    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = (crop_mask * 255).astype(np.uint8)
    rgba_path = f"{stem}_rgba.png"
    cv2.imwrite(rgba_path, rgba)
    print(f"   ✅ RGBA masked crop: {rgba_path}  (use with PhysX --remove_bg False)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--output", default="iron_bbox.png")
    parser.add_argument("--pad", type=int, default=60)
    args = parser.parse_args()

    prep(args.video, args.masks, args.frame, args.output, args.pad)
