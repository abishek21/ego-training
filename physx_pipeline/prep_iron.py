"""
Prepare iron input for PhysX-Anything.
---------------------------------------
1. Extract a chosen frame from the video where the iron is clearly visible
2. Use the SAM2 mask for that frame to cut out the iron
3. Produce an RGBA PNG (transparent background) — PhysX input format

If the SAM2 mask for the chosen frame exists, we use it.
Otherwise you can pass --frame to just extract the raw frame for manual masking.

Usage:
    python prep_iron.py --video /path/ego_press.mp4 \
        --masks ../sam2_pipeline/masks_full \
        --frame 250 --output iron_rgba.png
"""

import argparse
import cv2
import numpy as np
from pathlib import Path


def prep(video_path, masks_dir, frame_idx, output_path, pad=40):
    # ── Extract the frame ──
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"❌ Could not read frame {frame_idx}")
        return
    h, w = frame.shape[:2]
    print(f"📷 Frame {frame_idx}: {w}x{h}")

    # ── Load the SAM2 mask for this frame ──
    mask_file = Path(masks_dir) / "masks" / f"{frame_idx:05d}.npz"
    if not mask_file.exists():
        print(f"⚠️  No mask at {mask_file}")
        print("   Saving raw frame only (mask it manually or pick a frame with a mask).")
        raw_path = str(Path(output_path).with_suffix(".raw.png"))
        cv2.imwrite(raw_path, frame)
        print(f"   Saved raw frame: {raw_path}")
        return

    data = np.load(mask_file)
    # Use the first object (the iron)
    key = data.files[0]
    mask = data[key].astype(bool)  # (H, W)
    print(f"   Using mask '{key}' — {mask.sum()} pixels")

    # ── Crop to bounding box (with padding) ──
    ys, xs = np.where(mask)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)

    crop = frame[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]

    # ── Build RGBA (transparent background) ──
    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = (crop_mask * 255).astype(np.uint8)  # alpha from mask

    cv2.imwrite(output_path, rgba)
    print(f"✅ Saved RGBA cutout: {output_path} ({crop.shape[1]}x{crop.shape[0]})")

    # Also save a white-background preview (easier to eyeball)
    preview = crop.copy()
    preview[~crop_mask] = (255, 255, 255)
    prev_path = str(Path(output_path).with_suffix(".preview.png"))
    cv2.imwrite(prev_path, preview)
    print(f"   Preview (white bg): {prev_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--masks", required=True, help="SAM2 masks dir (has masks/ subdir)")
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--output", default="iron_rgba.png")
    parser.add_argument("--pad", type=int, default=40)
    args = parser.parse_args()

    prep(args.video, args.masks, args.frame, args.output, args.pad)
