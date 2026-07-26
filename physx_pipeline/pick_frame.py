"""
Scrub the video with the iron mask overlaid, pick the best frame.
------------------------------------------------------------------
Plays video with SAM2 iron mask. Pause and step to find the cleanest
view of the iron, then export that frame as an RGBA cutout for PhysX.

Controls:
    SPACE      = play / pause
    d / →      = next frame (when paused)
    a / ←      = previous frame (when paused)
    s          = SAVE current frame as RGBA iron cutout
    q / ESC    = quit

Usage:
    python pick_frame.py --video /path/ego_press.mp4 \
        --masks ../sam2_pipeline/masks_full --output iron_selected.png
"""

import argparse
import cv2
import numpy as np
from pathlib import Path


def load_mask(masks_dir, frame_idx):
    mask_file = Path(masks_dir) / "masks" / f"{frame_idx:05d}.npz"
    if not mask_file.exists():
        return None
    data = np.load(mask_file)
    return data[data.files[0]].astype(bool)


def save_cutout(frame, mask, frame_idx, output_path, pad=40):
    h, w = frame.shape[:2]
    ys, xs = np.where(mask)
    if len(xs) == 0:
        print("   ⚠️  Empty mask, cannot save")
        return
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)

    crop = frame[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]

    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = (crop_mask * 255).astype(np.uint8)

    out = Path(output_path)
    stem = out.with_suffix("")
    final = f"{stem}_f{frame_idx}.png"
    cv2.imwrite(final, rgba)

    preview = crop.copy()
    preview[~crop_mask] = (255, 255, 255)
    cv2.imwrite(f"{stem}_f{frame_idx}_preview.png", preview)

    print(f"   ✅ Saved: {final} ({crop.shape[1]}x{crop.shape[0]})")
    print(f"      Preview: {stem}_f{frame_idx}_preview.png")


def pick(video_path, masks_dir, output_path, display_w=1280):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = display_w / w
    disp_h = int(h * scale)

    print("=" * 60)
    print("FRAME PICKER")
    print("=" * 60)
    print("SPACE = play/pause | d/→ = next | a/← = prev")
    print("s = save cutout | q/ESC = quit")
    print(f"Total frames: {total}")
    print("=" * 60)

    frame_idx = 0
    paused = True

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            frame_idx = max(0, frame_idx - 1)
            continue

        mask = load_mask(masks_dir, frame_idx)

        disp = frame.copy()
        if mask is not None:
            # Green tinted overlay + contour
            overlay = disp.copy()
            overlay[mask] = (0, 255, 0)
            disp = cv2.addWeighted(disp, 0.7, overlay, 0.3, 0)
            mu8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mu8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(disp, contours, -1, (0, 255, 0), 3)
            area = int(mask.sum())
        else:
            area = 0

        disp = cv2.resize(disp, (display_w, disp_h))
        status = "PAUSED" if paused else "PLAYING"
        cv2.putText(disp, f"Frame {frame_idx}/{total}  [{status}]  mask_px={area}",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(disp, "s=save  space=play/pause  a/d=step  q=quit",
                    (15, disp_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Pick Frame (iron mask)", disp)

        key = cv2.waitKey(0 if paused else 30) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
        elif key in (ord('d'), 83):   # next
            frame_idx = min(total - 1, frame_idx + 1)
        elif key in (ord('a'), 81):   # prev
            frame_idx = max(0, frame_idx - 1)
        elif key == ord('s'):
            if mask is not None:
                print(f"💾 Saving frame {frame_idx}...")
                save_cutout(frame, mask, frame_idx, output_path)
            else:
                print(f"   ⚠️  No mask for frame {frame_idx}")
        else:
            if not paused:
                frame_idx = min(total - 1, frame_idx + 1)
                if frame_idx == total - 1:
                    paused = True

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--output", default="iron_selected.png")
    args = parser.parse_args()

    pick(args.video, args.masks, args.output)
