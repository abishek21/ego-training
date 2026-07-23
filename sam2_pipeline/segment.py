"""
Segment — Run SAM 2 video tracking using clicked prompts.
----------------------------------------------------------
Takes prompts.json (from annotate.py) and propagates object masks
across all frames of the video. Runs on GPU.

Saves per-frame masks as compressed .npz + a summary JSON.

Usage:
    python segment.py --video ego_press.mp4 --prompts prompts.json --output masks/
"""

import os
import cv2
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Workaround for cuDNN init errors on some driver/torch combos.
# Falls back to non-cuDNN CUDA kernels (slightly slower but reliable).
torch.backends.cudnn.enabled = False


def extract_frames(video_path: str, frames_dir: Path, max_frames: int = None):
    """SAM 2 video predictor needs frames as JPEG files in a directory."""
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if max_frames:
        total = min(total, max_frames)

    print(f"📹 Extracting {total} frames to {frames_dir}...")
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and idx >= max_frames:
            break
        # SAM 2 expects frames named as %05d.jpg
        cv2.imwrite(str(frames_dir / f"{idx:05d}.jpg"), frame)
        idx += 1
    cap.release()

    print(f"   Extracted {idx} frames ({frame_w}x{frame_h} @ {fps:.1f}fps)")
    return idx, fps, frame_w, frame_h


def run_segmentation(video_path, prompts_path, output_dir, max_frames=None):
    # Resolve to absolute paths (script may run from a different cwd)
    video_path = str(Path(video_path).resolve())
    prompts_path = str(Path(prompts_path).resolve())
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"

    # Load prompts
    with open(prompts_path) as f:
        prompts = json.load(f)

    # Extract frames
    n_frames, fps, frame_w, frame_h = extract_frames(video_path, frames_dir, max_frames)

    # Import SAM 2
    print("\n🔧 Loading SAM 2...")
    from sam2.build_sam import build_sam2_video_predictor

    # Absolute paths so this works regardless of working directory
    script_dir = Path(__file__).resolve().parent
    checkpoint = str(script_dir / "checkpoints" / "sam2.1_hiera_large.pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"  # resolved by hydra within sam2 pkg

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    print(f"   SAM 2 loaded on: {device}")

    # Initialize inference state on the extracted frames
    print("\n🚀 Initializing video predictor...")
    inference_state = predictor.init_state(video_path=str(frames_dir))

    # Add clicked points for each object on the prompt frame
    prompt_frame = prompts["frame_idx"]
    obj_labels = {}  # obj_id -> label name

    print(f"\n📍 Adding prompts on frame {prompt_frame}...")
    for obj_id_str, obj_data in prompts["objects"].items():
        obj_id = int(obj_id_str)
        obj_labels[obj_id] = obj_data["label"]

        points = np.array(obj_data["points"], dtype=np.float32)
        labels = np.array(obj_data["labels"], dtype=np.int32)

        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=prompt_frame,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )
        print(f"   Object {obj_id} ({obj_data['label']}): {len(points)} points")

    # Propagate through the whole video
    print("\n🎬 Propagating masks across all frames...")
    video_segments = {}  # frame_idx -> {obj_id: mask}

    for out_frame_idx, out_obj_ids, out_mask_logits in tqdm(
        predictor.propagate_in_video(inference_state), total=n_frames
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # Save masks — compressed npz per frame (only non-empty)
    print("\n💾 Saving masks...")
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(exist_ok=True)

    summary = {
        "video": video_path,
        "n_frames": n_frames,
        "fps": fps,
        "frame_width": frame_w,
        "frame_height": frame_h,
        "objects": obj_labels,
        "frames_with_masks": [],
    }

    for frame_idx, seg in tqdm(video_segments.items(), total=len(video_segments)):
        # Stack masks: shape (n_objects, H, W)
        obj_ids = sorted(seg.keys())
        if not obj_ids:
            continue

        mask_stack = {}
        for obj_id in obj_ids:
            mask = seg[obj_id].squeeze()  # (H, W) bool
            if mask.any():
                mask_stack[f"obj_{obj_id}"] = mask.astype(np.uint8)

        if mask_stack:
            np.savez_compressed(masks_dir / f"{frame_idx:05d}.npz", **mask_stack)
            summary["frames_with_masks"].append(frame_idx)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 60)
    print("✅ SEGMENTATION COMPLETE")
    print("=" * 60)
    print(f"   Masks saved to: {masks_dir}")
    print(f"   Frames with masks: {len(summary['frames_with_masks'])}/{n_frames}")
    print(f"   Objects tracked: {obj_labels}")
    print()
    print("   Next: python overlay.py --video <video> --masks <output_dir> --output segmented.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", default="masks_output")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit number of frames (for testing)")
    args = parser.parse_args()

    run_segmentation(args.video, args.prompts, args.output, args.max_frames)
