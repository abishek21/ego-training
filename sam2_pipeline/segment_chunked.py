"""
Segment (CHUNKED) — memory-safe SAM 2 tracking for LONG videos.
---------------------------------------------------------------
The non-chunked segment.py OOMs on long 1600x1300 clips because SAM 2's
init_state loads ALL frames into RAM (~22GB) AND its temporal memory features
grow every frame (~25GB by frame 1640) -> ~70GB+ for a full 3592-frame clip,
which fits in NEITHER the 50GB CPU cap NOR the 46GB GPU.

Fix: process in WINDOWS of `chunk` frames. Each window runs SAM 2 on only its
frames (bounded memory), then RE-SEEDS the next window with the previous
window's LAST-frame mask (via add_new_mask) so tracking stays continuous across
boundaries. Masks stream to disk. Memory stays flat regardless of clip length.

Reuses frames already extracted by segment.py at <output>/frames (00000.jpg...).
If not present, extracts them first.

Usage (run from sam2_pipeline/ AFTER the sam2 repo was moved out of it):
    python segment_chunked.py \
        --video camera_left_2min.mp4 \
        --prompts prompts_stereo.json \
        --output masks_left_full/ \
        --chunk 500 --overlap 1
"""

import os
import cv2
import json
import shutil
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cudnn.enabled = False


def extract_frames(video_path, frames_dir, max_frames=None):
    frames_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(frames_dir.glob("*.jpg"))
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if max_frames:
        total = min(total, max_frames)
    if len(existing) >= total:
        cap.release()
        print(f"♻️  Reusing {len(existing)} already-extracted frames.")
        return len(existing), fps, fw, fh
    print(f"📹 Extracting {total} frames -> {frames_dir}...")
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and idx >= max_frames):
            break
        cv2.imwrite(str(frames_dir / f"{idx:05d}.jpg"), frame)
        idx += 1
    cap.release()
    return idx, fps, fw, fh


def build_predictor():
    from sam2.build_sam import build_sam2_video_predictor
    script_dir = Path(__file__).resolve().parent
    ckpt = str(script_dir / "checkpoints" / "sam2.1_hiera_large.pt")
    cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_sam2_video_predictor(cfg, ckpt, device=device)
    return predictor, device


def link_chunk(frames_dir, chunk_dir, start, end):
    """Symlink frames [start,end) into chunk_dir renumbered 00000.jpg..."""
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True)
    for local, gi in enumerate(range(start, end)):
        src = frames_dir / f"{gi:05d}.jpg"
        dst = chunk_dir / f"{local:05d}.jpg"
        os.symlink(src.resolve(), dst)
    return end - start


def run(video_path, prompts_path, output_dir, chunk=500, max_frames=None):
    video_path = str(Path(video_path).resolve())
    prompts_path = str(Path(prompts_path).resolve())
    output_dir = Path(output_dir).resolve()
    frames_dir = output_dir / "frames"
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = output_dir / "_chunk_frames"

    with open(prompts_path) as f:
        prompts = json.load(f)

    n_frames, fps, fw, fh = extract_frames(video_path, frames_dir, max_frames)
    print(f"🧩 Chunked run: {n_frames} frames, window={chunk}")

    predictor, device = build_predictor()
    print(f"   SAM 2 on {device}")

    obj_labels = {int(k): v["label"] for k, v in prompts["objects"].items()}
    # carry[obj_id] = last-frame bool mask (H,W) to re-seed next chunk
    carry = None
    frames_with_masks = []

    starts = list(range(0, n_frames, chunk))
    for ci, start in enumerate(starts):
        end = min(start + chunk, n_frames)
        n_local = link_chunk(frames_dir, chunk_dir, start, end)
        state = predictor.init_state(
            video_path=str(chunk_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=False,   # keep growing state on the idle GPU
        )

        if ci == 0:
            # first chunk: use the human clicked points on local frame 0
            for oid_str, od in prompts["objects"].items():
                oid = int(oid_str)
                pts = np.array(od["points"], dtype=np.float32)
                lbs = np.array(od["labels"], dtype=np.int32)
                predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=0, obj_id=oid,
                    points=pts, labels=lbs)
        else:
            # re-seed from previous chunk's last mask on local frame 0
            for oid, mask in carry.items():
                if mask is None or not mask.any():
                    continue
                predictor.add_new_mask(
                    inference_state=state, frame_idx=0, obj_id=oid,
                    mask=torch.as_tensor(mask, dtype=torch.bool))

        # propagate this chunk, stream masks to disk at GLOBAL indices
        last_masks = {}
        for lidx, obj_ids, logits in tqdm(
            predictor.propagate_in_video(state), total=n_local,
            desc=f"chunk {ci+1}/{len(starts)} [{start}:{end}]"):
            gidx = start + lidx
            stack = {}
            for i, oid in enumerate(obj_ids):
                m = (logits[i] > 0.0).squeeze().cpu().numpy()
                last_masks[oid] = m
                if m.any():
                    stack[f"obj_{oid}"] = m.astype(np.uint8)
            if stack:
                np.savez_compressed(masks_dir / f"{gidx:05d}.npz", **stack)
                frames_with_masks.append(int(gidx))
            del logits

        carry = last_masks
        # free everything before next chunk
        predictor.reset_state(state)
        del state
        import gc; gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)

    summary = {
        "video": video_path, "n_frames": n_frames, "fps": fps,
        "frame_width": fw, "frame_height": fh, "objects": obj_labels,
        "frames_with_masks": sorted(frames_with_masks),
        "chunked": True, "chunk_size": chunk,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("✅ CHUNKED SEGMENTATION COMPLETE")
    print("=" * 60)
    print(f"   masks -> {masks_dir}")
    print(f"   frames with masks: {len(frames_with_masks)}/{n_frames}")
    print(f"   objects: {obj_labels}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--output", default="masks_output")
    p.add_argument("--chunk", type=int, default=500, help="frames per window")
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args()
    run(args.video, args.prompts, args.output, args.chunk, args.max_frames)
