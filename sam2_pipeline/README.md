# SAM 2 Object Segmentation Pipeline (GPU)

Click a few points on frame 1 → SAM 2 tracks objects across the whole video.

## Recommended GPU
- **24GB VRAM** (RTX 4090 / A10 / L4) — comfortable for 4K video with SAM 2 Large
- CUDA 12.1+

## Workflow

```
Step 1 (local or GPU):  Extract frame 1, click points on objects
Step 2 (GPU):           SAM 2 propagates masks across all frames
Step 3 (local):         Download masks, overlay on video
```

## Setup on RunPod

```bash
bash start.sh
```

This installs SAM 2, downloads the large checkpoint, and sets up the environment.

## Files

| File | Purpose |
|------|---------|
| `start.sh` | One-shot setup on GPU machine |
| `annotate.py` | Click UI — select points on frame 1, saves prompts.json |
| `segment.py` | Runs SAM 2 tracking using prompts.json → saves masks |
| `overlay.py` | Renders masks onto video |

## Usage

```bash
# 1. Annotate (can run locally with display, or use pre-saved coords)
python annotate.py --video /path/to/ego_press.mp4 --output prompts.json

# 2. Segment (on GPU)
python segment.py --video /path/to/ego_press.mp4 --prompts prompts.json --output masks/

# 3. Overlay (local)
python overlay.py --video /path/to/ego_press.mp4 --masks masks/ --output segmented.mp4
```
