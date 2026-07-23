"""
Object Detection Inference — Florence-2-Large
------------------------------------------------
Detects objects (iron box, gray sheet, black cloth) in ego-centric video frames.
Saves detections to JSON alongside hand keypoint predictions.

Run once, then use render.py to visualize.

Usage:
    python infer_objects.py
"""

import cv2
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# ─── Config ────────────────────────────────────────────────────────────────────
INPUT_VIDEO = "/Users/abishek/Downloads/ego_press.mp4"
OUTPUT_JSON = "/Users/abishek/ego_training/output/full_video/object_detections.json"

# Object prompts for Florence-2 grounding
TEXT_PROMPTS = "iron box . gray sheet . black cloth"

# Detection settings
CONFIDENCE_THRESHOLD = 0.25
FRAME_SKIP = 3  # Process every Nth frame (60fps → 20fps effective)
MAX_FRAMES = None  # Set to e.g. 600 for testing, None for full video

# Model
MODEL_ID = "microsoft/Florence-2-large"
# ────────────────────────────────────────────────────────────────────────────────


def run_inference():
    """Run Florence-2 object detection on video frames."""

    input_path = Path(INPUT_VIDEO)
    output_path = Path(OUTPUT_JSON)

    if not input_path.exists():
        print(f"❌ Input video not found: {INPUT_VIDEO}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"🔧 Loading Florence-2-Large ({MODEL_ID})...")
    print("   This may take a minute on first run (downloading ~1.5GB weights)...")

    from transformers import AutoProcessor, AutoModelForCausalLM

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Use MPS if available, otherwise CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    print(f"   Model loaded on: {device}")
    print(f"   Prompts: {TEXT_PROMPTS}")
    print(f"   Frame skip: every {FRAME_SKIP} frame(s)")
    print()

    # Open video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"❌ Cannot open video: {INPUT_VIDEO}")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📹 Input: {input_path.name}")
    print(f"   Resolution: {frame_w}x{frame_h} | FPS: {fps:.1f} | Frames: {total_frames}")
    print(f"   Will process: ~{total_frames // FRAME_SKIP} frames")
    print()

    # Output structure
    predictions = {
        "metadata": {
            "input_video": str(input_path),
            "frame_width": frame_w,
            "frame_height": frame_h,
            "fps": fps,
            "total_frames": total_frames,
            "model": MODEL_ID,
            "prompts": TEXT_PROMPTS,
            "frame_skip": FRAME_SKIP,
        },
        "frames": [],
    }

    print("🚀 Running object detection...")
    frame_num = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        if MAX_FRAMES and frame_num > MAX_FRAMES:
            break

        # Skip frames for speed
        if frame_num % FRAME_SKIP != 0:
            continue

        processed += 1

        # Convert BGR → RGB → PIL Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Florence-2 uses <OPEN_VOCABULARY_DETECTION> task with text prompt
        task = "<OPEN_VOCABULARY_DETECTION>"
        prompt = task + TEXT_PROMPTS

        inputs = processor(text=prompt, images=pil_image, return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )

        # Decode output
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        result = processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(frame_w, frame_h),
        )

        # Extract detections
        detections = []
        if task in result:
            det_result = result[task]
            bboxes = det_result.get("bboxes", [])
            labels = det_result.get("bboxes_labels", [])

            for bbox, label in zip(bboxes, labels):
                x1, y1, x2, y2 = bbox
                detections.append({
                    "label": label.strip(),
                    "confidence": 1.0,  # Florence-2 doesn't output per-box scores
                    "bbox": {
                        "x1": round(x1, 1),
                        "y1": round(y1, 1),
                        "x2": round(x2, 1),
                        "y2": round(y2, 1),
                    },
                    "bbox_normalized": {
                        "x1": round(x1 / frame_w, 4),
                        "y1": round(y1 / frame_h, 4),
                        "x2": round(x2 / frame_w, 4),
                        "y2": round(y2 / frame_h, 4),
                    },
                })

        frame_data = {
            "frame_num": frame_num,
            "timestamp_ms": int((frame_num / fps) * 1000),
            "objects": detections,
        }
        predictions["frames"].append(frame_data)

        # Progress
        if processed % 20 == 0:
            pct = (frame_num / (MAX_FRAMES or total_frames)) * 100
            n_obj = len(detections)
            print(f"   [{pct:5.1f}%] Frame {frame_num} — {n_obj} objects detected")

    cap.release()

    predictions["metadata"]["frames_processed"] = processed

    # Save to JSON
    with open(str(output_path), "w") as f:
        json.dump(predictions, f, indent=2)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print()
    print("=" * 60)
    print("✅ OBJECT DETECTION COMPLETE")
    print("=" * 60)
    print(f"   Saved to: {output_path}")
    print(f"   File size: {file_size_mb:.1f} MB")
    print(f"   Frames processed: {processed}")
    print(f"   Total detections: {sum(len(f['objects']) for f in predictions['frames'])}")
    print()

    # Quick distribution
    label_counts = {}
    for frame_data in predictions["frames"]:
        for obj in frame_data["objects"]:
            label_counts[obj["label"]] = label_counts.get(obj["label"], 0) + 1
    print("   📊 Detection distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"      {label:20s}: {count}")
    print()


if __name__ == "__main__":
    run_inference()
