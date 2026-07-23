"""
Step 1: INFERENCE — Run MediaPipe + Activity Recognition → Save to JSON
------------------------------------------------------------------------
Runs all the heavy computation once and saves structured predictions.
You only need to run this once per video.

Output: output/predictions.json

Usage:
    python infer.py
"""

import cv2
import json
import mediapipe as mp
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import deque
from typing import Optional, List, Tuple

# ─── Config ────────────────────────────────────────────────────────────────────
INPUT_VIDEO = "/Users/abishek/Downloads/ego_press.mp4"
OUTPUT_JSON = "/Users/abishek/ego_training/output/full_video/predictions.json"
MODEL_PATH = "/Users/abishek/ego_training/models/hand_landmarker.task"

# Processing
MAX_FRAMES = None  # None = process ALL frames

# MediaPipe
MAX_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Activity recognition thresholds
IRON_GRIP_CURL_THRESHOLD = 0.25   # Middle/Ring/Pinky curl for iron handle grip (lowered based on data)
IRON_GRIP_SPREAD_THRESHOLD = 0.07 # Thumb-Index must be spread apart (V-shape)
PINCH_DISTANCE_THRESHOLD = 0.06   # Thumb-Index close = pinch
VELOCITY_LOW = 8.0                # Below = static
WRIST_HISTORY_SIZE = 10
# ────────────────────────────────────────────────────────────────────────────────


# ─── Activity Classifier (Simplified) ──────────────────────────────────────────

class ActivityClassifier:
    """
    Simplified activity classification tuned for ironing.

    Actions:
      HOLDING IRON  → middle+ring+pinky curled, thumb+index spread (V-shape)
      PRESSING      → holding iron + low wrist velocity (actively ironing)
      PINCH         → thumb tip and index tip close together
      OPEN          → hand not gripping anything
    """

    FINGERS = {
        "thumb": [1, 2, 3, 4],
        "index": [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring": [13, 14, 15, 16],
        "pinky": [17, 18, 19, 20],
    }

    def __init__(self):
        self.wrist_history: dict = {}  # hand_label → deque of (x, y)

    def compute_finger_curl(self, landmarks, finger_indices) -> float:
        """0 = straight, 1 = fully curled."""
        mcp = np.array([landmarks[finger_indices[0]].x, landmarks[finger_indices[0]].y])
        pip = np.array([landmarks[finger_indices[1]].x, landmarks[finger_indices[1]].y])
        dip = np.array([landmarks[finger_indices[2]].x, landmarks[finger_indices[2]].y])
        tip = np.array([landmarks[finger_indices[3]].x, landmarks[finger_indices[3]].y])

        bone_length = (np.linalg.norm(pip - mcp) +
                       np.linalg.norm(dip - pip) +
                       np.linalg.norm(tip - dip))
        direct_dist = np.linalg.norm(tip - mcp)

        if bone_length < 1e-6:
            return 0.0
        return max(0.0, min(1.0, 1.0 - (direct_dist / bone_length)))

    def compute_thumb_index_distance(self, landmarks) -> float:
        """Normalized distance between thumb tip and index tip."""
        thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
        index_tip = np.array([landmarks[8].x, landmarks[8].y])
        return float(np.linalg.norm(thumb_tip - index_tip))

    def compute_wrist_velocity(self, hand_label: str, wrist_pos, frame_w, frame_h) -> float:
        """Track wrist and return smoothed velocity in px/frame."""
        if hand_label not in self.wrist_history:
            self.wrist_history[hand_label] = deque(maxlen=WRIST_HISTORY_SIZE)

        px = (wrist_pos.x * frame_w, wrist_pos.y * frame_h)
        history = self.wrist_history[hand_label]
        history.append(px)

        if len(history) < 3:
            return 0.0

        velocities = []
        for i in range(1, len(history)):
            dx = history[i][0] - history[i - 1][0]
            dy = history[i][1] - history[i - 1][1]
            velocities.append(np.sqrt(dx * dx + dy * dy))

        return float(np.mean(velocities[-5:]) if len(velocities) >= 5 else np.mean(velocities))

    def classify(self, landmarks, hand_label: str, frame_w: int, frame_h: int) -> dict:
        """Classify into one of: holding_iron, pressing, pinch, open."""

        # Compute per-finger curls
        curls = {}
        for finger_name, indices in self.FINGERS.items():
            curls[finger_name] = self.compute_finger_curl(landmarks, indices)

        curl_values = [curls["thumb"], curls["index"], curls["middle"], curls["ring"], curls["pinky"]]

        # Key measurements
        thumb_index_dist = self.compute_thumb_index_distance(landmarks)
        grip_curls = [curls["middle"], curls["ring"], curls["pinky"]]  # The 3 wrap fingers
        avg_grip_curl = np.mean(grip_curls)
        velocity = self.compute_wrist_velocity(hand_label, landmarks[0], frame_w, frame_h)

        # ─── Classification Rules (priority order) ─────────────

        # Key measurements for iron grip
        is_grip_curled = avg_grip_curl > IRON_GRIP_CURL_THRESHOLD
        is_thumb_index_spread = thumb_index_dist > IRON_GRIP_SPREAD_THRESHOLD

        # HOLDING IRON (check FIRST — higher priority than pinch)
        # When gripping iron handle: mid+ring+pinky wrap around,
        # AND either thumb-index spread (V-shape) OR close (wrapped around handle)
        # The key differentiator from pure pinch: the 3 grip fingers are curled
        if is_grip_curled:
            if velocity < VELOCITY_LOW:
                action = "pressing"
                confidence = float(avg_grip_curl)
            else:
                action = "holding_iron"
                confidence = float(avg_grip_curl)

        # PINCH: thumb + index tips close, BUT grip fingers NOT curled
        # Pure pinch = adjusting fabric with fingertips, other fingers relaxed
        elif thumb_index_dist < PINCH_DISTANCE_THRESHOLD:
            action = "pinch"
            confidence = float(1.0 - (thumb_index_dist / PINCH_DISTANCE_THRESHOLD))

        # OPEN: everything else
        else:
            action = "open"
            confidence = 0.5

        return {
            "action": action,
            "confidence": round(confidence, 3),
            "finger_curls": [round(c, 4) for c in curl_values],
            "thumb_index_dist": round(thumb_index_dist, 4),
            "wrist_velocity": round(velocity, 2),
        }


# ─── Contact Event Tracker ──────────────────────────────────────────────────────

class ContactEventTracker:
    """Tracks action transitions: open↔holding_iron, open↔pinch."""
    def __init__(self):
        self.prev_action: dict = {}
        self.cooldown: dict = {}

    def update(self, hand_label: str, action: str, frame_num: int) -> Optional[dict]:
        if hand_label in self.cooldown:
            self.cooldown[hand_label] -= 1
            if self.cooldown[hand_label] > 0:
                return None

        prev = self.prev_action.get(hand_label, "unknown")
        curr = action
        event = None

        if prev != curr and prev != "unknown":
            # Picked up iron or pinched fabric
            if prev == "open" and curr in ("holding_iron", "pressing", "pinch"):
                event = {
                    "frame": frame_num, "hand": hand_label,
                    "event_type": "grasp_initiated",
                    "from": prev, "to": curr,
                }
            # Released iron or fabric
            elif prev in ("holding_iron", "pressing", "pinch") and curr == "open":
                event = {
                    "frame": frame_num, "hand": hand_label,
                    "event_type": "release",
                    "from": prev, "to": curr,
                }
            # Started pressing (was just holding)
            elif prev == "holding_iron" and curr == "pressing":
                event = {
                    "frame": frame_num, "hand": hand_label,
                    "event_type": "pressing_start",
                    "from": prev, "to": curr,
                }
            # Lifted iron (stopped pressing)
            elif prev == "pressing" and curr == "holding_iron":
                event = {
                    "frame": frame_num, "hand": hand_label,
                    "event_type": "pressing_end",
                    "from": prev, "to": curr,
                }

            if event:
                self.cooldown[hand_label] = 15

        self.prev_action[hand_label] = curr
        return event


# ─── Main Inference Pipeline ────────────────────────────────────────────────────

def run_inference():
    """Run all inference and save predictions to JSON."""

    input_path = Path(INPUT_VIDEO)
    output_path = Path(OUTPUT_JSON)
    model_path = Path(MODEL_PATH)

    if not input_path.exists():
        print(f"❌ Input video not found: {INPUT_VIDEO}")
        return
    if not model_path.exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

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
    print(f"   Duration: {total_frames/fps:.1f}s")
    print()

    # MediaPipe setup
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    landmarker = HandLandmarker.create_from_options(options)

    # Activity modules
    classifier = ActivityClassifier()
    contact_tracker = ContactEventTracker()

    # Output structure
    predictions = {
        "metadata": {
            "input_video": str(input_path),
            "frame_width": frame_w,
            "frame_height": frame_h,
            "fps": fps,
            "total_frames": total_frames,
            "frames_processed": 0,
        },
        "frames": [],
        "contact_events": [],
    }

    print("🚀 Running inference...")
    print("   Actions: HOLDING_IRON | PRESSING | PINCH | OPEN")
    print()
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        if MAX_FRAMES and frame_num > MAX_FRAMES:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((frame_num / fps) * 1000)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        frame_data = {
            "frame_num": frame_num,
            "timestamp_ms": timestamp_ms,
            "hands": [],
        }

        if results.hand_landmarks:
            for hand_landmarks, hand_info in zip(
                results.hand_landmarks, results.handedness
            ):
                hand_label = hand_info[0].category_name

                # Keypoints (normalized 0-1)
                keypoints = []
                for lm in hand_landmarks:
                    keypoints.append({
                        "x": round(lm.x, 5),
                        "y": round(lm.y, 5),
                        "z": round(lm.z, 5),
                    })

                # Activity classification (single unified label)
                activity = classifier.classify(hand_landmarks, hand_label, frame_w, frame_h)

                # Contact events
                event = contact_tracker.update(hand_label, activity["action"], frame_num)
                if event:
                    predictions["contact_events"].append(event)
                    print(f"   ⚡ Frame {frame_num}: {event['hand']} — {event['event_type']} "
                          f"({event['from']} → {event['to']})")

                hand_data = {
                    "label": hand_label,
                    "keypoints": keypoints,
                    "activity": activity,
                }
                frame_data["hands"].append(hand_data)

        predictions["frames"].append(frame_data)

        if frame_num % 60 == 0:
            pct = (frame_num / (MAX_FRAMES or total_frames)) * 100
            print(f"   [{pct:5.1f}%] Frame {frame_num}")

    cap.release()
    landmarker.close()

    predictions["metadata"]["frames_processed"] = frame_num

    # Save to JSON
    with open(str(output_path), "w") as f:
        json.dump(predictions, f, indent=2)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print()
    print("=" * 60)
    print("✅ INFERENCE COMPLETE")
    print("=" * 60)
    print(f"   Saved to: {output_path}")
    print(f"   File size: {file_size_mb:.1f} MB")
    print(f"   Frames: {frame_num}")
    print(f"   Contact events: {len(predictions['contact_events'])}")
    print()
    print("   Next step: python render.py")


if __name__ == "__main__":
    run_inference()
