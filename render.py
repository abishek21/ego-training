"""
Step 2: RENDER — Read predictions.json + draw on video
-------------------------------------------------------
No inference needed! Reads saved predictions and renders overlays.
Classification rules live HERE so you can tweak without re-running inference.

Usage:
    python render.py
"""

import cv2
import json
import numpy as np
from pathlib import Path
from collections import deque
from typing import Tuple

# ─── Config ────────────────────────────────────────────────────────────────────
INPUT_VIDEO = "/Users/abishek/Downloads/ego_press.mp4"
PREDICTIONS_JSON = "/Users/abishek/ego_training/output/full_video/predictions.json"
OUTPUT_VIDEO = "/Users/abishek/ego_training/output/full_video/ego_press_activity.mp4"

# ─── Classification Rules (TWEAK THESE — no re-inference needed!) ─────────────
IRON_GRIP_CURL_THRESHOLD = 0.25   # Mid+Ring+Pinky avg curl to count as gripping
PINCH_DISTANCE_THRESHOLD = 0.06   # Thumb-Index distance for pinch
VELOCITY_LOW = 8.0                # Below this = static (pressing vs holding)

# Per-hand rules: Right hand holds iron, Left hand pinches fabric
# When both hands show similar curl patterns, this disambiguates
RIGHT_HAND_CAN_HOLD_IRON = True
LEFT_HAND_CAN_HOLD_IRON = False   # Left hand grip → classify as pinch instead
# ──────────────────────────────────────────────────────────────────────────────

# ─── Visualization Settings (TWEAK THESE FREELY) ──────────────────────────────
SKELETON_COLOR = (255, 255, 255)
THUMB_COLOR = (0, 165, 255)
INDEX_COLOR = (255, 0, 128)
MIDDLE_COLOR = (255, 255, 0)
RING_COLOR = (0, 255, 255)
PINKY_COLOR = (128, 0, 255)
WRIST_COLOR = (0, 200, 0)
KEYPOINT_RADIUS = 5
SKELETON_THICKNESS = 2

ACTION_COLORS = {
    "holding_iron": (0, 140, 255),    # Orange
    "pressing": (0, 0, 255),          # Red
    "pinch": (255, 0, 255),           # Magenta
    "open": (0, 255, 0),              # Green
}

EVENT_COLORS = {
    "grasp_initiated": (0, 200, 255),
    "release": (0, 255, 0),
    "pressing_start": (0, 100, 255),
    "pressing_end": (0, 255, 255),
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

FINGER_COLORS = [THUMB_COLOR, INDEX_COLOR, MIDDLE_COLOR, RING_COLOR, PINKY_COLOR]
FINGER_NAMES = ["Thumb", "Index", "Mid", "Ring", "Pinky"]
# ────────────────────────────────────────────────────────────────────────────────


def get_landmark_color(idx: int) -> tuple:
    if idx == 0: return WRIST_COLOR
    elif 1 <= idx <= 4: return THUMB_COLOR
    elif 5 <= idx <= 8: return INDEX_COLOR
    elif 9 <= idx <= 12: return MIDDLE_COLOR
    elif 13 <= idx <= 16: return RING_COLOR
    elif 17 <= idx <= 20: return PINKY_COLOR
    return WRIST_COLOR


# ─── Classification from raw measurements (runs at render time) ────────────────

def classify_action(activity: dict, hand_label: str) -> dict:
    """
    Re-classify action from raw measurements stored in predictions.json.
    Edit the rules here and just re-run render.py — no re-inference needed!
    """
    curls = activity["finger_curls"]  # [thumb, index, middle, ring, pinky]
    ti_dist = activity["thumb_index_dist"]
    velocity = activity["wrist_velocity"]

    grip_curls = [curls[2], curls[3], curls[4]]  # middle, ring, pinky
    avg_grip_curl = sum(grip_curls) / 3.0

    is_grip_curled = avg_grip_curl > IRON_GRIP_CURL_THRESHOLD

    # ─── Per-hand rules ───
    can_hold_iron = (RIGHT_HAND_CAN_HOLD_IRON if hand_label == "Right"
                     else LEFT_HAND_CAN_HOLD_IRON)

    # HOLDING IRON / PRESSING (only for the iron hand)
    if is_grip_curled and can_hold_iron:
        if velocity < VELOCITY_LOW:
            return {"action": "pressing", "confidence": round(avg_grip_curl, 3)}
        else:
            return {"action": "holding_iron", "confidence": round(avg_grip_curl, 3)}

    # PINCH: thumb + index close (or left hand with grip pattern = also pinch)
    if ti_dist < PINCH_DISTANCE_THRESHOLD or (is_grip_curled and not can_hold_iron):
        conf = 1.0 - (ti_dist / PINCH_DISTANCE_THRESHOLD) if ti_dist < PINCH_DISTANCE_THRESHOLD else avg_grip_curl
        return {"action": "pinch", "confidence": round(max(0, min(1, conf)), 3)}

    # OPEN: everything else
    return {"action": "open", "confidence": 0.5}


def classify_contact_events(frames_data: list) -> list:
    """Re-compute contact events from classified actions."""
    prev_action = {}
    cooldown = {}
    events = []

    for frame_pred in frames_data:
        frame_num = frame_pred["frame_num"]
        for hand in frame_pred["hands"]:
            hand_label = hand["label"]
            action = hand["_classified"]["action"]

            if hand_label in cooldown:
                cooldown[hand_label] -= 1
                if cooldown[hand_label] > 0:
                    continue

            prev = prev_action.get(hand_label, "unknown")
            if prev != action and prev != "unknown":
                event = None
                if prev == "open" and action in ("holding_iron", "pressing", "pinch"):
                    event = {"frame": frame_num, "hand": hand_label,
                             "event_type": "grasp_initiated", "from": prev, "to": action}
                elif prev in ("holding_iron", "pressing", "pinch") and action == "open":
                    event = {"frame": frame_num, "hand": hand_label,
                             "event_type": "release", "from": prev, "to": action}
                elif prev == "holding_iron" and action == "pressing":
                    event = {"frame": frame_num, "hand": hand_label,
                             "event_type": "pressing_start", "from": prev, "to": action}
                elif prev == "pressing" and action == "holding_iron":
                    event = {"frame": frame_num, "hand": hand_label,
                             "event_type": "pressing_end", "from": prev, "to": action}
                if event:
                    events.append(event)
                    cooldown[hand_label] = 15

            prev_action[hand_label] = action

    return events


def draw_hand_keypoints(frame, keypoints, frame_w, frame_h):
    """Draw hand skeleton + keypoints from saved predictions."""
    points = []
    for kp in keypoints:
        points.append((int(kp["x"] * frame_w), int(kp["y"] * frame_h)))

    # Skeleton
    for s, e in HAND_CONNECTIONS:
        if s < len(points) and e < len(points):
            cv2.line(frame, points[s], points[e], SKELETON_COLOR, SKELETON_THICKNESS)

    # Keypoints
    for idx, (px, py) in enumerate(points):
        color = get_landmark_color(idx)
        radius = KEYPOINT_RADIUS + 3 if idx in [4, 8, 12, 16, 20] else KEYPOINT_RADIUS
        cv2.circle(frame, (px, py), radius, color, -1)
        cv2.circle(frame, (px, py), radius, (0, 0, 0), 1)

    return frame


def draw_activity_panel(frame, hand_label: str, activity: dict,
                        frame_w: int, frame_h: int, panel_side: str = "left"):
    """Draw simplified activity panel for a hand."""
    h, w = frame.shape[:2]

    panel_w = 560
    panel_h = 280

    if panel_side == "left":
        panel_x = 20
    else:
        panel_x = w - panel_w - 20

    panel_y = h - panel_h - 20

    action = activity["action"]
    action_color = ACTION_COLORS.get(action, (200, 200, 200))

    # Semi-transparent background with action-colored border
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), action_color, 3)

    # Hand label
    cv2.putText(frame, f"{hand_label} Hand", (panel_x + 20, panel_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # ─── ACTION (big, prominent) ───
    action_label = action.replace("_", " ").upper()
    cv2.circle(frame, (panel_x + 25, panel_y + 85), 14, action_color, -1)
    cv2.putText(frame, action_label, (panel_x + 55, panel_y + 95),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, action_color, 3)
    cv2.putText(frame, f"{activity['confidence']:.0%}", (panel_x + 440, panel_y + 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

    # ─── Velocity bar ───
    velocity = activity["wrist_velocity"]
    vel_bar_max = 350
    vel_bar_width = min(int(velocity * 5), vel_bar_max)
    cv2.putText(frame, "Velocity:", (panel_x + 20, panel_y + 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.rectangle(frame, (panel_x + 140, panel_y + 120), (panel_x + 140 + vel_bar_max, panel_y + 140),
                  (50, 50, 50), -1)
    cv2.rectangle(frame, (panel_x + 140, panel_y + 120), (panel_x + 140 + vel_bar_width, panel_y + 140),
                  action_color, -1)
    cv2.putText(frame, f"{velocity:.1f} px/f", (panel_x + 140, panel_y + 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # ─── Thumb-Index distance ───
    ti_dist = activity["thumb_index_dist"]
    cv2.putText(frame, f"Thumb-Index dist: {ti_dist:.3f}", (panel_x + 300, panel_y + 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    # ─── Finger curl bars ───
    cv2.putText(frame, "Finger Curl:", (panel_x + 20, panel_y + 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)

    bar_base_y = panel_y + 260
    bar_max_h = 55
    finger_curls = activity["finger_curls"]

    for i, (name, curl) in enumerate(zip(FINGER_NAMES, finger_curls)):
        bar_x = panel_x + 40 + i * 100
        bar_h = int(curl * bar_max_h)
        bar_w = 50

        cv2.rectangle(frame, (bar_x, bar_base_y - bar_max_h), (bar_x + bar_w, bar_base_y),
                      (40, 40, 40), -1)
        cv2.rectangle(frame, (bar_x, bar_base_y - bar_h), (bar_x + bar_w, bar_base_y),
                      FINGER_COLORS[i], -1)
        cv2.putText(frame, name, (bar_x, bar_base_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, FINGER_COLORS[i], 1)
        cv2.putText(frame, f"{curl:.0%}", (bar_x + 5, bar_base_y - bar_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, FINGER_COLORS[i], 1)

    return frame


def draw_contact_event(frame, event: dict, frame_num: int):
    """Flash a contact event notification with fade-out."""
    h, w = frame.shape[:2]

    frames_since = frame_num - event["frame"]
    if frames_since > 45:
        return frame

    alpha = max(0.0, 1.0 - (frames_since / 45.0))
    color = EVENT_COLORS.get(event["event_type"], (255, 255, 255))

    text = f"{event['hand']}: {event['event_type'].replace('_', ' ').upper()}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    tx = (w - text_size[0]) // 2
    ty = 90

    overlay = frame.copy()
    cv2.rectangle(overlay, (tx - 20, ty - 35), (tx + text_size[0] + 20, ty + 15),
                  (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6 * alpha, frame, 1.0 - 0.6 * alpha, 0)

    color_faded = tuple(int(c * alpha) for c in color)
    cv2.putText(frame, text, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_faded, 2)

    return frame


def draw_header(frame, frame_num, total_frames, fps, num_hands):
    """Top info bar."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    info = f"Frame: {frame_num}/{total_frames} | Hands: {num_hands} | FPS: {fps:.0f}"
    cv2.putText(frame, info, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame, "EGO ACTIVITY RECOGNITION", (w - 420, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
    return frame


# ─── Main Render Pipeline ───────────────────────────────────────────────────────

def render_video():
    """Render predictions onto video frames."""

    input_path = Path(INPUT_VIDEO)
    pred_path = Path(PREDICTIONS_JSON)
    output_path = Path(OUTPUT_VIDEO)

    if not input_path.exists():
        print(f"❌ Input video not found: {INPUT_VIDEO}")
        return
    if not pred_path.exists():
        print(f"❌ Predictions not found: {PREDICTIONS_JSON}")
        print("   Run 'python infer.py' first!")
        return

    # Load predictions
    print(f"📂 Loading predictions from: {pred_path}")
    with open(str(pred_path), "r") as f:
        predictions = json.load(f)

    metadata = predictions["metadata"]
    frames_data = predictions["frames"]

    frame_w = metadata["frame_width"]
    frame_h = metadata["frame_height"]
    fps = metadata["fps"]
    total_frames = metadata["total_frames"]

    # Re-classify actions from raw measurements using rules defined above
    print("🧠 Applying classification rules...")
    print(f"   Right hand can hold iron: {RIGHT_HAND_CAN_HOLD_IRON}")
    print(f"   Left hand can hold iron:  {LEFT_HAND_CAN_HOLD_IRON}")
    print(f"   Iron grip curl threshold: {IRON_GRIP_CURL_THRESHOLD}")
    print(f"   Pinch distance threshold: {PINCH_DISTANCE_THRESHOLD}")

    for frame_pred in frames_data:
        for hand in frame_pred["hands"]:
            classified = classify_action(hand["activity"], hand["label"])
            # Store classification alongside raw data
            hand["_classified"] = classified

    # Re-compute contact events from new classifications
    contact_events = classify_contact_events(frames_data)

    print(f"   Loaded {len(frames_data)} frames, reclassified → {len(contact_events)} contact events")
    print()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))

    # Use avc1 (H.264) for better player compatibility; fall back to mp4v
    writer = None
    for codec in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        test_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_w, frame_h))
        if test_writer.isOpened():
            writer = test_writer
            print(f"   Using codec: {codec}")
            break
        test_writer.release()

    if writer is None:
        print("❌ Could not initialize video writer")
        cap.release()
        return

    # Track recent events for fade-out display
    recent_events: deque = deque(maxlen=5)
    event_idx = 0

    print("🎨 Rendering...")
    for frame_idx, frame_pred in enumerate(frames_data):
        ret, frame = cap.read()
        if not ret:
            break

        frame_num = frame_pred["frame_num"]

        # Draw hands (keypoints only — panels disabled)
        num_hands = len(frame_pred["hands"])
        for i, hand in enumerate(frame_pred["hands"]):
            # Keypoints
            frame = draw_hand_keypoints(frame, hand["keypoints"], frame_w, frame_h)

        # Header
        frame = draw_header(frame, frame_num, total_frames, fps, num_hands)

        writer.write(frame)

        if (frame_idx + 1) % 60 == 0:
            pct = ((frame_idx + 1) / len(frames_data)) * 100
            print(f"   [{pct:5.1f}%] Frame {frame_num}")

    cap.release()
    writer.release()

    print()
    print("=" * 60)
    print("✅ RENDER COMPLETE")
    print("=" * 60)
    print(f"   Output: {output_path}")
    print(f"   Frames rendered: {len(frames_data)}")
    print()


if __name__ == "__main__":
    render_video()
