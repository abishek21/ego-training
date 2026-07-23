"""
Ego-Centric Hand Keypoint Extraction
-------------------------------------
Converts raw egocentric video into annotated output with hand keypoints.
This is the first stage of: Raw Ego Video → High-Signal VLA Training Data.

Uses MediaPipe Tasks API (0.10.x) for hand landmark detection.

Usage:
    python extract_keypoints.py
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

# ─── Config ────────────────────────────────────────────────────────────────────
INPUT_VIDEO = "/Users/abishek/Downloads/ego_press.mp4"
OUTPUT_VIDEO = "/Users/abishek/ego_training/output/ego_press_keypoints.mp4"
MODEL_PATH = "/Users/abishek/ego_training/models/hand_landmarker.task"

# Processing settings
MAX_FRAMES = 600  # Set to None to process all frames. ~10 seconds at 60fps.

# MediaPipe settings
MAX_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Visualization settings
SKELETON_COLOR = (255, 255, 255)   # White connections
THUMB_COLOR = (0, 165, 255)        # Orange for thumb
INDEX_COLOR = (255, 0, 128)        # Pink for index finger
MIDDLE_COLOR = (255, 255, 0)       # Cyan for middle
RING_COLOR = (0, 255, 255)         # Yellow for ring
PINKY_COLOR = (128, 0, 255)        # Purple for pinky
WRIST_COLOR = (0, 200, 0)         # Green for wrist/palm

KEYPOINT_RADIUS = 5
SKELETON_THICKNESS = 2

# Hand connections (matching MediaPipe hand landmark topology)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring
    (0, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (5, 9), (9, 13), (13, 17),            # Palm
]
# ────────────────────────────────────────────────────────────────────────────────


def get_landmark_color(idx: int) -> tuple:
    """Color-code landmarks by finger group for clarity."""
    if idx == 0:
        return WRIST_COLOR
    elif 1 <= idx <= 4:
        return THUMB_COLOR
    elif 5 <= idx <= 8:
        return INDEX_COLOR
    elif 9 <= idx <= 12:
        return MIDDLE_COLOR
    elif 13 <= idx <= 16:
        return RING_COLOR
    elif 17 <= idx <= 20:
        return PINKY_COLOR
    return WRIST_COLOR


def draw_hand_landmarks(frame: np.ndarray, landmarks, handedness: str,
                        frame_w: int, frame_h: int) -> np.ndarray:
    """Draw keypoints and skeleton connections on a single hand."""

    # Convert normalized landmarks to pixel coords
    points = []
    for lm in landmarks:
        px = int(lm.x * frame_w)
        py = int(lm.y * frame_h)
        points.append((px, py))

    # Draw skeleton connections first (so dots appear on top)
    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(frame, points[start_idx], points[end_idx],
                     SKELETON_COLOR, SKELETON_THICKNESS)

    # Draw keypoints
    for idx, (px, py) in enumerate(points):
        color = get_landmark_color(idx)
        # Larger dots for fingertips (indices 4, 8, 12, 16, 20)
        radius = KEYPOINT_RADIUS + 3 if idx in [4, 8, 12, 16, 20] else KEYPOINT_RADIUS
        cv2.circle(frame, (px, py), radius, color, -1)
        cv2.circle(frame, (px, py), radius, (0, 0, 0), 1)  # Black outline

    # Label the hand (Left/Right)
    if points:
        wrist_x, wrist_y = points[0]
        label_pos = (wrist_x - 20, wrist_y + 30)
        cv2.putText(frame, handedness, label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame


def draw_info_overlay(frame: np.ndarray, frame_num: int, total_frames: int,
                      num_hands: int, fps: float) -> np.ndarray:
    """Draw info panel on the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent overlay bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Info text
    info = f"Frame: {frame_num}/{total_frames} | Hands: {num_hands} | FPS: {fps:.0f}"
    cv2.putText(frame, info, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Pipeline label
    cv2.putText(frame, "EGO KEYPOINT EXTRACTION", (w - 280, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1)

    return frame


def process_video():
    """Main processing pipeline."""

    input_path = Path(INPUT_VIDEO)
    output_path = Path(OUTPUT_VIDEO)
    model_path = Path(MODEL_PATH)

    # Verify input exists
    if not input_path.exists():
        print(f"❌ Input video not found: {INPUT_VIDEO}")
        return

    if not model_path.exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        print("   Download with:")
        print("   curl -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        return

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"❌ Cannot open video: {INPUT_VIDEO}")
        return

    # Video properties
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📹 Input: {input_path.name}")
    print(f"   Resolution: {frame_w}x{frame_h} | FPS: {fps:.1f} | Frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.1f}s")
    print(f"📤 Output: {output_path}")
    print()

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_w, frame_h))

    # Initialize MediaPipe Hand Landmarker (Tasks API)
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

    print("🚀 Processing frames...")
    frame_num = 0
    hands_detected_total = 0
    frames_to_process = MAX_FRAMES if MAX_FRAMES else total_frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        if MAX_FRAMES and frame_num > MAX_FRAMES:
            break

        # Convert BGR → RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Run hand detection (VIDEO mode requires timestamp in ms)
        timestamp_ms = int((frame_num / fps) * 1000)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Draw detections
        num_hands = 0
        if results.hand_landmarks:
            num_hands = len(results.hand_landmarks)
            hands_detected_total += num_hands

            for hand_landmarks, hand_info in zip(
                results.hand_landmarks, results.handedness
            ):
                handedness = hand_info[0].category_name
                frame = draw_hand_landmarks(
                    frame, hand_landmarks, handedness, frame_w, frame_h
                )

        # Draw info overlay
        frame = draw_info_overlay(frame, frame_num, total_frames, num_hands, fps)

        # Write frame
        writer.write(frame)

        # Progress
        if frame_num % 60 == 0 or frame_num == total_frames:
            pct = (frame_num / total_frames) * 100
            print(f"   [{pct:5.1f}%] Frame {frame_num}/{total_frames} — hands: {num_hands}")

    # Cleanup
    cap.release()
    writer.release()
    landmarker.close()

    print()
    print(f"✅ Done! Output saved to: {output_path}")
    print(f"   Avg hands/frame: {hands_detected_total/frame_num:.2f}")
    print(f"   Total frames processed: {frame_num}")


if __name__ == "__main__":
    process_video()
