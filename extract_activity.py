"""
Ego-Centric Activity Recognition Pipeline
-------------------------------------------
Layer 1: Hand Keypoints (MediaPipe)
Layer 2: Grasp Classification (finger geometry)
Layer 3: Action Phases (wrist trajectory + grasp state)
Layer 4: Contact Events (grasp transitions)

Raw Ego Video → High-Signal VLA Training Data

Usage:
    python extract_activity.py
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, List, Tuple

# ─── Config ────────────────────────────────────────────────────────────────────
INPUT_VIDEO = "/Users/abishek/Downloads/ego_press.mp4"
OUTPUT_VIDEO = "/Users/abishek/ego_training/output/ego_press_activity.mp4"
MODEL_PATH = "/Users/abishek/ego_training/models/hand_landmarker.task"

# Processing
MAX_FRAMES = 600  # Set to None for full video. ~10s at 60fps.

# MediaPipe
MAX_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Activity recognition thresholds
CURL_THRESHOLD = 0.55          # Finger curl ratio: >this = curled
PINCH_DISTANCE_THRESHOLD = 0.06  # Normalized dist for pinch detection
VELOCITY_LOW = 8.0             # Pixels/frame: below = static
VELOCITY_HIGH = 25.0           # Pixels/frame: above = fast motion
WRIST_HISTORY_SIZE = 10        # Frames to smooth wrist velocity

# Visualization
SKELETON_COLOR = (255, 255, 255)
THUMB_COLOR = (0, 165, 255)
INDEX_COLOR = (255, 0, 128)
MIDDLE_COLOR = (255, 255, 0)
RING_COLOR = (0, 255, 255)
PINKY_COLOR = (128, 0, 255)
WRIST_COLOR = (0, 200, 0)
KEYPOINT_RADIUS = 4
SKELETON_THICKNESS = 2

# Hand connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]
# ────────────────────────────────────────────────────────────────────────────────


# ─── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class GraspState:
    """Grasp classification for a single hand."""
    grasp_type: str = "unknown"       # power_grasp, open_palm, pinch, relaxed
    finger_curls: List[float] = field(default_factory=list)  # curl ratio per finger
    confidence: float = 0.0


@dataclass
class ActionPhase:
    """Action phase for a single hand."""
    phase: str = "idle"               # reaching, grasping, pressing, lifting, releasing, idle
    wrist_velocity: float = 0.0       # px/frame
    wrist_direction: str = "static"   # up, down, left, right, static


@dataclass
class ContactEvent:
    """A grasp/release contact event."""
    frame: int
    hand: str                         # Left/Right
    event_type: str                   # grasp_initiated, release, pre_grasp
    grasp_from: str                   # previous grasp state
    grasp_to: str                     # new grasp state


# ─── Grasp Classifier ─────────────────────────────────────────────────────────

class GraspClassifier:
    """Classifies grasp type from 21 hand landmarks using finger geometry."""

    # Finger landmark indices: [MCP, PIP, DIP, TIP]
    FINGERS = {
        "thumb": [1, 2, 3, 4],
        "index": [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring": [13, 14, 15, 16],
        "pinky": [17, 18, 19, 20],
    }

    def compute_finger_curl(self, landmarks, finger_indices: List[int]) -> float:
        """
        Compute curl ratio for a finger.
        0 = fully extended, 1 = fully curled.
        Uses ratio of tip-to-MCP distance vs total finger bone length.
        """
        mcp = np.array([landmarks[finger_indices[0]].x, landmarks[finger_indices[0]].y])
        pip = np.array([landmarks[finger_indices[1]].x, landmarks[finger_indices[1]].y])
        dip = np.array([landmarks[finger_indices[2]].x, landmarks[finger_indices[2]].y])
        tip = np.array([landmarks[finger_indices[3]].x, landmarks[finger_indices[3]].y])

        # Total bone chain length
        bone_length = (np.linalg.norm(pip - mcp) +
                       np.linalg.norm(dip - pip) +
                       np.linalg.norm(tip - dip))

        # Direct distance from MCP to TIP
        direct_dist = np.linalg.norm(tip - mcp)

        if bone_length < 1e-6:
            return 0.0

        # Ratio: 1 = straight, 0 = fully curled
        straightness = direct_dist / bone_length
        curl = 1.0 - straightness
        return max(0.0, min(1.0, curl))

    def compute_pinch_distance(self, landmarks) -> float:
        """Distance between thumb tip and index tip (normalized)."""
        thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
        index_tip = np.array([landmarks[8].x, landmarks[8].y])
        return np.linalg.norm(thumb_tip - index_tip)

    def classify(self, landmarks) -> GraspState:
        """Classify grasp type from landmarks."""
        curls = {}
        for finger_name, indices in self.FINGERS.items():
            curls[finger_name] = self.compute_finger_curl(landmarks, indices)

        curl_values = list(curls.values())
        pinch_dist = self.compute_pinch_distance(landmarks)

        # Classification logic
        non_thumb_curls = [curls["index"], curls["middle"], curls["ring"], curls["pinky"]]
        avg_curl = np.mean(non_thumb_curls)

        # Pinch: thumb + index close, other fingers variable
        if pinch_dist < PINCH_DISTANCE_THRESHOLD:
            grasp_type = "pinch"
            confidence = 1.0 - (pinch_dist / PINCH_DISTANCE_THRESHOLD)

        # Power grasp: most fingers curled
        elif avg_curl > CURL_THRESHOLD and sum(c > CURL_THRESHOLD for c in non_thumb_curls) >= 3:
            grasp_type = "power_grasp"
            confidence = avg_curl

        # Open palm: most fingers extended
        elif avg_curl < 0.3 and sum(c < 0.3 for c in non_thumb_curls) >= 3:
            grasp_type = "open_palm"
            confidence = 1.0 - avg_curl

        # Relaxed: partially curled
        else:
            grasp_type = "relaxed"
            confidence = 0.5

        return GraspState(
            grasp_type=grasp_type,
            finger_curls=curl_values,
            confidence=confidence,
        )


# ─── Action Phase Detector ─────────────────────────────────────────────────────

class ActionPhaseDetector:
    """Detects action phase from wrist trajectory + grasp state."""

    def __init__(self):
        self.wrist_history: dict = {}  # hand_label → deque of (x, y) positions

    def update(self, hand_label: str, wrist_pos: Tuple[float, float],
               grasp: GraspState) -> ActionPhase:
        """Determine action phase for a hand."""

        if hand_label not in self.wrist_history:
            self.wrist_history[hand_label] = deque(maxlen=WRIST_HISTORY_SIZE)

        history = self.wrist_history[hand_label]
        history.append(wrist_pos)

        # Compute velocity and direction
        if len(history) < 3:
            return ActionPhase(phase="idle", wrist_velocity=0, wrist_direction="static")

        # Average velocity over recent frames
        velocities = []
        for i in range(1, len(history)):
            dx = history[i][0] - history[i - 1][0]
            dy = history[i][1] - history[i - 1][1]
            velocities.append(np.sqrt(dx * dx + dy * dy))

        avg_velocity = np.mean(velocities[-5:]) if len(velocities) >= 5 else np.mean(velocities)

        # Direction (using last few frames)
        recent_dy = history[-1][1] - history[-3][1]
        recent_dx = history[-1][0] - history[-3][0]

        if abs(recent_dy) > abs(recent_dx):
            direction = "down" if recent_dy > 0 else "up"
        elif abs(recent_dx) > 2:
            direction = "right" if recent_dx > 0 else "left"
        else:
            direction = "static"

        if avg_velocity < VELOCITY_LOW:
            direction = "static"

        # Phase classification combining velocity + grasp + direction
        phase = self._classify_phase(grasp.grasp_type, avg_velocity, direction)

        return ActionPhase(
            phase=phase,
            wrist_velocity=avg_velocity,
            wrist_direction=direction,
        )

    def _classify_phase(self, grasp_type: str, velocity: float, direction: str) -> str:
        """Rule-based action phase from grasp + motion."""

        # Pressing: holding iron, low velocity, wrist static or moving sideways
        if grasp_type == "power_grasp" and velocity < VELOCITY_LOW:
            return "pressing"

        # Lifting: power grasp + wrist moving up
        if grasp_type == "power_grasp" and direction == "up" and velocity > VELOCITY_LOW:
            return "lifting"

        # Reaching: open hand + moving
        if grasp_type == "open_palm" and velocity > VELOCITY_LOW:
            return "reaching"

        # Grasping: fingers closing (transitional - detected via contact events)
        if grasp_type == "power_grasp" and velocity > VELOCITY_HIGH:
            return "repositioning"

        # Adjusting: pinch grip + any motion
        if grasp_type == "pinch":
            return "adjusting_fabric"

        # Smoothing: open palm + low velocity (pressing fabric flat)
        if grasp_type == "open_palm" and velocity < VELOCITY_LOW:
            return "smoothing"

        # Releasing: relaxed hand, was previously grasping
        if grasp_type == "relaxed" and velocity < VELOCITY_LOW:
            return "idle"

        if grasp_type == "relaxed" and velocity > VELOCITY_LOW:
            return "reaching"

        return "active"


# ─── Contact Event Tracker ──────────────────────────────────────────────────────

class ContactEventTracker:
    """Tracks grasp state transitions to identify contact events."""

    def __init__(self):
        self.prev_grasp: dict = {}  # hand_label → previous grasp type
        self.events: List[ContactEvent] = []
        self.cooldown: dict = {}    # hand_label → frames since last event

    def update(self, hand_label: str, grasp: GraspState, frame_num: int) -> Optional[ContactEvent]:
        """Check for contact event (grasp transition)."""

        # Cooldown to avoid flickering
        if hand_label in self.cooldown:
            self.cooldown[hand_label] -= 1
            if self.cooldown[hand_label] > 0:
                return None

        prev = self.prev_grasp.get(hand_label, "unknown")
        curr = grasp.grasp_type

        event = None

        # Detect meaningful transitions
        if prev != curr and prev != "unknown":
            # Open → Closed = grasp initiated
            if prev in ("open_palm", "relaxed") and curr in ("power_grasp", "pinch"):
                event = ContactEvent(
                    frame=frame_num,
                    hand=hand_label,
                    event_type="grasp_initiated",
                    grasp_from=prev,
                    grasp_to=curr,
                )
            # Closed → Open = release
            elif prev in ("power_grasp", "pinch") and curr in ("open_palm", "relaxed"):
                event = ContactEvent(
                    frame=frame_num,
                    hand=hand_label,
                    event_type="release",
                    grasp_from=prev,
                    grasp_to=curr,
                )
            # Open → Relaxed approaching object = pre-grasp
            elif prev == "open_palm" and curr == "relaxed":
                event = ContactEvent(
                    frame=frame_num,
                    hand=hand_label,
                    event_type="pre_grasp",
                    grasp_from=prev,
                    grasp_to=curr,
                )

            if event:
                self.events.append(event)
                self.cooldown[hand_label] = 15  # 15 frame cooldown

        self.prev_grasp[hand_label] = curr
        return event


# ─── Visualization ──────────────────────────────────────────────────────────────

def get_landmark_color(idx: int) -> tuple:
    if idx == 0: return WRIST_COLOR
    elif 1 <= idx <= 4: return THUMB_COLOR
    elif 5 <= idx <= 8: return INDEX_COLOR
    elif 9 <= idx <= 12: return MIDDLE_COLOR
    elif 13 <= idx <= 16: return RING_COLOR
    elif 17 <= idx <= 20: return PINKY_COLOR
    return WRIST_COLOR


def draw_hand_landmarks(frame, landmarks, frame_w, frame_h):
    """Draw keypoints and skeleton."""
    points = []
    for lm in landmarks:
        points.append((int(lm.x * frame_w), int(lm.y * frame_h)))

    for s, e in HAND_CONNECTIONS:
        if s < len(points) and e < len(points):
            cv2.line(frame, points[s], points[e], SKELETON_COLOR, SKELETON_THICKNESS)

    for idx, (px, py) in enumerate(points):
        color = get_landmark_color(idx)
        radius = KEYPOINT_RADIUS + 2 if idx in [4, 8, 12, 16, 20] else KEYPOINT_RADIUS
        cv2.circle(frame, (px, py), radius, color, -1)
        cv2.circle(frame, (px, py), radius, (0, 0, 0), 1)

    return frame


GRASP_COLORS = {
    "power_grasp": (0, 0, 255),      # Red
    "open_palm": (0, 255, 0),        # Green
    "pinch": (255, 0, 255),          # Magenta
    "relaxed": (200, 200, 200),      # Gray
    "unknown": (128, 128, 128),
}

PHASE_COLORS = {
    "pressing": (0, 100, 255),       # Orange
    "lifting": (0, 255, 255),        # Yellow
    "reaching": (255, 200, 0),       # Cyan
    "repositioning": (255, 100, 0),  # Blue
    "adjusting_fabric": (255, 0, 255),# Magenta
    "smoothing": (0, 255, 150),      # Light green
    "idle": (150, 150, 150),         # Gray
    "active": (200, 200, 200),
}


def draw_activity_panel(frame, hand_label, grasp: GraspState, phase: ActionPhase,
                        wrist_px: Tuple[int, int], frame_w: int, frame_h: int,
                        panel_side: str = "left"):
    """Draw large activity info panel for a hand."""
    h, w = frame.shape[:2]

    # Scaled panel dimensions (larger for 4K video readability)
    panel_w = 560
    panel_h = 320

    # Panel position based on which hand
    if panel_side == "left":
        panel_x = 20
    else:
        panel_x = w - panel_w - 20

    panel_y = h - panel_h - 20

    # Semi-transparent background with border
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
    # Border
    border_color = GRASP_COLORS.get(grasp.grasp_type, (128, 128, 128))
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), border_color, 3)

    # Hand label (large)
    cv2.putText(frame, f"{hand_label} Hand", (panel_x + 20, panel_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # ─── Grasp Type ───
    grasp_color = GRASP_COLORS.get(grasp.grasp_type, (128, 128, 128))
    grasp_label = grasp.grasp_type.replace("_", " ").upper()
    cv2.circle(frame, (panel_x + 25, panel_y + 80), 12, grasp_color, -1)
    cv2.putText(frame, f"Grasp: {grasp_label}", (panel_x + 50, panel_y + 87),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, grasp_color, 2)
    cv2.putText(frame, f"{grasp.confidence:.0%}", (panel_x + 420, panel_y + 87),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

    # ─── Action Phase ───
    phase_color = PHASE_COLORS.get(phase.phase, (200, 200, 200))
    phase_label = phase.phase.replace("_", " ").upper()
    cv2.circle(frame, (panel_x + 25, panel_y + 125), 12, phase_color, -1)
    cv2.putText(frame, f"Action: {phase_label}", (panel_x + 50, panel_y + 132),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, phase_color, 2)

    # ─── Velocity bar ───
    vel_bar_max = 350
    vel_bar_width = min(int(phase.wrist_velocity * 5), vel_bar_max)
    cv2.putText(frame, "Velocity:", (panel_x + 20, panel_y + 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)
    # Bar background
    cv2.rectangle(frame, (panel_x + 140, panel_y + 160), (panel_x + 140 + vel_bar_max, panel_y + 180),
                  (50, 50, 50), -1)
    # Bar fill
    cv2.rectangle(frame, (panel_x + 140, panel_y + 160), (panel_x + 140 + vel_bar_width, panel_y + 180),
                  phase_color, -1)
    cv2.putText(frame, f"{phase.wrist_velocity:.1f} px/f  [{phase.wrist_direction}]",
                (panel_x + 140, panel_y + 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # ─── Finger curl bars (larger) ───
    cv2.putText(frame, "Finger Curl:", (panel_x + 20, panel_y + 235),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)
    finger_names = ["Thumb", "Index", "Mid", "Ring", "Pinky"]
    finger_short = ["T", "I", "M", "R", "P"]
    finger_colors = [THUMB_COLOR, INDEX_COLOR, MIDDLE_COLOR, RING_COLOR, PINKY_COLOR]
    bar_base_y = panel_y + 300
    bar_max_h = 60

    for i, (name, curl) in enumerate(zip(finger_short, grasp.finger_curls)):
        bar_x = panel_x + 40 + i * 100
        bar_h = int(curl * bar_max_h)
        bar_w = 50

        # Bar background
        cv2.rectangle(frame, (bar_x, bar_base_y - bar_max_h), (bar_x + bar_w, bar_base_y),
                      (40, 40, 40), -1)
        # Bar fill
        cv2.rectangle(frame, (bar_x, bar_base_y - bar_h), (bar_x + bar_w, bar_base_y),
                      finger_colors[i], -1)
        # Label
        cv2.putText(frame, finger_names[i], (bar_x, bar_base_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, finger_colors[i], 1)
        # Curl value
        cv2.putText(frame, f"{curl:.0%}", (bar_x + 5, bar_base_y - bar_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, finger_colors[i], 1)

    return frame


def draw_contact_event(frame, event: ContactEvent, frame_num: int):
    """Flash a contact event notification."""
    h, w = frame.shape[:2]

    # Show event for 30 frames after it occurs
    frames_since = frame_num - event.frame
    if frames_since > 45:
        return frame

    # Fade out
    alpha = max(0.0, 1.0 - (frames_since / 45.0))

    event_colors = {
        "grasp_initiated": (0, 200, 255),  # Orange
        "release": (0, 255, 0),            # Green
        "pre_grasp": (255, 200, 0),        # Cyan
    }
    color = event_colors.get(event.event_type, (255, 255, 255))

    # Position at center-top
    text = f"⚡ {event.hand}: {event.event_type.replace('_', ' ').upper()}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    tx = (w - text_size[0]) // 2
    ty = 80

    # Background pill
    overlay = frame.copy()
    cv2.rectangle(overlay, (tx - 15, ty - 25), (tx + text_size[0] + 15, ty + 10),
                  (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6 * alpha, frame, 1.0 - 0.6 * alpha, 0)

    # Text with alpha
    color_faded = tuple(int(c * alpha) for c in color)
    cv2.putText(frame, text, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_faded, 2)

    return frame


def draw_header(frame, frame_num, total_frames, fps, num_hands):
    """Top info bar."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 45), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    info = f"Frame: {frame_num}/{total_frames} | Hands: {num_hands} | FPS: {fps:.0f}"
    cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "EGO ACTIVITY RECOGNITION", (w - 320, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2)
    return frame


# ─── Main Pipeline ──────────────────────────────────────────────────────────────

def process_video():
    """Full activity recognition pipeline."""

    input_path = Path(INPUT_VIDEO)
    output_path = Path(OUTPUT_VIDEO)
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
    print(f"📤 Output: {output_path}")
    print()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_w, frame_h))

    # MediaPipe
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

    # Activity recognition modules
    grasp_classifier = GraspClassifier()
    phase_detector = ActionPhaseDetector()
    contact_tracker = ContactEventTracker()

    # Recent events for display
    recent_events: deque = deque(maxlen=5)

    print("🚀 Processing with full activity recognition...")
    print("   Layers: Keypoints → Grasp → Action Phase → Contact Events")
    print()

    frame_num = 0
    stats = {"hands": 0, "grasps": {}, "phases": {}, "events": 0}

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

        num_hands = 0
        if results.hand_landmarks:
            num_hands = len(results.hand_landmarks)
            stats["hands"] += num_hands

            for i, (hand_landmarks, hand_info) in enumerate(
                zip(results.hand_landmarks, results.handedness)
            ):
                hand_label = hand_info[0].category_name  # "Left" or "Right"

                # Layer 1: Draw keypoints
                frame = draw_hand_landmarks(frame, hand_landmarks, frame_w, frame_h)

                # Layer 2: Grasp classification
                grasp = grasp_classifier.classify(hand_landmarks)
                stats["grasps"][grasp.grasp_type] = stats["grasps"].get(grasp.grasp_type, 0) + 1

                # Layer 3: Action phase
                wrist = hand_landmarks[0]
                wrist_px = (wrist.x * frame_w, wrist.y * frame_h)
                phase = phase_detector.update(hand_label, wrist_px, grasp)
                stats["phases"][phase.phase] = stats["phases"].get(phase.phase, 0) + 1

                # Layer 4: Contact events
                event = contact_tracker.update(hand_label, grasp, frame_num)
                if event:
                    recent_events.append(event)
                    stats["events"] += 1
                    print(f"   ⚡ Frame {frame_num}: {event.hand} — {event.event_type} "
                          f"({event.grasp_from} → {event.grasp_to})")

                # Draw activity panel
                panel_side = "left" if i == 0 else "right"
                frame = draw_activity_panel(frame, hand_label, grasp, phase,
                                            (int(wrist_px[0]), int(wrist_px[1])),
                                            frame_w, frame_h, panel_side)

        # Draw contact events (with fade-out)
        for event in recent_events:
            frame = draw_contact_event(frame, event, frame_num)

        # Header
        frame = draw_header(frame, frame_num, total_frames, fps, num_hands)

        writer.write(frame)

        if frame_num % 60 == 0:
            pct = (frame_num / (MAX_FRAMES or total_frames)) * 100
            print(f"   [{pct:5.1f}%] Frame {frame_num} — hands: {num_hands}")

    cap.release()
    writer.release()
    landmarker.close()

    # Summary
    print()
    print("=" * 60)
    print("✅ PIPELINE COMPLETE")
    print("=" * 60)
    print(f"   Output: {output_path}")
    print(f"   Frames processed: {frame_num}")
    print(f"   Avg hands/frame: {stats['hands']/frame_num:.2f}")
    print()
    print("   📊 Grasp Distribution:")
    for g, count in sorted(stats["grasps"].items(), key=lambda x: -x[1]):
        pct = count / stats["hands"] * 100 if stats["hands"] > 0 else 0
        print(f"      {g:20s}: {count:5d} ({pct:.1f}%)")
    print()
    print("   📊 Action Phase Distribution:")
    for p, count in sorted(stats["phases"].items(), key=lambda x: -x[1]):
        pct = count / stats["hands"] * 100 if stats["hands"] > 0 else 0
        print(f"      {p:20s}: {count:5d} ({pct:.1f}%)")
    print()
    print(f"   ⚡ Contact Events: {stats['events']}")
    for event in contact_tracker.events:
        print(f"      Frame {event.frame:4d} | {event.hand:5s} | "
              f"{event.event_type:18s} | {event.grasp_from} → {event.grasp_to}")
    print()


if __name__ == "__main__":
    process_video()
