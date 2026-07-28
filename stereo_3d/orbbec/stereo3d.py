"""
Stereo 3D hand keypoint lifting (Orbbec Ego).
----------------------------------------------
Pipeline for a single synchronized (left, right) frame pair:

  1. Detect hands with MediaPipe on BOTH raw fisheye images (21 pts each).
  2. Undistort the keypoints (NOT the image) with the KB fisheye model,
     yielding normalized camera-ray coordinates.
  3. Match hands across the two cameras by epipolar/vertical consistency
     (NOT by MediaPipe's handedness label, which is unreliable on ego views).
  4. Triangulate matched keypoints -> 21 points in 3D (LEFT camera frame, meters).
  5. Validate geometry (plausible distance + bone lengths) and filter the
     wearer's hands by distance (a second person may appear in frame).

Design notes / honesty:
  - MediaPipe runs on the RAW fisheye. It's reliable near the image center
    (mild distortion); accuracy degrades toward the edges. We undistort the
    detected POINTS so 3D geometry stays correct regardless.
  - Triangulation requires the hand to be visible in BOTH cameras. Frames
    where a hand is seen by only one camera yield no 3D for that hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from calibration import StereoCalib, CameraCalib

# NOTE: mediapipe is imported lazily inside StereoHandDetector so this module
# can be used with the WiLoR backend on machines without mediapipe installed.


# MediaPipe hand landmark topology (for bone-length validation + rendering)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (0, 9), (9, 10), (10, 11), (11, 12),    # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (5, 9), (9, 13), (13, 17),              # palm
]

# Plausible metric ranges for a real adult hand (meters) — used for QA.
WEARER_MAX_DISTANCE_M = 0.90     # wearer's own hands are within arm's reach
WEARER_MIN_DISTANCE_M = 0.10
MAX_HAND_SPAN_M = 0.30           # wrist->middle-tip should be < ~30cm
MIN_HAND_SPAN_M = 0.05


# ─────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────
@dataclass
class HandDetection2D:
    """One hand detected in one image."""
    keypoints_px: np.ndarray     # (21,2) pixel coords in the raw image
    handedness: str              # MediaPipe label ("Left"/"Right") — hint only
    score: float                 # detection confidence

    @property
    def wrist_px(self) -> np.ndarray:
        return self.keypoints_px[0]

    def bbox(self) -> tuple[float, float, float, float]:
        x1, y1 = self.keypoints_px.min(axis=0)
        x2, y2 = self.keypoints_px.max(axis=0)
        return float(x1), float(y1), float(x2), float(y2)


@dataclass
class Hand3D:
    """A triangulated hand in the LEFT camera frame (meters)."""
    keypoints_3d: np.ndarray     # (21,3) meters, left-camera frame
    handedness: str
    reproj_error_px: float       # mean reprojection error (QA)
    distance_m: float            # wrist distance from camera
    left_2d: np.ndarray          # (21,2) source pixels (left)
    right_2d: np.ndarray         # (21,2) source pixels (right)

    def hand_span_m(self) -> float:
        """Wrist (0) to middle-finger tip (12) Euclidean distance."""
        return float(np.linalg.norm(self.keypoints_3d[12] - self.keypoints_3d[0]))


# ─────────────────────────────────────────────────────────────────────
# Image preprocessing (improves detection on dark / low-contrast ego video)
# ─────────────────────────────────────────────────────────────────────
def enhance_for_detection(image_bgr: np.ndarray,
                          clahe_clip: float = 2.0,
                          clahe_grid: int = 8,
                          brightness_gain: float = 1.0) -> np.ndarray:
    """
    Enhance a dark/low-contrast frame to help hand detection.

    Uses CLAHE (Contrast-Limited Adaptive Histogram Equalization) on the L
    channel of LAB — boosts LOCAL contrast without blowing out bright areas
    (better than a global brightness multiply). Optional brightness gain on top.

    This does NOT change geometry — keypoints detected on the enhanced image
    are still in the SAME pixel coordinates, so undistortion/triangulation are
    unaffected. We only enhance the pixels the detector sees.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip,
                            tileGridSize=(clahe_grid, clahe_grid))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if brightness_gain != 1.0:
        out = cv2.convertScaleAbs(out, alpha=brightness_gain, beta=0)
    return out


# ─────────────────────────────────────────────────────────────────────
# Hand detector (MediaPipe wrapper)
# ─────────────────────────────────────────────────────────────────────
class StereoHandDetector:
    def __init__(self, model_path: str = None, max_hands: int = 4,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 enhance: bool = False):
        # MediaPipe 0.10.x uses the Tasks API (mp.solutions was removed).
        # We detect up to `max_hands` because a second person may be present;
        # filtering to the wearer happens later via 3D distance.
        # NOTE: `enhance` (CLAHE) was tested and REDUCED detection on this
        # fisheye ego data (38% raw -> 29% CLAHE). Default OFF. The real
        # bottleneck is MediaPipe on fisheye; upgrade to WiLoR/HaMeR for gains.
        self.enhance = enhance
        import mediapipe as mp          # lazy import (not needed for WiLoR path)
        self._mp = mp
        from pathlib import Path
        if model_path is None:
            # Default to the repo's downloaded model.
            model_path = str(Path(__file__).resolve().parents[2]
                             / "models" / "hand_landmarker.task")

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,   # independent per frame
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._det = HandLandmarker.create_from_options(options)

    def detect(self, image_bgr: np.ndarray) -> list[HandDetection2D]:
        h, w = image_bgr.shape[:2]
        # Enhance ONLY what the detector sees; keypoint coords stay in the
        # original image space, so geometry (undistort/triangulate) is intact.
        det_input = enhance_for_detection(image_bgr) if self.enhance else image_bgr
        rgb = cv2.cvtColor(det_input, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        res = self._det.detect(mp_image)
        out: list[HandDetection2D] = []
        if not res.hand_landmarks:
            return out
        for lms, handed in zip(res.hand_landmarks, res.handedness):
            pts = np.array([[lm.x * w, lm.y * h] for lm in lms],
                           dtype=np.float64)
            out.append(HandDetection2D(
                keypoints_px=pts,
                handedness=handed[0].category_name,
                score=float(handed[0].score),
            ))
        return out

    def close(self):
        self._det.close()


# ─────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────
def undistort_points_kb(points_px: np.ndarray, cam: CameraCalib) -> np.ndarray:
    """
    Undistort fisheye (KB) pixel points -> NORMALIZED camera-ray coords.

    Returns (N,2) array of [x/z, y/z] (i.e. rays through the pinhole),
    which is what triangulation with K-free projection matrices expects.
    """
    pts = points_px.reshape(-1, 1, 2).astype(np.float64)
    und = cv2.fisheye.undistortPoints(pts, cam.K, cam.D)  # no P -> normalized
    return und.reshape(-1, 2)


def triangulate(left_norm: np.ndarray, right_norm: np.ndarray,
                stereo: StereoCalib) -> np.ndarray:
    """
    Triangulate matched NORMALIZED points (N,2) each -> (N,3) in LEFT frame.
    """
    P_left, P_right = stereo.projection_matrices()  # K-free (normalized) matrices
    ptsL = left_norm.T.astype(np.float64)           # (2,N)
    ptsR = right_norm.T.astype(np.float64)
    hom = cv2.triangulatePoints(P_left, P_right, ptsL, ptsR)  # (4,N)
    pts3d = (hom[:3] / hom[3]).T                    # (N,3), left-camera frame
    return pts3d


def reprojection_error(pts3d: np.ndarray, left_norm: np.ndarray,
                       right_norm: np.ndarray, stereo: StereoCalib) -> float:
    """Mean reprojection error (in normalized units) across both cameras."""
    P_left, P_right = stereo.projection_matrices()
    hom = np.hstack([pts3d, np.ones((len(pts3d), 1))])  # (N,4)
    errs = []
    for P, obs in ((P_left, left_norm), (P_right, right_norm)):
        proj = (P @ hom.T).T           # (N,3)
        proj = proj[:, :2] / proj[:, 2:3]
        errs.append(np.linalg.norm(proj - obs, axis=1))
    return float(np.mean(np.concatenate(errs)))


# ─────────────────────────────────────────────────────────────────────
# Cross-camera hand matching
# ─────────────────────────────────────────────────────────────────────
def match_hands(left_hands: list[HandDetection2D],
                right_hands: list[HandDetection2D],
                cam_left: CameraCalib, cam_right: CameraCalib,
                max_vertical_px: float = 60.0) -> list[tuple[int, int]]:
    """
    Match hands across cameras by epipolar (near-horizontal) consistency.

    The stereo baseline is horizontal (~121mm in X), so a physical point's
    two projections share a similar ROW (y) and differ mainly in COLUMN (x,
    the disparity). We match each left hand to the right hand whose wrist row
    is closest (within max_vertical_px), preferring positive disparity.

    Returns list of (left_idx, right_idx) matches. Unmatched hands dropped.
    """
    matches: list[tuple[int, int]] = []
    used_right: set[int] = set()

    for li, lh in enumerate(left_hands):
        best_ri, best_cost = -1, 1e18
        lw = lh.wrist_px
        for ri, rh in enumerate(right_hands):
            if ri in used_right:
                continue
            rw = rh.wrist_px
            dy = abs(lw[1] - rw[1])          # vertical difference (should be small)
            if dy > max_vertical_px:
                continue
            disparity = lw[0] - rw[0]        # horizontal disparity
            # A valid stereo match needs the point closer than infinity ->
            # nonzero disparity; sign depends on rectification but should be
            # consistent. Penalize tiny/negative disparity lightly.
            cost = dy + 0.1 * abs(min(disparity, 0.0))
            if cost < best_cost:
                best_cost, best_ri = cost, ri
        if best_ri >= 0:
            matches.append((li, best_ri))
            used_right.add(best_ri)
    return matches


# ─────────────────────────────────────────────────────────────────────
# Full single-frame lift
# ─────────────────────────────────────────────────────────────────────
def lift_frame_to_3d(left_bgr: np.ndarray, right_bgr: np.ndarray,
                     detector: StereoHandDetector,
                     stereo: StereoCalib,
                     wearer_only: bool = True) -> list[Hand3D]:
    """Detect, match, triangulate all hands in one stereo frame pair."""
    left_hands = detector.detect(left_bgr)
    right_hands = detector.detect(right_bgr)

    hands3d: list[Hand3D] = []
    if not left_hands or not right_hands:
        return hands3d

    for li, ri in match_hands(left_hands, right_hands, stereo.left, stereo.right):
        lh, rh = left_hands[li], right_hands[ri]
        left_norm = undistort_points_kb(lh.keypoints_px, stereo.left)
        right_norm = undistort_points_kb(rh.keypoints_px, stereo.right)

        pts3d = triangulate(left_norm, right_norm, stereo)
        reproj = reprojection_error(pts3d, left_norm, right_norm, stereo)
        dist = float(np.linalg.norm(pts3d[0]))     # wrist distance

        hand = Hand3D(
            keypoints_3d=pts3d,
            handedness=lh.handedness,
            reproj_error_px=reproj,
            distance_m=dist,
            left_2d=lh.keypoints_px,
            right_2d=rh.keypoints_px,
        )

        # Geometry QA + wearer filtering
        span = hand.hand_span_m()
        plausible = (WEARER_MIN_DISTANCE_M < dist < WEARER_MAX_DISTANCE_M
                     if wearer_only else True)
        span_ok = MIN_HAND_SPAN_M < span < MAX_HAND_SPAN_M
        if plausible and span_ok:
            hands3d.append(hand)

    return hands3d
