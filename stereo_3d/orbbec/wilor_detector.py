"""
WiLoR hand detector — drop-in replacement for StereoHandDetector.
------------------------------------------------------------------
Same interface (`.detect(image_bgr) -> list[HandDetection2D]`) so the rest
of the stereo-3D pipeline (undistort points -> triangulate -> JSON) is
UNCHANGED. Only the 2D detection backend swaps.

WiLoR (CVPR 2025) is a SOTA "in-the-wild" hand model — far more robust on
fisheye / egocentric / occluded hands than MediaPipe. Runs on GPU.

Install (on the GPU pod):
    pip install git+https://github.com/warmshao/WiLoR-mini

The wilor-mini pipeline returns, per detected hand:
    {
      "hand_bbox": [x1,y1,x2,y2],
      "is_right": 0 or 1,
      "wilor_preds": {
          "pred_keypoints_2d": (1,21,2)  # image-pixel coords
          "pred_keypoints_3d": (1,21,3)  # up-to-scale, camera frame
          ...
      }
    }
We use pred_keypoints_2d (pixels) so our stereo triangulation gives TRUE
metric 3D (WiLoR's own 3D is single-image, up-to-scale).
"""

from __future__ import annotations

import numpy as np

# Reuse the shared 2D container so the pipeline is identical.
from stereo3d import HandDetection2D


class WiLoRDetector:
    def __init__(self, device: str = None, dtype=None,
                 verbose: bool = True):
        import torch
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
            WiLorHandPose3dEstimationPipeline,
        )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            dtype = torch.float16 if device == "cuda" else torch.float32

        self.device = device
        self._torch = torch
        if verbose:
            print(f"🔧 Loading WiLoR on {device} ({dtype})...")
        self.pipe = WiLorHandPose3dEstimationPipeline(device=torch.device(device),
                                                      dtype=dtype)
        if verbose:
            print("   WiLoR ready.")

    def detect(self, image_bgr: np.ndarray) -> list[HandDetection2D]:
        import cv2
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        outputs = self.pipe.predict(rgb)  # list of hand dicts

        results: list[HandDetection2D] = []
        for hand in outputs:
            preds = hand.get("wilor_preds", {})
            kp2d = preds.get("pred_keypoints_2d", None)
            if kp2d is None:
                continue
            kp2d = np.asarray(kp2d)
            # Shape can be (1,21,2) or (21,2); normalize to (21,2)
            if kp2d.ndim == 3:
                kp2d = kp2d[0]
            if kp2d.shape != (21, 2):
                continue

            is_right = int(hand.get("is_right", 1))
            handedness = "Right" if is_right == 1 else "Left"
            # WiLoR doesn't return a single scalar score; use bbox presence.
            score = float(hand.get("hand_bbox_score", 1.0)) \
                if "hand_bbox_score" in hand else 1.0

            results.append(HandDetection2D(
                keypoints_px=kp2d.astype(np.float64),
                handedness=handedness,
                score=score,
            ))
        return results

    def close(self):
        # WiLoR pipeline has no explicit close; free GPU memory.
        try:
            self._torch.cuda.empty_cache()
        except Exception:
            pass
