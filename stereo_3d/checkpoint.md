// ...existing code...

## CHECKPOINT — Stereo 3D Pipeline (last updated 2026-07-28)

### DONE ✅
- **calibration.py** — parses Orbbec `calibration_camera.yaml` + `calibration_imu.yaml`
  into validated dataclasses (`CameraCalib`, `StereoCalib`, `ImuCalib`). Rotation
  orthonormality checks, mm→m conversion, projection matrices, `T_cam_imu`, timeshift.
- **stereo3d.py** — `StereoHandDetector` (MediaPipe Tasks API, lazy import),
  `undistort_points_kb` (fisheye point undistort), `triangulate`, `lift_frame_to_3d`
  (cross-camera matching + wearer distance filter), `HAND_CONNECTIONS`.
- **wilor_detector.py** — `WiLoRDetector`, drop-in for `StereoHandDetector.detect()`.
- **process_clip.py** — full clip → `hands_3d.json` + `overlay_left.mp4`.
  Preserves exact `timestamp_us`, exact fps. `--detector mediapipe|wilor`.
- **Detector comparison (300 frames):** MediaPipe 38% both-cam → WiLoR **100%**.
  MediaPipe caps at 38% on fisheye ego; preprocessing (CLAHE/gamma/undistort) all
  made it WORSE — proven empirically. WiLoR is the fix (GPU pod, one pip install).
- **Full clip (3588 frames) with WiLoR:** 97% frames-with-hands, hand span
  mean 12.5cm (noisy), distance mean 46cm, reproj err 0.032.
- **Validated frame 75:** hand span 18.3cm (real anatomy) → geometry chain correct.
- **MANO keypoint order == MediaPipe order** (0=wrist,1-4 thumb,...17-20 pinky). Confirmed.

### KNOWN ISSUES ⚠️ (fix next)
- **2nd-person hand leakage** — distance filter (0.9m) too loose; a hand at x≈+0.46m,
  0.55m leaks in. Tighten to ~0.7m + add lateral-x bounds.
- **Hand-span noise** — spans range 0.08–0.19m; add span QA filter (reject <0.12 / >0.22)
  + light temporal smoothing.

### CURRENT OUTPUT: `hands_3d.json` schema
- `metadata`: device, baseline_mm, fps (exact), W/H, coordinate_frame="left_camera",
  units="meters", distortion_model="KB", counts.
- `frames[]`: `frame_idx`, `timestamp_us`, `hands[]`.
- `hands[]`: `handedness`, `keypoints_3d` (21×[x,y,z] m, left-cam frame),
  `wrist_distance_m`, `hand_span_m`, `reproj_error`.

### NEXT STEPS (in order)
1. **Wearer-filter + span QA refinement** (fixes the 2 known issues).
2. **Stage 2 — wrist 6DoF + joint angles** in CAMERA frame (pure geometry from
   keypoints_3d, NO IMU). Wrist pos = kp[0]; orientation from palm plane
   (kp[5]-kp[0] × kp[17]-kp[0]). Joint angles = angles between 3D bone segments.
3. **Stage 3 — VIO (IMU + stereo)** → camera poses → transform to WORLD frame.
   Hard step; use ORB-SLAM3 stereo-inertial (takes stereo + IMU + `T_cam_imu` +
   timeshift, all present in provider data). Needed only for consistent-over-time
   trajectories; camera-frame is fine for relative hand-object signals.
4. **Stage 4 — object 3D + 3D contact events** (hand-object distance thresholds).
5. **3D trajectory visualization**.

### GPU POD (WiLoR)
- Setup: `stereo_3d/orbbec/setup_wilor.sh` (one pip: WiLoR-mini, auto-downloads weights).
- Needs ~8GB VRAM (16GB comfortable). Run from `stereo_3d/orbbec`, `--detector wilor`.
- SCP data up: left/right mp4 + `*calibration_camera.yaml` + `*camera_left_pts.csv`
  (+ right pts for sync check) to `/workspace/ego-training/stereo_data/`.