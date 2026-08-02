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
- ~~**2nd-person hand leakage**~~ — RESOLVED (see 2026-08-02 update, `filter_wearer.py`).
- ~~**Hand-span noise**~~ — span QA + optional temporal smoothing added in `filter_wearer.py`.

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

---

## CHECKPOINT UPDATE — Stages 1-filter, 2, 3 complete (2026-08-02)

Architecture note: inference (`process_clip.py`, on the GPU pod) is kept separate
from all post-processing. Each post-step reads a JSON and writes a new JSON, core
untouched:  `hands_3d.json` -> filter -> pose -> world.

### DONE ✅ (this session)
- **filter_wearer.py** (post-proc) — keeps only the WEARER's hands. Cues from ego
  geometry: one-sided lateral gate `x > +0.30 m` (2nd person is off to the +x side),
  height gate `y > -0.20 m` (wearer hands enter low, intruder across table is higher),
  distance + span QA, optional temporal span smoothing, closest-N selection. Writes
  provenance (params + reject reasons) to metadata. → `hands_3d_wearer.json`
  (5131 hands, 2813/3588 frames).
- **render_overlay.py** (post-proc) — reproject 3D→2D (KB fisheye) onto left video,
  or compare raw-vs-filtered (green kept / red dropped). Verifies frame↔µs-pts chain
  (0 mismatches) + burns pts into frames. Prefers raw 2D pixels if present.
- **process_clip.py** — now also serializes `keypoints_2d_left/right` (self-contained
  JSON, auditable reproj_error). NOTE: only populated on a fresh pod re-run.
- **compute_hand_pose.py** (Stage 2, post-proc) — wrist 6DoF (`wrist_pose_cam`:
  position + quaternion + rotation_matrix, palm-plane frame), joint flexion angles
  [15] + MANO axis-angle [15,3], fingertips. CAMERA frame, pure geometry, no IMU.
  Geometric handedness QA (palm triple product) is a toggle (`--handedness-qa`) +
  `handedness_uncertain` flag. → `hands_3d_pose.json`.
  - **Handedness finding:** geometric vs detector agree **97.3%** on this clip —
    so the old "handedness unreliable on ego" note does NOT hold here; WiLoR's label
    is fine. (Fixed a chirality bug that first read 47.7%.)
- **Coverage/quality (raw hands_3d):** 96.5% frames have ≥1 3D hand; reproj median
  0.019, ~85% ≤0.05. Wearer-filtered: quality improves (median 0.015, worst tail cut).

### STAGE 3 — WORLD FRAME via SLAM (done, with caveats) ✅
- **slam_prep.py** — Orbbec stereo+IMU → EuRoC/ASL layout. Sync GATE all-pass:
  frame accounting, monotonic ts, stereo L/R skew ~85µs, IMU 999Hz brackets every
  frame (~33 samp/frame), µs→ns exact, Kalibr timeshift (−13.49ms) baked into cam
  clock, 1 frame trimmed to IMU span. Writes `frame_map.csv` (ts_ns→orig frame_idx),
  `timestamps.txt`, `imu0/data.csv`, DRAFT settings. Final: **3587 stereo frames**.
- **slam3/** — ORB-SLAM3 build. Local arm64 Docker OOM'd on `Optimizer.cc` at 6GB →
  moved to **RunPod (Ubuntu 22.04, 50GB RAM, x86)**, native `setup_slam_pod.sh`,
  built clean. (`/workspace` = 1.6PB net mount, disk non-issue.) EuRoC gate skipped
  (ETH server down) → validated directly on our data instead.
- **make_orbbec_settings.py** — generates ORB-SLAM3 settings from calibration, modeled
  on TUM-VI (KB fisheye stereo-inertial). Flags: `--flip-stereo`, `--tbc-direct`,
  `--no-imu`, `--ini-fast/--min-fast`, `--nfeatures`. (mm→m handled; int fps required.)
- **SLAM RESULTS (key story):**
  - Stereo-inertial needed **`--flip-stereo`** to init (ORB-SLAM3's `Stereo.T_c1_c2`
    wanted the INVERSE of provider convention → fixed "0-point maps").
  - Stereo-**inertial DIVERGED** to km-scale regardless of `IMU.T_b_c1` sign
    (both inv and direct blew up: ~4.8km then ~17km extent). IMU destabilizes on this
    near-stationary clip.
  - Stereo-**ONLY (no IMU) is CLEAN**: 100% frames tracked (3587), 1 map, bounded —
    **extent 0.18 m, path 7.2 m, median step 1.7 mm, ~0.06 m/s**. → `f_orbbec_stereo.txt`.
  - **Motion is genuinely tiny: ~18 cm translation, ~20° head rotation** over 2 min.
    Rotation-dominant, low-translation → classic hard-for-VIO ego clip; stereo-only is
    the right tool (stereo gives scale, no IMU-excitation needed).
- **fuse_world.py** (Stage 3d) — lifts hands to WORLD via `T_world_cam`. Join by
  nearest ts (**0 ns residual, exact**). Adds `keypoints_3d_world`, `wrist_pose_world`,
  per-frame `T_world_cam`. → `hands_3d_world.json` (3587 frames, 5129 hands).
  - Validated: rigid-transform preserves hand span (0.1094 cam == world); world extent
    shows head-rotation compensation. NOT ground-truth verified (needs static object).

### OUTPUT CHAIN + SCHEMAS
`hands_3d.json` (raw, cam) → `hands_3d_wearer.json` (+wearer filter) →
`hands_3d_pose.json` (+wrist 6DoF/joints, cam) → `hands_3d_world.json` (+world).
- world adds per hand: `keypoints_3d_world` (21×3), `wrist_pose_world` (pos+quat+R);
  per frame: `T_world_cam` {R,t}, `world_pose_available`.
- **World frame = first tracked camera pose; local, gravity-unaware, meters.**

### HONEST DELIVERABLE FRAMING (for client)
- This clip is **near-stationary manipulation** → **camera-frame is the primary,
  preferred deliverable** (translation-invariant, matches egocentric robot policies).
- **World-frame available** (stereo-only VO) but low-parallax; mainly compensates the
  ~20° head rotation. Bounded/metric/self-consistent, NOT ground-truth verified.
- IMU excluded from world-frame ON PURPOSE (it diverges on low-motion ego footage).

### NEXT STEPS (in order)
1. **3D trajectory visualization** — plot world/cam hand trajectories on 3D axes
   (client-facing artifact). IN PROGRESS.
2. **Stage 4 — object 3D + 3D contact events** (camera frame). Bonus: a static object
   gives the ground-truth point to finally validate the world trajectory.
3. **Pod hygiene** — stop the RunPod pod (code in git, trajectory pulled locally).
4. Optional: re-run `process_clip.py` on pod to populate `keypoints_2d_left/right`.

### ENVIRONMENT NOTES
- Local SLAM analysis/fusion: `venv` (numpy, opencv). Pod build: `setup_slam_pod.sh`.
- Large derived JSONs (`hands_3d*.json`), `slam_data/`, `slam_out/` are gitignored.
- Pod: `root@157.157.221.29 -p 24752`; repo at `/workspace/ego-training`, ORB-SLAM3
  at `/workspace/ORB_SLAM3`. Raw data uploaded to `stereo_data/shared/` (~140MB).