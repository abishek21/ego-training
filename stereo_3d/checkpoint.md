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
1. **3D trajectory visualization** — DONE. `plot_trajectory.py` (static PNG,
   `world_traj.png`) + `animate_trajectory.py` (3D skeleton video, `world_traj.mp4`).
2. **Stage 4 — objects + contact + activity** (camera frame). See STAGE 4 PLAN below.
3. **Pod hygiene** — stop the RunPod pod (code in git, trajectory pulled locally).
4. Optional: re-run `process_clip.py` on pod to populate `keypoints_2d_left/right`.

---

## STAGE 4 PLAN — objects + contact + activity (2026-08-02)

### SCENE REALITY (from actual frame inspection — see the ego view)
- Wearer stands at a **large shiny curved metal table** (specular, low-texture,
  strong fisheye curvature — this is why SLAM struggled).
- **Objects present:** a PILE of **red/orange printed package bags** ("福虾"/shrimp
  graphics, high-contrast) + **translucent plastic mesh/netting** (the "white bag of
  clips", semi-transparent) + a blue bin at back.
- **Two activities observed:**
  1. **Right hand picks a clip** from the white netting/bag (source).
  2. **Both hands insert the clip** into the handle of a red package bag (target).
- A **2nd person's hands** appear on the image-left (already handled by wearer filter).

### KEY DESIGN DECISIONS (honest, validated against the image)
- **DO NOT mask the individual clip** — too small (few px), occluded inside the hand
  exactly when it matters, low-contrast, deformable. Stereo-matching it = garbage 3D.
  → **Use the HAND as the clip proxy:** after grasp, clip position ≈ grasp point
  (thumb-tip kp4 ↔ index-tip kp8 midpoint, in 3D), clip orientation ≈ wrist pose.
  Document this as an explicit approximation (clip pose *within* fingers is NOT
  recovered — would need a marker or closer camera).
- **Red package bags = reliable SAM2 target** (large, high-contrast, textured).
- **Translucent netting = RISKY for SAM2** (transparent/mesh objects segment poorly).
  Test it, but DO NOT promise it; may fall back to a "source region" instead of a mask.
- **Pile problem:** many red bags → for "insert", mask the ACTIVE bag near the hands
  (hand-guided SAM2 prompt), not all bags.
- **Fisheye:** undistort mask POINTS (not the image) before lifting to 3D, same KB
  approach as hands. Edge objects have severe distortion.

### PLAN (validate-before-build: start with a SAM2 spike on ~4 frames)
- **4a — SAM2 on the 2 big objects** (red bag(s), + attempt netting). Reuse
  `sam2_pipeline/`. Prompt the active red bag near the hands. GATE: eyeball which
  objects segment cleanly on 3-4 representative frames BEFORE committing.
  - **SPIKE RESULT (2026-08-02): PASSED.** 100-frame spike on left cam, 2 objects
    (`package_bag` 20pts incl. negatives, `clips_bag` netting 14pts). BOTH masks
    clean — even the translucent netting held (better than expected). SAM2.1 Large
    on A40, ~3.3 it/s. Prompts saved: `sam2_pipeline/prompts_stereo.json` (frame 0).
  - **NOW RUNNING:** full left-cam segmentation (3587 frames, ~18min) →
    `masks_left_full/`. Right cam deferred (decide after checking left holds up).
  - **SAM2 gotcha (fixed):** repo cloned into `sam2_pipeline/sam2/` shadows the
    installed pkg. Fix: `mv sam2 /workspace/sam2_repo && pip install -e .`; run
    segment.py from `sam2_pipeline/`. Checkpoints stay in `sam2_pipeline/checkpoints`.
- **4b — lift masks to 3D** (stereo triangulate mask centroid/region; undistort pts) +
  **hand↔object contact events** (3D distance thresholds → grasp/release).
- **4c — activity segmentation** from hand↔object 3D proximity:
  "pick from white/source region" (right hand) / "insert into red bag" (both hands).
- **Object 6DoF (stretch):** position from mask 3D centroid; orientation from PCA on
  the object's 3D points. Needed for the "insert" (alignment) signal; position-only
  is quick, full orientation is the harder/higher-value part. Clip orientation comes
  from the hand (proxy), not the clip mask.
- **Bonus:** a static object (e.g. a red bag not being handled) = the ground-truth
  point to finally validate the world trajectory (Stage 3 was NOT gt-verified).

### "GOOD ENOUGH TO SHIP" vs "GREAT"
- Ship: 4a (red bags) + 4b (3D contact) + 4c (2 activities).
- Great: + grasp point on object + object 6DoF orientation for the insert.

---

## PARKED R&D — SO-101 retargeting (see RETARGETING_PLAN.md)
- User HAS a **LeRobot SO-101** arm (6-DoF: J1 base, J2 shoulder, J3 elbow, J4 wrist
  flex, J5 wrist roll, J6 gripper). Product direction = retargeting (human hand →
  robot gripper), beyond annotations. **Parallel track, sim-first, AFTER Stage 4.**
- Honest limit: SO-101 wrist has only pitch+roll (NO independent yaw) → can match
  wrist position (IK on J1-3) + pitch/roll (J4-5) + grasp (J6 from thumb-index
  aperture), but NOT full 3-DoF wrist orientation. Use existing IK libs (placo/ikpy),
  don't hand-roll. Workspace scaling (human ~50cm → SO-101 ~40cm) required.