# Copilot Instructions — ego-training

## Project Context
This repo converts **egocentric video** (RGB and stereo+IMU) into **high-signal training data** for VLA / robotics teams. Deliverables include hand keypoints, activity/contact events, object masks, and — the current focus — **metric 3D manipulation primitives** (3D hand keypoints, wrist 6DoF, joint angles, 3D contact events) from stereo egocentric video (Orbbec Ego device).

## Working Standards (IMPORTANT)
- **Production-level code. Do NOT oversimplify.** No toy shortcuts, no "good enough for a demo" hacks unless explicitly asked. Handle edge cases, occlusion, missing detections, and coordinate-frame correctness.
- **Be honest and critical.** If an approach is weak, lossy, or won't scale, say so plainly. Call out assumptions and where they break. Never overstate accuracy (e.g., don't invent metrics).
- **Preserve data integrity.** Timestamps (microsecond `pts`), FPS (use exact reported values, never round), and IMU/camera sync must never drift. Every derived frame must carry its exact original timestamp.
- **Coordinate frames matter.** Always be explicit about camera-frame vs world-frame. State units (meters). Undistort points (not whole images) for wide fisheye; use provided calibration (KB fisheye model, baseline, extrinsics).
- **Step-by-step.** The user prefers incremental, verifiable steps with visible results before moving on. Validate each stage (e.g., against ground truth where available) before building the next.
- **Explain the "why".** The user is upskilling in 3D/VLA. Briefly explain reasoning, tradeoffs, and where each choice bites — without being verbose.

## Technical Facts (current data)
- **Device:** Orbbec Ego — stereo (left/right), 1600×1300 @ ~30fps, IMU @ 1000Hz.
- **Distortion:** Kannala-Brandt ("KB") fisheye. Undistort POINTS, not images.
- **Stereo baseline:** ~120.78 mm (from `calibration_camera.yaml`).
- **Poses:** NOT provided → compute via VIO (IMU + stereo) when needed.
- **Depth:** NOT pre-computed (raw stereo) → triangulate matched keypoints.
- **Timestamp gotcha:** the shared `.mp4` is a 2-min clip (~3592 frames) but `*_pts.csv` covers the full recording (~31890 rows). Map video frame i → pts row i.

## Environments
- `venv/` — Python 3.13, RGB pipeline (MediaPipe, OpenCV).
- `venv_rgbd/` — Python 3.12, 3D/stereo work (open3d, pyyaml, MediaPipe, OpenCV). Use this for stereo_3d.

## Repo Conventions
- Inference and rendering are separated (run inference once → save JSON → iterate on rendering).
- Large artifacts (videos, `.npz` masks, `.ply`, `.bag`, model weights, venvs) are gitignored.
- Keep provider data under `stereo_data/`; derived outputs under `stereo_3d/`.

## Deliverable Framing (be accurate)
- The 3D hand/object trajectory data is **demonstration / imitation-learning data**, NOT "sim-ready assets." Do not conflate the two.
- Filter out non-wearer hands (a second person may appear) using **3D distance** (wearer's hands are closest to the ego camera, < ~0.8 m) plus a size prefilter.
