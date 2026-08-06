# CHECKPOINT — SO-101 Retargeting Track (last updated 2026-08-06)

## GOAL
Turn the human hand annotations (`hands_3d_pose.json`) into robot-executable
motion for a **LeRobot SO-101** (6-DoF: J1 base pan, J2 shoulder, J3 elbow,
J4 wrist flex, J5 wrist roll, J6 gripper). This is the "beyond annotations"
product direction: human demo → robot-ready action.

Sim-first (PyBullet on M3 Mac, no GPU). LeRobot NOT installed — not needed until
export (Phase 3). Dedicated env: `venv_retarget` (pybullet + numpy + scipy + opencv).

## WORKING STANDARD
Production honesty: every script has a "GAPS & ASSUMPTIONS" header. Nothing is
silently faked. Placeholders (workspace mapping, axis map, gripper direction) are
labelled as such and require calibration before real hardware.

---

## DONE ✅

### Phase 1a — arm loads in sim (`load_arm.py`)
- SO-101 URDF + 13 STL meshes downloaded to `retargeting/so101/`.
- Loads clean in PyBullet: 6 movable joints (idx 1,2,3,4,5,7), limits verified
  against URDF, end-effector = `gripperframe` (link idx 6).

### Phase 1b — gripper map (`gripper_map.py`)
- Human openness → SO-101 J6. Metric: `||thumb_tip-index_tip|| / ||wrist-mid_MCP||`
  (scale-invariant). Calibrated by robust p5/p95 (not min/max — outliers ~13x median).
- **A1 resolved:** J6 open/close DIRECTION confirmed correct in GUI (no --invert-j6).
- Known limits (flagged, fine-tune later): thumb-index only captures pinch not
  power grasp (`--metric multi` available); no contact-event override yet (A3).
- Finding: this clip's right hand never fully closes → gripper stays open/semi
  (faithful to a limited-range signal; A2). Proper fix = contact-event fusion.

### Phase 1c — position IK (`pose_ik.py`)
- Human wrist POSITION → workspace-mapped → PyBullet IK → J1–J5. Gripper from 1b.
- **Quality: 98% of frames within 2cm** (FK-verified position error; median ~0cm).
  High-error frames are reach-limit edges, flagged not hidden (B4).
- Live HUD of target/reached/error/joint-angles. Synced source video window.
- **B2 axis map validated in GUI** (hand up→arm up, forward→forward). Working
  region shifted forward (`--fwd`) so arm operates in front of base like the human.
- STILL position-only (B3): tool ORIENTATION not mapped yet (next increment;
  pitch/roll only — SO-101 has NO independent yaw).

### Trajectory gap-fill (`trajectory_filter.py`) — RECOVERED
- Raw wrist trajectory has HOLES (only 2436/3585 frames measured). Robot needs
  continuous motion. Method: **constant-velocity Kalman + RTS smoother** (position),
  **SLERP** (orientation), linear interp (gripper), **chi-square outlier gating**.
- Output `wrist_traj_filled.json`: 3585 frames, each tagged `source`
  (measured / filled / lowconf / outlier_rejected) + `confidence`.
  Counts: 2436 measured, 789 filled (short gaps ≤12), 351 lowconf (long gaps), ~9 rejected.
- ⚠️ **This file was WIPED (never committed) then REBUILT** from the surviving
  JSON's metadata spec. Rebuild reproduces the trajectory to 0.04cm median;
  `filled`/`lowconf` counts match exactly. NOW COMMITTED.
- HONEST: filled/lowconf are INFERRED. A grasp hidden entirely inside a gap is
  missed (gripper interpolated, no fingers seen there). Contact-fusion = TODO.

### Visualization (`overlay_filled.py`)
- Side-by-side MP4: LEFT = SO-101 from FILLED trajectory (IK + gripper), RIGHT =
  synced real video. HUD color-codes source (measured/filled/lowconf).
- Rendered offline (no realtime lag), written at low out-fps → **smooth + slow-mo**
  (every frame present, plays slower — not frame-dropped). → `filled_overlay.mp4`.
- Confirmed: gap-filling removed the jumpiness.

### Camera tooling (for the render view)
- `pick_camera.py` — passive GUI viewer, prints live cam params as you orbit
  (sliders/reset-camera versions auto-crashed on macOS → reverted to passive poll).
- `cam_previews.py` — offline 3×3 grid of candidate angles (crash-free alternative).
- `dump_view.py` — captures the EXACT view matrix from a GUI cam state (guards the
  GUI-vs-offline convention quirk). Verified: offline matrix == GUI matrix (diff 0).
- **Final render camera (user-approved):**
  `--cam-yaw -89.4 --cam-pitch -67.9 --cam-dist 0.8 --cam-target ~[0.13,-0.17,0.27]`
  (steep top-down-ish, aimed at the WORKING region — not the rest pose. Key lesson:
  the picker framed the arm at rest, but IK drives it elsewhere, so target must aim
  at the working region.) Baked as defaults in `overlay_filled.py` + `pose_ik.py`.

---

## KEY DECISIONS / LESSONS
- **Delta actions > absolute** for VLA training: origin-free, no hand-eye calib
  needed, portable across embodiments. Absolute needs a shared frame (calibration
  or markers — not recoverable from this clip). Object-relative is strongest but
  needs object 6DoF (deferred). For sim replay we use workspace-normalization
  (placeholder, B1) — fine for VISUALIZATION, not production ground truth.
- **wrist_traj_filled.json is a robot-control DISTILLATION** of hands_3d_pose.json
  (right wrist + 1 gripper scalar, gap-filled), NOT a richer/complete version —
  it DROPS the 21-pt skeleton, joint angles, 2nd hand, QA fields. pose.json remains
  the full hand annotation.
- **DISCIPLINE: commit code immediately.** Two scripts (trajectory_filter,
  overlay_filled) were wiped because never committed. Recovered, now enforced.

## NEXT STEPS (in order)
1. **Orientation IK** (B3) — add wrist pitch/roll to the tool target (no yaw on
   SO-101; prioritize position > approach axis > roll; report residual honestly).
2. **Contact-event gripper fusion** (A3) — override interpolated gripper with real
   grasp/release timing from `contact_events.json` (fixes grasps-inside-gaps).
3. **Canonical delta-action export** (Phase 0) — per-frame [pos, quat, dpos, drot,
   gripper] + source/confidence; the embodiment-independent trainable artifact.
4. **LeRobot export** (Phase 3) — canonical → SO-101 joint targets in LeRobot format.
5. **Physical SO-101 replay** — the money demo (human ego ↔ real robot).
6. Optional/brainstormed: **robot-overlaid-on-video** (AR). Level 2 = perspective-
   matched using ego intrinsics; honest caveats: fisheye vs pinhole, base placement
   is a plausible fiction, no depth occlusion. A VIZ artifact, not calibrated AR.

## ENV / FILES
- Env: `venv_retarget` (pybullet, numpy, scipy, opencv). URDF+meshes: `retargeting/so101/`.
- Scripts (committed): load_arm, gripper_map, trajectory_filter, pose_ik,
  overlay_filled, pick_camera, cam_previews, dump_view.
- Gitignored artifacts: `*.mp4`, `wrist_traj_filled.json`, `gripper_signal_*.json`,
  `so101/` meshes, `venv_retarget/`, preview PNGs.
- Reference: `RETARGETING_PLAN.md` (parked plan) + `RETARGETING_REFERENCE.md`
  (the multi-embodiment north-star; deliberately NOT fully built — SO-101 only,
  no premature Franka/UR5/ALOHA adapters or full IK optimizer).