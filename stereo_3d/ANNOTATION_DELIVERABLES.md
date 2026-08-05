# Egocentric Manipulation Dataset — Annotation Deliverables

## Overview

This dataset provides frame-accurate **3D hand, object, and interaction
annotations** derived from egocentric (head-mounted) stereo video of a manual
packing task. The annotations are intended as **demonstration data** for
imitation / robot-learning workflows.

Every annotation is time-synchronized to the source video using exact
per-frame **microsecond timestamps**, so any file can be aligned with the video
or with a robot control stream without drift.

## Coordinate Frames and Units (read first)

- All 3D positions are in **meters**.
- Two reference frames are used:

  **Camera frame** (`_cam` / `left_camera`)
  Origin at the camera; the frame moves with the wearer's head. This is the
  natural frame for relative hand-object manipulation and for egocentric robot
  policies. **It is the primary frame for this dataset.**

  **World frame** (`_world`)
  A fixed local frame anchored to the wearer's starting head pose. Useful when a
  consistent-over-time reference is needed. For this recording the wearer is
  largely stationary, so the world frame primarily compensates for head
  rotation; it is provided as a convenience layer on top of the camera-frame
  data.

- Hand keypoints follow the standard **21-point hand skeleton** ordering
  (0 = wrist; then thumb, index, middle, ring, pinky — four points each).
- Timestamps are integer microseconds (`timestamp_us`); the frame rate is the
  exact recorded rate (not rounded).

## Files

| File | Contents | Frame |
|------|----------|-------|
| `hands_3d_wearer.json` | 3D hand keypoints (wearer only) | camera |
| `hands_3d_pose.json` | + wrist 6DoF + finger/grasp configuration | camera |
| `hands_3d_world.json` | + hands expressed in the fixed world frame | world |
| `contact_events.json` | hand-object grasp/release timeline | — |
| `object_masks` (zip) | per-frame object segmentation masks | image |

---

### 1. `hands_3d_wearer.json` — 3D hands (camera frame)

The core 3D hand annotation, containing only the camera wearer's hands (a
second person occasionally appears in view and has been removed).

Per frame: `frame_idx`, `timestamp_us`, and `hands[]`. Each hand has:
- `handedness` — `"Left"` / `"Right"`
- `keypoints_3d` — 21 points, each `[x, y, z]` in meters (camera frame)
- `wrist_distance_m` — wrist distance from the camera
- `hand_span_m` — wrist-to-middle-fingertip distance (scale sanity value)
- a per-hand geometric quality value for filtering

**Use:** the foundational 3D hand trajectory — for retargeting hand motion to an
end-effector and for relative hand-object reasoning.

### 2. `hands_3d_pose.json` — hands + wrist 6DoF + finger configuration

Everything in File 1, plus derived kinematics per hand. **The richest
camera-frame deliverable.**

Added per hand:
- `wrist_pose_cam` — the wrist as a full **6DoF pose**: `position [x,y,z]` in
  meters, plus orientation as both a **quaternion** and a **rotation matrix**
- **joint angles** — per-finger flexion angles (degrees): how open/closed the
  hand is (grasp configuration)
- a per-joint rotation representation compatible with **standard parametric
  hand models (MANO-style joint ordering)**, for teams that consume that format
- `fingertips` — convenience subset of the five fingertip 3D points

**Use:** maps directly to a robot end-effector pose plus gripper/finger state.
The wrist 6DoF is the primary end-effector signal; finger angles indicate grasp
open/close.

> **Note on MANO-style rotations:** the flexion angles are the robust,
> recommended grasp signal. The full per-joint rotations are provided for format
> compatibility; their finer (distal/twist) components are inherently noisier and
> should be treated as secondary.

### 3. `hands_3d_world.json` — hands in the fixed world frame

Everything in File 2, additionally expressed in the fixed world frame.

Added:
- `keypoints_3d_world` — the 21 hand points in world coordinates
- `wrist_pose_world` — the wrist 6DoF in world coordinates
- per frame: the camera-to-world transform used, and a flag indicating whether a
  world pose was available for that frame

**Use:** when a consistent, head-motion-compensated frame is preferred. For this
mostly-stationary recording, **camera-frame remains the recommended primary
representation**; world-frame is an additional layer. World-frame values are
internally consistent and metric but are **not verified against an external
ground-truth reference**.

### 4. `contact_events.json` — hand-object interaction timeline

A timeline of when each hand makes and breaks contact with each tracked object —
the manipulation moments (when to grasp / when to release).

- `metadata` — tracked object names, a short method note and caveat
- `events[]` — each event has:
  - `type` — `"grasp"` (contact begins) or `"release"` (contact ends)
  - `frame_idx`, `timestamp_us`
  - `hand` — `"Left"` / `"Right"`
  - `object` — the object involved

**Use:** segments the demonstration into interaction phases (approach, grasp,
transport, release) and indicates when a gripper should close or open.

> **Note:** contact is **inferred from hand-to-object proximity** in the imagery,
> not from a physical touch sensor. It has been reviewed against the video and
> tracks the real interactions well; treat it as high-quality inferred contact.

### 5. Object masks — per-frame object segmentation

Per-frame segmentation masks for the two large workspace objects (the package
bag and the clips/source bag), for the full clip.

- one compressed mask file per frame, named by frame index
- each file contains a binary mask per tracked object at full image resolution
- a summary file lists the object names and the frames covered

Reading a mask file:
```python
import numpy as np
d = np.load("masks/00083.npz")
package_bag = d["obj_1"]   # (1300, 1600) uint8, 1 = package_bag pixels
clips_bag   = d["obj_2"]   # (1300, 1600) uint8, 1 = clips_bag pixels
```

**Use:** identifies which object is being manipulated, supports hand-object
association, and underlies the contact-event timeline (File 4).

---

## Honest Scope and Limitations

- **Camera-frame data (Files 1-2)** is the primary, recommended deliverable and
  is well suited to egocentric manipulation learning.
- **World-frame data (File 3)** is a convenience layer; for this near-stationary
  recording it mainly corrects head rotation and is not externally ground-truth
  verified.
- **Contact events (File 4)** are proximity-inferred, video-reviewed, and
  reliable for this task, but are not sensor-measured touch.
- **3D coverage** depends on each hand being visible to both stereo cameras;
  brief single-camera or heavily occluded moments may lack 3D for that hand.
- **Small handled items** (individual clips) are represented via the grasping
  hand rather than segmented directly, as they are too small and occluded to
  segment reliably.

## Frame / Time Alignment

All files share the same **frame indexing** and **microsecond timestamps** as the
source video, enabling exact alignment across annotations and with any external
(e.g. robot) clock.
