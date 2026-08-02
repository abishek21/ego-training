# Parked R&D Plan — Human→SO-101 Retargeting (product beyond annotations)

**Status:** PARKED (parallel R&D track). Resume after client Stage 4 (objects +
contact) is delivered. This is product development, NOT the paid deliverable —
keep the tracks separate.

## Why this matters (strategy)
Moves us from "annotation vendor" (commodity) to "retargeting product" (moat).
The unlock: we have BOTH real ego data AND a real arm on the standard platform
(LeRobot SO-101). Closing the loop — human ego video → our annotations →
retarget → SO-101 executes — is a demo no annotation vendor can show.

## Hardware: LeRobot SO-101 (6 DoF)
- J1 base rotation, J2 shoulder lift, J3 elbow flex, J4 wrist flex (pitch),
  J5 wrist roll, J6 gripper.
- **Key limitation (design around honestly):** wrist has only 2 orientation DoF
  (pitch J4 + roll J5) — **NO independent yaw.** Human hand has full 3-DoF
  orientation → we can match position + pitch + roll, must approximate/drop yaw.
- Parallel-jaw gripper (1 DoF) → maps cleanly from thumb-index aperture.

## Data → joint mapping
| SO-101 joint | driven by our annotation |
|---|---|
| J1-J3 (position) | wrist x,y,z via INVERSE KINEMATICS |
| J4-J5 (pitch,roll) | wrist_pose orientation (2 of 3 axes) |
| J6 (gripper) | thumb-tip(kp4) ↔ index-tip(kp8) distance |

## Two hard sub-problems
1. **Inverse Kinematics** (wrist pose → J1-J5). DO NOT hand-roll — use SO-101
   URDF + a library (placo / pybullet / ikpy).
2. **Workspace scaling** — human ~50cm volume → SO-101 ~40cm reach from fixed
   base. Translate + scale human workspace into robot workspace.

## Build path (SIM FIRST, honest pacing)
- **Step 0:** get SO-101 URDF (LeRobot repo), load in PyBullet/MuJoCo.
- **Step 1 (afternoon):** gripper only — thumb-index dist → J6. No IK. Easy win.
- **Step 2:** static pose IK — one frame → wrist target → IK → command in sim.
  Handle yaw limit explicitly.
- **Step 3:** trajectory replay in sim — reach-grasp segment, per-frame IK +
  gripper + SMOOTHING (raw data jittery) + joint velocity limits.
- **Step 4:** physical arm via LeRobot → the money demo (split-screen human ego
  ↔ SO-101 doing the same task).
- **Step 5 (frontier):** train a LeRobot policy on retargeted human demos.
  Open research Q: can human ego demos replace robot teleop demos?

## Honest cautions
- 5-DoF wrist can't match full 6-DoF human orientation → early demos APPROXIMATE.
  Frame as "retargeting proof," not "perfect replay."
- Sim before hardware — bad IK into a real servo = crash.
- Real R&D arc, not a weekend. Steps 1-2 quick; 3-5 are projects.
- Don't derail the paid client deliverable.

## To resume, need:
1. SO-101 URDF (LeRobot).
2. Sim-first confirmed (PyBullet easiest).
3. One clean grasp segment from our data as the test trajectory.
