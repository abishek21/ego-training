# Retargeting Reference (external, for guidance)

> Source: external LLM analysis (ChatGPT), saved as a north-star reference.
> This describes the MATURE product. We deliberately build only a SMALL slice
> first (SO-101, sim-first). See RETARGETING_PLAN.md for our actual phased plan.
> Treat the multi-embodiment / full-optimizer parts as FUTURE, not now.

## Core principle (adopt)
Do NOT convert 21 MANO keypoints directly to robot motor positions. Instead:

    human hand trajectory
      -> canonical embodiment-independent end-effector actions
        -> robot-specific IK / joint actions

One dataset then supports SO-101, Franka, UR5, ALOHA, etc.

## What our data already gives
- wrist 3D position        -> robot end-effector position
- wrist orientation (6DoF) -> end-effector orientation  (MORE important than the
                              21 keypoints for arm retargeting)
- finger opening           -> parallel-gripper opening
- thumb-finger distances   -> grasp intent / type
- contact events           -> gripper close/open timing
- left/right hands         -> two arms (bimanual)
- object-relative wrist    -> task-generalizable action (needs object 3D — we
                              deferred this; gate for later)

## Canonical format (target export)
Store THREE representations per frame:
- absolute:      [x,y,z, qx,qy,qz,qw, gripper]
- delta:         [dx,dy,dz, droll,dpitch,dyaw, dgripper]   (OpenVLA-style 7D)
- object-relative: T_object_hand = inv(T_world_object) @ T_world_hand   (LATER)
Keep absolute SE(3) — needed for IK, validation, retargeting.

## Coordinate transforms (the critical step)
    T_robot_hand(t) = T_robot_world * T_world_hand(t) * T_hand_tool
- T_world_hand : our measured wrist pose
- T_robot_world: aligns human workspace to robot workspace
- T_hand_tool  : converts human wrist frame to robot gripper/tool convention
Prefer OBJECT-RELATIVE mapping (needs object pose) OR WORKSPACE-NORMALIZED
mapping (per-axis scale human range -> robot reachable volume). We start with
workspace-normalized (no object pose needed).

## Gripper from keypoints
    g_raw = d(thumb_tip, index_tip) / d(wrist, middle_MCP)   # scale-invariant
    g = clip((g_raw - g_closed)/(g_open - g_closed), 0, 1)   # 0 closed, 1 open
Contact events OVERRIDE finger geometry for close/open timing.

## SO-101 retargeting (our target)
6 DoF: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.
IK objective (per frame): minimize weighted [ position error + orientation error
+ smoothness ||q - q_{t-1}|| + joint-limit penalty + collision penalty ].
LIMITATION: SO-101 cannot match full human wrist orientation (no independent
yaw). Priority weights (start): position 1.0, approach axis 0.5, full rot 0.15,
smoothness 0.1. Project human orientation onto a reachable manifold; relax
yaw/pitch during transport, tighten near contact.

## Trajectory processing (before IK)
1 reject low-confidence frames  2 interpolate short gaps  3 smooth position
(Savitzky-Golay / One-Euro)  4 smooth orientation on SO(3) via SLERP (NEVER
average Euler/quats)  5 resample to robot control freq (cubic for pos, SLERP for
rot, event-aware for gripper)  6 object-relative (later)  7 workspace map
8 sequential IK (init from previous q)  9 velocity/accel limits  10 collision
11 FK validation  12 export.

## Validation (honest — keep these fields)
Per frame store: ik_success, projection_distance_m, orientation_relaxed.
Dataset metrics: median/p95 EE position error, median orientation error,
contact-window alignment, jerk, IK success rate, collision-free rate.
Do NOT silently drop unreachable frames. Mark synthetic robot state as
retargeted/synthetic — NOT hardware-recorded proprioception.

## LeRobot export (our only export target for now)
frame = { observation.images.head, observation.state, action, timestamp,
          frame_index, episode_index, task_index }
- action = robot joint target generated from human motion
- observation.state: we have NO real robot proprioception unless replayed on
  robot/sim. Initially observation.state[t] = action[t-1] (or sim joint state),
  clearly labeled SYNTHETIC.

## Product hierarchy (eventual)
L1 human ground truth (MANO, wrist SE(3), contact, object poses, calib)
L2 canonical robot actions (abs EE, delta EE, object-relative, gripper, phases)
L3 embodiment retargeting (SO-101, Franka, UR5, ALOHA joints)
L4 validation (reachability, FK error, collisions, limits, smoothness)

## OUR SCOPE DISCIPLINE (important)
- Build L2 canonical export first (small, no robot needed).
- Then SO-101 ONLY, sim-first (PyBullet on M3 Mac — no GPU/Isaac).
- DEFER: other embodiments, collision/null-space optimizer, object-relative
  (needs object 3D), multi-format exporters. Add only on a paying reason.
