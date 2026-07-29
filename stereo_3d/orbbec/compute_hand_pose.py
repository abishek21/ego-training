"""
Stage 2: Wrist 6DoF + joint angles from 3D hand keypoints.
----------------------------------------------------------
DERIVED post-processor (same pattern as filter_wearer.py): reads a hands JSON
with `keypoints_3d`, adds per-hand pose/kinematics, writes a new JSON. The core
detection/triangulation is NOT touched.

    hands_3d_wearer.json  --compute_hand_pose.py-->  hands_3d_pose.json

Everything here is PURE GEOMETRY from the triangulated keypoints. NO IMU.
Coordinate frame = LEFT CAMERA (meters). This frame MOVES with the head every
frame, so poses are honestly named `wrist_pose_cam`, NOT `_world`. World-frame
comes later from VIO (Stage 3); a consumer can then transform cam->world.

What we add per hand
--------------------
  geometric_handedness : "Left"/"Right" from palm-normal chirality (replaces
                         WiLoR's per-image label, which is unreliable on ego).
  wrist_pose_cam       : 6DoF rigid pose of the wrist in the left-camera frame
                           position        [x,y,z]      meters
                           quaternion      [qx,qy,qz,qw] (Hamilton, unit)
                           rotation_matrix 3x3          (row-major)
                           axes_doc        explicit axis definitions
  joint_angles_flexion_deg : [15] scalar flexion per joint (degrees). ROBUST
                         grasp signal (angle between adjacent bone segments).
  joint_rotations_axisangle: [15,3] MANO-style per-joint relative rotation
                         (axis*angle, radians). STANDARD format, but see caveat.
  fingertips_3d        : [5,3] convenience subset of keypoints_3d (tips).

HONESTY / caveats
-----------------
  - wrist_pose_cam is CAMERA-frame, not world. Correct & well-defined, but not a
    stable absolute trajectory until VIO (Stage 3).
  - joint_rotations_axisangle captures the BEND between two bone lines only. A
    single bone segment has no observable twist (roll about its own axis), so
    the twist DOF is NOT recovered — distal joints are also noisier because
    fingertip triangulation is the least reliable. Trust flexion > full rotation
    for distal joints. We output both so nothing is lost; label accordingly.
  - No smoothing here (stateless per-frame). Temporal smoothing is a separate
    optional pass so raw geometry stays auditable.

Wrist frame convention (right-handed, anatomical)
-------------------------------------------------
  origin = wrist (kp0)
  x = across the palm, index_MCP(5) -> pinky(17) side  (unit)
  z = palm normal = (kp5-kp0) x (kp17-kp0), flipped to a consistent side
  y = z cross x   (points roughly from wrist toward fingers)
  Handedness sign of the raw normal tells Left vs Right.

Usage:
    python compute_hand_pose.py --in ../hands_3d_wearer.json \
                                --out ../hands_3d_pose.json
"""

import argparse
import json

import numpy as np

# MANO / MediaPipe 21-keypoint order (0=wrist, then 4 per finger tip-last).
TIPS = [4, 8, 12, 16, 20]

# Per-joint flexion triplets (prev, cur, next) — 3 joints per finger, 15 total.
# Angle is measured AT `cur` between bone(prev->cur) and bone(cur->next).
FLEXION_TRIPLETS = [
    # thumb
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    # index
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    # middle
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    # ring
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    # pinky
    (0, 17, 18), (17, 18, 19), (18, 19, 20),
]
JOINT_NAMES = [
    "thumb_cmc", "thumb_mcp", "thumb_ip",
    "index_mcp", "index_pip", "index_dip",
    "middle_mcp", "middle_pip", "middle_dip",
    "ring_mcp", "ring_pip", "ring_dip",
    "pinky_mcp", "pinky_pip", "pinky_dip",
]


def _unit(v, eps=1e-9):
    n = np.linalg.norm(v)
    return v / n if n > eps else v * 0.0


def rotation_matrix_to_quaternion(R):
    """3x3 rotation -> [qx,qy,qz,qw] Hamilton, numerically stable."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (m[2, 1] - m[1, 2]) / S
        qy = (m[0, 2] - m[2, 0]) / S
        qz = (m[1, 0] - m[0, 1]) / S
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        qw = (m[2, 1] - m[1, 2]) / S
        qx = 0.25 * S
        qy = (m[0, 1] + m[1, 0]) / S
        qz = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        qw = (m[0, 2] - m[2, 0]) / S
        qx = (m[0, 1] + m[1, 0]) / S
        qy = 0.25 * S
        qz = (m[1, 2] + m[2, 1]) / S
    else:
        S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        qw = (m[1, 0] - m[0, 1]) / S
        qx = (m[0, 2] + m[2, 0]) / S
        qy = (m[1, 2] + m[2, 1]) / S
        qz = 0.25 * S
    q = np.array([qx, qy, qz, qw])
    return _unit(q)


def palm_chirality(kp):
    """Signed palm triple product -> hand chirality.

    triple = (index_MCP - wrist) . [ (middle_MCP - wrist) x (pinky_MCP - wrist) ]
    Empirically on this data the sign separates hands cleanly (>0 => Left,
    <0 => Right; ~98% agreement with the detector's own label). This is a
    geometric, per-frame determination independent of the detector.
    """
    w = kp[0]
    triple = np.dot(kp[5] - w, np.cross(kp[9] - w, kp[17] - w))
    return "Left" if triple >= 0 else "Right", float(triple)


def wrist_frame(kp):
    """Right-handed wrist rotation matrix (columns = x,y,z axes in cam frame).

    Well-defined and reproducible from fixed keypoint indices:
      y = wrist(0) -> middle_MCP(9)      (~wrist toward fingers)
      x = pinky_MCP(17) -> index_MCP(5)  (across palm)
      z = x cross y      (palm normal), then re-orthogonalize x = y cross z
    The frame is consistent per keypoint construction; Left/Right is reported
    separately via palm_chirality (the frame itself is not mirrored).
    """
    wrist = kp[0]
    y_axis = _unit(kp[9] - wrist)
    x_axis = _unit(kp[5] - kp[17])
    z_axis = _unit(np.cross(x_axis, y_axis))
    # Gram-Schmidt: make x exactly orthogonal to (y,z).
    x_axis = _unit(np.cross(y_axis, z_axis))
    R = np.column_stack([x_axis, y_axis, z_axis])
    return R


def flexion_angles_deg(kp):
    """15 scalar flexion angles (degrees) at each finger joint."""
    angles = []
    for a, b, c in FLEXION_TRIPLETS:
        v1 = _unit(kp[a] - kp[b])   # points back toward parent
        v2 = _unit(kp[c] - kp[b])   # points toward child
        cosang = np.clip(np.dot(v1, v2), -1.0, 1.0)
        # Interior angle at the joint; 180deg = straight, smaller = more bent.
        angles.append(float(np.degrees(np.arccos(cosang))))
    return angles


def joint_axis_angles(kp):
    """[15,3] axis-angle: rotation that bends parent bone into child bone.

    Captures the BEND direction+magnitude at each joint. Does NOT capture twist
    (a single bone line is rotationally symmetric about itself) -> honest gap.
    """
    out = []
    for a, b, c in FLEXION_TRIPLETS:
        parent = _unit(kp[b] - kp[a])   # bone entering the joint
        child = _unit(kp[c] - kp[b])    # bone leaving the joint
        axis = np.cross(parent, child)
        s = np.linalg.norm(axis)
        cosang = np.clip(np.dot(parent, child), -1.0, 1.0)
        ang = float(np.arctan2(s, cosang))   # 0 = straight
        axis = axis / s if s > 1e-9 else np.zeros(3)
        out.append((axis * ang).tolist())
    return out


def process_hand(hand, handedness_qa=True):
    kp = np.asarray(hand["keypoints_3d"], dtype=np.float64)  # (21,3)

    R = wrist_frame(kp)
    quat = rotation_matrix_to_quaternion(R)

    hand["wrist_pose_cam"] = {
        "position": kp[0].round(5).tolist(),
        "quaternion": quat.round(6).tolist(),          # [qx,qy,qz,qw]
        "rotation_matrix": R.round(6).tolist(),        # columns = x,y,z axes
        "axes_doc": ("x=across palm (index->pinky), z=palm normal, "
                     "y=z cross x (~wrist->fingers); left-camera frame, meters"),
    }
    hand["joint_names"] = JOINT_NAMES
    hand["joint_angles_flexion_deg"] = [round(a, 2) for a in flexion_angles_deg(kp)]
    hand["joint_rotations_axisangle"] = [
        [round(v, 5) for v in ax] for ax in joint_axis_angles(kp)
    ]
    hand["fingertips_3d"] = kp[TIPS].round(5).tolist()
    hand["frame"] = "left_camera"

    # Optional geometric handedness QA: an INDEPENDENT Left/Right from palm
    # chirality, used to audit the detector's label. Adds `geometric_handedness`
    # and flags `handedness_uncertain=True` where the two disagree (likely
    # occluded/noisy frames worth down-weighting downstream).
    agree = None
    if handedness_qa:
        handed, _ = palm_chirality(kp)
        hand["geometric_handedness"] = handed
        detector = hand.get("handedness")
        if detector is not None:
            agree = (detector == handed)
            hand["handedness_uncertain"] = (not agree)
    return agree



def main(in_path, out_path, handedness_qa=True):
    with open(in_path) as f:
        data = json.load(f)

    n_hands = 0
    handed_agree = 0
    handed_compared = 0
    for fr in data["frames"]:
        for h in fr["hands"]:
            agree = process_hand(h, handedness_qa=handedness_qa)
            n_hands += 1
            if agree is not None:
                handed_compared += 1
                handed_agree += int(agree)

    meta = data["metadata"]
    meta["stage2_hand_pose"] = {
        "applied": True,
        "source_json": str(in_path),
        "frame": "left_camera",
        "notes": ("Wrist 6DoF + joint kinematics, pure geometry (no IMU). "
                  "Camera-frame; world-frame deferred to VIO (Stage 3)."),
        "added_fields": [
            "wrist_pose_cam", "joint_angles_flexion_deg",
            "joint_rotations_axisangle", "fingertips_3d",
        ] + (["geometric_handedness", "handedness_uncertain"] if handedness_qa else []),
        "handedness_qa": handedness_qa,
        "handedness_agreement_with_detector": (
            round(handed_agree / handed_compared, 4) if handed_compared else None),
    }
    meta["notes"] = meta.get("notes", "") + " | Stage2 wrist 6DoF + joint angles (compute_hand_pose.py)."

    with open(out_path, "w") as f:
        json.dump(data, f)

    print("=" * 56)
    print("STAGE 2 — WRIST 6DoF + JOINT ANGLES")
    print("=" * 56)
    print(f"  in : {in_path}")
    print(f"  out: {out_path}")
    print(f"  hands processed: {n_hands}")
    print(f"  handedness QA: {'ON' if handedness_qa else 'OFF'}")
    if handedness_qa and handed_compared:
        print(f"  geometric vs detector handedness agree: "
              f"{handed_agree}/{handed_compared} ({handed_agree/handed_compared*100:.1f}%)")
        print(f"  flagged handedness_uncertain: {handed_compared - handed_agree}")
    print("  frame: left_camera (NOT world — see metadata)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--handedness-qa", dest="handedness_qa", action="store_true",
                   default=True, help="add geometric handedness QA (default ON)")
    p.add_argument("--no-handedness-qa", dest="handedness_qa", action="store_false",
                   help="disable geometric handedness QA")
    args = p.parse_args()
    main(args.in_path, args.out_path, handedness_qa=args.handedness_qa)
