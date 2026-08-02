"""
3D hand trajectory visualization (client-facing artifact).
----------------------------------------------------------
Plots the wrist trajectories (Left + Right) on clean 3D axes, colored by time,
for BOTH the camera frame and the world frame side by side — so the viewer can
see the raw ego-frame motion vs the head-rotation-compensated world motion.

Reads hands_3d_world.json (needs keypoints_3d + keypoints_3d_world).

Usage:
    python plot_trajectory.py --hands ../hands_3d_world.json --out ../world_traj.png
    python plot_trajectory.py --hands ../hands_3d_world.json --out ../world_traj.png --show
"""

import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


HAND_CMAP = {"Left": "viridis", "Right": "plasma"}


def collect(frames, key):
    """Return {handedness: (Nx3 wrist positions, N frame indices)}."""
    out = {}
    for fr in frames:
        if key == "keypoints_3d_world" and not fr.get("world_pose_available", True):
            continue
        for h in fr["hands"]:
            if key not in h:
                continue
            wrist = np.asarray(h[key])[0]  # kp0
            hd = h.get("handedness", "?")
            out.setdefault(hd, ([], []))
            out[hd][0].append(wrist)
            out[hd][1].append(fr["frame_idx"])
    return {k: (np.asarray(v[0]), np.asarray(v[1])) for k, v in out.items()}


def set_equal_3d(ax, pts):
    """Equal aspect ratio so the trajectory isn't visually distorted."""
    if len(pts) == 0:
        return
    mins, maxs = pts.min(0), pts.max(0)
    center = (mins + maxs) / 2
    r = (maxs - mins).max() / 2 + 1e-3
    ax.set_xlim(center[0]-r, center[0]+r)
    ax.set_ylim(center[1]-r, center[1]+r)
    ax.set_zlim(center[2]-r, center[2]+r)


def plot_frame(ax, data, title):
    allpts = []
    for hd, (pts, fidx) in data.items():
        if len(pts) == 0:
            continue
        allpts.append(pts)
        c = np.linspace(0, 1, len(pts))
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=c,
                   cmap=HAND_CMAP.get(hd, "viridis"), s=6, alpha=0.7)
        # start/end markers
        ax.scatter(*pts[0], color="lime", s=60, marker="o",
                   edgecolor="k", label=f"{hd} start")
        ax.scatter(*pts[-1], color="red", s=60, marker="X",
                   edgecolor="k", label=f"{hd} end")
    allpts = np.vstack(allpts) if allpts else np.zeros((0, 3))
    set_equal_3d(ax, allpts)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=7)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hands", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    with open(args.hands) as f:
        data = json.load(f)
    frames = data["frames"]

    cam = collect(frames, "keypoints_3d")
    world = collect(frames, "keypoints_3d_world")

    fig = plt.figure(figsize=(15, 7))
    ax1 = fig.add_subplot(121, projection="3d")
    plot_frame(ax1, cam, "Wrist trajectory — CAMERA frame\n(moves with head)")
    ax2 = fig.add_subplot(122, projection="3d")
    plot_frame(ax2, world, "Wrist trajectory — WORLD frame\n(stereo-VO, head-motion compensated)")

    meta = data.get("metadata", {})
    fig.suptitle(
        f"Ego stereo 3D hand trajectories — {meta.get('device','')}  "
        f"(Left=green→ path, Right=purple→ path; ○ start, ✕ end; meters)",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=140)
    print(f"Wrote {args.out}")
    for name, d in (("camera", cam), ("world", world)):
        for hd, (pts, _) in d.items():
            if len(pts):
                ext = (pts.max(0)-pts.min(0)).round(3)
                print(f"  {name:6s} {hd:5s}: {len(pts)} pts  extent(m)={ext}")
    if args.show:
        matplotlib.use("MacOSX")
        plt.show()


if __name__ == "__main__":
    main()
