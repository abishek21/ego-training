"""
Animated 3D hand trajectory video (client-facing artifact).
-----------------------------------------------------------
Renders both hands' full 21-keypoint skeletons moving through 3D space over
time (world frame), with a fading wrist trail and a slowly rotating view for
depth. Much more compelling than a static scatter for showing manipulation.

Writes MP4 via OpenCV (no ffmpeg needed): each matplotlib 3D frame -> array.

Usage:
    python animate_trajectory.py --hands ../hands_3d_world.json \
        --out ../world_traj.mp4 --frame world --stride 2 --fps 30
    #   --frame camera  to animate the camera-frame version instead
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stereo3d import HAND_CONNECTIONS

HAND_COLOR = {"Left": "#2ca02c", "Right": "#9467bd"}   # green / purple


def key_for(frame_mode):
    return "keypoints_3d_world" if frame_mode == "world" else "keypoints_3d"


def compute_bounds(frames, key):
    pts = []
    for fr in frames:
        for h in fr["hands"]:
            if key in h:
                pts.append(np.asarray(h[key]))
    pts = np.vstack(pts)
    mins, maxs = pts.min(0), pts.max(0)
    center = (mins + maxs) / 2
    r = (maxs - mins).max() / 2 * 1.05
    return center, r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hands", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frame", choices=["world", "camera"], default="world")
    p.add_argument("--stride", type=int, default=2, help="use every Nth frame")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--trail", type=int, default=60, help="wrist trail length (frames)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    args = p.parse_args()

    with open(args.hands) as f:
        data = json.load(f)
    frames = data["frames"]
    key = key_for(args.frame)
    center, r = compute_bounds(frames, key)

    sel = frames[::args.stride]
    dpi = 100
    fig = plt.figure(figsize=(args.width/dpi, args.height/dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    writer = None
    for codec in ("avc1", "mp4v"):
        w = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*codec),
                            args.fps, (args.width, args.height))
        if w.isOpened():
            writer = w; break
        w.release()

    # rolling wrist trails per handedness
    trails = {"Left": [], "Right": []}
    label = f"{args.frame.upper()} frame"
    print(f"Rendering {len(sel)} frames -> {args.out}")
    for i, fr in enumerate(sel):
        ax.clear()
        ax.set_xlim(center[0]-r, center[0]+r)
        ax.set_ylim(center[1]-r, center[1]+r)
        ax.set_zlim(center[2]-r, center[2]+r)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
        ax.view_init(elev=18, azim=(i * 0.3) % 360)   # slow rotation
        ax.set_title(f"Ego 3D hands — {label}   frame {fr['frame_idx']}")

        present = set()
        for h in fr["hands"]:
            if key not in h:
                continue
            hd = h.get("handedness", "?")
            present.add(hd)
            kp = np.asarray(h[key])
            col = HAND_COLOR.get(hd, "#1f77b4")
            # bones
            for a, b in HAND_CONNECTIONS:
                ax.plot([kp[a, 0], kp[b, 0]], [kp[a, 1], kp[b, 1]],
                        [kp[a, 2], kp[b, 2]], color=col, linewidth=1.5)
            # joints
            ax.scatter(kp[:, 0], kp[:, 1], kp[:, 2], color=col, s=10)
            trails.setdefault(hd, []).append(kp[0])
        # update + draw trails (fading)
        for hd, tr in trails.items():
            if hd not in present:
                continue
            trail = np.asarray(tr[-args.trail:])
            if len(trail) > 1:
                ax.plot(trail[:, 0], trail[:, 1], trail[:, 2],
                        color=HAND_COLOR.get(hd, "#1f77b4"), alpha=0.5, linewidth=1)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frame = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        frame = cv2.resize(frame, (args.width, args.height))
        writer.write(frame)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(sel)}")

    writer.release()
    plt.close(fig)
    print(f"Done -> {args.out}")


if __name__ == "__main__":
    main()
