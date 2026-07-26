"""
Preview RealSense bag — show RGB + Depth side by side.
-------------------------------------------------------
Quick look at the raw streams before reconstruction.

Usage:
    python preview_bag.py --bag /path/to/d435i_walk_around.bag
"""

import argparse
import numpy as np
import cv2
import open3d as o3d


def preview(bag_path: str, delay_ms: int = 30):
    reader = o3d.t.io.RSBagReader()
    reader.open(bag_path)
    meta = reader.metadata
    print(f"📹 {meta.device_name} | {meta.width}x{meta.height} @ {meta.fps}fps")
    print("Press 'q' or ESC to quit, SPACE to pause/resume.")

    paused = False
    im = reader.next_frame()
    frame_num = 0

    while not reader.is_eof():
        if not paused:
            frame_num += 1
            color = im.color.as_tensor().numpy()               # HxWx3 uint8 (RGB)
            depth = im.depth.as_tensor().numpy().squeeze()      # HxW uint16

            # RGB → BGR for cv2 display
            color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

            # Colorize depth for visualization
            depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            # Side by side
            combined = np.hstack([color_bgr, depth_color])

            # Label
            cv2.putText(combined, f"RGB   |   Depth   (frame {frame_num})",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            cv2.imshow("RealSense Preview (RGB | Depth)", combined)
            im = reader.next_frame()

        key = cv2.waitKey(delay_ms) & 0xFF
        if key in (ord('q'), 27):  # q or ESC
            break
        elif key == ord(' '):
            paused = not paused

    reader.close()
    cv2.destroyAllWindows()
    print(f"Done. Viewed {frame_num} frames.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--delay", type=int, default=30, help="ms between frames")
    args = parser.parse_args()

    preview(args.bag, args.delay)
