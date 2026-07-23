"""
Annotate — Click points on frame 1 to define objects for SAM 2 to track.
------------------------------------------------------------------------
Opens frame 1 in a matplotlib window. You click on each object.
Left-click = positive point (this IS the object)
Right-click = negative point (this is NOT the object)
Press a number key (1-9) to switch object ID before clicking.
Press 's' to save and exit.

Saves prompts.json:
{
  "frame_idx": 0,
  "objects": {
     "1": {"label": "iron",  "points": [[x,y],...], "labels": [1,1,...]},
     "2": {"label": "sheet", "points": [[x,y],...], "labels": [1,...]},
  }
}

Usage:
    python annotate.py --video /path/to/ego_press.mp4 --output prompts.json
"""

import cv2
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import TextBox


# Object colors for visualization
OBJ_COLORS = {
    1: "red",
    2: "lime",
    3: "cyan",
    4: "yellow",
    5: "magenta",
}

# Default labels — edit as needed
DEFAULT_LABELS = {
    1: "iron",
    2: "sheet",
    3: "cloth",
}


def annotate(video_path: str, output_path: str, frame_idx: int = 0):
    # Read the target frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"❌ Could not read frame {frame_idx}")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # State
    state = {
        "current_obj": 1,
        "objects": {},  # obj_id -> {"points": [], "labels": []}
    }

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(frame_rgb)
    ax.set_title(
        "LEFT-click = add point | RIGHT-click = negative point\n"
        "Keys 1-5 = switch object | 's' = save & exit\n"
        f"Current object: {state['current_obj']} ({DEFAULT_LABELS.get(1, 'obj1')})"
    )

    plotted = []  # matplotlib artists for redrawing

    def redraw_title():
        obj = state["current_obj"]
        label = DEFAULT_LABELS.get(obj, f"obj{obj}")
        ax.set_title(
            "LEFT-click = add point | RIGHT-click = negative point\n"
            "Keys 1-5 = switch object | 's' = save & exit\n"
            f"Current object: {obj} ({label})"
        )
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x, y = int(event.xdata), int(event.ydata)
        obj = state["current_obj"]

        if obj not in state["objects"]:
            state["objects"][obj] = {"points": [], "labels": []}

        # Left = positive (1), Right = negative (0)
        is_positive = 1 if event.button == 1 else 0
        state["objects"][obj]["points"].append([x, y])
        state["objects"][obj]["labels"].append(is_positive)

        color = OBJ_COLORS.get(obj, "white")
        marker = "*" if is_positive else "x"
        artist = ax.plot(x, y, marker, color=color, markersize=18,
                         markeredgecolor="black", markeredgewidth=1.5)[0]
        plotted.append(artist)
        fig.canvas.draw_idle()

        sign = "+" if is_positive else "-"
        print(f"  Object {obj}: {sign}point at ({x}, {y})")

    def on_key(event):
        if event.key in "123456789":
            state["current_obj"] = int(event.key)
            redraw_title()
            print(f"→ Switched to object {event.key} "
                  f"({DEFAULT_LABELS.get(int(event.key), 'obj')})")
        elif event.key == "s":
            save_and_close()

    def save_and_close():
        # Build output
        out = {
            "frame_idx": frame_idx,
            "video": video_path,
            "objects": {},
        }
        for obj_id, data in state["objects"].items():
            out["objects"][str(obj_id)] = {
                "label": DEFAULT_LABELS.get(obj_id, f"obj{obj_id}"),
                "points": data["points"],
                "labels": data["labels"],
            }

        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)

        print(f"\n✅ Saved {len(out['objects'])} objects to {output_path}")
        for oid, d in out["objects"].items():
            n_pos = sum(d["labels"])
            n_neg = len(d["labels"]) - n_pos
            print(f"   Object {oid} ({d['label']}): {n_pos} positive, {n_neg} negative points")
        plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    print("=" * 60)
    print("ANNOTATION UI")
    print("=" * 60)
    print("LEFT-click  = positive point (this IS the object)")
    print("RIGHT-click = negative point (this is NOT the object)")
    print("Keys 1-5    = switch which object you're labeling")
    print("Key 's'     = save and exit")
    print()
    print("Object IDs:", DEFAULT_LABELS)
    print("=" * 60)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", default="prompts.json", help="Output prompts JSON")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to annotate")
    args = parser.parse_args()

    annotate(args.video, args.output, args.frame)
