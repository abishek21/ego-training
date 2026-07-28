"""
Download ONE small HOT3D-Clips clip from HuggingFace.
------------------------------------------------------
The full HOT3D is huge (27GB+). We grab a single .tar clip.

TRAIN clips include ground-truth annotations (hands.json, objects.json)
which the TEST clips withhold — so we use train for prototyping + validation.

Per-frame files in a train clip:
    image_214-1.jpg     RGB camera (fisheye)
    image_1201-1.jpg    stereo SLAM camera left (grayscale)
    image_1201-2.jpg    stereo SLAM camera right (grayscale)
    cameras.json        intrinsics + world poses per camera
    hands.json          GT: 3D wrist pose (umetrack) + MANO + 2D boxes
    objects.json        GT: object 6DoF poses + names + boxes
    hand_crops.json     hand crop regions
    info.json           frame metadata

Prereqs:
    pip install huggingface_hub
    hf auth login   # accept the HOT3D license on the dataset page first

Usage:
    python download_clip.py --split train --clip clip-001849
"""

import argparse
from huggingface_hub import HfApi, hf_hub_download
from pathlib import Path

REPO_ID = "bop-benchmark/hot3d"
OUT_DIR = Path(__file__).parent / "data"


def main(split: str, clip: str | None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    prefix = f"{split}_aria/"

    print(f"📂 Listing {prefix} clips...")
    files = api.list_repo_files(REPO_ID, repo_type="dataset")
    tars = sorted([f for f in files if f.startswith(prefix) and f.endswith(".tar")])
    if not tars:
        print(f"❌ No .tar clips under {prefix}")
        return
    print(f"   Found {len(tars)} clips.")

    target = f"{prefix}{clip}.tar" if clip else tars[0]
    if target not in tars:
        print(f"❌ {target} not found. First few available:")
        for t in tars[:5]:
            print("   ", t)
        return

    print(f"\n⬇️  Downloading: {target}")
    path = hf_hub_download(REPO_ID, target, repo_type="dataset", local_dir=str(OUT_DIR))
    print(f"✅ Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--clip", default="clip-001849",
                        help="Clip name e.g. clip-001849 (None = first)")
    args = parser.parse_args()
    main(args.split, args.clip)

