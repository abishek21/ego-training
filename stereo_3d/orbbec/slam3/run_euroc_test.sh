#!/usr/bin/env bash
# Stage 3b GATE: prove the ORB-SLAM3 install on a public EuRoC sample BEFORE
# trusting it on our data. Downloads MH_01_easy (~1.4 GB) and runs the
# stereo-inertial example. If a sane trajectory prints, the build is good.
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="orbslam3:arm64"
EUROC_DIR="${EUROC_DIR:-$PWD/euroc_MH01}"
ZIP="$EUROC_DIR/MH_01_easy.zip"
URL="http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip"

mkdir -p "$EUROC_DIR/mav0"
if [ ! -d "$EUROC_DIR/mav0/cam0" ]; then
  echo "Downloading EuRoC MH_01_easy (~1.4 GB)..."
  wget -c "$URL" -O "$ZIP"
  echo "Unzipping..."
  unzip -q -o "$ZIP" -d "$EUROC_DIR"
fi

# EuRoC ships a timestamps file in the ORB-SLAM3 repo; we generate one from the
# cam0 data.csv (ns timestamps) to avoid depending on the repo copy.
TIMES="$EUROC_DIR/MH01.txt"
if [ ! -f "$TIMES" ]; then
  awk -F',' 'NR>1 {print $1}' "$EUROC_DIR/mav0/cam0/data.csv" > "$TIMES"
fi

echo "Running ORB-SLAM3 stereo-inertial on MH_01_easy (headless)..."
docker run --rm --platform linux/arm64 \
  -v "$EUROC_DIR":/data \
  "$IMAGE" bash -lc '
    cd /orbslam3
    ./Examples/Stereo-Inertial/stereo_inertial_euroc \
      Vocabulary/ORBvoc.txt \
      Examples/Stereo-Inertial/EuRoC.yaml \
      /data \
      /data/MH01.txt \
      euroc_test_traj
    echo "--- trajectory head ---"
    head -3 f_euroc_test_traj.txt 2>/dev/null || echo "(no trajectory file?)"
    wc -l f_euroc_test_traj.txt 2>/dev/null || true
'
echo "If a KeyFrameTrajectory / f_*.txt with many rows printed above, the build WORKS."
