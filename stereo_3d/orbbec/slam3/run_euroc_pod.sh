#!/usr/bin/env bash
# Stage 3b GATE (NATIVE, no Docker) — prove the ORB-SLAM3 build on a public
# EuRoC sample directly on the pod. Downloads MH_01_easy (~1.4 GB), runs the
# stereo-inertial example, and checks a multi-row trajectory was produced.
#
# Run this on the POD after setup_slam_pod.sh. If it prints a trajectory with
# many rows, the build works -> proceed to scp our data + run our clip.
set -euo pipefail

ORB="${ORB_SLAM3_ROOT:-/workspace/ORB_SLAM3}"
EUROC_DIR="${EUROC_DIR:-/workspace/euroc_MH01}"
BIN="$ORB/Examples/Stereo-Inertial/stereo_inertial_euroc"
YAML="$ORB/Examples/Stereo-Inertial/EuRoC.yaml"
VOC="$ORB/Vocabulary/ORBvoc.txt"
URL="http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip"

[ -x "$BIN" ] || { echo "ERROR: binary not found: $BIN (run setup_slam_pod.sh first)"; exit 1; }

mkdir -p "$EUROC_DIR"
if [ ! -d "$EUROC_DIR/mav0/cam0" ]; then
  echo "Downloading EuRoC MH_01_easy (~1.4 GB)..."
  wget -c "$URL" -O "$EUROC_DIR/MH_01_easy.zip"
  echo "Unzipping..."
  unzip -q -o "$EUROC_DIR/MH_01_easy.zip" -d "$EUROC_DIR"
fi

# Build the timestamps file from cam0/data.csv (ns integers).
TIMES="$EUROC_DIR/MH01.txt"
[ -f "$TIMES" ] || awk -F',' 'NR>1 {print $1}' "$EUROC_DIR/mav0/cam0/data.csv" > "$TIMES"

echo "Running ORB-SLAM3 stereo-inertial on MH_01_easy (headless)..."
cd "$ORB"
"$BIN" "$VOC" "$YAML" "$EUROC_DIR" "$TIMES" euroc_test || true

echo "--- trajectory check ---"
for f in f_euroc_test.txt kf_euroc_test.txt CameraTrajectory.txt KeyFrameTrajectory.txt; do
  if [ -f "$f" ]; then
    echo "$f : $(wc -l < "$f") rows"; head -2 "$f"
  fi
done
echo "If a trajectory file with MANY rows appeared, the build WORKS."
echo "Then free disk:  rm -rf $EUROC_DIR"
