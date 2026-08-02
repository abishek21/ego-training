#!/usr/bin/env bash
# Build the arm64-native ORB-SLAM3 image. One-time (cached afterwards).
# On a 6 GB Docker VM use JOBS=4; drop to 2 if you hit OOM during the build.
set -euo pipefail

JOBS="${JOBS:-4}"
IMAGE="orbslam3:arm64"

cd "$(dirname "$0")"

echo "Building $IMAGE (arm64-native, JOBS=$JOBS)..."
echo "First build ~30-45 min (Pangolin + g2o + ORB-SLAM3). Cached after."
docker build \
  --platform linux/arm64 \
  --build-arg JOBS="$JOBS" \
  -t "$IMAGE" \
  .

echo "Done. Image: $IMAGE"
echo "Next: ./run_euroc_test.sh  (prove the install on a public EuRoC sample)"
