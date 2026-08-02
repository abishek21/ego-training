#!/usr/bin/env bash
# Stage 3c: run ORB-SLAM3 stereo-inertial on OUR Orbbec clip (slam_data/).
# Produces a camera trajectory (per-frame 6DoF poses) we later fuse into the
# world frame (Stage 3d).
#
# PREREQ: passes only AFTER run_euroc_test.sh works AND the DRAFT settings YAML
# has been reviewed (IMU.T_b_c1 convention/sign, stereo 'lapping' cols).
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="orbslam3:arm64"
SLAM_DATA="${SLAM_DATA:-$PWD/../../slam_data}"          # stereo_3d/slam_data
SETTINGS="${SETTINGS:-$SLAM_DATA/orbslam3_stereo_inertial_DRAFT.yaml}"
OUT="${OUT:-$PWD/../../slam_out}"                        # stereo_3d/slam_out
mkdir -p "$OUT"

# The euroc example expects <seq>/mav0/{cam0,cam1,imu0}. Our slam_data already
# has cam0/cam1/imu0 at top level -> present it as mav0 via a bind alias.
docker run --rm --platform linux/arm64 \
  -v "$SLAM_DATA":/data/mav0 \
  -v "$SETTINGS":/data/settings.yaml:ro \
  -v "$OUT":/out \
  "$IMAGE" bash -lc '
    cd /out
    /orbslam3/Examples/Stereo-Inertial/stereo_inertial_euroc \
      /orbslam3/Vocabulary/ORBvoc.txt \
      /data/settings.yaml \
      /data \
      /data/mav0/timestamps.txt \
      orbbec_traj
    echo "--- outputs ---"; ls -la /out
    echo "--- CameraTrajectory head ---"
    head -3 /out/f_orbbec_traj.txt 2>/dev/null || true
    wc -l /out/f_orbbec_traj.txt 2>/dev/null || true
'
echo "Trajectory -> $OUT (f_orbbec_traj.txt = per-frame TUM poses: ts tx ty tz qx qy qz qw)"
