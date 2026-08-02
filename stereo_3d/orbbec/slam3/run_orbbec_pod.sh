#!/usr/bin/env bash
# Stage 3c (NATIVE, no Docker) — run ORB-SLAM3 stereo-inertial on OUR Orbbec
# clip directly on the pod. Produces a per-frame camera trajectory we later
# fuse into the world frame (Stage 3d).
#
# PREREQ: run_euroc_pod.sh passed AND the DRAFT settings YAML was reviewed
# (IMU.T_b_c1 convention/sign, stereo 'lapping'). slam_data/ regenerated on pod
# via slam_prep.py.
set -euo pipefail

ORB="${ORB_SLAM3_ROOT:-/workspace/ORB_SLAM3}"
REPO="${REPO:-/workspace/ego-training}"
SLAM_DATA="${SLAM_DATA:-$REPO/stereo_3d/slam_data}"
SETTINGS="${SETTINGS:-$SLAM_DATA/orbbec_stereo_inertial.yaml}"
OUT="${OUT:-$REPO/stereo_3d/slam_out}"
BIN="$ORB/Examples/Stereo-Inertial/stereo_inertial_euroc"
VOC="$ORB/Vocabulary/ORBvoc.txt"

[ -x "$BIN" ] || { echo "ERROR: binary not found: $BIN"; exit 1; }
[ -d "$SLAM_DATA/cam0/data" ] || { echo "ERROR: no frames in $SLAM_DATA (run slam_prep.py)"; exit 1; }
[ -f "$SETTINGS" ] || { echo "ERROR: settings not found: $SETTINGS (run make_orbbec_settings.py)"; exit 1; }

# Headless pod: ORB-SLAM3's example opens a Pangolin viewer window. Provide a
# virtual framebuffer (Xvfb) so it runs offscreen instead of crashing on no-X.
if ! command -v xvfb-run >/dev/null 2>&1; then
  echo "Installing xvfb (one-time)..."
  apt-get update -qq && apt-get install -y -qq xvfb
fi

# The euroc example expects <seq>/mav0/{cam0,cam1,imu0}. Our slam_data already
# has cam0/cam1/imu0 at top level -> present it as mav0 via a symlink.
SEQ="$(dirname "$SLAM_DATA")/slam_seq"
mkdir -p "$SEQ"
ln -sfn "$SLAM_DATA" "$SEQ/mav0"

mkdir -p "$OUT"
cd "$OUT"
echo "Running ORB-SLAM3 on Orbbec clip (headless via Xvfb)..."
echo "  settings: $SETTINGS"
xvfb-run -a "$BIN" "$VOC" "$SETTINGS" "$SEQ" "$SLAM_DATA/timestamps.txt" orbbec_traj || true

echo "--- outputs ---"; ls -la "$OUT"
for f in f_orbbec_traj.txt kf_orbbec_traj.txt; do
  [ -f "$OUT/$f" ] && { echo "$f : $(wc -l < "$OUT/$f") rows"; head -2 "$OUT/$f"; }
done
echo "Trajectory (TUM: ts tx ty tz qx qy qz qw) -> $OUT/f_orbbec_traj.txt"
echo "scp this file back to your Mac; leave the 14GB frames on the pod."
