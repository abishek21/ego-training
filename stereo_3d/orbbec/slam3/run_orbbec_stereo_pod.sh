#!/usr/bin/env bash
# DIAGNOSTIC (NATIVE, pod): run ORB-SLAM3 STEREO-ONLY (no IMU) on our clip.
# Isolates whether divergence comes from the IMU (if this is bounded/sane) or
# the visual/stereo geometry (if this also diverges).
#
# Uses the `stereo_euroc` binary (non-inertial) + a --no-imu settings YAML.
set -euo pipefail

ORB="${ORB_SLAM3_ROOT:-/workspace/ORB_SLAM3}"
REPO="${REPO:-/workspace/ego-training}"
SLAM_DATA="${SLAM_DATA:-$REPO/stereo_3d/slam_data}"
SETTINGS="${SETTINGS:-$SLAM_DATA/orbbec_stereo_only.yaml}"
OUT="${OUT:-$REPO/stereo_3d/slam_out}"
BIN="$ORB/Examples/Stereo/stereo_euroc"
VOC="$ORB/Vocabulary/ORBvoc.txt"

[ -x "$BIN" ] || { echo "ERROR: binary not found: $BIN"; exit 1; }
[ -f "$SETTINGS" ] || { echo "ERROR: settings not found: $SETTINGS (make_orbbec_settings.py --no-imu)"; exit 1; }
command -v xvfb-run >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq xvfb; }

SEQ="$(dirname "$SLAM_DATA")/slam_seq"
mkdir -p "$SEQ"; ln -sfn "$SLAM_DATA" "$SEQ/mav0"
mkdir -p "$OUT"; cd "$OUT"

TIMES="$SLAM_DATA/timestamps.txt"
if [ -n "${NFRAMES:-}" ]; then
  TIMES="$OUT/timestamps_${NFRAMES}.txt"; head -n "$NFRAMES" "$SLAM_DATA/timestamps.txt" > "$TIMES"
  echo "  (fast mode: first $NFRAMES frames)"
fi

echo "Running ORB-SLAM3 STEREO-ONLY (no IMU) diagnostic..."
xvfb-run -a "$BIN" "$VOC" "$SETTINGS" "$SEQ" "$TIMES" orbbec_stereo || true
for f in f_orbbec_stereo.txt kf_orbbec_stereo.txt CameraTrajectory.txt; do
  [ -f "$OUT/$f" ] && { echo "$f : $(wc -l < "$OUT/$f") rows"; }
done
