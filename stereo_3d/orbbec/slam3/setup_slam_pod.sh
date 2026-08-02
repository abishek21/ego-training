#!/usr/bin/env bash
# Native ORB-SLAM3 (stereo-inertial) build for a RunPod CPU/GPU pod.
# Target: runpod/pytorch:2.4.0-...-ubuntu22.04  (Ubuntu 22.04, x86_64, 50GB RAM).
# No Docker here — the pod IS the container. High RAM => build at -j$(nproc).
#
# Ubuntu 22.04 ships OpenCV 4.5 (apt), so ORB-SLAM3's ">=4.4" check passes with
# no patch. We keep a harmless version-relax sed as a safety net anyway.
#
# Usage (on the pod):
#   bash setup_slam_pod.sh
# Produces: /workspace/ORB_SLAM3 (built), Pangolin installed system-wide.
set -euo pipefail

JOBS="$(nproc)"
PREFIX="${PREFIX:-/workspace}"
echo "== ORB-SLAM3 pod setup :: JOBS=$JOBS PREFIX=$PREFIX =="

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  build-essential cmake git pkg-config wget unzip ca-certificates \
  libopencv-dev python3-opencv \
  libeigen3-dev libboost-all-dev libssl-dev \
  libglew-dev libgl1-mesa-dev libglu1-mesa-dev \
  libpython2.7-dev libepoxy-dev

mkdir -p "$PREFIX"
cd "$PREFIX"

# --- Pangolin v0.6 (ORB-SLAM3-compatible API) ---
if [ ! -d Pangolin ]; then
  git clone --depth 1 --branch v0.6 https://github.com/stevenlovegrove/Pangolin.git
fi
cd Pangolin && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PANGOLIN_PYTHON=OFF
make -j"$JOBS"
make install
ldconfig
cd "$PREFIX"

# --- ORB-SLAM3 ---
if [ ! -d ORB_SLAM3 ]; then
  git clone --depth 1 https://github.com/UZ-SLAMLab/ORB_SLAM3.git
fi
cd ORB_SLAM3

# 22.04 / GCC 11 compatibility shims (common ORB-SLAM3 fixes) + relax OpenCV pin.
export CXXFLAGS="-std=c++14 -include memory -include cstdlib -include ctime"
grep -rl "OpenCV 4.4" . --include=CMakeLists.txt | xargs -r sed -i 's/OpenCV 4\.4/OpenCV 4.2/g' || true

# Thirdparty (DBoW2, g2o, Sophus)
for d in DBoW2 g2o Sophus; do
  echo "== Thirdparty/$d =="
  cd "Thirdparty/$d" && mkdir -p build && cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release
  make -j"$JOBS"
  cd "$PREFIX/ORB_SLAM3"
done

# Vocabulary
[ -f Vocabulary/ORBvoc.txt ] || (cd Vocabulary && tar -xf ORBvoc.txt.tar.gz)

# ORB-SLAM3 itself. 50GB RAM => -j$(nproc) is safe (this OOM'd at -j4 on 6GB).
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$JOBS"

echo "== DONE =="
echo "Executable: $PREFIX/ORB_SLAM3/Examples/Stereo-Inertial/stereo_inertial_euroc"
