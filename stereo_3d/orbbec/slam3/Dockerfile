# ORB-SLAM3 (stereo-inertial) — arm64-native build for Apple Silicon.
# ---------------------------------------------------------------------------
# Design for a RAM-constrained Docker VM (~6 GB):
#   - Ubuntu 20.04 + OpenCV 4.2 from APT (do NOT compile OpenCV -> would OOM).
#   - Pangolin v0.6 (matches ORB-SLAM3's expected API; newer breaks the build).
#   - Low build parallelism (-j2..-j4) to stay under the memory ceiling.
#   - Headless: we run pose export in batch; Pangolin viewer disabled at runtime
#     (no X11 on macOS). The lib still builds; we pass viewer flag off in code.
#
# ORB-SLAM3 is finicky to compile on modern toolchains. Ubuntu 20.04 (GCC 9)
# is the sweet spot with the fewest patches. Known fixes applied below:
#   - C++14, and a couple of missing <memory>/<cstdlib> includes via CXXFLAGS.
FROM --platform=linux/arm64 ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG JOBS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git pkg-config wget ca-certificates \
    libopencv-dev python3-opencv \
    libeigen3-dev \
    libboost-all-dev \
    libssl-dev \
    libglew-dev libgl1-mesa-dev libglu1-mesa-dev \
    libpython2.7-dev \
    libepoxy-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /deps

# --- Pangolin v0.6 (ORB-SLAM3-compatible API) ---
RUN git clone --depth 1 --branch v0.6 https://github.com/stevenlovegrove/Pangolin.git && \
    cd Pangolin && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PANGOLIN_PYTHON=OFF && \
    make -j${JOBS} && make install && ldconfig

# --- ORB-SLAM3 ---
WORKDIR /orbslam3
RUN git clone --depth 1 https://github.com/UZ-SLAMLab/ORB_SLAM3.git . || \
    git clone --depth 1 https://github.com/UZ-SLAMLab/ORB_SLAM3.git /orbslam3

# Compatibility shims for GCC 9 / newer std headers (common ORB-SLAM3 fixes).
ENV CXXFLAGS="-std=c++14 -include memory -include cstdlib -include ctime"

# Build the bundled thirdparty (DBoW2, g2o, Sophus) + ORB-SLAM3 itself.
# We call the pieces explicitly so we can cap parallelism and fail fast.
RUN set -e; \
    echo "== DBoW2 =="; cd /orbslam3/Thirdparty/DBoW2 && mkdir -p build && cd build && \
      cmake .. -DCMAKE_BUILD_TYPE=Release && make -j${JOBS}; \
    echo "== g2o =="; cd /orbslam3/Thirdparty/g2o && mkdir -p build && cd build && \
      cmake .. -DCMAKE_BUILD_TYPE=Release && make -j${JOBS}; \
    echo "== Sophus =="; cd /orbslam3/Thirdparty/Sophus && mkdir -p build && cd build && \
      cmake .. -DCMAKE_BUILD_TYPE=Release && make -j${JOBS}

# Uncompress the DBoW2 vocabulary (needed at runtime).
RUN cd /orbslam3/Vocabulary && tar -xf ORBvoc.txt.tar.gz

# ORB-SLAM3's CMake demands OpenCV >= 4.4, but Ubuntu 20.04 ships 4.2 (which
# works fine in practice). Relax the version pin in every CMakeLists that has
# it. Well-known community fix; purely a version-guard change, not code.
RUN grep -rl "OpenCV 4.4" /orbslam3 --include=CMakeLists.txt | \
      xargs -r sed -i 's/OpenCV 4\.4/OpenCV 4.2/g'

# Build ORB-SLAM3 library + example executables (stereo-inertial included).
RUN cd /orbslam3 && mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && make -j${JOBS}

ENV ORB_SLAM3_ROOT=/orbslam3
WORKDIR /work
CMD ["/bin/bash"]
