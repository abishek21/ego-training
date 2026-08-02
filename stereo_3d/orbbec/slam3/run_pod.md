# ORB-SLAM3 on a RunPod pod — run guide (Stage 3b/3c)

Pod: `runpod/pytorch:2.4.0-...-ubuntu22.04`, x86_64, **50 GB RAM / 9 vCPU / 40 GB disk**.
GPU unused (SLAM is CPU-bound). Big RAM removes the local 6 GB OOM problem.

> ⚠️ **Disk is the tight resource (40 GB).** The base image is ~15 GB. Extract
> stereo frames on the pod (~14 GB) and delete the EuRoC test data right after
> the gate. Consider `--gray` extraction to roughly halve frame size (TODO flag).

## 0. Code onto the pod (data comes LATER — build first)
```bash
cd /workspace
git clone https://github.com/abishek21/ego-training.git
cd ego-training
```

## 1. Build ORB-SLAM3 (one-time, ~15-20 min at -j9)
```bash
cd /workspace/ego-training/stereo_3d/orbbec/slam3
bash setup_slam_pod.sh
```

## 2. GATE — prove the build on public EuRoC (self-downloads; NO upload needed)
Do this BEFORE uploading our data — fail fast if the build is broken.
```bash
bash run_euroc_pod.sh         # downloads MH_01, runs, prints a trajectory
rm -rf /workspace/euroc_MH01  # free ~4 GB once it passes
```

## 3. Only now: scp up our small raw provider files (~few hundred MB)
```bash
# From LOCAL machine (do NOT upload slam_data/ — regenerated on the pod):
DATA="stereo_data/Stereo Video with IMU (shared)"
scp -P <PORT> \
  "$DATA/camera_left_2min.mp4" "$DATA/camera_right_2min.mp4" \
  "$DATA"/*calibration_camera.yaml "$DATA"/*calibration_imu.yaml \
  "$DATA"/*camera_left_pts.csv "$DATA"/*camera_right_pts.csv \
  "$DATA"/*imu_2min.csv \
  root@<POD_IP>:/workspace/ego-training/stereo_data/shared/
```

## 4. Python env + regenerate slam_data/ ON THE POD
```bash
cd /workspace/ego-training
python3 -m venv venv && . venv/bin/activate
pip install numpy opencv-python pyyaml
cd stereo_3d/orbbec
python slam_prep.py --data ../../stereo_data/shared --out ../slam_data --validate-only  # gate
python slam_prep.py --data ../../stereo_data/shared --out ../slam_data                  # full
```

## 5. Review DRAFT settings, then run on our clip
Verify the flagged `IMU.T_b_c1` convention/sign and stereo `lapping` cols in
`../slam_data/orbslam3_stereo_inertial_DRAFT.yaml` against ORB-SLAM3's reference
fisheye YAML (`/workspace/ORB_SLAM3/Examples/Stereo-Inertial/*.yaml`).
```bash
cd ../slam3
bash run_orbbec_pod.sh        # -> stereo_3d/slam_out/f_orbbec_traj.txt
```

## 6. Pull the trajectory back (tiny), leave the 14 GB PNGs on the pod
```bash
# From LOCAL:
scp -P <PORT> root@<POD_IP>:/workspace/ego-training/stereo_3d/slam_out/f_orbbec_traj.txt \
  stereo_3d/slam_out/
```

## Notes
- The NATIVE pod scripts are `setup_slam_pod.sh`, `run_euroc_pod.sh`,
  `run_orbbec_pod.sh` (no Docker). The `run_euroc_test.sh` / `run_orbbec.sh`
  (with `docker run`) are for the LOCAL Docker path only.
- 22.04 uses GCC 11; `setup_slam_pod.sh` includes the common ORB-SLAM3 shims.
  If a specific source file errors, capture it and patch that file.
