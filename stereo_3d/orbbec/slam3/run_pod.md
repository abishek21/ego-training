# ORB-SLAM3 on a RunPod pod — run guide (Stage 3b/3c)

Pod: `runpod/pytorch:2.4.0-...-ubuntu22.04`, x86_64, **50 GB RAM / 9 vCPU / 40 GB disk**.
GPU unused (SLAM is CPU-bound). Big RAM removes the local 6 GB OOM problem.

> ⚠️ **Disk is the tight resource (40 GB).** The base image is ~15 GB. Extract
> stereo frames on the pod (~14 GB) and delete the EuRoC test data right after
> the gate. Consider `--gray` extraction to roughly halve frame size (TODO flag).

## 0. Code + data onto the pod
```bash
# On the pod: get the CODE via git (data is gitignored, comes separately).
cd /workspace
git clone https://github.com/abishek21/ego-training.git
cd ego-training

# From LOCAL machine: scp up ONLY the small raw provider files (~few hundred MB).
# (Do NOT upload slam_data/ — we regenerate it on the pod.)
DATA="stereo_data/Stereo Video with IMU (shared)"
scp -P <PORT> \
  "$DATA/camera_left_2min.mp4" \
  "$DATA/camera_right_2min.mp4" \
  "$DATA"/*calibration_camera.yaml \
  "$DATA"/*calibration_imu.yaml \
  "$DATA"/*camera_left_pts.csv \
  "$DATA"/*camera_right_pts.csv \
  "$DATA"/*imu_2min.csv \
  root@<POD_IP>:/workspace/ego-training/stereo_data/shared/
```

## 1. Build ORB-SLAM3 (one-time, ~15-20 min at -j9)
```bash
cd /workspace/ego-training/stereo_3d/orbbec/slam3
bash setup_slam_pod.sh
```

## 2. Python env for slam_prep.py
```bash
cd /workspace/ego-training
python3 -m venv venv && . venv/bin/activate
pip install numpy opencv-python pyyaml
```

## 3. Regenerate slam_data/ ON THE POD (extract + sync-validate)
```bash
cd stereo_3d/orbbec
python slam_prep.py --data ../../stereo_data/shared --out ../slam_data --validate-only  # gate
python slam_prep.py --data ../../stereo_data/shared --out ../slam_data                  # full
```

## 4. GATE — prove the build on public EuRoC (then delete to free disk)
```bash
cd slam3
bash run_euroc_test.sh        # must print a multi-row trajectory
rm -rf euroc_MH01             # free ~4 GB before/after
```

## 5. Review DRAFT settings, then run on our clip
Edit `../slam_data/orbslam3_stereo_inertial_DRAFT.yaml` — verify the flagged
`IMU.T_b_c1` convention/sign and stereo `lapping` cols against ORB-SLAM3's
reference fisheye YAML (`/workspace/ORB_SLAM3/Examples/Stereo-Inertial/*.yaml`).
```bash
bash run_orbbec.sh            # -> ../../slam_out/f_orbbec_traj.txt
```

## 6. Pull the trajectory back (tiny), leave the 14 GB PNGs on the pod
```bash
# From LOCAL:
scp -P <PORT> root@<POD_IP>:/workspace/ego-training/stereo_3d/slam_out/f_orbbec_traj.txt \
  stereo_3d/slam_out/
```

## Notes
- The pod's `run_euroc_test.sh` / `run_orbbec.sh` currently call `docker run`.
  On the pod we run the binary DIRECTLY (no Docker) — see the native commands in
  those scripts' comments, or invoke `stereo_inertial_euroc` directly. (TODO:
  add a `--native` path; for now use the direct binary calls below.)
- Direct EuRoC run (native):
  ```bash
  cd /workspace/ORB_SLAM3
  ./Examples/Stereo-Inertial/stereo_inertial_euroc \
    Vocabulary/ORBvoc.txt Examples/Stereo-Inertial/EuRoC.yaml \
    <euroc_dir> <euroc_dir>/MH01.txt euroc_test
  ```
- Direct Orbbec run (native): point the same binary at `stereo_3d/slam_data`
  (as `<seq>/mav0`) + `timestamps.txt` + our settings YAML.
