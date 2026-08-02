"""
Generate an ORB-SLAM3 stereo-inertial settings YAML for the Orbbec Ego clip,
modeled EXACTLY on ORB-SLAM3's TUM-VI.yaml (the reference KB-fisheye stereo-
inertial config). Fills every field from our validated calibration.

Run on the pod (or locally) from stereo_3d/orbbec:
    python slam3/make_orbbec_settings.py \
        --data ../../stereo_data/shared \
        --out ../slam_data/orbbec_stereo_inertial.yaml

CONVENTIONS (documented, since these are the silent-failure risks):
  - Stereo.T_c1_c2 = pose of RIGHT cam (c2) in LEFT cam (c1) frame
      = right.[R_ref_cam | t_ref_cam]  (our triangulation already uses this).
  - IMU.T_b_c1 = pose of LEFT cam (c1) in IMU/body (b) frame = inv(T_cam_imu).
      Kalibr gives T_cam_imu (p_cam = T_cam_imu * p_imu), so cam-in-body is its
      inverse. If VIO init fails / gravity is wrong, try T_cam_imu directly
      (set --tbc-direct) — this is THE thing to flip during validation.
  - Translations converted to METERS. Provider stores mm.
  - IMU noise densities/walks + rate come straight from calibration_imu.yaml.
"""

import argparse
from pathlib import Path
import sys

import numpy as np

# calibration.py lives in the parent orbbec/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from calibration import load_stereo_calibration, load_imu_calibration


def mat_block(name, M):
    flat = ",\n          ".join(
        ", ".join(f"{v: .12g}" for v in row) for row in M
    )
    return (f"{name}: !!opencv-matrix\n"
            f"  rows: 4\n  cols: 4\n  dt: f\n"
            f"  data: [ {flat} ]\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=float, default=30.001336)
    p.add_argument("--nfeatures", type=int, default=1500,
                   help="more than TUM (1000): our images are 1600x1300")
    p.add_argument("--tbc-direct", action="store_true",
                   help="use T_cam_imu directly for IMU.T_b_c1 (default: inverse)")
    p.add_argument("--flip-stereo", action="store_true",
                   help="invert Stereo.T_c1_c2 (try if stereo init makes 0 points)")
    p.add_argument("--ini-fast", type=int, default=20,
                   help="ORB iniThFAST; lower (e.g. 10) for low-contrast ego video")
    p.add_argument("--min-fast", type=int, default=7,
                   help="ORB minThFAST; lower (e.g. 3) for low-contrast ego video")
    p.add_argument("--no-imu", action="store_true",
                   help="omit IMU fields (for stereo-only diagnostic run)")
    args = p.parse_args()

    data = Path(args.data)
    stereo = load_stereo_calibration(next(data.glob("*calibration_camera.yaml")))
    imu = load_imu_calibration(next(data.glob("*calibration_imu.yaml")))
    L, R = stereo.left, stereo.right

    # Stereo.T_c1_c2 : right cam pose in left frame (meters).
    T_c1_c2 = np.eye(4)
    T_c1_c2[:3, :3] = R.R_ref_cam
    T_c1_c2[:3, 3] = R.t_ref_cam  # already meters
    if args.flip_stereo:
        # ORB-SLAM3's T_c1_c2 convention/direction is ambiguous; if stereo
        # init produces ~0 points, the matrix is likely reversed -> invert it.
        T_c1_c2 = np.linalg.inv(T_c1_c2)

    # IMU.T_b_c1 : left cam pose in IMU/body frame. Provider T_cam_imu has mm
    # translation -> convert to meters, then invert (unless --tbc-direct).
    T_cam_imu = imu.T_cam_imu.copy()
    T_cam_imu[:3, 3] = T_cam_imu[:3, 3] / 1000.0
    T_b_c1 = T_cam_imu if args.tbc_direct else np.linalg.inv(T_cam_imu)

    W, H = L.width, L.height

    lines = []
    lines.append("%YAML:1.0")
    lines.append("")
    lines.append("# Orbbec Ego stereo-inertial — generated from provider")
    lines.append("# calibration, modeled on ORB-SLAM3 TUM-VI.yaml (KB fisheye).")
    lines.append('File.version: "1.0"')
    lines.append('Camera.type: "KannalaBrandt8"')
    lines.append("")
    lines.append(f"Camera1.fx: {L.fx:.10f}")
    lines.append(f"Camera1.fy: {L.fy:.10f}")
    lines.append(f"Camera1.cx: {L.cx:.10f}")
    lines.append(f"Camera1.cy: {L.cy:.10f}")
    lines.append(f"Camera1.k1: {L.D[0]:.12f}")
    lines.append(f"Camera1.k2: {L.D[1]:.12f}")
    lines.append(f"Camera1.k3: {L.D[2]:.12f}")
    lines.append(f"Camera1.k4: {L.D[3]:.12f}")
    lines.append("")
    lines.append(f"Camera2.fx: {R.fx:.10f}")
    lines.append(f"Camera2.fy: {R.fy:.10f}")
    lines.append(f"Camera2.cx: {R.cx:.10f}")
    lines.append(f"Camera2.cy: {R.cy:.10f}")
    lines.append(f"Camera2.k1: {R.D[0]:.12f}")
    lines.append(f"Camera2.k2: {R.D[1]:.12f}")
    lines.append(f"Camera2.k3: {R.D[2]:.12f}")
    lines.append(f"Camera2.k4: {R.D[3]:.12f}")
    lines.append("")
    lines.append(mat_block("Stereo.T_c1_c2", T_c1_c2))
    lines.append("# Stereo overlap columns. Full width as a starting point;")
    lines.append("# tighten if stereo matching is poor at the fisheye edges.")
    lines.append("Camera1.overlappingBegin: 0")
    lines.append(f"Camera1.overlappingEnd: {W - 1}")
    lines.append("Camera2.overlappingBegin: 0")
    lines.append(f"Camera2.overlappingEnd: {W - 1}")
    lines.append("")
    lines.append(f"Camera.width: {W}")
    lines.append(f"Camera.height: {H}")
    # ORB-SLAM3 requires an INTEGER fps. This is only a coarse hint (keyframe
    # heuristics); real frame timing comes from the microsecond timestamps we
    # feed, which stay exact. So rounding here does NOT affect sync integrity.
    lines.append(f"Camera.fps: {int(round(args.fps))}")
    lines.append("Camera.RGB: 0  # IR/gray")
    lines.append("Stereo.ThDepth: 40.0")
    lines.append("")
    if not args.no_imu:
        lines.append(mat_block("IMU.T_b_c1", T_b_c1))
        lines.append(f"IMU.NoiseGyro: {imu.gyro_noise_density:.10g}")
        lines.append(f"IMU.NoiseAcc: {imu.accel_noise_density:.10g}")
        lines.append(f"IMU.GyroWalk: {imu.gyro_random_walk:.10g}")
        lines.append(f"IMU.AccWalk: {imu.accel_random_walk:.10g}")
        lines.append(f"IMU.Frequency: {imu.update_rate_hz:.1f}")
        lines.append("")
    lines.append(f"ORBextractor.nFeatures: {args.nfeatures}")
    lines.append("ORBextractor.scaleFactor: 1.2")
    lines.append("ORBextractor.nLevels: 8")
    lines.append(f"ORBextractor.iniThFAST: {args.ini_fast}")
    lines.append(f"ORBextractor.minThFAST: {args.min_fast}")
    lines.append("")
    lines.append("Viewer.KeyFrameSize: 0.05")
    lines.append("Viewer.KeyFrameLineWidth: 1.0")
    lines.append("Viewer.GraphLineWidth: 0.9")
    lines.append("Viewer.PointSize: 2.0")
    lines.append("Viewer.CameraSize: 0.08")
    lines.append("Viewer.CameraLineWidth: 3.0")
    lines.append("Viewer.ViewpointX: 0.0")
    lines.append("Viewer.ViewpointY: -0.7")
    lines.append("Viewer.ViewpointZ: -3.5")
    lines.append("Viewer.ViewpointF: 500.0")

    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")
    print(f"  Stereo.T_c1_c2 t = {T_c1_c2[:3,3].round(5)} (baseline {np.linalg.norm(T_c1_c2[:3,3]):.5f} m){'  [FLIPPED]' if args.flip_stereo else ''}")
    print(f"  IMU.T_b_c1 mode = {'T_cam_imu DIRECT' if args.tbc_direct else 'inv(T_cam_imu)'}")
    print(f"  ORB: nFeatures={args.nfeatures} iniFAST={args.ini_fast} minFAST={args.min_fast}")
    print(f"  IMU noise: gyro {imu.gyro_noise_density:.3e} acc {imu.accel_noise_density:.3e} @ {imu.update_rate_hz}Hz")


if __name__ == "__main__":
    main()
