"""
Production calibration module for the Orbbec Ego stereo + IMU rig.
-------------------------------------------------------------------
Parses the provider's calibration YAMLs into validated, typed objects.
This is the foundation for all 3D geometry — correctness here is critical.

Coordinate conventions (from the provider's file header):
    Right-handed, OpenCV convention: X -> right, Y -> down, Z -> forward.
    Reference camera = cam_0 (IR_L). Right camera pose is given relative to it.
    Translations in the YAML are in MILLIMETERS; we convert to METERS here
    and keep everything in meters downstream.

Distortion model: Kannala-Brandt fisheye ("KB") with 4 coefficients (k1..k4),
matching cv2.fisheye.* expectations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

MM_TO_M = 1e-3


# ─────────────────────────────────────────────────────────────────────
# Single camera
# ─────────────────────────────────────────────────────────────────────
@dataclass
class CameraCalib:
    """Intrinsics + KB fisheye distortion + extrinsics for one camera.

    Extrinsics express the transform T_ref_from_cam:
        a point in THIS camera's frame -> the reference (left) camera frame.
    Reference camera (cam_0) has identity extrinsics.
    """
    cam_id: str
    name: str
    width: int
    height: int
    distortion_model: str
    K: np.ndarray            # (3,3) intrinsic matrix
    D: np.ndarray            # (4,) KB distortion coeffs [k1,k2,k3,k4]
    R_ref_cam: np.ndarray    # (3,3) rotation, cam -> reference
    t_ref_cam: np.ndarray    # (3,) translation in METERS, cam -> reference

    def __post_init__(self):
        self._validate()

    def _validate(self):
        assert self.K.shape == (3, 3), f"K must be 3x3, got {self.K.shape}"
        assert self.D.shape == (4,), f"KB D must be length-4, got {self.D.shape}"
        assert self.R_ref_cam.shape == (3, 3)
        assert self.t_ref_cam.shape == (3,)
        # Rotation matrix sanity: orthonormal, det ~ +1
        should_be_I = self.R_ref_cam @ self.R_ref_cam.T
        if not np.allclose(should_be_I, np.eye(3), atol=1e-3):
            raise ValueError(f"{self.name}: R is not orthonormal")
        det = np.linalg.det(self.R_ref_cam)
        if not np.isclose(det, 1.0, atol=1e-3):
            raise ValueError(f"{self.name}: det(R)={det:.4f}, expected ~1")

    @property
    def fx(self) -> float: return float(self.K[0, 0])
    @property
    def fy(self) -> float: return float(self.K[1, 1])
    @property
    def cx(self) -> float: return float(self.K[0, 2])
    @property
    def cy(self) -> float: return float(self.K[1, 2])

    def T_ref_cam(self) -> np.ndarray:
        """4x4 homogeneous transform: cam frame -> reference frame (meters)."""
        T = np.eye(4)
        T[:3, :3] = self.R_ref_cam
        T[:3, 3] = self.t_ref_cam
        return T


# ─────────────────────────────────────────────────────────────────────
# Stereo pair
# ─────────────────────────────────────────────────────────────────────
@dataclass
class StereoCalib:
    left: CameraCalib
    right: CameraCalib

    @property
    def baseline_m(self) -> float:
        """Distance between the two camera centers, in meters."""
        return float(np.linalg.norm(self.right.t_ref_cam - self.left.t_ref_cam))

    def R_left_right(self) -> np.ndarray:
        """Rotation from right camera frame into left (reference) camera frame."""
        # left is reference (identity); this is just right's rotation.
        return self.right.R_ref_cam

    def t_left_right(self) -> np.ndarray:
        """Translation of right camera origin in the left frame (meters)."""
        return self.right.t_ref_cam

    def projection_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Build 3x4 projection matrices for triangulation, in the LEFT
        (reference) camera frame, using NORMALIZED image coords.

        Because we triangulate UNDISTORTED-NORMALIZED points (see stereo3d),
        the intrinsics are folded out — P uses identity K and the relative
        pose between cameras.

            P_left  = [ I | 0 ]
            P_right = [ R_l_r^T | -R_l_r^T t_l_r ]   (world=left frame)

        We express the right camera's extrinsic as T_right_from_left =
        inverse of T_left_from_right.
        """
        # T_left_from_right : right-frame point -> left-frame point
        T_l_r = np.eye(4)
        T_l_r[:3, :3] = self.R_left_right()
        T_l_r[:3, 3] = self.t_left_right()
        # We need each camera's [R|t] that maps a LEFT-frame 3D point into
        # that camera's frame (projection convention).
        # Left camera IS the reference frame -> identity.
        P_left = np.hstack([np.eye(3), np.zeros((3, 1))])
        # Right camera: T_right_from_left = inv(T_left_from_right)
        T_r_l = np.linalg.inv(T_l_r)
        P_right = T_r_l[:3, :4]
        return P_left, P_right


# ─────────────────────────────────────────────────────────────────────
# IMU
# ─────────────────────────────────────────────────────────────────────
@dataclass
class ImuCalib:
    T_cam_imu: np.ndarray           # (4,4) IMU frame -> camera frame
    timeshift_cam_imu_s: float      # seconds; add to cam ts to align with imu
    update_rate_hz: float
    accel_noise_density: float
    accel_random_walk: float
    gyro_noise_density: float
    gyro_random_walk: float

    def __post_init__(self):
        assert self.T_cam_imu.shape == (4, 4)


# ─────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────
def _camera_from_dict(d: dict) -> CameraCalib:
    intr = d["intrinsics"]
    dist = d["distortion"]
    K = np.array([
        [intr["fx"], 0.0,        intr["cx"]],
        [0.0,        intr["fy"], intr["cy"]],
        [0.0,        0.0,        1.0],
    ], dtype=np.float64)
    D = np.array([dist["k1"], dist["k2"], dist["k3"], dist["k4"]], dtype=np.float64)

    ext = d["extrinsics"]
    R = np.array(ext["rotation"], dtype=np.float64)
    t_mm = np.array(ext["translation"], dtype=np.float64)
    t_m = t_mm * MM_TO_M  # provider stores millimeters

    return CameraCalib(
        cam_id=d["id"],
        name=d["name"],
        width=int(d["image_width"]),
        height=int(d["image_height"]),
        distortion_model=d["distortion_model"],
        K=K, D=D,
        R_ref_cam=R, t_ref_cam=t_m,
    )


def load_stereo_calibration(camera_yaml: str | Path) -> StereoCalib:
    """Parse the Orbbec camera calibration YAML into a StereoCalib."""
    camera_yaml = Path(camera_yaml)
    with open(camera_yaml) as f:
        raw = yaml.safe_load(f)

    cams = {c["name"]: _camera_from_dict(c) for c in raw["cameras"]}
    if "IR_L" not in cams or "IR_R" not in cams:
        raise ValueError(f"Expected IR_L and IR_R cameras, found {list(cams.keys())}")

    stereo = StereoCalib(left=cams["IR_L"], right=cams["IR_R"])

    # Sanity: baseline should be ~0.12 m for the Orbbec Ego
    b = stereo.baseline_m
    if not (0.05 < b < 0.30):
        raise ValueError(f"Suspicious stereo baseline: {b:.4f} m")

    return stereo


def load_imu_calibration(imu_yaml: str | Path) -> ImuCalib:
    """Parse the IMU calibration YAML (OpenCV %YAML:1.0 header handled)."""
    imu_yaml = Path(imu_yaml)
    text = imu_yaml.read_text()
    # Strip the OpenCV-style header line that PyYAML can't parse.
    text = "\n".join(
        ln for ln in text.splitlines()
        if not ln.strip().startswith("%YAML")
    )
    raw = yaml.safe_load(text)

    cam0 = raw["cam0"]
    imu0 = raw["imu0"]

    T_cam_imu = np.array(cam0["T_cam_imu"], dtype=np.float64)
    accel = imu0["accelerometer"]
    gyro = imu0["gyroscope"]

    return ImuCalib(
        T_cam_imu=T_cam_imu,
        timeshift_cam_imu_s=float(cam0.get("timeshift_cam_imu", 0.0)),
        update_rate_hz=float(imu0.get("update_rate", 0.0)),
        accel_noise_density=float(accel["noise_density"]),
        accel_random_walk=float(accel["random_walk"]),
        gyro_noise_density=float(gyro["noise_density"]),
        gyro_random_walk=float(gyro["random_walk"]),
    )


# ─────────────────────────────────────────────────────────────────────
# Self-test / summary
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Provider data folder")
    args = parser.parse_args()

    data = Path(args.data)
    cam_yaml = next(data.glob("*calibration_camera.yaml"))
    imu_yaml = next(data.glob("*calibration_imu.yaml"))

    stereo = load_stereo_calibration(cam_yaml)
    imu = load_imu_calibration(imu_yaml)

    print("=" * 60)
    print("STEREO CALIBRATION")
    print("=" * 60)
    for cam in (stereo.left, stereo.right):
        print(f"  {cam.name} ({cam.cam_id}): {cam.width}x{cam.height} | {cam.distortion_model}")
        print(f"     fx={cam.fx:.2f} fy={cam.fy:.2f} cx={cam.cx:.2f} cy={cam.cy:.2f}")
        print(f"     D(KB)={cam.D}")
        print(f"     t_ref_cam(m)={cam.t_ref_cam}")
    print(f"  Baseline: {stereo.baseline_m*1000:.2f} mm  ({stereo.baseline_m:.4f} m)")

    P_l, P_r = stereo.projection_matrices()
    print(f"  P_left=\n{P_l}")
    print(f"  P_right=\n{P_r.round(4)}")

    print()
    print("=" * 60)
    print("IMU CALIBRATION")
    print("=" * 60)
    print(f"  update_rate: {imu.update_rate_hz} Hz")
    print(f"  timeshift_cam_imu: {imu.timeshift_cam_imu_s*1000:.3f} ms")
    print(f"  accel noise_density={imu.accel_noise_density:.2e} random_walk={imu.accel_random_walk:.2e}")
    print(f"  gyro  noise_density={imu.gyro_noise_density:.2e} random_walk={imu.gyro_random_walk:.2e}")
    print(f"  T_cam_imu=\n{imu.T_cam_imu.round(4)}")
    print()
    print("✅ Calibration parsed and validated.")
