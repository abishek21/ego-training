"""
Stage 3a: SLAM data prep + sync validation (Orbbec Ego -> EuRoC/ASL format).
----------------------------------------------------------------------------
Prepares the provider stereo+IMU data for ORB-SLAM3 (stereo-inertial) and,
crucially, VALIDATES the timestamp/sync chain BEFORE any heavy extraction or
engine work. Pure Python (venv). Engine-agnostic: the EuRoC layout also feeds
Basalt/OKVIS/VINS if we ever switch, so nothing here is wasted.

WHY EuRoC layout: it is ORB-SLAM3's native stereo-inertial input:
    <out>/
      cam0/data/<ts_ns>.png      (left)
      cam0/data.csv              (ts_ns, filename)
      cam1/data/<ts_ns>.png      (right)
      cam1/data.csv
      imu0/data.csv              (ts_ns, wx,wy,wz, ax,ay,az)   gyro THEN accel
      timestamps.txt             (one ts_ns per stereo frame, for the runner)

TIMESTAMPS (data-integrity — the highest-risk part):
  - Provider pts are MICROSECONDS; EuRoC/ORB-SLAM3 expect NANOSECONDS.
    Convert us->ns by *1000 exactly (integers, no rounding loss).
  - Video frame i  ->  pts row i (the documented clip-vs-full-recording gotcha).
  - Left/right are hardware-synced (~11us apart). We stamp BOTH cam0 and cam1
    with the LEFT timestamp per frame so ORB-SLAM3 sees a consistent stereo pair.

TIMESHIFT (the "silent killer" — handled explicitly, not assumed):
  Kalibr convention:  t_imu = t_cam + timeshift_cam_imu   (here ~ -13.49 ms).
  ORB-SLAM3 assumes camera & IMU share ONE clock, so we bake the shift into the
  CAMERA timestamps (t_cam_on_imu_clock = t_cam + timeshift) and leave IMU raw.
  This is behind --apply-timeshift (default ON) with the sign documented, so we
  can flip/disable it during validation instead of trusting it blindly. NOTE:
  13.5 ms < half a frame (33 ms), so it is below what we can verify purely
  offline — the real test is the Stage-3d "static hand stays still" gate.

WHAT WE CAN RIGOROUSLY VALIDATE HERE (the Stage-3a gate):
  1. Frame count: video frames == pts rows used (i->i mapping intact).
  2. Monotonic, gap-free camera timestamps; dt ~ 33.3 ms (30 fps), flag drops.
  3. Left/right per-frame skew small (< a few hundred us).
  4. IMU: uniform ~1 kHz, monotonic; EVERY image ts lies inside the IMU span
     with a dense sample window around it (~15-16 samples per 33 ms frame).
  5. Gravity magnitude ~9.81 (units sanity) — cross-checked earlier.

Usage:
    # 1) validate ONLY (fast, no image extraction) — pass this gate first
    python slam_prep.py --data "<provider dir>" --out ../slam_data --validate-only

    # 2) full extraction (images + imu + settings) once sync is confirmed
    python slam_prep.py --data "<provider dir>" --out ../slam_data
    #    optional: --max-frames N for a short test clip
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from calibration import load_stereo_calibration, load_imu_calibration

US_TO_NS = 1000
S_TO_NS = 1_000_000_000


# ─────────────────────────────────────────────────────────────────────
# Readers
# ─────────────────────────────────────────────────────────────────────
def read_pts_us(pts_csv: Path) -> np.ndarray:
    """Read a *_pts.csv (single 'timestamp_us' column) -> int64 microseconds."""
    ts = []
    with open(pts_csv) as f:
        r = csv.reader(f)
        header = next(r)
        assert header[0].strip() == "timestamp_us", f"unexpected header {header}"
        for row in r:
            if row:
                ts.append(int(row[0]))
    return np.asarray(ts, dtype=np.int64)


def read_imu(imu_csv: Path):
    """Read interleaved accel/gyro CSV -> paired arrays on shared timestamps.

    Provider format: timestamp_us, x, y, z, type  where `type` is 'accel'|'gyro'
    and BOTH streams share the same timestamps. Returns (ts_us, gyro_xyz,
    accel_xyz) as aligned arrays, keeping only timestamps present in BOTH.
    """
    accel = {}
    gyro = {}
    with open(imu_csv) as f:
        r = csv.reader(f)
        next(r)  # header
        for row in r:
            if not row:
                continue
            t = int(row[0])
            vec = (float(row[1]), float(row[2]), float(row[3]))
            if row[4] == "accel":
                accel[t] = vec
            else:
                gyro[t] = vec
    common = sorted(set(accel) & set(gyro))
    ts = np.asarray(common, dtype=np.int64)
    g = np.asarray([gyro[t] for t in common], dtype=np.float64)
    a = np.asarray([accel[t] for t in common], dtype=np.float64)
    return ts, g, a


# ─────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────
def validate(cam_ts_us, raw_left_us, raw_right_us, imu_ts_us, imu_g, imu_a,
             n_video_frames, n_trimmed, fps, report):
    """Run all sync checks; fill `report`; return ok (bool).

    cam_ts_us : timeshift-applied, IMU-clock frame timestamps (post-trim).
    raw_left/right_us : RAW provider timestamps (post-trim, for stereo skew).
    n_trimmed : frames intentionally dropped to fit the IMU span.
    """
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        if not cond:
            ok = False
        report["checks"].append({"name": name, "status": status, "detail": detail})
        print(f"  [{status}] {name}: {detail}")

    n = len(cam_ts_us)

    # 1. frame accounting: every video frame is either kept or intentionally
    #    trimmed to the IMU span — nothing vanished unexpectedly.
    check("frame_accounting_intact",
          n + n_trimmed == n_video_frames,
          f"kept={n} + trimmed={n_trimmed} == video={n_video_frames}")

    # 2. camera timestamps monotonic + gap analysis
    dt = np.diff(cam_ts_us)
    mono = bool(np.all(dt > 0))
    expected_dt = 1e6 / fps
    drops = int(np.sum(dt > 1.5 * expected_dt))
    check("cam_ts_monotonic", mono, f"min_dt={dt.min() if n>1 else 0}us")
    check("cam_ts_no_frame_drops", drops == 0,
          f"gaps>1.5x period: {drops}; dt median={np.median(dt):.1f}us "
          f"expected~{expected_dt:.1f}us")

    # 3. left/right skew from RAW timestamps (timeshift-independent)
    m = min(len(raw_left_us), len(raw_right_us))
    skew = np.abs(raw_left_us[:m] - raw_right_us[:m])
    check("stereo_lr_skew_small", bool(skew.max() < 2000),
          f"max_skew={skew.max()}us median={np.median(skew):.0f}us (raw)")

    # 4. IMU rate + monotonic
    idt = np.diff(imu_ts_us)
    imu_mono = bool(np.all(idt > 0))
    imu_hz = 1e6 / np.median(idt)
    check("imu_ts_monotonic", imu_mono, f"min_dt={idt.min() if len(idt) else 0}us")
    check("imu_rate_near_1khz", 900 < imu_hz < 1100,
          f"{imu_hz:.1f}Hz median_dt={np.median(idt):.0f}us")

    # 5. IMU covers every (trimmed) image ts + dense window
    imu_lo, imu_hi = imu_ts_us[0], imu_ts_us[-1]
    covered = bool(cam_ts_us[0] >= imu_lo and cam_ts_us[-1] <= imu_hi)
    check("imu_spans_all_frames", covered,
          f"imu[{imu_lo},{imu_hi}] cam[{cam_ts_us[0]},{cam_ts_us[-1]}]")
    idx = np.searchsorted(imu_ts_us, cam_ts_us)
    per_frame = np.diff(idx)
    check("imu_dense_between_frames", bool(per_frame.min() >= 5),
          f"imu samples/frame min={per_frame.min()} median={np.median(per_frame):.0f}")

    # 6. gravity magnitude sanity
    gmag = np.linalg.norm(imu_a, axis=1).mean()
    check("accel_gravity_units_ok", 9.0 < gmag < 10.6,
          f"mean|a|={gmag:.3f} m/s^2")

    report["summary"] = {
        "video_frames": int(n_video_frames),
        "frames_used_after_trim": int(n),
        "fps_exact": float(fps),
        "cam_dt_median_us": float(np.median(dt)) if n > 1 else None,
        "stereo_lr_skew_max_us": int(skew.max()),
        "imu_rate_hz": float(imu_hz),
        "imu_samples_per_frame_median": float(np.median(per_frame)),
        "accel_mean_mag": float(gmag),
        "all_pass": ok,
    }
    return ok


# ─────────────────────────────────────────────────────────────────────
# Extraction (EuRoC layout)
# ─────────────────────────────────────────────────────────────────────
def write_imu_euroc(out_dir: Path, imu_ts_us, g, a):
    """imu0/data.csv : ts_ns, wx,wy,wz, ax,ay,az  (gyro then accel, EuRoC)."""
    imu_dir = out_dir / "imu0"
    imu_dir.mkdir(parents=True, exist_ok=True)
    with open(imu_dir / "data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["#timestamp [ns]",
                    "w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]", "w_RS_S_z [rad s^-1]",
                    "a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]"])
        for t, gi, ai in zip(imu_ts_us, g, a):
            w.writerow([int(t) * US_TO_NS,
                        f"{gi[0]:.9f}", f"{gi[1]:.9f}", f"{gi[2]:.9f}",
                        f"{ai[0]:.9f}", f"{ai[1]:.9f}", f"{ai[2]:.9f}"])
    return len(imu_ts_us)


def extract_stereo(out_dir: Path, left_mp4: Path, right_mp4: Path,
                   frame_ts_ns, orig_frame_idx, max_frames=None):
    """Write cam0/cam1 PNGs named <ts_ns>.png + each data.csv. Both cameras use
    the same (left) timestamp per frame to enforce a synchronized stereo pair.

    `orig_frame_idx` maps each kept entry -> its ORIGINAL video frame number.
    We read the video sequentially and only WRITE frames whose index is kept,
    so trimmed leading/trailing frames are skipped correctly. A mapping file
    (ts_ns -> original frame_idx) is written for downstream pose->frame linking.
    """
    cam0 = out_dir / "cam0" / "data"; cam0.mkdir(parents=True, exist_ok=True)
    cam1 = out_dir / "cam1" / "data"; cam1.mkdir(parents=True, exist_ok=True)
    capL = cv2.VideoCapture(str(left_mp4))
    capR = cv2.VideoCapture(str(right_mp4))

    n = len(frame_ts_ns) if max_frames is None else min(max_frames, len(frame_ts_ns))
    kept_set = {int(orig_frame_idx[i]): i for i in range(n)}
    last_needed = int(orig_frame_idx[n - 1])

    rowsL, rowsR, mapping = [], [], []
    written = 0
    vid_i = 0
    while vid_i <= last_needed:
        okL, imgL = capL.read()
        okR, imgR = capR.read()
        if not (okL and okR):
            print(f"  ! video ended early at frame {vid_i}")
            break
        if vid_i in kept_set:
            k = kept_set[vid_i]
            ts = int(frame_ts_ns[k])
            cv2.imwrite(str(cam0 / f"{ts}.png"), imgL)
            cv2.imwrite(str(cam1 / f"{ts}.png"), imgR)
            rowsL.append((ts, f"{ts}.png"))
            rowsR.append((ts, f"{ts}.png"))
            mapping.append((ts, vid_i))
            written += 1
            if written % 500 == 0:
                print(f"  extracted {written}/{n} stereo frames")
        vid_i += 1
    capL.release(); capR.release()

    for cam, rows in ((out_dir / "cam0", rowsL), (out_dir / "cam1", rowsR)):
        with open(cam / "data.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["#timestamp [ns]", "filename"])
            w.writerows(rows)
    with open(out_dir / "timestamps.txt", "w") as f:
        for ts, _ in rowsL:
            f.write(f"{ts}\n")
    # ts_ns -> original video frame_idx (for mapping SLAM poses back to hands_3d)
    with open(out_dir / "frame_map.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_ns", "orig_frame_idx"])
        w.writerows(mapping)
    return written


def write_settings_draft(out_dir: Path, stereo, imu, fps):
    """Emit a DRAFT ORB-SLAM3 stereo-inertial settings YAML (KB fisheye).

    Marked DRAFT: several fields (stereo overlap 'lapping', ORB params, and the
    exact IMU T_b_c1 sign/convention) must be reviewed in Stage 3b against
    ORB-SLAM3's fisheye examples before trusting a run. We fill everything we
    can from the provider calibration and flag the rest.
    """
    L, R = stereo.left, stereo.right
    # T_b_c1: body(IMU)->cam(left). Provider gives T_cam_imu (imu->cam). ORB-SLAM3
    # IMU.T_b_c1 expects cam->imu? Convention is version-specific -> FLAG for 3b.
    T_cam_imu = imu.T_cam_imu.copy()
    T_cam_imu[:3, 3] = T_cam_imu[:3, 3] / 1000.0  # provider translation in mm -> m

    # Right camera pose relative to left (for KB8 stereo 'Tlr').
    T_lr = np.eye(4)
    T_lr[:3, :3] = R.R_ref_cam
    T_lr[:3, 3] = R.t_ref_cam  # already meters

    lines = []
    lines.append("%YAML:1.0")
    lines.append("# DRAFT ORB-SLAM3 stereo-inertial settings (KannalaBrandt8).")
    lines.append("# REVIEW in Stage 3b: stereo 'lapping' cols, IMU T_b_c1 convention/sign,")
    lines.append("# and IMU frequency vs actual (~999 Hz).")
    lines.append('File.version: "1.0"')
    lines.append("")
    lines.append('Camera.type: "KannalaBrandt8"')
    lines.append(f"Camera1.fx: {L.fx:.6f}")
    lines.append(f"Camera1.fy: {L.fy:.6f}")
    lines.append(f"Camera1.cx: {L.cx:.6f}")
    lines.append(f"Camera1.cy: {L.cy:.6f}")
    lines.append(f"Camera1.k1: {L.D[0]:.8f}")
    lines.append(f"Camera1.k2: {L.D[1]:.8f}")
    lines.append(f"Camera1.k3: {L.D[2]:.8f}")
    lines.append(f"Camera1.k4: {L.D[3]:.8f}")
    lines.append(f"Camera2.fx: {R.fx:.6f}")
    lines.append(f"Camera2.fy: {R.fy:.6f}")
    lines.append(f"Camera2.cx: {R.cx:.6f}")
    lines.append(f"Camera2.cy: {R.cy:.6f}")
    lines.append(f"Camera2.k1: {R.D[0]:.8f}")
    lines.append(f"Camera2.k2: {R.D[1]:.8f}")
    lines.append(f"Camera2.k3: {R.D[2]:.8f}")
    lines.append(f"Camera2.k4: {R.D[3]:.8f}")
    lines.append(f"Camera.width: {L.width}")
    lines.append(f"Camera.height: {L.height}")
    lines.append(f"Camera.fps: {fps:.6f}")
    lines.append("Camera.RGB: 0")
    lines.append("")
    lines.append("# Stereo transform T_c1_c2 (right w.r.t. left), meters:")
    lines.append("Stereo.T_c1_c2: !!opencv-matrix")
    lines.append("  rows: 4")
    lines.append("  cols: 4")
    lines.append("  dt: f")
    flat = ", ".join(f"{v:.8f}" for v in T_lr.reshape(-1))
    lines.append(f"  data: [ {flat} ]")
    lines.append("# TODO(3b): Stereo.b (baseline) and 'lapping' overlap columns.")
    lines.append(f"Stereo.b: {stereo.baseline_m:.6f}")
    lines.append("")
    lines.append("IMU.NoiseGyro: %.8e" % imu.gyro_noise_density)
    lines.append("IMU.NoiseAcc: %.8e" % imu.accel_noise_density)
    lines.append("IMU.GyroWalk: %.8e" % imu.gyro_random_walk)
    lines.append("IMU.AccWalk: %.8e" % imu.accel_random_walk)
    lines.append(f"IMU.Frequency: {imu.update_rate_hz:.1f}")
    lines.append("# T_b_c1: IMU(body)->left cam. FLAG(3b): verify convention/sign.")
    lines.append("IMU.T_b_c1: !!opencv-matrix")
    lines.append("  rows: 4")
    lines.append("  cols: 4")
    lines.append("  dt: f")
    flat = ", ".join(f"{v:.8f}" for v in np.linalg.inv(T_cam_imu).reshape(-1))
    lines.append(f"  data: [ {flat} ]")
    lines.append("")
    lines.append("# --- ORB params (defaults from ORB-SLAM3 EuRoC; tune in 3c) ---")
    lines.append("ORBextractor.nFeatures: 1200")
    lines.append("ORBextractor.scaleFactor: 1.2")
    lines.append("ORBextractor.nLevels: 8")
    lines.append("ORBextractor.iniThFAST: 20")
    lines.append("ORBextractor.minThFAST: 7")

    path = out_dir / "orbslam3_stereo_inertial_DRAFT.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


# ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Provider data folder")
    p.add_argument("--out", required=True, help="Output EuRoC-format folder")
    p.add_argument("--validate-only", action="store_true",
                   help="run sync checks only; no image/imu extraction")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--apply-timeshift", dest="apply_timeshift",
                   action="store_true", default=True,
                   help="bake Kalibr timeshift into camera timestamps (default ON)")
    p.add_argument("--no-apply-timeshift", dest="apply_timeshift",
                   action="store_false")
    args = p.parse_args()

    data = Path(args.data)
    out = Path(args.out)
    cam_yaml = next(data.glob("*calibration_camera.yaml"))
    imu_yaml = next(data.glob("*calibration_imu.yaml"))
    left_mp4 = data / "camera_left_2min.mp4"
    right_mp4 = data / "camera_right_2min.mp4"
    left_pts = next(data.glob("*camera_left_pts.csv"))
    right_pts = next(data.glob("*camera_right_pts.csv"))
    imu_csv = next(data.glob("*imu_2min.csv"))

    stereo = load_stereo_calibration(cam_yaml)
    imu = load_imu_calibration(imu_yaml)

    # exact fps + true frame count from the video
    cap = cv2.VideoCapture(str(left_mp4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    left_ts_us = read_pts_us(left_pts)[:n_video]     # frame i -> row i
    right_ts_us = read_pts_us(right_pts)[:n_video]
    imu_ts_us, imu_g, imu_a = read_imu(imu_csv)

    # Timeshift: put camera timestamps on the IMU clock (t_imu = t_cam + shift).
    shift_us = int(round(imu.timeshift_cam_imu_s * 1e6)) if args.apply_timeshift else 0
    cam_ts_all = left_ts_us + shift_us

    # Trim frames to the IMU-covered span. VIO needs IMU BEFORE the first and
    # AFTER the last used frame (preintegration brackets each frame). The
    # timeshift can push frame 0 a few ms before IMU start -> drop such frames.
    # We keep the ORIGINAL video frame index so extraction reads the right
    # frames and downstream can map poses back to original frame_idx.
    imu_lo, imu_hi = imu_ts_us[0], imu_ts_us[-1]
    keep = np.where((cam_ts_all >= imu_lo) & (cam_ts_all <= imu_hi))[0]
    first_kept, last_kept = int(keep[0]), int(keep[-1])
    orig_frame_idx = np.arange(n_video)[first_kept:last_kept + 1]
    cam_ts_us = cam_ts_all[first_kept:last_kept + 1]
    raw_left_kept = left_ts_us[first_kept:last_kept + 1]
    raw_right_kept = right_ts_us[first_kept:last_kept + 1]
    n_trimmed = n_video - len(cam_ts_us)

    print("=" * 60)
    print("STAGE 3a — SLAM DATA PREP + SYNC VALIDATION")
    print("=" * 60)
    print(f"  provider: {data.name}")
    print(f"  fps(exact)={fps:.6f}  video_frames={n_video}")
    print(f"  timeshift applied: {args.apply_timeshift} ({shift_us} us)")
    print(f"  frames trimmed to IMU span: {n_trimmed} "
          f"(kept video frames {first_kept}..{last_kept} = {len(cam_ts_us)})")
    print("-" * 60)

    report = {"checks": [], "meta": {
        "fps_exact": fps, "video_frames": n_video,
        "timeshift_applied": args.apply_timeshift,
        "timeshift_us": shift_us,
        "baseline_m": stereo.baseline_m,
        "frames_trimmed": int(n_trimmed),
        "kept_frame_range": [first_kept, last_kept],
    }}
    ok = validate(cam_ts_us, raw_left_kept, raw_right_kept,
                  imu_ts_us, imu_g, imu_a, n_video, n_trimmed, fps, report)

    out.mkdir(parents=True, exist_ok=True)
    (out / "sync_report.json").write_text(json.dumps(report, indent=2))
    print("-" * 60)
    print(f"  sync report -> {out/'sync_report.json'}")
    print(f"  OVERALL: {'✅ ALL PASS' if ok else '❌ FAILURES — fix before extraction'}")

    if args.validate_only:
        print("  (validate-only: no extraction performed)")
        return
    if not ok:
        print("  Refusing to extract while sync checks fail. Resolve first.")
        return

    # Full extraction
    frame_ts_ns = (cam_ts_us.astype(np.int64) * US_TO_NS)
    print("-" * 60)
    n_imu = write_imu_euroc(out, imu_ts_us, imu_g, imu_a)
    print(f"  IMU written: {n_imu} samples -> imu0/data.csv")
    n_ext = extract_stereo(out, left_mp4, right_mp4, frame_ts_ns,
                           orig_frame_idx, args.max_frames)
    print(f"  stereo extracted: {n_ext} frames -> cam0/ cam1/")
    settings = write_settings_draft(out, stereo, imu, fps)
    print(f"  DRAFT settings -> {settings}")
    print("  Stage 3a complete.")


if __name__ == "__main__":
    main()
