"""
RGB-D Scene Reconstruction from RealSense D435i bag
-----------------------------------------------------
Pipeline:
  1. Read RGB + Depth frames from .bag (intrinsics embedded)
  2. Estimate camera poses via RGB-D odometry (frame-to-frame)
  3. Fuse depth frames into a TSDF volume
  4. Extract a 3D mesh + colored point cloud

Output: reconstructed mesh (.ply) + point cloud (.ply)

Usage:
    python reconstruct.py --bag /path/to/d435i_walk_around.bag --output out/
"""

import argparse
import numpy as np
import open3d as o3d
from pathlib import Path


def reconstruct(bag_path: str, output_dir: str, max_frames: int = None,
                frame_skip: int = 1):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Read bag ──────────────────────────────────────────────
    print(f"📂 Opening bag: {bag_path}")
    reader = o3d.t.io.RSBagReader()
    reader.open(bag_path)
    meta = reader.metadata

    W, H = meta.width, meta.height
    fps = meta.fps
    depth_scale = meta.depth_scale
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        W, H,
        meta.intrinsics.intrinsic_matrix[0, 0],  # fx
        meta.intrinsics.intrinsic_matrix[1, 1],  # fy
        meta.intrinsics.intrinsic_matrix[0, 2],  # cx
        meta.intrinsics.intrinsic_matrix[1, 2],  # cy
    )

    print(f"   Device: {meta.device_name}")
    print(f"   {W}x{H} @ {fps}fps | depth_scale={depth_scale:.1f}")
    print()

    # ── TSDF volume for fusion ────────────────────────────────
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=4.0 / 512.0,   # ~7.8mm voxels
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    # ── Odometry setup ────────────────────────────────────────
    odo_option = o3d.pipelines.odometry.OdometryOption()
    prev_rgbd = None
    cur_pose = np.identity(4)  # global camera pose (world = first frame)

    print("🚀 Processing frames (odometry + TSDF fusion)...")
    frame_num = 0
    used = 0

    im = reader.next_frame()
    while not reader.is_eof():
        frame_num += 1
        if max_frames and frame_num > max_frames:
            break
        if frame_num % frame_skip != 0:
            im = reader.next_frame()
            continue

        # Convert to legacy Open3D images
        color_np = im.color.as_tensor().numpy()          # HxWx3 uint8
        depth_np = im.depth.as_tensor().numpy().squeeze()  # HxW uint16

        color = o3d.geometry.Image(np.ascontiguousarray(color_np))
        depth = o3d.geometry.Image(np.ascontiguousarray(depth_np))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth,
            depth_scale=depth_scale,
            depth_trunc=4.0,
            convert_rgb_to_intensity=False,
        )

        # Estimate relative pose vs previous frame
        if prev_rgbd is not None:
            success, trans, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                rgbd, prev_rgbd, intrinsic, np.identity(4),
                o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                odo_option,
            )
            if success:
                cur_pose = cur_pose @ trans
            # if odometry fails, keep previous pose (skip integration jump)

        # Integrate into TSDF at the current global pose
        volume.integrate(rgbd, intrinsic, np.linalg.inv(cur_pose))

        prev_rgbd = rgbd
        used += 1

        if used % 30 == 0:
            print(f"   Integrated {used} frames (video frame {frame_num})")

        im = reader.next_frame()

    reader.close()

    print(f"\n✅ Fused {used} frames")

    # ── Extract mesh ──────────────────────────────────────────
    print("🔨 Extracting mesh...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    mesh_path = output_dir / "scene_mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    print(f"   Saved mesh: {mesh_path} ({len(mesh.vertices)} verts, {len(mesh.triangles)} tris)")

    # ── Extract point cloud ───────────────────────────────────
    print("☁️  Extracting point cloud...")
    pcd = volume.extract_point_cloud()
    pcd_path = output_dir / "scene_pointcloud.ply"
    o3d.io.write_point_cloud(str(pcd_path), pcd)
    print(f"   Saved point cloud: {pcd_path} ({len(pcd.points)} points)")

    print()
    print("=" * 60)
    print("✅ RECONSTRUCTION COMPLETE")
    print("=" * 60)
    print(f"   Mesh:        {mesh_path}")
    print(f"   Point cloud: {pcd_path}")
    print(f"   View with:   python view.py --file {mesh_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, help="Path to RealSense .bag")
    parser.add_argument("--output", default="rgbd_output")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit frames for testing")
    parser.add_argument("--frame-skip", type=int, default=1,
                        help="Process every Nth frame")
    args = parser.parse_args()

    reconstruct(args.bag, args.output, args.max_frames, args.frame_skip)
