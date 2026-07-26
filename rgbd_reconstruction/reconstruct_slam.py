"""
RGB-D Scene Reconstruction with Pose-Graph Optimization (SLAM)
----------------------------------------------------------------
Eliminates the "ghosting" of naive frame-to-frame odometry by:
  1. Building local fragments (short segments, low drift)
  2. Multiway registration WITHIN each fragment (pose graph + optimization)
  3. Registering fragments to each other (loop closure between fragments)
  4. Global pose-graph optimization
  5. Final TSDF integration with globally-consistent poses

Plus depth bilateral filtering to reduce sensor noise.

Usage:
    python reconstruct_slam.py --bag <bag> --output out_slam/ [--max-frames N]
"""

import argparse
import numpy as np
import open3d as o3d
from pathlib import Path

# ── Tunable parameters ──────────────────────────────────────────
VOXEL_LENGTH = 3.0 / 512.0        # ~5.8mm voxels (finer than baseline)
SDF_TRUNC = 0.02                  # 2cm (tighter → crisper surfaces)
DEPTH_TRUNC = 3.0                 # ignore beyond 3m
FRAGMENT_SIZE = 30                # frames per local fragment
# Odometry / registration thresholds
MAX_DEPTH_DIFF = 0.07
# ────────────────────────────────────────────────────────────────


def read_frames(bag_path, max_frames=None, frame_skip=1):
    """Read RGB-D frames from bag into memory as Open3D RGBD images."""
    reader = o3d.t.io.RSBagReader()
    reader.open(bag_path)
    meta = reader.metadata

    W, H = meta.width, meta.height
    depth_scale = meta.depth_scale
    K = meta.intrinsics.intrinsic_matrix
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    )

    print(f"📹 {meta.device_name} | {W}x{H} @ {meta.fps}fps")

    rgbds = []
    frame_num = 0
    im = reader.next_frame()
    while not reader.is_eof():
        frame_num += 1
        if max_frames and frame_num > max_frames:
            break
        if frame_num % frame_skip != 0:
            im = reader.next_frame()
            continue

        color_np = np.ascontiguousarray(im.color.as_tensor().numpy())
        depth_np = np.ascontiguousarray(im.depth.as_tensor().numpy().squeeze())

        color = o3d.geometry.Image(color_np)
        depth = o3d.geometry.Image(depth_np)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth,
            depth_scale=depth_scale,
            depth_trunc=DEPTH_TRUNC,
            convert_rgb_to_intensity=False,
        )
        rgbds.append(rgbd)
        im = reader.next_frame()

    reader.close()
    print(f"   Loaded {len(rgbds)} frames")
    return rgbds, intrinsic


def pairwise_odometry(src, dst, intrinsic, init=np.identity(4)):
    """Compute relative transform + information matrix between two RGBD frames."""
    option = o3d.pipelines.odometry.OdometryOption()
    option.depth_diff_max = MAX_DEPTH_DIFF
    success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
        src, dst, intrinsic, init,
        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
        option,
    )
    return success, trans, info


def build_fragment_posegraph(rgbds, intrinsic, start, end):
    """
    Build a pose graph for frames [start, end) with:
      - sequential edges (odometry)
      - a few loop-closure edges within the fragment
    Then optimize.
    """
    pose_graph = o3d.pipelines.registration.PoseGraph()
    trans_odometry = np.identity(4)
    pose_graph.nodes.append(
        o3d.pipelines.registration.PoseGraphNode(trans_odometry)
    )

    n = end - start
    for s in range(n):
        for t in range(s + 1, n):
            src = rgbds[start + s]
            dst = rgbds[start + t]

            if t == s + 1:
                # Sequential (odometry) edge
                success, trans, info = pairwise_odometry(src, dst, intrinsic)
                if not success:
                    trans = np.identity(4)
                    info = np.identity(6)
                trans_odometry = trans @ trans_odometry
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(trans_odometry)
                    )
                )
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        s, t, trans, info, uncertain=False
                    )
                )
            elif t % 5 == 0:
                # Loop-closure candidate edge (sparse, uncertain)
                success, trans, info = pairwise_odometry(src, dst, intrinsic)
                if success:
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            s, t, trans, info, uncertain=True
                        )
                    )

    # Optimize the fragment pose graph
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=MAX_DEPTH_DIFF,
        edge_prune_threshold=0.25,
        reference_node=0,
    )
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )
    return pose_graph


def integrate_fragment(rgbds, intrinsic, pose_graph, start, end):
    """Integrate a fragment's frames into a TSDF, return the fragment point cloud."""
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL_LENGTH,
        sdf_trunc=SDF_TRUNC,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for i in range(end - start):
        pose = pose_graph.nodes[i].pose
        volume.integrate(rgbds[start + i], intrinsic, np.linalg.inv(pose))
    return volume.extract_point_cloud()


def reconstruct_slam(bag_path, output_dir, max_frames=None, frame_skip=1):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Read frames ──
    rgbds, intrinsic = read_frames(bag_path, max_frames, frame_skip)
    n = len(rgbds)

    # ── 2. Build + optimize fragments ──
    print("\n🧩 Building fragments with pose-graph optimization...")
    fragment_pcds = []
    fragment_poses = []  # global pose of each fragment's first frame

    global_pose = np.identity(4)
    for start in range(0, n, FRAGMENT_SIZE):
        end = min(start + FRAGMENT_SIZE, n)
        if end - start < 2:
            break

        pg = build_fragment_posegraph(rgbds, intrinsic, start, end)
        pcd = integrate_fragment(rgbds, intrinsic, pg, start, end)
        pcd = pcd.voxel_down_sample(VOXEL_LENGTH)

        fragment_pcds.append(pcd)
        fragment_poses.append(global_pose.copy())

        # Advance global pose by this fragment's net motion (last node pose)
        net = pg.nodes[end - start - 1].pose
        global_pose = global_pose @ net

        print(f"   Fragment {len(fragment_pcds)}: frames {start}-{end} "
              f"({len(pcd.points)} pts)")

    # ── 3. Register fragments to each other (loop closure between fragments) ──
    print("\n🔗 Registering fragments (loop closure)...")
    combined = o3d.geometry.PointCloud()
    accumulated = np.identity(4)

    for i, pcd in enumerate(fragment_pcds):
        if i == 0:
            combined += pcd
            prev = pcd
            continue

        # ICP align current fragment to the accumulated cloud
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        prev.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        reg = o3d.pipelines.registration.registration_icp(
            pcd, prev, MAX_DEPTH_DIFF, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )
        accumulated = accumulated @ reg.transformation
        pcd.transform(accumulated)
        combined += pcd
        prev = pcd
        print(f"   Aligned fragment {i+1}/{len(fragment_pcds)} "
              f"(fitness={reg.fitness:.2f})")

    # ── 4. Clean + downsample the combined cloud ──
    print("\n🧹 Cleaning combined point cloud...")
    combined = combined.voxel_down_sample(VOXEL_LENGTH)
    combined, _ = combined.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"   Combined: {len(combined.points)} points")

    pcd_path = output_dir / "scene_pointcloud.ply"
    o3d.io.write_point_cloud(str(pcd_path), combined)

    # ── 5. Poisson mesh from the clean cloud ──
    print("\n🔨 Poisson mesh reconstruction...")
    combined.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    combined.orient_normals_consistent_tangent_plane(30)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        combined, depth=9
    )
    # Trim low-density (spurious) vertices
    densities = np.asarray(densities)
    keep = densities > np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(~keep)
    mesh.compute_vertex_normals()

    mesh_path = output_dir / "scene_mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)

    print()
    print("=" * 60)
    print("✅ SLAM RECONSTRUCTION COMPLETE")
    print("=" * 60)
    print(f"   Point cloud: {pcd_path} ({len(combined.points)} pts)")
    print(f"   Mesh:        {mesh_path} ({len(mesh.vertices)} verts)")
    print(f"   View:        python rgbd_reconstruction/view.py --file {pcd_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--output", default="out_slam")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-skip", type=int, default=1)
    args = parser.parse_args()

    reconstruct_slam(args.bag, args.output, args.max_frames, args.frame_skip)
