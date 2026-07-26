"""
View a reconstructed mesh or point cloud in an interactive 3D window.
----------------------------------------------------------------------
Mouse: rotate | Scroll: zoom | Shift+drag: pan

Usage:
    python view.py --file out_test/scene_mesh.ply
    python view.py --file out_test/scene_pointcloud.ply
"""

import argparse
import numpy as np
import open3d as o3d


def view(file_path: str):
    print(f"📂 Loading: {file_path}")

    is_mesh = "mesh" in file_path.lower()
    if is_mesh:
        geom = o3d.io.read_triangle_mesh(file_path)
        geom.compute_vertex_normals()
        print(f"   Mesh: {len(geom.vertices)} vertices, {len(geom.triangles)} triangles")
        pts = np.asarray(geom.vertices)
    else:
        geom = o3d.io.read_point_cloud(file_path)
        print(f"   Point cloud: {len(geom.points)} points")
        pts = np.asarray(geom.points)

    # Compute geometry center + extent for auto-framing
    center = pts.mean(axis=0)
    bbox = geom.get_axis_aligned_bounding_box()
    extent = np.linalg.norm(bbox.get_extent())
    print(f"   Center: {center.round(2)} | Extent: {extent:.2f} m")

    print("\nControls: drag=rotate | scroll=zoom | shift+drag=pan | q=quit")

    # Use the Visualizer class for fine control over the camera
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Reconstruction", width=1400, height=900)
    vis.add_geometry(geom)

    # Render options
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.12])  # dark background
    opt.point_size = 2.0
    opt.light_on = True
    if is_mesh:
        opt.mesh_show_back_face = True

    # Camera — frame the object tightly
    ctr = vis.get_view_control()
    ctr.set_lookat(center)
    ctr.set_front([0.0, -0.3, -1.0])   # look slightly from above
    ctr.set_up([0.0, -1.0, 0.0])
    ctr.set_zoom(0.45)                  # tighter than default (lower = closer)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    args = parser.parse_args()
    view(args.file)
