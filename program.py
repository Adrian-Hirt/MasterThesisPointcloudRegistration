import pointcloud_generation
import global_reg
import open3d as o3d
from utils import pointcloud_utils
import ransac_test
import numpy as np
import copy
import find_planes_for_intersection_of_lines
import pca_methods

def run_program(captured_point_cloud, cad_model_point_cloud, visualize=True):
  print('Cleaning point cloud... ')
  cleaned_pcd = pointcloud_utils.remove_outliers_statistically(captured_point_cloud, visualize=False)

  # Rotate the pointcloud to match PCA directions (needed for the following algorithms to work)
  print('Rotating point cloud... ')
  rotated_captured_point_cloud, _ = pca_methods.rotate_pcd_to_match_pcas(cleaned_pcd)

  # Get the new pcas to only compute them once
  print('Computing PCAs of point cloud... ')
  new_pcas = pca_methods.get_pcas_from_first_n_walls(rotated_captured_point_cloud.uniform_down_sample(10), 5, print_messages=False)

  # Find planes
  print('Find planes in point cloud... ')
  planes, axis_aligned_bboxes = find_planes_for_intersection_of_lines.find_planes(rotated_captured_point_cloud, show_steps=visualize, pcas=new_pcas)

  # Visualize if enabled
  if visualize:
    pointcloud_utils.visualize_planes(planes, rotated_captured_point_cloud.uniform_down_sample(50), additional_data=axis_aligned_bboxes)

  # Find intersection of lines
  print('Compute intersections of planes... ')
  final_lineset, _, boundary_lineset = ransac_test.find_intersections_of_lines(rotated_captured_point_cloud, planes, axis_aligned_bboxes, visualize_steps=False)


  # Visualize if enabled
  if visualize:
    o3d.visualization.draw_geometries([final_lineset])

  # Compute a sampled pointcloud, i.e. a pointcloud which has less noise, from the line segments we computed earlier
  print('Creating sampled point cloud... ')
  sampled_pcd, _, sampled_floor_pcd = pointcloud_generation.extrude_line_segments(
    rotated_captured_point_cloud.uniform_down_sample(10),
    final_lineset,
    boundary_lineset,
    sampled_point_count=200_000
  )

  # Visualize if enabled
  if visualize:
    o3d.visualization.draw_geometries([sampled_pcd, rotated_captured_point_cloud])

  # Fit the pointclouds into eachother globally
  print('Computing global registration... ')
  global_fit_transformation, _ = global_reg.find_optimal_global_fit(cad_model_point_cloud, sampled_pcd, captured_floor_pointcloud=sampled_floor_pcd, visualize=False)

  # Compute the inverse transform for the CAD model
  inverse_global_fit_transformation = np.linalg.inv(global_fit_transformation)

  # Visualize if enabled
  if visualize:
    cad_model_pointcloud_global_fit = copy.deepcopy(cad_model_point_cloud)
    cad_model_pointcloud_global_fit.transform(inverse_global_fit_transformation)
    o3d.visualization.draw_geometries([sampled_pcd, cad_model_pointcloud_global_fit], window_name='Point clouds aligned after global fit step')

  # Get the local alignment with ICP
  print('Refining registration locally... ')
  local_opt_transform, _ = global_reg.local_optimization(cad_model_point_cloud, sampled_pcd, inverse_global_fit_transformation)

  # Compute the transformation from the global align & the ICP align
  final_transformation = np.matmul(local_opt_transform, inverse_global_fit_transformation)

  # Visualize if enabled
  if visualize:
    cad_model_pointcloud_copy = copy.deepcopy(cad_model_point_cloud)
    cad_model_pointcloud_copy.transform(final_transformation)
    o3d.visualization.draw_geometries([sampled_pcd, cad_model_pointcloud_copy])
    o3d.visualization.draw_geometries([rotated_captured_point_cloud, cad_model_pointcloud_copy])

  # Replace poins in cad model pointcloud with the captured ones
  print('Merging point clouds... ')
  resulting_pointcloud = pointcloud_utils.replace_cad_points_with_captured(cad_model_point_cloud, rotated_captured_point_cloud, final_transformation)

  return resulting_pointcloud, final_transformation

if __name__ == '__main__':
  # Read pointclouds
  captured_point_cloud = o3d.io.read_point_cloud('testdata/captured_1.ply')
  cad_model_pointcloud = o3d.io.read_point_cloud('testdata/home_cad.ply').paint_uniform_color([1, 0.706, 0])

  # Visualize input
  o3d.visualization.draw_geometries([captured_point_cloud, cad_model_pointcloud])

  # Run the program
  resulting_pointcloud, final_transformation = run_program(captured_point_cloud, cad_model_pointcloud, visualize=False)

  # Visualize result
  o3d.visualization.draw_geometries([resulting_pointcloud])