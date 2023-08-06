import open3d as o3d
from utils import pointcloud_utils
import pca_methods
from tqdm import tqdm
import numpy as np
from point_cloud_grid import FlatPointcloudGrid

POINT_COUNTER = 3
BIN_COUNT_THRESHOLD = 50

def find_outer_bound_planes(pcd, visualize = False, pcas=None, print_messages=False):
  # Get the PCAs if they're not given as param
  if pcas is None:
    pcas = pca_methods.get_pcas_from_first_n_walls(pcd, 5, print_messages=print_messages)

  # Clean the pointcloud
  cleaned_pcd = pointcloud_utils.remove_outliers_with_clustering(pcd, visualize=False, print_messages=print_messages)

  # And flatten the pointcloud
  flattened_pcd = pointcloud_utils.flatten_pointcloud(cleaned_pcd, target_z=-0.1)

  # Create the FlatPointcloudGrid
  grid = FlatPointcloudGrid(flattened_pcd, bin_count=40)
  lineset = grid.get_lineset()

  # For visualization only
  x_bin_points = []
  y_bin_points = []

  y_indexes_found = []
  x_indexes_found = []
  relevant_bin_indices = []

  x_planes = []
  y_planes = []

  # Our results
  planes = []
  aabboxes = []

  # --- Increasing X direction -------------------------------------------------------------------------------------

  for x_idx in range(grid.x_bins):
    current_x_run_y_indexes_found = []
    for y_idx in range(grid.y_bins):
      bin_count = grid.get_bin_item_count(x_idx, y_idx)

      if bin_count > BIN_COUNT_THRESHOLD:
        if not y_idx in y_indexes_found:
          x_bin_points.append(grid.get_middlepoint_of_bin(x_idx, y_idx))
          y_indexes_found.append(y_idx)
          current_x_run_y_indexes_found.append(y_idx)
          relevant_bin_indices.append([x_idx, y_idx])

    prev = False
    prevprev = False
    current_counter = 0
    line_found = False

    if len(current_x_run_y_indexes_found) >= POINT_COUNTER:
      for y_idx in range(grid.y_bins):
        current = y_idx in current_x_run_y_indexes_found

        if current:
          # If current has a point, we can have:
          #  prev has a point => ok, increase normaly
          #  prev does not have a point, but prevprev does => prev was a hole, increase by 2
          #  both prev and prevprev do not have a point => set to 1
          if prev:
            current_counter += 1
          elif not prev and prevprev:
            current_counter += 2
          elif not prev and not prevprev:
            current_counter = 1
        else:
          # If the current does not have a point:
          #  previous has a point => do nothing, current might be a "hole"
          #  previous does not have a point => reset to zero, hole "too big"
          if prev:
            pass
          else:
            current_counter = 0

        # Set new prev values
        prevprev = prev
        prev = current

        if current_counter >= POINT_COUNTER:
          line_found = True
          break

    if line_found:
      x_planes.append(x_idx)

  # --- Decreasing X direction -------------------------------------------------------------------------------------

  y_indexes_found = []
  x_indexes_found = []
  relevant_bin_indices = []

  for x_idx in range(grid.x_bins - 1, -1, -1):
    current_x_run_y_indexes_found = []
    for y_idx in range(grid.y_bins):
      bin_count = grid.get_bin_item_count(x_idx, y_idx)

      if bin_count > BIN_COUNT_THRESHOLD:
        if not y_idx in y_indexes_found:
          x_bin_points.append(grid.get_middlepoint_of_bin(x_idx, y_idx))
          y_indexes_found.append(y_idx)
          current_x_run_y_indexes_found.append(y_idx)
          relevant_bin_indices.append([x_idx, y_idx])

    prev = False
    prevprev = False
    current_counter = 0
    line_found = False

    if len(current_x_run_y_indexes_found) >= POINT_COUNTER:
      for y_idx in range(grid.y_bins):
        current = y_idx in current_x_run_y_indexes_found

        if current:
          # If current has a point, we can have:
          #  prev has a point => ok, increase normaly
          #  prev does not have a point, but prevprev does => prev was a hole, increase by 2
          #  both prev and prevprev do not have a point => set to 1
          if prev:
            current_counter += 1
          elif not prev and prevprev:
            current_counter += 2
          elif not prev and not prevprev:
            current_counter = 1
        else:
          # If the current does not have a point:
          #  previous has a point => do nothing, current might be a "hole"
          #  previous does not have a point => reset to zero, hole "too big"
          if prev:
            pass
          else:
            current_counter = 0

        # Set new prev values
        prevprev = prev
        prev = current

        if current_counter >= POINT_COUNTER:
          line_found = True
          break

    if line_found:
      x_planes.append(x_idx)

  # --- Create X planes -------------------------------------------------------------------------------------

  for x_plane in x_planes:
    current_points = []

    for pt3D in tqdm(pcd.points, disable=(not print_messages)):
      x_bin, _ = grid.get_bin_indices(pt3D)

      if x_bin == x_plane:
        current_points.append(pt3D)

    plane_pointcloud = pointcloud_utils.pointcloud_from_points(current_points, uniform_color=[1, 0, 0])
    plane_pointcloud.paint_uniform_color([0.5, 0.5, 0.5])

    plane, _, _ = pca_methods.use_pcas_for_constrained_ransac(pcas, plane_pointcloud, direction='x', iterations=300, print_messages=print_messages)

    planes.append(plane)
    aabboxes.append(o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(np.array(current_points))))

  # --- Increasing Y direction -------------------------------------------------------------------------------------

  y_indexes_found = []
  x_indexes_found = []
  relevant_bin_indices = []

  for y_idx in range(grid.y_bins):
    current_x_run_y_indexes_found = []
    for x_idx in range(grid.x_bins):
      bin_count = grid.get_bin_item_count(x_idx, y_idx)

      if bin_count > BIN_COUNT_THRESHOLD:
        if not x_idx in x_indexes_found:
          y_bin_points.append(grid.get_middlepoint_of_bin(x_idx, y_idx))
          x_indexes_found.append(x_idx)
          current_x_run_y_indexes_found.append(x_idx)
          relevant_bin_indices.append([x_idx, y_idx])

    prev = False
    prevprev = False
    current_counter = 0
    line_found = False

    if len(current_x_run_y_indexes_found) >= POINT_COUNTER:
      for x_idx in range(grid.x_bins):
        current = x_idx in current_x_run_y_indexes_found

        if current:
          # If current has a point, we can have:
          #  prev has a point => ok, increase normaly
          #  prev does not have a point, but prevprev does => prev was a hole, increase by 2
          #  both prev and prevprev do not have a point => set to 1
          if prev:
            current_counter += 1
          elif not prev and prevprev:
            current_counter += 2
          elif not prev and not prevprev:
            current_counter = 1
        else:
          # If the current does not have a point:
          #  previous has a point => do nothing, current might be a "hole"
          #  previous does not have a point => reset to zero, hole "too big"
          if prev:
            pass
          else:
            current_counter = 0

        # Set new prev values
        prevprev = prev
        prev = current

        if current_counter >= POINT_COUNTER:
          line_found = True
          break

    if line_found:
      y_planes.append(y_idx)

  # --- Decreasing Y direction -------------------------------------------------------------------------------------

  y_indexes_found = []
  x_indexes_found = []
  relevant_bin_indices = []

  for y_idx in range(grid.y_bins - 1, -1, -1):
    current_x_run_y_indexes_found = []
    for x_idx in range(grid.x_bins):
      bin_count = grid.get_bin_item_count(x_idx, y_idx)

      if bin_count > BIN_COUNT_THRESHOLD:
        if not x_idx in x_indexes_found:
          y_bin_points.append(grid.get_middlepoint_of_bin(x_idx, y_idx))
          x_indexes_found.append(x_idx)
          current_x_run_y_indexes_found.append(x_idx)
          relevant_bin_indices.append([x_idx, y_idx])

    prev = False
    prevprev = False
    current_counter = 0
    line_found = False

    if len(current_x_run_y_indexes_found) >= POINT_COUNTER:
      for x_idx in range(grid.x_bins):
        current = x_idx in current_x_run_y_indexes_found

        if current:
          # If current has a point, we can have:
          #  prev has a point => ok, increase normaly
          #  prev does not have a point, but prevprev does => prev was a hole, increase by 2
          #  both prev and prevprev do not have a point => set to 1
          if prev:
            current_counter += 1
          elif not prev and prevprev:
            current_counter += 2
          elif not prev and not prevprev:
            current_counter = 1
        else:
          # If the current does not have a point:
          #  previous has a point => do nothing, current might be a "hole"
          #  previous does not have a point => reset to zero, hole "too big"
          if prev:
            pass
          else:
            current_counter = 0

        # Set new prev values
        prevprev = prev
        prev = current

        if current_counter >= POINT_COUNTER:
          line_found = True
          break

    if line_found:
      y_planes.append(y_idx)

  if visualize:
    x_bin_points_pcd = pointcloud_utils.pointcloud_from_points(x_bin_points, uniform_color=[0, 1, 0])
    y_bin_points_pcd = pointcloud_utils.pointcloud_from_points(y_bin_points, uniform_color=[1, 0, 0])
    o3d.visualization.draw_geometries([flattened_pcd, lineset, x_bin_points_pcd, y_bin_points_pcd], window_name='Showing points pcds')

  # --- Create Y planes -------------------------------------------------------------------------------------

  for y_plane in y_planes:
    current_points = []

    for pt3D in tqdm(pcd.points, disable=(not print_messages)):
      _, y_bin = grid.get_bin_indices(pt3D)

      if y_bin == y_plane:
        current_points.append(pt3D)

    plane_pointcloud = pointcloud_utils.pointcloud_from_points(current_points, uniform_color=[1, 0, 0])
    plane_pointcloud.paint_uniform_color([0.5, 0.5, 0.5])

    plane, _, _ = pca_methods.use_pcas_for_constrained_ransac(pcas, plane_pointcloud, direction='y', iterations=300, print_messages=print_messages)

    planes.append(plane)
    aabboxes.append(o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(np.array(current_points))))

  if visualize:
    pointcloud_utils.visualize_planes(planes, pcd, show_visualization=True, window_name='Showing planes', additional_data=aabboxes)

  return planes, aabboxes
