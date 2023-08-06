import open3d as o3d
from utils import geometry, pointcloud_utils
import ransac_test
import copy
import numpy as np
import outer_bound_planes

def merge_planes(planes, center_point, direction, pointcloud):
  neighbours = []

  for plane in planes:
    neighbours.append(find_neighbours(plane, planes, center_point, direction))

  # Remove duplicates
  neighbours = unique_array_of_arrays(neighbours)

  # Merge the arrays, getting a boolean value wether they converged
  converged, merge_sets = merge_array_with_convergence(neighbours)

  while not converged:
    converged, merge_sets = merge_array_with_convergence(merge_sets)

  result = []

  # Now, merge the planes in the merge sets into one
  for to_merge in merge_sets:
    if len(to_merge) == 1:
      result.append(planes[to_merge[0]])
    else:
      current_planes = np.array(planes)[to_merge]

      # For each point cloud, get the number of points
      plane_counts = []

      for plane in current_planes:
        plane_count = 0
        for pt3D in pointcloud.points:
          distance_to_plane = geometry.point_plane_distance(pt3D, plane)
          if distance_to_plane <= 0.25:
            plane_count += 1
        plane_counts.append(plane_count)

      # Keep the plane with the largest number of points within
      relevant_plane_idx = np.argmax(plane_counts)

      relevant_plane = current_planes[relevant_plane_idx]
      result.append(relevant_plane)

  return result

def find_neighbours(plane, planes, center_point, direction):
  point_on_plane, _ = geometry.line_plane_intersection(center_point, direction, plane)

  neighbour_indices = []

  for index, other_plane in enumerate(planes):
    distance_to_other = geometry.point_plane_distance(point_on_plane, other_plane)
    if distance_to_other < 2: # TODO: this should not be hardcoded
      neighbour_indices.append(index)

  return neighbour_indices

def unique_array_of_arrays(array_of_arrays):
  result = []

  for array in array_of_arrays:
    if array not in result:
      result.append(array)

  return result

def arrays_overlap(array_a, array_b):
  return not set(array_a).isdisjoint(array_b)

def merge_arrays(array_a, array_b):
  result_set = set(array_a)
  result_set.update(array_b)
  return list(result_set)

def merge_array_with_convergence(arrays):
  previous_arrays = copy.deepcopy(arrays)
  result = []

  for index, array in enumerate(arrays):
    current = copy.copy(array)
    for other_index, other_array in enumerate(arrays):
      if index == other_index:
        continue

      if arrays_overlap(current, other_array):
        current = merge_arrays(current, other_array)

    result.append(current)

  # Make unique
  result = unique_array_of_arrays(result)

  converged = previous_arrays == result
  return converged, result

def cleanup_ransac_planes(planes, point_cloud, visualize=False):
  x_planes = []
  y_planes = []
  floor_plane = None

  for plane in planes:
    normal = geometry.plane_normal(plane)

    amax = np.argmax(normal)
    if amax == 0:
      x_planes.append(plane)
    elif amax == 1:
      y_planes.append(plane)
    else:
      floor_plane = plane

  # If either no x or y planes are found, return directly,
  # as the algorithm does not work otherwise
  if len(x_planes) == 0 or len(y_planes) == 0:
    return planes

  x_planes_keep_idx = []
  y_planes_keep_idx = []
  x_planes_discard_idx = []
  y_planes_discard_idx = []

  for index, x_plane in enumerate(x_planes):
    counter = 0
    x_plane_normal = geometry.plane_normal(x_plane)
    for y_plane in y_planes:
      y_plane_normal = geometry.plane_normal(y_plane)
      dot = np.dot(x_plane_normal, y_plane_normal)
      if 0.02 < abs(dot) < 0.98:
        counter += 1
    if counter > len(y_planes) * 0.75:
      x_planes_discard_idx.append(index)
    else:
      x_planes_keep_idx.append(index)

  for index, y_plane in enumerate(y_planes):
    counter = 0
    y_plane_normal = geometry.plane_normal(y_plane)
    for x_plane in x_planes:
      x_plane_normal = geometry.plane_normal(x_plane)
      dot = np.dot(x_plane_normal, y_plane_normal)
      if 0.02 < abs(dot) < 0.98:
        counter += 1
    if counter > len(y_planes) * 0.75:
      y_planes_discard_idx.append(index)
    else:
      y_planes_keep_idx.append(index)

  x_planes_kept = np.array(x_planes)[x_planes_keep_idx]
  y_planes_kept = np.array(y_planes)[y_planes_keep_idx]

  x_planes_discarded = np.array(x_planes)[x_planes_discard_idx]
  y_planes_discarded = np.array(y_planes)[y_planes_discard_idx]

  kept_planes = np.append(x_planes_kept, y_planes_kept, axis=0)
  kept_planes = np.append([floor_plane], kept_planes, axis=0)
  discarded_planes = np.append(x_planes_discarded, y_planes_discarded, axis=0)

  if visualize:
    pointcloud_utils.visualize_planes(kept_planes, point_cloud.voxel_down_sample(0.25), window_name='Kept RANSAC planes')
    pointcloud_utils.visualize_planes(discarded_planes, point_cloud.voxel_down_sample(0.25), window_name='Discarded RANSAC planes')

  return kept_planes

# This assumes the pointcloud is already aligned with the principal directions, if not, use the
# `rotate_pcd_to_match_pcas` method from `pca_methods`.
def find_planes(pointcloud, show_steps=False, pcas=None):
  # Find planes with RANSAC, which might or might not include all the important outer walls
  print('  Finding RANSAC planes')
  _, _, planes, _, _, _ = ransac_test.find_planes_with_pca(pointcloud, 25, create_ceiling=False, keep_misaligned_walls=False, pcas=pcas)
  if show_steps:
    pointcloud_utils.visualize_planes(planes, pointcloud.voxel_down_sample(0.25), window_name='RANSAC Planes')

  # Cleanup the ransac planes, as some might be too much off
  print('  Cleaning up RANSAC planes')
  planes = cleanup_ransac_planes(planes, pointcloud, visualize=show_steps)

  # Get the outer planes we need to have a complete bound around the pointcloud
  print('  Finding outer planes')
  outer_planes, _ = outer_bound_planes.find_outer_bound_planes(pointcloud.uniform_down_sample(10), pcas=pcas, visualize=False)
  if show_steps:
    pointcloud_utils.visualize_planes(outer_planes, pointcloud.voxel_down_sample(0.25), window_name='Outer Planes')

  # Sort planes and outer planes into X and Y planes
  x_planes = []
  y_planes = []

  # Sort outer planes
  for outer_plane in outer_planes:
    amax = np.argmax(np.abs(geometry.plane_normal(outer_plane)))
    if amax == 0:
      x_planes.append(outer_plane)
    elif amax == 1:
      y_planes.append(outer_plane)
    else:
      raise RuntimeError('Not possible')

  # Sort RANSAC planes. Please note that the RANSAC planes should include a floor plane!
  floor_plane = None

  for plane in planes:
    amax = np.argmax(np.abs(geometry.plane_normal(plane)))
    if amax == 0:
      x_planes.append(plane)
    elif amax == 1:
      y_planes.append(plane)
    else:
      # Floor is present here as well
      floor_plane = plane

  # Get the center point to use as the anchor for merging the planes
  center_point = pointcloud.get_center()

  # Visualization only
  if show_steps:
    pointcloud_utils.visualize_planes(x_planes, pointcloud.voxel_down_sample(0.25), window_name='X planes')
    pointcloud_utils.visualize_planes(y_planes, pointcloud.voxel_down_sample(0.25), window_name='Y planes')

  # Show all planes before merging
  if show_steps:
    all_planes = np.append(x_planes, y_planes, axis=0)
    pointcloud_utils.visualize_planes(all_planes, pointcloud.voxel_down_sample(0.25), window_name='All planes')

  # Merge X and Y planes
  print('  Merging planes')
  x_planes_merged = merge_planes(x_planes, center_point, [1, 0, 0], pointcloud)
  y_planes_merged = merge_planes(y_planes, center_point, [0, 1, 0], pointcloud)

  # Put all planes into an array
  all_planes_merged = np.append(x_planes_merged, y_planes_merged, axis=0)

  # Visualization only
  if show_steps:
    pointcloud_utils.visualize_planes(all_planes_merged, pointcloud.voxel_down_sample(0.25), window_name='All planes merged')

  # Put the floor plane at the start of the array of all planes
  all_planes_merged_with_floor = np.append(np.array([floor_plane]), all_planes_merged, axis=0)

  # Create bounding boxes for all planes
  print('  Creating bounding boxes for planes')
  aabboxes = []
  for plane in all_planes_merged_with_floor:
    # Get all points for the current plane
    current_plane_points = []

    for pt3D in pointcloud.points:
      distance_to_plane = geometry.point_plane_distance(pt3D, plane)
      if distance_to_plane <= 0.25:
        current_plane_points.append(pt3D)

    # Create the bounding box
    aabboxes.append(o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(np.array(current_plane_points))))

  # Visualization only
  if show_steps:
    pointcloud_utils.visualize_planes(all_planes_merged_with_floor, pointcloud.voxel_down_sample(0.25), additional_data=aabboxes)

  return all_planes_merged_with_floor, aabboxes
