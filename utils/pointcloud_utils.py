import open3d as o3d
import numpy as np
from utils import geometry
import sys
import copy

def remove_outliers_with_clustering(pointcloud, visualize=False, print_messages=False):
  labels = np.array(pointcloud.cluster_dbscan(eps=0.2, min_points=20, print_progress=print_messages))

  if print_messages:
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")

  point_cloud_points = []
  point_cloud_colors = []
  outlier_points = []
  outlier_colors = []

  counter = 0

  for pt3D, color in zip(pointcloud.points, pointcloud.colors):
    if labels[counter] != -1:
      point_cloud_points.append(pt3D)
      point_cloud_colors.append(color)
    else:
      outlier_points.append(pt3D)
      outlier_colors.append([1, 0, 0])
    counter += 1

  new_point_cloud = pointcloud_from_points(point_cloud_points, colors=point_cloud_colors)
  outlier_point_cloud = pointcloud_from_points(outlier_points, colors=outlier_colors)

  if visualize:
    o3d.visualization.draw_geometries([new_point_cloud, outlier_point_cloud])

  return new_point_cloud

def remove_outliers_statistically(point_cloud, visualize=True):
  cleaned_point_cloud, ind = point_cloud.remove_statistical_outlier(nb_neighbors=4, std_ratio=25.0)

  if visualize:
    inlier_cloud = point_cloud.select_by_index(ind)
    outlier_cloud = point_cloud.select_by_index(ind, invert=True)

    outlier_cloud.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

  return cleaned_point_cloud

def fit_ransac_plane(pointcloud, visualize=False, print_messages=True):
  aabbox = pointcloud.get_axis_aligned_bounding_box()
  extent = aabbox.get_extent()
  distance_threshold = np.max(extent) / 250

  plane_model, inliers = pointcloud.segment_plane(distance_threshold=distance_threshold, ransac_n=3, num_iterations=500)

  [a, b, c, d] = plane_model
  if print_messages:
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

  if visualize:
    inlier_cloud = pointcloud.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pointcloud.select_by_index(inliers, invert=True)
    outlier_cloud.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

  return plane_model

def split_pc_points_into_clusters(points, cluster_labels, label_count, min_cluster_size=None):
  # 2D Array to hold the clusters
  clusters = [[] for x in range(label_count)]

  assert(len(points) == len(cluster_labels))

  for point, label in zip(points, cluster_labels):
    # Remove outliers
    if label >= 0:
      clusters[label].append(point)

  # If we're given a min cluster size, we only keep the clusters
  # which are larger than this minimum size.
  if min_cluster_size != None:
    result = []

    for cluster in clusters:
      if len(cluster) >= min_cluster_size:
        result.append(cluster)

    return result
  else:
    return clusters

def visualize_planes(planes, pointcloud, show_visualization=True, window_name='Open 3D', additional_data=None, plane_colors_list=None):
  points = []
  colors = []

  plane_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
  plane_colors_len = len(plane_colors)

  for pt3D, color in zip(pointcloud.points, pointcloud.colors):
    closest_distance = sys.float_info.max
    closest_plane_idx = None

    for index, plane in enumerate(planes):
      distance_to_plane = geometry.point_plane_distance(pt3D, plane)
      if closest_distance > distance_to_plane:
        closest_distance = distance_to_plane
        closest_plane_idx = index

    points.append(pt3D)

    if closest_distance <= 0.5:
      if plane_colors_list is None:
        colors.append(plane_colors[closest_plane_idx % plane_colors_len])
      else:
        colors.append(plane_colors_list[closest_plane_idx])
    else:
      colors.append(color)

  if len(points) == 0:
    print('No points found, skipping')
    return None

  plane_pointcloud = pointcloud_from_points(points, colors=colors)

  if show_visualization:
    if additional_data:
      o3d.visualization.draw_geometries([plane_pointcloud] + additional_data, window_name=window_name)
    else:
      o3d.visualization.draw_geometries([plane_pointcloud], window_name=window_name)

  return plane_pointcloud

def pointcloud_from_points(points, colors=None, normals=None, uniform_color=None):
  if len(points) == 0:
    raise Exception('No points given, cannot construct a pointcloud')

  pointcloud = o3d.geometry.PointCloud()
  pointcloud.points = o3d.utility.Vector3dVector(np.array(points))

  if colors is not None and uniform_color is not None:
    raise Exception('Cannot give colors and uniform_color at the same time')

  if colors is not None:
    if len(colors) != len(points):
      raise Exception('colors and points arrays do not have the same length')
    pointcloud.colors = o3d.utility.Vector3dVector(np.array(colors))

  if normals is not None:
    if len(normals) != len(points):
      raise Exception('normals and points arrays do not have the same length')
    pointcloud.normals = o3d.utility.Vector3dVector(np.array(normals))

  if uniform_color is not None:
    pointcloud.paint_uniform_color(uniform_color)

  return pointcloud

def flatten_pointcloud(pointcloud, color=[0.5, 0.5, 0.5], target_z=None):
  flattened_pointcloud = o3d.geometry.PointCloud()
  min_z = sys.float_info.max

  if target_z is None:
    for pt3D in pointcloud.points:
      min_z = min(pt3D[2], min_z)

  for pt3D in pointcloud.points:
    if target_z is None:
      pt = [pt3D[0], pt3D[1], min_z]
    else:
      pt = [pt3D[0], pt3D[1], target_z]
    flattened_pointcloud.points.append(pt)
    flattened_pointcloud.colors.append(color)

  return flattened_pointcloud

def replace_cad_points_with_captured(cad_model_pointcloud, captured_pointcloud, cad_model_transformation, visualize=False):
  # Copy the original CAD pointcloud
  cad_model_pointcloud_copy = copy.deepcopy(cad_model_pointcloud)

  # Apply transformation to the CAD pointcloud
  cad_model_pointcloud_copy.transform(cad_model_transformation)

  # Build a KDtree
  pcd_tree = o3d.geometry.KDTreeFlann(cad_model_pointcloud_copy)

  # Get the point density
  point_density = get_point_density_based_on_k_neighbours(cad_model_pointcloud_copy, 5) / 4

  found_idx = np.array([], dtype=np.int32)

  # For each point in the captured pointcloud, get the neighbours in a certain radius
  for pt3D in captured_pointcloud.points:
    k, idx, _ = pcd_tree.search_radius_vector_3d(pt3D, point_density)
    if k > 0:
      found_idx = np.append(found_idx, np.asarray(idx))

  # Only keep unique indices
  found_idx = np.unique(found_idx)

  # Filter the points and colors to only include the points that aren't too close to a point
  # in the captured pointcloud
  kept_points = np.delete(np.asarray(cad_model_pointcloud_copy.points), found_idx, axis=0)
  kept_colors = np.delete(np.asarray(cad_model_pointcloud_copy.colors), found_idx, axis=0)

  # Merge the captured points with the kept points
  points = np.append(kept_points, captured_pointcloud.points, axis=0)
  colors = np.append(kept_colors, captured_pointcloud.colors, axis=0)

  # Create a new pointcloud
  merged_pcd = pointcloud_from_points(points, colors=colors)

  if visualize:
    o3d.visualization.draw_geometries([merged_pcd])

  return merged_pcd

def get_point_density_based_on_k_neighbours(pointcloud, k):
  # Build a KDtree
  pcd_tree = o3d.geometry.KDTreeFlann(pointcloud)

  # The max radius we're searching for is a fraction of the max extent
  max_radius = pointcloud.get_axis_aligned_bounding_box().get_max_extent() / 20

  average_distance = 0
  counter = 0

  for pt3D in pointcloud.points:
    _, _, distances = pcd_tree.search_hybrid_vector_3d(pt3D, max_radius, k)

    for distance in distances[1:]:
      average_distance += distance
      counter += 1

  average_distance /= counter

  return average_distance