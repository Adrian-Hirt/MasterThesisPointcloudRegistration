import open3d as o3d
import numpy as np
from utils import geometry, helpers, pointcloud_utils, geometry
import copy
import random
import functools
import operator
import pca_methods

def extrude_line_segments(pointcloud, lineset, boundary_lineset, sampled_point_count=100_000, scatter=True, z_height=None, rotmat=None):
  sampled_points = []

  if rotmat is not None:
    lineset.rotate(rotmat, pointcloud.get_center())
    boundary_lineset.rotate(rotmat, pointcloud.get_center())

  line_segments = helpers.line_endpoints_from_lineset(lineset)
  bound_line_segments = helpers.line_endpoints_from_lineset(boundary_lineset)

  total_line_len = 0


  for line_segment in line_segments:
    line_start, line_end = line_segment
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)
    total_line_len += line_len

  avg_line_length = total_line_len / len(line_segments)

  if z_height is None:
    choices = np.random.choice(len(lineset.points), 3, replace=False)
    points = np.asarray(lineset.points)[choices]
    floor_plane = geometry.plane_from_three_points(points)
    z_height = resolve_z_height(pointcloud, floor_plane=floor_plane)

  if scatter:
    x_y_scatter = avg_line_length / 50.0
    floor_scatter = z_height / 20

  # If desired number of points is given, calculate how many points
  # we need to sample "per line length unit":
  points_per_line_unit = sampled_point_count / total_line_len

  bboxes = []

  for line_segment in line_segments:
    segment_points = []
    line_start, line_end = line_segment
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)

    # Compute how many points to sample, the larger the line
    # the more points will be sampled
    samples = int(points_per_line_unit * line_len)

    amax = np.argmax(np.abs(line_vec))

    for _ in range(samples):
      # Choose a random point along the line
      current_point = line_start + random.uniform(0, 1) * line_vec

      # "Scatter" x and y values a bit, if enabled
      if scatter:
        if amax == 0:
          current_point[1] += random.uniform(-x_y_scatter, x_y_scatter)
        elif amax == 1:
          current_point[0] += random.uniform(-x_y_scatter, x_y_scatter)

      # Set a random z value. Please note, that we might need
      # to add a vector, if the wall is not perfectly aligned with the floor plane!
      current_point[2] = random.uniform(current_point[2], current_point[2] + z_height)

      sampled_points.append(current_point)
      segment_points.append(current_point)

    if helpers.array_of_arrays_contains(bound_line_segments, line_segment) and len(segment_points) > 0:
      bboxes.append(o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(np.array(segment_points))))

  sampled_pcd = pointcloud_utils.pointcloud_from_points(sampled_points, uniform_color=[0, 1, 0])
  sampled_walls_pcd = copy.deepcopy(sampled_pcd)

  # Sample the floor!
  floor_sampled_points = []
  bbox = sampled_pcd.get_axis_aligned_bounding_box()

  flattened_line_segments = functools.reduce(operator.iconcat, line_segments, [])
  unique_line_segment_points = np.unique(np.array(flattened_line_segments), axis=0)
  choices = np.random.choice(unique_line_segment_points.shape[0], 3, replace=False)

  # TOOD: the floor was already computed previously, I don't need to do this again!
  selected_points = unique_line_segment_points[choices]
  a, b, c, d = geometry.plane_from_three_points(selected_points)

  max_point = bbox.get_max_bound()
  min_point = bbox.get_min_bound()

  # Floor samples
  floor_sample_count = int(sampled_point_count / 4)

  pcd_middle_height = min_point[2] + (z_height / 2)

  for _ in range(floor_sample_count):
    x = random.uniform(min_point[0], max_point[0])
    y = random.uniform(min_point[1], max_point[1])
    z = (-a * x -b * y - d) / c

    current_floor_point = np.array([x, y, z])

    # Scatter floor point a bit if enabled
    if scatter:
      current_floor_point[2] += random.uniform(-floor_scatter, floor_scatter)

    # Do the "raycasting" to check wether a point lies inside or outside
    outside_point = np.array([min_point[0] - 5, y - 0.01, pcd_middle_height - 0.01])
    inside_point = np.array([x, y, pcd_middle_height])
    direction = inside_point - outside_point

    counter = 0

    for segment_bbox in bboxes:
      intersects, t_near, t_far = geometry.line_aabb_intersection(outside_point, direction, segment_bbox)
      if intersects:
        if 0 <= t_near <= 1 or 0 <= t_far <= 1:
          counter += 1

    # Only keep point if it's "inside" the pointcloud walls
    if counter % 2 == 1:
      sampled_pcd.points.append(current_floor_point)
      sampled_pcd.colors.append([0, 0, 1])
      floor_sampled_points.append(current_floor_point)

  sampled_floor_pcd = pointcloud_utils.pointcloud_from_points(floor_sampled_points, uniform_color=[0, 0, 0])

  if rotmat is not None:
    rotmat_inv = np.linalg.inv(rotmat)
    sampled_pcd.rotate(rotmat_inv, pointcloud.get_center())
    sampled_walls_pcd.rotate(rotmat_inv, pointcloud.get_center())
    sampled_floor_pcd.rotate(rotmat_inv, pointcloud.get_center())

  return sampled_pcd, sampled_walls_pcd, sampled_floor_pcd

def resolve_z_height(pointcloud, floor_plane=None, print_messages=False):
  if floor_plane is None:
    max_extent = pointcloud.get_axis_aligned_bounding_box().get_max_extent()
    distance_threshold = max_extent / 500
    voxel_size = distance_threshold / 5
    pcas = pca_methods.get_pcas_from_first_n_walls(pointcloud, 5, print_messages=False)
    floor_plane, _, _ = pca_methods.use_pcas_for_constrained_ransac(pcas, pointcloud.voxel_down_sample(voxel_size), direction='z', iterations=300, distance_threshold=distance_threshold)

  pointcloud_ds = copy.deepcopy(pointcloud).uniform_down_sample(100)

  floor_normal = geometry.normalized(geometry.plane_normal(floor_plane))
  floor_step = np.array(floor_normal)

  # If the z-extent of the oriented bounding box is rather small, the
  # floor_step is made smaller, such that we don't have a too large
  # step for small point clouds.
  z_extent = pointcloud.get_oriented_bounding_box().extent[2]

  if z_extent < 10:
    # Make it such that at least 8 steps fit into the z-extent,
    # to accomodate any outliers
    max_step_length = z_extent / 8

    # Multiply the floor step by the factor (step has unit length)
    floor_step *= max_step_length

  point = geometry.point_on_plane(floor_plane)
  current_point = np.array(copy.deepcopy(point))
  initial_z_height = point[2]

  # Pick the point from the pointcloud with the highest z value
  max_z_point = pointcloud.points[np.argmax(pointcloud.points, axis=0)[2]]

  flipped_floor = False

  if not geometry.point_above_plane(max_z_point, floor_plane):
    floor_step *= -1
    flipped_floor = True

  while True:
    current_point += floor_step
    current_plane = geometry.plane_from_point_and_normal(current_point, floor_normal)

    total = 0
    above_count = 0

    for pt3D in pointcloud_ds.points:
      total += 1
      if flipped_floor:
        if not geometry.point_above_plane(pt3D, current_plane):
          above_count += 1
      else:
        if geometry.point_above_plane(pt3D, current_plane):
          above_count += 1

    if print_messages:
      print(above_count / total)

    if above_count / total < 0.01:
      break

  z_height = current_point[2] - initial_z_height

  return z_height
