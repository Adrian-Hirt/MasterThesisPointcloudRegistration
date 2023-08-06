import open3d as o3d
import numpy as np
from tqdm import tqdm
from utils import geometry, helpers, pointcloud_utils
from itertools import repeat
import random
import pca_methods
from point_cloud_grid import FlatPointcloudGrid
import sys
import math

NORMAL_VECTOR_THRESHOLD = 0.1
POINTS_BELOW_FLOOR_THRESHOLD = 1.0
STOP_ITERATIONS_THRESHOLD = 50
PCA_DOT_THRESHOLD = 0.99

def find_planes(pointcloud, iterations, create_ceiling=True, print_messages=True, keep_invalid_floor=False):
  current_step_pointcloud = pointcloud

  resulting_wall_points = []
  resulting_floor_points = []
  axis_aligned_bboxes = []
  oriented_bboxes = []
  planes = []
  floor_found = False
  wall_pointclouds = []
  floor_pointcloud = None

  aabbox = pointcloud.get_axis_aligned_bounding_box()
  extent = aabbox.get_extent()
  distance_threshold = np.max(extent) / 500

  for _ in range(0, iterations):
    valid_floor = False
    valid_wall = False

    # Get the plane
    plane = pointcloud_utils.fit_ransac_plane(current_step_pointcloud, visualize=False, print_messages=print_messages)

    # Check wether the plane is "horizontal" or "vertical"
    plane_normal = geometry.plane_normal(plane)

    if -NORMAL_VECTOR_THRESHOLD <= plane_normal[0] <= NORMAL_VECTOR_THRESHOLD and -NORMAL_VECTOR_THRESHOLD <= plane_normal[1] <= NORMAL_VECTOR_THRESHOLD:
      # Plane is horizontal. For a horizontal plane to make any sense (as we assume
      # the only horizontal planes are floors), we require the plane to be at the "bottom"
      # of the pointcloud, i.e. only a small percentage of points is allowed to be below
      # this plane. Otherwise, it's a wrong plane and should be discarded
      if print_messages:
        print('Horizontal')

      under_count = 0
      above_count = 0
      total = 0

      # Check how many points are "under" the floor plane
      for pt3D in tqdm(current_step_pointcloud.points, disable=(not print_messages)):
        # points.append(pt3D)
        total += 1
        if geometry.point_above_plane(pt3D, plane):
          # point_colors.append([0.0, 1.0, 0.0])
          above_count += 1
        else:
          # Check wether the point is close enough to the floor
          # to be still considered floor. If yes, we keep it,
          # otherwise the point is discarded
          distance = geometry.point_plane_distance(pt3D, plane)
          if distance < distance_threshold:
            pass
            # point_colors.append([1.0, 0.0, 0.0])
          else:
            under_count += 1
            # point_colors.append([0.0, 0.0, 1.0])

      percentage_under = float(under_count) / total
      percentage_under *= 100.0

      if print_messages:
        print(f"There are {under_count} points under the plane and {above_count} above of total {total}, for a total of {percentage_under}% under the plane")

      # If the percentage of points below the plane is higher than a certain threshold,
      # we consider the plane to be an invalid floor
      if not keep_invalid_floor and percentage_under > POINTS_BELOW_FLOOR_THRESHOLD:
        if print_messages:
          print('Invalid floor, ignoring')
      else:
        valid_floor = True
    elif -NORMAL_VECTOR_THRESHOLD <= plane_normal[2] <= NORMAL_VECTOR_THRESHOLD:
      # Plane is vertical, can just keep going
      if print_messages:
        print('Vertical')
      valid_wall = True
      pass
    else:
      # Plane is crooked, ignore plane, Maybe we need to remove some points, such
      # that we don't get the same plane in the next iteration
      if print_messages:
        print('Other')
      pass

    remaining_points = []
    current_valid_points = []

    # Now, remove all points lying on (or within a threshold of) the plane
    for pt3D in tqdm(current_step_pointcloud.points, disable=(not print_messages)):
      distance_to_plane = geometry.point_plane_distance(pt3D, plane)
      if distance_to_plane > distance_threshold:
        remaining_points.append(pt3D)
      elif valid_floor and not floor_found:
        resulting_floor_points.append(pt3D)
        current_valid_points.append(pt3D)
      elif valid_wall:
        resulting_wall_points.append(pt3D)
        current_valid_points.append(pt3D)

    if (valid_floor and not floor_found) or valid_wall:
      axis_aligned_bboxes.append(o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(np.array(current_valid_points))))
      oriented_bboxes.append(o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(np.array(current_valid_points))))
      planes.append(plane)

    if valid_wall:
      wall_pointclouds.append(pointcloud_utils.pointcloud_from_points(current_valid_points))

    if valid_floor and not floor_found:
      floor_pointcloud = pointcloud_utils.pointcloud_from_points(current_valid_points)

    if valid_floor:
      floor_found = True

    # If there are less points than a threshold, we stop
    if len(remaining_points) < STOP_ITERATIONS_THRESHOLD:
      if print_messages:
        print(f"Stopping as we have less than {STOP_ITERATIONS_THRESHOLD} points")
      break

    current_step_pointcloud = pointcloud_utils.pointcloud_from_points(remaining_points)

  final_points = []
  final_colors = []

  if create_ceiling:
    max_wall_z = -100000
    avg_floor_z = np.average(np.array(resulting_floor_points)[:, 2])

    for pt3D in resulting_wall_points:
      max_wall_z = max(max_wall_z, pt3D[2])

    # print(f"max_z is {max_z}")

    distance = max_wall_z - avg_floor_z

    # Copy all floor points and shift them up to make the "ceiling"
    ceiling_points = []
    for pt3D in resulting_floor_points:
      dist = distance + pt3D[2]
      point = [pt3D[0], pt3D[1], dist]
      ceiling_points.append(point)

  # First add the floor points
  final_points += resulting_floor_points
  final_colors += repeat([1.0, 0.0, 0.0], len(resulting_floor_points))

  # And then add the wall points
  final_points += resulting_wall_points
  final_colors += repeat([0.0, 1.0, 0.0], len(resulting_wall_points))

  if create_ceiling:
    # And finally, the ceiling points
    final_points += ceiling_points
    final_colors += repeat([0.0, 0.0, 1.0], len(ceiling_points))

    # Create new AABB for ceiling points
    axis_aligned_bboxes.append(o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(np.array(ceiling_points))))
    oriented_bboxes.append(o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(np.array(ceiling_points))))
    ceiling_pcd = pointcloud_utils.pointcloud_from_points(ceiling_points)
    plane = pointcloud_utils.fit_ransac_plane(ceiling_pcd, visualize=False)
    planes.append(plane)

  if final_points:
    final_pointcloud = pointcloud_utils.pointcloud_from_points(final_points, colors=final_colors)
  else:
    final_pointcloud = None

  # o3d.visualization.draw_geometries([final_pointcloud] + axis_aligned_bboxes)

  return axis_aligned_bboxes, oriented_bboxes, planes, final_pointcloud, wall_pointclouds, floor_pointcloud

def find_planes_with_pca(pointcloud, iterations, create_ceiling=True, keep_misaligned_walls=True, visualize_fit_ransac_plane=False, pcas=None, print_messages=False):
  current_step_pointcloud = pointcloud.uniform_down_sample(10)

  resulting_wall_points = []
  resulting_floor_points = []
  axis_aligned_bboxes = []
  oriented_bboxes = []
  planes = []
  floor_found = False
  wall_pointclouds = []
  floor_pointcloud = None

  aabbox = pointcloud.get_axis_aligned_bounding_box()
  extent = aabbox.get_extent()
  distance_threshold = np.max(extent) / 250

  if pcas is None:
    pcas = pca_methods.get_pcas_from_first_n_walls(pointcloud.uniform_down_sample(10), 5, print_messages=False)

  for _ in range(0, iterations):
    valid_floor = False
    valid_wall = False
    wall_aligned = False

    # Get the plane
    plane = pointcloud_utils.fit_ransac_plane(current_step_pointcloud, visualize=visualize_fit_ransac_plane, print_messages=False)

    # Check wether the plane is "horizontal" or "vertical"
    plane_normal = geometry.plane_normal(plane)

    if -NORMAL_VECTOR_THRESHOLD <= plane_normal[0] <= NORMAL_VECTOR_THRESHOLD and -NORMAL_VECTOR_THRESHOLD <= plane_normal[1] <= NORMAL_VECTOR_THRESHOLD:
      # Plane is horizontal. For a horizontal plane to make any sense (as we assume
      # the only horizontal planes are floors), we require the plane to be at the "bottom"
      # of the pointcloud, i.e. only a small percentage of points is allowed to be below
      # this plane. Otherwise, it's a wrong plane and should be discarded
      if print_messages:
        print('Horizontal')

      under_count = 0
      above_count = 0
      total = 0

      # Check how many points are "under" the floor plane
      for pt3D in tqdm(current_step_pointcloud.points, disable=(not print_messages)):
        # points.append(pt3D)
        total += 1
        if geometry.point_above_plane(pt3D, plane):
          # point_colors.append([0.0, 1.0, 0.0])
          above_count += 1
        else:
          # Check wether the point is close enough to the floor
          # to be still considered floor. If yes, we keep it,
          # otherwise the point is discarded
          distance = geometry.point_plane_distance(pt3D, plane)
          if distance < distance_threshold:
            pass
          else:
            under_count += 1

      percentage_under = float(under_count) / total
      percentage_under *= 100.0

      if print_messages:
        print(f"There are {under_count} points under the plane and {above_count} above of total {total}, for a total of {percentage_under}% under the plane")

      # If the percentage of points below the plane is higher than a certain threshold,
      # we consider the plane to be an invalid floor
      if percentage_under > POINTS_BELOW_FLOOR_THRESHOLD:
        if print_messages:
          print('Invalid floor, ignoring')
      else:
        valid_floor = True
    elif -NORMAL_VECTOR_THRESHOLD <= plane_normal[2] <= NORMAL_VECTOR_THRESHOLD:
      # Plane is vertical, need to check wether the plane is somewhat "parallel" to the
      # PCAs of the pointcloud. For that, we check wether the dot product of the normal
      # of the plane is "close enough" to 1 (parallel) or -1 (antiparallel, still okay)
      # of the first two pca vectors.
      if abs(np.dot(plane_normal, pcas[0])) > PCA_DOT_THRESHOLD or abs(np.dot(plane_normal, pcas[1])) > PCA_DOT_THRESHOLD:
        if print_messages:
          print('Vertical, aligned with PCA')
        wall_aligned = True
        valid_wall = True
      else:
        if print_messages:
          print('Vertical, NOT aligned with PCA')
        valid_wall = keep_misaligned_walls
    else:
      # Plane is crooked, ignore plane, Maybe we need to remove some points, such
      # that we don't get the same plane in the next iteration
      if print_messages:
        print('Other')

    remaining_points = []
    current_valid_points = []

    # Now, remove all points lying on (or within a threshold of) the plane
    for pt3D in tqdm(current_step_pointcloud.points, disable=(not print_messages)):
      distance_to_plane = geometry.point_plane_distance(pt3D, plane)
      if distance_to_plane > distance_threshold:
        remaining_points.append(pt3D)
      elif valid_floor and not floor_found:
        resulting_floor_points.append(pt3D)
        current_valid_points.append(pt3D)
      elif valid_wall:
        resulting_wall_points.append(pt3D)
        current_valid_points.append(pt3D)

    if (valid_floor and not floor_found) or valid_wall:
      axis_aligned_bboxes.append(o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(np.array(current_valid_points))))
      oriented_bboxes.append(o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(np.array(current_valid_points))))
      planes.append(plane)

    if valid_wall:
      if wall_aligned:
        color = [[1, 0, 0]] * len(current_valid_points)
      else:
        color = [[1, 1, 0]] * len(current_valid_points)
      wall_pointclouds.append(pointcloud_utils.pointcloud_from_points(current_valid_points, colors=color))

    if valid_floor and not floor_found:
      floor_pointcloud = pointcloud_utils.pointcloud_from_points(current_valid_points, colors=([[0, 0, 1]] * len(current_valid_points)))

    if valid_floor:
      floor_found = True

    # If there are less points than a threshold, we stop
    if len(remaining_points) < STOP_ITERATIONS_THRESHOLD:
      if print_messages:
        print(f"Stopping as we have less than {STOP_ITERATIONS_THRESHOLD} points")
      break

    current_step_pointcloud = pointcloud_utils.pointcloud_from_points(remaining_points)

  if print_messages:
    print('iterations over')

  final_points = []
  final_colors = []

  if create_ceiling:
    max_wall_z = -100000
    avg_floor_z = np.average(np.array(resulting_floor_points)[:, 2])

    for pt3D in resulting_wall_points:
      max_wall_z = max(max_wall_z, pt3D[2])

    distance = max_wall_z - avg_floor_z

    # Copy all floor points and shift them up to make the "ceiling"
    ceiling_points = []
    for pt3D in resulting_floor_points:
      dist = distance + pt3D[2]
      point = [pt3D[0], pt3D[1], dist]
      ceiling_points.append(point)

  # First add the floor points
  final_points += resulting_floor_points
  final_colors += repeat([1.0, 0.0, 0.0], len(resulting_floor_points))

  # And then add the wall points
  final_points += resulting_wall_points
  final_colors += repeat([0.0, 1.0, 0.0], len(resulting_wall_points))

  if create_ceiling:
    # And finally, the ceiling points
    final_points += ceiling_points
    final_colors += repeat([0.0, 0.0, 1.0], len(ceiling_points))

    # Create new AABB for ceiling points
    axis_aligned_bboxes.append(o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(np.array(ceiling_points))))
    oriented_bboxes.append(o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(np.array(ceiling_points))))
    ceiling_pcd = pointcloud_utils.pointcloud_from_points(ceiling_points)
    plane = pointcloud_utils.fit_ransac_plane(ceiling_pcd, visualize=False)
    planes.append(plane)

  final_pointcloud = pointcloud_utils.pointcloud_from_points(final_points, colors=final_colors)

  return axis_aligned_bboxes, oriented_bboxes, planes, final_pointcloud, wall_pointclouds, floor_pointcloud

def find_intersections_of_lines(pointcloud, planes, axis_aligned_bboxes, visualize_steps=False):
  pointcloud = pointcloud.uniform_down_sample(10)

  assert(len(planes) == len(axis_aligned_bboxes))

  bboxes = axis_aligned_bboxes

  x_lines = []
  y_lines = []
  z_lines = []

  pcd_bbox = pointcloud.get_axis_aligned_bounding_box()
  min_b = pcd_bbox.min_bound
  max_b = pcd_bbox.max_bound

  bboxes[0].min_bound = [min_b[0], min_b[1], bboxes[0].min_bound[2]]
  bboxes[0].max_bound = [max_b[0], max_b[1], bboxes[0].max_bound[2]]

  # --------------------------------------------------------------------------------------------------------------------------------------------
  #
  # Get lines and sort them in x, y and z heading lines.
  #
  # --------------------------------------------------------------------------------------------------------------------------------------------
  for i in range(len(planes)):
    for j in range(i + 1, len(planes), 1):
      point, direction = geometry.plane_plane_intersection(planes[i], planes[j])

      line_start = None
      line_end = None

      if point is None:
        # print('No intersection found')
        continue

      # Plot the line
      current = point.copy()
      unit_direction = geometry.normalized(direction)

      intersects_i, t_near_i, t_far_i = geometry.line_aabb_intersection(point, direction, bboxes[i])
      intersects_j, t_near_j, t_far_j = geometry.line_aabb_intersection(point, direction, bboxes[j])

      if not intersects_i and not intersects_j:
        continue

      # Move the "start" point into one of the two bounding boxes
      if intersects_i:
        start_point = point + (t_near_i * direction)
      elif intersects_j:
        start_point = point + (t_near_j * direction)
      else:
        raise RuntimeError('Line needs to intersect at least one of the bboxes!')

      current = start_point.copy()

      # This "sampling" approach probably also could be replaced by the line_aabb_intersection apporoach!
      while True:
        if geometry.point_within_bounding_box(current, bboxes[i]) or geometry.point_within_bounding_box(current, bboxes[j]):
          line_start = current
          current += unit_direction
        else:
          break

      current = start_point.copy()

      while True:
        if geometry.point_within_bounding_box(current, bboxes[i]) or geometry.point_within_bounding_box(current, bboxes[j]):
          line_end = current
          current -= unit_direction
        else:
          break

      if line_start is not None and line_end is not None:
        # Check wether the line is "heading" towards x or y
        argmax = np.argmax(np.abs(direction))

        if argmax == 0:
          x_lines.append([line_start, line_end])
        elif argmax == 1:
          y_lines.append([line_start, line_end])
        else:
          # For z-lines, we want to make sure the end is at least at "ground level",
          # i.e. the intersection of the ground plane and the line
          line_ground_intersection, _ = geometry.line_plane_intersection(line_start, geometry.normalized(line_end - line_start), planes[0])
          if line_ground_intersection is not None:
            if line_start[2] < line_end[2]:
              # Line start is the "lower" point, check wether the intersection
              # with the ground is even lower, if yes use this as the start
              if line_start[2] > line_ground_intersection[2]:
                line_start = line_ground_intersection
            else:
              # Line end is the "lower" point, check wether the intersection
              # with the ground is even lower, if yes use this as the end
              if line_end[2] > line_ground_intersection[2]:
                line_end = line_ground_intersection
            z_lines.append([line_start, line_end])

  if visualize_steps:
    linesets = []
    if x_lines:
      lineset_x = helpers.create_lineset_from_line_endpoints(x_lines)
      lineset_x.paint_uniform_color([1, 0, 0])
      linesets.append(lineset_x)
    if y_lines:
      lineset_y = helpers.create_lineset_from_line_endpoints(y_lines)
      lineset_y.paint_uniform_color([0, 1, 0])
      linesets.append(lineset_y)
    if z_lines:
      lineset_z = helpers.create_lineset_from_line_endpoints(z_lines)
      lineset_z.paint_uniform_color([0, 0, 1])
      linesets.append(lineset_z)

    o3d.visualization.draw_geometries([pointcloud] + linesets, window_name='Showing x, y and z linesets with pointcloud')
    o3d.visualization.draw_geometries(linesets, window_name='Showing x, y and z linesets')

  # --------------------------------------------------------------------------------------------------------------------------------------------
  #
  # Get the intersection points of the lines, sorting them in "close" intersection points (i.e. points which probably are an
  # intersection) and "far" intersection points (i.e. points which most likely are not intersections).
  #
  # --------------------------------------------------------------------------------------------------------------------------------------------
  close_intersection_points = []
  far_intersection_points = []

  for line_x in x_lines:
    for line_y in y_lines:
      point_a, point_b, distance = geometry.closest_distance_between_lines(line_x, line_y)
      if math.isclose(0.0, distance, abs_tol=1e-9):
        close_intersection_points.append(point_a)
      else:
        far_intersection_points.append(point_a)
        far_intersection_points.append(point_b)

  lineset_x = helpers.create_lineset_from_line_endpoints(x_lines)
  lineset_x.paint_uniform_color([1, 0, 0])
  lineset_y = helpers.create_lineset_from_line_endpoints(y_lines)
  lineset_y.paint_uniform_color([0, 1, 0])

  if visualize_steps:
    intersection_pc_close = pointcloud_utils.pointcloud_from_points(close_intersection_points, colors=([[1, 1, 0]] * len(close_intersection_points))) # Yellow
    if far_intersection_points:
      intersection_pc_far = pointcloud_utils.pointcloud_from_points(far_intersection_points, colors=([[0, 0, 1]] * len(far_intersection_points))) # Blue
      o3d.visualization.draw_geometries([lineset_x, lineset_y, intersection_pc_far, intersection_pc_close], window_name='Showing intersection points')
    else:
      o3d.visualization.draw_geometries([lineset_x, lineset_y, intersection_pc_close], window_name='Showing intersection points')

  # --------------------------------------------------------------------------------------------------------------------------------------------
  #
  # Remove all z points which don't have ANY points (including the floor) near them, as they probably are outside of the
  # usable space, and therefore can be removed safely
  #
  # --------------------------------------------------------------------------------------------------------------------------------------------
  pc_grid = FlatPointcloudGrid(pointcloud.voxel_down_sample(0.25), bin_count=30)
  grid_lineset = pc_grid.get_lineset()

  # For the boundary lineset, check for each line if there are ANY points near it, if not, it's probably
  # an outlier and should be discarded
  kept_z_lines = []
  discarded_z_lines = []

  for z_line in z_lines:
    line_start, line_end = z_line
    line_midpoint = (line_end + line_start) / 2
    count = pc_grid.get_point_neighbor_count(line_midpoint)
    if count > 1:
      kept_z_lines.append(z_line)
    else:
      discarded_z_lines.append(z_line)

  kept_z_lines_lineset = helpers.create_lineset_from_line_endpoints(kept_z_lines, color=[0, 1, 0])

  if visualize_steps:
    if len(discarded_z_lines) > 0:
      discarded_z_lines_lineset = helpers.create_lineset_from_line_endpoints(discarded_z_lines, color=[1, 0, 0])
      o3d.visualization.draw_geometries([kept_z_lines_lineset, discarded_z_lines_lineset, lineset_x, lineset_y], window_name='Showing removed (red) and kept (green) z-lines after checking wether they have points nearby')
    else:
      o3d.visualization.draw_geometries([kept_z_lines_lineset, lineset_x, lineset_y], window_name='Showing removed (red) and kept (green) z-lines after checking wether they have points nearby')

  # --------------------------------------------------------------------------------------------------------------------------------------------
  #
  # Filter the intersection points, removing points where we don't have a "corresponding" z-line (i.e. a z-line
  # which is **really** close to the intersection).
  #
  # --------------------------------------------------------------------------------------------------------------------------------------------
  kept_intersection_points = []
  clostest_z_lines = []
  discarded_intersection_points = []

  for intersection_point in close_intersection_points:
    closest_line = None
    closest_distance = sys.float_info.max

    for line in kept_z_lines:
      point, _, distance = geometry.closest_distance_between_lines(line, [intersection_point, intersection_point + np.array([0.01, 0.01, 0.01])])
      if distance < closest_distance:
        closest_distance = distance
        closest_line = line

    if math.isclose(0.0, closest_distance, abs_tol=1e-9):
      kept_intersection_points.append(intersection_point)
      clostest_z_lines.append(closest_line)
    else:
      discarded_intersection_points.append(intersection_point)

  if visualize_steps:
    lineset_z = helpers.create_lineset_from_line_endpoints(clostest_z_lines)
    lineset_z.paint_uniform_color([0, 0, 1])
    kept_intersection_points_pcd = pointcloud_utils.pointcloud_from_points(kept_intersection_points, colors=([[0, 0, 1]] * len(kept_intersection_points)))
    discarded_intersection_point_pcd = pointcloud_utils.pointcloud_from_points(discarded_intersection_points, colors=([[1, 0, 0]] * len(discarded_intersection_points)))
    o3d.visualization.draw_geometries([lineset_x, lineset_y, kept_intersection_points_pcd, discarded_intersection_point_pcd, lineset_z], window_name='Showing kept (blue) and discarded (red) intersection points')

  # --------------------------------------------------------------------------------------------------------------------------------------------
  #
  # Filter the lines, such that each line only exists between their "outermost" intersection points. For this, we need to keep
  # track for each line which intersection points belong to it, but for now, we just check for each line which points lie on the line.
  #
  # --------------------------------------------------------------------------------------------------------------------------------------------
  new_x_line_points = []
  new_y_line_points = []

  for x_line in x_lines:
    relevant_points = []
    for intersection_point in kept_intersection_points:
      if geometry.point_between_other_two_on_line(intersection_point, x_line[0], x_line[1]):
        relevant_points.append(intersection_point)

    # Now that we have the relevant points, get the two that are the "furthest out"
    # of all the points. For that we just get the one with the shortest distance
    # to the end, and the one with the shortest distance to the start
    found_end = None
    found_start = None
    end_distance = 10000000
    start_distance = 10000000

    for point in relevant_points:
      new_start_distance = geometry.point_point_distance(point, x_line[0])
      new_end_distance = geometry.point_point_distance(point, x_line[1])

      if start_distance > new_start_distance:
        start_distance = new_start_distance
        found_start = point

      if end_distance > new_end_distance:
        end_distance = new_end_distance
        found_end = point

    if found_start is not None and found_end is not None and not np.array_equal(found_start, found_end):
      new_x_line_points.append([found_start, found_end])

  for y_line in y_lines:
    relevant_points = []
    for intersection_point in kept_intersection_points:
      if geometry.point_between_other_two_on_line(intersection_point, y_line[0], y_line[1]):
        relevant_points.append(intersection_point)

    # Now that we have the relevant points, get the two that are the "furthest out"
    # of all the points. For that we just get the one with the shortest distance
    # to the end, and the one with the shortest distance to the start
    found_end = None
    found_start = None
    end_distance = 10000000
    start_distance = 10000000

    for point in relevant_points:
      new_start_distance = geometry.point_point_distance(point, y_line[0])
      new_end_distance = geometry.point_point_distance(point, y_line[1])

      if start_distance > new_start_distance:
        start_distance = new_start_distance
        found_start = point

      if end_distance > new_end_distance:
        end_distance = new_end_distance
        found_end = point

    if found_start is not None and found_end is not None and not np.array_equal(found_start, found_end):
      new_y_line_points.append([found_start, found_end])

  if visualize_steps:
    new_lineset_x = helpers.create_lineset_from_line_endpoints(new_x_line_points)
    new_lineset_x.paint_uniform_color([1, 0, 0])
    new_lineset_y = helpers.create_lineset_from_line_endpoints(new_y_line_points)
    new_lineset_y.paint_uniform_color([0, 1, 0])

    o3d.visualization.draw_geometries([new_lineset_x, new_lineset_y, pointcloud], window_name='Showing lines bound to outermost intersection points with pointcloud')
    o3d.visualization.draw_geometries([new_lineset_x, new_lineset_y], window_name='Showing lines bound to outermost intersection points')

  # --------------------------------------------------------------------------------------------------------------------------------------------
  #
  # Filter z-lines
  #
  # --------------------------------------------------------------------------------------------------------------------------------------------
  boundary_z_lines = []
  other_z_lines = []

  # For each z-line, check wether is is "close" to an end or start of a x or y line, if
  # yes, it's probably part of the boundary and should be kept in any case
  for z_line in kept_z_lines:
    boundary = False
    for other_line in new_x_line_points + new_y_line_points:
      _, _, distance_start = geometry.closest_distance_between_lines(z_line, [other_line[0], other_line[0] + np.array([0.01, 0.01, 0.01])])
      _, _, distance_end = geometry.closest_distance_between_lines(z_line, [other_line[1], other_line[1] + np.array([0.01, 0.01, 0.01])])

      if distance_start < 0.1 or distance_end < 0.1:
        boundary_z_lines.append(z_line)
        boundary = True
        break

    if not boundary:
      other_z_lines.append(z_line)


  boundary_lineset = helpers.create_lineset_from_line_endpoints(boundary_z_lines, color=[0, 0, 1])

  # o3d.visualization.draw_geometries([boundary_lineset, other_lineset, new_lineset_x, new_lineset_y])

  floor_plane = planes[0]

  walls_points = []

  for pt3D in pointcloud.points:
    distance_to_plane = geometry.point_plane_distance(pt3D, floor_plane)
    if distance_to_plane > 0.5 and geometry.point_above_plane(pt3D, floor_plane):
      walls_points.append(pt3D)

  wall_pcd = pointcloud_utils.pointcloud_from_points(walls_points, colors=([[0.5, 0.5, 0.5]] * len(walls_points)))
  wall_pcd.voxel_down_sample(0.25)

  pc_grid = FlatPointcloudGrid(wall_pcd, bin_count=50)

  low_z = []
  high_z = []

  # We should keep the lines which are "low", but are the "outermost"
  # line on another line, to keep the "outline" of the building intact.

  # IMPROVEMENT: check that both walls contain at least N points near the intersection
  for z_line in other_z_lines:
    line_start, line_end = z_line
    line_midpoint = (line_end + line_start) / 2
    count = pc_grid.get_point_neighbor_count(line_midpoint)
    if count > 200:
      high_z.append(z_line)
    else:
      low_z.append(z_line)

  if visualize_steps:
    z_linesets = []

    if len(low_z) > 0:
      z_linesets.append(helpers.create_lineset_from_line_endpoints(low_z, color=[1, 0, 1]))

    if len(high_z):
      z_linesets.append(helpers.create_lineset_from_line_endpoints(high_z, color=[0, 1, 0]))

    o3d.visualization.draw_geometries([boundary_lineset, new_lineset_x, new_lineset_y, *z_linesets], window_name='Showing boundary z (blue), high-count z (green) and low-count z (purple)')
    o3d.visualization.draw_geometries([boundary_lineset, new_lineset_x, new_lineset_y, pointcloud, *z_linesets], window_name='Showing boundary z (blue), high-count z (green) and low-count z (purple)')

  kept_z_lines = high_z + boundary_z_lines

  # --------------------------------------------------------------------------------------------------------------------------------------------
  #
  # Divide the lines into individual segments, "throwing away" the segment if either the start or the end are not near one of the "kept_z_lines"
  #
  # --------------------------------------------------------------------------------------------------------------------------------------------
  line_segments = []
  line_colors = []
  intersection_points = []

  for line in new_x_line_points + new_y_line_points:
    relevant_points = []
    for other_line in new_x_line_points + new_y_line_points:
      if np.array_equal(line, other_line):
        continue
      intersection_point, _, distance = geometry.closest_distance_between_lines(line, other_line)

      # TODO: use math.isclose
      if distance > 0.1:
        continue
      curr = {
        "point": intersection_point,
        "dist": geometry.point_point_distance(intersection_point, line[0])
      }
      relevant_points.append(curr)

    # Order the points by their distance to the "start" point
    sorted_relevant_points = sorted(relevant_points, key=lambda d: d['dist'])

    # Only keep the points, don't need the whole thing anymore
    sorted_relevant_points = [d['point'] for d in sorted_relevant_points]

    intersection_points += sorted_relevant_points

    current_start = sorted_relevant_points[0]

    for current_end in sorted_relevant_points[1:]:
      color = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
      line_segments.append([current_start, current_end])
      line_colors.append(color)
      current_start = current_end

  kept_segments = []
  discarded_segments = []

  for line_segment in line_segments:
    found_close_end = False
    found_close_start = False
    for z_line in kept_z_lines:
      _, _, distance_start = geometry.closest_distance_between_lines(z_line, [line_segment[0], line_segment[0] + np.array([0.01, 0.01, 0.01])])
      _, _, distance_end = geometry.closest_distance_between_lines(z_line, [line_segment[1], line_segment[1] + np.array([0.01, 0.01, 0.01])])

      if distance_start < 0.1:
        found_close_start = True

      if distance_end < 0.1:
        found_close_end = True

    if found_close_start and found_close_end:
      kept_segments.append(line_segment)
    else:
      discarded_segments.append(line_segment)

  if visualize_steps:
    z_linesets = []

    if len(high_z):
      z_linesets.append(helpers.create_lineset_from_line_endpoints(high_z, color=[0, 1, 0]))

    if len(kept_segments):
      z_linesets.append(helpers.create_lineset_from_line_endpoints(kept_segments, color=[0, 1, 0]))

    if len(discarded_segments):
      z_linesets.append(helpers.create_lineset_from_line_endpoints(discarded_segments, color=[1, 0, 0]))

    intersection_points_pcd = pointcloud_utils.pointcloud_from_points(intersection_points, colors=([[1, 0, 0]] * len(intersection_points)))
    o3d.visualization.draw_geometries([boundary_lineset, *z_linesets, intersection_points_pcd, pointcloud], window_name='Showing kept lines (green) and discarded lines (red) with pointcloud')
    o3d.visualization.draw_geometries([boundary_lineset, *z_linesets, intersection_points_pcd], window_name='Showing kept lines (green) and discarded lines (red)')

    # new_z_lineset = helpers.create_lineset_from_line_endpoints(kept_z_lines, color=[1, 0, 0])
    # o3d.visualization.draw_geometries([kept_lineset, new_z_lineset], window_name='Showing kepy lineset only')

  # --------------------------------------------------------------------------------------------------------------------------------------------
  #
  # Get boundary segments, which should NOT be filtered out in any case.
  #
  # --------------------------------------------------------------------------------------------------------------------------------------------
  x_segments = []
  y_segments = []

  for segment in kept_segments:
    direction = segment[1] - segment[0]
    argmax = np.argmax(np.abs(direction))

    if argmax == 0:
      x_segments.append(segment)
    elif argmax == 1:
      y_segments.append(segment)
    else:
      raise RuntimeError('Implausible')

  # For each line, sample points in a certain distance.
  # From this, we make a pointcloud, which we then plug into a grid.
  # With this grid, we can then check for each (sampled) line, wether
  # there are any lines occupying the cells in the orthogonal directions
  # of the line. If we find a direction where most are unoccupied,
  # we probably have a boundary line.
  # ------------------------------------------------------------------------------
  sampled_points = []

  for segment in x_segments:
    direction = geometry.normalized(segment[1] - segment[0]) / 4

    if 0 in direction:
      continue

    count = ((segment[1] - segment[0]) / direction)[0]
    current = segment[0].copy()

    for _ in range(int(count)):
      sampled_points.append(current)
      current = current.copy() + direction

    sampled_points.append(segment[1])

  sampled_lines = pointcloud_utils.pointcloud_from_points(sampled_points, colors=([[0.5, 0.5, 0.5]] * len(sampled_points)))

  padding = np.min(pointcloud.get_axis_aligned_bounding_box().get_extent()) / 30
  sampled_grid = FlatPointcloudGrid(sampled_lines, 50, padding=padding)

  bound = []
  insi = []

  for segment in x_segments:
    # Check whether the bin before the line start is occupied,
    # and wherther the bin after the end is occupied
    line_midpoint = (segment[1] + segment[0]) / 2
    line_midpoint[2] = 0

    line_x_bin, line_y_bin = sampled_grid.get_bin_indices(line_midpoint)

    up_counter = 0
    down_counter = 0

    for i in range(line_y_bin):
      count = sampled_grid.get_bin_item_count(line_x_bin, i)
      down_counter += count

    for i in range(line_y_bin + 1, sampled_grid.y_bins, 1):
      count = sampled_grid.get_bin_item_count(line_x_bin, i)
      up_counter += count

    if down_counter == 0 or up_counter == 0:
      bound.append(segment)
    else:
      insi.append(segment)

  # ------------------------------------------------------------------------------
  sampled_points = []

  for segment in y_segments:
    direction = geometry.normalized(segment[1] - segment[0]) / 4
    count = ((segment[1] - segment[0]) / direction)[0]
    current = segment[0].copy()

    for _ in range(int(count)):
      sampled_points.append(current)
      current = current.copy() + direction

    sampled_points.append(segment[1])

  sampled_lines = pointcloud_utils.pointcloud_from_points(sampled_points, colors=([[0.5, 0.5, 0.5]] * len(sampled_points)))

  padding = np.min(pointcloud.get_axis_aligned_bounding_box().get_extent()) / 30
  sampled_grid = FlatPointcloudGrid(sampled_lines, 50, padding=padding)

  sampling_points = []

  for segment in y_segments:
    line_midpoint = (segment[1] + segment[0]) / 2
    line_midpoint[2] = 0
    line_x_bin, line_y_bin = sampled_grid.get_bin_indices(line_midpoint)

    up_counter = 0
    down_counter = 0

    for i in range(line_x_bin):
      count = sampled_grid.get_bin_item_count(i, line_y_bin)
      point = sampled_grid.get_middlepoint_of_bin(i, line_y_bin)
      sampling_points.append(point)
      down_counter += count

    for i in range(line_x_bin + 1, sampled_grid.x_bins, 1):
      count = sampled_grid.get_bin_item_count(i, line_y_bin)
      point = sampled_grid.get_middlepoint_of_bin(i, line_y_bin)
      sampling_points.append(point)
      up_counter += count

    if down_counter == 0 or up_counter == 0:
      bound.append(segment)
    else:
      insi.append(segment)

  if visualize_steps:
    boundset = helpers.create_lineset_from_line_endpoints(bound, color=[0.3, 0.3, 0.3])
    insiset = helpers.create_lineset_from_line_endpoints(insi, color=[0, 0, 1])

    o3d.visualization.draw_geometries([insiset, boundset], window_name='Showing inside lineset (blue) and boundary set (grey)')

  # --------------------------------------------------------------------------------------------------------------------------------------------
  #
  # Apply the "grid filter" to the individual "segments" made by the lines. However, this should probably only be applied to
  # line_segments which are not considered a "boundary" segment.
  #
  # --------------------------------------------------------------------------------------------------------------------------------------------
  floor_plane = planes[0]

  walls_points = []

  for pt3D in pointcloud.points:
    distance_to_plane = geometry.point_plane_distance(pt3D, floor_plane)
    if distance_to_plane > 0.5 and geometry.point_above_plane(pt3D, floor_plane):
      walls_points.append(pt3D)

  wall_pcd = pointcloud_utils.pointcloud_from_points(walls_points, colors=([[0.5, 0.5, 0.5]] * len(walls_points)))
  wall_pcd.voxel_down_sample(0.25)

  pc_grid = FlatPointcloudGrid(wall_pcd, bin_count=100)
  grid_lineset = pc_grid.get_lineset()

  kept_line_segments = []
  kept_segment_colors = []

  for line_segment in insi:
    # Sample along the axis and get for each sample point the indices of the bin.
    current = line_segment[0].copy()

    idxset = set()

    vect = (line_segment[1] - line_segment[0]) / 20

    for _ in range(20):
      x_bin, y_bin = pc_grid.get_bin_indices(current)
      idxset.add((x_bin, y_bin))
      current += vect

    idxlen = len(idxset)

    total_count = 0

    zero_count_bin = 0
    high_count_bin = 0
    low_count_bin = 0

    for x_bin, y_bin in idxset:
      count = pc_grid.get_bin_item_count(x_bin, y_bin)
      total_count += count
      if count == 0:
        zero_count_bin += 1
      elif count < 20:
        low_count_bin += 1
      else:
        high_count_bin += 1

    total_count /= idxlen

    if high_count_bin > 0:
      zero_to_nonzero_fraction = zero_count_bin / (low_count_bin + high_count_bin)
      low_to_high_fraction = low_count_bin / high_count_bin
    else:
      zero_to_nonzero_fraction = 1
      low_to_high_fraction = 1

    line_segment_flat = [[line_segment[0][0], line_segment[0][1], 0.0], [line_segment[1][0], line_segment[1][1], 0.0]]

    # Red: too many zero bins
    # orange: too many low bins compared to high bins (outlier high bins)
    # green: ok
    # kept_line_segments.append(line_segment_flat)

    if zero_to_nonzero_fraction >= 0.5:
      # kept_segment_colors.append([1, 0, 0])
      pass
    elif low_to_high_fraction >= 0.25:
      # kept_segment_colors.append([1, 0.65, 0])
      pass
    else:
      kept_line_segments.append(line_segment)
      kept_segment_colors.append([0, 1, 0])


  boundset = helpers.create_lineset_from_line_endpoints(bound, color=[0.3, 0.3, 0.3])

  if len(kept_line_segments) > 0:
    kept_lineset = helpers.create_lineset_from_line_endpoints(kept_line_segments)
    kept_lineset.colors = o3d.utility.Vector3dVector(np.array(kept_segment_colors))
  else:
    kept_lineset = None


  if visualize_steps:
    if kept_lineset is None:
      o3d.visualization.draw_geometries([boundset], window_name='Showing kept lines (green) after grid filter and boundary lines (grey)')
    else:
      o3d.visualization.draw_geometries([boundset, kept_lineset], window_name='Showing kept lines (green) after grid filter and boundary lines (grey)')

  if kept_lineset is None:
    final_line_segments = bound
  else:
    final_line_segments = kept_line_segments + bound

  final_lineset = helpers.create_lineset_from_line_endpoints(final_line_segments)

  if visualize_steps:
    o3d.visualization.draw_geometries([final_lineset], window_name='Showing final lineset')

  return final_lineset, kept_lineset, boundset
