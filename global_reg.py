import open3d as o3d
import copy
import numpy as np
import ransac_test
from utils import geometry, pointcloud_utils
import math
import pca_methods
import scipy
from tqdm import tqdm
import pointcloud_generation
import warnings

def flip_floor_normal_if_needed(point_cloud, normal, floor_plane):
  POINTS_COUNT = 300

  # Get POINTS_COUNT random points, and check wether they are in the direction of the normal
  # away from the plane or the other. If the majority is in the negative direction,
  # we need to flip the normal
  selected_points = np.asarray(point_cloud.points)[np.random.choice(len(point_cloud.points), size=POINTS_COUNT, replace=False), :]

  above_counter = 0

  for pt3D in selected_points:
    if geometry.point_above_plane(pt3D, floor_plane):
      above_counter += 1

  # Flip normal if less than 50% of the points above
  if (above_counter / POINTS_COUNT) < 0.5:
    return geometry.stretch_vector(normal, -1)
  else:
    return normal

def find_optimal_global_fit(building_cad_model_pointcloud, captured_pointcloud, building_cad_model_floor_pointcloud=None, captured_floor_pointcloud=None, visualize=True):
  if building_cad_model_floor_pointcloud is None:
    _, _, building_cad_model_planes, _, _, _ = ransac_test.find_planes(building_cad_model_pointcloud, 1, create_ceiling=False, print_messages=False)
    building_cad_model_plane = building_cad_model_planes[0]
  else:
    building_cad_model_plane = pointcloud_utils.fit_ransac_plane(building_cad_model_floor_pointcloud, visualize=False, print_messages=False)

  if captured_floor_pointcloud is None:
    _, _, captured_pointcloud_planes, _, _, _ = ransac_test.find_planes(captured_pointcloud, 1, create_ceiling=False, print_messages=False)
    captured_pointcloud_plane = captured_pointcloud_planes[0]
  else:
    captured_pointcloud_plane = pointcloud_utils.fit_ransac_plane(captured_floor_pointcloud, visualize=False, print_messages=False)

  # -- Rough aligning -------------------------------------------------------------------------------------------------------------

  # --------------------------------------------------------------------------------------------------------------------------------------------
  #
  # Orient ground planes, such that they lie on the same plane
  #
  # --------------------------------------------------------------------------------------------------------------------------------------------
  rough_align_captured_pointcloud_copy = copy.deepcopy(captured_pointcloud)

  # Get normal vector of building_cad_model_pointcloud and captured_pointcloud
  building_cad_model_pointcloud_normal = geometry.plane_normal(building_cad_model_plane)
  captured_pointcloud_normal = geometry.plane_normal(captured_pointcloud_plane)

  # Check for both normals wether they point into the right direction or if they need to be flipped
  captured_pointcloud_normal = flip_floor_normal_if_needed(captured_pointcloud, captured_pointcloud_normal, captured_pointcloud_plane)
  building_cad_model_pointcloud_normal = flip_floor_normal_if_needed(building_cad_model_pointcloud, building_cad_model_pointcloud_normal, building_cad_model_plane)

  warnings.filterwarnings(action='ignore', category=UserWarning)
  rotation_scipi, _ = scipy.spatial.transform.Rotation.align_vectors([building_cad_model_pointcloud_normal], [captured_pointcloud_normal])
  warnings.filterwarnings(action='default', category=UserWarning)
  rotmat_scipi = rotation_scipi.as_matrix()

  rough_align_captured_pointcloud_copy.rotate(rotmat_scipi)
  captured_pointcloud_normal_rotated = rotmat_scipi.dot(captured_pointcloud_normal)

  flipped = False
  additional_rotmat = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
  ])

  # Check wether we need to flip the rotation matrix
  if math.isclose(np.dot(building_cad_model_pointcloud_normal, captured_pointcloud_normal_rotated), -1, abs_tol=0.01):
    print('need to flip!')
    flipped = True
    # If yes, rotate the pcd about 180 degrees on the x-axis

    rough_align_captured_pointcloud_copy.rotate(additional_rotmat)

  # o3d.visualization.draw_geometries([building_cad_model_pointcloud, rough_align_captured_pointcloud_copy], window_name='Alignment after aligning floor planes')

  building_cad_model_pointcloud_pcas = pca_methods.get_pcas_from_first_n_walls(building_cad_model_pointcloud, 4, print_messages=False)
  captured_pointcloud_pcas = pca_methods.get_pcas_from_first_n_walls(rough_align_captured_pointcloud_copy, 4, print_messages=False)

  relevant_building_cad_model_pointcloud_pcas = building_cad_model_pointcloud_pcas
  relevant_captured_pointcloud_pcas = captured_pointcloud_pcas

  relevant_building_cad_model_pointcloud_pcas[2] = building_cad_model_pointcloud_normal
  relevant_captured_pointcloud_pcas[2] = captured_pointcloud_normal_rotated

  rotmat_scipi_2, _ = scipy.spatial.transform.Rotation.align_vectors(relevant_building_cad_model_pointcloud_pcas, relevant_captured_pointcloud_pcas)
  rotmat_scipi_2 = rotmat_scipi_2.as_matrix()
  rough_align_captured_pointcloud_copy.rotate(rotmat_scipi_2)

  if flipped:
    rotation = np.matmul(rotmat_scipi_2, np.matmul(additional_rotmat, rotmat_scipi))
  else:
    rotation = np.matmul(rotmat_scipi_2, rotmat_scipi)

  center_align_vector = captured_pointcloud.get_center()

  rot4x4 = np.array([
    [rotation[0, 0], rotation[0, 1], rotation[0, 2], 0],
    [rotation[1, 0], rotation[1, 1], rotation[1, 2], 0],
    [rotation[2, 0], rotation[2, 1], rotation[2, 2], 0],
    [0,              0,              0,              1]
  ])

  translate4x4 = np.array([
    [1, 0, 0, center_align_vector[0]],
    [0, 1, 0, center_align_vector[1]],
    [0, 0, 1, center_align_vector[2]],
    [0, 0, 0, 1             ]
  ])

  negative_translate4x4 = np.array([
    [1, 0, 0, -center_align_vector[0]],
    [0, 1, 0, -center_align_vector[1]],
    [0, 0, 1, -center_align_vector[2]],
    [0, 0, 0, 1              ]
  ])

  planes_align_transformation = np.matmul(translate4x4, np.matmul(rot4x4, negative_translate4x4))

  # -- Rough aligning done --------------------------------------------------------------------------------------------------------

  # o3d.visualization.draw_geometries([building_cad_model_pointcloud, rough_align_captured_pointcloud_copy], window_name='Alignment after rough aligning')

  # -- Initial scaling based on height --------------------------------------------------------------------------------------------

  cad_height = pointcloud_generation.resolve_z_height(building_cad_model_pointcloud, floor_plane=building_cad_model_plane, print_messages=False)
  captured_height = pointcloud_generation.resolve_z_height(rough_align_captured_pointcloud_copy, print_messages=False)
  initial_scale = cad_height / captured_height

  # -- More fine grained aligning -------------------------------------------------------------------------------------------------

  fine_grained_align_captured_pointcloud_copy = copy.deepcopy(captured_pointcloud)
  fine_grained_align_captured_pointcloud_copy.transform(planes_align_transformation)

  bbox_building_cad_model_pointcloud = building_cad_model_pointcloud.get_axis_aligned_bounding_box()
  bbox_captured_pointcloud = fine_grained_align_captured_pointcloud_copy.get_axis_aligned_bounding_box()

  move_length = bbox_building_cad_model_pointcloud.get_max_bound() - bbox_captured_pointcloud.get_max_bound()

  rotation_matrix_90_deg = np.array([
    [0, -1, 0],
    [1,  0, 0],
    [0, 0,  1]
  ])

  max_fitness_steps = []

  # Setup ranges. Y scale range is always relative to x_scale and contains
  # 11 entries (5 smaller, same value, 5 larger).
  x_move_steps = 30
  y_move_steps = 30

  # Scale range can be rather small, the height was already fitted in. We're using a multiplicative range,
  # i.e. the value will be multiplied with the initial scale, in steps of 2 percents
  x_scale_range = np.arange(0.84, 1.18, 0.02)
  x_move_range = range(0, x_move_steps + 1, 1)
  y_move_range = range(0, y_move_steps + 1, 1)

  # 4 iterations for rotation by 90 deg, 5 iterations for the y_scale
  total_iterations = 4 * len(x_scale_range) * 5 * len(x_move_range) * len(y_move_range)

  building_cad_model_pointcloud_ds = building_cad_model_pointcloud.voxel_down_sample(np.max(bbox_building_cad_model_pointcloud.get_extent()) / 100)
  fine_grained_align_captured_pointcloud_copy_ds = fine_grained_align_captured_pointcloud_copy.voxel_down_sample(np.max(bbox_captured_pointcloud.get_extent()) / 100)

  captured_pointcloud_rotation_loop_copy = copy.deepcopy(fine_grained_align_captured_pointcloud_copy_ds)

  BBOX_THRESHOLD = 1.1

  # Visualization only
  if visualize:
    live_vis = o3d.visualization.Visualizer()
    live_vis.create_window()
    live_vis.add_geometry(building_cad_model_pointcloud_ds)

    optimal_vis = o3d.visualization.Visualizer()
    optimal_vis.create_window()
    optimal_vis.add_geometry(building_cad_model_pointcloud_ds)

  with tqdm(total=total_iterations) as pbar:
    # Rotate by 90, 180 and 270 degrees, as the directions of the pointclouds
    # might not agree.
    for rotation_count in range(4):
      optimal = None

      max_fitness = -1
      max_fitness_step = None

      for x_scale_factor in x_scale_range:
        for y_scale_factor in np.arange(x_scale_factor - 0.2, x_scale_factor + 0.3, 0.1):
          x_scale = x_scale_factor * initial_scale
          y_scale = y_scale_factor * initial_scale

          # Z-scale stays!
          transformation_matrix = np.array([
            [x_scale, 0,       0,             0],
            [0,       y_scale, 0,             0],
            [0,       0,       initial_scale, 0],
            [0,       0,       0,             1]
          ])

          # Make copy
          captured_pointcloud_scale_loop_copy = copy.deepcopy(captured_pointcloud_rotation_loop_copy)

          # Scale
          captured_pointcloud_scale_loop_copy.transform(transformation_matrix)

          bbox = captured_pointcloud_scale_loop_copy.get_axis_aligned_bounding_box()
          bbox_extent = bbox.get_extent()
          bbox_source_extent = bbox_building_cad_model_pointcloud.get_extent()

          # Discard too large ones
          if (bbox_extent[0] / bbox_source_extent[0]) > BBOX_THRESHOLD or (bbox_extent[1] / bbox_source_extent[1]) > BBOX_THRESHOLD:
            pbar.update(len(x_move_range) * len(y_move_range))
            continue

          # Move
          bbox_captured_pointcloud_scale_loop_copy = captured_pointcloud_scale_loop_copy.get_axis_aligned_bounding_box()
          min_bound_vect = bbox_building_cad_model_pointcloud.get_min_bound() - bbox_captured_pointcloud_scale_loop_copy.get_min_bound()
          captured_pointcloud_scale_loop_copy.translate(min_bound_vect)

          move_length = bbox_building_cad_model_pointcloud.get_max_bound() - captured_pointcloud_scale_loop_copy.get_axis_aligned_bounding_box().get_max_bound()
          move_x_length = move_length[0]
          move_x_step = move_x_length / x_move_steps # OPTIMIZATION: Enforce a minimal length for this, as to not make too many very small steps
          move_x_vect = np.array([move_x_step, 0, 0])

          move_y_length = move_length[1]
          move_y_step = move_y_length / y_move_steps
          move_y_vect = np.array([0, move_y_step, 0])

          if move_x_length < 0 or move_y_length < 0:
            pbar.update(len(x_move_range) * len(y_move_range))
            continue

          for x_move_count in x_move_range:
            captured_pointcloud_move_x_loop_copy = copy.deepcopy(captured_pointcloud_scale_loop_copy)
            captured_pointcloud_move_x_loop_copy.translate(x_move_count * move_x_vect)

            for y_move_count in y_move_range:
              captured_pointcloud_move_y_loop_copy = copy.deepcopy(captured_pointcloud_move_x_loop_copy)
              captured_pointcloud_move_y_loop_copy.translate(y_move_count * move_y_vect)

              # Visualization only
              if visualize:
                live_vis.add_geometry(captured_pointcloud_move_y_loop_copy)
                live_vis.poll_events()
                live_vis.update_renderer()

              evaluation_fitness = get_evaluation_fitness(building_cad_model_pointcloud_ds, captured_pointcloud_move_y_loop_copy)

              if evaluation_fitness > max_fitness:
                max_fitness = evaluation_fitness
                # print(max_fitness)
                max_fitness_step = [x_scale, y_scale, (x_move_count * move_x_vect), (y_move_count * move_y_vect), rotation_count]

                # Visualization only
                if visualize and optimal is not None:
                    optimal_vis.remove_geometry(optimal)

                # Visualization only
                if visualize:
                  optimal = copy.deepcopy(captured_pointcloud_move_y_loop_copy)
                  optimal_vis.add_geometry(optimal)
                  optimal_vis.poll_events()
                  optimal_vis.update_renderer()

              # Visualization only
              if visualize:
                live_vis.remove_geometry(captured_pointcloud_move_y_loop_copy)

              pbar.update(1)

      captured_pointcloud_rotation_loop_copy.rotate(rotation_matrix_90_deg)
      # Visualization only
      if visualize:
        optimal_vis.remove_geometry(optimal)

      if max_fitness_step is not None:
        max_fitness_steps.append([max_fitness_step, max_fitness])

  # Visualization only
  if visualize:
    optimal_vis.destroy_window()
    live_vis.destroy_window()

  # --------------------------------------------------------------------------------------
  #  Compute transformation

  max_step_value = -1
  max_step_transform = None

  for max_fitness_step, fitness_value in max_fitness_steps:
    final_x_scale, final_y_scale, final_x_move, final_y_move, rotation_count = max_fitness_step

    scaling_matrix = np.array([
      [final_x_scale, 0,             0,             0],
      [0,             final_y_scale, 0,             0],
      [0,             0,             initial_scale, 0],
      [0,             0,             0,             1]
    ])

    # Initial rotate about z axis
    if rotation_count == 0:
      rotation_matrix = np.identity(4)
    else:
      if rotation_count == 1:
        angle = math.pi / 2
      elif rotation_count == 2:
        angle = math.pi
      elif rotation_count == 3:
        angle = 1.5 * math.pi

      rotation_matrix = np.array([
        [math.cos(angle), -math.sin(angle), 0, 0],
        [math.sin(angle),  math.cos(angle), 0, 0],
        [0,                0,               1, 0],
        [0,                0,               0, 1]
      ])

    intermediate_transform = np.matmul(np.matmul(scaling_matrix, rotation_matrix), planes_align_transformation)

    intermediate_result = copy.deepcopy(captured_pointcloud)
    intermediate_result.transform(intermediate_transform)

    intermediate_result_bbox = intermediate_result.get_axis_aligned_bounding_box()
    min_bound_vect_2 = bbox_building_cad_model_pointcloud.get_min_bound() - intermediate_result_bbox.get_min_bound()

    # Move
    translation_min_bound_vec = np.array([
      [1, 0, 0, min_bound_vect_2[0]],
      [0, 1, 0, min_bound_vect_2[1]],
      [0, 0, 1, min_bound_vect_2[2]],
      [0, 0, 0, 1]
    ])

    final_x_move_mat = np.array([
      [1, 0, 0, final_x_move[0]],
      [0, 1, 0, final_x_move[1]],
      [0, 0, 1, final_x_move[2]],
      [0, 0, 0, 1]
    ])

    final_y_move_mat = np.array([
      [1, 0, 0, final_y_move[0]],
      [0, 1, 0, final_y_move[1]],
      [0, 0, 1, final_y_move[2]],
      [0, 0, 0, 1]
    ])

    final_transformation = np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(final_y_move_mat, final_x_move_mat), translation_min_bound_vec), scaling_matrix), rotation_matrix), planes_align_transformation)

    if fitness_value > max_step_value:
      max_step_value = fitness_value
      max_step_transform = final_transformation

  # Visualization only
  if visualize:
    best_result = copy.deepcopy(captured_pointcloud)
    best_result.transform(max_step_transform)
    o3d.visualization.draw_geometries([building_cad_model_pointcloud, best_result], window_name="Best result")

  return max_step_transform, max_step_value

def get_evaluation_fitness(building_cad_model_pointcloud, captured_pointcloud, transformation=None):
  evaluation_threshold = building_cad_model_pointcloud.get_axis_aligned_bounding_box().get_extent()[0] / 50

  if transformation is None:
    transformation = np.identity(4)

  evaluation = o3d.pipelines.registration.evaluate_registration(building_cad_model_pointcloud, captured_pointcloud, evaluation_threshold, transformation)
  return evaluation.fitness

def local_optimization(building_cad_model_pointcloud, captured_pointcloud, transformation):
  building_cad_model_pointcloud_transformed = copy.deepcopy(building_cad_model_pointcloud).transform(transformation)

  # This proved to be a sensible threshold
  threshold = building_cad_model_pointcloud_transformed.get_axis_aligned_bounding_box().get_max_extent() / 50

  # Keep track of the best transformation
  best_transformation = None
  best_fitness = 0

  for scale_x in np.arange(0.95, 1.06, 0.01):
    for scale_y in np.arange(0.95, 1.06, 0.01):
      scale_copy = copy.deepcopy(building_cad_model_pointcloud_transformed)

      scaling_matrix = np.array([
        [scale_x, 0,       0, 0],
        [0,       scale_y, 0, 0],
        [0,       0,       1, 0],
        [0,       0,       0, 1]
      ])

      scale_copy.transform(scaling_matrix)

      reg_p2p = o3d.pipelines.registration.registration_icp(
        scale_copy, captured_pointcloud, threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

      current_fitness = get_evaluation_fitness(scale_copy, captured_pointcloud, reg_p2p.transformation)

      if current_fitness > best_fitness:
        best_transformation = np.matmul(reg_p2p.transformation, scaling_matrix)
        best_fitness = current_fitness

  return best_transformation, best_fitness
