import open3d as o3d
import numpy as np
import ransac_test
import random
from utils import geometry, helpers
import math
import copy
from tqdm import tqdm
import itertools
import scipy

def find_pca_of_pointcloud(pointcloud):
  # Voxel-downsample the cloud to remove "clusters" where we have
  # too many points, as the density may vary greatly over the cloud
  pointcloud = pointcloud.voxel_down_sample(0.5)

  # Compute the mean and the covariance matrix of the downsampled
  # pointcloud.
  mean, covariance = pointcloud.compute_mean_and_covariance()

  # Compute the eigenvalues and eigenvectors of the covariance
  # matrix. The eigenvectors are the principal directions of
  # the pointcloud
  eigenvalues, eigenvectors = np.linalg.eig(covariance)

  return eigenvectors

def visualize_pca_with_pointcloud(eigenvectors, pointcloud, color=[1, 0, 0], additional_data=None, stretch_factor=5):
  points = []
  zero_point = [0, 0, 0]

  # For each eigenvector, create a line between the origin and its
  # endpoint, given by the eigenvector. We stretch the line a bit,
  # such that it's easier to see.
  for eigenvector in eigenvectors:
    eigenvector = geometry.stretch_vector(geometry.normalized(eigenvector), stretch_factor)
    points.append([zero_point, eigenvector])

  # Create a new lineset and paint it red
  lineset = helpers.create_lineset_from_line_endpoints(points)
  lineset.paint_uniform_color(color)

  visualization_data = [lineset]

  if pointcloud is not None:
    visualization_data.append(pointcloud)

  if additional_data is not None:
    if np.isscalar(additional_data):
      visualization_data.append(additional_data)
    else:
      visualization_data += additional_data

  o3d.visualization.draw_geometries(visualization_data)

# Get the first n walls, compute the PCA for each, and then average the PCAs,
# such that we get the a good approximation of the general PCA of the room
# layout.
def get_pcas_from_first_n_walls(pointcloud, walls_count, visualize=False, stretch_factor=5, print_messages=True, recursion_depth=0):
  find_planes_wall_count = walls_count
  _, _, _, _, wall_pointclouds, _ = ransac_test.find_planes(pointcloud, find_planes_wall_count, create_ceiling=False, print_messages=print_messages)

  zero_point = [0, 0, 0]

  # Take minumim of walls_count or length of the floor pointclouds, to avoid index out of bounds
  stop = min(walls_count, len(wall_pointclouds))

  all_eigenvectors = []

  # Get eigenvectors of the first few wall pointclouds
  for separate_pointcloud in wall_pointclouds[:stop]:
    all_eigenvectors.append(find_pca_of_pointcloud(separate_pointcloud))

  zero_point = [0, 0, 0]
  points = []

  x_evs = []
  y_evs = []
  z_evs = []

  # Sort eigenvectors into x, y and z eigenvectors
  for eigenvectors in all_eigenvectors:
    x_found = False
    y_found = False
    z_found = False
    for eigenvector in eigenvectors:
      amax = np.argmax(np.abs(eigenvector))
      if amax == 0 and not x_found:
        x_found = True
        x_evs.append(eigenvector)
      elif amax == 1 and not y_found:
        y_found = True
        y_evs.append(eigenvector)
      elif amax == 2 and not z_found:
        z_found = True
        z_evs.append(eigenvector)
      else:
        raise Exception("Inplausible argmax")

  # Flip the eigenvectors, such that they all point into the right direction
  for idx, x_ev in enumerate(x_evs):
    if x_ev[0] < 0:
      x_evs[idx] = -1 * x_ev

  for idx, y_ev in enumerate(y_evs):
    if y_ev[1] < 0:
      y_evs[idx] = -1 * y_ev

  for idx, z_ev in enumerate(z_evs):
    if z_ev[2] < 0:
      z_evs[idx] = -1 * z_ev


  # Compute the averages of all eigenvector directions
  x_evs = np.array(x_evs)
  y_evs = np.array(y_evs)
  z_evs = np.array(z_evs)

  # If we did not find any EVs, try again with a higher wall count, except if we
  # already did try to do it recursively too often
  if len(x_evs) == 0:
    if recursion_depth >= 3:
      raise RuntimeError('Did not manage to find any PCAs!')
    else:
      if print_messages:
        print('Did not find PCAs, trying again with more walls to find')
      return get_pcas_from_first_n_walls(pointcloud, walls_count + 2, visualize=visualize, stretch_factor=stretch_factor, print_messages=print_messages, recursion_depth=recursion_depth + 1)

  final_x_ev = [np.mean(x_evs[:, 0]), np.mean(x_evs[:, 1]), np.mean(x_evs[:, 2])]
  final_y_ev = [np.mean(y_evs[:, 0]), np.mean(y_evs[:, 1]), np.mean(y_evs[:, 2])]
  final_z_ev = [np.mean(z_evs[:, 0]), np.mean(z_evs[:, 1]), np.mean(z_evs[:, 2])]

  if visualize:
    points = []
    zero_point = [0, 0, 0]

    for eigenvector in [final_x_ev, final_y_ev, final_z_ev]:
      eigenvector = geometry.stretch_vector(geometry.normalized(eigenvector), stretch_factor)
      points.append([zero_point, eigenvector])

    lineset = helpers.create_lineset_from_line_endpoints(points)
    lineset.paint_uniform_color([1, 0, 0])
    evsall = find_pca_of_pointcloud(pointcloud)
    visualize_pca_with_pointcloud(evsall, pointcloud, color=[0, 1, 0], additional_data=[lineset])

  if print_messages:
    print('PCAs found!')

  return [final_x_ev, final_y_ev, final_z_ev]

def use_pcas_for_constrained_ransac(pcas, point_cloud, direction='x', iterations=300, angle_check=False, distance_threshold=0.1, print_messages=False):
  x_pcc, y_pcc, z_pcc = pcas

  if direction == 'x':
    normal_vector = x_pcc
  elif direction == 'y':
    normal_vector = y_pcc
  elif direction == 'z':
    normal_vector = z_pcc

  a, b, c = normal_vector

  current_point = random.choice(point_cloud.points)
  best_inlier_counter = 0
  best_plane = None

  if angle_check:
    assert(len(point_cloud.points) == len(point_cloud.normals))

  ANGLE_OFFSET = math.pi / 10

  for iteration in tqdm(range(iterations), disable=(not print_messages)):
    x0, y0, z0 = current_point
    d = -(a*x0 + b*y0 + c*z0)
    plane = [a, b, c, d]
    inlier_counter = 0
    non_angle_counter = 0

    # TODO: check if this can be made faster by parallelizing
    for pt3D, pt3Dnormal in itertools.zip_longest(point_cloud.points, point_cloud.normals):
      distance_to_plane = geometry.point_plane_distance(pt3D, plane)

      if distance_to_plane <= distance_threshold:
        if angle_check:
          angle = geometry.angle_between_vectors(normal_vector, pt3Dnormal)
          if angle < ANGLE_OFFSET or (math.pi - ANGLE_OFFSET) < angle:
            inlier_counter += 1
          else:
            non_angle_counter += 1
        else:
          inlier_counter += 1

    if inlier_counter > best_inlier_counter:
      best_plane = plane
      best_inlier_counter = inlier_counter

    current_point = random.choice(point_cloud.points)

  # print(f"Found plane: {best_plane} with {best_inlier_counter} inliers")

  # pointcloud_utils.visualize_planes([best_plane], point_cloud)

  return best_plane, inlier_counter, non_angle_counter

def rotate_pcd_to_match_pcas(pointcloud):
  # Get PCAs and rotate the pointcloud
  pcas = get_pcas_from_first_n_walls(pointcloud.uniform_down_sample(10), 5, print_messages=False)

  xyz = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
  ])

  # Get rotation matrix
  rotation, _ = scipy.spatial.transform.Rotation.align_vectors(xyz, np.array(pcas))
  rotmat = rotation.as_matrix()

  # Rotate the pointcloud
  result = copy.deepcopy(pointcloud)
  result.rotate(rotmat)

  return result, rotmat
