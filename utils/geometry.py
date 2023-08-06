import math
import numpy as np
import open3d as o3d
from typing import List, Union
import random

def point_plane_distance(point, plane):
  x, y, z = point
  a, b, c, d = plane
  distance = abs(a * x + b * y + c * z + d)
  norm = math.sqrt(a**2 + b**2 + c**2)
  return distance / norm

def point_above_plane(point, plane):
  x, y, z = point
  a, b, c, d = plane
  return np.dot((a,b,c,d), (x,y,z,1)) > 0

def plane_normal(plane):
  a, b, c, d = plane
  return [a, b, c]

def line_plane_intersection(point, direction, plane):
  # Plane = Ax + By + Cz + D = 0
  a, b, c, d = plane

  # Point = (x0, y0, z0)
  x0, y0, z0 = point

  # Direction = (xd, yd, zd)
  xd, yd, zd = direction

  # Any point on the vector can be written as:
  #   p(t) = point + t * dir = (x0 + t*xd, y0 + t*yd, z0 + t*zd)
  #
  # Now, if we plug this into the plane equation, we get:
  #   A * (x0 + t*xd) + B * (y0 + t*yd) + C * (z0 + t*zd) + D = 0
  #
  # Through rewriting, we get:
  #   = Ax0 + Atxd + By0 + Btyd + Cz0 + Ctzd + D = 0
  #   = Ax0 + Atxd + By0 + Btyd + Cz0 + Ctzd = -D
  #   = t * (Axd + Byd + Czd) + Ax0 + By0 + Cz0 = -D
  #   = t * (Axd + Byd + Czd) = -Ax0 - By0 - Cz0 - D
  #   = t = (-Ax0 - By0 - Cz0 - D) / (Axd + Byd + Czd)

  Ax0 = a * x0
  By0 = b * y0
  Cz0 = c * z0
  Axd = a * xd
  Byd = b * yd
  Czd = c * zd

  t = (-Ax0 - By0 - Cz0 - d) / (Axd + Byd + Czd)

  # Resulting point is (x0 + t*xd, y0 + t*yd, z0 + t*zd)
  xn = x0 + t * xd
  yn = y0 + t * yd
  zn = z0 + t * zd

  # Check wether the point actually lies on the plane, if not,
  # return None.
  # Please note that the intersection may lie on and point
  # along the line, i.e. it might be "in front" or "on the back"
  # of the point. If you want to only include points "in front"
  # of the point, you need to check wether t is negative or positive,
  # which is returned as the second return value.
  plane_equation_value = a * xn + b * yn + c * zn + d

  if math.isclose(plane_equation_value, 0, abs_tol=1e-09):
    return [xn, yn, zn], t >= 0.0
  else:
    return None, None

def point_within_oriented_bounding_box(point: List[int], bounding_box: o3d.geometry.OrientedBoundingBox) -> bool:
  R = bounding_box.R
  center = bounding_box.center
  extent_x, extent_y, extent_z = bounding_box.extent

  dx = R.dot(np.array([1, 0, 0]))
  dy = R.dot(np.array([0, 1, 0]))
  dz = R.dot(np.array([0, 0, 1]))

  d = point - center

  res_x = abs(d.dot(dx)) <= extent_x / 2
  res_y = abs(d.dot(dy)) <= extent_y / 2
  res_z = abs(d.dot(dz)) <= extent_z / 2

  return res_x and res_y and res_z

def point_within_axis_aligned_bounding_box(point: List[int], bounding_box: o3d.geometry.AxisAlignedBoundingBox) -> bool:
  min_x, min_y, min_z = bounding_box.get_min_bound()
  max_x, max_y, max_z = bounding_box.get_max_bound()

  px, py, pz = point

  return min_x <= px <= max_x and min_y <= py <= max_y and min_z <= pz <= max_z

def point_within_bounding_box(point: List[int], bounding_box: Union[o3d.geometry.OrientedBoundingBox, o3d.geometry.AxisAlignedBoundingBox]) -> bool:
  if bounding_box is o3d.geometry.OrientedBoundingBox:
    return point_within_oriented_bounding_box(point, bounding_box)
  else:
    return point_within_axis_aligned_bounding_box(point, bounding_box)

def normalized(vector):
  if np.linalg.norm(vector) == 0.0:
    return np.array([0.0, 0.0, 0.0])
  return vector / np.linalg.norm(vector)

def plane_plane_intersection(plane_a: List[float], plane_b: List[float]):
  # Compute the normalized direction vector
  direction_vector = normalized(np.cross(plane_a[:-1], plane_b[:-1]))

  # Create the right hand side of our equation we want to solve,
  # which is:
  #  a_a  b_a  c_a
  #  a_b  b_b  c_b
  #  a_d  b_d  c_d
  #
  # Where x_a denotes params from plane_a, x_b params from plane_b and
  # x_d denotes params form the direction_vector
  #
  rhs = np.array((plane_a[:-1], plane_b[:-1], direction_vector))

  # Similarly, create the left hand side of the equation as:
  #   d_a  d_b  0
  # where the 0 is put is as we have a free variable
  lhs = np.array((-plane_a[-1], -plane_b[-1], 0.))

  try:
    point_on_line = np.linalg.solve(rhs, lhs)
  except np.linalg.LinAlgError:
    return None, None

  return point_on_line, direction_vector

def parallel_planes_distance(plane_1: List[float], plane_2: List[float]) -> float:
  # First, check that the planes are actually parallel. Please note
  # that for parallel planes, a, b and c should be equal
  a1, b1, c1, d1 = plane_1
  a2, b2, c2, d2 = plane_2

  normal_1 = plane_normal(plane_1)
  normal_2 = plane_normal(plane_2)

  angle_between = angle_between_vectors(normal_1, normal_2)

  if not math.isclose(angle_between, 0.0):
    raise RuntimeError('Planes are not parallel!')

  distance = abs(d2 - d1) / math.sqrt(a1**2 + b1 **2 + c1**2)

  return distance

# Returns a value in the interval [0, pi]
def angle_between_vectors(v1, v2):
  v1_u = normalized(v1)
  v2_u = normalized(v2)
  return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def stretch_vector(vector, factor):
  result = []

  for dimension in vector:
    result.append(dimension * factor)

  return result

def rotation_matrix_from_vectors(source, destination):
  """ Find the rotation matrix that aligns source to destination
  :param source: A 3d "source" vector
  :param destination: A 3d "destination" vector
  :return mat: A transform matrix (3x3) which when applied to source, aligns it with destination.
  """
  # Normalize the two vectors
  a = normalized(source)
  b = normalized(destination)

  # Compute the crossproduct of the two vectors, and from that the
  # cosine and sine between the two angles.
  cross_product = np.cross(a, b)
  angle_cos = np.dot(a, b)
  angle_sin = np.linalg.norm(cross_product)

  # Build the cross-product matrix [v]
  cross_product_matrix = np.array([
    [0, -cross_product[2], cross_product[1]],
    [cross_product[2], 0, -cross_product[0]],
    [-cross_product[1], cross_product[0], 0]
  ])

  # Finally, find the rotation matrix as R = I + [v] + [v]^2 * (1 - cos) / sin^2
  rotation_matrix = np.eye(3) + cross_product_matrix + cross_product_matrix.dot(cross_product_matrix) * ((1 - angle_cos) / (angle_sin**2))

  return rotation_matrix

# Adapted from https://math.stackexchange.com/a/4289668
def closest_distance_between_lines(line_a, line_b):
  line_a_start, line_a_end = line_a
  line_b_start, line_b_end = line_b

  vect_a = line_a_end - line_a_start
  vect_b = line_b_end - line_b_start
  vect_b_start_a_start = line_b_start - line_a_start

  v22 = np.dot(vect_b, vect_b)
  v11 = np.dot(vect_a, vect_a)

  v21 = np.dot(vect_b, vect_a)
  v21_1 = np.dot(vect_b_start_a_start, vect_a)
  v21_2 = np.dot(vect_b_start_a_start, vect_b)

  denominator = v21 * v21 - v22 * v11

  if np.isclose(denominator, 0.):
    s = 0
    t = (v11 * s - v21_1) / v21
  else:
    s = (v21_2 * v21 - v22 * v21_1) / denominator
    t = (-v21_1 * v21 + v11 * v21_2) / denominator

  # Clamp s and t to the 0, 1 range
  s = max(min(s, 1.), 0.)
  t = max(min(t, 1.), 0.)

  p_a = line_a_start + s * vect_a
  p_b = line_b_start + t * vect_b

  return p_a, p_b, np.linalg.norm(p_a - p_b)

def point_point_distance(point_a, point_b):
  difference_vector = np.array(point_a) - np.array(point_b)
  return np.linalg.norm(difference_vector)

def point_between_other_two_on_line(point, start_point, end_point):
  whole_distance = point_point_distance(start_point, end_point)
  point_distance = point_point_distance(start_point, point) + point_point_distance(point, end_point)
  return math.isclose(whole_distance, point_distance)

def line_aabb_intersection(line_origin, line_direction, axis_aligned_bbox):
  bbox_min = axis_aligned_bbox.get_min_bound()
  bbox_max = axis_aligned_bbox.get_max_bound()

  t0 = (bbox_min - line_origin) / line_direction
  t1 = (bbox_max - line_origin) / line_direction

  t_min = np.minimum(t0, t1)
  t_max = np.maximum(t0, t1)

  t_near = max(max(t_min[0], t_min[1]), t_min[2])
  t_far = min(min(t_max[0], t_max[1]), t_max[2])

  return t_near <= t_far, t_near, t_far

def plane_from_three_points(points):
  # Get points and divide them up into their parts
  p1, p2, p3 = points
  x1, y1, z1 = p1
  x2, y2, z2 = p2
  x3, y3, z3 = p3

  # Compute vectors between points
  a1 = x2 - x1
  b1 = y2 - y1
  c1 = z2 - z1
  a2 = x3 - x1
  b2 = y3 - y1
  c2 = z3 - z1

  # Compute parameters of plane
  a = b1 * c2 - b2 * c1
  b = a2 * c1 - a1 * c2
  c = a1 * b2 - b1 * a2
  d = (- a * x1 - b * y1 - c * z1)

  return a, b, c, d

def plane_from_point_and_normal(point, normal):
  a, b, c = normal
  d = - point.dot(normal)

  return [a, b, c, d]

def point_on_plane(plane):
  # Point [x, y, z] and plane [a, b, c, d]
  # must satisfy: ax + by + cd + d = 0.
  a, b, c, d = plane

  # Pick x, y at random in interval [0, 1]
  x = random.uniform(0, 1)
  y = random.uniform(0, 1)

  # Compute z with:
  #  ax + by + cz + d = 0
  #  cz = -ax - by - d
  #   z = (-ax - by - d) / c

  z  = (-a*x - b*y - d) / c

  return np.array([x, y, z])