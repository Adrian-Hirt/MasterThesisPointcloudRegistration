import hashlib
import pickle
import bz2
import open3d as o3d
import numpy as np

def get_keys_from_value(d, val):
  return [k for k, v in d.items() if v == val]

def hash_point_coordinates(point):
  string = str(point)
  return hashlib.md5(string.encode()).hexdigest()

def save_data_as_compressed_pickle(data, filename):
  with bz2.BZ2File(filename, 'w') as file:
    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

def load_data_from_compressed_pickle(filename):
  with bz2.BZ2File(filename, 'rb') as file:
    data = pickle.load(file)
    return data

def save_bounding_boxes_as_compressed_pickle(data, filename):
  serialized = []

  for bbox in data:
    if bbox is o3d.geometry.OrientedBoundingBox:
      bbox_result = serialize_oriented_bbox(bbox)
    else:
      bbox_result = serialize_axis_aligned_bbox(bbox)

    serialized.append(bbox_result)

  save_data_as_compressed_pickle(serialized, filename)

def serialize_axis_aligned_bbox(bounding_box):
  return [bounding_box.get_min_bound(), bounding_box.get_max_bound()]

def serialize_oriented_bbox(bounding_box):
  return [bounding_box.get_center(), bounding_box.R, bounding_box.extent]

def axis_aligned_bbox_from_serialized_data(data):
  return o3d.geometry.AxisAlignedBoundingBox(data[0], data[1])

def oriented_bbox_from_serialized_data(data):
  return o3d.geometry.OrientedBoundingBox(data[0], data[1], data[2])

def save_lineset_as_compressed_pickle(lineset, filename):
  data = []
  lines = []
  points = []

  for point in lineset.points:
    points.append(point)

  for line in lineset.lines:
    lines.append(line)

  data.append(points)
  data.append(lines)

  save_data_as_compressed_pickle(data, filename)

def lineset_from_compressed_pickle(filename):
  data = load_data_from_compressed_pickle(filename)
  points = o3d.utility.Vector3dVector(np.array(data[0]))
  lines = o3d.utility.Vector2iVector(np.array(data[1]))
  return o3d.geometry.LineSet(points, lines)

def bounding_boxes_from_compressed_pickle(filename):
  data = load_data_from_compressed_pickle(filename)

  result = []

  for bbox_data in data:
    if len(bbox_data) == 3:
      bbox = oriented_bbox_from_serialized_data(bbox_data)
    else:
      bbox = axis_aligned_bbox_from_serialized_data(bbox_data)
    result.append(bbox)

  return result

# line_endpoints must be an array of arrays with 2 points
# each, which denote the start and end of the line we want
# to visualize
def create_lineset_from_line_endpoints(line_endpoints, line_colors=None, color=None):
  points = []
  point_indices = []
  counter = 0

  if len(line_endpoints) == 0:
    raise RuntimeError('Cannot build a lineset from zero lines!')

  for line in line_endpoints:
    points.append(line[0])
    points.append(line[1])
    point_indices.append([counter, counter + 1])
    counter += 2

  points = o3d.utility.Vector3dVector(np.array(points))
  point_indices = o3d.utility.Vector2iVector(np.array(point_indices))

  lineset = o3d.geometry.LineSet(points, point_indices)

  if color is not None:
    line_colors = [color] * len(line_endpoints)

  if line_colors is not None:
    lineset.colors = o3d.utility.Vector3dVector(np.array(line_colors))

  return lineset

def array_of_arrays_contains(array_of_arrays, array):
  for current_array in array_of_arrays:
    if np.array_equal(current_array, array):
      return True

  return False

def line_endpoints_from_lineset(lineset):
  points = lineset.points

  line_points = []

  for [line_start_idx, line_end_idx] in lineset.lines:
    line_start = points[line_start_idx]
    line_end = points[line_end_idx]

    line_points.append([line_start, line_end])

  return line_points
