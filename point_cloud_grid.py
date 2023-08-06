import numpy as np
from utils import helpers
import sys
import math
import numpy as np

class FlatPointcloudGrid:
  def __init__(self, pointcloud, bin_count=200, padding=0):
    self.build_matrix(pointcloud, bin_count, padding)

  def build_matrix(self, pointcloud, bin_count, padding):
    max_x = sys.float_info.min
    max_y = sys.float_info.min
    min_x = sys.float_info.max
    min_y = sys.float_info.max

    for pt3D in pointcloud.points:
      xp, yp, zp = pt3D

      max_x = max(max_x, xp)
      max_y = max(max_y, yp)

      min_x = min(min_x, xp)
      min_y = min(min_y, yp)

    # Add / subrtract padding
    max_x += padding
    max_y += padding
    min_x -= padding
    min_y -= padding

    # Compute size of "floor"
    x_dist = abs(max_x - min_x)
    y_dist = abs(max_y - min_y)

    # Compute number of bins. We use a fixed size, which is 1 / bin_count-th
    # of the longer side of the cloud
    bin_side_dimension = max(x_dist, y_dist) / bin_count

    # Compute number of bins
    x_bins = int(math.ceil(x_dist / bin_side_dimension)) + 1
    y_bins = int(math.ceil(y_dist / bin_side_dimension)) + 1

    matrix = np.zeros((x_bins, y_bins), dtype=np.int32)

    for pt3D in pointcloud.points:
      xp, yp, zp = pt3D
      # Shift to coordinates starting at 0, 0
      xp += abs(min_x)
      yp += abs(min_y)

      x_bin = math.floor(xp / bin_side_dimension)
      y_bin = math.floor(yp / bin_side_dimension)

      matrix[x_bin][y_bin] += 1

    self.pointcloud = pointcloud
    self.matrix = matrix
    self.bin_side_dimension = bin_side_dimension
    self.min_x = min_x
    self.min_y = min_y
    self.max_x = max_x
    self.max_y = max_y
    self.x_bins = x_bins
    self.y_bins = y_bins
    self.x_dist = x_dist
    self.y_dist = y_dist
    self.bin_count = bin_count

  def get_point_neighbor_count(self, point):
    x_bin, y_bin = self.get_bin_indices(point)

    if x_bin is None or y_bin is None:
      return 0

    return self.matrix[x_bin][y_bin]

  def get_bin_indices(self, point):
    xp, yp, zp = point

    # Shift to coordinates starting at 0, 0
    xp += abs(self.min_x)
    yp += abs(self.min_y)

    x_bin = math.floor(xp / self.bin_side_dimension)
    y_bin = math.floor(yp / self.bin_side_dimension)

    if x_bin < 0 or x_bin > self.bin_count or y_bin < 0 or y_bin > self.bin_count:
      return None, None

    return x_bin, y_bin

  def get_middlepoint_of_bin(self, x_bin, y_bin):
    xp = x_bin * self.bin_side_dimension + (self.bin_side_dimension / 2)
    yp = y_bin * self.bin_side_dimension + (self.bin_side_dimension / 2)

    xp -= abs(self.min_x)
    yp -= abs(self.min_y)

    return [xp, yp, 0]

  def get_bin_item_count(self, x_bin, y_bin):
    return self.matrix[x_bin][y_bin]

  def get_lineset(self):
    # Make lines from the bins
    points = []
    current_x = self.min_x
    current_y = self.min_y

    # Make line from [min_x, min_y] to [min_x, max_y]
    max_x = self.max_x + self.bin_side_dimension
    max_y = self.max_y + self.bin_side_dimension

    for y in range(self.y_bins + 1):
      line_start = [self.min_x, current_y, 0]
      line_end = [max_x, current_y, 0]

      points.append([line_start, line_end])
      current_y += self.bin_side_dimension

    for x in range(self.x_bins + 1):
      line_start = [current_x, self.min_y, 0]
      line_end = [current_x, max_y, 0]

      points.append([line_start, line_end])
      current_x += self.bin_side_dimension

    return helpers.create_lineset_from_line_endpoints(points)
