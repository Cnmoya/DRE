from scipy.ndimage import map_coordinates
import numpy as np


def subgrid_cube(original_points, cube, idx_min, samples=3, window=3, order=3):
    sub_points = []
    sub_index = []
    for x, i_min in zip(original_points, idx_min):
        i0 = max(0, i_min - window // 2)
        i1 = min(len(x) - 1, i_min + window // 2)
        sub_x = np.linspace(x[i0], x[i1], len(x[i0:i1+1]) * (samples + 1) - samples)
        sub_points.append(sub_x)
        x_idx = np.linspace(0, len(x) - 1, len(x))
        sub_x_idx = np.linspace(x_idx[i0], x_idx[i1], len(x_idx[i0:i1+1]) * (samples + 1) - samples)
        sub_index.append(sub_x_idx)
    sub_grid = np.array(np.meshgrid(*sub_index, indexing='ij'))
    interpolated_cube = map_coordinates(cube, sub_grid, order=order)
    return sub_points, interpolated_cube


def interpolated_min(cube, original_points, idx_min, samples=3, window=3, order=3):
    sub_points, interpolated_cube = subgrid_cube(original_points, cube, idx_min,
                                                 samples, window, order)
    new_idx_min = np.unravel_index(np.argmin(interpolated_cube), interpolated_cube.shape)
    interp_min = [sub_points[i][new_idx_min[i]] for i in range(len(new_idx_min))]
    return interp_min
