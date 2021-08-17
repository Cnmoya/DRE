from scipy.ndimage import map_coordinates
import numpy as np
from itertools import product


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


def subgrid_min(cube, original_points, idx_min, samples=3, window=3, order=3):
    sub_points, interpolated_cube = subgrid_cube(original_points, cube, idx_min,
                                                 samples, window, order)
    new_idx_min = np.unravel_index(np.argmin(interpolated_cube), interpolated_cube.shape)
    interp_min = [sub_points[i][new_idx_min[i]] for i in range(len(new_idx_min))]
    return interp_min


def parabola_1d(x, y):
    coef_mat = np.zeros((x.shape[0], 3))
    coef_mat[:, 0] = 1
    coef_mat[:, 1] = x
    coef_mat[:, 2] = x ** 2
    if x.shape[0] == 3:
        return np.linalg.solve(coef_mat, y)
    else:
        return np.linalg.lstsq(coef_mat, y, rcond=None)[0]


def fit_parabola_1d(cube, idx_min, log_r, window=3):
    i_min = idx_min[2]
    i_min = max(i_min, window // 2)
    i_min = min(i_min, len(log_r) - 1 - window // 2)

    r_slice = slice(i_min - window // 2, i_min + window // 2 + 1)
    window_r = log_r[r_slice]
    window_chi = cube[idx_min[0], idx_min[1], r_slice]

    # y = a*x^2 + b*x + c
    c, b, a = parabola_1d(window_r, window_chi)
    parabola_r_min = -b / (2 * a)

    # border cases
    outside_window = ((parabola_r_min < window_r[0]) or
                      (parabola_r_min > window_r[-1]))
    outside_models = ((parabola_r_min < log_r[0]) or
                      (parabola_r_min > log_r[-1]))
    if outside_models or a < 0:
        parabola_r_min = np.nan
        parabola_r_std = np.nan
    elif outside_window:
        parabola_r_min = log_r[idx_min[2]]
        parabola_r_std = np.nan
    else:
        parabola_r_std = 1 / np.sqrt(2 * a)
    return parabola_r_min, parabola_r_std


def parabola_nd(x, y):
    dim = x.shape[1]
    coef_mat = np.zeros((x.shape[0], 1 + 2 * dim + (dim ** 2 - dim) // 2))
    coef_mat[:, 0] = 1
    for d1 in range(dim):
        coef_mat[:, 1 + d1] = x[:, d1]
        coef_mat[:, 1 + dim + d1] = x[:, d1] ** 2
        for d2 in range(d1 + 1, dim):
            coef_mat[:, 2 * dim + d1 + d2] = x[:, d1] * x[:, d2]

    return np.linalg.lstsq(coef_mat, y, rcond=None)[0]


def find_parabola_nd_minimum(coefs, dim):
    linear_coefs = coefs[1:dim + 1]
    square_diag_coefs = coefs[1 + dim:1 + 2 * dim]
    square_mixed_coefs = coefs[1 + 2 * dim:]

    hessian_matrix = np.zeros((dim, dim))
    hessian_matrix[np.triu_indices(dim, 1)] = square_mixed_coefs
    hessian_matrix[np.tril_indices(dim, -1)] = square_mixed_coefs
    np.fill_diagonal(hessian_matrix, 2 * square_diag_coefs)

    x_min = np.empty(dim)
    cov_matrix = np.empty((dim, dim))

    if (np.linalg.eigvals(hessian_matrix) > 0).all():
        x_min = np.linalg.solve(hessian_matrix, -linear_coefs)
        cov_matrix = np.linalg.inv(hessian_matrix)
    else:
        x_min[:] = np.nan
        cov_matrix[:] = np.nan

    return x_min, cov_matrix


def fit_parabola_nd(cube, idx_min, log_r, ax_ratio, angle, window=3):
    points = (ax_ratio, angle, log_r)
    window_points = []
    window_slices = []
    for x, i_min in zip(points, idx_min):
        i_min = max(i_min, window // 2)
        i_min = min(i_min, len(x) - 1 - window // 2)
        x_slice = slice(i_min - window // 2, i_min + window // 2 + 1)
        window_slices.append(x_slice)
        window_points.append(x[x_slice])
    window_chi = cube[tuple(window_slices)]
    window_grid = np.array(list(product(*window_points)))
    parabola_coefs = parabola_nd(window_grid, window_chi.flatten())
    parabola_min, cov_matrix = find_parabola_nd_minimum(parabola_coefs, dim=len(points))

    # border cases
    not_border = []
    not_min = np.isnan(parabola_min).any()
    outside_window = []
    outside_models = []
    for i in range(len(parabola_min)):
        not_border.append((idx_min[i] != 0) and (idx_min[i] != len(points[i]) - 1))
        outside_window.append((parabola_min[i] < window_points[i][0]) or
                              (parabola_min[i] > window_points[i][-1]))
        outside_models.append((parabola_min[i] < points[i][0]) or
                              (parabola_min[i] > points[i][-1]))
    if np.any(outside_models):
        parabola_min[:] = np.nan
        cov_matrix[:] = np.nan
    elif (not_min and np.any(not_border)) or np.any(outside_window):
        parabola_min = [points[i][idx_min[i]] for i in range(len(points))]
        cov_matrix[:] = np.nan
    return parabola_min, cov_matrix


