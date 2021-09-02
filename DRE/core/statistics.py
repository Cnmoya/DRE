import numpy as np


def cube_gradient(cube, idx_min, steps):
    grad = np.array(np.gradient(cube, *steps))
    return grad[:, idx_min[0], idx_min[1], idx_min[2]]


def gradient_norm(cube, idx_min, steps):
    window = 3
    window_slices = []
    for i_min in idx_min:
        x_slice = slice(i_min - window // 2, i_min + window // 2 + 1)
        window_slices.append(x_slice)
    gradient = cube_gradient(cube[tuple(window_slices)], idx_min, steps)
    return np.linalg.norm(gradient)


def cube_hessian(cube, idx_min, steps):
    window = 5
    window_slices = []
    for i_min in idx_min:
        x_slice = slice(i_min - window // 2, i_min + window // 2 + 1)
        window_slices.append(x_slice)
    hessian = np.zeros((cube.ndim, cube.ndim) + cube[idx_min].shape)
    for i, grad_i in enumerate(np.gradient(cube, *steps)):
        for j, grad_ij in enumerate(np.gradient(grad_i, *steps)):
            hessian[i, j] = grad_ij
    return hessian[:, :, idx_min[0], idx_min[1], idx_min[2]]


def params_std(chi_cube, idx_min, steps):
    info_matrix = -cube_hessian(-np.log(chi_cube), idx_min, steps)
    cov_matrix = np.linalg.inv(info_matrix)
    variance = np.array(np.diag(cov_matrix))
    variance[variance < 0] = np.nan
    return np.sqrt(variance)
