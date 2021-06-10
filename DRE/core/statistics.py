import numpy as np


def cube_gradient(cube, min_index, steps):
    grad = np.array(np.gradient(cube, *steps))
    return grad[:, min_index[0], min_index[1], min_index[2]]


def gradient_norm(cube, min_index, steps):
    return np.linalg.norm(cube_gradient(cube, min_index, steps))


def cube_hessian(cube, min_index, steps):
    hessian = np.zeros((cube.ndim, cube.ndim) + cube.shape)
    for i, grad_i in enumerate(np.gradient(cube, *steps)):
        for j, grad_ij in enumerate(np.gradient(grad_i, *steps)):
            hessian[i, j] = grad_ij
    return hessian[:, :, min_index[0], min_index[1], min_index[2]]


def params_std(chi_cube, min_index, steps):
    info_matrix = -cube_hessian(-np.log(np.sqrt(chi_cube)), min_index, steps)
    cov_matrix = np.linalg.inv(info_matrix)
    variance = np.diag(cov_matrix)
    variance[variance < 0] = np.nan
    return np.sqrt(variance)
