from DRE.core import ModelsIO as MIO
import numpy as np
from h5py import File


def E_fit(_cube: np.ndarray((10, 13, 21, 128, 128), '>f4'),
          data: np.ndarray((128, 128), '>f4'),
          seg: np.ndarray((128, 128), '>f4'),
          noise: np.ndarray((128, 128), '>f4')) -> np.ndarray((10, 13, 21), '>f4'):

    scaled_models: np.ndarray((10, 13, 21, 128, 128), '>f4')
    flux_models: np.ndarray((10, 13, 21), '>f4')
    flux_data: np.float('>f4')
    X: np.ndarray((10, 13, 21), '>f4')
    resta: np.ndarray((10, 13, 21, 128, 128), '>f4')
    residuo: np.ndarray((10, 13, 21, 128, 128), '>f4')
    chi: np.ndarray((10, 13, 21), '>f4')
    area: int

    flux_models = np.einsum("ijkxy,xy->ijk", _cube, seg)
    flux_data = np.einsum("xy,xy", data, seg)
    X = flux_data / flux_models
    scaled_models = X[:, :, :, np.newaxis, np.newaxis] * _cube
    resta = data - scaled_models
    residuo = (resta ** 2) / np.sqrt(np.abs(scaled_models) + noise ** 2)
    chi = np.einsum("ijkxy,xy->ijk", residuo, seg)

    area = seg.sum()
    chi = chi / area
    return chi


def read_obj_h5(name):
    # debe ser
    try:
        with File(name, 'r') as f:
            data = f['obj'][:, :]
            seg = f['seg'][:, :]
            rms = f['rms'][:, :]

            return data, seg, rms
        # rms = MIO.fits.open(name.replace('objs','noise'))[1].data
        # seg =  MIO.fits.open(name.replace('object',"segment").replace("objs","segs"))[1].data

    except IOError:
        print("{} not found".format(name))
        return False, False, False


# se necesita esta funcion??
def read_obj(name):
    try:
        data = MIO.fits.open(name)[1].data
        rms = MIO.fits.open(name.replace('objs', 'noise'))[1].data
        seg = MIO.fits.open(name.replace('object', "segment").replace("objs", "segs"))[1].data

    except IOError:
        print("{} not found".format(name))
        return False, False, False
    noise = np.median(rms)
    d_t = np.tile(data, (10, 13, 21))
    s_t = np.tile(seg, (10, 13, 21))
    d_t = d_t.reshape((10, 13, 128, 21, 128))
    s_t = s_t.reshape((10, 13, 128, 21, 128))
    return d_t, s_t, noise


def feed(name, cube):
    """
    From a name and a models cube, run an object through the routine
    Outputs the numpy array of the chi_cube

    """

    a, b, s = read_obj_h5(name)
    if a is not False:
        chi = E_fit(cube, a, b, noise=s)
        # outchi = MIO.fits.ImageHDU(data=chi)
        # outchi.writeto(name.replace('cut_object',"chi_cube"),overwrite=True)
        return chi
    else:
        return False


def save_chi(name, cube):
    """

    Parameters

    name : str of output file
    cube : crunch.feed output
    """

    outchi = MIO.fits.ImageHDU(data=cube)
    outchi.writeto(name, overwrite=True)
    return True


def get_cube(name):
    cube = MIO.ModelsCube(name)
    cube = cube.data.reshape((10, 13, 128, 21, 128))
    cube = np.swapaxes(cube, 2, 3)  # new shape (10, 13, 21, 128, 128)
    return cube


def chi_index(chi_name):
    """

    Parameters
    ----------
    chi_name : chi_cube fits filename.

    Returns
    -------
    tuple (i,j,k) of the index which minimize the residuals.

    """

    chi_cube = MIO.fits.open(chi_name)
    i, j, k = np.unravel_index(np.argmin(chi_cube[1].data), shape=(10, 13, 21))
    return i, j, k


def pond_rad_like(chi_name, logh):
    i, j, k = chi_index(chi_name)
    chi_cubo = MIO.fits.open(chi_name)[1].data
    weights = np.e ** (chi_cubo[i, j, :])
    r_weight = 0
    for r in range(21):
        r_weight += (10 ** (logh[r])) / weights[r]

    r_chi = np.log10(r_weight / np.sum(1. / weights))

    r_var = 0
    for r in range(21):
        r_var += ((logh[r] - r_chi) ** 2) / (weights[r])
    r_var = r_var / np.sum(1. / weights)

    return r_chi, r_var


def pond_rad(chi_name, logh):
    i, j, k = chi_index(chi_name)
    chi_cubo = MIO.fits.open(chi_name)[1].data
    weights = chi_cubo[i, j, :]
    r_weight = 0
    for r in range(21):
        r_weight += (10 ** (logh[r])) / weights[r]

    r_chi = np.log10(r_weight / np.sum(1. / weights))

    r_var = 0
    for r in range(21):
        r_var += ((logh[r] - r_chi) ** 2) / (weights[r])
    r_var = r_var / np.sum(1. / weights)
    return r_chi, r_var


def pond_rad_3d(chi_name, logh):
    chi_cubo = MIO.fits.open(chi_name)[1].data
    sqrt_chi = np.sqrt(chi_cubo)
    r_weight = 0
    for e in range(10):
        for t in range(13):
            for r in range(21):
                r_weight += (10 ** (logh[r])) / sqrt_chi[e, t, r]

    r_chi = np.log10(r_weight / np.sum(1. / sqrt_chi))

    r_var = 0
    for e in range(10):
        for t in range(13):
            for r in range(21):
                r_var += ((logh[r] - r_chi) ** 2) / (chi_cubo[e, t, r])

    r_var = r_var / np.sum(1. / chi_cubo)
    return r_chi, r_var


def make_mosaic(obj, chi, cube):
    """

    Parameters
    ----------
    obj : str
        DESCRIPTION.
    chi : str
        DESCRIPTION.
    cube : numpy array
        DESCRIPTION.

    Returns
    -------
    Bool

    Builds a mosaic containing the data,segment,model and residual
    """

    i, j, k = chi_index(chi)
    model = cube[i, j, k]
    gal, seg, noise = read_obj(obj)
    gal = gal[i, j, k]
    seg = seg[i, j, k]
    output = chi.replace('chi_cube', 'mosaic').replace('cut_object', 'mosaic')

    fg = np.sum(gal * seg)
    fm1 = np.sum(model * seg)
    aux = np.zeros((128, 128 * 4))
    aux[:, 0:128] = gal
    aux[:, 128:256] = seg * (fg / seg.sum())
    aux[:, 256:384] = model * (fg / fm1)
    aux[:, 384:] = gal - model * (fg / fm1)

    gg = MIO.fits.ImageHDU(data=aux)
    gg.writeto(output, overwrite=True)
    return True


def make_mosaic_h5(obj, chi, cube):
    """

    Parameters
    ----------
    obj : str
        DESCRIPTION.
    chi : str
        DESCRIPTION.
    cube : numpy array
        DESCRIPTION.

    Returns
    -------
    Bool

    Builds a mosaic containing the data,segment,model and residual
    """

    i, j, k = chi_index(chi)
    model = cube[i, j, k]
    output = chi.replace('chi_cube', 'mosaic').replace('cut', 'mosaic')
    with File(obj, 'r') as f:
        gal = f['obj'][:, :]
        seg = f['seg'][:, :]

        fg = np.sum(gal * seg)
        fm1 = np.sum(model * seg)
        aux = np.zeros((128, 128 * 4))
        aux[:, 0:128] = gal
        aux[:, 128:256] = seg * (fg / seg.sum())
        aux[:, 256:384] = model * (fg / fm1)
        aux[:, 384:] = gal - model * (fg / fm1)

        gg = MIO.fits.ImageHDU(data=aux)
        gg.writeto(output, overwrite=True)

    return True
