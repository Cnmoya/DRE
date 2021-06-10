from astropy.io import fits
import os
import numpy as np
import DRE
from DRE.misc.h5py_compression import compression_types
from DRE.misc.interpolation import interpolated_min
from DRE.core.statistics import gradient_norm, params_std


class ModelsCube:
    def __init__(self, models_file=None, out_compression='none'):

        self.models = None
        self.convolved_models = None
        self.header = None
        self.original_shape = None

        self.log_r = None
        self.angle = None
        self.ax_ratio = None

        self.compression = compression_types[out_compression]

        if models_file is None:
            dre_dir = os.path.dirname(os.path.realpath(DRE.__file__))
            models_file = os.path.join(dre_dir, 'models', 'modelbulge.fits')
        self.load_models(models_file)

    def __getitem__(self, index):
        return self.models.__getitem__(index)

    @property
    def shape(self):
        return self.models.shape

    def load_models(self, models_file):
        cube = fits.getdata(models_file).astype('float')
        self.original_shape = cube.shape
        cube = cube.reshape(10, 13, 128, 21, 128)
        cube = cube.swapaxes(2, 3)
        self.models = cube
        self.header = fits.getheader(models_file)
        self.log_r = np.arange(self.header["NLOGH"]) * self.header["DLOGH"] + self.header["LOGH0"]
        self.angle = np.arange(self.header["NPOSANG"]) * self.header["DPOSANG"] + self.header["POSANG0"]
        self.ax_ratio = np.arange(self.header["NAXRAT"]) * self.header["DAXRAT"] + self.header["AXRAT0"]

    def save_model(self, output_file):
        cube = self.models.swapaxes(2, 3)
        cube = cube.reshape(self.original_shape)
        models_fits = fits.ImageHDU(data=cube)
        models_fits.writeto(output_file, overwrite=True)

    def convolve(self, psf_file, *args, **kwargs):
        pass

    def dre_fit(self, data, segment, noise):
        pass

    def pond_rad_3d(self, chi_cube, log_r_min):
        r_chi = np.sum((10 ** self.log_r) / chi_cube)
        r_chi = r_chi / np.sum(1. / chi_cube)
        log_r_chi = np.log10(r_chi)

        r_var = np.sum(((10 ** self.log_r - 10 ** log_r_min) ** 2) / chi_cube)
        r_var = r_var / np.sum(1. / chi_cube)
        log_r_var = np.log10(r_var)

        r_chi_var = np.sum(((10 ** self.log_r - r_chi) ** 2) / chi_cube)
        r_chi_var = r_chi_var / np.sum(1. / chi_cube)
        log_r_chi_var = np.log10(r_chi_var)
        return log_r_chi, log_r_var, log_r_chi_var

    def get_parameters(self, chi_cube):
        e, t, r = np.unravel_index(np.nanargmin(chi_cube), chi_cube.shape)
        min_chi = np.nanmin(chi_cube)
        log_r_chi, log_r_var, log_r_chi_var = self.pond_rad_3d(chi_cube, self.log_r[r])
        interp_ax_ratio, interp_angle, interp_r = interpolated_min(chi_cube,
                                                                   (self.ax_ratio, self.angle, self.log_r),
                                                                   (e, t, r))
        steps = (self.header["DAXRAT"], self.header["DPOSANG"], self.header["DLOGH"])
        grad = gradient_norm(chi_cube, (e, t, r), steps)
        ax_ratio_std, angle_std, log_r_std = params_std(chi_cube, (e, t, r), steps)

        parameters = {'R_IDX': r, 'E_IDX': e, 'T_IDX': t,
                      'LOGR': self.log_r[r], 'AX_RATIO': self.ax_ratio[e], 'ANGLE': self.angle[t],
                      'LOGR_CHI': log_r_chi, 'LOGR_VAR': log_r_var, 'LOGR_CHI_VAR': log_r_chi_var,
                      'LOGR_INTERP': interp_r, 'AX_RATIO_INTERP': interp_ax_ratio,
                      'ANGLE_INTERP': interp_angle, 'LOGR_STD': log_r_std, 'AX_RATIO_STD': ax_ratio_std,
                      'ANGLE_STD': angle_std, 'CHI': min_chi, 'GRAD': grad}
        return parameters

    def make_mosaic(self, data, segment, model_index):
        model = self.convolved_models[model_index]
        flux_model = np.einsum("xy,xy", model, segment)
        flux_data = np.einsum("xy,xy", data, segment)
        scaled_model = (flux_data / flux_model) * model
        mosaic = np.zeros((4, 128, 128))
        mosaic[0] = data
        mosaic[1] = segment * (flux_data / segment.sum())
        mosaic[2] = scaled_model
        mosaic[3] = data - scaled_model
        mosaic = mosaic.reshape(128 * 4, 128).T
        return mosaic
