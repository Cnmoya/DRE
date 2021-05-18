from astropy.io import fits
import os
import numpy as np
import DRE
from DRE.misc.h5py_compression import compression_types


class ModelsCube:
    def __init__(self, models_file=None, out_compression='none'):

        self.models = None
        self.header = None
        self.original_shape = None

        self.compression = compression_types[out_compression]

        if models_file is None:
            dre_dir = os.path.dirname(os.path.realpath(DRE.__file__))
            models_file = os.path.join(dre_dir, 'models', 'modelbulge.fits')
        self.load_models(models_file)

    def __getitem__(self, index):
        i, j, k = index
        return self.models[i, j, k]

    @property
    def shape(self):
        return self.models.shape

    @property
    def log_r(self):
        return [self.header["LOGH0"] + i * self.header["DLOGH"] for i in range(self.header["NLOGH"])]

    @property
    def angle(self):
        return [self.header["POSANG0"] + i * self.header["DPOSANG"] for i in range(self.header["NPOSANG"])]

    @property
    def ax_ratio(self):
        return [self.header["AXRAT0"] + i * self.header["DAXRAT"] for i in range(self.header["NAXRAT"])]

    def load_models(self, models_file):
        cube = fits.getdata(models_file).astype('float')
        self.original_shape = cube.shape
        cube = cube.reshape(10, 13, 128, 21, 128)
        cube = cube.swapaxes(2, 3)
        self.models = cube
        self.header = fits.getheader(models_file)

    def save_model(self, output_file):
        cube = self.models.swapaxes(2, 3)
        cube = cube.reshape(self.original_shape)
        models_fits = fits.ImageHDU(data=cube)
        models_fits.writeto(output_file, overwrite=True)

    def convolve(self, psf_file: str):
        pass

    def dre_fit(self, data, segment, noise):
        pass

    def fit_file(self, input_file, output_file, progress_status=''):
        pass

    def fit_dir(self, input_dir, output_dir):
        # list with input files in input_dir
        _, _, files = next(os.walk(input_dir))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        for i, filename in enumerate(files):
            input_file = f"{input_dir}/{filename}"
            name, _ = os.path.basename(filename).replace('_cuts.h5', '')
            output_file = f"{output_dir}/{name}_chi.h5"
            if os.path.isfile(output_file):
                os.remove(output_file)
            # fit all cuts in each file
            self.fit_file(input_file, output_file, progress_status=f"({i + 1}/{len(files)})")

    # noinspection PyTypeChecker
    def make_mosaic_h5(self, data, segment, index, output_file):
        e, t, r = index
        model = self.models[e, t, r]
        flux_model = np.einsum("xy,xy", model, segment)
        flux_data = np.einsum("xy,xy", data, segment)
        scaled_model = (flux_data / flux_model) * model
        mosaic = np.zeros((4, 128, 128))
        mosaic[0] = data
        mosaic[1] = segment * (flux_data / segment.sum())
        mosaic[2] = scaled_model
        mosaic[3] = data - scaled_model

        mosaic_fits = fits.ImageHDU(data=mosaic.reshape(128 * 4, 128).T)
        mosaic_fits.writeto(output_file, overwrite=True)

    def get_parameters(self, chi_cube):
        e, t, r = np.unravel_index(np.argmin(chi_cube[1].data), shape=(10, 13, 21))
        min_chi = np.min(chi_cube)
        return self.log_r[r], self.ax_ratio[e], self.angle[t], min_chi

    def pond_rad_3d(self, chi_cube):
        sqrt_chi = np.sqrt(chi_cube)
        r_weight = 0
        for e in range(chi_cube.shape[0]):
            for t in range(chi_cube.shape[1]):
                for r in range(chi_cube.shape[2]):
                    r_weight += (10 ** self.log_r[r]) / sqrt_chi[e, t, r]

        log_r_pond = np.log10(r_weight / np.sum(1. / sqrt_chi))

        log_r_var = 0
        for e in range(10):
            for t in range(13):
                for r in range(21):
                    log_r_var += ((self.log_r[r] - log_r_pond) ** 2) / (chi_cube[e, t, r])

        log_r_var = log_r_var / np.sum(1. / log_r_pond)
        return log_r_pond, log_r_var