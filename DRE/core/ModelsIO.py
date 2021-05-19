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

        self.log_r = None
        self.angle = None
        self.ax_ratio = None

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

    def convolve(self, psf_file: str):
        pass

    def dre_fit(self, data, segment, noise):
        pass

    def fit_file(self, input_file, output_file, progress_status=''):
        pass

    def fit_dir(self, input_dir='Cuts', output_dir='Chi'):
        # list with input files in input_dir
        _, _, files = next(os.walk(input_dir))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        for i, filename in enumerate(files):
            input_file = f"{input_dir}/{filename}"
            name = os.path.basename(filename).replace('_cuts.h5', '')
            output_file = f"{output_dir}/{name}_chi.h5"
            if os.path.isfile(output_file):
                os.remove(output_file)
            # fit all cuts in each file
            self.fit_file(input_file, output_file, progress_status=f"({i + 1}/{len(files)})")

    def get_parameters(self, chi_cube):
        e, t, r = np.unravel_index(np.nanargmin(chi_cube), chi_cube.shape)
        min_chi = np.nanmin(chi_cube)
        return self.ax_ratio[e], self.angle[t], self.log_r[r], min_chi, (e, t, r)


    def pond_rad_3d(self, chi_cube):
        r_pond = np.sum((10 ** self.log_r) / chi_cube)
        r_pond = r_pond / np.sum(1. / chi_cube)
        log_r_pond = np.log10(r_pond)

        r_var = np.sum(((10 ** self.log_r - r_pond) ** 2) / chi_cube)
        r_var = r_var / np.sum(1. / chi_cube)
        log_r_var = np.log10(r_var)
        return log_r_pond, log_r_var
