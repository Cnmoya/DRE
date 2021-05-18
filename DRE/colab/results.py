from h5py import File
from astropy.table import QTable, join, vstack
import astropy.units as u
from astropy.io import fits, ascii
import os
import numpy as np
import matplotlib.pyplot as plt


class Result:

    def __init__(self, model):
        self.model = model
        self.table = QTable()
        self.name = None
        self.model_index = []
        self.image = None

    def __getitem__(self, item):
        return self.table.__getitem__(item)

    def __repr__(self):
        return self.table.__repr__()

    def load_file(self, chi_file):
        with File(chi_file, 'r') as chi_h5f:
            self.name = os.path.basename(chi_file).replace('_chi.h5', '')

            names = list(chi_h5f.keys())

            self.table['EXT_NUMBER'] = np.zeros(len(names), dtype='int')
            self.table['NUMBER'] = np.zeros(len(names), dtype='int')
            self.table['LOGR'] = np.zeros(len(names), dtype='float')
            self.table['LOGR_POND'] = np.zeros(len(names), dtype='float')
            self.table['LOGR_VAR'] = np.zeros(len(names), dtype='float')
            self.table['AX_RATIO'] = np.zeros(len(names), dtype='float')
            self.table['ANGLE'] = np.zeros(len(names), dtype='float') * u.deg
            self.table['CHI'] = np.zeros(len(names), dtype='float')
            for i, name in enumerate(names):
                ext, numb = name.split('_')
                self.table['EXT_NUMBER'][i] = int(ext)
                self.table['NUMBER'][i] = int(numb)

                chi_cube = chi_h5f[name][:]
                self.model_index.append(np.unravel_index(np.argmin(chi_cube), chi_cube.shape))
                ratio, angle, logr, chi = self.model.get_parameters(chi_cube)
                self.table['LOGR'][i] = logr
                self.table['AX_RATIO'][i] = ratio
                self.table['ANGLE'][i] = angle * u.deg
                self.table['CHI'][i] = chi
                logr_pond, logr_var = self.model.pond_rad_3d(chi_cube)
                self.table['LOGR_POND'][i] = logr_pond
                self.table['LOGR_VAR'][i] = logr_var

    def visualize_detections(self):
        pass

    def hist(self, key=None, bins=20):
        if key:
            plt.hist(self.table[key], bins=bins)
            plt.xlabel(key.lower(), fontsize=12)
            plt.show()
        else:
            plt.figure(figsize=(8, 8))
            for i, key in enumerate(['LOGR', 'LOGR_POND', 'AX_RATIO', 'ANGLE']):
                plt.subplot(2, 2, i+1)
                plt.hist(self.table[key], bins=bins)
                plt.xlabel(key.lower(), fontsize=12)
            plt.show()

    def plot(self, x_key, y_key):
        plt.scatter(self.table[x_key], self.table[y_key])
        plt.xlabel(x_key.lower(), fontsize=12)
        plt.xlabel(y_key.lower(), fontsize=12)
        plt.show()

    def join_catalog(self, cat_table):
        self.table = join(self.table, QTable(cat_table), join_type='inner')


class Results:
    def __init__(self, model):
        self.model = model
        self.results = []
        self.total_results = Result(self.model)
        self.total_results.name = "Total Results"

    def __getitem__(self, item):
        return self.results[item]

    def load_results(self, chi_dir, images_dir=None, catalogs_dir=None):
        _, _, files = next(os.walk(chi_dir))
        for chi_file in files:
            result = Result(self.model)
            result.load_file(f"{chi_dir}/{chi_file}")
            self.results.append(result)

        if images_dir is not None:
            self.set_images(images_dir)
        if catalogs_dir is not None:
            self.set_catalogs(catalogs_dir)

        self.total_results.table = vstack(self.results)

    def set_images(self, images_dir):
        for result in self.results:
            result.image = f"{images_dir}/{result.name}.fits"

    def set_catalogs(self, catalogs_dir):
        for result in self.results:
            if os.path.isfile(f"{catalogs_dir}/{result.name}_cat.fits"):
                cat = fits.open(f"{catalogs_dir}/{result.name}_cat.fits")
            elif os.path.isfile(f"{catalogs_dir}/{result.name}_cat.cat"):
                cat = ascii.read(f"{catalogs_dir}/{result.name}_cat.cat", format='sextractor')
            else:
                raise ValueError("Can't find catalog in known format, "
                                 "check that the filename is name_cat.cat or name_cat.fits")
            result.join_catalog(cat)
