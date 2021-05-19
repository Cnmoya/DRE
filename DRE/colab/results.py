from h5py import File
from astropy.table import QTable, join, vstack
import astropy.units as u
from astropy.io import fits, ascii
import os
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from astropy.visualization import quantity_support

quantity_support()


class Result:

    def __init__(self, model):
        self.model = model
        self.table = QTable()
        self.name = None
        self.image = None
        self.cuts = None

    def __getitem__(self, item):
        return self.table.__getitem__(item)

    def __repr__(self):
        return self.table.__repr__()

    def __len__(self):
        return len(self.table)

    def loc(self, cat_number, ext_number=0):
        return self.table.loc['EXT_NUMBER', ext_number].loc['NUMBER', cat_number]

    def row(self, i):
        return self.table.loc['ROW', i]

    def load_file(self, chi_file):
        with File(chi_file, 'r') as chi_h5f:
            self.name = os.path.basename(chi_file).replace('_chi.h5', '')

            names = list(chi_h5f.keys())

            self.table['ROW'] = np.arange(len(names), dtype='int')
            self.table['EXT_NUMBER'] = np.zeros(len(names), dtype='int')
            self.table['NUMBER'] = np.zeros(len(names), dtype='int')
            self.table['R_IDX'] = np.zeros(len(names), dtype='int')
            self.table['E_IDX'] = np.zeros(len(names), dtype='int')
            self.table['T_IDX'] = np.zeros(len(names), dtype='int')
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
                ratio, angle, logr, chi, (e_idx, t_idx, r_idx) = self.model.get_parameters(chi_cube)
                self.table['LOGR'][i] = logr
                self.table['AX_RATIO'][i] = ratio
                self.table['ANGLE'][i] = angle * u.deg
                self.table['CHI'][i] = chi
                self.table['R_IDX'][i] = r_idx
                self.table['E_IDX'][i] = e_idx
                self.table['T_IDX'][i] = t_idx

                logr_pond, logr_var = self.model.pond_rad_3d(chi_cube)
                self.table['LOGR_POND'][i] = logr_pond
                self.table['LOGR_VAR'][i] = logr_var

            self.table.add_index('ROW')
            self.table.add_index('EXT_NUMBER')
            self.table.add_index('NUMBER')

    def visualize_detections(self):
        pass

    def hist(self, key=None, **kwargs):
        if key:
            plt.hist(self.table[key], **kwargs)
            plt.xlabel(key.lower(), fontsize=12)
            plt.show()
        else:
            plt.figure(figsize=(8, 8))
            for i, key in enumerate(['LOGR', 'LOGR_POND', 'AX_RATIO', 'ANGLE']):
                plt.subplot(2, 2, i + 1)
                plt.hist(self.table[key], **kwargs)
                plt.xlabel(key.lower(), fontsize=12)
            plt.show()

    def plot(self, x_key, y_key):
        plt.scatter(self.table[x_key], self.table[y_key])
        plt.xlabel(x_key.lower(), fontsize=12)
        plt.xlabel(y_key.lower(), fontsize=12)
        plt.show()

    def join_catalog(self, cat_table):
        self.table = join(self.table, QTable(cat_table), join_type='inner')
        self.table.add_index('ROW')
        self.table.add_index('EXT_NUMBER')
        self.table.add_index('NUMBER')

    def make_mosaic(self, i, save=False, mosaics_dir='Mosaics', cmap='gray', **kwargs):
        if self.cuts:
            row = self.row(i)
            cat_number, ext_number = row['NUMBER', 'EXT_NUMBER']
            e, t, r = row['E_IDX', 'T_IDX', 'R_IDX']
            model = cp.asnumpy(self.model[e, t, r])

            with File(f"{self.cuts}/{self.name}_cuts.h5", 'r') as cuts_h5f:
                cuts = cuts_h5f[f'{ext_number}_{cat_number}']
                data = cuts['obj'][:]
                segment = cuts['seg'][:]

            flux_model = np.einsum("xy,xy", model, segment)
            flux_data = np.einsum("xy,xy", data, segment)
            scaled_model = (flux_data / flux_model) * model
            mosaic = np.zeros((4, 128, 128))
            mosaic[0] = data
            mosaic[1] = segment * (flux_data / segment.sum())
            mosaic[2] = scaled_model
            mosaic[3] = data - scaled_model
            mosaic = mosaic.reshape(128 * 4, 128).T
            if save:
                if not os.path.isdir(mosaics_dir):
                    os.mkdir(mosaics_dir)
                mosaic_fits = fits.ImageHDU(data=mosaic)
                mosaic_fits.writeto(f"{mosaics_dir}/{self.name}_{ext_number:02d}_{cat_number:04d}_mosaic.fits",
                                    overwrite=True)
            else:
                plt.figure(figsize=(15, 5))
                plt.imshow(mosaic, cmap, **kwargs)
                plt.axis('off')
                plt.show()
        else:
            print("You should define the cuts image first")


class Results:
    def __init__(self, model, chi_dir='Chi', images_dir='Tile', cuts_dir='Cuts', catalogs_dir='Sextracted'):
        self.model = model
        self.results = []
        self.total_results = Result(self.model)
        self.total_results.name = "Total Results"
        self.load_results(chi_dir, images_dir, cuts_dir, catalogs_dir)

    def __getitem__(self, item):
        return self.results[item]

    def __len__(self):
        return len(self.results)

    def load_results(self, chi_dir, images_dir, cuts_dir, catalogs_dir):
        _, _, files = next(os.walk(chi_dir))
        for chi_file in files:
            result = Result(self.model)
            result.load_file(f"{chi_dir}/{chi_file}")
            self.results.append(result)

        self.set_images(images_dir)
        self.set_cuts(cuts_dir)
        self.set_catalogs(catalogs_dir)

        self.total_results.table = vstack([result.table for result in self.results])

    def set_images(self, images_dir):
        for result in self.results:
            result.image = f"{images_dir}/{result.name}.fits"

    def set_cuts(self, cuts_dir):
        for result in self.results:
            result.cuts = cuts_dir

    def set_catalogs(self, catalogs_dir):
        for result in self.results:
            if os.path.isdir(f"{catalogs_dir}/{result.name}"):
                cat_dir = f"{catalogs_dir}/{result.name}"
            else:
                cat_dir = catalogs_dir
            if os.path.isfile(f"{cat_dir}/{result.name}_cat.fits"):
                cat = fits.open(f"{cat_dir}/{result.name}_cat.fits")
                result.join_catalog(cat)
            elif os.path.isfile(f"{cat_dir}/{result.name}_cat.cat"):
                cat = ascii.read(f"{cat_dir}/{result.name}_cat.cat", format='sextractor')
                result.join_catalog(cat)