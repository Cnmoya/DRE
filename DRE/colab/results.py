from h5py import File
from astropy.table import QTable, join, vstack
from astropy.io import fits
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import quantity_support
from DRE.misc.read_catalog import cat_to_table

quantity_support()


class Result:

    def __init__(self, model, output_dir):
        self.model = model
        self.output_dir = output_dir
        self.table = QTable()
        self.name = None
        self.image = None
        self.cuts = None
        self.psf = None

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

    def save(self):
        self.table.write(f"{self.output_dir}/{self.name}_tab.fits", overwrite=True)

    def load_summary(self, summary):
        self.table = cat_to_table(summary)

    def load_chi(self, chi_file):
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
            self.table['ANGLE'] = np.zeros(len(names), dtype='float')
            self.table['CHI'] = np.zeros(len(names), dtype='float')

            for i, name in enumerate(names):
                ext, numb = name.split('_')
                self.table['EXT_NUMBER'][i] = int(ext)
                self.table['NUMBER'][i] = int(numb)

                chi_cube = chi_h5f[name][:]
                ratio, angle, logr, chi, (e_idx, t_idx, r_idx) = self.model.get_parameters(chi_cube)
                self.table['LOGR'][i] = logr
                self.table['AX_RATIO'][i] = ratio
                self.table['ANGLE'][i] = angle
                self.table['CHI'][i] = chi
                self.table['R_IDX'][i] = r_idx
                self.table['E_IDX'][i] = e_idx
                self.table['T_IDX'][i] = t_idx

                logr_pond, logr_var = self.model.pond_rad_3d(chi_cube)
                self.table['LOGR_POND'][i] = logr_pond
                self.table['LOGR_VAR'][i] = logr_var

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
            model_idx = row['E_IDX', 'T_IDX', 'R_IDX']

            with File(f"{self.cuts}/{self.name}_cuts.h5", 'r') as cuts_h5f:
                cuts = cuts_h5f[f'{ext_number}_{cat_number}']
                data = cuts['obj'][:]
                segment = cuts['seg'][:]

            self.model.convolve(self.psf)
            mosaic = self.model.make_mosaic(data, segment, model_idx)

            if save:
                os.makedirs(mosaics_dir, exist_ok=True)
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
    def __init__(self, model, output_dir='Summary', chi_dir='Chi', images_dir='Tiles', cuts_dir='Cuts', psf_dir='PSF',
                 catalogs_dir='Sextracted'):
        self.model = model
        self.output_dir = output_dir
        self.results = []
        self.total_results = Result(self.model, self.output_dir)
        self.total_results.name = "Total Results"
        self.load_results(chi_dir, images_dir, cuts_dir, psf_dir, catalogs_dir)

    def __getitem__(self, item):
        return self.results[item]

    def __len__(self):
        return len(self.results)

    def load_results(self, chi_dir, images_dir, cuts_dir, psf_dir, catalogs_dir):
        if os.path.isdir(self.output_dir):
            _, _, files = next(os.walk(self.output_dir))
            for summary_file in sorted(files):
                result = Result(self.model, self.output_dir)
                result.load_summary(f"{self.output_dir}/{summary_file}")
                self.results.append(result)
        else:
            _, _, files = next(os.walk(chi_dir))
            for chi_file in sorted(files):
                result = Result(self.model, self.output_dir)
                result.load_chi(f"{chi_dir}/{chi_file}")
                self.results.append(result)

        self.set_images(images_dir)
        self.set_cuts(cuts_dir)
        self.set_psf(psf_dir)
        self.set_catalogs(catalogs_dir)

        self.total_results.table = vstack([result.table for result in self.results])

    def set_images(self, images_dir):
        for result in self.results:
            result.image = f"{images_dir}/{result.name}.fits"

    def set_cuts(self, cuts_dir):
        for result in self.results:
            result.cuts = cuts_dir

    def set_psf(self, psf_dir):
        for result in self.results:
            result.psf = f"{psf_dir}/{result.name}_psf.h5"

    def set_catalogs(self, catalogs_dir):
        for result in self.results:
            if os.path.isdir(f"{catalogs_dir}/{result.name}"):
                basename = f"{catalogs_dir}/{result.name}/{result.name}"
            else:
                basename = f"{catalogs_dir}/{result.name}"
            cat = cat_to_table(basename)
            result.join_catalog(cat)
