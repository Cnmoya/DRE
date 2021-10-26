import numpy as np
from h5py import File
from astropy.table import QTable, join, vstack
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.visualization import quantity_support
from DRE.misc.read_catalog import cat_to_table
from collections import defaultdict
import os

quantity_support()


class Result:

    def __init__(self, model=None, output_dir=None, result_id=0):
        self.model = model
        self.output_dir = output_dir
        self.result_id = result_id
        self.table = QTable()
        self.name = None
        self.image = None
        self.cuts = None
        self.psf = None

    def __getitem__(self, key):
        return self.table.__getitem__(key)

    def __setitem__(self, key, value):
        return self.table.__setitem__(key, value)

    def __repr__(self):
        return self.table.__repr__()

    def __len__(self):
        return len(self.table)

    def show(self):
        return self.table.show_in_notebook()

    def loc(self, cat_number, ext_number=0):
        return self.table.loc['EXT_NUMBER', ext_number].loc['NUMBER', cat_number]

    def row(self, i):
        return self.table.loc['ROW', i]

    def save(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self.table.write(os.path.join(self.output_dir, f"{self.name}_dre.fits"), overwrite=True)

    def load_summary(self, summary):
        self.name = os.path.basename(summary).replace('_dre.fits', '')
        self.table = QTable.read(summary)
        self.table['ROW'] = np.arange(len(self.table))
        self.table['RESULT_ID'] = np.ones(len(self)) * self.result_id
        self.table.add_index('ROW')
        self.table.add_index('EXT_NUMBER')
        self.table.add_index('NUMBER')

    def load_chi(self, chi_file):
        self.name = os.path.basename(chi_file).replace('_chi.h5', '')
        parameters = defaultdict(list)
        with File(chi_file, 'r') as chi_h5f:
            self.name = os.path.basename(chi_file).replace('_chi.h5', '')
            names = list(chi_h5f.keys())
            for i, name in enumerate(names):
                parameters['ROW'].append(i)
                ext, numb = name.split('_')
                parameters['EXT_NUMBER'].append(int(ext))
                parameters['NUMBER'].append(int(numb))

                chi_cube = chi_h5f[name][:]
                params = self.model.get_parameters(chi_cube)
                for key, value in params.items():
                    parameters[key].append(value)
        self.table = QTable(parameters)
        self.table['RESULT_ID'] = self.result_id

    def visualize_detections(self):
        pass

    def hist(self, key=None, **kwargs):
        if key:
            plt.figure(figsize=(6, 6))
            plt.hist(self.table[key], **kwargs)
            plt.xlabel(key, fontsize=14)
            plt.show()
        else:
            plt.figure(figsize=(8, 8))
            for i, (key, label) in enumerate([('INDEX', r'$n$'), ('AX_RATIO', 'a/b'),
                                              ('ANGLE', r'$\theta$'), ('LOGR', r'$Log_{10}R$')]):
                plt.subplot(2, 2, i + 1)
                plt.hist(self.table[key], bins=self.model.shape[i], **kwargs)
                plt.xlabel(label, fontsize=14)
            plt.show()

    def plot(self, x_key=None, y_key=None, s=5, **kwargs):
        if x_key is not None and y_key is not None:
            plt.scatter(self.table[x_key], self.table[y_key], s=s, **kwargs)
            plt.xlabel(x_key.lower(), fontsize=14)
            plt.ylabel(y_key.lower(), fontsize=14)
            plt.show()
        else:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.scatter(self.table['LOGR_CHI'], self.table['LOGR_CHI_VAR'], s=s, **kwargs)
            plt.xlabel(r'$Log_{10}R_{\chi}$', fontsize=14)
            plt.ylabel(r'$\Delta^2 R_{\chi}$', fontsize=14)
            plt.subplot(1, 2, 2)
            plt.scatter(self.table['LOGR_VAR'], self.table['LOGR_CHI_VAR'], s=s, **kwargs)
            plt.xlabel(r'$\Delta^2 R$', fontsize=14)
            plt.ylabel(r'$\Delta^2 R_{\chi}$', fontsize=14)
            plt.tight_layout()
            plt.show()

    def join_catalog(self, cat_table, keys=None, table_names=('1', '2')):
        self.table = join(self.table, QTable(cat_table), join_type='inner', keys=keys, table_names=table_names)
        if 'EXT_NUMBER' not in self.table.colnames:
            self.table['EXT_NUMBER'] = self.table[f'EXT_NUMBER_{table_names[0]}']
        if 'NUMBER' not in self.table.colnames:
            self.table['NUMBER'] = self.table[f'NUMBER_{table_names[0]}']
        self.table.sort(['EXT_NUMBER', 'NUMBER'])
        self.table['ROW'] = np.arange(len(self.table))
        self.table.add_index('ROW')
        self.table.add_index('EXT_NUMBER')
        self.table.add_index('NUMBER')

    def get_data(self, i):
        row = self.row(i)
        cat_number, ext_number = row['NUMBER', 'EXT_NUMBER']

        with File(os.path.join(self.cuts, f"{self.name}_cuts.h5"), 'r') as cuts_h5f:
            cuts = cuts_h5f[f'{ext_number:02d}_{cat_number:04d}']
            data = cuts['obj'][:]
            segment = cuts['seg'][:]
            noise = cuts['rms'][:]
        return data, segment, noise

    def make_mosaic(self, i, save=False, mosaics_dir='Mosaics', cmap='gray', figsize=(15, 5), **kwargs):
        if self.cuts:
            row = self.row(i)
            cat_number, ext_number = row['NUMBER', 'EXT_NUMBER']
            data, segment, _ = self.get_data(i)

            mosaic = self.model.make_mosaic(data, segment, tuple(row['MODEL_IDX']), psf_file=self.psf)

            if save:
                os.makedirs(mosaics_dir, exist_ok=True)
                mosaic_fits = fits.ImageHDU(data=mosaic)
                mosaic_fits.writeto(os.path.join(mosaics_dir,
                                                 f"{self.name}_{ext_number:02d}_{cat_number:04d}_mosaic.fits"),
                                    overwrite=True)
            else:
                plt.figure(figsize=figsize)
                plt.imshow(mosaic, cmap, **kwargs)
                plt.axis('off')
                plt.show()
        else:
            print("You should define the cuts image first")

    def make_residuals(self, i, src_index_idx=-1, ax_ratio_idx=-1, save=False, residuals_dir='Residuals',
                       cmap='plasma', figsize=(20, 15), **kwargs):
        if self.cuts:
            row = self.row(i)
            cat_number, ext_number = row['NUMBER', 'EXT_NUMBER']
            data, segment, _ = self.get_data(i)

            if self.psf:
                self.model.convolve(self.psf, to_cpu=True)
            residual = self.model.make_residual(data, segment)

            if save:
                os.makedirs(residuals_dir, exist_ok=True)
                mosaic_fits = fits.ImageHDU(data=residual)
                mosaic_fits.writeto(os.path.join(residuals_dir,
                                                 f"{self.name}_{ext_number:02d}_{cat_number:04d}_residual.fits"),
                                    overwrite=True)
            else:
                residual_slice = residual[src_index_idx, ax_ratio_idx]
                plt.figure(figsize=figsize)
                title = f'a/b = {self.model.ax_ratio[ax_ratio_idx]:.1f}, n = {self.model.src_index[src_index_idx]:.1f}'
                plt.suptitle(title, fontsize=20, y=0.85)
                plt.imshow(residual_slice, cmap=cmap, **kwargs)
                plt.axis('off')
                plt.show()
        else:
            print("You should define the cuts image first")


class Results:
    def __init__(self, model=None, output_dir='Summary', chi_dir='Chi', images_dir='Tiles', cuts_dir='Cuts',
                 psf_dir='PSF', catalogs_dir='Sextracted', recompute=False):
        self.output_dir = output_dir
        self.results = []
        self.all_ = Result()
        self.all_.name = "Total Results"
        self.load_results(model, chi_dir, images_dir, cuts_dir, psf_dir, catalogs_dir, recompute)

    def __getitem__(self, item):
        return self.results[item]

    def __len__(self):
        return len(self.results)

    def save(self):
        for result in self.results:
            result.save()

    def load_results(self, model, chi_dir, images_dir, cuts_dir, psf_dir, catalogs_dir, recompute):
        if os.path.isdir(self.output_dir) and not recompute:
            print(f"loading results from {self.output_dir}")
            files = os.listdir(self.output_dir)
            for i, summary_file in enumerate(sorted(files)):
                result = Result(model, self.output_dir, result_id=i)
                result.load_summary(os.path.join(self.output_dir, summary_file))
                self.results.append(result)
        elif model is not None:
            print(f"loading results from {chi_dir}")
            files = os.listdir(chi_dir)
            for i, chi_file in enumerate(sorted(files)):
                result = Result(model, self.output_dir, result_id=i)
                result.load_chi(os.path.join(chi_dir, chi_file))
                self.results.append(result)
            self.set_catalogs(catalogs_dir)
        else:
            print(f"Can't load summary from {self.output_dir}, please set a model to compute the parameters")

        if os.path.isdir(images_dir):
            self.set_images(images_dir)
        else:
            pass
        if os.path.isdir(cuts_dir):
            self.set_cuts(cuts_dir)
        else:
            print(f"Can't find {cuts_dir} directory")
        if os.path.isdir(psf_dir):
            self.set_psf(psf_dir)
        else:
            print(f"Can't find {psf_dir} directory")

    @property
    def all(self):
        self.all_.table = vstack([result.table for result in self.results])
        return self.all_

    def set_images(self, images_dir):
        for result in self.results:
            result.image = os.path.join(images_dir, f"{result.name}.fits")

    def set_cuts(self, cuts_dir):
        for result in self.results:
            result.cuts = cuts_dir

    def set_psf(self, psf_dir):
        for result in self.results:
            result.psf = os.path.join(psf_dir, f"{result.name}.psf")

    def set_catalogs(self, catalogs_dir):
        for result in self.results:
            if os.path.isdir(os.path.join(catalogs_dir, result.name)):
                cat_file = os.path.join(catalogs_dir, result.name, f"{result.name}_cat.fits")
            else:
                cat_file = os.path.join(catalogs_dir, f"{result.name}_cat.fits")
            cat = cat_to_table(cat_file)
            result.join_catalog(cat)

    def hist(self, key=None, **kwargs):
        self.all.hist(key, **kwargs)

    def plot(self, x_key=None, y_key=None, s=5, **kwargs):
        self.all.plot(x_key, y_key, s, **kwargs)

    def show(self):
        return self.all.show()
