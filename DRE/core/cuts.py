from astropy.io import fits, ascii
from astropy.table import Table
from astropy.nddata import Cutout2D
from scipy.ndimage import shift
from photutils.centroids import centroid_1dg
from h5py import File
import numpy as np
import os
from DRE.misc.progress_bar import progress
from DRE.misc.h5py_compression import compression_types
from DRE.misc.read_catalog import cat_to_table


class Cutter:

    def __init__(self, margin=80, max_stellarity=0.5, compression='none'):

        self.margin = margin
        self.max_stellarity = max_stellarity
        self.compression = compression_types[compression]

    def condition(self, _row, header):
        inside_x = self.margin < _row['X_IMAGE'] < header['NAXIS1'] - self.margin
        inside_y = self.margin < _row["Y_IMAGE"] < header['NAXIS2'] - self.margin
        is_galaxy = _row['CLASS_STAR'] < self.max_stellarity
        if inside_x and inside_y and is_galaxy:
            return True
        else:
            return False

    @staticmethod
    def cut_object(fits_data, cat_row, ext_number, size=128):
        return Cutout2D(fits_data[ext_number].data,
                        (cat_row["X_IMAGE"] - 1, cat_row["Y_IMAGE"] - 1),
                        size).data.copy()

    def cut_image(self, cat, out_name, seg, obj, data, noise, progress_status):
        cut = 0
        with File(out_name, 'w') as h5_file:
            for j, row in enumerate(cat):
                ext_number = row['EXT_NUMBER'] if 'EXT_NUMBER' in row.keys() else 0
                if self.condition(row, data[ext_number].header):
                    # este es para filtrar los nan (ocultos por sextractor)
                    data_cut = self.cut_object(data, row, ext_number)
                    if not np.isnan(np.sum(data_cut)):
                        # cortes
                        obj_cut = self.cut_object(obj, row, ext_number)
                        seg_cut = self.cut_object(seg, row, ext_number)
                        rms_cut = self.cut_object(noise, row, ext_number)

                        seg_mask = seg_cut == row["NUMBER"]
                        seg_cut[~seg_mask] = 0
                        seg_cut[seg_mask] = 1

                        # algo raro aca, shifts muy altos
                        mini_obj = self.cut_object(obj, row, ext_number, size=12)
                        xo, yo = centroid_1dg(mini_obj)
                        x_shift, y_shift = 5.5 - xo, 5.5 - yo
                        h5_group = h5_file.create_group(f"{ext_number}_{row['NUMBER']}")
                        h5_group.create_dataset('obj', data=shift(obj_cut, (y_shift, x_shift)),
                                                dtype='float32', **self.compression)
                        h5_group.create_dataset('seg', data=shift(seg_cut, (y_shift, x_shift)),
                                                dtype='float32', **self.compression)
                        h5_group.create_dataset('rms', data=shift(rms_cut, (y_shift, x_shift)),
                                                dtype='float32', **self.compression)

                        progress(j + 1, len(cat), progress_status)
                        cut += 1
        print(f"\n{progress_status}: {cut} cuts")

    def cut_tiles(self, tiles='Tiles', sextracted='Sextracted', output='Cuts'):
        _, _, files = next(os.walk(tiles))
        if not os.path.exists(output):
            os.mkdir(output)
        for i, filename in enumerate(sorted(files)):
            name, _ = os.path.splitext(os.path.split(filename)[1])
            if os.path.isdir(f"{sextracted}/{name}"):
                basename = f"{sextracted}/{name}/{name}"
            else:
                basename = f"{sextracted}/{name}"

            seg = fits.open(f"{basename}_seg.fits")
            obj = fits.open(f"{basename}_nb.fits")
            noise = fits.open(f"{basename}_rms.fits")
            data = fits.open(f"{tiles}/{name}.fits")
            cat = cat_to_table(basename)

            out_name = f"{output}/{name}_cuts.h5"
            if os.path.isfile(out_name):
                os.remove(out_name)
            progress_status = f"({i + 1}/{len(files)})"
            print(f"{progress_status}: {name}")
            self.cut_image(cat, out_name,
                           seg, obj, data, noise,
                           progress_status)
            seg.close()
            obj.close()
            noise.close()
            data.close()

