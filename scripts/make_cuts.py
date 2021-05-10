#!/bin/env python

from astropy.io import fits, ascii
from astropy.nddata import Cutout2D
from scipy.ndimage import shift
from photutils.centroids import centroid_1dg
from h5py import File
import numpy as np
import argparse
import os
import sys


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write(f'{status}: [{bar}] {percents}%\r')
    sys.stdout.flush()


def condition(_row, header, margin, max_stellarity):
    inside_x = margin < _row['X_IMAGE'] < header['NAXIS1'] - margin
    inside_y = margin < _row["Y_IMAGE"] < header['NAXIS2'] - margin
    is_galaxy = _row['CLASS_STAR'] < max_stellarity
    if inside_x and inside_y and is_galaxy:
        return True
    else:
        return False


def cut_object(fits_data, cat_row, ext_number, size=128):
    return Cutout2D(fits_data[ext_number].data,
                    (cat_row["X_IMAGE"] - 1, cat_row["Y_IMAGE"] - 1),
                    size).data.copy()


def cut_image(cats, out_name, seg, obj, data, noise, margin, max_stellarity, compression, progress_status):
    cut = 0
    with File(out_name, 'w') as h5_file:
        for j, row in enumerate(cats):
            cat_digits = len(str(len(cats)))
            ext_number = row['EXT_NUMBER'] if 'EXT_NUMBER' in row else 0
            if condition(row, data[ext_number].header, margin, max_stellarity):
                # este es para filtrar los nan (ocultos por sextractor)
                data_cut = cut_object(data, row, ext_number)
                if not np.isnan(np.sum(data_cut)):
                    # cortes
                    obj_cut = cut_object(obj, row, ext_number)
                    seg_cut = cut_object(seg, row, ext_number)
                    rms_cut = cut_object(noise, row, ext_number)

                    seg_mask = seg_cut == row["NUMBER"]
                    seg_cut[~seg_mask] = 0
                    seg_cut[seg_mask] = 1

                    # algo raro aca, shifts muy altos
                    mini_obj = cut_object(obj, row, ext_number, size=12)
                    xo, yo = centroid_1dg(mini_obj)
                    x_shift, y_shift = 5.5 - xo, 5.5 - yo
                    h5_group = h5_file.create_group(f"{ext_number:02d}_{row['NUMBER']:0{cat_digits}d}")
                    h5_group.create_dataset('obj', data=shift(obj_cut, (y_shift, x_shift)))
                    h5_group.create_dataset('seg', data=shift(seg_cut, (y_shift, x_shift)),
                                            dtype='float32', **compression)
                    h5_group.create_dataset('rms', data=shift(rms_cut, (y_shift, x_shift)),
                                            dtype='float32', **compression)

                    progress(j + 1, len(cats), progress_status)
                    cut += 1
    print(f"\n{progress_status}: {cut} cuts")


def cut_tiles(args):
    _, _, files = next(os.walk(args.tile))
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    for i, filename in enumerate(files):
        name, _ = os.path.splitext(os.path.split(filename)[1])
        basename = f"{args.sextracted}/{name}/{name}"
        seg = fits.open(f"{basename}_seg.fits")
        obj = fits.open(f"{basename}_obj_nb.fits")
        noise = fits.open(f"{basename}_rms.fits")
        cats = ascii.read(f"{basename}_cat.cat", format='sextractor')
        data = fits.open(f"{args.tile}/{name}.fits")

        out_name = f"{args.output}/{name}_cuts.h5"
        if os.path.isfile(out_name):
            os.remove(out_name)
        progress_status = f"({i + 1}/{len(files)})"
        print(f"{progress_status}: {name}")
        cut_image(cats, out_name,
                  seg, obj, data, noise,
                  args.margin, args.max_stellarity, args.compression,
                  progress_status)


if __name__ == "__main__":

    def parse_arguments():
        parser = argparse.ArgumentParser()

        parser.add_argument("-t", "--tile", help="root directory to original data (def: Tile)",
                            default="Tile", type=str)
        parser.add_argument("-s", "--sextracted", help="root directory to sextracted data (def: Sextracted)",
                            type=str, default="Sextracted")
        parser.add_argument("-o", "--output", help="output directory (def: Cuts)", type=str, default="Cuts")
        parser.add_argument("--margin", help="exclude objects outside a margin (def: 80)", type=int, default=80)
        parser.add_argument("--max-stellarity",
                            help="maximum SexTractor stellarity to be considered a galaxy (def: 0.5)",
                            default=0.5)
        parser.add_argument("--compression", help="compresion level for the h5 output file,"
                                                  "lower is faster (def: medium)",
                            choices=["none", "low", "medium", "high"], default="medium")

        args = parser.parse_args()

        compression_types = {"none": dict(),
                             "low": {"compression": "lzf"},
                             "medium": {"compression": "gzip", "compression_opts": 4},
                             "high": {"compression": "gzip", "compression_opts": 9}}
        args.compression = compression_types[args.compression]

        return args


    args_ = parse_arguments()
    cut_tiles(args_)
