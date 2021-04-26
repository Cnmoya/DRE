from astropy.io import fits, ascii
from astropy.nddata import Cutout2D
from scipy.ndimage import shift
from photutils.centroids import centroid_1dg
from h5py import File
import argparse
import os


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", help="file with list of images to cut (def: tile_names.txt)",
                        type=str, default="tile_names.txt")
    parser.add_argument("-t", "--tile", help="root directory to original data (def: Tile)", default="Tile")
    parser.add_argument("-s", "--sextracted", help="root directory to sextracted data (def: Sextracted)",
                        type=str, default="Sextracted")
    parser.add_argument("-o", "--output", help="output directory (def: Output)", type=str, default="Output")
    parser.add_argument("--margin", help="exclude objects outside a margin (def: 80)", type=int, default=80)
    parser.add_argument("--max-stellarity", help="maximum SexTractor stellarity to be considered a galaxy (def: 0.5)",
                        default=0.5)

    args_ = parser.parse_args()
    return args_


def condition(_row, header, margin, max_stellarity):
    inside_x = margin < _row['X_IMAGE'] < header['NAXIS1'] - margin
    inside_y = margin < _row["Y_IMAGE"] < header['NAXIS2'] - margin
    is_galaxy = _row['CLASS_STAR'] < max_stellarity
    if inside_x and inside_y and is_galaxy:
        return True
    else:
        return False


def cut_object(fits_data, cat_row, size=128):
    return Cutout2D(fits_data[cat_row['EXT_NUMBER']].data,
                    (cat_row["X_IMAGE"] - 1, cat_row["Y_IMAGE"] - 1),
                    size).data.copy()


def cut_all(name, output, seg, nob, data, cats, noise, margin, max_stellarity):
    for row in cats[1].data:
        if condition(row, data[row['EXT_NUMBER']].header, margin, max_stellarity):
            db1 = cut_object(nob, row, size=12)
            xo, yo = centroid_1dg(db1)
            x_set, y_set = 5.5 - xo, 5.5 - yo
            db = cut_object(nob, row)
            ds = cut_object(seg, row)
            drms = cut_object(noise, row)
            mds = ds == row["NUMBER"]
            ds[~mds] = 0
            ds[mds] = 1
            with File(f"{output}/{name}/{name}_cut_{row['EXT_NUMBER']:02d}_{row['NUMBER']:03d}.h5", 'w') as h5f:
                h5f.create_dataset('obj', data=shift(db, (y_set, x_set)))
                h5f.create_dataset('seg', data=shift(ds, (y_set, x_set)))
                h5f.create_dataset('rms', data=shift(drms, (y_set, x_set)))
        else:
            pass


if __name__ == "__main__":
    args = parse_arguments()

    names = [i.strip() for i in open(args.input, 'r')]
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    for name in names:
        print(name)
        if not os.path.exists(os.path.join(args.output, name)):
            os.mkdir(os.path.join(args.output, name))
        seg = fits.open(f"{args.sextracted}/{name}_seg.fits")
        nob = fits.open(f"{args.sextracted}/{name}_nb.fits")
        data = fits.open(f"{args.tile}/{name}.fit")
        noise = fits.open(f"{args.sextracted}{name}_rms.fits")

        cats = ascii.read(f"{args.sextracted}/{name}_catalog.cat", format='sextractor')

        cut_all(name, args.output, seg, nob, data, cats, noise)
