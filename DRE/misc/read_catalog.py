import os
from astropy.io import ascii, fits
from astropy.table import Table


def cat_to_table(basename):
    if os.path.isfile(f"{basename}_cat.fits"):
        cat_fits = fits.open(f"{basename}_cat.fits")
        data_idx = 1
        for i in range(len(cat_fits)):
            if cat_fits[i].name == 'LDAC_OBJECTS':
                data_idx = i
        cat = Table(cat_fits[data_idx].data)
        cat_fits.close()
    elif os.path.isfile(f"{basename}_cat.cat"):

        cat = Table(ascii.read(f"{basename}_cat.cat", format='sextractor'))
    else:
        raise ValueError("Can't find catalog in known format, "
                         "check that the filename is name_cat.cat or name_cat.fits")
    return cat
