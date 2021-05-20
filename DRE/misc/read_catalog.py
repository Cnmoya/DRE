from astropy.io import fits
from astropy.table import Table


def cat_to_table(basename):
    cat_fits = fits.open(f"{basename}_cat.fits")
    data_idx = 1
    for i in range(len(cat_fits)):
        if cat_fits[i].name == 'LDAC_OBJECTS':
            data_idx = i
    cat = Table(cat_fits[data_idx].data)
    cat_fits.close()
    return cat
