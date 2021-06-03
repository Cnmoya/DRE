from astropy.io import fits
from astropy.table import Table, vstack


def cat_to_table(filename):
    cat_fits = fits.open(filename)
    tables = []
    for i in range(len(cat_fits)):
        if cat_fits[i].name == 'LDAC_OBJECTS':
            tables.append(Table(cat_fits[i].data))
    cat = vstack(tables)
    cat_fits.close()
    return cat
