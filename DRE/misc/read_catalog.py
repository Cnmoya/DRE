from astropy.io import fits
from astropy.table import QTable, vstack


def cat_to_table(filename):
    try:
        cat_fits = fits.open(filename)
        tables = []
        for i in range(len(cat_fits)):
            if cat_fits[i].name == 'LDAC_OBJECTS':
                tables.append(QTable(cat_fits[i].data))
        cat = vstack(tables)
        cat_fits.close()
        return cat
    except FileNotFoundError:
        print(f"Warning: can't find the catalog {filename}, omitting this operation")
        return QTable()
