from astropy.table import Table, join
from astropy.io import ascii
from collections import defaultdict
from DRE.misc.read_catalog import cat_to_table
import os


class Summary:

    def __init__(self, name):
        self.name = name
        self.parameters = defaultdict(list)
        self.row_idx = 0

    def append(self, params):
        self.parameters['ROW'].append(self.row_idx)
        self.row_idx += 1
        for key, value in params.items():
            self.parameters[key].append(value)

    def save(self, catalogs_dir='Sextracted'):
        os.makedirs('Summary', exist_ok=True)
        table = Table(self.parameters)
        if os.path.isdir(f"{catalogs_dir}/{self.name}"):
            cat_file = f"{catalogs_dir}/{self.name}/{self.name}_cat.fits"
        else:
            cat_file = f"{catalogs_dir}/{self.name}_cat.fits"
        table = join(table, cat_to_table(cat_file), join_type='inner')
        if 'VIGNET' in table.colnames:
            table.remove_column('VIGNET')
        ascii.write(table=table, output=f"Summary/{self.name}.dat", overwrite=True)
