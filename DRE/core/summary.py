from astropy.table import QTable, join
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

    def save(self, save_dir='Summary', catalogs_dir='Sextracted'):
        os.makedirs(save_dir, exist_ok=True)
        table = QTable(self.parameters)
        if os.path.isdir(os.path.join(catalogs_dir, self.name)):
            cat_file = os.path.join(catalogs_dir, self.name, f"{self.name}_cat.fits")
        else:
            cat_file = os.path.join(catalogs_dir, f"{self.name}_cat.fits")
        table = join(table, cat_to_table(cat_file), join_type='inner')
        if 'VIGNET' in table.colnames:
            table.remove_column('VIGNET')
        table.write(os.path.join(save_dir, f"{self.name}_dre.fits"), overwrite=True)
