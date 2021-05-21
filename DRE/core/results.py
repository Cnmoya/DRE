from astropy.table import Table
from collections import defaultdict
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

    def save(self):
        os.makedirs('Summary', exist_ok=True)
        table = Table(self.parameters)
        table.write(f'Summary/{self.name}_tab.fits', overwrite=True)