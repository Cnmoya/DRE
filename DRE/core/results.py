from astropy.table import Table
from collections import defaultdict
import os


class Summary:

    def __init__(self, name):
        self.name = name
        self.parameters = defaultdict(list)

    def append(self, params):
        for key, value in params.items():
            self.parameters[key].append(value)

    def save(self):
        os.makedirs('Summary', exist_ok=True)
        table = Table(self.parameters)
        table.write(f'Summary/{self.name}_tab.fits', overwrite=True)
