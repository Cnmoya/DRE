from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import h5py


def psf_clean(psf):
    mask = psf < 0
    psf[mask] = 0
    _sum = psf.sum()
    psf = psf / _sum
    return psf


psf_file = [i.strip() for i in open("psf_tiles.txt", "r")]
for f in psf_file:
    with h5py.File('{}'.format(f.replace(".psf", ".h5"), 'w')) as F:
        lista = []
        data = fits.open(f)
        for i in range(1, 33):
            psf = psf_clean(data[i].data[0][0][0])
            lista.append(psf)
        F['data'] = lista
