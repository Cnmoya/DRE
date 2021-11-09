from astropy.io import fits
import numpy


def psf_clean(psf):
    psf[psf < 0] = 0
    psf = psf / psf.sum()
    return psf


def get_psf(filename, ext_number=0):
    with fits.open(filename) as hdul:
        data = hdul[ext_number + 1].data
        # psf to order 0
        psf = psf_clean(data[0][0][0])
        return psf
