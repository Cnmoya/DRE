from astropy.io import fits
import os


def psf_clean(psf):
    psf[psf < 0] = 0
    psf = psf / psf.sum()
    return psf


def get_psf(filename):
    _, file_extension = os.path.splitext(filename)
    if file_extension == '.fits':
        data = fits.getdata(filename)
        psf = psf_clean(data)
        return psf
    elif file_extension == '.psf':
        with fits.open(filename) as hdul:
            # PSF in PSFex format:
            for i in range(1, len(hdul)):
                if hdul[i].header['ACCEPTED'] != 0:
                    data = hdul[i].data
                    # psf to order 0
                    psf = psf_clean(data[0][0][0])
                    return psf
            raise ValueError(f"No accepted PSF found in '{filename}', check the log from PSFex")
    else:
        raise ValueError(f"Unknown extension '{file_extension}' for the PSF file, \n"
                         f"should be '.fits' or '.psf'")
