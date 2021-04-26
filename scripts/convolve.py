from astropy.io import fits
import h5py
from astropy.convolution import convolve

models = fits.open('Models/modelbulge.fits')
psf_file = [i.strip().replace("psf", "h5") for i in open("psf_tiles.txt", "r")]

for f in psf_file:
    psf = h5py.File(f, 'r')
    cubo = models[0].data.copy()
    for i in range(10):
        cubo[i, :, :] = convolve(cubo[i, :, :], psf['data'][12])
    aux = fits.ImageHDU(data=cubo, header=models[0].header)
    aux.writeto("Models/models_{}".format(f.replace('Catalogs/', "").replace(".psf", ".fits")))
