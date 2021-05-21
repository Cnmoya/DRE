from astropy.io import fits


class ModelsCube:
    def __init__(self, _file):
        self.data = fits.getdata(_file)

        self.header = fits.getheader(_file)

    def __getitem__(self, index):
        i, j, k = index
        return self.data[k, j * 128:(j + 1) * 128, i * 128:(i + 1) * 128]

    @property
    def shape(self):
        return self.data.shape[-1::]

    @property
    def LOGH(self):
        return [self.header["LOGH0"] + i * self.header["DLOGH"] for i in range(self.header["NLOGH"])]

    @property
    def POSANG(self):
        return [self.header["POSANG0"] + i * self.header["DPOSANG"] for i in range(self.header["NPOSANG"])]

    @property
    def AXRAT(self):
        return [self.header["AXRAT0"] + i * self.header["DAXRAT"] for i in range(self.header["NAXRAT"])]

    def convolve(self, psf):
        import astropy.convolution as convolution
        kpsf = convolution.CustomKernel(psf)
        for k in range(10):
            self.data[k, :, :] = convolution.convolve(self.data[k, :, :], kpsf)

        print('convolved!')
