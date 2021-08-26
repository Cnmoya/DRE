from astropy.io import fits
import os
import numpy as np
import numpy
import DRE
from DRE.misc.h5py_compression import compression_types
from DRE.misc.interpolation import fit_parabola_1d


class ModelsCube:
    """
    This object stores the models and performs the convolution and fit operation, also computes the parameters for
    the fit results and makes a mosaic to visualize it

    Attributes
    ----------
    models : ndarray
        numpy/cupy array with the cube of models, the axes are sorted as (ax_ratio, angle, log_r, x_image, y_image)
    convolved_models : ndarray
        numpy/cupy array with the cube of models convolved with a PSF
    header : dict
        astropy header of the models fits file
    original_shape : tuple
        shape of the models as saved in the fits file (ax_ratio, angle x x_image, log_r x y_image)
    src_index : ndarray
        numpy array with the Sérsic index axis
    log_r : ndarray
        numpy array with the log_r axis
    angle : ndarray
        numpy array with the angle axis
    ax_ratio : ndarray
        numpy array with the ax_ratio axis
    compression : dict
        dictionary with arguments for H5Py compression


    Methods
    -------
    load_models(models_file):
        loads the models fits file and reshapes it
    save_models(output_file):
        saves the convolved models into a fits file
    convolve(psf_file, *args, **kwargs):
        convolves the models whit the psf, is implemented in the child classes ModelsCPU and ModelsGPU depending on the
        acceleration method
    dre_fit(data, segment, noise, backend=numpy)
        performs the fit and returns a numpy/cupy array chi_squared residual between the model and the data inside the
        segment
    pond_rad_3d(chi_cube)
        computes a weighted radius which weights are 1/chi_squared for each model, also computes a wighted variance
        respect the minimum and the weighted radius, they are returned in log10 scale
    get_parameters(chi_cube)
        find the model that minimizes the chi_square and computes the parameters for this model
    make_mosaic(data, segment, model_index)
        makes a image with the data, the segment and the model, all scaled to the data  flux
    """

    def __init__(self, models_file=None, out_compression='none'):
        """
        Parameters
        ----------
        models_file : str
            the path to the fits file with the models
        out_compression : str
            compression level for the HDF5 output file, can be 'none', 'low', 'medium' or 'high'
        """

        self.models = None
        self.convolved_models = None
        self.header = None
        self.original_shape = None

        self.src_index = None
        self.log_r = None
        self.angle = None
        self.ax_ratio = None
        self.x_image = None
        self.y_image = None

        self.compression = compression_types[out_compression]

        if models_file is None:
            dre_dir = os.path.dirname(os.path.realpath(DRE.__file__))
            models_file = os.path.join(dre_dir, 'models', 'modelbulge.fits')
        self.load_models(models_file)

    def __getitem__(self, index):
        return self.models.__getitem__(index)

    @property
    def shape(self):
        """
        Returns
        -------
        tuple
            tuple with the shape of the models array
        """
        return self.models.shape

    @property
    def axes_names(self):
        """
        Returns
        -------
        tuple
            name of each axis
        """
        return 'src_index', 'ax_ratio', 'angle', 'log_r', 'x_image', 'y_image'

    @property
    def axes(self):
        """
        Returns
        -------
        tuple
            arrays of each axis
        """
        return self.src_index, self.ax_ratio, self.angle, self.log_r, self.x_image, self.y_image

    def load_models(self, models_file):
        """
        loads the models fits file and reshapes it, also loads the header and computes the axes

        Parameters
        ----------
        models_file : str
            the path to the fits file with the models
        """
        self.header = fits.getheader(models_file)

        cube = fits.getdata(models_file).astype('float')
        self.original_shape = cube.shape
        if "NINDEX" in self.header:
            cube = cube.reshape(self.header["NINDEX"], self.header["NAXRAT"], self.header["NPOSANG"],
                                self.header["BOXSIZE"], self.header["NLOGH"], self.header["BOXSIZE"])
        else:
            # the old cube without Sérsic index
            cube = cube.reshape(1, self.header["NAXRAT"], self.header["NPOSANG"],
                                self.header["BOXSIZE"], self.header["NLOGH"], self.header["BOXSIZE"])
            self.header["INDEX0"] = 4.0
            self.header["NINDEX"] = 1
            self.header["DINDEX"] = 0
        # swap log_r and x_image
        cube = cube.swapaxes(-2, -3)
        self.models = cube

        self.src_index = np.arange(self.header["NINDEX"]) * self.header["DINDEX"] + self.header["INDEX0"]
        self.log_r = np.arange(self.header["NLOGH"]) * self.header["DLOGH"] + self.header["LOGH0"]
        self.angle = np.arange(self.header["NPOSANG"]) * self.header["DPOSANG"] + self.header["POSANG0"]
        self.ax_ratio = np.arange(self.header["NAXRAT"]) * self.header["DAXRAT"] + self.header["AXRAT0"]
        self.x_image = np.arange(self.header["BOXSIZE"])
        self.y_image = np.arange(self.header["BOXSIZE"])

    def save_models(self, output_file):
        """128
        saves the convolved models into a fits file at the specified directory

        Parameters
        ----------
        output_file : str
            the path to the fits file to save the models
        """

        cube = self.convolved_models.swapaxes(-2, -3)
        cube = cube.reshape(self.original_shape)
        models_hdu = fits.ImageHDU(data=cube)
        header_hdu = fits.PrimaryHDU(header=self.header)
        hdul = fits.HDUList([header_hdu, models_hdu])
        hdul.writeto(output_file, overwrite=True)

    def convolve(self, psf_file, *args, **kwargs):
        """
        convolves the models with the PSF and stores them in the convolved_models attribute,
        is implemented in the child classes ModelsCPU and ModelsGPU depending on the acceleration method

        Parameters
        ----------
        psf_file : str
            the path to the file with the PSF in the format of PSFex output
        """

        pass

    def dre_fit(self, data, segment, noise, backend=numpy):
        """
        performs the fit with this steps:
            - masks the models, the data and the noise with the segment,
              all the following operations are only in the segment
            - compute the models flux and the object flux, scales the model to match the object flux
            - compute the residual between the model and the data divided by the variance for each pixel,
              the variance is considered: sqrt(scaled_model)^2 + noise
            - compute the chi-squared: sum the residuals and divide by the number of pixels

        Parameters
        ----------
        data : ndarray
            numpy/cupy array corresponding to a science image cut with the object at the center
        segment : ndarray
            numpy/cupy array corresponding to a segmentation image cut
        noise : ndarray
            numpy/cupy array corresponding to a background RMS image cut
        backend : module, optional
            module tu use as backend, should be numpy or cupy

        Returns
        -------
        ndarray
            numpy/cupy array with the chi-square for each model
        """

        # mask all elements, faster with index for large arrays
        mask_idx = backend.where(segment)
        models = self.convolved_models[..., mask_idx[0], mask_idx[1]]
        data = data[mask_idx[0], mask_idx[1]]
        noise = noise[mask_idx[0], mask_idx[1]]

        flux_models = backend.sum(models, axis=-1)
        flux_data = backend.nansum(data, axis=-1)
        scale = flux_data / flux_models
        scaled_models = scale[..., backend.newaxis] * models
        diff = data - scaled_models
        residual = (diff ** 2) / (scaled_models + noise ** 2)
        chi = backend.nanmean(residual, axis=-1)

        return chi

    def pond_rad_3d(self, chi_cube, log_r_min):
        """
        DEPRECATION WARNING: this method is probably going to be deprecated in the near future

        computes a weighted radius which weights are 1/chi_squared for each model, also computes a wighted variance
        respect the minimum and the weighted radius

        Parameters
        ---------
        chi_cube : ndarray
            numpy array  with the chi_squared resulting from the DRE fit, if using cupy array it must be converted
            to numpy array before using this method
        log_r_min : float
            value of log_r at the model which minimizes the chi_squared

        Returns
        -------
        log_r_chi : float
            log10 of the weighted radius
        log_r_var : float
            log10 of the variance respect to the radius of the optimal model
        log_r_chi_var : float
            log10 of the variance respect to the weighted radius
        """

        r_chi = np.sum((10 ** self.log_r) / chi_cube)
        r_chi = r_chi / np.sum(1. / chi_cube)
        log_r_chi = np.log10(r_chi)

        r_var = np.sum(((10 ** self.log_r - 10 ** log_r_min) ** 2) / chi_cube)
        r_var = r_var / np.sum(1. / chi_cube)
        log_r_var = np.log10(r_var)

        r_chi_var = np.sum(((10 ** self.log_r - r_chi) ** 2) / chi_cube)
        r_chi_var = r_chi_var / np.sum(1. / chi_cube)
        log_r_chi_var = np.log10(r_chi_var)
        return log_r_chi, log_r_var, log_r_chi_var

    def get_parameters(self, chi_cube):
        """
        find the model that minimizes the chi_square and computes the parameters for this model

        Parameters
        ----------
        chi_cube : ndarray
            numpy array  with the chi_squared resulting from the DRE fit, if using cupy array it must be converted
            to numpy array before using this method

        Returns
        -------
        parameters : dict
            dictionary with the parameters
        """
        parameters = dict()

        parameters['MODEL_IDX'] = np.unravel_index(np.nanargmin(chi_cube), chi_cube.shape)
        n, e, t, r = parameters['MODEL_IDX']
        parameters['CHI'] = np.nanmin(chi_cube)
        parameters['CHI_STD'] = np.std(chi_cube)
        parameters['LOGR_CHI'], parameters['LOGR_VAR'], parameters['LOGR_CHI_VAR'] = self.pond_rad_3d(chi_cube,
                                                                                                      self.log_r[r])

        parameters['LOGR_PARAB'], parameters['LOGR_STD'] = fit_parabola_1d(np.log(chi_cube),
                                                                           parameters['MODEL_IDX'],
                                                                           self.log_r)

        parameters.update({'INDEX': self.src_index[n], 'LOGR': self.log_r[r],
                           'AX_RATIO': self.ax_ratio[e], 'ANGLE': self.angle[t]})
        return parameters

    def make_mosaic(self, data, segment, model_index):
        """
        makes a image with the data, the segment and the model, all scaled to the data  flux

        Parameters
        ----------
        data : ndarray
            numpy array corresponding to a science image cut with the object at the center
        segment : ndarray
            numpy/cupy array corresponding to a segmentation image cut
        model_index : tuple
            index of the optimal model

        Returns
        -------
        ndarray
            numpy array with the mosaic image
        """

        model = self.convolved_models[model_index]
        flux_model = np.einsum("xy,xy", model, segment)
        flux_data = np.einsum("xy,xy", data, segment)
        scaled_model = (flux_data / flux_model) * model
        mosaic = np.zeros((4, len(self.x_image), len(self.y_image)))
        mosaic[0] = data
        mosaic[1] = segment * (flux_data / segment.sum())
        mosaic[2] = scaled_model
        mosaic[3] = data - scaled_model
        mosaic = mosaic.reshape(len(self.x_image) * 4, len(self.y_image)).T
        return mosaic
