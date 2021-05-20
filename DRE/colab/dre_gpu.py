import cupy as cp
from opt_einsum import contract_expression
from h5py import File
from tqdm import tqdm
import matplotlib.pyplot as plt
from DRE.core.ModelsIO import ModelsCube
from .convolve_gpu import gpu_fftconvolve


class ModelGPU(ModelsCube):
    def __init__(self, models_file=None, out_compression='none'):
        super().__init__(models_file, out_compression)
        self.convolved = False

        self.to_gpu()

        # optimized einstein sum contractions for GPU
        self.contract_cube_x_image = contract_expression("ijkxy,xy->ijk", (10, 13, 21, 128, 128), (128, 128))
        self.contract_image_x_image = contract_expression("xy,xy->", (128, 128), (128, 128))
        self.contract_scale_x_model = contract_expression("ijk,ijkxy->ijkxy", (10, 13, 21), (10, 13, 21, 128, 128))

    def to_gpu(self):
        self.models = cp.array(self.models)

    def convolve(self, psf_file, progress_status=''):
        with File(psf_file, 'r') as psf_h5f:
            psf = cp.array(psf_h5f['psf'][:])
        self.convolved_models = cp.zeros(self.models.shape)
        for i in range(self.convolved_models.shape[0]):
            self.convolved_models[i] = gpu_fftconvolve(self.models[i], psf[cp.newaxis, cp.newaxis],
                                                       axes=(-2, -1))

    def dre_fit(self, data, segment, noise):
        # enviar a la GPU
        data = cp.array(data)
        segment = cp.array(segment)
        noise = cp.array(noise)

        flux_models = self.contract_cube_x_image(self.convolved_models, segment, backend='cupy')
        flux_data = self.contract_image_x_image(data, segment, backend='cupy')
        scale = flux_data / flux_models
        scaled_models = self.contract_scale_x_model(scale, self.convolved_models, backend='cupy')
        diff = data - scaled_models
        residual = (diff ** 2) / (scaled_models + noise ** 2)
        chi = self.contract_cube_x_image(residual, segment, backend='cupy')

        area = segment.sum()
        chi = chi / area
        return chi

    def fit_file(self, input_file, output_file, psf, progress_status=''):
        self.convolve(psf)
        with File(input_file, 'r') as input_h5f:
            names = list(input_h5f.keys())
        for name in tqdm(names, desc=progress_status, mininterval=0.5):
            with File(input_file, 'r') as input_h5f:
                data = input_h5f[name]
                chi = self.dre_fit(data['obj'][:], data['seg'][:], data['rms'][:])
            if not cp.isnan(chi).all():
                with File(output_file, 'a') as output_h5f:
                    output_h5f.create_dataset(f'{name}', data=cp.asnumpy(chi),
                                              dtype='float32', **self.compression)

    def visualize_model(self, ratio_idx, psf=None, figsize=(20, 20), vmin=0, vmax=100, cmap='gray'):
        plt.figure(figsize=figsize)
        if psf is not None:
            self.convolve(psf)
        else:
            self.convolved_models = self.models.copy()
        models_slice = self.convolved_models.swapaxes(2, 3)[ratio_idx]
        models_slice = models_slice.reshape(self.models.shape[1] * 128, self.models.shape[2] * 128)
        plt.imshow(cp.asnumpy(models_slice), vmin=vmin, vmax=vmax, cmap=cmap)
        plt.axis('off')
        plt.show()
