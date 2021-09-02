import cupy as cp
import cupy
from h5py import File
from tqdm import tqdm
import matplotlib.pyplot as plt
from DRE.core.models import ModelsCube
from .convolve_gpu import gpu_fftconvolve
from DRE.misc.read_psf import get_psf
import os


class ModelGPU(ModelsCube):
    def __init__(self, models_file=None, out_compression='none'):
        super().__init__(models_file, out_compression)
        self.convolved = False

        self.to_gpu()

    def to_gpu(self):
        self.models = cp.array(self.models)

    def convolve(self, psf_file, to_cpu=False, *args, **kwargs):
        psf = cp.array(get_psf(psf_file))
        self.convolved_models = cp.zeros(self.models.shape)
        for i in range(self.convolved_models.shape[0]):
            for j in range(self.convolved_models.shape[1]):
                self.convolved_models[i, j] = gpu_fftconvolve(self.models[i, j], psf[cp.newaxis, cp.newaxis],
                                                              axes=(-2, -1))
        if to_cpu:
            self.convolved_models = cp.asnumpy(self.convolved_models)

    def fit_file(self, input_file, output_file, psf, progress_status=''):
        self.convolve(psf)
        with File(input_file, 'r') as input_h5f:
            names = list(input_h5f.keys())
        for name in tqdm(names, desc=progress_status, mininterval=0.5):
            with File(input_file, 'r') as input_h5f:
                data = input_h5f[name]
                chi = self.dre_fit(cp.array(data['obj'][:]),
                                   cp.array(data['seg'][:]),
                                   cp.array(data['rms'][:]),
                                   backend=cupy)
            if not cp.isnan(chi).all():
                with File(output_file, 'a') as output_h5f:
                    output_h5f.create_dataset(f'{name}', data=cp.asnumpy(chi),
                                              dtype='float32', **self.compression)

    def fit_dir(self, input_dir='Cuts', output_dir='Chi', psf_dir='PSF'):
        # list with input files in input_dir
        files = os.listdir(input_dir)
        os.makedirs(output_dir, exist_ok=True)
        for i, filename in enumerate(sorted(files)):
            input_file = os.path.join(input_dir, filename)
            name = os.path.basename(filename).replace('_cuts.h5', '')
            output_file = os.path.join(output_dir, f"{name}_chi.h5")
            psf = os.path.join(psf_dir, f"{name}.psf")
            if os.path.isfile(output_file):
                os.remove(output_file)
            # fit all cuts in each file
            self.fit_file(input_file, output_file, psf, progress_status=f"({i + 1}/{len(files)})")

    def visualize_model(self, ax_ratio_idx, src_index_idx=-1,
                        psf=None, figsize=(20, 15), vmin=0, vmax=100, cmap='gray'):
        plt.figure(figsize=figsize)
        if psf is not None:
            self.convolve(psf)
        else:
            self.convolved_models = self.models.copy()
        models_slice = self.convolved_models[src_index_idx, ax_ratio_idx].swapaxes(-2, -3)
        models_slice = models_slice.reshape(self.original_shape[-2:])
        plt.suptitle(f'a/b = {self.ax_ratio[ax_ratio_idx]:.1f}, n = {self.src_index[src_index_idx]:.1f}',
                     fontsize=20, y=0.85)
        plt.imshow(cp.asnumpy(models_slice), vmin=vmin, vmax=vmax, cmap=cmap)
        plt.axis('off')
        plt.show()
