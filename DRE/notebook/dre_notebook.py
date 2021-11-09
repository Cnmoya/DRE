from h5py import File
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from DRE.core.models import ModelsCube
import numpy as np
import os
from matplotlib.widgets import Slider


class ModelNB(ModelsCube):
    def __init__(self, models_file=None, out_compression='none'):
        super().__init__(models_file, out_compression)
        self.convolved = False
        self.convolved_models = self.models.copy()

    def fit_file(self, input_file, output_file, psf=None, convolve=True, progress_status=''):
        if os.path.isfile(output_file):
            os.remove(output_file)
        if convolve:
            self.convolve(psf)
        with File(input_file, 'r') as input_h5f:
            names = list(input_h5f.keys())
        for name in tqdm(names, desc=progress_status, mininterval=0.5):
            with File(input_file, 'r') as input_h5f:
                data = input_h5f[name]
                chi = self.dre_fit(data['obj'][:],
                                   data['seg'][:],
                                   data['rms'][:])
            if not np.isnan(chi).all():
                with File(output_file, 'a') as output_h5f:
                    output_h5f.create_dataset(f'{name}', data=chi,
                                              dtype='float32', **self.compression)

    def fit_dir(self, input_dir='Cuts', chi_dir='Chi', psf_dir='PSF', convolve=True):
        # list with input files in input_dir
        files = os.listdir(input_dir)
        os.makedirs(chi_dir, exist_ok=True)
        for i, filename in enumerate(sorted(files)):
            input_file = os.path.join(input_dir, filename)
            name = os.path.basename(filename).replace('_cuts.h5', '')
            output_file = os.path.join(chi_dir, f"{name}_chi.h5")
            psf = os.path.join(psf_dir, f"{name}.psf")
            # fit all cuts in each file
            self.fit_file(input_file, output_file, psf, convolve, progress_status=f"({i + 1}/{len(files)})")

    def visualize_model(self, src_index_idx=0, ax_ratio_idx=0, figsize=(10, 7), vmin=0, vmax=100, cmap='gray'):
        fig, ax = plt.subplots(figsize=figsize)

        def make_slice(index_i, ratio_i):
            models_slice = self.convolved_models[index_i, ratio_i].swapaxes(-2, -3)
            models_slice = models_slice.reshape(self.original_shape[-2:])
            return models_slice

        fig.suptitle(f'a/b = {self.ax_ratio[ax_ratio_idx]:.1f}, n = {self.src_index[src_index_idx]:.1f}',
                     fontsize=20, y=0.85)
        ax.imshow(make_slice(src_index_idx, ax_ratio_idx), vmin=vmin, vmax=vmax, cmap=cmap)
        plt.axis('off')

        plt.subplots_adjust(left=0, top=0.8)

        index_slider = Slider(
            ax=plt.axes([0.18, 0., 0.5, 0.03]),
            label='n',
            valmin=0,
            valmax=self.shape[0] - 1,
            valstep=1,
            valinit=src_index_idx)
        ratio_slider = Slider(
            ax=plt.axes([0.18, 0.05, 0.5, 0.03]),
            label='a/b',
            valmin=0,
            valmax=self.shape[1] - 1,
            valstep=1,
            valinit=ax_ratio_idx, )

        def update(val):
            ax.imshow(make_slice(index_slider.val, ratio_slider.val), vmin=vmin, vmax=vmax, cmap=cmap)
            plt.suptitle(f'a/b = {self.ax_ratio[ratio_slider.val]:.1f}, n = {self.src_index[index_slider.val]:.1f}',
                         fontsize=20, y=0.85)
            fig.canvas.draw_idle()

        index_slider.on_changed(update)
        ratio_slider.on_changed(update)

        plt.show()

