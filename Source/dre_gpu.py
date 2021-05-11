#!/bin/env python

from astropy.io import fits
import cupy as cp
from cupyx.scipy.ndimage import convolve
from opt_einsum import contract_expression
from h5py import File
import argparse
import os
import sys
import time
import datetime
from tqdm import tqdm


class ModelGPU:
    def __init__(self, models_file, out_compression=None, convolved=False):
        if out_compression is None:
            out_compression = dict()
        self.models = None
        self.load_models(models_file)

        self.convolved = convolved
        self.compression = out_compression

        self.cube_x_image = contract_expression("ijkxy,xy->ijk", (10, 13, 21, 128, 128), (128, 128))
        self.image_x_image = contract_expression("xy,xy->", (128, 128), (128, 128))
        self.scale_model = contract_expression("ijk,ijkxy->ijkxy", (10, 13, 21), (10, 13, 21, 128, 128))

    def load_models(self, models_file):
        cube = fits.getdata(models_file)
        cube = cube.reshape((10, 13, 128, 21, 128))
        cube = cube.swapaxes(2, 3)
        # enviar a la GPU
        cube = cp.array(cube)
        self.models = cube

    def convolve(self, psf_file):
        if not self.convolved:
            print("Convolving...")
            start = time.time()
            with open(psf_file, 'r') as psf_h5f:
                psf = cp.array(psf_h5f[:])
            for i in range(self.models.shape[0]):
                for j in range(self.models.shape[1]):
                    for k in range(self.models.shape[2]):
                        self.models[i, j, k] = convolve(self.models[i, j, k], psf, mode='nearest')
            print(f"Done! ({time.time() - start:2.2f}s)")
            self.convolved = True
        else:
            print("This model has already been convolved, you should create a new model")

    def E_fit_gpu(self, data, seg, noise):
        # enviar a la GPU
        data = cp.array(data)
        seg = cp.array(seg)
        noise = cp.array(noise)

        flux_models = self.cube_x_image(self.models, seg, backend='cupy')
        flux_data = self.image_x_image(data, seg, backend='cupy')
        X = flux_data / flux_models
        scaled_models = self.scale_model(X, self.models, backend='cupy')
        diff = data - scaled_models
        residual = (diff ** 2) / (cp.sqrt((scaled_models + noise ** 2)))
        chi = self.cube_x_image(residual, seg, backend='cupy')

        area = seg.sum()
        chi = chi / area
        return chi

    def fit_data(self, input_file, output_file, progress_status=''):
        start = time.time()
        with File(input_file, 'r') as input_h5f:
            names = list(input_h5f.keys())
        with tqdm(names) as pbar:
            pbar.set_description(progress_status)
            for name in names:
                with File(input_file, 'r') as input_h5f:
                    data = input_h5f[name]
                    chi = self.E_fit_gpu(data['obj'][:], data['seg'][:], data['rms'][:])
                with File(output_file, 'a') as output_h5f:
                    output_h5f.create_dataset(f'{name}', data=cp.asnumpy(chi),
                                              dtype='float32', **self.compression)
                pbar.update()
        time_delta = time.time() - start
        mean_time = time_delta / len(names)
        print(f"\n{progress_status}: finished in {datetime.timedelta(seconds=time_delta)}s ({mean_time:1.3f} s/obj)")


def feed_model(model, input_dir, output_dir):
    # generamos una lista de archivos de entrada
    _, _, files = next(os.walk(input_dir))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for i, filename in enumerate(files):
        input_file = f"{input_dir}/{filename}"
        name, _ = os.path.splitext(os.path.split(filename)[1])
        output_file = f"{output_dir}/{name}_chi.h5"
        if os.path.isfile(output_file):
            os.remove(output_file)
        # realizamos un fit en GPU
        model.fit_data(input_file, output_file, progress_status=f"({i + 1}/{len(files)})")


# chequamos que existe una GPU
try:
    cp.cuda.runtime.getDevice()
except cp.cuda.runtime.CUDARuntimeError:
    print("cudaErrorNoDevice: no CUDA-capable device is detected")
    sys.exit(1)


if __name__ == "__main__":
    def parse_arguments():
        parser = argparse.ArgumentParser()

        parser.add_argument("model", help="models cube file", type=str)
        parser.add_argument("--psf", help="psf file", type=str, default=None)
        parser.add_argument("-i", "--input", help="root directory to cuts of data (def: Cuts)",
                            default="Cuts", type=str)
        parser.add_argument("-o", "--output", help="output directory (def: Chi)", type=str, default="Chi")
        parser.add_argument("--compression", help="compresion level for the h5 output file,"
                                                  "lower is faster (def: medium)",
                            choices=["none", "low", "medium", "high"], default="medium")

        args_ = parser.parse_args()

        compression_types = {"none": dict(),
                             "low": {"compression": "lzf"},
                             "medium": {"compression": "gzip", "compression_opts": 4},
                             "high": {"compression": "gzip", "compression_opts": 9}}
        args_.compression = compression_types[args_.compression]

        return args_

    args = parse_arguments()

    model_ = ModelGPU(args.model, args.compression)
    if args.psf:
        model_.convolve("psf")
    feed_model(model_, args.input, args.output)
