#!/bin/env python

from astropy.io import fits
import multiprocessing as mp
from threading import Thread
import ctypes
import numpy as np
from astropy.convolution import convolve_fft
from crunch import E_fit
from h5py import File
import argparse
import os
import sys
import time
import datetime


class ModelCPU:
    def __init__(self, models_file, n_cpu, chunk_size, out_compression=None):
        if out_compression is None:
            out_compression = dict()
        self.models = None
        self.load_models(models_file)
        self.chunk_size = chunk_size
        self.compression = out_compression

        self.n_cpu = n_cpu

        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        self.feed_thread = None
        self.output_thread = None
        self.processes = []

    def load_models(self, models_file):
        cube = fits.getdata(models_file)
        cube = cube.reshape((10, 13, 128, 21, 128))
        cube = cube.swapaxes(2, 3)
        self.models = cube

    def convolve(self, psf_file):
        print("Convolving...")
        start = time.time()
        with open(psf_file, 'r') as psf_h5f:
            psf = psf_h5f[:]
        for i in range(self.models.shape[0]):
            for j in range(self.models.shape[1]):
                for k in range(self.models.shape[2]):
                    self.models[i, j, k] = convolve_fft(self.models[i, j, k], psf)
        print(f"Done! ({time.time() - start:2.2f}s)")

    def to_shared_mem(self):
        shape = self.models.shape
        shared_array_base = mp.Array(ctypes.c_float, int(np.prod(shape)))
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array = shared_array.reshape(shape)
        shared_array[:] = self.models
        self.models = shared_array

    def cpu_worker(self, input_q, output_q):
        for name, data, segment, noise in iter(input_q.get, ('STOP', '', '', '')):
            chi = E_fit(self.models, data, segment, noise)
            output_q.put((name, chi))

    def feed_processes(self, names, input_file):
        for name in names:
            while self.input_queue.qsize() > self.chunk_size:
                time.sleep(0.5)
            with File(input_file, 'r') as input_h5f:
                data = input_h5f[name]
                self.input_queue.put((name, data['obj'][:], data['seg'][:], data['rms'][:]))
        for i in range(self.n_cpu):
            self.input_queue.put(('STOP', '', '', ''))

    def get_output(self, n_tasks, output_file, progress_status):
        for i in range(n_tasks):
            name, chi = self.output_queue.get()
            with File(output_file, 'a') as output_h5f:
                output_h5f.create_dataset(f'{name}', data=chi,
                                          dtype='float32', **self.compression)
            progress(i+1, n_tasks, progress_status)

    def start_processes(self, names, input_file, output_file, progress_status):
        self.feed_thread = Thread(target=self.feed_processes, args=(names, input_file))
        self.output_thread = Thread(target=self.get_output, args=(len(names), output_file, progress_status))
        self.feed_thread.start()
        self.output_thread.start()
        self.processes = []
        for i in range(self.n_cpu):
            p = mp.Process(target=self.cpu_worker, args=(self.input_queue, self.output_queue))
            self.processes.append(p)
            p.start()

    def stop_processes(self):
        for p in self.processes:
            p.join()
        self.feed_thread.join()
        self.output_thread.join()

    def fit_data(self, input_file, output_file, progress_status):
        start = time.time()
        with File(input_file, 'r') as input_h5f:
            names = list(input_h5f.keys())
        print(f"{progress_status}: {input_file}\t{len(names)} objects")
        self.start_processes(names, input_file, output_file, progress_status)
        self.stop_processes()
        time_delta = time.time() - start
        mean_time = time_delta / len(names)
        print(f"\n{progress_status}: finished in {datetime.timedelta(seconds=time_delta)}s ({mean_time:1.3f} s/obj)")


def feed_model(model, input_dir, output_dir):
    # enviamos el modelo a memoria compartida entre procesos
    model.to_shared_mem()
    # generamos una lista de archivos de entrada
    _, _, files = next(os.walk(input_dir))
    if not os.path.exists(args.output):
        os.mkdir(output_dir)
    for i, filename in enumerate(files):
        input_file = f"{input_dir}/{filename}"
        name, _ = os.path.splitext(os.path.split(filename)[1])
        output_file = f"{output_dir}/{name}_chi.h5"
        if os.path.isfile(output_file):
            os.remove(output_file)
        # realizamos un fit en paralelo
        model.fit_data(input_file, output_file, progress_status=f"({i + 1}/{len(files)})")


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write(f'{status}: [{bar}] {percents}%\r')
    sys.stdout.flush()


if __name__ == "__main__":
    def parse_arguments():
        parser = argparse.ArgumentParser()

        parser.add_argument("model", help="models cube file", type=str)
        parser.add_argument("--psf", help="psf file", type=str, default=None)
        parser.add_argument("-i", "--input", help="root directory to cuts of data (def: Cuts)",
                            default="Cuts", type=str)
        parser.add_argument("-o", "--output", help="output directory (def: Chi)", type=str, default="Chi")
        parser.add_argument("--cpu", help="Number of cpu's to use", type=int, default=1)
        parser.add_argument("--chunk", type=int, default=100)
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

    model_ = ModelCPU(args.model, args.cpu, args.chunk, args.compression)
    if args.psf:
        model_.convolve("psf")
    feed_model(model_, args.input, args.output)
