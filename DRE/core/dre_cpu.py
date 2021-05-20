import multiprocessing as mp
import os
from threading import Thread
import ctypes
import numpy as np
from scipy.signal import fftconvolve
from functools import partial
from h5py import File
import time
import datetime
from DRE.core.ModelsIO import ModelsCube
from DRE.core.results import Summary
from DRE.misc.progress_bar import progress
from astropy.io import fits


class ModelCPU(ModelsCube):
    def __init__(self, models_file=None, n_cpu=1, chunk_size=100, out_compression='none', save_mosaics=False):
        super().__init__(models_file, out_compression)

        self.chunk_size = chunk_size
        self.n_cpu = n_cpu

        self.conv_pool = mp.Pool(self.n_cpu)
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        self.feed_thread = None
        self.output_thread = None
        self.processes = []

        # send all attributes to shared memory
        self.models = self.to_shared_mem(self.models)
        self.log_r = self.to_shared_mem(self.log_r)
        self.ax_ratio = self.to_shared_mem(self.ax_ratio)
        self.angle = self.to_shared_mem(self.angle)
        self.save_mosaics = mp.Value(ctypes.c_bool, save_mosaics)

    @staticmethod
    def to_shared_mem(array):
        shape = array.shape
        shared_array_base = mp.Array(ctypes.c_float, int(np.prod(shape)))
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array = shared_array.reshape(shape)
        shared_array[:] = array
        return shared_array

    def convolve(self, psf_file, progress_status=''):
        print(f"{progress_status}: Convolving...")
        start = time.time()
        with File(psf_file, 'r') as psf_h5f:
            psf = psf_h5f['psf'][:]
        psf = psf[np.newaxis, np.newaxis, :]
        convolve = partial(fftconvolve, in2=psf, mode='same', axes=(-2, -1))
        convolved = self.conv_pool.map(convolve, self.models)
        self.convolved_models = self.to_shared_mem(np.array(list(convolved)))
        print(f"{progress_status}: Convolved! ({time.time() - start:2.2f}s)")

    def dre_fit(self, data, segment, noise):
        flux_models = np.einsum("ijkxy,xy->ijk", self.convolved_models, segment)
        flux_data = np.einsum("xy,xy", data, segment)
        scale = flux_data / flux_models
        scaled_models = scale[:, :, :, np.newaxis, np.newaxis] * self.convolved_models
        diff = data - scaled_models
        residual = (diff ** 2) / (scaled_models + noise ** 2)
        chi = np.einsum("ijkxy,xy->ijk", residual, segment)

        area = segment.sum()
        chi = chi / area
        return chi

    def cpu_worker(self, input_q, output_q):
        for name, data, segment, noise in iter(input_q.get, ('STOP', '', '', '')):
            chi_cube = self.dre_fit(data, segment, noise)
            parameters = None
            mosaic = None
            success = not np.isnan(chi_cube).all()
            if success:
                parameters = self.get_parameters(chi_cube)
                ext, numb = name.split('_')
                parameters['EXT_NUMBER'] = int(ext)
                parameters['NUMBER'] = int(numb)
                if self.save_mosaics:
                    model_idx = (parameters['E_IDX'], parameters['T_IDX'], parameters['R_IDX'])
                    mosaic = self.make_mosaic(data, segment, model_idx)
            output_q.put((name, success, chi_cube, parameters, mosaic))

    def feed_processes(self, names, input_file):
        for name in names:
            while self.input_queue.qsize() > self.chunk_size:
                time.sleep(0.5)
            with File(input_file, 'r') as input_h5f:
                data = input_h5f[name]
                # subscribes input to workers queue
                self.input_queue.put((name, data['obj'][:], data['seg'][:], data['rms'][:]))
        for i in range(self.n_cpu):
            self.input_queue.put(('STOP', '', '', ''))

    def get_output(self, input_name, n_tasks, output_file, table, progress_status):
        for i in range(n_tasks):
            name, success, chi_cube, params, mosaic = self.output_queue.get()
            if success:
                table.append(params)
                with File(output_file, 'a') as output_h5f:
                    output_h5f.create_dataset(f'{name}', data=chi_cube,
                                              dtype='float32', **self.compression)
                if self.save_mosaics:
                    os.makedirs(f"Mosaics/{input_name}", exist_ok=True)
                    mosaic_fits = fits.ImageHDU(data=mosaic)
                    mosaic_fits.writeto(f"Mosaics/{input_name}/{input_name}_{name}_mosaic.fits", overwrite=True)
            progress(i+1, n_tasks, progress_status)

    def start_processes(self, input_name, names, input_file, output_file, table, progress_status):
        self.feed_thread = Thread(target=self.feed_processes, args=(names, input_file))
        self.output_thread = Thread(target=self.get_output, args=(input_name, len(names), output_file,
                                                                  table, progress_status))
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

    def fit_file(self, input_name, input_file, output_file, psf, progress_status=''):
        start = time.time()
        with File(input_file, 'r') as input_h5f:
            names = list(input_h5f.keys())
        print(f"{progress_status}: {input_file}\t{len(names)} objects")
        # table with summary
        table = Summary(input_name)
        # convolve with the psf
        self.convolve(psf, progress_status)
        # fit in parallel
        self.start_processes(input_name, names, input_file, output_file, table, progress_status)
        self.stop_processes()
        # save summary
        table.save()
        time_delta = time.time() - start
        obj_s = len(names) / time_delta
        print(f"\n{progress_status}: finished in {datetime.timedelta(seconds=time_delta)}s ({obj_s:1.3f} obj/s)")
