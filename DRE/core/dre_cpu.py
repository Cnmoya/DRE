import multiprocessing as mp
from threading import Thread
import ctypes
import numpy as np
from astropy.convolution import convolve_fft
from h5py import File
import time
import datetime
from DRE.core.ModelsIO import ModelsCube
from DRE.misc.progress_bar import progress


class ModelCPU(ModelsCube):
    def __init__(self, models_file=None, n_cpu=1, chunk_size=100, out_compression='none'):
        super().__init__(models_file, out_compression)

        self.chunk_size = chunk_size
        self.n_cpu = n_cpu

        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        self.feed_thread = None
        self.output_thread = None
        self.processes = []

    def convolve(self, psf_file):
        print("Convolving...")
        start = time.time()
        with File(psf_file, 'r') as psf_h5f:
            psf = psf_h5f['psf'][:]
        for e in range(self.models.shape[0]):
            for t in range(self.models.shape[1]):
                for r in range(self.models.shape[2]):
                    self.models[e, t, r] = convolve_fft(self.models[e, t, r], psf)
        print(f"Convolved! ({time.time() - start:2.2f}s)")

    def dre_fit(self, data, segment, noise):
        flux_models = np.einsum("ijkxy,xy->ijk", self.models, segment)
        flux_data = np.einsum("xy,xy", data, segment)
        scale = flux_data / flux_models
        scaled_models = scale[:, :, :, np.newaxis, np.newaxis] * self.models
        resta = data - scaled_models
        residuo = (resta ** 2) / (scaled_models + noise ** 2)
        chi = np.einsum("ijkxy,xy->ijk", residuo, segment)

        area = segment.sum()
        chi = chi / area
        return chi

    def to_shared_mem(self):
        shape = self.models.shape
        shared_array_base = mp.Array(ctypes.c_float, int(np.prod(shape)))
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array = shared_array.reshape(shape)
        shared_array[:] = self.models
        self.models = shared_array

    def cpu_worker(self, input_q, output_q):
        for name, data, segment, noise in iter(input_q.get, ('STOP', '', '', '')):
            chi = self.dre_fit(data, segment, noise)
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

    def fit_file(self, input_file, output_file, progress_status=''):
        start = time.time()
        with File(input_file, 'r') as input_h5f:
            names = list(input_h5f.keys())
        print(f"{progress_status}: {input_file}\t{len(names)} objects")
        self.start_processes(names, input_file, output_file, progress_status)
        self.stop_processes()
        time_delta = time.time() - start
        mean_time = time_delta / len(names)
        print(f"\n{progress_status}: finished in {datetime.timedelta(seconds=time_delta)}s ({mean_time:1.3f} s/obj)")
