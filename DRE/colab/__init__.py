import cupy
import sys

# check if GPU exists
try:
    cupy.cuda.runtime.getDevice()
except cupy.cuda.runtime.CUDARuntimeError:
    print("cudaErrorNoDevice: no CUDA-capable device is detected")
    sys.exit(1)