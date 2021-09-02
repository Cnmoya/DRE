import cupy as cp
from scipy.fft._helper import _init_nd_shape_and_axes
from scipy.fft import next_fast_len


def _centered(arr, newshape):
    # Return the center newshape portion of the array.
    newshape = cp.asarray(newshape)
    currshape = cp.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def gpu_fftconvolve(in1, in2, axes=None):
    _, axes = _init_nd_shape_and_axes(in1, shape=None, axes=axes)

    s1 = in1.shape
    s2 = in2.shape

    shape = [max((s1[i], s2[i])) if i not in axes else s1[i] + s2[i] - 1
             for i in range(in1.ndim)]

    fshape = [next_fast_len(shape[a], True) for a in axes]

    sp1 = cp.fft.rfft2(in1, fshape, axes=axes)
    sp2 = cp.fft.rfft2(in2, fshape, axes=axes)

    ret = cp.fft.irfft2(sp1 * sp2, fshape, axes=axes)

    fslice = tuple([slice(sz) for sz in shape])
    ret = ret[fslice]

    return _centered(ret, s1).copy()
