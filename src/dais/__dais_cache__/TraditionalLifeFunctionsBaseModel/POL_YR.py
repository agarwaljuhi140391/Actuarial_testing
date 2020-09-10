
#THIS CODE IS AUTO-GENERATED by timeloop_tempate, DO NOT DIRECTLY EDIT UNLESS YOU KNOW WHAT YOU ARE DOING!

from numba import njit, void, prange
import numpy as np
import math
from numpy import float64, int32, int64

@njit(nogil=True, cache=True)
def POL_YR(T):
	return (T+11)//12


@njit(nogil=True, parallel=True, cache=True)
def wrapped_POL_YR(T):
	arr_POL_YR=np.zeros(T.shape,dtype=int32)
	for y in prange(T.shape[0]):
		for t in range(T.shape[1]):
			arr_POL_YR[y,t]=POL_YR(T[y,t])
	return arr_POL_YR