
#THIS CODE IS AUTO-GENERATED by timeloop_tempate, DO NOT DIRECTLY EDIT UNLESS YOU KNOW WHAT YOU ARE DOING!

from numba import njit, void, prange
import numpy as np
import math
from numpy import float64, int32, int64

@njit(nogil=True, cache=True)
def ATTAINED_AGE(POL_YR,AGE_AT_ENTRY):
	return AGE_AT_ENTRY+POL_YR-1


@njit(nogil=True, parallel=True, cache=True)
def wrapped_ATTAINED_AGE(POL_YR,AGE_AT_ENTRY):
	arr_ATTAINED_AGE=np.zeros(POL_YR.shape,dtype=int32)
	for y in prange(POL_YR.shape[0]):
		for t in range(POL_YR.shape[1]):
			arr_ATTAINED_AGE[y,t]=ATTAINED_AGE(POL_YR[y,t],AGE_AT_ENTRY[y,0])
	return arr_ATTAINED_AGE