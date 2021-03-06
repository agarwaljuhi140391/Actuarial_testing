
#THIS CODE IS AUTO-GENERATED by simple_template, DO NOT DIRECTLY EDIT UNLESS YOU KNOW WHAT YOU ARE DOING!

from numba import njit, void, prange
import numpy as np
import math
from numpy import float64, int32, int64

@njit(nogil=True, cache=True)
def COMM_S_RATE(COMM_LIMIT_TYPE_derived,COMM_S_RATE_PC_1):
	if COMM_LIMIT_TYPE_derived==999999:
		return 0
	else:
		return COMM_S_RATE_PC_1/100


@njit(nogil=True, parallel=True, cache=True)
def wrapped_COMM_S_RATE(COMM_LIMIT_TYPE_derived,COMM_S_RATE_PC_1):
	arr_COMM_S_RATE=np.zeros((COMM_LIMIT_TYPE_derived.shape[0],1),dtype=float64)
	for y in prange(COMM_LIMIT_TYPE_derived.shape[0]):
		arr_COMM_S_RATE[y,0]=COMM_S_RATE(COMM_LIMIT_TYPE_derived[y,0],COMM_S_RATE_PC_1[y,0])
	return arr_COMM_S_RATE
