
#THIS CODE IS AUTO-GENERATED by simple_template, DO NOT DIRECTLY EDIT UNLESS YOU KNOW WHAT YOU ARE DOING!

from numba import njit, void, prange
import numpy as np
import math
from numpy import float64, int32, int64

@njit(nogil=True, cache=True)
def PREM_FREQ(PREM_MODE):
	if PREM_MODE==0:
		return 0
	elif PREM_MODE==1:
		return 1
	elif PREM_MODE in [2,3]:
		return 2
	elif PREM_MODE in [4,5,6]:
		return 12
	else:
		return -1 #Error value


@njit(nogil=True, parallel=True, cache=True)
def wrapped_PREM_FREQ(PREM_MODE):
	arr_PREM_FREQ=np.zeros((PREM_MODE.shape[0],1),dtype=int32)
	for y in prange(PREM_MODE.shape[0]):
		arr_PREM_FREQ[y,0]=PREM_FREQ(PREM_MODE[y,0])
	return arr_PREM_FREQ
