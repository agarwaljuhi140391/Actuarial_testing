
#THIS CODE IS AUTO-GENERATED by simple_template, DO NOT DIRECTLY EDIT UNLESS YOU KNOW WHAT YOU ARE DOING!

from numba import njit, void, prange
import numpy as np
import math
from numpy import float64, int32, int64

@njit(nogil=True, cache=True)
def SPIKE_LAPSE_PERIOD2(SPIKE_RATE_IND,SPIKE_LAPSE_PERIOD):
	if (SPIKE_RATE_IND==0):
		return 0
	else:
		return SPIKE_LAPSE_PERIOD


@njit(nogil=True, parallel=True, cache=True)
def wrapped_SPIKE_LAPSE_PERIOD2(SPIKE_RATE_IND,SPIKE_LAPSE_PERIOD):
	arr_SPIKE_LAPSE_PERIOD2=np.zeros((SPIKE_RATE_IND.shape[0],1),dtype=int32)
	for y in prange(SPIKE_RATE_IND.shape[0]):
		arr_SPIKE_LAPSE_PERIOD2[y,0]=SPIKE_LAPSE_PERIOD2(SPIKE_RATE_IND[y,0],SPIKE_LAPSE_PERIOD[y,0])
	return arr_SPIKE_LAPSE_PERIOD2
