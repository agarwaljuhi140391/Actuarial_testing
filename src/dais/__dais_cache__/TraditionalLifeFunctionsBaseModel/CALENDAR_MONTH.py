
#THIS CODE IS AUTO-GENERATED by timeloop_tempate, DO NOT DIRECTLY EDIT UNLESS YOU KNOW WHAT YOU ARE DOING!

from numba import njit, void, prange
import numpy as np
import math
from numpy import float64, int32, int64

@njit(nogil=True, cache=True)
def CALENDAR_MONTH(ENTRY_MONTH,T):
	return ((ENTRY_MONTH+T+10)%12)+1


@njit(nogil=True, parallel=True, cache=True)
def wrapped_CALENDAR_MONTH(ENTRY_MONTH,T):
	arr_CALENDAR_MONTH=np.zeros(T.shape,dtype=int32)
	for y in prange(T.shape[0]):
		for t in range(T.shape[1]):
			arr_CALENDAR_MONTH[y,t]=CALENDAR_MONTH(ENTRY_MONTH[y,0],T[y,t])
	return arr_CALENDAR_MONTH