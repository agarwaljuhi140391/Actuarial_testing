
#THIS CODE IS AUTO-GENERATED by timeloop_tempate, DO NOT DIRECTLY EDIT UNLESS YOU KNOW WHAT YOU ARE DOING!

from numba import njit, void, prange
import numpy as np
import math
from numpy import float64, int32, int64

@njit(nogil=True, cache=True)
def PPIA_INC_IF(T,DURATIONIF_M, PPIA_INCOME_RATE,SUM_ASSURED2,NO_POLS_IFSM):
	if T< DURATIONIF_M:
		return 0
	else:
		return PPIA_INCOME_RATE*SUM_ASSURED2*NO_POLS_IFSM


@njit(nogil=True, parallel=True, cache=True)
def wrapped_PPIA_INC_IF(T,DURATIONIF_M,PPIA_INCOME_RATE,SUM_ASSURED2,NO_POLS_IFSM):
	arr_PPIA_INC_IF=np.zeros(T.shape,dtype=float64)
	for y in prange(T.shape[0]):
		for t in range(T.shape[1]):
			arr_PPIA_INC_IF[y,t]=PPIA_INC_IF(T[y,t],DURATIONIF_M[y,0],PPIA_INCOME_RATE[y,t],SUM_ASSURED2[y,t],NO_POLS_IFSM[y,t])
	return arr_PPIA_INC_IF
