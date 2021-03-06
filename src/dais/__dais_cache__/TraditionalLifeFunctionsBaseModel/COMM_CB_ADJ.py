
#THIS CODE IS AUTO-GENERATED by timeloop_tempate, DO NOT DIRECTLY EDIT UNLESS YOU KNOW WHAT YOU ARE DOING!

from numba import njit, void, prange
import numpy as np
import math
from numpy import float64, int32, int64

@njit(nogil=True, cache=True)
def COMM_CB_ADJ(POL_YR,NBYM,COMM_CB_ST_YM,COMM_CB_ADJ_RATE):
	if POL_YR>1	and NBYM>=COMM_CB_ST_YM:
		return COMM_CB_ADJ_RATE
	else:
		return 1


@njit(nogil=True, parallel=True, cache=True)
def wrapped_COMM_CB_ADJ(POL_YR,NBYM,COMM_CB_ST_YM,COMM_CB_ADJ_RATE):
	arr_COMM_CB_ADJ=np.zeros(POL_YR.shape,dtype=float64)
	for y in prange(POL_YR.shape[0]):
		for t in range(POL_YR.shape[1]):
			arr_COMM_CB_ADJ[y,t]=COMM_CB_ADJ(POL_YR[y,t],NBYM[y,0],COMM_CB_ST_YM[0,0],COMM_CB_ADJ_RATE[0,0])
	return arr_COMM_CB_ADJ
