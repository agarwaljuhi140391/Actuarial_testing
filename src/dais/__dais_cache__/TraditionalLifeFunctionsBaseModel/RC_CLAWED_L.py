
#THIS CODE IS AUTO-GENERATED by timeloop_tempate, DO NOT DIRECTLY EDIT UNLESS YOU KNOW WHAT YOU ARE DOING!

from numba import njit, void, prange
import numpy as np
import math
from numpy import float64, int32, int64

@njit(nogil=True, cache=True)
def RC_CLAWED_L(T,DURATIONIF_M,REN_POL_TERM_Y,PARTIAL_RENEW_COMM_EARNED,COMM_RECOV_PC,COMM_CB_ADJ,NO_LAPSE):
	if T<DURATIONIF_M or T>REN_POL_TERM_Y*12:
		return 0
	else:
		return PARTIAL_RENEW_COMM_EARNED*COMM_RECOV_PC*COMM_CB_ADJ*NO_LAPSE


@njit(nogil=True, parallel=True, cache=True)
def wrapped_RC_CLAWED_L(T,DURATIONIF_M,REN_POL_TERM_Y,PARTIAL_RENEW_COMM_EARNED,COMM_RECOV_PC,COMM_CB_ADJ,NO_LAPSE):
	arr_RC_CLAWED_L=np.zeros(T.shape,dtype=float64)
	for y in prange(T.shape[0]):
		for t in range(T.shape[1]):
			arr_RC_CLAWED_L[y,t]=RC_CLAWED_L(T[y,t],DURATIONIF_M[y,0],REN_POL_TERM_Y[y,0],PARTIAL_RENEW_COMM_EARNED[y,t],COMM_RECOV_PC[y,t],COMM_CB_ADJ[y,t],NO_LAPSE[y,t])
	return arr_RC_CLAWED_L