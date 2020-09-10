
#THIS CODE IS AUTO-GENERATED by timeloop_tempate, DO NOT DIRECTLY EDIT UNLESS YOU KNOW WHAT YOU ARE DOING!

from numba import njit, void, prange
import numpy as np
import math
from numpy import float64, int32, int64

@njit(nogil=True, cache=True)
def COMM_CLAWED(T,DURATIONIF_M,REN_POL_TERM_Y,IC_CLAWED_L,RC_CLAWED_L):
	if (T<=DURATIONIF_M) or (T>REN_POL_TERM_Y*12):
		return 0
	else:
		return IC_CLAWED_L+RC_CLAWED_L


@njit(nogil=True, parallel=True, cache=True)
def wrapped_COMM_CLAWED(T,DURATIONIF_M,REN_POL_TERM_Y,IC_CLAWED_L,RC_CLAWED_L):
	arr_COMM_CLAWED=np.zeros(T.shape,dtype=float64)
	for y in prange(T.shape[0]):
		for t in range(T.shape[1]):
			arr_COMM_CLAWED[y,t]=COMM_CLAWED(T[y,t],DURATIONIF_M[y,0],REN_POL_TERM_Y[y,0],IC_CLAWED_L[y,t],RC_CLAWED_L[y,t])
	return arr_COMM_CLAWED