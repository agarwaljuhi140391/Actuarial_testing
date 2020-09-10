
#THIS CODE IS AUTO-GENERATED by timeloop_tempate, DO NOT DIRECTLY EDIT UNLESS YOU KNOW WHAT YOU ARE DOING!

from numba import njit, void, prange
import numpy as np
import math
from numpy import float64, int32, int64

@njit(nogil=True, cache=True)
def COMM_RECOV_PC(COMM_CB_CODE,REN_T,COMM_CB_RATE,MAX_COMM_CB_M):
	if (COMM_CB_CODE==999999) or (REN_T>MAX_COMM_CB_M):
		return 0
	else:
		return COMM_CB_RATE


@njit(nogil=True, parallel=True, cache=True)
def wrapped_COMM_RECOV_PC(COMM_CB_CODE,REN_T,COMM_CB_RATE,MAX_COMM_CB_M):
	arr_COMM_RECOV_PC=np.zeros(REN_T.shape,dtype=float64)
	for y in prange(REN_T.shape[0]):
		for t in range(REN_T.shape[1]):
			arr_COMM_RECOV_PC[y,t]=COMM_RECOV_PC(COMM_CB_CODE[y,0],REN_T[y,t],COMM_CB_RATE[y,t],MAX_COMM_CB_M[0,0])
	return arr_COMM_RECOV_PC
