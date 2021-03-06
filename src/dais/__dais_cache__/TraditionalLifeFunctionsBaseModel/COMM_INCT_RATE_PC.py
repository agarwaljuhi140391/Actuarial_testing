
#THIS CODE IS AUTO-GENERATED by timeloop_tempate, DO NOT DIRECTLY EDIT UNLESS YOU KNOW WHAT YOU ARE DOING!

from numba import njit, void, prange
import numpy as np
import math
from numpy import float64, int32, int64

@njit(nogil=True, cache=True)
def COMM_INCT_RATE_PC(POL_YR,MAX_COMM_PAYBL_Y,COMM_INCT_RATE_PC_1,REN_COMM_INCT_RATE_PC_1):
	if POL_YR>MAX_COMM_PAYBL_Y:
		return 0
	else:
		if POL_YR==1:
			return COMM_INCT_RATE_PC_1
		else:
			return REN_COMM_INCT_RATE_PC_1


@njit(nogil=True, parallel=True, cache=True)
def wrapped_COMM_INCT_RATE_PC(POL_YR,MAX_COMM_PAYBL_Y,COMM_INCT_RATE_PC_1,REN_COMM_INCT_RATE_PC_1):
	arr_COMM_INCT_RATE_PC=np.zeros(POL_YR.shape,dtype=float64)
	for y in prange(POL_YR.shape[0]):
		for t in range(POL_YR.shape[1]):
			arr_COMM_INCT_RATE_PC[y,t]=COMM_INCT_RATE_PC(POL_YR[y,t],MAX_COMM_PAYBL_Y[y,0],COMM_INCT_RATE_PC_1[y,0],REN_COMM_INCT_RATE_PC_1[y,0])
	return arr_COMM_INCT_RATE_PC
