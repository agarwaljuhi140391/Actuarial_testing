
#THIS CODE IS AUTO-GENERATED by timeloop_tempate, DO NOT DIRECTLY EDIT UNLESS YOU KNOW WHAT YOU ARE DOING!

from numba import njit, void, prange
import numpy as np
import math
from numpy import float64, int32, int64

@njit(nogil=True, cache=True)
def COMM_CAP_AG_FAC(POL_YR,MAX_COMM_PAYBL_Y,CP_TYPE_patch,COMM_S_AG_FAC,REN_COMM_S_AG_FAC):
	if POL_YR>MAX_COMM_PAYBL_Y:
		return 0
	else:
		if (POL_YR==1) or (CP_TYPE_patch==2):
			return COMM_S_AG_FAC
		else:
			return 	REN_COMM_S_AG_FAC


@njit(nogil=True, parallel=True, cache=True)
def wrapped_COMM_CAP_AG_FAC(POL_YR,MAX_COMM_PAYBL_Y,CP_TYPE_patch,COMM_S_AG_FAC,REN_COMM_S_AG_FAC):
	arr_COMM_CAP_AG_FAC=np.zeros(POL_YR.shape,dtype=float64)
	for y in prange(POL_YR.shape[0]):
		for t in range(POL_YR.shape[1]):
			arr_COMM_CAP_AG_FAC[y,t]=COMM_CAP_AG_FAC(POL_YR[y,t],MAX_COMM_PAYBL_Y[y,0],CP_TYPE_patch[y,0],COMM_S_AG_FAC[y,0],REN_COMM_S_AG_FAC[y,0])
	return arr_COMM_CAP_AG_FAC
