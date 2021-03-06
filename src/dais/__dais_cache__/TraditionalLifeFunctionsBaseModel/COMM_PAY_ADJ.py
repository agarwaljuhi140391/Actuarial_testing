
#THIS CODE IS AUTO-GENERATED by timeloop_tempate, DO NOT DIRECTLY EDIT UNLESS YOU KNOW WHAT YOU ARE DOING!

from numba import njit, void, prange
import numpy as np
import math
from numpy import float64, int32, int64

@njit(nogil=True, cache=True)
def COMM_PAY_ADJ(REN_T,CP_TYPE_ADJ_patch,CP_TYPE_patch,PREM_MODE,REN_POL_YR,PREM_FREQ,):
	if ((CP_TYPE_ADJ_patch != 3) and (CP_TYPE_patch != 0))  or PREM_MODE==0 or ((CP_TYPE_patch != 0) and (REN_POL_YR>1)):
		return 1
	else:
		if REN_T==1:
			return PREM_FREQ
		else:
			return 0


@njit(nogil=True, parallel=True, cache=True)
def wrapped_COMM_PAY_ADJ(REN_T,CP_TYPE_ADJ_patch,CP_TYPE_patch,PREM_MODE,REN_POL_YR,PREM_FREQ):
	arr_COMM_PAY_ADJ=np.zeros(REN_T.shape,dtype=float64)
	for y in prange(REN_T.shape[0]):
		for t in range(REN_T.shape[1]):
			arr_COMM_PAY_ADJ[y,t]=COMM_PAY_ADJ(REN_T[y,t],CP_TYPE_ADJ_patch[y,0],CP_TYPE_patch[y,0],PREM_MODE[y,0],REN_POL_YR[y,t],PREM_FREQ[y,0])
	return arr_COMM_PAY_ADJ
