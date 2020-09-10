
#THIS CODE IS AUTO-GENERATED by timeloop_tempate, DO NOT DIRECTLY EDIT UNLESS YOU KNOW WHAT YOU ARE DOING!

from numba import njit, void, prange
import numpy as np
import math
from numpy import float64, int32, int64

@njit(nogil=True, cache=True)
def PPIA_INCOME_RATE(T,SUM_ASSURED2,PREPAID_PREMIUM_PRESENT_VALUE,PREM_INC_PP):
	#This complains about a runtime warning, but I can't figure out where it comes from
	if T>1:
		return 0
	elif SUM_ASSURED2==0:
		return 0
	else:
		return PREPAID_PREMIUM_PRESENT_VALUE*PREM_INC_PP/SUM_ASSURED2


@njit(nogil=True, parallel=True, cache=True)
def wrapped_PPIA_INCOME_RATE(T,SUM_ASSURED2,PREPAID_PREMIUM_PRESENT_VALUE,PREM_INC_PP):
	arr_PPIA_INCOME_RATE=np.zeros(T.shape,dtype=float64)
	for y in prange(T.shape[0]):
		for t in range(T.shape[1]):
			arr_PPIA_INCOME_RATE[y,t]=PPIA_INCOME_RATE(T[y,t],SUM_ASSURED2[y,t],PREPAID_PREMIUM_PRESENT_VALUE[y,0],PREM_INC_PP[y,t])
	return arr_PPIA_INCOME_RATE