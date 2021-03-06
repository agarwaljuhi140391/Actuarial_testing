
#THIS CODE IS AUTO-GENERATED by timeloop_tempate, DO NOT DIRECTLY EDIT UNLESS YOU KNOW WHAT YOU ARE DOING!

from numba import njit, void, prange
import numpy as np
import math
from numpy import float64, int32, int64

@njit(nogil=True, cache=True)
def LAPSE_SPIKE_RATE(SPIKE_RATE_BE,SPIKE_RATE_IND,POL_YR,PREM_PAYBL_Y,REMAINING_PREMIUM_TERM,SPIKE_LAPSE_PERIOD2):
	if (SPIKE_RATE_IND==0) or (POL_YR>PREM_PAYBL_Y+1):
		return 0
	else:
		if (SPIKE_LAPSE_PERIOD2>REMAINING_PREMIUM_TERM):
			if (REMAINING_PREMIUM_TERM<-1) or (REMAINING_PREMIUM_TERM>4):
				return 0
			else:
				return SPIKE_RATE_BE
		else:
			return 0


@njit(nogil=True, parallel=True, cache=True)
def wrapped_LAPSE_SPIKE_RATE(SPIKE_RATE_BE,SPIKE_RATE_IND,POL_YR,PREM_PAYBL_Y,REMAINING_PREMIUM_TERM,SPIKE_LAPSE_PERIOD2):
	arr_LAPSE_SPIKE_RATE=np.zeros(SPIKE_RATE_BE.shape,dtype=float64)
	for y in prange(SPIKE_RATE_BE.shape[0]):
		for t in range(SPIKE_RATE_BE.shape[1]):
			arr_LAPSE_SPIKE_RATE[y,t]=LAPSE_SPIKE_RATE(SPIKE_RATE_BE[y,t],SPIKE_RATE_IND[y,0],POL_YR[y,t],PREM_PAYBL_Y[y,0],REMAINING_PREMIUM_TERM[y,t],SPIKE_LAPSE_PERIOD2[y,0])
	return arr_LAPSE_SPIKE_RATE
