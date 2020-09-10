
#THIS CODE IS AUTO-GENERATED by timeloop_tempate, DO NOT DIRECTLY EDIT UNLESS YOU KNOW WHAT YOU ARE DOING!

from numba import njit, void, prange
import numpy as np
import math
from numpy import float64, int32, int64

@njit(nogil=True, cache=True)
def COMM_MILE_PAID(T,DURATIONIF_M,REN_POL_TERM_Y,ANN_TERM_Y,MM_MILE_PAYBL,NO_HTHY_POLS_IFSM):
	if (T<=DURATIONIF_M) or (T>(REN_POL_TERM_Y+ANN_TERM_Y)*12):
		return 0
	else:
		return MM_MILE_PAYBL*NO_HTHY_POLS_IFSM


@njit(nogil=True, parallel=True, cache=True)
def wrapped_COMM_MILE_PAID(T,DURATIONIF_M,REN_POL_TERM_Y,ANN_TERM_Y,MM_MILE_PAYBL,NO_HTHY_POLS_IFSM):
	arr_COMM_MILE_PAID=np.zeros(T.shape,dtype=float64)
	for y in prange(T.shape[0]):
		for t in range(T.shape[1]):
			arr_COMM_MILE_PAID[y,t]=COMM_MILE_PAID(T[y,t],DURATIONIF_M[y,0],REN_POL_TERM_Y[y,0],ANN_TERM_Y[y,0],MM_MILE_PAYBL[y,t],NO_HTHY_POLS_IFSM[y,t])
	return arr_COMM_MILE_PAID
