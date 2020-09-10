
#THIS CODE IS AUTO-GENERATED by timeloop_tempate, DO NOT DIRECTLY EDIT UNLESS YOU KNOW WHAT YOU ARE DOING!

from numba import njit, void, prange
import numpy as np
import math
from numpy import float64, int32, int64

@njit(nogil=True, cache=True)
def RC_PAID(T,DURATIONIF_M,RC_PAYBL_PP,NO_HTHY_POLS_IFSM):
	if T<=DURATIONIF_M:
		return 0
	else:
		return RC_PAYBL_PP*NO_HTHY_POLS_IFSM


@njit(nogil=True, parallel=True, cache=True)
def wrapped_RC_PAID(T,DURATIONIF_M,RC_PAYBL_PP,NO_HTHY_POLS_IFSM):
	arr_RC_PAID=np.zeros(T.shape,dtype=float64)
	for y in prange(T.shape[0]):
		for t in range(T.shape[1]):
			arr_RC_PAID[y,t]=RC_PAID(T[y,t],DURATIONIF_M[y,0],RC_PAYBL_PP[y,t],NO_HTHY_POLS_IFSM[y,t])
	return arr_RC_PAID