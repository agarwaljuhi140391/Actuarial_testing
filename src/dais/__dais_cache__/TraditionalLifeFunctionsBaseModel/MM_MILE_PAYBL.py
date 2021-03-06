
#THIS CODE IS AUTO-GENERATED by complex_template, DO NOT DIRECTLY EDIT UNLESS YOU KNOW WHAT YOU ARE DOING!

from numba import njit, void, prange
import numpy as np
import math
from numpy import float64, int32, int64
from dais.models.simple_dispatcher import irr_est

@njit(nogil=True, cache=True)
def MM_MILE_PAYBL(T,MAX_COMM_MILE_M,REN_POL_TERM_Y,ANN_TERM_Y,REN_T,POL_TERM_Y,CONSUMPTION_TAX_PC_derived,COMM_MILAGE_BEFORE_TAX,MM_MILE_PAYBL):
	for t in range(T.shape[0]):
		if (T[t]>(REN_POL_TERM_Y[0]+ANN_TERM_Y[0])*12) or (REN_T[t]>min(MAX_COMM_MILE_M[0],POL_TERM_Y[0]*12)):
			MM_MILE_PAYBL[t]=0
		else:
			if REN_T[t]==1:
				MM_MILE_PAYBL[t] = COMM_MILAGE_BEFORE_TAX[t] * (1 + CONSUMPTION_TAX_PC_derived[min(T.shape[0]-1,t+MAX_COMM_MILE_M[0]-1)]/100)
			else:
				MM_MILE_PAYBL[t] = MM_MILE_PAYBL[t-1]


@njit(nogil=True, parallel=True, cache=True)
def wrapped_MM_MILE_PAYBL(T,MAX_COMM_MILE_M,REN_POL_TERM_Y,ANN_TERM_Y,REN_T,POL_TERM_Y,CONSUMPTION_TAX_PC_derived,COMM_MILAGE_BEFORE_TAX):
	arr_MM_MILE_PAYBL=np.zeros((T.shape[0],T.shape[1]),dtype=float64)
	for y in prange(arr_MM_MILE_PAYBL.shape[0]):
		MM_MILE_PAYBL(T[y,:],MAX_COMM_MILE_M[y,:],REN_POL_TERM_Y[y,:],ANN_TERM_Y[y,:],REN_T[y,:],POL_TERM_Y[y,:],CONSUMPTION_TAX_PC_derived[y,:],COMM_MILAGE_BEFORE_TAX[y,:],arr_MM_MILE_PAYBL[y,:])
	return arr_MM_MILE_PAYBL
