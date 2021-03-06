
#THIS CODE IS AUTO-GENERATED by simple_template, DO NOT DIRECTLY EDIT UNLESS YOU KNOW WHAT YOU ARE DOING!

from numba import njit, void, prange
import numpy as np
import math
from numpy import float64, int32, int64

@njit(nogil=True, cache=True)
def ANN_TERM_Y(PROD_CODE,k_term,DTH_ANN_GUAR,AGE_AT_ENTRY,POL_TERM_Y):
	if PROD_CODE==1208:
		if k_term==99:
			return max(DTH_ANN_GUAR,80-(AGE_AT_ENTRY+POL_TERM_Y))
		else:
			return DTH_ANN_GUAR
	else:
		return 0


@njit(nogil=True, parallel=True, cache=True)
def wrapped_ANN_TERM_Y(PROD_CODE,k_term,DTH_ANN_GUAR,AGE_AT_ENTRY,POL_TERM_Y):
	arr_ANN_TERM_Y=np.zeros((PROD_CODE.shape[0],1),dtype=int32)
	for y in prange(PROD_CODE.shape[0]):
		arr_ANN_TERM_Y[y,0]=ANN_TERM_Y(PROD_CODE[y,0],k_term[y,0],DTH_ANN_GUAR[y,0],AGE_AT_ENTRY[y,0],POL_TERM_Y[y,0])
	return arr_ANN_TERM_Y
