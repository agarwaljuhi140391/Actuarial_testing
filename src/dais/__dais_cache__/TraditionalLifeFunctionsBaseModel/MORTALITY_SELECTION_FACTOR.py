
#THIS CODE IS AUTO-GENERATED by timeloop_tempate, DO NOT DIRECTLY EDIT UNLESS YOU KNOW WHAT YOU ARE DOING!

from numba import njit, void, prange
import numpy as np
import math
from numpy import float64, int32, int64

@njit(nogil=True, cache=True)
def MORTALITY_SELECTION_FACTOR(MORT_SEL_FAC_BE, POL_YR, REN_POL_TERM_Y, ANN_TERM_Y):
	if POL_YR > (REN_POL_TERM_Y + ANN_TERM_Y):
		return 0
	else:
		return MORT_SEL_FAC_BE/100


@njit(nogil=True, parallel=True, cache=True)
def wrapped_MORTALITY_SELECTION_FACTOR(MORT_SEL_FAC_BE,POL_YR,REN_POL_TERM_Y,ANN_TERM_Y):
	arr_MORTALITY_SELECTION_FACTOR=np.zeros(MORT_SEL_FAC_BE.shape,dtype=float64)
	for y in prange(MORT_SEL_FAC_BE.shape[0]):
		for t in range(MORT_SEL_FAC_BE.shape[1]):
			arr_MORTALITY_SELECTION_FACTOR[y,t]=MORTALITY_SELECTION_FACTOR(MORT_SEL_FAC_BE[y,t],POL_YR[y,t],REN_POL_TERM_Y[y,0],ANN_TERM_Y[y,0])
	return arr_MORTALITY_SELECTION_FACTOR
