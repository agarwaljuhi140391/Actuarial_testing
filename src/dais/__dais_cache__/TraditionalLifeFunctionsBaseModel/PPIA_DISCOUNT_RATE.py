
#THIS CODE IS AUTO-GENERATED by simple_template, DO NOT DIRECTLY EDIT UNLESS YOU KNOW WHAT YOU ARE DOING!

from numba import njit, void, prange
import numpy as np
import math
from numpy import float64, int32, int64

@njit(nogil=True, cache=True)
def PPIA_DISCOUNT_RATE(PPIA_DISC_PC,PREM_FREQ):
	if PREM_FREQ==0:
		return 0
	else:
		return (1+PPIA_DISC_PC/100)**(1/PREM_FREQ)-1


@njit(nogil=True, parallel=True, cache=True)
def wrapped_PPIA_DISCOUNT_RATE(PPIA_DISC_PC,PREM_FREQ):
	arr_PPIA_DISCOUNT_RATE=np.zeros((PPIA_DISC_PC.shape[0],1),dtype=float64)
	for y in prange(PPIA_DISC_PC.shape[0]):
		arr_PPIA_DISCOUNT_RATE[y,0]=PPIA_DISCOUNT_RATE(PPIA_DISC_PC[y,0],PREM_FREQ[y,0])
	return arr_PPIA_DISCOUNT_RATE