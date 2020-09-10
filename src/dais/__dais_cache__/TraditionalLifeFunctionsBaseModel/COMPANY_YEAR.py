
#THIS CODE IS AUTO-GENERATED by complex_template, DO NOT DIRECTLY EDIT UNLESS YOU KNOW WHAT YOU ARE DOING!

from numba import njit, void, prange
import numpy as np
import math
from numpy import float64, int32, int64
from dais.models.simple_dispatcher import irr_est

@njit(nogil=True, cache=True)
def COMPANY_YEAR(FISCAL_YEAR,min_ppia_pc_lkup_yr,max_ppia_pc_lkup_yr,COMPANY_YEAR):
	COMPANY_YEAR[0]=min(max(FISCAL_YEAR[0],min_ppia_pc_lkup_yr[0]),max_ppia_pc_lkup_yr[0])


@njit(nogil=True, parallel=True, cache=True)
def wrapped_COMPANY_YEAR(FISCAL_YEAR,min_ppia_pc_lkup_yr,max_ppia_pc_lkup_yr):
	arr_COMPANY_YEAR=np.zeros((FISCAL_YEAR.shape[0],1),dtype=int32)
	for y in prange(arr_COMPANY_YEAR.shape[0]):
		COMPANY_YEAR(FISCAL_YEAR[y,:],min_ppia_pc_lkup_yr[y,:],max_ppia_pc_lkup_yr[y,:],arr_COMPANY_YEAR[y,:])
	return arr_COMPANY_YEAR
