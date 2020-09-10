import pandas as pd
import numpy as np
from numpy import int32, int64, float64
from numba import vectorize, prange

class BaseFunctions:
	single_derivations={
		'DURATIONIF_M': {'type': int32, 'numba':'v','shape':1}
# 		, 'PREM_FREQ': {'type': int32, 'numba':'v','shape':1}

		, 'POL_YR': {'type': int32, 'numba':'v','shape':2}
		, 'POL_MTH': {'type': int32, 'numba':'v','shape':2}
		, 'ATTAINED_AGE': {'type': int32, 'numba':'v','shape':2}
		, 'ATTAINED_AGE_Y': {'type': int32, 'numba':'v','shape':2}
		, 'REMAINING_PREMIUM_TERM': {'type': int32, 'numba':'v','shape':2}
	}
	
	complex_derivations={}
	
	summaries={}
	
	def DURATIONIF_M(ENTRY_YEAR, ENTRY_MONTH, val_yr, val_mth):
		return (val_yr - ENTRY_YEAR)*12 + val_mth - ENTRY_MONTH+1

	def POL_YR(T):
		return (T+11)//12
		
	def POL_MTH(T,POL_YR):
		return T-(POL_YR-1)*12
		
	def ATTAINED_AGE(POL_YR,AGE_AT_ENTRY):
		return AGE_AT_ENTRY+POL_YR-1
	def ATTAINED_AGE_Y(POL_YR,AGE_AT_ENTRY2):
		return AGE_AT_ENTRY2+POL_YR-1

	def REMAINING_PREMIUM_TERM(T, PREM_PAYBL_Y,POL_YR,REN_POL_TERM_Y):
		if T>REN_POL_TERM_Y*12:
			return 0
		else:
			return PREM_PAYBL_Y-POL_YR