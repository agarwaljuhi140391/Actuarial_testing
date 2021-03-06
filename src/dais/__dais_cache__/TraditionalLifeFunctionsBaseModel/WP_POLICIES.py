
#THIS CODE IS AUTO-GENERATED by derivedcomplex_template, DO NOT DIRECTLY EDIT UNLESS YOU KNOW WHAT YOU ARE DOING!

from numba import njit, void, prange
import numpy as np
import math
from numpy import float64, int32, int64

@njit(nogil=True, cache=True)
def WP_POLICIES(T,MATURITY_RATE,DEATH_RATE_MONTHLY, LAPSE_RATE_MONTHLY, NO_HTHY_WP
		,REN_POL_TERM_Y,ANN_TERM_Y,PREM_WAIVER_IND,DURATIONIF_M
		,NO_WP_POLS_IFSM,NO_WP_DEATHS,NO_WP_SURRS,NO_WP_POLS_IF,NO_WP_MATS):
	# NO_WP_POLS_IF_start=np.zeros(T.shape[0],dtype=float64)
	for t in range(T.shape[0]):
		if t==0:
			if PREM_WAIVER_IND[0]==1:
				NO_WP_POLS_IF_start=1
			else:
				NO_WP_POLS_IF_start=0
			NO_WP_MATS[t]=NO_WP_POLS_IF_start * MATURITY_RATE[t]
			NO_WP_POLS_IFSM[t]=NO_WP_POLS_IF_start
		else:
			NO_WP_MATS[t]=NO_WP_POLS_IF[t-1] * MATURITY_RATE[t]
			NO_WP_POLS_IFSM[t]=NO_WP_POLS_IF[t-1] - NO_WP_MATS[t]
		
		NO_WP_DEATHS[t]=NO_WP_POLS_IFSM[t] * DEATH_RATE_MONTHLY[t]
		NO_WP_SURRS[t]=NO_WP_POLS_IFSM[t] * (1-DEATH_RATE_MONTHLY[t]) * LAPSE_RATE_MONTHLY[t]
		if T[t]>(REN_POL_TERM_Y[0]+ANN_TERM_Y[0])*12:
			NO_WP_POLS_IF[t]=0
		else:
			if T[t]<=DURATIONIF_M[0]:
				NO_WP_POLS_IF[t]=NO_WP_POLS_IF_start
			else:
				NO_WP_POLS_IF[t]=NO_WP_POLS_IFSM[t]-(NO_WP_DEATHS[t]+NO_WP_SURRS[t])+NO_HTHY_WP[t]


@njit(nogil=True, parallel=True, cache=True)
def wrapped_WP_POLICIES(T,MATURITY_RATE,DEATH_RATE_MONTHLY,LAPSE_RATE_MONTHLY,NO_HTHY_WP,REN_POL_TERM_Y,ANN_TERM_Y,PREM_WAIVER_IND,DURATIONIF_M):
	arr_NO_WP_POLS_IFSM=np.zeros(T.shape,float64)
	arr_NO_WP_DEATHS=np.zeros(T.shape,float64)
	arr_NO_WP_SURRS=np.zeros(T.shape,float64)
	arr_NO_WP_POLS_IF=np.zeros(T.shape,float64)
	arr_NO_WP_MATS=np.zeros(T.shape,float64)
	for y in prange(T.shape[0]):
		WP_POLICIES(T[y,:],MATURITY_RATE[y,:],DEATH_RATE_MONTHLY[y,:],LAPSE_RATE_MONTHLY[y,:],NO_HTHY_WP[y,:],REN_POL_TERM_Y[y,:],ANN_TERM_Y[y,:],PREM_WAIVER_IND[y,:],DURATIONIF_M[y,:],arr_NO_WP_POLS_IFSM[y,:],arr_NO_WP_DEATHS[y,:],arr_NO_WP_SURRS[y,:],arr_NO_WP_POLS_IF[y,:],arr_NO_WP_MATS[y,:])
	return {'NO_WP_POLS_IFSM': arr_NO_WP_POLS_IFSM,'NO_WP_DEATHS': arr_NO_WP_DEATHS,'NO_WP_SURRS': arr_NO_WP_SURRS,'NO_WP_POLS_IF': arr_NO_WP_POLS_IF,'NO_WP_MATS': arr_NO_WP_MATS}
