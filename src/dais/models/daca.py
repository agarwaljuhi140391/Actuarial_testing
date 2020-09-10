import pandas as pd
import numpy as np
from numpy import int32, int64, float64
from numba import vectorize, prange
from .base import BaseFunctions

#THIS CAN BE A PRE-CALC, NEEDED for all SIMULATIONs? there's one thing that is needed per simulation.
#Can we separate per simulation from one-off upgront? how do we handle that in the model?
#Do one set of stuff up-front?
#Wrap in a for t in range():
#Then wrap in a for p in prange(): for c in prange():
#each time you hit a collapser you need to end the for loops, collapse, then restart the for loops once you're back to PC.

class Daca():
	single_derivations=dict(
		MONTH = {'type': int32, 'numba':'v','shape':2}
		, YEAR = {'type': int32, 'numba':'v','shape':2}
		, ADJ_CA_DEP = {'type': float64, 'numba':'v','shape':2}
		, ADJ_DA_DEP = {'type': float64, 'numba':'v','shape':2}
		, ADJ_CA_WTHDRW = {'type': float64, 'numba':'v','shape':2}
		, ADJ_DA_WTHDRW = {'type': float64, 'numba':'v','shape':2}
		, ADJ_COUP = {'type': float64, 'numba':'v','shape':2}
		, ADJ_DIV = {'type': float64, 'numba':'v','shape':2}
		, ADJ_CA_BAL = {'type': float64, 'numba':'v','shape':2}
		, ADJ_DA_BAL = {'type': float64, 'numba':'v','shape':2}
		, ADJ_CASH_COUP = {'type': float64, 'numba':'v','shape':2}
		, CA_DEP_TAKEUP_RATE = {'type': float64, 'numba':'v','shape':2}
		, DA_DEP_TAKEUP_RATE = {'type': float64, 'numba':'v','shape':2}
		, CA_WITHDRW_RATE = {'type': float64, 'numba':'v','shape':2}
		, DA_WITHDRW_RATE = {'type': float64, 'numba':'v','shape':2}
		, CRED_RATE_CAP = {'type': float64, 'numba':'v','shape':2}
		, CRED_RATE_CA_DEP = {'type': float64, 'numba':'v','shape':2}
		, CRED_RATE_DA_DEP = {'type': float64, 'numba':'v','shape':2}
		
		
		, DEF_INTERP = {'type': float64, 'numba':'j','shape':2}
		, ADJ_CASH_DIV = {'type': float64, 'numba':'j','shape':2}
		, DEFL_NET_CF = {'type': float64, 'numba':'j','shape':2}
	)
	
	complex_derivations={
		'CREDITING_RATE': {'outvars':{
			'FIRST_PRIN_CRED_RATE': {'type': float64, 'shape':2}
			, 'K_FACTOR_ADJ': {'type': float64, 'shape':2}
			, 'CRED_RATE_AFTER_K': {'type': float64, 'shape':2}
			, 'SMOOTHING_FUND_RET': {'type': float64, 'shape':2}
			, 'SMOOTHING_COMP_1': {'type': float64, 'shape':2}
			, 'CRED_RATE_CA_FUND_BAL': {'type': float64, 'shape':2}
			, 'CRED_RATE_DA_FUND_BAL': {'type': float64, 'shape':2}
			, 'SMOOTHING_FUND_RATIO': {'type': float64, 'shape':2}
			, 'SMOOTHING_RELEASE': {'type': float64, 'shape':2}
			, 'CRED_RATE_SMOOTH_RELEASE': {'type': float64, 'shape':2}
			, 'CRED_RATE_AFTER_YOY_CAP': {'type': float64, 'shape':2}
			, 'MKT_ADJ': {'type': float64, 'shape':2}
			, 'CRED_RATE_AFTER_MKT_ADJ': {'type': float64, 'shape':2}
			, 'FINAL_CRED_RATE': {'type': float64, 'shape':2}
			, 'SMOOTHING_COMP_2_4': {'type': float64, 'shape':2}
			, 'SMOOTHING_FUND': {'type': float64, 'shape':2}
			, 'CA_CRED_INT': {'type': float64, 'shape':2}
			, 'DA_CRED_INT': {'type': float64, 'shape':2}
			, 'CRED_RATE_CA_WITHDRW': {'type': float64, 'shape':2}
			, 'CRED_RATE_DA_WITHDRW': {'type': float64, 'shape':2}
			, 'CRED_B4_MKT_ADJ': {'type': float64, 'shape':2}
	}}
	}
	
	summaries={}
	
	mappings={
		# 'EQUITY_I_FAV': {'source':'ASSETS_EQUITY', 'alignment':'PC', 'type':float64, 'shape':1, 'mapping':{'POOL':'POOL','CATEGORY':'CATEGORY'}}

		'DET_COUPON_OUT': {'type':float64, 'shape':2, 'mapping':{'CFLOW_IDX':'CFLOW_IDX','col':'T'}}
		, 'DET_CA_COU_INC': {'type':float64, 'shape':2, 'mapping':{'CFLOW_IDX':'CFLOW_IDX','col':'T'}}
		, 'DET_DA_DIV_INC': {'type':float64, 'shape':2, 'mapping':{'CFLOW_IDX':'CFLOW_IDX','col':'T'}}
		, 'DET_CA_BEN_OUT': {'type':float64, 'shape':2, 'mapping':{'CFLOW_IDX':'CFLOW_IDX','col':'T'}}
		, 'DET_DA_BEN_OUT': {'type':float64, 'shape':2, 'mapping':{'CFLOW_IDX':'CFLOW_IDX','col':'T'}}
		, 'DET_CASHD_OUT': {'type':float64, 'shape':2, 'mapping':{'CFLOW_IDX':'CFLOW_IDX','col':'T'}}
		, 'DET_CA_FUND_IF': {'type':float64, 'shape':2, 'mapping':{'CFLOW_IDX':'CFLOW_IDX','col':'T'}}
		, 'DET_DA_FUND_IF': {'type':float64, 'shape':2, 'mapping':{'CFLOW_IDX':'CFLOW_IDX','col':'T'}}
		, 'pFAV_INV_RET_RATE': {'type':float64, 'shape':2, 'mapping':{'POOL':'INV_RET_POOL_ID','col':'T'}}
		, 'BONUS_ADJ_PCT': {'type':float64, 'shape':2, 'mapping':{'PROD_SERIES':'PROD_SERIES','col':'T'}}
		, 'PAR_YIELD': {'type':float64, 'shape':2, 'mapping':{'POOL':'INV_RET_POOL_ID','col':'T'}}
		, 'FX_INDEX': {'type':float64, 'shape':2, 'mapping':{'SIMULATION':'SIMULATION','ECONOMY':'ECONOMY','col':'YEAR'}}
		, 'VALN_DEF': {'type':float64, 'shape':2, 'mapping':{'SIMULATION':'SIMULATION','ECONOMY':'ECONOMY','col':'YEAR'}}
	}
	
	def DEFL_NET_CF(CRED_RATE_CA_DEP,CRED_RATE_DA_DEP,CRED_RATE_CA_WITHDRW,CRED_RATE_DA_WITHDRW,DEF_INTERP,DEFL_NET_CF):
		for t in range(1,DEFL_NET_CF.shape[0]):
			DEFL_NET_CF[t] = DEFL_NET_CF[t-1] + (CRED_RATE_CA_WITHDRW[t] + CRED_RATE_DA_WITHDRW[t] - CRED_RATE_CA_DEP[t] - CRED_RATE_DA_DEP[t]) * DEF_INTERP[t]

	def CREDITING_RATE(pFAV_INV_RET_RATE,MNTHLY_INV_EXP_PC,DC_SPREAD_PC,CRED_TOL_PC,CRED_SMTH_Y,CRED_Z_PC,PAR_YIELD,CRED_MXBM_PC
			,CRED_RATE_CA_DEP,CRED_RATE_DA_DEP,CA_WITHDRW_RATE,DA_WITHDRW_RATE,ADJ_CA_BAL,ADJ_DA_BAL,INIT_SMTH_FUND_PC
			,CRED_RATE_CAP,DCA_MIN_PC,CRED_MA_S_PC,CRED_MA_Y_PC,DACA_FLAG,INIT_DCAI_PC,CRED_ADJ_CAP,CRED_ULT_FLR_PC,CRED_ULT_CAP_PC
			,FIRST_PRIN_CRED_RATE,K_FACTOR_ADJ,CRED_RATE_AFTER_K,SMOOTHING_FUND_RET,SMOOTHING_COMP_1,SMOOTHING_FUND_RATIO,SMOOTHING_RELEASE
			,CRED_RATE_SMOOTH_RELEASE,CRED_RATE_AFTER_YOY_CAP,MKT_ADJ,CRED_RATE_AFTER_MKT_ADJ,FINAL_CRED_RATE,SMOOTHING_COMP_2_4,SMOOTHING_FUND,CRED_B4_MKT_ADJ
			,CRED_RATE_CA_FUND_BAL,CRED_RATE_DA_FUND_BAL,CA_CRED_INT,DA_CRED_INT,CRED_RATE_CA_WITHDRW,CRED_RATE_DA_WITHDRW
	):
		CRED_B4_MKT_ADJ[0] = INIT_DCAI_PC[0] / 100
		SMOOTHING_FUND[0] = 0 if DACA_FLAG[0]==0 else (ADJ_CA_BAL[0] + ADJ_DA_BAL[0]) * INIT_SMTH_FUND_PC[0] / 100
		CRED_RATE_CA_FUND_BAL[0] = ADJ_CA_BAL[0]
		CRED_RATE_DA_FUND_BAL[0] = ADJ_DA_BAL[0]
		for t in range(1,FIRST_PRIN_CRED_RATE.shape[0]):
			FIRST_PRIN_CRED_RATE[t] = (pFAV_INV_RET_RATE[t] - MNTHLY_INV_EXP_PC[0])*12 - DC_SPREAD_PC[t]/100 - CRED_B4_MKT_ADJ[t-1]
			K_FACTOR_ADJ[t] = FIRST_PRIN_CRED_RATE[t] * CRED_Z_PC[0] / 100
			CRED_RATE_AFTER_K[t] = CRED_B4_MKT_ADJ[t-1] + K_FACTOR_ADJ[t]
			SMOOTHING_FUND_RET[t] = SMOOTHING_FUND[t-1] * (pFAV_INV_RET_RATE[t] - MNTHLY_INV_EXP_PC[0])
			temp = CRED_RATE_CA_FUND_BAL[t-1] + CRED_RATE_DA_FUND_BAL[t-1] + CRED_RATE_CA_DEP[t] + CRED_RATE_DA_DEP[t]
			SMOOTHING_COMP_1[t] = (temp)\
									 * (FIRST_PRIN_CRED_RATE[t] - K_FACTOR_ADJ[t])/12
			SMOOTHING_FUND_RATIO[t] = 0 if temp==0 else (SMOOTHING_FUND[t-1] + SMOOTHING_FUND_RET[t] + SMOOTHING_COMP_1[t])/temp
			SMOOTHING_RELEASE[t] = 0 if ((abs(SMOOTHING_FUND_RATIO[t])<CRED_TOL_PC[0]) or (CRED_SMTH_Y[0]==0)) else SMOOTHING_FUND_RATIO[t]/CRED_SMTH_Y[0]/12
			CRED_RATE_SMOOTH_RELEASE[t] = CRED_RATE_AFTER_K[t] + SMOOTHING_RELEASE[t]
			CRED_B4_MKT_ADJ[t] = max(min(CRED_RATE_SMOOTH_RELEASE[t],CRED_RATE_CAP[t]),DCA_MIN_PC[0])
			CRED_RATE_AFTER_YOY_CAP[t] = max(min(CRED_B4_MKT_ADJ[t],CRED_B4_MKT_ADJ[t-1] + CRED_ADJ_CAP[0]/100),CRED_B4_MKT_ADJ[t-1] - CRED_ADJ_CAP[0]/100)
			MKT_ADJ[t] = (max(PAR_YIELD[t] - CRED_MA_S_PC[0]/100,DCA_MIN_PC[0]/100) - CRED_B4_MKT_ADJ[t]) * CRED_MA_Y_PC[0]/100
			CRED_RATE_AFTER_MKT_ADJ[t] = CRED_B4_MKT_ADJ[t] + MKT_ADJ[t]
			FINAL_CRED_RATE[t] = max(min(max(min(CRED_RATE_AFTER_MKT_ADJ[t],CRED_B4_MKT_ADJ[t-1] + CRED_ADJ_CAP[0]/100),CRED_B4_MKT_ADJ[t-1] - CRED_ADJ_CAP[0]/100),CRED_ULT_CAP_PC[0]/100),CRED_ULT_FLR_PC[0]/100)
			SMOOTHING_COMP_2_4[t] = (temp) * (CRED_RATE_AFTER_K[t] - CRED_RATE_AFTER_YOY_CAP[t])/12
			SMOOTHING_FUND[t] = 0 if DACA_FLAG[0]==0 else (SMOOTHING_FUND[t-1] + SMOOTHING_FUND_RET[t] + SMOOTHING_COMP_1[t] + SMOOTHING_COMP_2_4[t])
			CA_CRED_INT[t] = (CRED_RATE_CA_FUND_BAL[t-1] + CRED_RATE_CA_DEP[t]) * FINAL_CRED_RATE[t] / 12
			DA_CRED_INT[t] = (CRED_RATE_DA_FUND_BAL[t-1] + CRED_RATE_DA_DEP[t]) * FINAL_CRED_RATE[t] / 12
			CRED_RATE_CA_WITHDRW[t] = (CRED_RATE_CA_FUND_BAL[t-1] + CRED_RATE_CA_DEP[t] + CA_CRED_INT[t]) * CA_WITHDRW_RATE[t]
			CRED_RATE_DA_WITHDRW[t] = (CRED_RATE_DA_FUND_BAL[t-1] + CRED_RATE_DA_DEP[t] + DA_CRED_INT[t]) * DA_WITHDRW_RATE[t]
			CRED_RATE_CA_FUND_BAL[t] = CRED_RATE_CA_FUND_BAL[t-1] + CRED_RATE_CA_DEP[t] + CA_CRED_INT[t] - CRED_RATE_CA_WITHDRW[t]
			CRED_RATE_DA_FUND_BAL[t] = CRED_RATE_DA_FUND_BAL[t-1] + CRED_RATE_DA_DEP[t] + DA_CRED_INT[t] - CRED_RATE_DA_WITHDRW[t]

	def CRED_RATE_DA_DEP(DA_DEP_TAKEUP_RATE,ADJ_CASH_DIV):
		return DA_DEP_TAKEUP_RATE * ADJ_CASH_DIV

	def CRED_RATE_CA_DEP(CA_DEP_TAKEUP_RATE,ADJ_CASH_COUP):
		return CA_DEP_TAKEUP_RATE * ADJ_CASH_COUP
	
	def CRED_RATE_CAP(PAR_YIELD,CRED_MXBM_PC,CRED_MBMM_PC):
		return PAR_YIELD * CRED_MXBM_PC + CRED_MBMM_PC
	
	def DA_WITHDRW_RATE(DACA_FLAG,CA_WITHDRW_RATE,ADJ_DA_BAL,ADJ_DA_WTHDRW):
		if DACA_FLAG==2:
			return CA_WITHDRW_RATE
		else:
			return 0 if (ADJ_DA_BAL + ADJ_DA_WTHDRW)==0 else ADJ_DA_WTHDRW/(ADJ_DA_BAL + ADJ_DA_WTHDRW)
	
	def CA_WITHDRW_RATE(ADJ_CA_BAL,ADJ_CA_WTHDRW):
		return 0 if (ADJ_CA_BAL + ADJ_CA_WTHDRW)==0 else ADJ_CA_WTHDRW/(ADJ_CA_BAL + ADJ_CA_WTHDRW)
	
	def DA_DEP_TAKEUP_RATE(CA_DEP_TAKEUP_RATE,DACA_FLAG,ADJ_DIV,ADJ_DA_DEP):
		if DACA_FLAG==2:
			return CA_DEP_TAKEUP_RATE
		else:
			return 0 if ADJ_DIV==0 else ADJ_DA_DEP/ADJ_DIV

	def CA_DEP_TAKEUP_RATE(ADJ_COUP,ADJ_CA_DEP):
		return 0 if ADJ_COUP==0 else ADJ_CA_DEP/ADJ_COUP

	def ADJ_CASH_DIV(ADJ_DIV,BONUS_ADJ_PCT,ADJ_CASH_DIV):
		for t in range(ADJ_CASH_DIV.shape[0]):
			if t==0:
				ADJ_CASH_DIV[t] = 0
			else:
				ADJ_CASH_DIV[t] = ADJ_DIV[t] * BONUS_ADJ_PCT[t-1] / 100

	def ADJ_CASH_COUP(FX_INDEX,LIAB_SCALAR,DET_COUPON_OUT):
		return FX_INDEX * LIAB_SCALAR * DET_COUPON_OUT
	
	def ADJ_DA_BAL(FX_INDEX,LIAB_SCALAR,DET_DA_FUND_IF):
		return FX_INDEX * LIAB_SCALAR * DET_DA_FUND_IF

	def ADJ_CA_BAL(FX_INDEX,LIAB_SCALAR,DET_CA_FUND_IF):
		return FX_INDEX * LIAB_SCALAR * DET_CA_FUND_IF

	def ADJ_DIV(FX_INDEX,LIAB_SCALAR,DET_CASHD_OUT):
		return FX_INDEX * LIAB_SCALAR * DET_CASHD_OUT

	def ADJ_COUP(FX_INDEX,LIAB_SCALAR,DET_COUPON_OUT):
		return FX_INDEX * LIAB_SCALAR * DET_COUPON_OUT

	def ADJ_DA_WTHDRW(FX_INDEX,LIAB_SCALAR,DET_DA_BEN_OUT):
		return FX_INDEX * LIAB_SCALAR * DET_DA_BEN_OUT

	def ADJ_CA_WTHDRW(FX_INDEX,LIAB_SCALAR,DET_CA_BEN_OUT):
		return FX_INDEX * LIAB_SCALAR * DET_CA_BEN_OUT

	def ADJ_DA_DEP(FX_INDEX,LIAB_SCALAR,DET_DA_DIV_INC):
		return FX_INDEX * LIAB_SCALAR * DET_DA_DIV_INC
		
	def ADJ_CA_DEP(FX_INDEX,LIAB_SCALAR,DET_CA_COU_INC):
		return FX_INDEX * LIAB_SCALAR * DET_CA_COU_INC
	
	def DEF_INTERP(MONTH,VALN_DEF,DEF_INTERP):
		for t in range(DEF_INTERP.shape[0]):
			if t==0:
				DEF_INTERP[t]=1
			else:
				if MONTH[t]==1:
					if VALN_DEF[t-1]==0:
						DEF_INTERP[t]=2
					else:
						DEF_INTERP[t]=DEF_INTERP[t-1]*(VALN_DEF[t]/VALN_DEF[t-1])**(1/12)
				else:
					if DEF_INTERP[t-2]==0:
						DEF_INTERP[t]=3
					else:
						DEF_INTERP[t]=DEF_INTERP[t-1]*DEF_INTERP[t-1]/DEF_INTERP[t-2] #this only works if you start in december...
	
	def MONTH(T):
		return (T-1)%12+1
		
	def YEAR(T,val_yr):
		return val_yr+1+(T-1)//12
