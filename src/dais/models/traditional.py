from . import common
import pandas as pd
from numba import jit
import numpy as np
import inspect
# from ..tools import _inspect_and_jit
import functools
from numpy import int64, float64


class TraditionalLifeFunctions:
	"""This is a specific class implementing Traditional Life products """
	single_derivations={
		'DURATIONIF_M': int64,
		'POL_YR': int64,
		'POL_MTH': int64,
		'ATTAINED_AGE': int64,
		'ANN_TERM_Y': int64,
		'Base_Decrement_WOP_Rates': float64,
		'WOP_SEX_IDX': int64,
		'Policy_Year_Reset_At_renewal': int64,
		'Remaining_Prem_Term': int64,
		'LAPSE_SPIKE_RATE': float64,
		'BASE_LAPSE_RATE': float64,
		'LAPSE_CHAN_IDX_patch': int64,
		'PREM_FREQ': int64,
		'LAPSE_PREMIUM_FREQ_IDX': int64,
		'Spike_Adjusted_Lapse_Rate': float64,
		'LAPSE_RATE_ANNUAL': float64,
		'LAPSE_PAD_PC': float64,
		'LAPSE_RATE_MTH': float64,
# 		'MORTALITY_SELECTION_FACTOR': float64,
# 		'MORTALITY_SELECTION_FACTOR_col': int64,
# 		'MORT_SEL_SEX_IDX': int64,
# 		'MORT_SEX_IDX': int64,
# 		'MORT_SMK_IDX': int64,
# 		'BASE_DEATH_RATE': float64,
# 		'DEATH_RATE_ANNUAL': float64
	}
	
	def DEATH_RATE_ANNUAL(MORTALITY_SELECTION_FACTOR, BASE_DEATH_RATE, SUBS_PC, DEATH_RATE_ANNUAL):
		for i in range(DEATH_RATE_ANNUAL.shape[0]):
			if BASE_DEATH_RATE[i]==1:
				DEATH_RATE_ANNUAL[i]=1
			else:
				#MORT_PAD is empty so I'm just hard-coding for now
				DEATH_RATE_ANNUAL[i]=BASE_DEATH_RATE[i]*MORTALITY_SELECTION_FACTOR[i]*SUBS_PC[i]*(1+0)
	
	def MORT_SMK_IDX(SMOKER_STAT,MORT_SMK_IND,MORT_SMK_IDX):
		for i in range(MORT_SMK_IDX.shape[0]):
			if MORT_SMK_IND[i]==0:
				MORT_SMK_IDX[i]=999999
			else:
				MORT_SMK_IDX[i]=SMOKER_STAT[i]
	
	def BASE_DEATH_RATE(MORT_RATE_BE, REN_POL_TERM_Y, ANN_TERM_Y, BASE_DEATH_RATE):
		for i in range(BASE_DEATH_RATE.shape[0]):
			if (REN_POL_TERM_Y[i] + ANN_TERM_Y[i])>0:
				BASE_DEATH_RATE[i] = MORT_RATE_BE[i]
	
	def MORTALITY_SELECTION_FACTOR_col(BASE_POL_TERM_Y, RENEWAL_COUNT, max_mort_sel_lkup_yr, POL_YR, MORTALITY_SELECTION_FACTOR_col):
		for i in range(MORTALITY_SELECTION_FACTOR_col.shape[0]):
			MORTALITY_SELECTION_FACTOR_col[i]=max(
													min(POL_YR[i]+BASE_POL_TERM_Y[i]*RENEWAL_COUNT[i],max_mort_sel_lkup_yr),
													1)
	
	def MORTALITY_SELECTION_FACTOR(POL_YR,REN_POL_TERM_Y,ANN_TERM_Y,MORT_SEL_FAC_BE,MORTALITY_SELECTION_FACTOR):
		for i in range(MORTALITY_SELECTION_FACTOR.shape[0]):
			if POL_YR[i]>(REN_POL_TERM_Y[i]+ANN_TERM_Y[i]):
				MORTALITY_SELECTION_FACTOR[i]=0
			else:
				MORTALITY_SELECTION_FACTOR[i]=MORT_SEL_FAC_BE[i]/100
	
	def LAPSE_RATE_MTH(LAPSE_RATE_ANNUAL, LAPSE_RATE_MTH):
		for i in range(LAPSE_RATE_MTH.shape[0]):
			LAPSE_RATE_MTH[i] = 1 - (1 - LAPSE_RATE_ANNUAL[i]**(1/12))
	
	def LAPSE_PAD_PC(LAPSE_PAD_CODE, LAPSE_PAD_PC):
		for i in range(LAPSE_PAD_PC.shape[0]):
			if LAPSE_PAD_CODE[i] == 999999:
				LAPSE_PAD_PC[i] = 0
			else:
			#THIS CODE IS BROKEN IN THE SPREADSHEET
				LAPSE_PAD_PC[i] = 0
	
	def LAPSE_RATE_ANNUAL(Spike_Adjusted_Lapse_Rate, LAPSE_PAD_PC, LAPSE_RATE_ANNUAL):
		for i in range(LAPSE_RATE_ANNUAL.shape[0]):
			LAPSE_RATE_ANNUAL[i] = Spike_Adjusted_Lapse_Rate[i] * (1 + LAPSE_PAD_PC[i]/100)
	
	def Spike_Adjusted_Lapse_Rate(SPIKE_RATE_IND, LAPSE_SPIKE_RATE, BASE_LAPSE_RATE, SPIKE_LAPSE_PERIOD
					, POL_YR, REN_POL_TERM_Y, RENEWAL_COUNT, DOWN_PAYMENT, Remaining_Prem_Term, Spike_Adjusted_Lapse_Rate):
		for i in range(Spike_Adjusted_Lapse_Rate.shape[0]):
			#I THINK THE EXCEL HAS A BUG ON DOWN_PAYMENT. Y/N vs 0/1 is mishandled.
			if( SPIKE_RATE_IND[i] == 0) or (POL_YR[i] > REN_POL_TERM_Y[i]) or (RENEWAL_COUNT[i] == 1 and DOWN_PAYMENT[i] == 1):
				Spike_Adjusted_Lapse_Rate[i] = BASE_LAPSE_RATE[i]
			elif Remaining_Prem_Term[i] == -1:
				Spike_Adjusted_Lapse_Rate[i] = LAPSE_SPIKE_RATE[i]
			elif (Remaining_Prem_Term[i] < SPIKE_LAPSE_PERIOD[i]) and (Remaining_Prem_Term[i] > -1):
				Spike_Adjusted_Lapse_Rate[i] = BASE_LAPSE_RATE[i] - (BASE_LAPSE_RATE[i] - LAPSE_SPIKE_RATE[i])*(1 - Remaining_Prem_Term[i]/SPIKE_LAPSE_PERIOD[i])
			else:
				Spike_Adjusted_Lapse_Rate[i] = BASE_LAPSE_RATE[i]
	
	def PREM_FREQ(PREM_MODE,PREM_FREQ):
		for i in range(PREM_FREQ.shape[0]):
			if PREM_MODE[i]==0:
				PREM_FREQ[i]=0
			elif PREM_MODE[i]==1:
				PREM_FREQ[i]=1
			elif PREM_MODE[i] in [2,3]:
				PREM_FREQ[i]=2
			elif PREM_MODE[i] in [4,5,6]:
				PREM_FREQ[i]=12
			else:
				PREM_FREQ[i]=999999
	
	def LAPSE_PREMIUM_FREQ_IDX(LAPSE_PREM_FREQ_IND,PREM_FREQ,LAPSE_PREMIUM_FREQ_IDX):
		for i in range(LAPSE_PREMIUM_FREQ_IDX.shape[0]):
			if LAPSE_PREM_FREQ_IND[i]==0:
				LAPSE_PREMIUM_FREQ_IDX[i]=999999
			else:
				LAPSE_PREMIUM_FREQ_IDX[i]=PREM_FREQ[i]
	
	def LAPSE_CHAN_IDX_patch(LAPSE_CHAN_IND,CHAN_CODE,LAPSE_CHAN_IDX_patch):
		for i in range(LAPSE_CHAN_IDX_patch.shape[0]):
			if LAPSE_CHAN_IND[i]==0:
				LAPSE_CHAN_IDX_patch[i]=999999
			else:
				LAPSE_CHAN_IDX_patch[i]=CHAN_CODE[i]
	
	def BASE_LAPSE_RATE(LAPSE_CODE, POL_YR, REN_POL_TERM_Y, Policy_Year_Reset_At_renewal, max_lapse_lkup_yr, LAPSE_RATE_BE, BASE_LAPSE_RATE):
		for i in range(BASE_LAPSE_RATE.shape[0]):
			if (LAPSE_CODE[i]==999999) or (POL_YR[i] > REN_POL_TERM_Y[i]):
				BASE_LAPSE_RATE[i]=0
			elif Policy_Year_Reset_At_renewal[i] > max_lapse_lkup_yr:
				BASE_LAPSE_RATE[i]=0
			elif POL_YR[i] >=1:
				BASE_LAPSE_RATE[i]=LAPSE_RATE_BE[i]/100
			else:
				BASE_LAPSE_RATE[i]=0
	
	def LAPSE_SPIKE_RATE(SPIKE_RATE_IND, UB_PT, SPIKE_RATE_BE, POL_YR, POL_MTH, PREM_PAYBL_Y, Remaining_Prem_Term, LAPSE_SPIKE_RATE):
		for i in range(LAPSE_SPIKE_RATE.shape[0]):
			if (SPIKE_RATE_IND[i] == 0) or (POL_YR[i] > (PREM_PAYBL_Y[i]+1)):
				LAPSE_SPIKE_RATE[i]=0
			elif (UB_PT[i] > Remaining_Prem_Term[i]) and (POL_MTH[i] >= 1):
				LAPSE_SPIKE_RATE[i]=SPIKE_RATE_BE[i]
			else:
				LAPSE_SPIKE_RATE[i]=0

	def Remaining_Prem_Term(REN_POL_TERM_Y, DURATIONIF_M, PREM_PAYBL_Y, POL_YR, Remaining_Prem_Term):
		for i in range(Remaining_Prem_Term.shape[0]):
			if DURATIONIF_M[i] > REN_POL_TERM_Y[i]*12:
				Remaining_Prem_Term[i] = 0
			else:
				Remaining_Prem_Term[i] = PREM_PAYBL_Y[i] - POL_YR[i]
	
	def Policy_Year_Reset_At_renewal(POL_YR, POL_TERM_Y, Policy_Year_Reset_At_renewal):
		for i in range(Policy_Year_Reset_At_renewal.shape[0]):
			Policy_Year_Reset_At_renewal[i] = ((POL_YR[i] + POL_TERM_Y[i] - 1) % POL_TERM_Y[i])+1
		
	def DURATIONIF_M(ENTRY_YEAR,ENTRY_MONTH,val_yr,val_mth,DURATIONIF_M):
		for i in range(DURATIONIF_M.shape[0]):
			DURATIONIF_M[i]=(val_yr-ENTRY_YEAR[i])*12 + val_mth - ENTRY_MONTH[i]+1

	def POL_YR(DURATIONIF_M,POL_YR):
		for i in range(POL_YR.shape[0]):
			POL_YR[i]=DURATIONIF_M[i] // 12 + 1

	def POL_MTH(DURATIONIF_M,POL_MTH):
		for i in range(POL_MTH.shape[0]):
			POL_MTH[i]=DURATIONIF_M[i] % 12 + 1
			
	def ATTAINED_AGE(AGE_AT_ENTRY,POL_YR,ATTAINED_AGE):
		for i in range(ATTAINED_AGE.shape[0]):
			ATTAINED_AGE[i]=AGE_AT_ENTRY[i] + POL_YR[i] - 1
	
	def ANN_TERM_Y(PROD_CODE,k_term,DTH_ANN_GUAR,AGE_AT_ENTRY,POL_TERM_Y,ANN_TERM_Y):
		for i in range(ANN_TERM_Y.shape[0]):
			if PROD_CODE[i]==1208:
				if k_term[i]==99:
					ANN_TERM_Y[i]=max(DTH_ANN_GUAR[i],80-(AGE_AT_ENTRY[i]+POL_TERM_Y[i]))
				else:
					ANN_TERM_Y[i]=DTH_ANN_GUAR[i]
			else:
				ANN_TERM_Y[i]=0
				
	def MORT_SEL_SEX_IDX(JUV_IND,MORT_SEL_SEX_IND, SEX2, SEX, MORT_SEL_SEX_IDX):
		for i in range(MORT_SEL_SEX_IDX.shape[0]):
			if MORT_SEL_SEX_IND[i]==0:
				MORT_SEL_SEX_IDX[i]=999999
			else:
				if JUV_IND[i]==1:
					MORT_SEL_SEX_IDX[i]=SEX2[i]
				else:
					MORT_SEL_SEX_IDX[i]=SEX[i]

	def MORT_SEX_IDX(JUV_IND,CMT_MORT_SEX_IND, SEX2, SEX, MORT_SEX_IDX):
		for i in range(MORT_SEX_IDX.shape[0]):
			if CMT_MORT_SEX_IND[i]==0:
				MORT_SEX_IDX[i]=999999
			else:
				if JUV_IND[i]==1:
					MORT_SEX_IDX[i]=SEX2[i]
				else:
					MORT_SEX_IDX[i]=SEX[i]
	
	def WOP_SEX_IDX(DEC_WOP_SEX_IND, SEX, WOP_SEX_IDX):
		for i in range(WOP_SEX_IDX.shape[0]):
			if DEC_WOP_SEX_IND[i]==1:
				WOP_SEX_IDX[i]=SEX[i]
			else:
				WOP_SEX_IDX[i]=999999
				
	def Base_Decrement_WOP_Rates(DEC_WOP_CODE, WOP_RATE_BE, ANN_TERM_Y, REN_POL_TERM_Y, ATTAINED_AGE, max_incidence_lkup_age, POL_YR, POL_MTH, Base_Decrement_WOP_Rates):
		for i in range(Base_Decrement_WOP_Rates.shape[0]):
			if DEC_WOP_CODE[i] == 999999:
				Base_Decrement_WOP_Rates[i]=0
			elif ATTAINED_AGE[i] > max_incidence_lkup_age:
				Base_Decrement_WOP_Rates[i]=0
			elif POL_YR[i]>(REN_POL_TERM_Y[i]+ANN_TERM_Y[i]):
				Base_Decrement_WOP_Rates[i]=0
			elif POL_MTH[i]>=1:
				Base_Decrement_WOP_Rates[i]=WOP_RATE_BE[i]
					
				