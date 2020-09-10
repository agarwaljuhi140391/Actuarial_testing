import numpy as np
import math
from numpy import int32, int64, float64
from .base import BaseFunctions

class TraditionalLifeFunctions(BaseFunctions):
	"""This is a specific class implementing Traditional Life products """
	single_derivations=dict(**BaseFunctions.single_derivations
		, PREM_FREQ= {'type': int32, 'numba':'v','shape':1}
		, LAPSE_CHAN_IDX= {'type': int64, 'numba':'v','shape':1}
		, LAPSE_PREMIUM_FREQ_IDX= {'type': int64, 'numba':'v','shape':1}
		, LAPSE_PAD_PC= {'type': float64, 'numba':'v','shape':1}
		, MORT_SEL_SEX_IDX= {'type': int32, 'numba':'v','shape':1}
		, MORT_SEX_IDX= {'type': int32, 'numba':'v','shape':1}
		, MORT_SMK_IDX= {'type': int32, 'numba':'v','shape':1}
		, MORT_RATE_MAX_AGE= {'type': int32, 'numba':'v','shape':1}
		, WOP_SEX_IDX= {'type': int32, 'numba':'v','shape':1}
		, WOP_EXP_SEX_IDX= {'type': int32, 'numba':'v','shape':1}
		, PPIA_DISCOUNT_RATE= {'type': float64, 'numba':'v','shape':1}
		, PREPAID_PREMIUM_PRESENT_VALUE= {'type': float64, 'numba':'v','shape':1}
		, SPIKE_LAPSE_PERIOD2= {'type': int32, 'numba':'v','shape':1}
		, SP_IND={'type': int32, 'numba':'v','shape':1}
		, COMM_AG_FAC_1_derived={'type': int32, 'numba':'v','shape':1}
		, REN_COMM_AG_FAC_1_derived={'type': int32, 'numba':'v','shape':1}
		, COMM_ADJ_MULT={'type': float64, 'numba':'v','shape':1}
		, COMM_S_RATE={'type': float64, 'numba':'v','shape':1}
		, REN_COMM_S_RATE={'type': float64, 'numba':'v','shape':1}
		, COMM_S_AG_FAC={'type': float64, 'numba':'v','shape':1}
		, REN_COMM_S_AG_FAC={'type': float64, 'numba':'v','shape':1}
		, COMM_LIMIT_TYPE_derived={'type': float64, 'numba':'v','shape':1}
		, IC_ADD_BASE_PC_derived={'type': float64, 'numba':'v','shape':1}
		, IC_ADD_SPEC_PC_derived={'type': float64, 'numba':'v','shape':1}
		, COMM_MILE_PC_derived={'type': float64, 'numba':'v','shape':1}
		, COMM_CB_CODE_patch={'type': int32, 'numba':'v','shape':1}
		, ANN_TERM_Y= {'type': int32, 'numba':'v','shape':1}
		
		, COMPANY_YEAR= {'type': int32, 'numba':'j','shape':1}
			
		, POLICY_YR_RESET_AT_RENEWAL= {'type': int64, 'numba':'v','shape':2}
		, BASE_LAPSE_RATES= {'type': float64, 'numba':'v','shape':2}
		, BASE_DEATH_RATES= {'type': float64, 'numba':'v','shape':2}
		, SPIKE_ADJUSTED_LAPSE_RATE= {'type': float64, 'numba':'v','shape':2}
		, LAPSE_RATE_ANNUAL= {'type': float64, 'numba':'v','shape':2}
		, LAPSE_RATE_MONTHLY= {'type': float64, 'numba':'v','shape':2}
		, MORTALITY_SELECTION_FACTOR_col= {'type': int32, 'numba':'v','shape':2}
		, MORTALITY_SELECTION_FACTOR= {'type': float64, 'numba':'v','shape':2}
		, MORT_AGE_col= {'type': int32, 'numba':'v','shape':2}
		, DEATH_RATE_ANNUAL= {'type': float64, 'numba':'v','shape':2}
		, DEATH_RATE_MONTHLY= {'type': float64, 'numba':'v','shape':2}
		, BASE_DECREMENT_WOP_RATE= {'type': float64, 'numba':'v','shape':2}
		, DECREMENT_WOP_EXPERIENCE= {'type': float64, 'numba':'v','shape':2}
		, WOP_EXP_FAC_BE_col= {'type': int32, 'numba':'v','shape':2}
		, DECREMENT_WOP_RATES_ANNUAL= {'type': float64, 'numba':'v','shape':2}
		, DECREMENT_WOP_RATES_MONTHLY= {'type': float64, 'numba':'v','shape':2}
		, MATURITY_RATE= {'type': float64, 'numba':'v','shape':2}
		, SUM_ASSURED2= {'type': float64, 'numba':'v','shape':2}
		, REN_T= {'type': int32, 'numba':'v','shape':2}
		, PREM_FRAC_PP= {'type': float64, 'numba':'v','shape':2}
		, ANN_PREM_PP= {'type': float64, 'numba':'v','shape':2}
		, PREM_INC_PP= {'type': float64, 'numba':'v','shape':2}
		, PREM_INC= {'type': float64, 'numba':'v','shape':2}
		, CALENDAR_YEAR= {'type': int32, 'numba':'v','shape':2}
		, CALENDAR_MONTH= {'type': int32, 'numba':'v','shape':2}
		, FISCAL_MONTH= {'type': int32, 'numba':'v','shape':2}
		, FISCAL_YEAR= {'type': int32, 'numba':'v','shape':2}
		, PPIA_INCOME_RATE= {'type': float64, 'numba':'v','shape':2}
		, NO_POLS_IFSM= {'type': float64, 'numba':'v','shape':2}
		, PPIA_INC_IF= {'type': float64, 'numba':'v','shape':2}
		, PPIA_OFFSET_RATE= {'type': float64, 'numba':'v','shape':2}
		, PPIA_OFFSET_IF= {'type': float64, 'numba':'v','shape':2}
		, LAPSE_SPIKE_RATE= {'type': float64, 'numba':'v','shape':2}
		, COMM_RATE_PC={'type': float64, 'numba':'v','shape':2}
		# , COMM_AG_FAC={'type': float64, 'numba':'v','shape':2}
		, COMM_RATE_BEFORE_ROUND={'type': float64, 'numba':'v','shape':2}
		, COMM_INCT_RATE_PC={'type': float64, 'numba':'v','shape':2}
		, COMM_MODAL_FAC={'type': float64, 'numba':'v','shape':2}
		, IC_INCT_PAYBL_PP={'type': float64, 'numba':'v','shape':2}
		, BASE_COMM_CAP_RATE={'type': float64, 'numba':'v','shape':2}
		, COMM_CAP_RATE={'type': float64, 'numba':'v','shape':2}
		, COMM_CAP_AG_FAC={'type': float64, 'numba':'v','shape':2}
		, COMM_RATE_ADJ_PC={'type': float64, 'numba':'v','shape':2}
		, RC_BASE_PAYBL_PP={'type': float64, 'numba':'v','shape':2}
		, COMM_PAY_ADJ={'type': float64, 'numba':'v','shape':2}
		, IC_BASE_PAYBL_PP={'type': float64, 'numba':'v','shape':2}
		, COMM_CAP={'type': float64, 'numba':'v','shape':2}
		, IC_ADD_PAYBL_PP={'type': float64, 'numba':'v','shape':2}
		, REN_POL_YR={'type': float64, 'numba':'v','shape':2}
		, RENEW_COMM_BEFORE_TAX={'type': float64, 'numba':'v','shape':2}
		, CONSUMPTION_TAX_PC_derived={'type': float64, 'numba':'v','shape':2}
		, RC_PAYBL_PP={'type': float64, 'numba':'v','shape':2}
		, INITIAL_COMM_BEFORE_TAX={'type': float64, 'numba':'v','shape':2}
		, IC_PAYBL_PP={'type': float64, 'numba':'v','shape':2}
		, COMM_BON_PC_derived={'type': float64, 'numba':'v','shape':2}
		, COMM_MILAGE_RATE={'type': float64, 'numba':'v','shape':2}
		, COMM_CB_ADJ={'type': float64, 'numba':'v','shape':2}
		, COMM_RECOV_PC={'type': float64, 'numba':'v','shape':2}
		, NO_LAPSE={'type': float64, 'numba':'v','shape':2}
		, RC_CLAWED_L={'type': float64, 'numba':'v','shape':2}
		, IC_CLAWED_L={'type': float64, 'numba':'v','shape':2}
		, IC_PAID={'type': float64, 'numba':'v','shape':2}
		, RC_PAID={'type': float64, 'numba':'v','shape':2}
		, COMM_MILE_PAID={'type': float64, 'numba':'v','shape':2}
		, COMM_CLAWED={'type': float64, 'numba':'v','shape':2}
		, TOT_COMM={'type': float64, 'numba':'v','shape':2}

		, DISC_PREM_INC={'type': float64, 'numba':'j','shape':2}
		, COMM_RATE_ACCUM_PC={'type': float64, 'numba':'j','shape':2}
		, COMM_MILAGE_BEFORE_TAX={'type': float64, 'numba':'j','shape':2}
		, MM_MILE_PAYBL={'type': float64, 'numba':'j','shape':2}
		, PARTIAL_RENEW_COMM_EARNED={'type': float64, 'numba':'j','shape':2}
		, PARTIAL_INIT_COMM_EARNED={'type': float64, 'numba':'j','shape':2}
		, DISC_TOT_COMM={'type': float64, 'numba':'j','shape':2}

	)

	complex_derivations={
		'HEALTHY_POLICIES': {'outvars':{
			'NO_HTHY_POLS_IFSM': {'type': float64, 'shape':2}
			, 'NO_HTHY_DEATHS': {'type': float64, 'shape':2}
			, 'NO_HTHY_SURRS': {'type': float64, 'shape':2}
			, 'NO_HTHY_WP': {'type': float64, 'shape':2}
			, 'NO_HTHY_POLS_IF': {'type': float64, 'shape':2}
			, 'NO_HTHY_MATS': {'type': float64, 'shape':2}
		}}
		, 'WP_POLICIES': {'outvars':{
			'NO_WP_POLS_IFSM': {'type': float64, 'shape':2}
			, 'NO_WP_DEATHS': {'type': float64, 'shape':2}
			, 'NO_WP_SURRS': {'type': float64, 'shape':2}
			, 'NO_WP_POLS_IF': {'type': float64, 'shape':2}
			, 'NO_WP_MATS': {'type': float64, 'shape':2}
		}}
	}
	
	#The code assume that vars are shape 2 and byvars are shape 1
	summaries={
		'Summary_PROD_CODE': {'byvars':['PROD_CODE'],'vars':['DISC_PREM_INC','DISC_TOT_COMM'],'func':'sum'}
	}
	
	
	mappings={
		'LAPSE_CODE': {'source': 'ASSPT_CODE_BE', 'type': int32, 'shape':1, 'mapping':{'PROD_CODE':'BASE_PROD_CODE_patch'}}
		, 'LAPSE_CHAN_IND': {'source': 'ASSPT_LKUP_IND_BE', 'type': int32, 'shape':1, 'mapping':{'PROD_CODE':'BASE_PROD_CODE_patch'}}
		, 'SPIKE_RATE_IND': {'source': 'ASSPT_LKUP_IND_BE', 'type': int32, 'shape':1, 'mapping':{'PROD_CODE':'BASE_PROD_CODE_patch'}}
		, 'SPIKE_LAPSE_PERIOD': {'source': 'SPIKE_RATE_PERIOD_BE', 'type': int32, 'shape':1, 'mapping':{'ISS_YR_IDX':'ISS_YR_IDX','INDEX':'SPIKE_RATE_PERIOD_INDEX'}}
		, 'UB_PT': {'source': 'SPIKE_RATE_PERIOD_BE', 'type': int32, 'shape':1, 'mapping':{'ISS_YR_IDX':'ISS_YR_IDX','INDEX':'SPIKE_RATE_PERIOD_INDEX'}}
		, 'LAPSE_PREM_FREQ_IND': {'source': 'ASSPT_LKUP_IND_BE', 'type': int32, 'shape':1, 'mapping':{'PROD_CODE':'BASE_PROD_CODE_patch'}}
		, 'LAPSE_PAD_CODE': {'source': 'ASSPT_CODE_BE', 'type': int32, 'shape':1, 'mapping':{'PROD_CODE':'BASE_PROD_CODE_patch'}}
		, 'MORT_SEL_CODE': {'source': 'ASSPT_CODE_BE', 'type': int32, 'shape':1, 'mapping':{'PROD_CODE':'PROD_CODE'}}
		, 'MORT_CODE': {'source': 'ASSPT_CODE_BE', 'type': int32, 'shape':1, 'mapping':{'PROD_CODE':'PROD_CODE'}}
		, 'DEC_WOP_CODE': {'source': 'ASSPT_CODE_BE', 'type': int32, 'shape':1, 'mapping':{'PROD_CODE':'PROD_CODE'}}
		, 'MORT_SEL_SEX_IND': {'source': 'ASSPT_LKUP_IND_BE', 'type': int32, 'shape':1, 'mapping':{'PROD_CODE':'PROD_CODE'}}
		, 'MORT_SEX_IND': {'source': 'ASSPT_LKUP_IND_BE', 'type': int32, 'shape':1, 'mapping':{'PROD_CODE':'PROD_CODE'}}
		, 'DEC_WOP_SEX_IND': {'source': 'ASSPT_LKUP_IND_BE', 'type': int32, 'shape':1, 'mapping':{'PROD_CODE':'PROD_CODE'}}
		, 'MORT_SMK_IND': {'source': 'ASSPT_LKUP_IND_BE', 'type': int32, 'shape':1, 'mapping':{'PROD_CODE':'PROD_CODE'}}
		, 'JUV_IND': {'source': 'BEN_CTRL', 'type': int32, 'shape':1, 'mapping':{'PROD_CODE':'PROD_CODE','Generation':'GENERATION'}}
		, 'DEC_WOP_EXP_CODE': {'source': 'ASSPT_CODE_BE', 'type': int32, 'shape':1, 'mapping':{'PROD_CODE':'PROD_CODE'}}
		, 'DEC_WOP_EXP_SEX_IND': {'source': 'ASSPT_LKUP_IND_BE', 'type': int32, 'shape':1, 'mapping':{'PROD_CODE':'PROD_CODE'}}
		, 'DEC_WOP_IND': {'source': 'BEN_CTRL', 'type': int32, 'shape':1, 'mapping':{'PROD_CODE':'PROD_CODE','Generation':'GENERATION'}}
		, 'REN_RATE': {'source': 'ASSPT_BE', 'type': float64, 'shape':1, 'mapping':{'ISS_YR_IDX':'ISS_YR_IDX','PROD_CODE':'PROD_CODE','INDEX':'ASSPT_BE_index'}}
		, 'ANN_RATE': {'source': 'ASSPT_BE', 'type': float64, 'shape':1, 'mapping':{'ISS_YR_IDX':'ISS_YR_IDX','PROD_CODE':'PROD_CODE','INDEX':'ASSPT_BE_index'}}
		, 'PPIA_DISC_PC': {'source': 'PPIA_RATES', 'type': float64, 'shape':1, 'mapping':{'COMPANY_YR':'COMPANY_YEAR'}}
		, 'MAX_COMM_PAYBL_Y': {'source': 'COMM_CTRL', 'type': int32, 'shape':1, 'mapping':{'PROD_CODE':'PROD_CODE','COMM_ID':'COMM_ID','SMOKER_STAT':'SMOKER_STAT','SP_IND':'SP_IND'}}
		, 'COMM_ADJ_MULT_TYPE': {'source': 'COMM_CTRL', 'type': int32, 'shape':1, 'mapping':{'PROD_CODE':'PROD_CODE','COMM_ID':'COMM_ID','SMOKER_STAT':'SMOKER_STAT','SP_IND':'SP_IND'}}
		, 'COMM_LIMIT_TYPE': {'source': 'COMM_CTRL', 'type': int32, 'shape':1, 'mapping':{'PROD_CODE':'PROD_CODE','COMM_ID':'COMM_ID','SMOKER_STAT':'SMOKER_STAT','SP_IND':'SP_IND'}}
		, 'COMM_LIMIT_ADJ': {'source': 'COMM_LIMIT_ADJ_TBL', 'type': int32, 'shape':1, 'mapping':{'COMM_LIMIT_TYPE':'COMM_LIMIT_TYPE_derived','INDEX':'COMM_LIMIT_ADJ_index'}}
		, 'IC_ADD_BASE_IND': {'source': 'INIT_COMM_ADD_IND', 'type': int32, 'shape': 1, 'mapping':{'PROD_CODE':'PROD_CODE', 'PREM_FREQ':'PREM_FREQ',	'COMM_RULE':'COMM_RULE'}}
		, 'IC_ADD_SPEC_IND': {'source': 'INIT_COMM_ADD_IND', 'type': int32, 'shape': 1, 'mapping':{'PROD_CODE':'PROD_CODE', 'PREM_FREQ':'PREM_FREQ',	'COMM_RULE':'COMM_RULE'}}
		, 'IC_ADD_BASE_PC': {'source': 'INIT_COMM_ADD', 'type': float64, 'shape': 1, 'mapping':{'COMM_RULE':'COMM_RULE', 'AG_RANK_ORIG':'AG_RANK_ORIG'}}
		, 'IC_ADD_SPEC_PC': {'source': 'INIT_COMM_ADD', 'type': float64, 'shape': 1, 'mapping':{'COMM_RULE':'COMM_RULE', 'AG_RANK_ORIG':'AG_RANK_ORIG'}}
		, 'COMM_MILE_PC': {'source': 'COMM_MILE_PC_TAB', 'type': float64, 'shape': 1, 'mapping':{'COMM_RULE':'COMM_RULE', 'AG_RANK_ORIG':'AG_RANK_ORIG'}}
		, 'COMM_MILE_PC_IND': {'source': 'COMM_MILE_PC_IND_TAB', 'type': float64, 'shape': 1, 'mapping':{'PROD_CODE':'PROD_CODE', 'PREM_FREQ':'PREM_FREQ', 'COMM_RULE':'COMM_RULE'}}
		
		, 'MORT_SEL_FAC_BE': {'type': float64
				, 'shape':2
				, 'mapping':{'ISS_YR_IDX': 'ISS_YR_IDX'
							, 'MORT_SEL_CODE': 'MORT_SEL_CODE'
							, 'MORT_SEL_SEX_IDX': 'MORT_SEL_SEX_IDX'
							, 'col': 'MORTALITY_SELECTION_FACTOR_col'}}
		, 'MORT_RATE_BE': {'autocapcol': True
				, 'type': float64
				, 'shape':2
				, 'mapping':{'ISS_YR_IDX': 'ISS_YR_IDX'
							, 'MORT_CODE': 'MORT_CODE'
							, 'MORT_SMK_IDX': 'MORT_SMK_IDX'
							, 'MORT_SEX_IDX': 'MORT_SEX_IDX'
							, 'col': 'MORT_AGE_col'}}
		# , 'MORT_RATE_BE2': {'autocapcol': True
				# , 'type': float64
				# , 'shape':2
				# , 'mapping':{'ISS_YR_IDX': 'ISS_YR_IDX'
							# , 'MORT_CODE': 'MORT_CODE'
							# , 'MORT_SMK_IDX': 'MORT_SMK_IDX'
							# , 'MORT_SEX_IDX': 'MORT_SEX_IDX'
							# , 'col': 'MORT_AGE_col'}}
		, 'LAPSE_RATE_BE': {'autocapcol': True, 'type': float64, 'shape':2, 'mapping':{'ISS_YR_IDX': 'ISS_YR_IDX'
																, 'LAPSE_CODE': 'LAPSE_CODE'
																, 'PREM_FREQ': 'LAPSE_PREMIUM_FREQ_IDX'
																, 'LAPSE_CHAN_IDX': 'LAPSE_CHAN_IDX'
																, 'col': 'POLICY_YR_RESET_AT_RENEWAL'}}
		# , 'LAPSE_RATE_BE2': {'autocapcol': True, 'type': float64, 'shape':2, 'mapping':{'ISS_YR_IDX': 'ISS_YR_IDX'
																# , 'LAPSE_CODE': 'LAPSE_CODE'
																# , 'PREM_FREQ': 'LAPSE_PREMIUM_FREQ_IDX'
																# , 'LAPSE_CHAN_IDX': 'LAPSE_CHAN_IDX'
																# , 'col': 'POLICY_YR_RESET_AT_RENEWAL'}}
		, 'WOP_RATE_BE': {'autocapcol': True, 'type': float64, 'shape':2, 'mapping':{'ISS_YR_IDX': 'ISS_YR_IDX'
																, 'WOP_CODE': 'DEC_WOP_CODE'
																, 'WOP_SEX_IDX': 'WOP_SEX_IDX'
																, 'col': 'ATTAINED_AGE'
																}}
		, 'WOP_EXP_FAC_BE': {'type': float64, 'shape':2, 'mapping':{'ISS_YR_IDX': 'ISS_YR_IDX'
																, 'WOP_EXP_CODE': 'DEC_WOP_EXP_CODE'
																, 'WOP_EXP_SEX_IDX': 'WOP_EXP_SEX_IDX'
																, 'col': 'WOP_EXP_FAC_BE_col'
																}}
		, 'SPIKE_RATE_BE': {'autocapcol': True, 'type': float64, 'shape':2, 'mapping':{'ISS_YR_IDX': 'ISS_YR_IDX'
																, 'SPIKE_LAPSE_LKUP_IDX': 'UB_PT'
																, 'col': 'REMAINING_PREMIUM_TERM'
																}}
		, 'CONSUMPTION_TAX_PC': {'autocapcol': True, 'type': float64, 'shape':2, 'mapping':{'ISS_YR_IDX': 'ISS_YR_IDX'
																, 'col': 'FISCAL_YEAR'
																}}
		, 'COMM_BON_BE': {'autocapcol': True, 'type': float64, 'shape':2, 'mapping':{'ISS_YR_IDX': 'ISS_YR_IDX'
																, 'col': 'CALENDAR_YEAR'
																}}
		, 'COMM_CB_RATE': {'autocapcol': True, 'type': float64, 'shape':2, 'mapping':{'COMM_CB_CODE': 'COMM_CB_CODE_patch'
																, 'CP_TYPE':'CP_TYPE_patch'
																, 'PREM_FREQ':'PREM_FREQ'
																, 'col': 'REN_T'
																}}	
}

	def DISC_TOT_COMM(T
			, DURATIONIF_M,REN_POL_TERM_Y,ANN_TERM_Y
			, M_DISC_RATE
			, TOT_COMM
			, DISC_TOT_COMM):
		for t in range(T.shape[0],-1,-1):#Note this counts down
			if t==T.shape[0]:
				DISC_TOT_COMM[t]=0
			else:
				if (T[t]<DURATIONIF_M[0]) or (T[t]>(REN_POL_TERM_Y[0]+ANN_TERM_Y[0])*12) or (M_DISC_RATE[0]==-1):
					DISC_TOT_COMM[t]=0
				else:
					DISC_TOT_COMM[t]=-1*TOT_COMM[t+1]+DISC_TOT_COMM[t+1]/(1+M_DISC_RATE[0])

	# def DISC_TOT_COMM(T
			# , DURATIONIF_M,REN_POL_TERM_Y,ANN_TERM_Y
			# , M_DISC_RATE
			# , TOT_COMM
			# , DISC_TOT_COMM):
		# for i in prange(T.shape[0]):
			# for t in range(T.shape[1],-1,-1):#Note this counts down
				# if t==T.shape[1]:
					# DISC_TOT_COMM[i,t]=0
				# else:
					# if (T[i,t]<DURATIONIF_M[i,0]) or (T[i,t]>(REN_POL_TERM_Y[i,0]+ANN_TERM_Y[i,0])*12) or (M_DISC_RATE[0,0]==-1):
						# DISC_TOT_COMM[i,t]=0
					# else:
						# DISC_TOT_COMM[i,t]=-1*TOT_COMM[i,t+1]+DISC_TOT_COMM[i,t+1]/(1+M_DISC_RATE[0,0])


	def TOT_COMM(COMM_CLAWED,COMM_MILE_PAID,IC_PAID,RC_PAID):
		return COMM_CLAWED+COMM_MILE_PAID+IC_PAID+RC_PAID

	def COMM_CLAWED(T,DURATIONIF_M,REN_POL_TERM_Y,IC_CLAWED_L,RC_CLAWED_L):
		if (T<=DURATIONIF_M) or (T>REN_POL_TERM_Y*12):
			return 0
		else:
			return IC_CLAWED_L+RC_CLAWED_L

	def COMM_MILE_PAID(T,DURATIONIF_M,REN_POL_TERM_Y,ANN_TERM_Y,MM_MILE_PAYBL,NO_HTHY_POLS_IFSM):
		if (T<=DURATIONIF_M) or (T>(REN_POL_TERM_Y+ANN_TERM_Y)*12):
			return 0
		else:
			return MM_MILE_PAYBL*NO_HTHY_POLS_IFSM

	def IC_PAID(T,DURATIONIF_M,IC_PAYBL_PP,NO_HTHY_POLS_IFSM):
		if T<=DURATIONIF_M:
			return 0
		else:
			return IC_PAYBL_PP*NO_HTHY_POLS_IFSM
	def RC_PAID(T,DURATIONIF_M,RC_PAYBL_PP,NO_HTHY_POLS_IFSM):
		if T<=DURATIONIF_M:
			return 0
		else:
			return RC_PAYBL_PP*NO_HTHY_POLS_IFSM

	def COMM_CB_CODE_patch(COMM_CB_CODE):
		if COMM_CB_CODE==999999:
			return 1
		else:
			return COMM_CB_CODE

	def NO_LAPSE(NO_HTHY_SURRS,NO_WP_SURRS):
		return NO_HTHY_SURRS+NO_WP_SURRS

	def IC_CLAWED_L(T,DURATIONIF_M,REN_POL_TERM_Y,PARTIAL_INIT_COMM_EARNED,COMM_RECOV_PC,COMM_CB_ADJ,NO_LAPSE):
		if T<DURATIONIF_M or T>REN_POL_TERM_Y*12:
			return 0
		else:
			return PARTIAL_INIT_COMM_EARNED*COMM_RECOV_PC*COMM_CB_ADJ*NO_LAPSE
		
	def RC_CLAWED_L(T,DURATIONIF_M,REN_POL_TERM_Y,PARTIAL_RENEW_COMM_EARNED,COMM_RECOV_PC,COMM_CB_ADJ,NO_LAPSE):
		if T<DURATIONIF_M or T>REN_POL_TERM_Y*12:
			return 0
		else:
			return PARTIAL_RENEW_COMM_EARNED*COMM_RECOV_PC*COMM_CB_ADJ*NO_LAPSE

	def COMM_RECOV_PC(COMM_CB_CODE,REN_T,COMM_CB_RATE,MAX_COMM_CB_M):
		if (COMM_CB_CODE==999999) or (REN_T>MAX_COMM_CB_M):
			return 0
		else:
			return COMM_CB_RATE

	def COMM_CB_ADJ(POL_YR,NBYM,COMM_CB_ST_YM,COMM_CB_ADJ_RATE):
		if POL_YR>1	and NBYM>=COMM_CB_ST_YM:
			return COMM_CB_ADJ_RATE
		else:
			return 1

	def MM_MILE_PAYBL(T,MAX_COMM_MILE_M,REN_POL_TERM_Y,ANN_TERM_Y,REN_T,POL_TERM_Y,CONSUMPTION_TAX_PC_derived,COMM_MILAGE_BEFORE_TAX,MM_MILE_PAYBL):
		for t in range(T.shape[0]):
			if (T[t]>(REN_POL_TERM_Y[0]+ANN_TERM_Y[0])*12) or (REN_T[t]>min(MAX_COMM_MILE_M[0],POL_TERM_Y[0]*12)):
				MM_MILE_PAYBL[t]=0
			else:
				if REN_T[t]==1:
					MM_MILE_PAYBL[t] = COMM_MILAGE_BEFORE_TAX[t] * (1 + CONSUMPTION_TAX_PC_derived[min(T.shape[0]-1,t+MAX_COMM_MILE_M[0]-1)]/100)
				else:
					MM_MILE_PAYBL[t] = MM_MILE_PAYBL[t-1]

	# def MM_MILE_PAYBL(T,MAX_COMM_MILE_M,REN_POL_TERM_Y,ANN_TERM_Y,REN_T,POL_TERM_Y,CONSUMPTION_TAX_PC_derived,COMM_MILAGE_BEFORE_TAX,MM_MILE_PAYBL):
		# for i in prange(T.shape[0]):
			# for t in range(T.shape[1]):
				# if (T[i,t]>(REN_POL_TERM_Y[i,0]+ANN_TERM_Y[i,0])*12) or (REN_T[i,t]>min(MAX_COMM_MILE_M[0,0],POL_TERM_Y[i,0]*12)):
					# MM_MILE_PAYBL[i,t]=0
				# else:
					# if REN_T[i,t]==1:
						# MM_MILE_PAYBL[i,t] = COMM_MILAGE_BEFORE_TAX[i,t] * (1 + CONSUMPTION_TAX_PC_derived[i,min(T.shape[1]-1,t+MAX_COMM_MILE_M[0,0]-1)]/100)
					# else:
						# MM_MILE_PAYBL[i,t] = MM_MILE_PAYBL[i,t-1]


	def COMM_MILE_PC_derived(COMM_MILE_PC,COMM_MILE_PC_IND,AG_SYS_CODE):
		if (AG_SYS_CODE != 1) or (COMM_MILE_PC_IND != 1):
			return 0
		else:
			return COMM_MILE_PC

	def COMM_MILAGE_RATE(COMM_MILE_PC_derived,T,REN_POL_TERM_Y):
		if T>REN_POL_TERM_Y*12:
			return 0
		else:
			return COMM_MILE_PC_derived
	
	def COMM_MILAGE_BEFORE_TAX(COMM_MILAGE_RATE,T,REN_POL_TERM_Y,ANN_TERM_Y,REN_T,MAX_COMM_MILE_M,POL_TERM_Y,CP_TYPE_patch
			,IC_BASE_PAYBL_PP,COMM_CAP,IC_INCT_PAYBL_PP,PREM_FREQ,COMM_MILE_FAC,COMM_MILAGE_BEFORE_TAX):
		for t in range(T.shape[0]):
			if (T[t]>(REN_POL_TERM_Y[0]+ANN_TERM_Y[0])*12) or (REN_T[t]>min(MAX_COMM_MILE_M[0],POL_TERM_Y[0]*12)):
				COMM_MILAGE_BEFORE_TAX[t] = 0
			else:
				if REN_T[t]==1:
					if min(MAX_COMM_MILE_M[0],POL_TERM_Y[t])==0:
						COMM_MILAGE_BEFORE_TAX[t]=0
					else:
						COMM_MILAGE_BEFORE_TAX[t] = (min(IC_BASE_PAYBL_PP[t],COMM_CAP[t])-IC_INCT_PAYBL_PP[t])*(
						PREM_FREQ[0]**(2 if CP_TYPE_patch[0]==0 else 1)
						)*COMM_MILAGE_RATE[t]*COMM_MILE_FAC[0]/(min(MAX_COMM_MILE_M[0],POL_TERM_Y[t]*12))
				else:
					COMM_MILAGE_BEFORE_TAX[t]=COMM_MILAGE_BEFORE_TAX[t-1]

	# def COMM_MILAGE_BEFORE_TAX(COMM_MILAGE_RATE,T,REN_POL_TERM_Y,ANN_TERM_Y,REN_T,MAX_COMM_MILE_M,POL_TERM_Y,CP_TYPE_patch
			# ,IC_BASE_PAYBL_PP,COMM_CAP,IC_INCT_PAYBL_PP,PREM_FREQ,COMM_MILE_FAC,COMM_MILAGE_BEFORE_TAX):
		# for i in prange(T.shape[0]):
			# for t in range(T.shape[1]):
				# if (T[i,t]>(REN_POL_TERM_Y[i,0]+ANN_TERM_Y[i,0])*12) or (REN_T[i,t]>min(MAX_COMM_MILE_M[0,0],POL_TERM_Y[i,0]*12)):
					# COMM_MILAGE_BEFORE_TAX[i,t] = 0
				# else:
					# if REN_T[i,t]==1:
						# COMM_MILAGE_BEFORE_TAX[i,t] = (min(IC_BASE_PAYBL_PP[i,t],COMM_CAP[i,t])-IC_INCT_PAYBL_PP[i,t])*(
							# PREM_FREQ[i,0]**(2 if CP_TYPE_patch[i,0]==0 else 1)
							# )*COMM_MILAGE_RATE[i,t]*COMM_MILE_FAC[0,0]/(min(MAX_COMM_MILE_M[0,0],POL_TERM_Y[i,t]*12))
					# else:
						# COMM_MILAGE_BEFORE_TAX[i,t]=COMM_MILAGE_BEFORE_TAX[i,t-1]


	def COMM_BON_PC_derived(COMM_BON_BE,T,REN_POL_TERM_Y):
		if T>REN_POL_TERM_Y*12:
			return 0
		else:
			return COMM_BON_BE
	
	def IC_PAYBL_PP(INITIAL_COMM_BEFORE_TAX,CONSUMPTION_TAX_PC_derived):
		return INITIAL_COMM_BEFORE_TAX*(1+CONSUMPTION_TAX_PC_derived/100)

	def RC_PAYBL_PP(RENEW_COMM_BEFORE_TAX,CONSUMPTION_TAX_PC_derived,COMM_BON_PC_derived):
		return RENEW_COMM_BEFORE_TAX*(1+CONSUMPTION_TAX_PC_derived/100)*(1+COMM_BON_PC_derived/100)
		
	def INITIAL_COMM_BEFORE_TAX(IC_BASE_PAYBL_PP,COMM_CAP,IC_ADD_PAYBL_PP,COMM_PAY_ADJ):
		return (min(IC_BASE_PAYBL_PP,COMM_CAP)+IC_ADD_PAYBL_PP) * COMM_PAY_ADJ
	
	def RENEW_COMM_BEFORE_TAX(RC_BASE_PAYBL_PP,COMM_CAP,COMM_PAY_ADJ):
		return min(RC_BASE_PAYBL_PP,COMM_CAP) * COMM_PAY_ADJ
		
	def CONSUMPTION_TAX_PC_derived(CONSUMPTION_TAX_PC,T,REN_POL_TERM_Y):
		if T>REN_POL_TERM_Y*12:
			return 0
		else:
			#simplifying this logic - might hide a mistake
			return CONSUMPTION_TAX_PC
	
	def PARTIAL_INIT_COMM_EARNED(T,REN_POL_TERM_Y,REN_T,IC_PAYBL_PP,PARTIAL_INIT_COMM_EARNED):
		for t in range(T.shape[0]):
			if T[t]>(REN_POL_TERM_Y[0]*12):
				PARTIAL_INIT_COMM_EARNED[t] = 0
			else:
				if (T[t] != 1) and (REN_T[t]==1):
					PARTIAL_INIT_COMM_EARNED[t] = IC_PAYBL_PP[t]
				else:
					PARTIAL_INIT_COMM_EARNED[t] = PARTIAL_INIT_COMM_EARNED[t-1] + IC_PAYBL_PP[t]

	# def PARTIAL_INIT_COMM_EARNED(T,REN_POL_TERM_Y,REN_T,IC_PAYBL_PP,PARTIAL_INIT_COMM_EARNED):
		# for i in prange(T.shape[0]):
			# for t in range(T.shape[1]):
				# if T[i,t]>(REN_POL_TERM_Y[i,0]*12):
					# PARTIAL_INIT_COMM_EARNED[i,t] = 0
				# else:
					# if (T[i,t] != 1) and (REN_T[i,t]==1):
						# PARTIAL_INIT_COMM_EARNED[i,t] = IC_PAYBL_PP[i,t]
					# else:
						# PARTIAL_INIT_COMM_EARNED[i,t] = PARTIAL_INIT_COMM_EARNED[i,t-1] + IC_PAYBL_PP[i,t]


	def PARTIAL_RENEW_COMM_EARNED(T,REN_POL_TERM_Y,REN_T,RC_PAYBL_PP,CONSUMPTION_TAX_PC_derived,PARTIAL_RENEW_COMM_EARNED):
		for t in range(T.shape[0]):
			if T[t]>(REN_POL_TERM_Y[0]*12):
				PARTIAL_RENEW_COMM_EARNED[t] = 0
			else:
				if (T[t] != 1) and (REN_T[t]==1):
					PARTIAL_RENEW_COMM_EARNED[t] = RC_PAYBL_PP[t]/(1+CONSUMPTION_TAX_PC_derived[t]/100)
				else:
					PARTIAL_RENEW_COMM_EARNED[t] = PARTIAL_RENEW_COMM_EARNED[t-1] + RC_PAYBL_PP[t]/(1+CONSUMPTION_TAX_PC_derived[t]/100)

	# def PARTIAL_RENEW_COMM_EARNED(T,REN_POL_TERM_Y,REN_T,RC_PAYBL_PP,CONSUMPTION_TAX_PC_derived,PARTIAL_RENEW_COMM_EARNED):
		# for i in prange(T.shape[0]):
			# for t in range(T.shape[1]):
				# if T[i,t]>(REN_POL_TERM_Y[i,0]*12):
					# PARTIAL_RENEW_COMM_EARNED[i,t] = 0
				# else:
					# if (T[i,t] != 1) and (REN_T[i,t]==1):
						# PARTIAL_RENEW_COMM_EARNED[i,t] = RC_PAYBL_PP[i,t]/(1+CONSUMPTION_TAX_PC_derived[i,t]/100)
					# else:
						# PARTIAL_RENEW_COMM_EARNED[i,t] = PARTIAL_RENEW_COMM_EARNED[i,t-1] + RC_PAYBL_PP[i,t]/(1+CONSUMPTION_TAX_PC_derived[i,t]/100)


	def IC_ADD_PAYBL_PP(IC_BASE_PAYBL_PP,COMM_CAP,IC_INCT_PAYBL_PP,IC_ADD_BASE_PC_derived,IC_ADD_SPEC_PC_derived):
		return (min(IC_BASE_PAYBL_PP,COMM_CAP) - IC_INCT_PAYBL_PP) * (IC_ADD_BASE_PC_derived+IC_ADD_SPEC_PC_derived)

	def IC_ADD_BASE_PC_derived(IC_ADD_BASE_IND,AG_SYS_CODE,IC_ADD_BASE_PC):
		if (IC_ADD_BASE_IND==0) or (AG_SYS_CODE != 2):
			return 0
		else:
			return IC_ADD_BASE_PC
	def IC_ADD_SPEC_PC_derived(IC_ADD_SPEC_IND,AG_SYS_CODE,IC_ADD_SPEC_PC):
		if (IC_ADD_SPEC_IND==0) or (AG_SYS_CODE != 2):
			return 0
		else:
			return IC_ADD_SPEC_PC

	def COMM_CAP(COMM_CAP_RATE,SUM_ASSURED,PREM_FREQ):
		if COMM_CAP_RATE==999999:
			return 999999
		else:
			return COMM_CAP_RATE*SUM_ASSURED/(10*max(1,PREM_FREQ))

	def IC_BASE_PAYBL_PP(REN_T,IC_PERIOD_M,COMM_RATE_ADJ_PC,COMM_MODAL_FAC,PREM_INC_PP):
		if REN_T>IC_PERIOD_M:
			return 0
		else:
			return COMM_RATE_ADJ_PC*COMM_MODAL_FAC*PREM_INC_PP
			
	def REN_POL_YR(RENEWABLE_IND,POL_YR,POL_TERM_Y):
		if RENEWABLE_IND==0:
			return POL_YR
		else:
			return POL_YR-POL_TERM_Y*max(0,(POL_YR-1)//POL_TERM_Y)		

	def COMM_PAY_ADJ(REN_T,CP_TYPE_ADJ_patch,CP_TYPE_patch,PREM_MODE,REN_POL_YR,PREM_FREQ,):
		if ((CP_TYPE_ADJ_patch != 3) and (CP_TYPE_patch != 0))  or PREM_MODE==0 or ((CP_TYPE_patch != 0) and (REN_POL_YR>1)):
			return 1
		else:
			if REN_T==1:
				return PREM_FREQ
			else:
				return 0

	def RC_BASE_PAYBL_PP(REN_T,RC_START_M,COMM_RATE_ADJ_PC,COMM_MODAL_FAC,PREM_INC_PP):
		if REN_T<RC_START_M:
			return 0
		else:
			return COMM_RATE_ADJ_PC*COMM_MODAL_FAC*PREM_INC_PP

	def COMM_RATE_ADJ_PC(POL_YR,CP_TYPE_patch,COMM_ID,COMM_RATE_BEFORE_ROUND):
		if (((POL_YR==1) and (CP_TYPE_patch==1)) or (CP_TYPE_patch==2)) and COMM_ID<=5:
			# return round(COMM_RATE_BEFORE_ROUND,1)
			return math.floor(COMM_RATE_BEFORE_ROUND*10)/10
		else:
			return COMM_RATE_BEFORE_ROUND

	def COMM_CAP_AG_FAC(POL_YR,MAX_COMM_PAYBL_Y,CP_TYPE_patch,COMM_S_AG_FAC,REN_COMM_S_AG_FAC):
		if POL_YR>MAX_COMM_PAYBL_Y:
			return 0
		else:
			if (POL_YR==1) or (CP_TYPE_patch==2):
				return COMM_S_AG_FAC
			else:
				return 	REN_COMM_S_AG_FAC

	def BASE_COMM_CAP_RATE(POL_YR,MAX_COMM_PAYBL_Y,COMM_S_RATE,REN_COMM_S_RATE):
		if POL_YR>MAX_COMM_PAYBL_Y:
			return 0
		else:
			if POL_YR==1:
				return COMM_S_RATE
			else:
				return REN_COMM_S_RATE

	def COMM_S_AG_FAC(COMM_ID,COMM_S_AG_FAC_1):
		if COMM_ID==5:
			return COMM_S_AG_FAC_1
		else:
			return 1
	def REN_COMM_S_AG_FAC(COMM_ID,REN_COMM_S_AG_FAC_1):
		if COMM_ID==5:
			return REN_COMM_S_AG_FAC_1
		else:
			return 1

	def COMM_S_RATE(COMM_LIMIT_TYPE_derived,COMM_S_RATE_PC_1):
		if COMM_LIMIT_TYPE_derived==999999:
			return 0
		else:
			return COMM_S_RATE_PC_1/100
	def REN_COMM_S_RATE(COMM_LIMIT_TYPE_derived,REN_COMM_S_RATE_PC_1):
		if COMM_LIMIT_TYPE_derived==999999:
			return 0
		else:
			return REN_COMM_S_RATE_PC_1/100
	
	def COMM_LIMIT_TYPE_derived(COMM_LIMIT_TYPE,COMM_ID,MIN_COMM_ID):
		if COMM_ID<MIN_COMM_ID:
			return 999999
		else:
			return COMM_LIMIT_TYPE

	def COMM_CAP_RATE(POL_YR,CP_TYPE_patch,COMM_ID,COMM_LIMIT_TYPE_derived,BASE_COMM_CAP_RATE,COMM_CAP_AG_FAC,COMM_LIMIT_ADJ):
		if COMM_LIMIT_TYPE_derived==999999:
			return 999999
		else:
			if (((POL_YR==1) and ((CP_TYPE_patch==0) or (CP_TYPE_patch==1))) or (CP_TYPE_patch==2)) and COMM_ID<=5:
				# return round(BASE_COMM_CAP_RATE * COMM_CAP_AG_FAC * COMM_LIMIT_ADJ,1)
				return math.floor(BASE_COMM_CAP_RATE * COMM_CAP_AG_FAC * COMM_LIMIT_ADJ*10)/10
			else:
				return BASE_COMM_CAP_RATE * COMM_CAP_AG_FAC * COMM_LIMIT_ADJ

	def IC_INCT_PAYBL_PP(REN_T,IC_PERIOD_M,COMM_INCT_RATE_PC,COMM_MODAL_FAC,PREM_INC_PP):
		if REN_T>IC_PERIOD_M:
			return 0
		else:
			return COMM_INCT_RATE_PC*COMM_MODAL_FAC*PREM_INC_PP

	def COMM_MODAL_FAC(POL_YR,MAX_COMM_PAYBL_Y,COMM_MODAL_FAC_1,REN_COMM_MODAL_FAC_1):
		if POL_YR>MAX_COMM_PAYBL_Y:
			return 0
		else:
			if POL_YR==1:
				return COMM_MODAL_FAC_1
			else:
				return 	REN_COMM_MODAL_FAC_1

	def COMM_INCT_RATE_PC(POL_YR,MAX_COMM_PAYBL_Y,COMM_INCT_RATE_PC_1,REN_COMM_INCT_RATE_PC_1):
		if POL_YR>MAX_COMM_PAYBL_Y:
			return 0
		else:
			if POL_YR==1:
				return COMM_INCT_RATE_PC_1
			else:
				return REN_COMM_INCT_RATE_PC_1

	def COMM_RATE_BEFORE_ROUND(CP_TYPE_patch,POL_YR,COMM_ADJ_MULT,COMM_AG_FAC_1_derived,COMM_RATE_PC,COMM_RATE_ACCUM_PC):
		if CP_TYPE_patch==0:
			if POL_YR>1:
				return 0
			else:
				return COMM_RATE_ACCUM_PC
		else:
			return COMM_RATE_PC*COMM_AG_FAC_1_derived*COMM_ADJ_MULT

	def COMM_ADJ_MULT(COMM_ADJ_MULT_TYPE,POL_TERM_Y,PREM_PAYBL_Y):
		if COMM_ADJ_MULT_TYPE==1:
			if POL_TERM_Y==0:
				return -1 #Error scenario
			else:
				return PREM_PAYBL_Y / POL_TERM_Y
		else:
			return 1

	def COMM_AG_FAC_1_derived(COMM_AG_FAC_1,COMM_ID):
		if COMM_ID==5:
			return COMM_AG_FAC_1
		else:
			return 1
	def REN_COMM_AG_FAC_1_derived(REN_COMM_AG_FAC_1,COMM_ID):
		if COMM_ID==5:
			return REN_COMM_AG_FAC_1
		else:
			return 1
			
	# def COMM_AG_FAC(POL_YR,MAX_COMM_PAYBL_Y,CP_TYPE_patch,COMM_AG_FAC_1_derived,REN_COMM_AG_FAC_1_derived):
		# if POL_YR>MAX_COMM_PAYBL_Y:
			# return 0
		# else:
			# if (POL_YR==1) or (CP_TYPE_patch==2):
				# return COMM_AG_FAC_1_derived
			# else:
				# return REN_COMM_AG_FAC_1_derived
				

	def COMM_RATE_ACCUM_PC(POL_MTH,COMM_ACCUM_FAC,COMM_RATE_PC,COMM_RATE_ACCUM_PC):
		for t in range(POL_MTH.shape[0],-1,-1): #This goes backwards
			if (t+1)>POL_MTH.shape[0]:
				COMM_RATE_ACCUM_PC[t]=0
			else:
				if POL_MTH[t]==1:
					COMM_RATE_ACCUM_PC[t] = (COMM_RATE_PC[t]*COMM_ACCUM_FAC[0]) + COMM_RATE_ACCUM_PC[t+1]
				else:
					COMM_RATE_ACCUM_PC[t] = COMM_RATE_ACCUM_PC[t+1]

	# def COMM_RATE_ACCUM_PC(POL_MTH,COMM_ACCUM_FAC,COMM_RATE_PC,COMM_RATE_ACCUM_PC):
		# for i in prange(POL_MTH.shape[0]):
			# for t in range(POL_MTH.shape[1],-1,-1): #This goes backwards
				# if (t+1)>POL_MTH.shape[1]:
					# COMM_RATE_ACCUM_PC[i,t]=0
				# else:
					# if POL_MTH[i,t]==1:
						# COMM_RATE_ACCUM_PC[i,t] = (COMM_RATE_PC[i,t]*COMM_ACCUM_FAC[i,0]) + COMM_RATE_ACCUM_PC[i,t+1]
					# else:
						# COMM_RATE_ACCUM_PC[i,t] = COMM_RATE_ACCUM_PC[i,t+1]


	def SP_IND(PREM_FREQ):
		if PREM_FREQ==0:
			return 1
		else:
			return 0

	def COMM_RATE_PC(POL_YR,MAX_COMM_PAYBL_Y,COMM_RATE_PC_1,REN_COMM_RATE_PC_1):
		if POL_YR>MAX_COMM_PAYBL_Y:
			return 0
		else:
			if POL_YR==1:
				return COMM_RATE_PC_1/100
			else:
				return REN_COMM_RATE_PC_1/100

	def SPIKE_LAPSE_PERIOD2(SPIKE_RATE_IND,SPIKE_LAPSE_PERIOD):
		if (SPIKE_RATE_IND==0):
			return 0
		else:
			return SPIKE_LAPSE_PERIOD
	
	def LAPSE_SPIKE_RATE(SPIKE_RATE_BE,SPIKE_RATE_IND,POL_YR,PREM_PAYBL_Y,REMAINING_PREMIUM_TERM,SPIKE_LAPSE_PERIOD2):
		if (SPIKE_RATE_IND==0) or (POL_YR>PREM_PAYBL_Y+1):
			return 0
		else:
			if (SPIKE_LAPSE_PERIOD2>REMAINING_PREMIUM_TERM):
				if (REMAINING_PREMIUM_TERM<-1) or (REMAINING_PREMIUM_TERM>4):
					return 0
				else:
					return SPIKE_RATE_BE
			else:
				return 0

	def DISC_PREM_INC(T
			, DURATIONIF_M,REN_POL_TERM_Y,ANN_TERM_Y
			, M_DISC_RATE
			, PPIA_OFFSET_IF, PPIA_INC_IF, PREM_INC
			, DISC_PREM_INC):
		for t in range(T.shape[0],-1,-1):#Note this counts down
			if t==T.shape[0]:
				DISC_PREM_INC[t]=0
			else:
				if (T[t]<DURATIONIF_M[0]) or (T[t]>(REN_POL_TERM_Y[0]+ANN_TERM_Y[0])*12) or (M_DISC_RATE[0]==-1):
					DISC_PREM_INC[t]=0
				else:
					DISC_PREM_INC[t]=PPIA_OFFSET_IF[t+1]+PPIA_INC_IF[t+1]+PREM_INC[t+1]+DISC_PREM_INC[t+1]/(1+M_DISC_RATE[0])

	# def DISC_PREM_INC(T
			# , DURATIONIF_M,REN_POL_TERM_Y,ANN_TERM_Y
			# , M_DISC_RATE
			# , PPIA_OFFSET_IF, PPIA_INC_IF, PREM_INC
			# , DISC_PREM_INC):
		# for i in prange(T.shape[0]):
			# for t in range(T.shape[1],-1,-1):#Note this counts down
				# if t==T.shape[1]:
					# DISC_PREM_INC[i,t]=0
				# else:
					# if (T[i,t]<DURATIONIF_M[i,0]) or (T[i,t]>(REN_POL_TERM_Y[i,0]+ANN_TERM_Y[i,0])*12) or (M_DISC_RATE[0,0]==-1):
						# DISC_PREM_INC[i,t]=0
					# else:
						# DISC_PREM_INC[i,t]=PPIA_OFFSET_IF[i,t+1]+PPIA_INC_IF[i,t+1]+PREM_INC[i,t+1]+DISC_PREM_INC[i,t+1]/(1+M_DISC_RATE[0,0])


	def PPIA_OFFSET_IF(T,DURATIONIF_M,PPIA_OFFSET_RATE,SUM_ASSURED2,NO_POLS_IFSM):
		if T< DURATIONIF_M:
			return 0
		else:
			return PPIA_OFFSET_RATE*SUM_ASSURED2*NO_POLS_IFSM

	def PPIA_OFFSET_RATE(T,PPIA_IND,PPIA_TERM_M,POL_YR,REN_POL_TERM_Y,SUM_ASSURED2,PREM_INC_PP):
		#This complains about a runtime warning, but I can't figure out where it comes from
		if (PPIA_IND==0) or (T>PPIA_TERM_M) or (T<=1) or (POL_YR>REN_POL_TERM_Y):
			return 0
		else:
			if SUM_ASSURED2==0:
				return 0
			else:
				return -PREM_INC_PP/SUM_ASSURED2

	def PPIA_INC_IF(T,DURATIONIF_M, PPIA_INCOME_RATE,SUM_ASSURED2,NO_POLS_IFSM):
		if T< DURATIONIF_M:
			return 0
		else:
			return PPIA_INCOME_RATE*SUM_ASSURED2*NO_POLS_IFSM
	
	def NO_POLS_IFSM(NO_HTHY_POLS_IFSM,NO_WP_POLS_IFSM):
		return NO_HTHY_POLS_IFSM+NO_WP_POLS_IFSM

	def PPIA_INCOME_RATE(T,SUM_ASSURED2,PREPAID_PREMIUM_PRESENT_VALUE,PREM_INC_PP):
		#This complains about a runtime warning, but I can't figure out where it comes from
		if T>1:
			return 0
		elif SUM_ASSURED2==0:
			return 0
		else:
			return PREPAID_PREMIUM_PRESENT_VALUE*PREM_INC_PP/SUM_ASSURED2

	def PREPAID_PREMIUM_PRESENT_VALUE(PPIA_NO_TIME,PPIA_DISCOUNT_RATE):
		if PPIA_DISCOUNT_RATE==0:
			return 0
		else:
			return ((1-(1+PPIA_DISCOUNT_RATE))**(1-PPIA_NO_TIME))/PPIA_DISCOUNT_RATE

	def PPIA_DISCOUNT_RATE(PPIA_DISC_PC,PREM_FREQ):
		if PREM_FREQ==0:
			return 0
		else:
			return (1+PPIA_DISC_PC/100)**(1/PREM_FREQ)-1
	
	# def COMPANY_YEAR(FISCAL_YEAR,min_ppia_pc_lkup_yr,max_ppia_pc_lkup_yr,COMPANY_YEAR):
		# for i in prange(COMPANY_YEAR.shape[0]):
			# COMPANY_YEAR[i,0]=min(max(FISCAL_YEAR[i,0],min_ppia_pc_lkup_yr[0,0]),max_ppia_pc_lkup_yr[0,0])

	def COMPANY_YEAR(FISCAL_YEAR,min_ppia_pc_lkup_yr,max_ppia_pc_lkup_yr,COMPANY_YEAR):
		COMPANY_YEAR[0]=min(max(FISCAL_YEAR[0],min_ppia_pc_lkup_yr[0]),max_ppia_pc_lkup_yr[0])


	def FISCAL_YEAR(CALENDAR_YEAR,CALENDAR_MONTH,FISCAL_MONTH):
		if FISCAL_MONTH>=CALENDAR_MONTH:
			return CALENDAR_YEAR-1
		else:
			return CALENDAR_YEAR
	def FISCAL_MONTH(ENTRY_MONTH,COM_YEAR_END,T):
		return 1+(ENTRY_MONTH-1-COM_YEAR_END+T+23)%12
	def CALENDAR_MONTH(ENTRY_MONTH,T):
		return ((ENTRY_MONTH+T+10)%12)+1
	def CALENDAR_YEAR(T,ENTRY_YEAR,ENTRY_MONTH):
		return (ENTRY_YEAR*12+ENTRY_MONTH+T-2)//12

	def PREM_INC(PREM_INC_PP,T,DURATIONIF_M,NO_HTHY_POLS_IFSM):
		if T<=DURATIONIF_M:
			return 0
		else:
			return PREM_INC_PP*NO_HTHY_POLS_IFSM

	def PREM_INC_PP(ANN_PREM_PP,PREM_FRAC_PP):
		return ANN_PREM_PP*PREM_FRAC_PP

	def ANN_PREM_PP(PROD_CODE, PREM_STEP_DUR, POL_YR, PREM_STEP_RATIO):
	#	this is fake code to avoid commutation
		if PROD_CODE==1221:
			fake_annual_prem=0.1122
		elif PROD_CODE==1212:
			fake_annual_prem=0.177588
		elif PROD_CODE==1211:
			fake_annual_prem=2.444875
		elif PROD_CODE==1207:
			fake_annual_prem=0.017688
		else:
			fake_annual_prem=0
		if (POL_YR>PREM_STEP_DUR) and (PREM_STEP_RATIO!=0):
			return fake_annual_prem
		else:
			return fake_annual_prem
		

	def REN_T(RENEWABLE_IND,T,POL_TERM_Y):
		if RENEWABLE_IND==0:
			return T
		else:
			if POL_TERM_Y==0:
				return 0
			else:
				return (T-1)%(POL_TERM_Y*12)+1

	def PREM_FRAC_PP(PREM_FREQ,T,REN_POL_TERM_Y,REN_T,PREM_PAYBL_Y):
		if PREM_FREQ==0:
			if T==1:
				return 1
			else:
				return 0
		else:
			if T>(REN_POL_TERM_Y*12 + 1):
				return 0
			else:
				if (REN_T==((PREM_PAYBL_Y*12)//(12/PREM_FREQ))*(12/PREM_FREQ)+1) or (T==(REN_POL_TERM_Y*12)+1):
					return max(PREM_PAYBL_Y*12-((REN_T-1)//((12/PREM_FREQ))+1)*12/PREM_FREQ,0)/12
				else:
					if (REN_T<=(PREM_PAYBL_Y*12)) and ((((T+11)*PREM_FREQ)%12)==0):
						return 1/PREM_FREQ
					else:
						return 0
					

	def SUM_ASSURED2(T,REN_POL_TERM_Y,ANN_TERM_Y,SUM_ASSURED):
		#THERE'S SOMETHING COMPLICATED GOINNG ON IN THE SPREADSHEET THAT DIVES IN TO COMMUTATION. I'M GOING TO PAUSE THIS TRAIN FOR NOW
		if T>(REN_POL_TERM_Y+ANN_TERM_Y)*12:
			return 0
		else:
			return SUM_ASSURED

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

	# def WP_POLICIES(T,MATURITY_RATE,DEATH_RATE_MONTHLY, LAPSE_RATE_MONTHLY, NO_HTHY_WP
			# ,REN_POL_TERM_Y,ANN_TERM_Y,PREM_WAIVER_IND,DURATIONIF_M
			# ,NO_WP_POLS_IFSM,NO_WP_DEATHS,NO_WP_SURRS,NO_WP_POLS_IF,NO_WP_MATS):
		# NO_WP_POLS_IF_start=np.zeros(T.shape[0],dtype=float64)
		# for i in prange(T.shape[0]):
			# for t in range(T.shape[1]):
				# if t==0:
					# if PREM_WAIVER_IND[i,0]==1:
						# NO_WP_POLS_IF_start[i]=1
					# else:
						# NO_WP_POLS_IF_start[i]=0
					# NO_WP_MATS[i,t]=NO_WP_POLS_IF_start[i] * MATURITY_RATE[i,t]
					# NO_WP_POLS_IFSM[i,t]=NO_WP_POLS_IF_start[i]
				# else:
					# NO_WP_MATS[i,t]=NO_WP_POLS_IF[i,t-1] * MATURITY_RATE[i,t]
					# NO_WP_POLS_IFSM[i,t]=NO_WP_POLS_IF[i,t-1] - NO_WP_MATS[i,t]
				
				# NO_WP_DEATHS[i,t]=NO_WP_POLS_IFSM[i,t] * DEATH_RATE_MONTHLY[i,t]
				# NO_WP_SURRS[i,t]=NO_WP_POLS_IFSM[i,t] * (1-DEATH_RATE_MONTHLY[i,t]) * LAPSE_RATE_MONTHLY[i,t]
				# if T[i,t]>(REN_POL_TERM_Y[i,0]+ANN_TERM_Y[i,0])*12:
					# NO_WP_POLS_IF[i,t]=0
				# else:
					# if T[i,t]<=DURATIONIF_M[i,0]:
						# NO_WP_POLS_IF[i,t]=NO_WP_POLS_IF_start[i]
					# else:
						# NO_WP_POLS_IF[i,t]=NO_WP_POLS_IFSM[i,t]-(NO_WP_DEATHS[i,t]+NO_WP_SURRS[i,t])+NO_HTHY_WP[i,t]


	def HEALTHY_POLICIES(REN_POL_TERM_Y,ANN_TERM_Y,DURATIONIF_M,PREM_WAIVER_IND
			,T,MATURITY_RATE,DEATH_RATE_MONTHLY,LAPSE_RATE_MONTHLY,DECREMENT_WOP_RATES_MONTHLY
			,NO_HTHY_POLS_IFSM,NO_HTHY_DEATHS,NO_HTHY_SURRS,NO_HTHY_WP,NO_HTHY_POLS_IF,NO_HTHY_MATS):
		# NO_HTHY_POLS_IF_start=np.zeros(T.shape[0],dtype=float64)
		for t in range(T.shape[0]):
			if t==0:
				if PREM_WAIVER_IND[0]==1:
					NO_HTHY_POLS_IF_start=0
				else:
					NO_HTHY_POLS_IF_start=1
				NO_HTHY_MATS[t]=NO_HTHY_POLS_IF_start * MATURITY_RATE[t]
				NO_HTHY_POLS_IFSM[t]=NO_HTHY_POLS_IF_start
			else:
				NO_HTHY_MATS[t]=NO_HTHY_POLS_IF[t-1] * MATURITY_RATE[t]
				NO_HTHY_POLS_IFSM[t]=NO_HTHY_POLS_IF[t-1] - NO_HTHY_MATS[t]
				
			NO_HTHY_DEATHS[t]=NO_HTHY_POLS_IFSM[t]*DEATH_RATE_MONTHLY[t]
			NO_HTHY_SURRS[t]=NO_HTHY_POLS_IFSM[t]*(1-DEATH_RATE_MONTHLY[t])*LAPSE_RATE_MONTHLY[t]
			NO_HTHY_WP[t]=(NO_HTHY_POLS_IFSM[t]-NO_HTHY_DEATHS[t]-NO_HTHY_SURRS[t])*DECREMENT_WOP_RATES_MONTHLY[t]
			if T[t]>(REN_POL_TERM_Y[0]+ANN_TERM_Y[0])*12:
				NO_HTHY_POLS_IF[t]=0
			else:
				if T[t]<=DURATIONIF_M[0]:
					NO_HTHY_POLS_IF[t]=NO_HTHY_POLS_IF_start
				else:	
					NO_HTHY_POLS_IF[t]=NO_HTHY_POLS_IFSM[t]-(NO_HTHY_DEATHS[t]+NO_HTHY_SURRS[t]+NO_HTHY_WP[t])

	# def HEALTHY_POLICIES(REN_POL_TERM_Y,ANN_TERM_Y,DURATIONIF_M,PREM_WAIVER_IND
			# ,T,MATURITY_RATE,DEATH_RATE_MONTHLY,LAPSE_RATE_MONTHLY,DECREMENT_WOP_RATES_MONTHLY
			# ,NO_HTHY_POLS_IFSM,NO_HTHY_DEATHS,NO_HTHY_SURRS,NO_HTHY_WP,NO_HTHY_POLS_IF,NO_HTHY_MATS):
		# NO_HTHY_POLS_IF_start=np.zeros(T.shape[0],dtype=float64)
		# for i in prange(T.shape[0]):
			# for t in range(T.shape[1]):
				# if t==0:
					# if PREM_WAIVER_IND[i,0]==1:
						# NO_HTHY_POLS_IF_start[i]=0
					# else:
						# NO_HTHY_POLS_IF_start[i]=1
					# NO_HTHY_MATS[i,t]=NO_HTHY_POLS_IF_start[i] * MATURITY_RATE[i,t]
					# NO_HTHY_POLS_IFSM[i,t]=NO_HTHY_POLS_IF_start[i]
				# else:
					# NO_HTHY_MATS[i,t]=NO_HTHY_POLS_IF[i,t-1] * MATURITY_RATE[i,t]
					# NO_HTHY_POLS_IFSM[i,t]=NO_HTHY_POLS_IF[i,t-1] - NO_HTHY_MATS[i,t]
					
				# NO_HTHY_DEATHS[i,t]=NO_HTHY_POLS_IFSM[i,t]*DEATH_RATE_MONTHLY[i,t]
				# NO_HTHY_SURRS[i,t]=NO_HTHY_POLS_IFSM[i,t]*(1-DEATH_RATE_MONTHLY[i,t])*LAPSE_RATE_MONTHLY[i,t]
				# NO_HTHY_WP[i,t]=(NO_HTHY_POLS_IFSM[i,t]-NO_HTHY_DEATHS[i,t]-NO_HTHY_SURRS[i,t])*DECREMENT_WOP_RATES_MONTHLY[i,t]
				# if T[i,t]>(REN_POL_TERM_Y[i,0]+ANN_TERM_Y[i,0])*12:
					# NO_HTHY_POLS_IF[i,t]=0
				# else:
					# if T[i,t]<=DURATIONIF_M[i,0]:
						# NO_HTHY_POLS_IF[i,t]=NO_HTHY_POLS_IF_start[i]
					# else:	
						# NO_HTHY_POLS_IF[i,t]=NO_HTHY_POLS_IFSM[i,t]-(NO_HTHY_DEATHS[i,t]+NO_HTHY_SURRS[i,t]+NO_HTHY_WP[i,t])


	def MATURITY_RATE(T,REN_POL_TERM_Y,ANN_TERM_Y,POL_TERM_Y,RENEWABLE_IND,REN_RATE,ANN_RATE):
		if T==(REN_POL_TERM_Y+ANN_TERM_Y)*12+1:
			return 1
		else:
			if POL_TERM_Y==0:
				return 0
			if ((T>POL_TERM_Y*12) and (T%(POL_TERM_Y*12)==1) and (RENEWABLE_IND==1)) or (T==(REN_POL_TERM_Y*12+1)):
				return 1-REN_RATE-ANN_RATE
			else:
				return 0


	def DECREMENT_WOP_RATES_MONTHLY(DECREMENT_WOP_RATES_ANNUAL):
		return 1-(1-DECREMENT_WOP_RATES_ANNUAL)**(1/12)

	def DECREMENT_WOP_RATES_ANNUAL(T, DURATIONIF_M, WP_RIDER_IND, DEC_WOP_IND, BASE_DECREMENT_WOP_RATE, DECREMENT_WOP_EXPERIENCE, DEC_WOP_PAD_PC):
		if T<=DURATIONIF_M:
			return 0
		else:
			if (WP_RIDER_IND==0) and (DEC_WOP_IND==0):
				return 0
			else:
				return BASE_DECREMENT_WOP_RATE * DECREMENT_WOP_EXPERIENCE * (1+DEC_WOP_PAD_PC)

	def DECREMENT_WOP_EXPERIENCE(DEC_WOP_CODE, POL_YR, REN_POL_TERM_Y, ANN_TERM_Y, WOP_EXP_FAC_BE):
		if DEC_WOP_CODE==999999:
			return 0
		else:
			if POL_YR>(REN_POL_TERM_Y+ANN_TERM_Y):
				return 0
			else:
				return WOP_EXP_FAC_BE/100

	def WOP_EXP_FAC_BE_col(POL_YR, MAX_EXP_LKUP_Y):
		return min(POL_YR,MAX_EXP_LKUP_Y)
		

	def WOP_EXP_SEX_IDX(DEC_WOP_EXP_SEX_IND,SEX):
		if DEC_WOP_EXP_SEX_IND==0:
			return 999999
		else:
			return SEX

	def BASE_DECREMENT_WOP_RATE(WOP_RATE_BE, POL_YR, REN_POL_TERM_Y, ANN_TERM_Y, DEC_WOP_CODE):
		if DEC_WOP_CODE==999999:
			return 0
		elif POL_YR>(REN_POL_TERM_Y+ANN_TERM_Y):
			return 0
		else:
			return WOP_RATE_BE

	def WOP_SEX_IDX(DEC_WOP_SEX_IND, SEX):
		if DEC_WOP_SEX_IND==1:
			return SEX
		else:
			return 999999

	def MORT_RATE_MAX_AGE(SEX):
		if SEX==0:
			return 106
		else:
			return 109

	def DEATH_RATE_MONTHLY(DEATH_RATE_ANNUAL,MORT_RATE_MAX_AGE,ATTAINED_AGE,POL_MTH):
		if MORT_RATE_MAX_AGE==ATTAINED_AGE:
			#THIS IS REALLY HACKY! WHOEVER IS LEFT IS KILLED OFF!
			extra_factor=(1-POL_MTH/12)
		else:
			extra_factor=1
		return (1-(1-DEATH_RATE_ANNUAL)**(1/12))*extra_factor

	def DEATH_RATE_ANNUAL(BASE_DEATH_RATES, MORTALITY_SELECTION_FACTOR, SUBS_PC, T, MORT_PAD_PC, DURATIONIF_M):
		if T<=DURATIONIF_M:
			return 0
		else:
			return BASE_DEATH_RATES*MORTALITY_SELECTION_FACTOR*SUBS_PC*(1+MORT_PAD_PC/100)/100
	
	def BASE_DEATH_RATES(MORT_RATE_BE, POL_YR, REN_POL_TERM_Y, ANN_TERM_Y):
		if POL_YR > (REN_POL_TERM_Y + ANN_TERM_Y):
			return 0
		else:
			return MORT_RATE_BE
	
	def MORT_SMK_IDX(JUV_IND,MORT_SMK_IND, SMOKER_STAT):
		if MORT_SMK_IND==0:
			return 999999
		else:
			return SMOKER_STAT
			
	def MORT_SEL_SEX_IDX(JUV_IND,MORT_SEX_IND, SEX2, SEX):
		if MORT_SEX_IND==0:
			return 999999
		else:
			if JUV_IND==1:
				return SEX2
			else:
				return SEX
	def MORT_SEX_IDX(JUV_IND,MORT_SEX_IND, SEX2, SEX):
	#THIS HAS A CMT LOOKUP IN THE BE SECTION IN THE EXCEL. IS THIS A MISTAKE?!
		if MORT_SEX_IND==0:
			return 999999
		else:
			if JUV_IND==1:
				return SEX2
			else:
				return SEX
	
	def ANN_TERM_Y(PROD_CODE,k_term,DTH_ANN_GUAR,AGE_AT_ENTRY,POL_TERM_Y):
		if PROD_CODE==1208:
			if k_term==99:
				return max(DTH_ANN_GUAR,80-(AGE_AT_ENTRY+POL_TERM_Y))
			else:
				return DTH_ANN_GUAR
		else:
			return 0
	
	def MORTALITY_SELECTION_FACTOR(MORT_SEL_FAC_BE, POL_YR, REN_POL_TERM_Y, ANN_TERM_Y):
		if POL_YR > (REN_POL_TERM_Y + ANN_TERM_Y):
			return 0
		else:
			return MORT_SEL_FAC_BE/100

	def MORTALITY_SELECTION_FACTOR_col(BASE_POL_TERM_Y, RENEWAL_COUNT, max_mort_sel_lkup_yr, POL_YR):
		return max(
			min(POL_YR+BASE_POL_TERM_Y*RENEWAL_COUNT,max_mort_sel_lkup_yr)
				,1)
		
	def LAPSE_PAD_PC(LAPSE_PAD_CODE):
		if LAPSE_PAD_CODE == 999999:
			return 0
		else:
		#THIS CODE IS BROKEN IN THE SPREADSHEET
			return 0

	def LAPSE_RATE_MONTHLY(LAPSE_RATE_ANNUAL):
		return 1-(1-LAPSE_RATE_ANNUAL)**(1/12)
	
	def LAPSE_RATE_ANNUAL(T,DURATIONIF_M,SPIKE_ADJUSTED_LAPSE_RATE, LAPSE_PAD_PC):
		if T<=DURATIONIF_M:
			return 0
		else:
			return SPIKE_ADJUSTED_LAPSE_RATE * (1 + LAPSE_PAD_PC/100)
	
	def SPIKE_ADJUSTED_LAPSE_RATE(SPIKE_LAPSE_PERIOD2,LAPSE_SPIKE_RATE,REMAINING_PREMIUM_TERM,SPIKE_RATE_IND,POL_YR,REN_POL_TERM_Y,DOWN_PAYMENT,RENEWAL_COUNT,BASE_LAPSE_RATES):
		#Bug in down_payment in the excel
		if (SPIKE_RATE_IND==0) or (POL_YR>REN_POL_TERM_Y) or ((DOWN_PAYMENT==1) and (RENEWAL_COUNT==0)):
			return BASE_LAPSE_RATES
		else:
			if REMAINING_PREMIUM_TERM==-1:
				return LAPSE_SPIKE_RATE
			else:
				if (REMAINING_PREMIUM_TERM<SPIKE_LAPSE_PERIOD2) and (REMAINING_PREMIUM_TERM>-1):
					if SPIKE_LAPSE_PERIOD2 !=0:
						return BASE_LAPSE_RATES-(BASE_LAPSE_RATES-LAPSE_SPIKE_RATE)*(1-REMAINING_PREMIUM_TERM/SPIKE_LAPSE_PERIOD2)
					else:
						return BASE_LAPSE_RATES
				else:
					return BASE_LAPSE_RATES

	def BASE_LAPSE_RATES(LAPSE_CODE, POL_YR, REN_POL_TERM_Y
				, LAPSE_RATE_BE):
		if (LAPSE_CODE==999999) or (POL_YR > REN_POL_TERM_Y):
			return 0
		else:
			return LAPSE_RATE_BE/100
	
	def LAPSE_PREMIUM_FREQ_IDX(PREM_FREQ,LAPSE_PREM_FREQ_IND):
		if LAPSE_PREM_FREQ_IND==0:
			return 999999
		else:
			return PREM_FREQ

	def LAPSE_CHAN_IDX(LAPSE_CHAN_IND):
		if LAPSE_CHAN_IND==0:
			return 999999
		else:
			return LAPSE_CHAN_IND
	
	def PREM_FREQ(PREM_MODE):
		if PREM_MODE==0:
			return 0
		elif PREM_MODE==1:
			return 1
		elif PREM_MODE in [2,3]:
			return 2
		elif PREM_MODE in [4,5,6]:
			return 12
		else:
			return -1 #Error value
	 		
	def MORT_AGE_col(JUV_IND,ATTAINED_AGE,ATTAINED_AGE_Y):
		if JUV_IND==1:
			return ATTAINED_AGE_Y
		else:
			return ATTAINED_AGE
			
	def POLICY_YR_RESET_AT_RENEWAL(POL_YR, POL_TERM_Y):
		if POL_TERM_Y==0:
			return 0
		else:
			return ((POL_YR+POL_TERM_Y-1)%POL_TERM_Y)+1

			