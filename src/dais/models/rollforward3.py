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

class RFwdFunctions():
	single_derivations=dict(
		MONTH_T = {'type': int32, 'alignment':'', 'numba':'v','shape':2,'loop':1}
		, YEAR_T = {'type': int32, 'alignment':'', 'numba':'v','shape':2,'loop':1}
		# , cEQUITY_RET_RATE = {'type': float64, 'alignment':'C', 'numba':'j','shape':2,'loop':1}
		
		, pcEQUITY_RET = {'type': float64, 'alignment':'PC', 'numba':'v','shape':2,'loop':1}
		, pTOTAL_FAV = {'type': float64, 'alignment':'P', 'numba':'v','shape':2,'loop':1}
		, rfpBOX = {'type': float64, 'alignment':'P', 'numba':'v','shape':2,'loop':1}
		, pCASH_RET = {'type': float64, 'alignment':'P', 'numba':'v','shape':2,'loop':1}
		, new_money_temp = {'type': float64, 'alignment':'P', 'numba':'v','shape':2,'loop':1}
		, cNEW_MONEY_TO_BOND = {'type': float64, 'alignment':'C', 'numba':'v','shape':2,'loop':1}
		, cNEW_MONEY_TO_EQUITY = {'type': float64, 'alignment':'C', 'numba':'v','shape':2,'loop':1}
		, pTOTAL_REALIGN = {'type': float64, 'alignment':'P', 'numba':'v','shape':2,'loop':1}
		, rfpREALIGN_TOTAL = {'type': float64, 'alignment':'P', 'numba':'v','shape':2,'loop':1}
		
		, pcBOND_TARGET = {'type': float64, 'alignment':'PC', 'numba':'v','shape':2,'loop':1}
		, pcEQUITY_TARGET = {'type': float64, 'alignment':'PC', 'numba':'v','shape':2,'loop':1}
		, pcREBAL_BUY = {'type': float64, 'alignment':'PC', 'numba':'v','shape':2,'loop':1}
		, pcREBAL_SELL = {'type': float64, 'alignment':'PC', 'numba':'v','shape':2,'loop':1}
		, pREBAL_CASH = {'type': float64, 'alignment':'P', 'numba':'v','shape':2,'loop':1}
		, pcBOND_REALIGN_TARGET = {'type': float64, 'alignment':'PC', 'numba':'v','shape':2,'loop':1}
		, rfpcEQUITY_REALIGN_TARGET = {'type': float64, 'alignment':'PC', 'numba':'v','shape':2,'loop':1}
		, pcREALIGN_BUY = {'type': float64, 'alignment':'PC', 'numba':'v','shape':2,'loop':1}
		, pcREALIGN_SELL = {'type': float64, 'alignment':'PC', 'numba':'v','shape':2,'loop':1}
		, pcyREBAL_BUYSELL_PCTS = {'type': float64, 'alignment':'PCY', 'numba':'v','shape':2,'loop':1}
		, rfpcyREALIGN_BUYSELL_PCTS = {'type': float64, 'alignment':'PCY', 'numba':'v','shape':2,'loop':1}
		
		, pcyREBAL_MV = {'type': float64, 'alignment':'PCY', 'numba':'v','shape':2,'loop':1}
		, pcyREALIGN_MV = {'type': float64, 'alignment':'PCY', 'numba':'v','shape':2,'loop':1}
		, pcyREBAL_FAV = {'type': float64, 'alignment':'PCY', 'numba':'v','shape':2,'loop':1}
		, pcyREALIGN_FAV = {'type': float64, 'alignment':'PCY', 'numba':'v','shape':2,'loop':1}
		, pcREBAL_REALIZED_GAINLOSS = {'type': float64, 'alignment':'PC', 'numba':'v','shape':2,'loop':1}
		, pcREALIGN_REALIZED_GAINLOSS = {'type': float64, 'alignment':'PC', 'numba':'v','shape':2,'loop':1}
		, pcREALIGN_UNREALIZED_GAINLOSS = {'type': float64, 'alignment':'PC', 'numba':'v','shape':2,'loop':1}
		, pcBOND_FAV_INV_RETURN = {'type': float64, 'alignment':'PC', 'numba':'v','shape':2,'loop':1}
		, pcEQUITY_FAV_INV_RETURN = {'type': float64, 'alignment':'PC', 'numba':'v','shape':2,'loop':1}
		, pFAV_INV_RET_RATE = {'type': float64, 'alignment':'P', 'numba':'v','shape':2,'loop':1}
		, pBOX_INC = {'type': float64, 'alignment':'P', 'numba':'v','shape':2,'loop':1}
		
		# , pcyBUYSELL_PCTS = {'type': float64, 'alignment':'PCY', 'numba':'j','loop_location':'start','shape':2,'loop':1}
		, pcyBOND_FAV = {'type': float64, 'alignment':'PCY', 'numba':'v','shape':2,'loop':1}
		# , rfpcyBOND_FAV_monthly = {'type': float64, 'alignment':'PCY', 'numba':'j','loop_location':'start','shape':2,'loop':1,'monthly':True}
		, pcyBOND_MV = {'type': float64, 'alignment':'PCY', 'numba':'v','shape':2,'loop':1}
		 # ,pcyFAV_monthly = {'type': float64, 'alignment':'PCY', 'numba':'j','shape':2,'loop':1,'monthly':True}
		, pcyBOND_CF = {'type': float64, 'alignment':'PCY', 'numba':'v','shape':2,'loop':1}
		, rfpREALIGN_CASH = {'type': float64, 'alignment':'P', 'numba':'v','shape':2,'loop':1}
		, pcBOND_INV_EXP = {'type': float64, 'alignment':'PC', 'numba':'v','shape':2,'loop':1}
		, pcEQUITY_INV_EXP = {'type': float64, 'alignment':'PC', 'numba':'v','shape':2,'loop':1}
		, pTOTAL_FAV_INV_RETURN = {'type': float64, 'alignment':'P', 'numba':'v','shape':2,'loop':1}
		
		# , rfpREALIGN_CASH = {'type': float64, 'alignment':'P', 'numba':'v','shape':2,'loop':1}

		, ones = {'type':float64, 'alignment':'PCY', 'numba':'v','shape':2}
		, zeros = {'type':float64, 'alignment':'PCY', 'numba':'v','shape':2}
		, pzeros = {'type':float64, 'alignment':'P', 'numba':'v','shape':2}
	)
	
	complex_derivations={}
	
	summaries={}
	
	mappings={
		'EQUITY_I_FAV': {'source':'ASSETS_EQUITY', 'alignment':'PC', 'type':float64, 'shape':1, 'mapping':{'POOL':'POOL','CATEGORY':'CATEGORY'}}
		, 'BOND_DRIFT_UPPER_PC': {'source':'STRATEGY_STATIC', 'alignment':'C','type':float64, 'shape':1, 'mapping':{'POOL':'POOL','CATEGORY':'CATEGORY'}}
		, 'BOND_DRIFT_LOWER_PC': {'source':'STRATEGY_STATIC', 'alignment':'C','type':float64, 'shape':1, 'mapping':{'POOL':'POOL','CATEGORY':'CATEGORY'}}
		, 'EQUITY_DRIFT_UPPER_PC': {'source':'STRATEGY_STATIC', 'alignment':'C','type':float64, 'shape':1, 'mapping':{'POOL':'POOL','CATEGORY':'CATEGORY'}}
		, 'EQUITY_DRIFT_LOWER_PC': {'source':'STRATEGY_STATIC', 'alignment':'C','type':float64, 'shape':1, 'mapping':{'POOL':'POOL','CATEGORY':'CATEGORY'}}
		, 'BOND_NEW_MONEY_P_PC': {'source':'STRATEGY_STATIC', 'alignment':'C','type':float64, 'shape':1, 'mapping':{'POOL':'POOL','CATEGORY':'CATEGORY'}}
		, 'BOND_NEW_MONEY_N_PC': {'source':'STRATEGY_STATIC', 'alignment':'C','type':float64, 'shape':1, 'mapping':{'POOL':'POOL','CATEGORY':'CATEGORY'}}
		, 'EQUITY_NEW_MONEY_P_PC': {'source':'STRATEGY_STATIC', 'alignment':'C','type':float64, 'shape':1, 'mapping':{'POOL':'POOL','CATEGORY':'CATEGORY'}}
		, 'EQUITY_NEW_MONEY_N_PC': {'source':'STRATEGY_STATIC', 'alignment':'C','type':float64, 'shape':1, 'mapping':{'POOL':'POOL','CATEGORY':'CATEGORY'}}
		, 'CASH_NOMINAL': {'source':'ASSETS_CASH', 'alignment':'P','type':float64, 'shape':1, 'mapping':{'POOL':'POOL'}}
		, 'MAINT_EXP_PC': {'source':'STRATEGY_STATIC_POOL', 'alignment':'P','type':float64, 'shape':1, 'mapping':{'POOL':'POOL'}}
		, 'MV_init': {'source':'MV_init','alignment':'PCY', 'type':float64, 'shape':1, 'mapping':{'POOL':'POOL','CATEGORY':'CATEGORY','PURCHASE_YEAR_CAPPED':'PURCHASE_YEAR_CAPPED'}}
		, 'FAV_init': {'source':'FAV_init','alignment':'PCY', 'type':float64, 'shape':1, 'mapping':{'POOL':'POOL','CATEGORY':'CATEGORY','PURCHASE_YEAR_CAPPED':'PURCHASE_YEAR_CAPPED'}}
		
		# , 'EQUITY_RET_IDX': {'alignment':'C', 'type':float64, 'shape':2, 'mapping':{'CATEGORY':'CATEGORY','col':'rfYEAR'}}
		, 'EQUITY_RET_RATE': {'alignment':'C', 'type':float64, 'shape':2, 'mapping':{'CATEGORY':'CATEGORY','col':'rfMONTH'}}
		, 'BONDS_CATEGORY_MIX': {'alignment':'PC', 'type':float64, 'shape':2, 'mapping':{'POOL':'POOL','CATEGORY':'CATEGORY','col':'rfYEAR'}}
		, 'EQUITY_CATEGORY_MIX': {'alignment':'PC', 'type':float64, 'shape':2, 'mapping':{'POOL':'POOL','CATEGORY':'CATEGORY','col':'rfYEAR'}}
		, 'MATH_RES_IF_PL': {'alignment':'P', 'type':float64, 'shape':2, 'monthly': False, 'mapping':{'POOL':'POOL','col':'rfMONTH'}}
		, 'DA_FUND_FL': {'alignment':'P', 'type':float64, 'shape':2, 'monthly': False, 'mapping':{'POOL':'POOL','col':'rfMONTH'}}
		, 'CA_FUND_FL': {'alignment':'P', 'type':float64, 'shape':2, 'monthly': False, 'mapping':{'POOL':'POOL','col':'rfMONTH'}}
		, 'SOLV_MARG_IF_FL': {'alignment':'P', 'type':float64, 'shape':2, 'monthly': False, 'mapping':{'POOL':'POOL','col':'rfMONTH'}}
		, 'CASH_RET_PC': {'alignment':'P', 'type':float64, 'shape':2, 'mapping':{'POOL':'POOL','col':'rfYEAR'}}
		, 'LIAB_CF_PL': {'alignment':'P', 'type':float64, 'shape':2, 'mapping':{'POOL':'POOL','col':'rfMONTH'}}
		, 'MV_monthly': {'alignment':'PCY', 'type':float64, 'shape':2, 'mapping':{'POOL':'POOL','CATEGORY':'CATEGORY','PURCHASE_YEAR_CAPPED':'PURCHASE_YEAR_CAPPED','col':'rfMONTH'}} #Bond naming?
		, 'FAV_monthly': {'alignment':'PCY', 'type':float64, 'shape':2, 'mapping':{'POOL':'POOL','CATEGORY':'CATEGORY','PURCHASE_YEAR_CAPPED':'PURCHASE_YEAR_CAPPED','col':'rfMONTH'}} #Bond naming?
		, 'CF_monthly': {'alignment':'PCY', 'type':float64, 'shape':2, 'mapping':{'POOL':'POOL','CATEGORY':'CATEGORY','PURCHASE_YEAR_CAPPED':'PURCHASE_YEAR_CAPPED','col':'rfMONTH'}} #Bond naming?
		, 'MV': {'alignment':'PCY', 'type':float64, 'shape':2, 'monthly': False, 'mapping':{'POOL':'POOL','CATEGORY':'CATEGORY','PURCHASE_YEAR_CAPPED':'PURCHASE_YEAR_CAPPED','col':'rfMONTH'}} #Bond naming?
		, 'FAV': {'alignment':'PCY', 'type':float64, 'shape':2, 'monthly': False, 'mapping':{'POOL':'POOL','CATEGORY':'CATEGORY','PURCHASE_YEAR_CAPPED':'PURCHASE_YEAR_CAPPED','col':'rfMONTH'}} #Bond naming?
		# , 'CF': {'alignment':'PCY', 'type':float64, 'shape':2, 'monthly': False, 'mapping':{'POOL':'POOL','CATEGORY':'CATEGORY','PURCHASE_YEAR_CAPPED':'PURCHASE_YEAR_CAPPED','col':'rfMONTH'}} #Bond naming?
		, 'pcDENOM': {'alignment':'PC', 'type':float64, 'shape':2, 'monthly': False, 'mapping':{'POOL':'POOL','CATEGORY':'CATEGORY','col':'rfYEAR'}} #Specially derived
		, #hmmm, 4 dimensional inputs? purchase year and calendar year (as well as pool, category)? That would be 2 'col' type variables...
	}

	collapsers={
		'pcBOND_CF': {'source':'pcyBOND_CF', 'alignment':'PC','func':'sum', 'loop':1}
		, 'pBOND_CF': {'source':'pcyBOND_CF', 'alignment':'P','func':'sum', 'loop':1}
		, 'pBOND_FAV': {'source':'pcyBOND_FAV', 'alignment':'P','func':'sum','loop':1}
		, 'pcBOND_FAV': {'source':'pcyBOND_FAV', 'alignment':'PC','func':'sum','loop':1}
		, 'pEQUITY_RET': {'source':'pcEQUITY_RET', 'alignment':'P','func':'sum','loop':1}
		# , 
		, 'new_money': {'source':'new_money_temp','alignment':'','func':'sum','loop':1}
		, 'pBOND_TARGET': {'source':'pcBOND_TARGET', 'alignment':'P','func':'sum','loop':1}
		, 'pEQUITY_TARGET': {'source':'pcEQUITY_TARGET', 'alignment':'P','func':'sum','loop':1}
		, 'pBOND_REALIGN_TARGET': {'source':'pcBOND_REALIGN_TARGET', 'alignment':'P','func':'sum','loop':1}
		, 'pEQUITY_REALIGN_TARGET': {'source':'pcEQUITY_REALIGN_TARGET', 'alignment':'P','func':'sum','loop':1}
		, 'rfpEQUITY_REALIGN_TARGET': {'source':'rfpcEQUITY_REALIGN_TARGET', 'alignment':'P','func':'sum','loop':1}
		, 'pcBOND_MV': {'source':'pcyBOND_MV','alignment':'PC','func':'sum','loop':1}
		# , 'pcFAV_monthly': {'source':'pcyFAV_monthly','alignment':'PC','func':'sum','loop':1,'monthly':True}
		, 'pcREBAL_MV': {'source':'pcyREBAL_MV','alignment':'PC','func':'sum','loop':1}
		, 'pcREBAL_FAV': {'source':'pcyREBAL_FAV','alignment':'PC','func':'sum','loop':1}
		, 'pcREALIGN_MV': {'source':'pcyREALIGN_MV','alignment':'PC','func':'sum','loop':1}
		, 'pcREALIGN_FAV': {'source':'pcyREALIGN_FAV','alignment':'PC','func':'sum','loop':1}
		, 'pcLAGGED_MV': {'source':'pcyLAGGED_MV','alignment':'PC','func':'sum','loop':1}
		, 'pcLAGGED_FAV': {'source':'pcyLAGGED_FAV','alignment':'PC','func':'sum','loop':1}
		, 'pLAGGED_FAV': {'source':'pcyLAGGED_FAV','alignment':'P','func':'sum','loop':1}
		, 'pEQUITY_FAV_INV_RETURN': {'source':'pcEQUITY_FAV_INV_RETURN','alignment':'P','func':'sum','loop':1}
		, 'pBOND_FAV_INV_RETURN': {'source':'pcBOND_FAV_INV_RETURN','alignment':'P','func':'sum','loop':1}
	}
	
	rollforwards={
		'pcyREALIGN_BUYSELL_PCTS':{'initial':'ones','loop_var':'rfpcyREALIGN_BUYSELL_PCTS','loop':1,'alignment':'PCY'}
		, 'pBOX':{'initial':'zeros','loop_var':'rfpBOX','loop':1,'alignment':'P'}
		, 'pREALIGN_CASH':{'initial':'CASH_NOMINAL','loop_var':'rfpREALIGN_CASH','alignment':'P','loop':1}
		, 'pcEQUITY_REALIGN_TARGET':{'initial':'EQUITY_I_FAV','loop_var':'rfpcEQUITY_REALIGN_TARGET','loop':1,'alignment':'PC'}
		, 'pcyLAGGED_MV':{'initial':'MV_init','loop_var':'pcyREALIGN_MV','loop':1,'alignment':'PCY'}
		, 'pcyLAGGED_FAV':{'initial':'FAV_init','loop_var':'pcyREALIGN_FAV','loop':1,'alignment':'PCY'}
		, 'pLAGGED_REALIGN_TOTAL':{'initial':'pzeros','loop_var':'rfpREALIGN_TOTAL','loop':1,'alignment':'P'}
	}
	
	alignments={
		'P':'POOL'
		, 'C':'CATEGORY'
		, 'Y':'PURCHASE_YEAR_CAPPED'
	}
	
	def pBOX_INC(MONTH_T,rfpBOX,pBOX,LIAB_CF_PL,pBOND_CF,pCASH_RET):
		if MONTH_T==1:
			return rfpBOX - LIAB_CF_PL -pBOND_CF - pCASH_RET
		else:
			return rfpBOX - pBOX - LIAB_CF_PL -pBOND_CF - pCASH_RET
	
	def pFAV_INV_RET_RATE(T,pTOTAL_FAV_INV_RETURN,pLAGGED_REALIGN_TOTAL,pBOX, pLAGGED_FAV,pREALIGN_CASH,pEQUITY_REALIGN_TARGET):
		if T==0:
			denom = (pLAGGED_FAV + pREALIGN_CASH + pEQUITY_REALIGN_TARGET - pBOX)
		else:
			denom = (pLAGGED_REALIGN_TOTAL - pBOX)
		if abs(denom)<0.01:
			return 0
		else:
			return pTOTAL_FAV_INV_RETURN / denom
	
	def pTOTAL_FAV_INV_RETURN(pEQUITY_FAV_INV_RETURN,pBOND_FAV_INV_RETURN,pCASH_RET):
		return pBOND_FAV_INV_RETURN + pEQUITY_FAV_INV_RETURN + pCASH_RET
	
	def pcEQUITY_FAV_INV_RETURN(pcEQUITY_REALIGN_TARGET,pcEQUITY_RET):
		return pcEQUITY_RET - pcEQUITY_REALIGN_TARGET

	def pcBOND_FAV_INV_RETURN(pcBOND_FAV,pcBOND_CF,pcLAGGED_FAV):
		return pcBOND_FAV + pcBOND_CF - pcLAGGED_FAV
	
	def pcEQUITY_INV_EXP(pcEQUITY_REALIGN_TARGET,pcEQUITY_RET,MAINT_EXP_PC):
		return (pcEQUITY_REALIGN_TARGET+pcEQUITY_RET)*(1-(1-MAINT_EXP_PC/100)**(1/12))/2
	
	def pcBOND_INV_EXP(pcLAGGED_MV,pcBOND_MV,MAINT_EXP_PC):
		return (pcLAGGED_MV+pcBOND_MV)*(1-(1-MAINT_EXP_PC/100)**(1/12))/2
	
	def pcREALIGN_UNREALIZED_GAINLOSS(pcREALIGN_MV,pcREALIGN_FAV):
		return pcREALIGN_MV - pcREALIGN_FAV

	def pcREALIGN_REALIZED_GAINLOSS(MONTH_T,pcREBAL_MV,pcREBAL_FAV,pcREALIGN_SELL):
		if MONTH_T==12:
			return -1*(pcREBAL_MV - pcREBAL_FAV) * pcREALIGN_SELL
		else:
			return 0
	
	def pcREBAL_REALIZED_GAINLOSS(MONTH_T,pcBOND_MV,pcBOND_FAV,pcREBAL_SELL):
		if MONTH_T==12:
			return -1*(pcBOND_MV - pcBOND_FAV) * pcREBAL_SELL
		else:
			return 0

	def pcyREALIGN_FAV(MONTH_T,FAV_monthly,rfpcyREALIGN_BUYSELL_PCTS,pcyREBAL_FAV):
		if MONTH_T==12:
			return FAV_monthly*rfpcyREALIGN_BUYSELL_PCTS
		else:
			return pcyREBAL_FAV

	def pcyREBAL_FAV(FAV_monthly,pcyREBAL_BUYSELL_PCTS):
		return FAV_monthly * pcyREBAL_BUYSELL_PCTS

	def pcyREALIGN_MV(MONTH_T,MV_monthly,pcyREBAL_MV,rfpcyREALIGN_BUYSELL_PCTS):
		if MONTH_T==12:
			return MV_monthly * rfpcyREALIGN_BUYSELL_PCTS
		else:
			return pcyREBAL_MV

	def pcyREBAL_MV(MV_monthly,pcyREBAL_BUYSELL_PCTS):
		return MV_monthly*pcyREBAL_BUYSELL_PCTS
	
	def pcyBOND_MV(MV_monthly,pcyREALIGN_BUYSELL_PCTS):
		return MV_monthly * pcyREALIGN_BUYSELL_PCTS

	
	# def pcyBUYSELL_PCTS(pcREBAL_SELL,T,pcREBAL_BUY,pcREALIGN_SELL,pcREALIGN_BUY,PURCHASE_YEAR_CAPPED,pcyBUYSELL_PCTS):
		# This receives this period and last period. "0" is therefore last period
		# if PURCHASE_YEAR_CAPPED[0]==(T[0]+1+2019):
			# pcyBUYSELL_PCTS[1] = pcREBAL_BUY[0]*(1+pcREALIGN_SELL[0])+pcREALIGN_BUY[0]
		# elif PURCHASE_YEAR_CAPPED[0]<(T[0]+1+2019):
			# if T[0]==0:
				# pcyBUYSELL_PCTS[1] = 1 * ((1+pcREBAL_SELL[0])*(1+pcREALIGN_SELL[0]))
			# else:
				# pcyBUYSELL_PCTS[1] = pcyBUYSELL_PCTS[0] * ((1+pcREBAL_SELL[0])*(1+pcREALIGN_SELL[0]))

	def pcyREBAL_BUYSELL_PCTS(MONTH_T,YEAR_T,PURCHASE_YEAR_CAPPED,pcREBAL_BUY,pcREBAL_SELL,pcyREALIGN_BUYSELL_PCTS):
		if MONTH_T==12:
			if PURCHASE_YEAR_CAPPED==YEAR_T:
				return pcREBAL_BUY
			elif PURCHASE_YEAR_CAPPED<YEAR_T:
				return pcyREALIGN_BUYSELL_PCTS*(1+pcREBAL_SELL)
			else:
				return pcyREALIGN_BUYSELL_PCTS
		else:
			return pcyREALIGN_BUYSELL_PCTS

	def rfpcyREALIGN_BUYSELL_PCTS(MONTH_T,YEAR_T,PURCHASE_YEAR_CAPPED,pcREALIGN_BUY,pcREALIGN_SELL,pcyREBAL_BUYSELL_PCTS):
		if MONTH_T==12:
			if PURCHASE_YEAR_CAPPED==YEAR_T:
				return pcyREBAL_BUYSELL_PCTS*(1+pcREALIGN_SELL)+pcREALIGN_BUY
			elif PURCHASE_YEAR_CAPPED<YEAR_T:
				return pcyREBAL_BUYSELL_PCTS*(1+pcREALIGN_SELL)
			else:
				return pcyREBAL_BUYSELL_PCTS
		else:
			return pcyREBAL_BUYSELL_PCTS
	
	def pcyBOND_FAV(FAV_monthly,pcyREALIGN_BUYSELL_PCTS):
		#This receives this period and last period. "0" is therefore last period
		return FAV_monthly*pcyREALIGN_BUYSELL_PCTS

	def pcyBOND_CF(CF_monthly,pcyREALIGN_BUYSELL_PCTS):
		# for mt in range(12):
			# rfpcyBOND_CF[mt] = CF_monthly[mt+12]*pcyBUYSELL_PCTS[1]
		return CF_monthly*pcyREALIGN_BUYSELL_PCTS

	def new_money_temp(POOL,MONTH_T,rfpBOX):
		if MONTH_T==12:
			if POOL==10:
				return rfpBOX
			else:
				return 0
		else:
			return 0
	
	def rfpREALIGN_TOTAL(pBOND_REALIGN_TARGET,rfpEQUITY_REALIGN_TARGET,rfpREALIGN_CASH,rfpBOX):
		return pBOND_REALIGN_TARGET+rfpEQUITY_REALIGN_TARGET+rfpREALIGN_CASH+rfpBOX
	
	def rfpREALIGN_CASH(MONTH_T,pREALIGN_CASH,pTOTAL_REALIGN,pBOND_REALIGN_TARGET,rfpEQUITY_REALIGN_TARGET):
		if MONTH_T==12:
			return pTOTAL_REALIGN-pBOND_REALIGN_TARGET-rfpEQUITY_REALIGN_TARGET
		else:
			return pREALIGN_CASH
	
	def pcREALIGN_SELL(MONTH_T,POOL,CATEGORY,pcBOND_REALIGN_TARGET,pcBOND_TARGET):
		if MONTH_T==12:
			if (CATEGORY>=5) or (POOL==7): #this is horrible hard-coding, it should be fed by a table somewhere
				return 0
			else:
				if pcBOND_REALIGN_TARGET > pcBOND_TARGET:
					return 0
				else:
					if abs(pcBOND_TARGET + 0)<10e-10:
						return -1
					else:
						return -(1-pcBOND_REALIGN_TARGET/pcBOND_TARGET)
		else:
			return 0
	
	def pcREALIGN_BUY(MONTH_T,POOL,pcDENOM,pcBOND_REALIGN_TARGET,pcBOND_TARGET):
		if MONTH_T==12:
			if (pcDENOM==0) or (POOL==10):
				return 0
			else:
				return max(pcBOND_REALIGN_TARGET-pcBOND_TARGET,0)/pcDENOM
		else:
			return 0
	
	def rfpcEQUITY_REALIGN_TARGET(MONTH_T,POOL,pcEQUITY_TARGET,pTOTAL_REALIGN,EQUITY_CATEGORY_MIX
				,EQUITY_DRIFT_UPPER_PC,EQUITY_DRIFT_LOWER_PC):
		if MONTH_T==12:
			if POOL==10:
				return max(
						min(pcEQUITY_TARGET
						,EQUITY_DRIFT_UPPER_PC*pTOTAL_REALIGN/100)
						,EQUITY_DRIFT_LOWER_PC*pTOTAL_REALIGN/100)
			else:
				return pTOTAL_REALIGN * EQUITY_CATEGORY_MIX/100
		else:
			return pcEQUITY_TARGET

	def pcBOND_REALIGN_TARGET(MONTH_T,POOL,pcBOND_TARGET,pTOTAL_REALIGN,BONDS_CATEGORY_MIX
				,BOND_DRIFT_UPPER_PC,BOND_DRIFT_LOWER_PC):
		if MONTH_T==12:
			if POOL==10:
				return max(
						min(pcBOND_TARGET
						,BOND_DRIFT_UPPER_PC*pTOTAL_REALIGN/100)
						,BOND_DRIFT_LOWER_PC*pTOTAL_REALIGN/100)
			else:
				return pTOTAL_REALIGN * BONDS_CATEGORY_MIX/100
		else:
			return pcBOND_TARGET
		
	
	def pTOTAL_REALIGN(MATH_RES_IF_PL,DA_FUND_FL,CA_FUND_FL,SOLV_MARG_IF_FL):
		return MATH_RES_IF_PL + DA_FUND_FL + CA_FUND_FL + SOLV_MARG_IF_FL
	
	def pcREBAL_SELL(MONTH_T,CATEGORY,POOL,pcBOND_TARGET,pcBOND_FAV,cNEW_MONEY_TO_BOND):
		if MONTH_T==12:
			if (CATEGORY>=5) or (POOL==7): #this is horrible hard-coding, it should be fed by a table somewhere
				return 0
			else:
				temp_NEW_MONEY_TO_BOND=0
				if POOL==10:
					temp_NEW_MONEY_TO_BOND=cNEW_MONEY_TO_BOND
				if pcBOND_TARGET > (pcBOND_FAV + temp_NEW_MONEY_TO_BOND):
					return 0
				else:
					if abs(pcBOND_FAV + temp_NEW_MONEY_TO_BOND)<10e-10:
						return -1
					else:
						return -(1-pcBOND_TARGET/(pcBOND_FAV + temp_NEW_MONEY_TO_BOND))
		else:
			return 0

	def pcREBAL_BUY(MONTH_T,POOL,cNEW_MONEY_TO_BOND,pcBOND_FAV,pcDENOM,pcBOND_TARGET):
		if MONTH_T==12:
			if POOL==10:
				if (cNEW_MONEY_TO_BOND + pcBOND_FAV)==0:
					temp_pcREBAL_BUY = 0
				else:
					temp_pcREBAL_BUY = cNEW_MONEY_TO_BOND * pcBOND_TARGET / (cNEW_MONEY_TO_BOND + pcBOND_FAV)
			else:
				temp_pcREBAL_BUY = pcBOND_TARGET - pcBOND_FAV
			if pcDENOM==0:
				return 0
			else:
				return max(temp_pcREBAL_BUY,0)/pcDENOM
		else:
			return 0
	
	def pREBAL_CASH(MONTH_T,pREALIGN_CASH,pTOTAL_FAV,pBOND_TARGET,pEQUITY_TARGET):
		if MONTH_T==12:
			return pTOTAL_FAV - pBOND_TARGET - pEQUITY_TARGET
		else:
			return pREALIGN_CASH
		
	def pcEQUITY_TARGET(POOL,MONTH_T,pTOTAL_FAV,EQUITY_CATEGORY_MIX
			,pcEQUITY_RET,EQUITY_DRIFT_UPPER_PC,EQUITY_DRIFT_LOWER_PC,cNEW_MONEY_TO_EQUITY):
		if MONTH_T==12:
			if POOL==10:
				return max(
						min(cNEW_MONEY_TO_EQUITY + pcEQUITY_RET
						,EQUITY_DRIFT_UPPER_PC*pTOTAL_FAV/100)
						,EQUITY_DRIFT_LOWER_PC*pTOTAL_FAV/100)
			else:
				return pTOTAL_FAV * EQUITY_CATEGORY_MIX/100
		else:
			return pcEQUITY_RET
	
	def pcBOND_TARGET(POOL,MONTH_T,pTOTAL_FAV,BONDS_CATEGORY_MIX
			,pcBOND_FAV,BOND_DRIFT_UPPER_PC,BOND_DRIFT_LOWER_PC,cNEW_MONEY_TO_BOND):
		if MONTH_T==12:
			if POOL==10:
				return max(
						min(cNEW_MONEY_TO_BOND + pcBOND_FAV
						,BOND_DRIFT_UPPER_PC*pTOTAL_FAV/100)
						,BOND_DRIFT_LOWER_PC*pTOTAL_FAV/100)
			else:
				return pTOTAL_FAV * BONDS_CATEGORY_MIX/100
		else:
			return pcBOND_FAV

	def cNEW_MONEY_TO_EQUITY(MONTH_T,EQUITY_NEW_MONEY_P_PC,EQUITY_NEW_MONEY_N_PC,new_money):
		if MONTH_T==12:
			if new_money>=0:
				return EQUITY_NEW_MONEY_P_PC*new_money/100
			else:
				return EQUITY_NEW_MONEY_N_PC*new_money/100
		else:
			return 0
	
	def cNEW_MONEY_TO_BOND(MONTH_T,BOND_NEW_MONEY_P_PC,BOND_NEW_MONEY_N_PC,new_money):
		if MONTH_T==12:
			if new_money>=0:
				return BOND_NEW_MONEY_P_PC*new_money/100
			else:
				return BOND_NEW_MONEY_N_PC*new_money/100
		else:
			return 0
	
	def pTOTAL_FAV(pBOND_FAV,pEQUITY_RET,pREALIGN_CASH,rfpBOX):
		return pBOND_FAV + pEQUITY_RET + pREALIGN_CASH + rfpBOX
	
# 		for mt in range(1,13):
	def rfpBOX(MONTH_T,pBOND_CF,pBOX,CASH_RET_PC,LIAB_CF_PL,pCASH_RET):
		if MONTH_T==1:
			return                            LIAB_CF_PL*((1+CASH_RET_PC/100)**0.5) + pBOND_CF + pCASH_RET
		else:
			return pBOX*(1+CASH_RET_PC/100) + LIAB_CF_PL*((1+CASH_RET_PC/100)**0.5) + pBOND_CF + pCASH_RET

	def pCASH_RET(CASH_RET_PC,pREALIGN_CASH):
		return CASH_RET_PC*pREALIGN_CASH/100

	def pcEQUITY_RET(pcEQUITY_REALIGN_TARGET,EQUITY_RET_RATE):
		return pcEQUITY_REALIGN_TARGET * (1 + EQUITY_RET_RATE)
				
	def ones(FAV_monthly):
		return 1
		
	def zeros(FAV_monthly):
		return 0
	def pzeros(CASH_NOMINAL):
		return 0
		
	def MONTH_T(T):
		return T%12+1
		
	def YEAR_T(T):
		return 2020+T//12
