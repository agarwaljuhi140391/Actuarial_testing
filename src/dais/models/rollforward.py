import pandas as pd
import numpy as np
from numpy import int32, int64, float64
from numba import vectorize, prange
from .base import BaseFunctions

class RollForwardFunctions():
	single_derivations=dict(
		BONDS_FAV_T0 = {'type': float64, 'numba':'j','shape':1}
		
		, CALENDAR_YEAR = {'type': int32, 'numba':'v','shape':2}
		, CALENDAR_MONTH = {'type': int32, 'numba':'v','shape':2}
		, BONDS_CATEGORY_MIX_ANNUAL = {'type': float64, 'numba':'v','shape':2}
		, EQUITY_CATEGORY_MIX = {'type': float64, 'numba':'v','shape':2}
		
		, EQUITY_RET_RATE = {'type': float64, 'numba':'j','shape':2}
	)
	
	complex_derivations={
		'ROLL_FORWARD':{
			'outvars': {
				'test': {'type': float64, 'shape':2}
			}
		}
	}
	summaries={

	}

	mappings={
# 		'ECONOMY': {'source':'ASSETS_EQUITY','shape':1,'type':int32, 'mapping':{'POOL':'POOL','CATEGORY':'CATEGORY'}}
		'EQUITY_I_FAV': {'source':'ASSETS_EQUITY','shape':1,'type':float64,'mapping':{'POOL':'POOL','CATEGORY':'CATEGORY'}}
		, 'CASH_NOMINAL': {'source':'ASSETS_CASH','shape':1,'type':float64,'mapping':{'POOL':'POOL'}}
		, 'BOND_NEW_MONEY_P_PC': {'source':'STRATEGY_STATIC','shape':1,'type':float64, 'mapping':{'CATEGORY':'CATEGORY'}}
		, 'BOND_NEW_MONEY_N_PC': {'source':'STRATEGY_STATIC','shape':1,'type':float64, 'mapping':{'CATEGORY':'CATEGORY'}}
		, 'BOND_DRIFT_UPPER_PC': {'source':'STRATEGY_STATIC','shape':1,'type':float64, 'mapping':{'CATEGORY':'CATEGORY'}}
		, 'BOND_DRIFT_LOWER_PC': {'source':'STRATEGY_STATIC','shape':1,'type':float64, 'mapping':{'CATEGORY':'CATEGORY'}}
		, 'EQUITY_NEW_MONEY_P_PC': {'source':'STRATEGY_STATIC','shape':1,'type':float64, 'mapping':{'CATEGORY':'CATEGORY'}}
		, 'EQUITY_NEW_MONEY_N_PC': {'source':'STRATEGY_STATIC','shape':1,'type':float64, 'mapping':{'CATEGORY':'CATEGORY'}}
		, 'EQUITY_DRIFT_UPPER_PC': {'source':'STRATEGY_STATIC','shape':1,'type':float64, 'mapping':{'CATEGORY':'CATEGORY'}}
		, 'EQUITY_DRIFT_LOWER_PC': {'source':'STRATEGY_STATIC','shape':1,'type':float64, 'mapping':{'CATEGORY':'CATEGORY'}}
	
		, 'BONDS_FAV': {'autocapcol': True,'type':float64, 'shape':2, 'mapping':{'POOL':'POOL','CATEGORY':'CATEGORY','col':'T'}}
		, 'BONDS_CF': {'autocapcol': True,'type':float64, 'shape':2, 'mapping':{'POOL':'POOL','CATEGORY':'CATEGORY','col':'T'}}
		, 'EQUITY_RET_IDX': {'autocapcol': True,'type':float64, 'shape':2, 'mapping':{'SIMULATION':'SIMULATION','ECONOMY':'ECONOMY','CLASS':'CATEGORY','col':'CALENDAR_YEAR'}}
		, 'CASH_RET_PC': {'autocapcol': True,'type':float64, 'shape':2, 'mapping':{'SIMULATION':'SIMULATION','ECONOMY':'ECONOMY','col':'CALENDAR_YEAR'}}
		, 'BONDS_CATEGORY_MIX': {'autocapcol': True,'type':float64, 'shape':2, 'mapping':{'INV_STR_RULES':'INV_STR_RULES','POOL':'POOL','CATEGORY':'CATEGORY','col':'CALENDAR_YEAR'}}
		, 'EQUITY_CATEGORY_MIX': {'autocapcol': True,'type':float64, 'shape':2, 'mapping':{'INV_STR_RULES':'INV_STR_RULES','POOL':'POOL','CATEGORY':'CATEGORY','col':'CALENDAR_YEAR'}}
	}
	
	def ROLL_FORWARD(test):
		"""
		Start at T=0 with bond FAV, equity FAV and CASH FAV/NOMINAL. Box at T=0 is set to 0.
		Move to T=1
		Calculate Equity FAV at T=1. Take equity at T=0 and multiply by (1+EQUITY_RET_RATE).
		Calculate Cash FAV for the pool at T=1. Equal to cash position at T=0
		Calculate TOTAL_ACF for the pool at T=1. Equal to the sum of the BONDS_CF for the pool at T=1 (across categories) plus Cash at T=1 times CASH_RET_PC.  
		Calculate Box FAV for the pool at T=1. Equal to Box at T=0 (0) times (1+CASH_RET_PC) plus CASH_FLOW_FL for the pool times (1+CASH_RET_PC)**0.5 plus TOTAL_ACF.
		Calculate Total_FAV_Before_Dec for the pool at T=1 as sum of Bond FAV, Equity FAV, Cash FAV, Box FAV.
		Move to T=2
		Calc Eq FAV by taking Eq FAV at T1 and multiply again.
		Cash FAV equal to val at T1.
		TOTAL_ACF at T=2. Equal to sum of BONDS_CF (sum across categories plus Cash time CASH_RET_PC again.
		Box FAV equal to Box at T=1 time (1+CASH_RET_PC) plus CASH_FLOW_FL for the pool time (as above) plus TOTAL_ACF)
		Total_FAV as above
		Rinse and repat until T=12
		Do the above, then:
			Total Rebalancing target for the pool, equal to Total_FAV.
			New money is equal to Box at T=12.
			Split the New money to bonds by category across pool 10 only ? according to the BOND_NEW_MONEY_P_PC
			Bond Rebalance:
				for the non-10 pool: Multiply Bonds FAV (at pool and category) by BONDS_CATEGORY_MIX (at pool and category)
				for the 10 pool: max(
										min(Bond_FAV + New Money (total) * if New money is +ve then BOND_NEW_MONEY_P_PC else BOND_NEW_MONEY_N_PC, or Total Rebalancing for 10 times drift upper bound)
										,Total Rebalancing for 10 times drift lower bound)
			Equity rebalance:
				as for bonds, but with equity bounds, equity mix, etc.
			Calculate buy/sell by pool and category by subtracting Bond FAV (as calculated in main section) from the rebalance figures in the special T12 logic. Maxed at 0.
				For pool 10, you do something special.
				Take new money to bond pool 10 by category multiply by ratio of total after rebalance/(FAV before and after), divide by ?sum of new bond values for puchase this year?
			Calculate buy/sell by pool and category by subtracting Equity FAV.
				Pool 10: 
		"""
	
	def BONDS_CATEGORY_MIX_ANNUAL(BONDS_CATEGORY_MIX,CALENDAR_MONTH):
		if CALENDAR_MONTH==12:
			return BONDS_CATEGORY_MIX
		else:
			return 999999
	
	def EQUITY_CATEGORY_MIX(EQUITY_CATEGORY_MIX,CALENDAR_MONTH):
		if CALENDAR_MONTH==12:
			return EQUITY_CATEGORY_MIX
		else:
			return 999999
				
	def BONDS_FAV_T0(BONDS_FAV,BONDS_FAV_T0):
		BONDS_FAV_T0[0] = BONDS_FAV[0]
			
	def EQUITY_RET_RATE(CALENDAR_MONTH,EQUITY_RET_IDX,EQUITY_RET_RATE):
		for t in range(EQUITY_RET_IDX.shape[0]):
			if t>0:
				if (EQUITY_RET_IDX[t-1] != 0) and (CALENDAR_MONTH[t]==1):
					EQUITY_RET_RATE[t] = (EQUITY_RET_IDX[t]/EQUITY_RET_IDX[t-1])**(1/12)-1
				else:
					EQUITY_RET_RATE[t] = EQUITY_RET_RATE[t-1]

	def CALENDAR_YEAR(T,val_yr,val_mth):
		return (val_yr*12+val_mth+T+1-2)//12
	def CALENDAR_MONTH(T,val_mth):
		return (val_mth+T+1+10)%12+1