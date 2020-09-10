import pandas as pd
import numpy as np
from numpy import int32, int64, float64
from numba import vectorize, prange
from .base import BaseFunctions

class HKDCashFunctions:
	single_derivations=dict(
		CALENDAR_YEAR = {'type': int32, 'numba':'v','shape':2}
	)
	complex_derivations={
		'VALUATION':{'outvars':{
			'ASSET_TOT_ROR': {'type': float64, 'shape':2}
			, 'TOT_CASH_BOND_RETURN': {'type': float64, 'shape':2}
			, 'ASSET_MV': {'type': float64, 'shape':2}
		}}
	}
	
	summaries={}
	
	mappings={
		'EQUITY_SCALAR':{'source':'ASSET_POOL_BASIS', 'type':float64,'shape':1,'mapping':{'Pool':'POOL'}}
		
		,'CASH_PROPNHELD_CL':{'type':float64,'shape':2,'mapping':{'Pool':'POOL','col':'CALENDAR_YEAR'}}
		,'BOND_INC_CF_FL':{'type':float64,'shape':2,'mapping':{'Pool':'POOL','col':'CALENDAR_YEAR'}}
		,'CASH_ADJ_FL':{'type':float64,'shape':2,'mapping':{'Pool':'POOL','col':'CALENDAR_YEAR'}}
		,'ZCB':{'type':float64,'shape':2,'mapping':{'SIMULATION':'SIMULATION','ECONOMY':'ECONOMY','col':'CALENDAR_YEAR'}}
	}
	
	def VALUATION(CALENDAR_YEAR,val_yr,ZCB,CASH_PROPNHELD_CL,EQUITY_SCALAR,NOMINAL,ASSET_SCALAR,BOND_INC_CF_FL,CASH_ADJ_FL
				,ASSET_TOT_ROR,TOT_CASH_BOND_RETURN,ASSET_MV):
		for y in prange(CALENDAR_YEAR.shape[0]):
			for t in range(CALENDAR_YEAR.shape[1]):
				if CALENDAR_YEAR[y,t]==val_yr[y,0]:
					ASSET_TOT_ROR[y,t]=0
					TOT_CASH_BOND_RETURN[y,t]=0
					ASSET_MV[y,t]=ASSET_SCALAR[y,0]*CASH_PROPNHELD_CL[y,t]*EQUITY_SCALAR[y,0]*NOMINAL[y,0]
				else:
					ASSET_TOT_ROR[y,t]=1/ZCB[y,t-1]-1
					TOT_CASH_BOND_RETURN[y,t]=ASSET_MV[y,t-1]*ASSET_TOT_ROR[y,t]+BOND_INC_CF_FL[y,t]*((1+ASSET_TOT_ROR[y,t])**0.5-1)
					ASSET_MV[y,t]=(ASSET_MV[y,t-1]*(1+ASSET_TOT_ROR[y,t])+BOND_INC_CF_FL[y,t]*((1+ASSET_TOT_ROR[y,t])**0.5))*CASH_PROPNHELD_CL[y,t]+CASH_ADJ_FL[y,t]
	
	def CALENDAR_YEAR(T,val_yr):
		return val_yr+T
		
class USDCashFunctions: