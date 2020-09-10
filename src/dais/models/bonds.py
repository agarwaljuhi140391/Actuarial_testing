import pandas as pd
import numpy as np
from numpy import int32, int64, float64
from numba import vectorize, prange
from .base import BaseFunctions
from .simple_dispatcher import irr_est


class BondFunctions():
	single_derivations=dict(
	# **BaseFunctions.single_derivations,
		BOND_REMAINING_PERIOD_T0 = {'type': int32, 'numba':'v','shape':1}
		, REDEMPTION_TERM = {'type': int32, 'numba':'v','shape':1}
		, ACCRUED_INTEREST_T0 = {'type': float64, 'numba':'v','shape':1}
		, BOOK_YIELD_NPVRATE = {'type': float64, 'numba':'v','shape':1}
		, PURCHASE_T0 = {'type': int32, 'numba':'v','shape':1}
		, BV_CLEAN_T0 = {'type': float64, 'numba':'v','shape':1}
		, PURCHASE_YEAR_CAPPED = {'type': int32, 'numba':'v','shape':1}

		, NET_PRESENT_VALUE_T0 = {'type': float64, 'numba':'j','shape':1}
		, BOOK_YIELD = {'type': float64, 'numba':'j','shape':1}
		, MV_T0 = {'type': float64, 'numba':'j','shape':1}
		, COUPON_RATE_RESET = {'type': float64, 'numba':'j','shape':1}

		, CALENDAR_YEAR = {'type': int32, 'numba':'v','shape':2}
		, CALENDAR_YEAR_VT = {'type': int32, 'numba':'v','shape':2}
		, BVCF = {'type': float64, 'numba':'v','shape':2}
		, CALENDAR_MONTH = {'type': int32, 'numba':'v','shape':2}
		, CALENDAR_MONTH_VT = {'type': int32, 'numba':'v','shape':2}
		, CF_T0 = {'type': float64, 'numba':'v','shape':2}
		, REDEMPTION = {'type': float64, 'numba':'v','shape':2}
		, ACCRUED_INTEREST = {'type': float64, 'numba':'v','shape':2}
		, BV = {'type': float64, 'numba':'v','shape':2}
		, FAV = {'type': float64, 'numba':'v','shape':2}
		# , BOND_REMAINING_PERIOD = {'type': float64, 'numba':'v','shape':2}
		, BOND_REMAINING_PERIOD_VT = {'type': int32, 'numba':'v','shape':2}
		
		, COUPON_RATE_T0 = {'type': float64, 'numba':'j','shape':2}

	)
	
	complex_derivations={
		'BOND_VALUES': {'outvars':{
			  'MV': {'type': float64, 'shape':2,'2d':'VT'}
			, 'NPV_BY': {'type': float64, 'shape':2,'2d':'VT'}
			, 'CF': {'type': float64, 'shape':2,'2d':'VT'}
			, 'COUPON_RATE_RESET_T': {'type': float64, 'shape':2,'2d':'VT'}
			}
		}
	}
	summaries={
		'Summary_POOL_CATEGORY': {
			'byvars':['POOL','CATEGORY']
			,'vars':['MV','FAV','CF','ACCRUED_INTEREST']
			,'func':'sum'
		}
	}

	mappings={
		'YIELD_IDX': {'source':'YIELDS', 'type':int32, 'shape':1, 'mapping':{'ECONOMY':'ECONOMY','SIMULATION':'SIMULATION'}}

		, 'FX_INDEX': {'autocapcol': True,'type':float64, 'shape':2, 'mapping':{'SIMULATION':'SIMULATION','ECONOMY':'ECONOMY','col':'CALENDAR_YEAR_VT'}}
	}
	
	def PURCHASE_YEAR_CAPPED(PURCHASE_YEAR):
		return max(2019,PURCHASE_YEAR)

	def PURCHASE_T0(PURCHASE_YEAR,val_yr):
		return max(0,(PURCHASE_YEAR-val_yr)*12)
	
	def FAV(VT,BASIS_FLAG,MV,MV_T0,BV_CLEAN_T0,ACCRUED_INTEREST_T0,BV,PURCHASE_T0,ACCRUED_INTEREST):
		if (VT-PURCHASE_T0) > 0:
			if BASIS_FLAG==1:
				return MV
			else:
				return BV+ACCRUED_INTEREST
		elif (VT-PURCHASE_T0)==0:
			if BASIS_FLAG==1:
				return MV_T0
			else:
				return BV_CLEAN_T0+ACCRUED_INTEREST_T0
		else:
			return 0
	
	def BV_CLEAN_T0(I_ABV,ASSET_SCALAR):
		return I_ABV*ASSET_SCALAR
	
	def BV(NPV_BY,ACCRUED_INTEREST):
		return NPV_BY-ACCRUED_INTEREST
	
	def BOND_REMAINING_PERIOD_VT(REDEMP_MONTH,REDEMP_YEAR,CALENDAR_MONTH_VT,CALENDAR_YEAR_VT):
		if CALENDAR_YEAR_VT>REDEMP_YEAR:
			return 0
		elif CALENDAR_YEAR_VT==REDEMP_YEAR:
			if CALENDAR_MONTH_VT>REDEMP_MONTH:
				return 0
			else:
				return REDEMP_MONTH-CALENDAR_MONTH_VT
		else:
			return (REDEMP_YEAR-CALENDAR_YEAR_VT)*12+(REDEMP_MONTH-CALENDAR_MONTH_VT)
	
	def CALENDAR_YEAR_VT(VT,val_yr,val_mth):
		return (val_yr*12+val_mth+VT+1-2)//12
	def CALENDAR_MONTH_VT(VT,val_mth):
		return (val_mth+VT+1+10)%12+1
	
	def ACCRUED_INTEREST(COUPON_FREQ,REDEMP_AMT,COUPON_RATE_RESET_T,BOND_REMAINING_PERIOD_VT,ASSET_SCALAR):
		if COUPON_FREQ==0:
			return 0
		else:
			return REDEMP_AMT*COUPON_RATE_RESET_T*((-BOND_REMAINING_PERIOD_VT)%(12/COUPON_FREQ))/12*ASSET_SCALAR

	def COUPON_RATE_RESET(COUPON_RATE_T0,COUPON_FREQ,COUPON_RATE_RESET):
		COUPON_RATE_RESET[0]=COUPON_FREQ[0]*COUPON_RATE_T0[1]


	def BOOK_YIELD_NPVRATE(BOOK_YIELD):
		return (1+BOOK_YIELD)**(1/12)-1

	def BOOK_YIELD(BVCF,BOOK_YIELD):
		BOOK_YIELD[0]=(1+irr_est(BVCF,0.1))**12-1  #0.01 is the guess
	
	def BVCF(T,I_ABV,ASSET_SCALAR,ACCRUED_INTEREST_T0,CF_T0):
		if T==0:
			return -I_ABV*ASSET_SCALAR-ACCRUED_INTEREST_T0
		else:
			return CF_T0
	
	def ACCRUED_INTEREST_T0(REDEMP_AMT,COUPON_PC,BOND_REMAINING_PERIOD_T0,COUPON_FREQ,ASSET_SCALAR):
		if COUPON_FREQ==0:
			return 0
		else:
			return REDEMP_AMT*(COUPON_PC/100)*(-BOND_REMAINING_PERIOD_T0%(12/COUPON_FREQ))/12*ASSET_SCALAR

	def REDEMPTION_TERM(BOND_REMAINING_PERIOD_T0):
		return BOND_REMAINING_PERIOD_T0

	def BOND_REMAINING_PERIOD_T0(REDEMP_YEAR,REDEMP_MONTH,val_yr,val_mth):
		return (REDEMP_YEAR-val_yr)*12+(REDEMP_MONTH-val_mth)
		
	def BOND_VALUES(VT,FX_INDEX,MV_T0,BVCF,ACCRUED_INTEREST_T0,CALENDAR_YEAR,MV_NUMPERIODS,COUPON_FREQ,COUPON_PC,COUPON_RATE_RESET,ASSET_TYPE,REF_RATE_MARGIN_PC
					,PURCHASE_T0,YIELD_IDX,CALENDAR_MONTH,REDEMP_YEAR,REDEMP_MONTH,BOND_REMAINING_PERIOD_VT,ASSET_SCALAR
					,YIELDS,BOOK_YIELD_NPVRATE,REDEMPTION,REDEMP_AMT,val_yr
					,COUPON_RATE_RESET_T,MV,NPV_BY,CF
				   ):
		for t in range(PURCHASE_T0[0],min(BOND_REMAINING_PERIOD_VT[1+PURCHASE_T0[0]]+2,MV_NUMPERIODS[0])):
			if t==PURCHASE_T0[0]:
				MV[t]=MV_T0[0]
				NPV_BY[t]=-(BVCF[0]+ACCRUED_INTEREST_T0[0])
				COUPON_RATE_RESET_T[t]=COUPON_RATE_RESET[0]
			else:
				for ty in range(t,min(t+BOND_REMAINING_PERIOD_VT[t]+1,CALENDAR_YEAR.shape[0])):
					if (COUPON_FREQ[0]==0) or (CALENDAR_YEAR[ty]>REDEMP_YEAR[0]):
						inner_COUPON_RATE2 = 0
					else:
						if ASSET_TYPE[0]==11:
							inner_COUPON_RATE2 = COUPON_PC[0]/(100*COUPON_FREQ[0])
						elif ASSET_TYPE[0]==21:
							if ty==t:
								inner_COUPON_RATE2 = COUPON_RATE_RESET_T[t-1]/(COUPON_FREQ[0])
							elif (BOND_REMAINING_PERIOD_VT[t]+1-(ty-t))%(12/COUPON_FREQ[0])==0:
								inner_COUPON_RATE2 = (((YIELDS[YIELD_IDX[0]+t,ty-t-1]/YIELDS[YIELD_IDX[0]+t,ty-t])**12+REF_RATE_MARGIN_PC[0]/100)**(1/COUPON_FREQ[0])-1)
							else:
								# icr[y,t] = icr[y,t]
								pass
						else:
							inner_COUPON_RATE2 = 0
					#END COUPON RATE CALC
					##########
					#CALCULALATE THE COUPON AMOUNT
					if COUPON_FREQ[0]==0:
						inner_COUPON2 = 0
					else:
						if BOND_REMAINING_PERIOD_VT[t]-(ty-t)>=0:
							if (BOND_REMAINING_PERIOD_VT[t]-(ty-t))%(12/COUPON_FREQ[0])==0:
								inner_COUPON2 = inner_COUPON_RATE2*REDEMP_AMT[0]*ASSET_SCALAR[0]*FX_INDEX[t-PURCHASE_T0[0]]
							else:
								inner_COUPON2 = 0
						else:
							inner_COUPON2 = 0
					#END COUPON CALC
					##########
					if ty>t:
						MV[t]=MV[t]+(inner_COUPON2 + FX_INDEX[t-PURCHASE_T0[0]]*REDEMPTION[ty]*ASSET_SCALAR[0])*YIELDS[YIELD_IDX[0]+t,ty-t]
						NPV_BY[t]=NPV_BY[t]+(inner_COUPON2 + FX_INDEX[t-PURCHASE_T0[0]]*REDEMPTION[ty]*ASSET_SCALAR[0])/(1+BOOK_YIELD_NPVRATE[0])**(ty-t)
					if ty==(t+1):
						COUPON_RATE_RESET_T[t]=inner_COUPON_RATE2*COUPON_FREQ[0]
					if ty==t:
						CF[t]=inner_COUPON2 + FX_INDEX[t-PURCHASE_T0[0]]*REDEMPTION[ty]*ASSET_SCALAR[0]
		
	def NET_PRESENT_VALUE_T0(MV_NUMPERIODS,YIELDS,YIELD_IDX
							,CALENDAR_YEAR,CALENDAR_MONTH,REDEMP_MONTH,REDEMP_YEAR
							,COUPON_PC,REDEMP_AMT,COUPON_FREQ,ASSET_TYPE,ASSET_SCALAR
							,COUPON_PLACE,COUPON_RATE,REDEMPTION,REF_RATE_MARGIN_PC
							,NET_PRESENT_VALUE_T0):
		if ASSET_TYPE[0]==11:
			if COUPON_FREQ[0]==0:
				COUPON_RATE[0]=0
			else:
				COUPON_RATE[0]=COUPON_PC[0]/(100*COUPON_FREQ[0])
		for ty in range(CALENDAR_YEAR.shape[0]): #output time
			if COUPON_FREQ[0]==0:
				COUPON_RATE[0]=0
			else:
				if ASSET_TYPE[0]==21:
					if ty==0:
						COUPON_RATE[0]=0
					if (ty==1):
						COUPON_RATE[0]=COUPON_PC[0]/(100*COUPON_FREQ[0])
					elif (CALENDAR_MONTH[ty]-REDEMP_MONTH[0]-1)%(12/COUPON_FREQ[0])==0:
						COUPON_RATE[0]=(((YIELDS[YIELD_IDX[0],ty-1]/YIELDS[YIELD_IDX[0],ty])**12+REF_RATE_MARGIN_PC[0]/100)**(1/COUPON_FREQ[0])-1)
					else:
						pass #no change
				else:
					COUPON_RATE[0]=0 #unknown asset type
			if (ty==0) or (CALENDAR_YEAR[ty]>REDEMP_YEAR[0]) or ((CALENDAR_YEAR[ty]==REDEMP_YEAR[0]) and (CALENDAR_MONTH[ty]>REDEMP_MONTH[0])):
				COUPON_PLACE[0]=0
			else:
				if (CALENDAR_MONTH[ty]-REDEMP_MONTH[0])%(12/COUPON_FREQ[0])==0:
					COUPON_PLACE[0]=COUPON_RATE[0]*ASSET_SCALAR[0]*REDEMP_AMT[0]
				else:
					COUPON_PLACE[0]=0
			NET_PRESENT_VALUE_T0[0]=NET_PRESENT_VALUE_T0[0]+YIELDS[YIELD_IDX[0],ty]*(COUPON_PLACE[0]+ASSET_SCALAR[0]*REDEMPTION[ty])

# 	def MV_T0(CF_T0,YIELDS,YIELD_IDX,MV_T0):
# 		for t in range(CF_T0.shape[0]): #cf time
# 			MV_T0[0]=MV_T0[0]+YIELDS[YIELD_IDX[0],t]*CF_T0[t]

	def MV_T0(PURCHASE_T0,CF_T0,YIELDS,YIELD_IDX,REDEMPTION_TERM,MV_T0):
		for t in range(PURCHASE_T0[0]+1,REDEMPTION_TERM[0]+1+PURCHASE_T0[0]): #cf time
			MV_T0[0]=MV_T0[0]+YIELDS[YIELD_IDX[0]+PURCHASE_T0[0],t-PURCHASE_T0[0]]*CF_T0[t]


		
	def CF_T0(CALENDAR_YEAR,REDEMP_YEAR,CALENDAR_MONTH,REDEMP_MONTH,COUPON_FREQ
				,COUPON_RATE_T0,REDEMP_AMT,ASSET_SCALAR,REDEMPTION):
		if (COUPON_FREQ==0) or (CALENDAR_YEAR>REDEMP_YEAR) or ((CALENDAR_YEAR==REDEMP_YEAR) and (CALENDAR_MONTH>REDEMP_MONTH)):
			return 0
		elif (CALENDAR_MONTH-REDEMP_MONTH)%(12/COUPON_FREQ)==0:
			return ASSET_SCALAR*(REDEMP_AMT*COUPON_RATE_T0+REDEMPTION)
		else:
			return ASSET_SCALAR*REDEMPTION
	
	def COUPON_RATE_T0(YIELDS,YIELD_IDX,ASSET_TYPE,COUPON_PC,COUPON_FREQ,CALENDAR_YEAR,CALENDAR_MONTH,REDEMP_YEAR,REDEMP_MONTH,REF_RATE_MARGIN_PC,COUPON_RATE_T0):
		for t in range(COUPON_RATE_T0.shape[0]): #cf time
			if (COUPON_FREQ[0]==0) or (CALENDAR_YEAR[t]>REDEMP_YEAR[0]):
				COUPON_RATE_T0[t]=0
			else:
				if ASSET_TYPE[0]==11:
					COUPON_RATE_T0[t]=COUPON_PC[0]/(100*COUPON_FREQ[0])
				elif ASSET_TYPE[0]==21:
					if t==0:
						COUPON_RATE_T0[t]=0
					elif t==1:
						COUPON_RATE_T0[t]=COUPON_PC[0]/(100*COUPON_FREQ[0])
					elif (CALENDAR_MONTH[t]-REDEMP_MONTH[0]-1)%(12/COUPON_FREQ[0])==0:
						COUPON_RATE_T0[t]=(((YIELDS[YIELD_IDX[0],t-1]/YIELDS[YIELD_IDX[0],t])**12+REF_RATE_MARGIN_PC[0]/100)**(1/COUPON_FREQ[0])-1)
					else:
						COUPON_RATE_T0[t]=COUPON_RATE_T0[t-1]
				else:
					COUPON_RATE_T0[t]=0

	def REDEMPTION(CALENDAR_YEAR,REDEMP_YEAR,CALENDAR_MONTH,REDEMP_MONTH,REDEMP_AMT):
		if (CALENDAR_MONTH==REDEMP_MONTH) and (CALENDAR_YEAR==REDEMP_YEAR):
			return REDEMP_AMT
		else:
			return 0

	def CALENDAR_YEAR(T,val_yr,val_mth):
		return (val_yr*12+val_mth+T+1-2)//12
	def CALENDAR_MONTH(T,val_mth):
		return (val_mth+T+1+10)%12+1

class BondFunctionsAsset11(BondFunctions):
	complex_derivations=dict(BondFunctions.complex_derivations)
	complex_derivations.update({
		'BOND_VALUES': {'outvars':{
			  'MV': {'type': float64, 'shape':2,'2d':'VT'}
			, 'NPV_BY': {'type': float64, 'shape':2,'2d':'VT'}
			, 'CF': {'type': float64, 'shape':2,'2d':'VT'}
			, 'COUPON_RATE_RESET_T': {'type': float64, 'shape':2,'2d':'VT'}
			}
		}
	})
	
	def BOND_VALUES(VT,FX_INDEX,MV_T0,BVCF,ACCRUED_INTEREST_T0,CALENDAR_YEAR,MV_NUMPERIODS,COUPON_FREQ,COUPON_PC,COUPON_RATE_RESET,ASSET_TYPE,REF_RATE_MARGIN_PC
			,PURCHASE_T0,YIELD_IDX,CALENDAR_MONTH,REDEMP_YEAR,REDEMP_MONTH,BOND_REMAINING_PERIOD_VT,ASSET_SCALAR
			,YIELDS,BOOK_YIELD_NPVRATE,REDEMPTION,REDEMP_AMT,val_yr
			,COUPON_RATE_RESET_T,MV,NPV_BY,CF
		   ):
		if (COUPON_FREQ[0])==0:
			inner_COUPON_RATE2 = 0
		else:
			inner_COUPON_RATE2 = COUPON_PC[0]/(100*COUPON_FREQ[0])
		for t in prange(PURCHASE_T0[0],min(BOND_REMAINING_PERIOD_VT[1+PURCHASE_T0[0]]+2,MV_NUMPERIODS[0])):
			if t==PURCHASE_T0[0]:
				MV[t]=MV_T0[0]
				NPV_BY[t]=-(BVCF[0]+ACCRUED_INTEREST_T0[0])
				COUPON_RATE_RESET_T[t]=COUPON_RATE_RESET[0]
			else:
				for ty in range(t,min(t+BOND_REMAINING_PERIOD_VT[t]+1,CALENDAR_YEAR.shape[0])):
					##########
					#CALCULALATE THE COUPON AMOUNT
					if COUPON_FREQ[0]==0:
						inner_COUPON2 = 0
					else:
						if BOND_REMAINING_PERIOD_VT[t]-(ty-t)>=0:
							if (BOND_REMAINING_PERIOD_VT[t]-(ty-t))%(12/COUPON_FREQ[0])==0:
								inner_COUPON2 = FX_INDEX[t-PURCHASE_T0[0]]*inner_COUPON_RATE2*REDEMP_AMT[0]*ASSET_SCALAR[0]
							else:
								inner_COUPON2 = 0
						else:
							inner_COUPON2 = 0
					#END COUPON CALC
					##########
					if ty>t:
						MV[t]=MV[t]+(inner_COUPON2 + FX_INDEX[t-PURCHASE_T0[0]]*REDEMPTION[ty]*ASSET_SCALAR[0])*YIELDS[YIELD_IDX[0]+t,ty-t]
						NPV_BY[t]=NPV_BY[t]+(inner_COUPON2 + FX_INDEX[t-PURCHASE_T0[0]]*REDEMPTION[ty]*ASSET_SCALAR[0])/(1+BOOK_YIELD_NPVRATE[0])**(ty-t)
					if ty==(t+1):
						COUPON_RATE_RESET_T[t]=inner_COUPON_RATE2*COUPON_FREQ[0]
					if ty==t:
						CF[t]=inner_COUPON2 + FX_INDEX[t-PURCHASE_T0[0]]*REDEMPTION[ty]*ASSET_SCALAR[0]

class NewBondFunctions(BondFunctions):
	single_derivations=dict(BondFunctions.single_derivations)
	single_derivations.update(dict(
		  PURCHASE_MONTH = {'type': int32, 'numba':'v','shape':1}
		, REDEMP_MONTH = {'type': int32, 'numba':'v','shape':1}
		, REDEMP_YEAR = {'type': int32, 'numba':'v','shape':1}
		, PURCHASE_T0 = {'type': int32, 'numba':'v','shape':1}
		, COUPON_RATE_RESET = {'type': float64, 'numba':'v','shape':1}
		, ACCRUED_INTEREST_T0 = {'type': float64, 'numba':'v','shape':1}
		, BVCF = {'type': float64, 'numba':'v','shape':1}
		, BOOK_YIELD_NPVRATE = {'type': float64, 'numba':'v','shape':1}
		
		, NPV_CF_T0 = {'type': float64, 'numba':'j','shape':1}
		, BV_CLEAN_T0 = {'type': float64, 'numba':'v','shape':1} #this overrides a 'v' style input in the base class
		
		, CALENDAR_YEAR = {'type': int32, 'numba':'v','shape':2}
		, CALENDAR_MONTH = {'type': int32, 'numba':'v','shape':2}
		, REDEMPTION_TIMING = {'type': int32, 'numba':'v','shape':2}
		, REDEMPTION = {'type': float64, 'numba':'v','shape':2}
		, COUPON_TIMING = {'type': int32, 'numba':'v','shape':2}
		, CF_T0 = {'type': float64, 'numba':'v','shape':2}
		, BOND_REMAINING_PERIOD_VT = {'type': int32, 'numba':'v','shape':2}
		, CALENDAR_YEAR_VT = {'type': int32, 'numba':'v','shape':2}
		, CALENDAR_MONTH_VT = {'type': int32, 'numba':'v','shape':2}
		
		, COUPON_RATE_T0 = {'type': float64, 'numba':'j','shape':2}	
	))
	del single_derivations['REDEMPTION_TERM'] #this comes from the MPF file instead
	
	complex_derivations=dict(BondFunctions.complex_derivations)
	complex_derivations.update(
		INITIAL_COUPON_RATE = {'outvars':
			{
				'inner_ICR_num': {'type': float64, 'shape':1}
				,'inner_ICR_denom': {'type': float64, 'shape':1}
				,'COUPON_PC': {'type': float64, 'shape':1}
			}
		}
	)

	summaries=dict(BondFunctions.summaries)
	summaries.update({
	})
	
	def NPV_CF_T0(CF_T0,PURCHASE_T0,BOOK_YIELD_NPVRATE,NPV_CF_T0):
		for t in range(PURCHASE_T0[0]+1,CF_T0.shape[0]):
			NPV_CF_T0[0] = NPV_CF_T0[0] + CF_T0[t]/(1+BOOK_YIELD_NPVRATE[0])**(t-PURCHASE_T0[0])
	
	def BV_CLEAN_T0(NPV_CF_T0):
		return NPV_CF_T0 + 0 #ACCURED_INTEREST
	
	
	def REDEMPTION(CALENDAR_YEAR,REDEMP_YEAR,CALENDAR_MONTH,REDEMP_MONTH,REDEMP_AMT):
		if (CALENDAR_MONTH==REDEMP_MONTH) and (CALENDAR_YEAR==REDEMP_YEAR):
			return REDEMP_AMT
		else:
			return 0
	
	def BOOK_YIELD_NPVRATE(COUPON_PC):
		return ((1+COUPON_PC/100)**(1/12)-1)
	
	def BOND_REMAINING_PERIOD_VT(REDEMP_MONTH,REDEMP_YEAR,CALENDAR_MONTH_VT,CALENDAR_YEAR_VT):
		if CALENDAR_YEAR_VT>REDEMP_YEAR:
			return 0
		elif CALENDAR_YEAR_VT==REDEMP_YEAR:
			if CALENDAR_MONTH_VT>REDEMP_MONTH:
				return 0
			else:
				return REDEMP_MONTH-CALENDAR_MONTH_VT
		else:
			return (REDEMP_YEAR-CALENDAR_YEAR_VT)*12+(REDEMP_MONTH-CALENDAR_MONTH_VT)
	
	def CALENDAR_YEAR_VT(VT,val_yr,val_mth):
		return (val_yr*12+val_mth+VT+1-2)//12
	def CALENDAR_MONTH_VT(VT,val_mth):
		return (val_mth+VT+1+10)%12+1
	
	def COUPON_RATE_RESET(COUPON_PC,COUPON_FREQ):
		return COUPON_PC*COUPON_FREQ/100

	def CF_T0(COUPON_TIMING,REDEMPTION_TIMING
				,COUPON_RATE_T0,REDEMP_AMT,ASSET_SCALAR):
		return ASSET_SCALAR*REDEMP_AMT*(COUPON_TIMING*COUPON_RATE_T0+REDEMPTION_TIMING)

	def COUPON_RATE_T0(PURCHASE_T0,YIELDS,YIELD_IDX,ASSET_TYPE,COUPON_PC,COUPON_FREQ,CALENDAR_YEAR,CALENDAR_MONTH,REDEMP_YEAR,REDEMP_MONTH,REF_RATE_MARGIN_PC,COUPON_RATE_T0):
		for t in range(PURCHASE_T0[0],COUPON_RATE_T0.shape[0]): #cf time
			if (COUPON_FREQ[0]==0) or (CALENDAR_YEAR[t]>REDEMP_YEAR[0]):
				COUPON_RATE_T0[t]=0
			else:
				if ASSET_TYPE[0]==11:
					COUPON_RATE_T0[t]=COUPON_PC[0]/(100*COUPON_FREQ[0])
				elif ASSET_TYPE[0]==21:
					if t<PURCHASE_T0[0]:
						COUPON_RATE_T0[t]=0
					elif t==PURCHASE_T0[0]:
						COUPON_RATE_T0[t]=COUPON_PC[0]/(100*COUPON_FREQ[0])
					elif (CALENDAR_MONTH[t]-REDEMP_MONTH[0]-1)%(12/COUPON_FREQ[0])==0:
						COUPON_RATE_T0[t]=(((YIELDS[YIELD_IDX[0]+PURCHASE_T0[0],t-PURCHASE_T0[0]-1]/YIELDS[YIELD_IDX[0]+PURCHASE_T0[0],t-PURCHASE_T0[0]])**12+REF_RATE_MARGIN_PC[0]/100)**(1/COUPON_FREQ[0])-1)
					else:
						COUPON_RATE_T0[t]=COUPON_RATE_T0[t-1]
				else:
					COUPON_RATE_T0[t]=0

	def INITIAL_COUPON_RATE(PURCHASE_T0,ASSET_TYPE,YIELDS,YIELD_IDX,COUPON_TIMING,REDEMPTION_TIMING,REF_RATE_MARGIN_PC
		,inner_ICR_num, inner_ICR_denom,COUPON_PC):
		for t in range(PURCHASE_T0[0],COUPON_TIMING.shape[0]):
			if ASSET_TYPE[0]==11:
				if t>PURCHASE_T0[0]:
					inner_ICR_num[0]=inner_ICR_num[0]+YIELDS[YIELD_IDX[0]+PURCHASE_T0[0],t-PURCHASE_T0[0]]*REDEMPTION_TIMING[t]
					inner_ICR_denom[0]=inner_ICR_denom[0]+YIELDS[YIELD_IDX[0]+PURCHASE_T0[0],t-PURCHASE_T0[0]]*COUPON_TIMING[t]
			else:
				if t==(PURCHASE_T0[0]+1):
					COUPON_PC[0]=100*(YIELDS[YIELD_IDX[0]+PURCHASE_T0[0],t-PURCHASE_T0[0]]**(-12) -1 + REF_RATE_MARGIN_PC[0]/100)
		if ASSET_TYPE[0]==11:
			if inner_ICR_denom[0]==0:
				COUPON_PC[0]=0
			else:
				COUPON_PC[0]=100*(1-inner_ICR_num[0])/inner_ICR_denom[0]
	
	def COUPON_TIMING(T,PURCHASE_T0,REDEMPTION_TERM,COUPON_FREQ):
		if T>=PURCHASE_T0:
			if T<=(REDEMPTION_TERM+PURCHASE_T0):
				if COUPON_FREQ==0:
					return 0
				else:
					if (REDEMPTION_TERM+PURCHASE_T0-T)%(12/COUPON_FREQ)==0:
						return 1
					else:
						return 0
			else:
				return 0
		else:
			return 0
	
	def REDEMPTION_TIMING(CALENDAR_YEAR,CALENDAR_MONTH,REDEMP_YEAR,REDEMP_MONTH):
		if (CALENDAR_YEAR==REDEMP_YEAR) and (CALENDAR_MONTH==REDEMP_MONTH):
			return 1
		else:
			return 0
	
	def PURCHASE_T0(PURCHASE_YEAR,val_yr):
		return max(0,(PURCHASE_YEAR-val_yr)*12)
	def REDEMP_YEAR(REDEMPTION_TERM,PURCHASE_YEAR):
		return REDEMPTION_TERM//12 + PURCHASE_YEAR
	def REDEMP_MONTH(REDEMPTION_TERM):
		return (REDEMPTION_TERM-1)%12+1
	def PURCHASE_MONTH(PURCHASE_YEAR):
		return 12
	def CALENDAR_YEAR(T,val_yr,val_mth):
		return (val_yr*12+val_mth+T+1-2)//12
	def CALENDAR_MONTH(T,val_mth):
		return (val_mth+T+1+10)%12+1
		
	def ACCRUED_INTEREST_T0(REDEMP_YEAR):
		return 0
	def BVCF(REDEMP_YEAR):
		return 0

	def BOND_REMAINING_PERIOD_T0(REDEMP_YEAR,REDEMP_MONTH,val_yr,val_mth):
		return (REDEMP_YEAR-val_yr)*12+(REDEMP_MONTH-val_mth)
