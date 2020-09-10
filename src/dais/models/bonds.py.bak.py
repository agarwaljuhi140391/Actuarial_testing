import pandas as pd
import numpy as np
from numpy import int32, int64, float64
from numba import vectorize, prange
from .base import BaseFunctions
from .simple_dispatcher import irr_est

# class BondFunctions(BaseFunctions):
class BondFunctions2():
	single_derivations=dict(
		
		REMAIN_YR = {'type': int32, 'numba':'j','shape':1}
	
		, CALENDAR_YEAR = {'type': int32, 'numba':'v','shape':2}
		, FX_RATE = {'type': float64, 'numba':'v','shape':2}
		, MARKET_VALUE = {'type': float64, 'numba':'v','shape':2}
		, INTEREST_ACCRU_INDEX = {'type': int32, 'numba':'v','shape':2}
		, ACCRUED_INTEREST = {'type': float64, 'numba':'v','shape':2}
		, BOND_REMAINING_PERIOD = {'type': int32, 'numba':'v','shape':2}
		, ABV_PER_UNIT = {'type': float64, 'numba':'v','shape':2}
		, ABV = {'type': float64, 'numba':'v','shape':2}
	)
	
	complex_derivations={}
	
	summaries={}
	
	mappings={
		'PRESENT_VALUES': {'autocapcol':True,'type':float64,'shape':2,'mapping':{'SEGMENT_NO':'SEGMENT_NO','col':'CALENDAR_YEAR'}}
		, 'VALNDEF': {'autocapcol': True,'type':float64, 'shape':2, 'mapping':{'SIMULATION':'SIMULATION','ECONOMY':'ECONOMY','TERM':'VALNDEFN_TERM','col':'CALENDAR_YEAR'}}
		, 'BASE_VALNDEF': {'autocapcol': True,'type':float64, 'shape':2, 'mapping':{'SIMULATION':'SIMULATION','ECONOMY':'BASE_ECONOMY','TERM':'VALNDEFN_TERM','col':'CALENDAR_YEAR'}}
		, 'RET_CASH_ECON_PC': {'autocapcol': True,'type':float64, 'shape':2, 'mapping':{'Economy':'ECONOMY','col':'CALENDAR_YEAR'}}
		, 'SSA_BOND_PROPHELD': {'autocapcol': True,'type':float64, 'shape':2, 'mapping':{'bondnumber':'SEGMENT_NO','col':'CALENDAR_YEAR'}}
	}
	
	def ABV(ABV_PER_UNIT,SSA_BOND_PROPHELD,FX_RATE):
		return ABV_PER_UNIT*SSA_BOND_PROPHELD*FX_RATE
	
	def ABV_PER_UNIT(CALENDAR_YEAR,val_yr,I_ABV,I_ACC_INT,ACCRUED_INTEREST,BOND_REMAINING_PERIOD,REMAIN_YR,REDEMP_AMT):
		if CALENDAR_YEAR==val_yr:
			return I_ABV+I_ACC_INT+ACCRUED_INTEREST
		else:
			return I_ABV*BOND_REMAINING_PERIOD/REMAIN_YR + REDEMP_AMT*(REMAIN_YR-BOND_REMAINING_PERIOD)/REMAIN_YR + ACCRUED_INTEREST
	
	def REMAIN_YR(BOND_REMAINING_PERIOD,REMAIN_YR):
		for y in prange(BOND_REMAINING_PERIOD.shape[0]):
			REMAIN_YR[y,0]=BOND_REMAINING_PERIOD[y,0]
	
	def ACCRUED_INTEREST(INTEREST_ACCRU_INDEX,COUPON_PC,COUPON_FREQ,REDEMP_AMT):
		if COUPON_FREQ==0:
			return 0
		else:
			return REDEMP_AMT*COUPON_PC/100/COUPON_FREQ*INTEREST_ACCRU_INDEX/(12/COUPON_FREQ)
	
	def INTEREST_ACCRU_INDEX(COUPON_FREQ,REDEMP_MONTH,val_yr,CALENDAR_YEAR):
		if CALENDAR_YEAR==val_yr:
			return 0
		else:
			return (12-REDEMP_MONTH)%(12/COUPON_FREQ)
	
	def BOND_REMAINING_PERIOD(REDEMP_YEAR,REDEMP_MONTH,CALENDAR_YEAR):
		if CALENDAR_YEAR>=REDEMP_YEAR:
			return 0
		else:
			return (REDEMP_YEAR-CALENDAR_YEAR)*12+(REDEMP_MONTH-12)
	
	def MARKET_VALUE(PRESENT_VALUES,FX_RATE,SSA_BOND_PROPHELD):
		return PRESENT_VALUES*FX_RATE*SSA_BOND_PROPHELD
	
	def FX_RATE(VALNDEF,BASE_VALNDEF):
		if BASE_VALNDEF==0:
			return 0
		else:
			return VALNDEF/BASE_VALNDEF
	
	def CALENDAR_YEAR(T,val_yr):
		return val_yr+T

class BondFunctions():
	single_derivations=dict(
	# **BaseFunctions.single_derivations,
		COUPON_PLACE = {'type': float64, 'numba':'v','shape':1}
		, COUPON_RATE = {'type': float64, 'numba':'v','shape':1}
		, BOND_REMAINING_PERIOD_T0 = {'type': int32, 'numba':'v','shape':1}
		, ACCRUED_INTEREST_T0 = {'type': float64, 'numba':'v','shape':1}
		, BOOK_YIELD_NPVRATE = {'type': float64, 'numba':'v','shape':1}

		, NET_PRESENT_VALUE_T0 = {'type': float64, 'numba':'j','shape':1}
		, BOOK_YIELD = {'type': float64, 'numba':'j','shape':1}
		, MV_T0 = {'type': float64, 'numba':'j','shape':1}
		, COUPON_RATE_RESET = {'type': float64, 'numba':'j','shape':1}

		, CALENDAR_YEAR = {'type': int32, 'numba':'v','shape':2}
		, BVCF = {'type': float64, 'numba':'v','shape':2}
		# , CALENDAR_YEAR_SUB1 = {'type': int32, 'numba':'v','shape':2}
		, CALENDAR_MONTH = {'type': int32, 'numba':'v','shape':2}
		# , FX_START = {'type': float64, 'numba':'v','shape':2}
		, COUPON = {'type': float64, 'numba':'v','shape':2}
		, CF_T0 = {'type': float64, 'numba':'v','shape':2}
		, REDEMPTION = {'type': float64, 'numba':'v','shape':2}
		# , TCBA = {'type': float64, 'numba':'v','shape':2}
		# , ADJ_FACTOR = {'type': float64, 'numba':'v','shape':2}
		# , TCAA = {'type': float64, 'numba':'v','shape':2}
		, BOND_REMAINING_PERIOD = {'type': float64, 'numba':'v','shape':2}
		# , ZCB2 = {'type': float64, 'numba':'v','shape':2}

		# , FX_MID = {'type': float64, 'numba':'j','shape':2}
		# , TCAA_ANN = {'type': float64, 'numba':'j','shape':2}
		, COUPON_RATE_T0 = {'type': float64, 'numba':'j','shape':2}

		# , NET_PRESENT_VALUE = {'type': float64, 'numba':'jbond','shape':2}
		# , PRESENT_VALUES = {'type': float64, 'numba':'jbond','shape':2}
	)
	
	complex_derivations={
		BOND_VALUES: {'outvars':{
			'MV': {'type': float64, 'shape':2,'2d':'VT'}
			, 'NPV_BY': {'type': float64, 'shape':2,'2d':'VT'}
			, 'inner_COUPON_RATE': {'type': float64, 'shape':2,'2d':'VT'}
			, 'inner_COUPON': {'type': float64, 'shape':2,'2d':'VT'}
	}

	summaries={}

	mappings={
		  # 'ZCB_BASE_FRATE': {'source': 'ECON', 'type':float64, 'shape':1, 'mapping':{'Economy':'ECONOMY'}}
		'YIELD_IDX': {'source':'YIELDS', 'type':int32, 'shape':1, 'mapping':{'ECONOMY':'ECONOMY','SIMULATION':'SIMULATION'}}

		, 'VALNDEF': {'autocapcol': True,'type':float64, 'shape':2, 'mapping':{'SIMULATION':'SIMULATION','ECONOMY':'ECONOMY','TERM':'VALNDEFN_TERM','col':'CALENDAR_YEAR_SUB1'}}
		, 'BASE_VALNDEF': {'autocapcol': True,'type':float64, 'shape':2, 'mapping':{'SIMULATION':'SIMULATION','ECONOMY':'BASE_ECONOMY','TERM':'VALNDEFN_TERM','col':'CALENDAR_YEAR_SUB1'}}
		, 'RET_CASH_ECON_PC': {'autocapcol': True,'type':float64, 'shape':2, 'mapping':{'Economy':'ECONOMY','col':'CALENDAR_YEAR'}}
		, 'SSA_BOND_PROPHELD': {'autocapcol': True,'type':float64, 'shape':2, 'mapping':{'bondnumber':'SEGMENT_NO','col':'CALENDAR_YEAR_SUB1'}}
	}

	def BOND_REMAINING_PERIOD(REDEMP_YEAR,REDEMP_MONTH,CALENDAR_YEAR,CALENDAR_MONTH):
		if CALENDAR_YEAR>REDEMP_YEAR:
			return 0
		elif CALENDAR_YEAR==REDEMP_YEAR:
			if CALENDAR_MONTH>REDEMP_MONTH:
				return 0
			else:
				return REDEMP_MONTH-CALENDAR_MONTH
		else:
			return (REDEMP_YEAR-CALENDAR_YEAR)*12+(REDEMP_MONTH-CALENDAR_MONTH)
			
	def COUPON_RATE_RESET(COUPON_RATE_T0,COUPON_FREQ,COUPON_RATE_RESET):
		for y in prange(COUPON_RATE_T0.shape[0]):
			COUPON_RATE_RESET[y,0]=COUPON_FREQ[y,0]*COUPON_RATE_T0[y,1]
		
	def BOOK_YIELD_NPVRATE(BOOK_YIELD):
		return (1+BOOK_YIELD)**(1/12)-1


	def BOOK_YIELD(BVCF,BOOK_YIELD):
		for y in prange(BOOK_YIELD.shape[0]):
			BOOK_YIELD[y,0]=(1+irr_est(BVCF[y]))**12-1
	
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

	def BOND_REMAINING_PERIOD_T0(REDEMP_YEAR,REDEMP_MONTH,val_yr,val_mth):
		return (REDEMP_YEAR-val_yr)*12+(REDEMP_MONTH-val_mth)

	def COUPON_PLACE(REDEMP_YEAR):
		return 0
	def COUPON_RATE(REDEMP_YEAR):
		return 0
		
	def MV_T0(CF_T0,YIELDS,YIELD_IDX,MV_T0):
		for y in prange(CF_T0.shape[0]): #bonds
			for t in range(CF_T0.shape[1]): #cf time
				MV_T0[y,0]=MV_T0[y,0]+YIELDS[YIELD_IDX[y,0],t]*CF_T0[y,t]
		
	def CF_T0(CALENDAR_YEAR,REDEMP_YEAR,CALENDAR_MONTH,REDEMP_MONTH,COUPON_FREQ
				,COUPON_RATE_T0,REDEMP_AMT,ASSET_SCALAR,REDEMPTION):
		if (COUPON_FREQ==0) or (CALENDAR_YEAR>REDEMP_YEAR) or ((CALENDAR_YEAR==REDEMP_YEAR) and (CALENDAR_MONTH>REDEMP_MONTH)):
			return 0
		elif (CALENDAR_MONTH-REDEMP_MONTH)%(12/COUPON_FREQ)==0:
			return ASSET_SCALAR*(REDEMP_AMT*COUPON_RATE_T0+REDEMPTION)
		else:
			return ASSET_SCALAR*REDEMPTION
	
	def COUPON_RATE_T0(YIELDS,YIELD_IDX,ASSET_TYPE,COUPON_PC,COUPON_FREQ,CALENDAR_YEAR,CALENDAR_MONTH,REDEMP_YEAR,REDEMP_MONTH,REF_RATE_MARGIN_PC,COUPON_RATE_T0):
		for y in prange(COUPON_RATE_T0.shape[0]): #bonds
			for t in range(COUPON_RATE_T0.shape[1]): #cf time
				if (COUPON_FREQ[y,0]==0) or (CALENDAR_YEAR[y,t]>REDEMP_YEAR[y,0]):
					COUPON_RATE_T0[y,t]=0
				else:
					if ASSET_TYPE[y,0]==11:
						COUPON_RATE_T0[y,t]=COUPON_PC[y,0]/(100*COUPON_FREQ[y,0])
					elif ASSET_TYPE[y,0]==21:
						if t==0:
							COUPON_RATE_T0[y,t]=0
						elif t==1:
							COUPON_RATE_T0[y,t]=COUPON_PC[y,0]/(100*COUPON_FREQ[y,0])
						elif (CALENDAR_MONTH[y,t]-REDEMP_MONTH[y,0]-1)%(12/COUPON_FREQ[y,0])==0:
							COUPON_RATE_T0[y,t]=(((YIELDS[YIELD_IDX[y,0],t-1]/YIELDS[YIELD_IDX[y,0],t])**12+REF_RATE_MARGIN_PC[y,0]/100)**(1/COUPON_FREQ[y,0])-1)
						else:
							COUPON_RATE_T0[y,t]=COUPON_RATE_T0[y,t-1]
					else:
						COUPON_RATE_T0[y,t]=0

	def NET_PRESENT_VALUE_T0(MV_NUMPERIODS,YIELDS,YIELD_IDX
							,CALENDAR_YEAR,CALENDAR_MONTH,REDEMP_MONTH,REDEMP_YEAR
							,COUPON_PC,REDEMP_AMT,COUPON_FREQ,ASSET_TYPE,ASSET_SCALAR
							,COUPON_PLACE,COUPON_RATE,REDEMPTION,REF_RATE_MARGIN_PC
							,NET_PRESENT_VALUE_T0):
		for y in prange(NET_PRESENT_VALUE_T0.shape[0]): #bonds
	#         t=0
			if ASSET_TYPE[y,0]==11:
				if COUPON_FREQ[y,0]==0:
					COUPON_RATE[y,0]=0
				else:
					COUPON_RATE[y,0]=COUPON_PC[y,0]/(100*COUPON_FREQ[y,0])
			for ty in range(CALENDAR_YEAR.shape[1]): #output time
				if COUPON_FREQ[y,0]==0:
					COUPON_RATE[y,0]=0
				else:
					if ASSET_TYPE[y,0]==21:
						if ty==0:
							COUPON_RATE[y,0]=0
						if (ty==1):
							COUPON_RATE[y,0]=COUPON_PC[y,0]/(100*COUPON_FREQ[y,0])
						elif (CALENDAR_MONTH[y,ty]-REDEMP_MONTH[y,0]-1)%(12/COUPON_FREQ[y,0])==0:
							COUPON_RATE[y,0]=(((YIELDS[YIELD_IDX[y,0],ty-1]/YIELDS[YIELD_IDX[y,0],ty])**12+REF_RATE_MARGIN_PC[y,0]/100)**(1/COUPON_FREQ[y,0])-1)
						else:
							pass #no change
					else:
						COUPON_RATE[y,0]=0 #unknown asset type
				if (ty==0) or (CALENDAR_YEAR[y,ty]>REDEMP_YEAR[y,0]) or ((CALENDAR_YEAR[y,ty]==REDEMP_YEAR[y,0]) and (CALENDAR_MONTH[y,ty]>REDEMP_MONTH[y,0])):
					COUPON_PLACE[y,0]=0
				else:
					if (CALENDAR_MONTH[y,ty]-REDEMP_MONTH[y,0])%(12/COUPON_FREQ[y,0])==0:
						COUPON_PLACE[y,0]=COUPON_RATE[y,0]*ASSET_SCALAR[y,0]*REDEMP_AMT[y,0]
					else:
						COUPON_PLACE[y,0]=0
	#             print(y,ty,ASSET_TYPE[y,0],REDEMP_MONTH[y,0],CALENDAR_MONTH[y,ty]
	#                   , COUPON_RATE[y,0]
	#                   , YIELDS[YIELD_IDX[y,0],ty]
	#                   ,COUPON_PLACE[y,0]
	#                  ,REDEMPTION[y,ty]*ASSET_SCALAR[y,0])
				NET_PRESENT_VALUE_T0[y,0]=NET_PRESENT_VALUE_T0[y,0]+YIELDS[YIELD_IDX[y,0],ty]*(COUPON_PLACE[y,0]+ASSET_SCALAR[y,0]*REDEMPTION[y,ty])

		
	def NET_PRESENT_VALUE(MV_NUMPERIODS,YIELDS,YIELD_IDX
							,CALENDAR_YEAR,CALENDAR_MONTH,REDEMP_MONTH,REDEMP_YEAR
							,COUPON_PC,REDEMP_AMT,COUPON_FREQ,ASSET_TYPE,ASSET_SCALAR
							,COUPON_PLACE,REDEMPTION,REF_RATE_MARGIN_PC
							,NET_PRESENT_VALUE):
		for y in prange(NET_PRESENT_VALUE.shape[0]): #bonds
			for t in range(NET_PRESENT_VALUE.shape[1]): #CF time (1200?)
				for ty in range(t,CALENDAR_YEAR.shape[1]): #output time
					if (CALENDAR_YEAR[y,ty]>REDEMP_YEAR[y,0]) or ((CALENDAR_YEAR[y,ty]==REDEMP_YEAR[y,0]) and (CALENDAR_MONTH[y,ty]>REDEMP_MONTH[y,0])):
						COUPON_PLACE[y,0]=0
					else:
						if COUPON_FREQ[y,0]==0:
							COUPON_PLACE[y,0]=0
						else:
							if (CALENDAR_MONTH[y,ty]-REDEMP_MONTH[y,0])%(12/COUPON_FREQ[y,0])==0:
								if ASSET_TYPE[y,0]==11:
									COUPON_PLACE[y,0]=REDEMP_AMT[y,0]*COUPON_PC[y,0]/(100*COUPON_FREQ[y,0])
								elif ASSET_TYPE[y,0]==21:
									if ty==0:
										COUPON_PLACE[y,0]=0
									elif (ty==t) or (ty==1):
										COUPON_PLACE[y,0]=REDEMP_AMT[y,0]*COUPON_PC[y,0]/(100*COUPON_FREQ[y,0])
									else:
										COUPON_PLACE[y,0]=REDEMP_AMT[y,0]*(((YIELDS[YIELD_IDX[y,0]+t,ty-t-1]/YIELDS[YIELD_IDX[y,0]+t,ty-t])**12+REF_RATE_MARGIN_PC[y,0]/100)**(1/COUPON_FREQ[y,0])-1)
								else:
									COUPON_PLACE[y,0]=0
							else:
								COUPON_PLACE[y,0]=0
					if ty>t:
						NET_PRESENT_VALUE[y,t]=NET_PRESENT_VALUE[y,t]+ASSET_SCALAR[y,0]*YIELDS[YIELD_IDX[y,0]+t,ty-t]*(COUPON_PLACE[y,0]+REDEMPTION[y,ty])

	def PRESENT_VALUES(YIELDS,YIELD_IDX,TCBA,NUMYEARS,PRESENT_VALUES):
		for y in prange(TCBA.shape[0]): #bonds
			for t in prange(TCBA.shape[1]): #CF time (1200)
				for ty in range(YIELDS.shape[0]): #term time (100)
					if (t+12*ty)<TCBA.shape[1]:
						PRESENT_VALUES[y,ty]=PRESENT_VALUES[y,ty]+YIELDS[YIELD_IDX[y,0]+ty,t]*TCBA[y,t+12*ty]

	def TCAA_ANN(TCAA,CALENDAR_MONTH,TCAA_ANN):
		for y in prange(TCAA_ANN.shape[0]):
			for t in range(TCAA_ANN.shape[1]):
				TCAA_ANN[y,t]=0
				if (CALENDAR_MONTH[y,t]==12):
					for tt in range(12):
						TCAA_ANN[y,t]=TCAA_ANN[y,t]+TCAA[y,t-tt]

	def TCAA(TCBA,ADJ_FACTOR,FX_MID,SSA_BOND_PROPHELD,BOND_ASSET_SCALAR):
		return TCBA*ADJ_FACTOR*FX_MID*SSA_BOND_PROPHELD*BOND_ASSET_SCALAR

	def ADJ_FACTOR(CALENDAR_YEAR,CALENDAR_MONTH,val_yr,RET_CASH_ECON_PC):
		if CALENDAR_YEAR==val_yr:
			return (1+RET_CASH_ECON_PC/100)**-0.5
		else:
			return (1+RET_CASH_ECON_PC/100)**((6-CALENDAR_MONTH)/12)

	def TCBA(COUPON,REDEMPTION,ASSET_SCALAR):
		return (COUPON+REDEMPTION)*ASSET_SCALAR

	def REDEMPTION(CALENDAR_YEAR,REDEMP_YEAR,CALENDAR_MONTH,REDEMP_MONTH,REDEMP_AMT):
		if (CALENDAR_MONTH==REDEMP_MONTH) and (CALENDAR_YEAR==REDEMP_YEAR):
			return REDEMP_AMT
		else:
			return 0

	# def COUPON2(COUPON_FREQ,REDEMP_AMT,COUPON_PC,CALENDAR_YEAR,CALENDAR_MONTH,REDEMP_MONTH,REDEMP_YEAR,ASSET_TYPE):
		# if (CALENDAR_YEAR>REDEMP_YEAR) or ((CALENDAR_YEAR==REDEMP_YEAR) and (CALENDAR_MONTH>REDEMP_MONTH)):
			# return 0
		# else:
			# if COUPON_FREQ==0:
				# return 0
			# else:
				# if (CALENDAR_MONTH-REDEMP_MONTH)%(12/COUPON_FREQ)==0:
					# if ASSET_TYPE==11:
						# return REDEMP_AMT*COUPON_PC/(100*COUPON_FREQ)
					# elif ASSET_TYPE==21:

				# else:
					# return 0


	def COUPON(COUPON_FREQ,REDEMP_AMT,COUPON_PC,CALENDAR_YEAR,CALENDAR_MONTH,REDEMP_MONTH,REDEMP_YEAR):
		if (CALENDAR_YEAR>REDEMP_YEAR) or ((CALENDAR_YEAR==REDEMP_YEAR) and (CALENDAR_MONTH>REDEMP_MONTH)):
			return 0
		else:
			if COUPON_FREQ==0:
				return 0
			else:
				if (CALENDAR_MONTH-REDEMP_MONTH)%(12/COUPON_FREQ)==0:
					return REDEMP_AMT*COUPON_PC/(100*COUPON_FREQ)
				else:
					return 0

	def FX_MID(FX_START,FX_MID):
		for y in prange(FX_MID.shape[0]):
			for t in range(FX_MID.shape[1]):
				if t>(FX_MID.shape[1]-12):
					FX_MID[y,t]=1
				else:
					FX_MID[y,t]=(FX_START[y,t]*FX_START[y,t+12])**0.5

	def FX_START(VALNDEF,BASE_VALNDEF):
		if BASE_VALNDEF==0:
			return 0
		else:
			return VALNDEF/BASE_VALNDEF

	def CALENDAR_YEAR_SUB1(CALENDAR_YEAR):
		return CALENDAR_YEAR-1
	def CALENDAR_YEAR(T,val_yr,val_mth):
		return (val_yr*12+val_mth+T+1-2)//12
	def CALENDAR_MONTH(T,val_mth):
		return (val_mth+T+1+10)%12+1
