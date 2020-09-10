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
		pcEQUITY_RET = {'type': float64, 'alignment':'PC', 'numba':'v','shape':2,'loop':1}
# 		, pCALENDAR_YEAR = {'type': int32, 'alignment':'P', 'numba':'v','shape':2}
# 		, pcCALENDAR_YEAR = {'type': int32, 'alignment':'P', 'numba':'v','shape':2}
# 		, pcyCALENDAR_YEAR = {'type': int32, 'alignment':'P', 'numba':'v','shape':2}
# 		, pMONTH_T = {'type': int32, 'alignment':'P', 'numba':'v','shape':2}
# 		, pcMONTH_T = {'type': int32, 'alignment':'P', 'numba':'v','shape':2}
# 		, pcyMONTH_T = {'type': int32, 'alignment':'P', 'numba':'v','shape':2}
		

		, pTOTAL_FAV = {'type': float64, 'alignment':'P', 'numba':'v','shape':2,'loop':1}
		, pBOX = {'type': float64, 'alignment':'P', 'numba':'v','shape':2,'loop':1}
		, new_money_temp = {'type': float64, 'alignment':'P', 'numba':'v','shape':2,'loop':1}
		, cNEW_MONEY_TO_BOND = {'type': float64, 'alignment':'C', 'numba':'v','shape':2,'loop':1}
		, cNEW_MONEY_TO_EQUITY = {'type': float64, 'alignment':'C', 'numba':'v','shape':2,'loop':1}
		, pTOTAL_REALIGN = {'type': float64, 'alignment':'P', 'numba':'v','shape':2,'loop':1}
		# , rfpREALIGN_CASH = {'type': float64, 'alignment':'P', 'numba':'v','shape':2,'loop':1}
		, pREALIGN_TOTAL = {'type': float64, 'alignment':'P', 'numba':'v','shape':2,'loop':1}
		
		, pcBOND_TARGET = {'type': float64, 'alignment':'PC', 'numba':'v','shape':2,'loop':1}
		, pcEQUITY_TARGET = {'type': float64, 'alignment':'PC', 'numba':'v','shape':2,'loop':1}
		, pcREBAL_BUY =  {'type': float64, 'alignment':'PC', 'numba':'v','shape':2,'loop':1}
		, pcREBAL_SELL =  {'type': float64, 'alignment':'PC', 'numba':'v','shape':2,'loop':1}
		, pcBOND_REALIGN_TARGET =  {'type': float64, 'alignment':'PC', 'numba':'v','shape':2,'loop':1}
		, rfpcEQUITY_REALIGN_TARGET =  {'type': float64, 'alignment':'PC', 'numba':'v','shape':2,'loop':1}
		, pcREALIGN_BUY =  {'type': float64, 'alignment':'PC', 'numba':'v','shape':2,'loop':1}
		, pcREALIGN_SELL =  {'type': float64, 'alignment':'PC', 'numba':'v','shape':2,'loop':1}
		
		
# 		, pREBAL_CASH =  {'type': float64, 'alignment':'P', 'numba':'v','shape':2,'loop':1}
		, rfpcyBOND_FAV = {'type': float64, 'alignment':'PCY', 'numba':'j','loop_location':'start','shape':2,'loop':1}
		, rfpcyBOND_CF = {'type': float64, 'alignment':'PCY', 'numba':'j','loop_location':'start','shape':2,'loop':1,'monthly':True}
		, pcyBUYSELL_PCTS = {'type': float64, 'alignment':'PCY', 'numba':'j','loop_location':'end','shape':2,'loop':1}
		, rfpREALIGN_CASH = {'type': float64, 'alignment':'P', 'numba':'v','shape':2,'loop':1}
# 		, rfpcyBOND_MV = {'type': float64, 'alignment':'PCY', 'numba':'v','shape':2,'loop':1}

		, cEQUITY_RET_RATE = {'type': float64, 'alignment':'C', 'numba':'j','shape':2}
		
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
		
		, 'EQUITY_RET_IDX': {'alignment':'C', 'type':float64, 'shape':2, 'mapping':{'CATEGORY':'CATEGORY','col':'rfYEAR'}}
		, 'BONDS_CATEGORY_MIX': {'alignment':'PC', 'type':float64, 'shape':2, 'mapping':{'POOL':'POOL','CATEGORY':'CATEGORY','col':'rfYEAR'}}
		, 'EQUITY_CATEGORY_MIX': {'alignment':'PC', 'type':float64, 'shape':2, 'mapping':{'POOL':'POOL','CATEGORY':'CATEGORY','col':'rfYEAR'}}
		, 'MATH_RES_IF_PL': {'alignment':'P', 'type':float64, 'shape':2, 'monthly': False, 'mapping':{'POOL':'POOL','col':'rfMONTH'}}
		, 'DA_FUND_FL': {'alignment':'P', 'type':float64, 'shape':2, 'monthly': False, 'mapping':{'POOL':'POOL','col':'rfMONTH'}}
		, 'CA_FUND_FL': {'alignment':'P', 'type':float64, 'shape':2, 'monthly': False, 'mapping':{'POOL':'POOL','col':'rfMONTH'}}
		, 'SOLV_MARG_IF_FL': {'alignment':'P', 'type':float64, 'shape':2, 'monthly': False, 'mapping':{'POOL':'POOL','col':'rfMONTH'}}
		, 'CASH_RET_PC': {'alignment':'P', 'type':float64, 'shape':2, 'mapping':{'POOL':'POOL','col':'rfYEAR'}}
		, 'LIAB_CF_PL': {'alignment':'P', 'type':float64, 'shape':2, 'monthly': True, 'mapping':{'POOL':'POOL','col':'rfMONTH'}}
		, 'MV': {'alignment':'PCY', 'type':float64, 'shape':2, 'monthly': False, 'mapping':{'POOL':'POOL','CATEGORY':'CATEGORY','PURCHASE_YEAR_CAPPED':'PURCHASE_YEAR_CAPPED','col':'rfMONTH'}} #Bond naming?
		, 'FAV': {'alignment':'PCY', 'type':float64, 'shape':2, 'monthly': False, 'mapping':{'POOL':'POOL','CATEGORY':'CATEGORY','PURCHASE_YEAR_CAPPED':'PURCHASE_YEAR_CAPPED','col':'rfMONTH'}} #Bond naming?
		, 'CF': {'alignment':'PCY', 'type':float64, 'shape':2, 'monthly': True, 'mapping':{'POOL':'POOL','CATEGORY':'CATEGORY','PURCHASE_YEAR_CAPPED':'PURCHASE_YEAR_CAPPED','col':'rfMONTH'}} #Bond naming?
		, 'pcDENOM': {'alignment':'PC', 'type':float64, 'shape':2, 'monthly': False, 'mapping':{'POOL':'POOL','CATEGORY':'CATEGORY','col':'rfYEAR'}} #Specially derived
		, #hmmm, 4 dimensional inputs? purchase year and calendar year (as well as pool, category)? That would be 2 'col' type variables...
	}

	collapsers={
		'pBOND_CF': {'source':'pcyBOND_CF', 'alignment':'P','func':'sum','monthly': True, 'loop':1}
		# , 'pcBOND_CF': {'source':'pcyBOND_CF', 'alignment':'PC','func':'sum'}
		, 'pBOND_FAV': {'source':'pcyBOND_FAV', 'alignment':'P','func':'sum','loop':1}
		, 'pcBOND_FAV': {'source':'pcyBOND_FAV', 'alignment':'PC','func':'sum','loop':1}
		, 'pEQUITY_RET': {'source':'pcEQUITY_RET', 'alignment':'P','func':'sum','loop':1}
		, 'pcBOND_CF': {'source':'pcyBOND_CF', 'alignment':'PC','func':'sum','monthly': True, 'loop':1}
# 		, 'pcBOND_MV': {'source':'sum_MV', 'alignment':'PC','func':'sum'}
		, 'new_money': {'source':'new_money_temp','alignment':'','func':'sum','loop':1}
# 		, 'pBOND_TARGET': {'source':'pcBOND_TARGET','alignment':'P','func':'sum','loop':1}
# 		, 'pEQUITY_TARGET': {'source':'pcEQUITY_TARGET','alignment':'P','func':'sum','loop':1}
		# , 'pcDENOM': {'source':'pcyBOND_MV', 'alignment':'PC','func':'sum','loop':1}
		, 'pBOND_REALIGN_TARGET': {'source':'pcBOND_REALIGN_TARGET', 'alignment':'P','func':'sum','loop':1}
		, 'pEQUITY_REALIGN_TARGET': {'source':'pcEQUITY_REALIGN_TARGET', 'alignment':'P','func':'sum','loop':1}
		, 'rfpEQUITY_REALIGN_TARGET': {'source':'rfpcEQUITY_REALIGN_TARGET', 'alignment':'P','func':'sum','loop':1}
	}
	
	rollforwards={
		  'pcyBOND_FAV':{'initial':'FAV','initial_start':0,'loop_var':'rfpcyBOND_FAV','loop':1,'alignment':'PCY'}
		, 'pcyBOND_CF':{'initial':'CF','initial_start':0,'loop_var':'rfpcyBOND_CF','loop':1,'alignment':'PCY','monthly':True}
		, 'pREALIGN_CASH':{'initial':'CASH_NOMINAL','loop_var':'rfpREALIGN_CASH','alignment':'P','loop':1}
		, 'pcEQUITY_REALIGN_TARGET':{'initial':'EQUITY_I_FAV','loop_var':'rfpcEQUITY_REALIGN_TARGET','loop':1,'alignment':'PC'}
	}
	
	alignments={
		'P':'POOL'
		, 'C':'CATEGORY'
		, 'Y':'PURCHASE_YEAR_CAPPED'
	}

	def pcyBUYSELL_PCTS(pcREBAL_SELL,T,pcREBAL_BUY,pcREALIGN_SELL,pcREALIGN_BUY,PURCHASE_YEAR_CAPPED,pcyBUYSELL_PCTS):
		#This receives this period and last period. "0" is therefore last period
		if PURCHASE_YEAR_CAPPED[0]==(T[0]+1+2019):
			pcyBUYSELL_PCTS[1] = pcREBAL_BUY[0]*(1+pcREALIGN_SELL[0])+pcREALIGN_BUY[0]
		elif PURCHASE_YEAR_CAPPED[0]<(T[0]+1+2019):
			if T[0]==0:
				pcyBUYSELL_PCTS[1] = 1 * ((1+pcREBAL_SELL[0])*(1+pcREALIGN_SELL[0]))
			else:
				pcyBUYSELL_PCTS[1] = pcyBUYSELL_PCTS[0] * ((1+pcREBAL_SELL[0])*(1+pcREALIGN_SELL[0]))

	
	def rfpcyBOND_FAV(FAV,pcyBUYSELL_PCTS,rfpcyBOND_FAV):
		#This receives this period and last period. "0" is therefore last period
		rfpcyBOND_FAV[0] = FAV[1]*pcyBUYSELL_PCTS[1]

	def rfpcyBOND_CF(CF,pcyBUYSELL_PCTS,rfpcyBOND_CF):
		#This receives this period and last period. "0" is therefore last period
		for mt in range(12):
			rfpcyBOND_CF[mt] = CF[mt+12]*pcyBUYSELL_PCTS[1]

	def new_money_temp(POOL,pBOX):
		if POOL==10:
			return pBOX
		else:
			return 0
	
	def pREALIGN_TOTAL(pBOND_REALIGN_TARGET,pEQUITY_REALIGN_TARGET,pREALIGN_CASH):
		return pBOND_REALIGN_TARGET+pEQUITY_REALIGN_TARGET+pREALIGN_CASH
	
	def rfpREALIGN_CASH(pTOTAL_REALIGN,pBOND_REALIGN_TARGET,rfpEQUITY_REALIGN_TARGET):
		return pTOTAL_REALIGN-pBOND_REALIGN_TARGET-rfpEQUITY_REALIGN_TARGET
	
	def pcREALIGN_SELL(POOL,CATEGORY,pcBOND_REALIGN_TARGET,pcBOND_TARGET):
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
	
	def pcREALIGN_BUY(POOL,pcDENOM,pcBOND_REALIGN_TARGET,pcBOND_TARGET):
		if (pcDENOM==0) or (POOL==10):
			return 0
		else:
			return max(pcBOND_REALIGN_TARGET-pcBOND_TARGET,0)/pcDENOM
	
	def rfpcEQUITY_REALIGN_TARGET(POOL,pcEQUITY_TARGET,pTOTAL_REALIGN,EQUITY_CATEGORY_MIX
				,EQUITY_DRIFT_UPPER_PC,EQUITY_DRIFT_LOWER_PC):
		if POOL==10:
			return max(
					min(pcEQUITY_TARGET
					,EQUITY_DRIFT_UPPER_PC*pTOTAL_REALIGN/100)
					,EQUITY_DRIFT_LOWER_PC*pTOTAL_REALIGN/100)
		else:
			return pTOTAL_REALIGN * EQUITY_CATEGORY_MIX/100

	def pcBOND_REALIGN_TARGET(POOL,pcBOND_TARGET,pTOTAL_REALIGN,BONDS_CATEGORY_MIX
				,BOND_DRIFT_UPPER_PC,BOND_DRIFT_LOWER_PC):
		if POOL==10:
			return max(
					min(pcBOND_TARGET
					,BOND_DRIFT_UPPER_PC*pTOTAL_REALIGN/100)
					,BOND_DRIFT_LOWER_PC*pTOTAL_REALIGN/100)
		else:
			return pTOTAL_REALIGN * BONDS_CATEGORY_MIX/100
		
	
	def pTOTAL_REALIGN(MATH_RES_IF_PL,DA_FUND_FL,CA_FUND_FL,SOLV_MARG_IF_FL):
		return MATH_RES_IF_PL + DA_FUND_FL + CA_FUND_FL + SOLV_MARG_IF_FL
	
	def pcREBAL_SELL(CATEGORY,POOL,pcBOND_TARGET,pcBOND_FAV,cNEW_MONEY_TO_BOND):
		if (CATEGORY>4) or (POOL==7): #this is horrible hard-coding, it should be fed by a table somewhere
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

	def pcREBAL_BUY(POOL,cNEW_MONEY_TO_BOND,pcBOND_FAV,pcDENOM,pcBOND_TARGET):
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
	
	def pREBAL_CASH(pTOTAL_FAV,pBOND_TARGET,pEQUITY_TARGET):
		return pTOTAL_FAV + pBOND_TARGET + pEQUITY_TARGET
		
	def pcEQUITY_TARGET(POOL,pTOTAL_FAV,EQUITY_CATEGORY_MIX
			,pcEQUITY_RET,EQUITY_DRIFT_UPPER_PC,EQUITY_DRIFT_LOWER_PC,cNEW_MONEY_TO_EQUITY):
		if POOL==10:
			return max(
					min(cNEW_MONEY_TO_EQUITY + pcEQUITY_RET
					,EQUITY_DRIFT_UPPER_PC*pTOTAL_FAV/100)
					,EQUITY_DRIFT_LOWER_PC*pTOTAL_FAV/100)
		else:
			return pTOTAL_FAV * EQUITY_CATEGORY_MIX/100
	
	def pcBOND_TARGET(POOL,pTOTAL_FAV,BONDS_CATEGORY_MIX
			,pcBOND_FAV,BOND_DRIFT_UPPER_PC,BOND_DRIFT_LOWER_PC,cNEW_MONEY_TO_BOND):
		if POOL==10:
			return max(
					min(cNEW_MONEY_TO_BOND + pcBOND_FAV
					,BOND_DRIFT_UPPER_PC*pTOTAL_FAV/100)
					,BOND_DRIFT_LOWER_PC*pTOTAL_FAV/100)
		else:
			return pTOTAL_FAV * BONDS_CATEGORY_MIX/100

	# def rfpREALIGN_CASH(pTOTAL_FAV):
		# return pTOTAL_FAV #placeholder
# 	def rfpcyBOND_MV(pTOTAL_FAV):
# 		return pTOTAL_FAV

		
	def cNEW_MONEY_TO_EQUITY(EQUITY_NEW_MONEY_P_PC,EQUITY_NEW_MONEY_N_PC,new_money):
		if new_money>=0:
			return EQUITY_NEW_MONEY_P_PC*new_money/100
		else:
			return EQUITY_NEW_MONEY_N_PC*new_money/100
	
	def cNEW_MONEY_TO_BOND(BOND_NEW_MONEY_P_PC,BOND_NEW_MONEY_N_PC,new_money):
		if new_money>=0:
			return BOND_NEW_MONEY_P_PC*new_money/100
		else:
			return BOND_NEW_MONEY_N_PC*new_money/100
	
	def pTOTAL_FAV(pBOND_FAV,pEQUITY_RET,pREALIGN_CASH,pBOX):
		return pBOND_FAV + pEQUITY_RET + pREALIGN_CASH + pBOX
	
# 		for mt in range(1,13):
	def pBOX(pBOND_CF,CASH_RET_PC,LIAB_CF_PL,pREALIGN_CASH): #this needs to run thru a single year. non-standard time. how should variables be referred to?
		box=0
		for mt in range(12):
			box = box*(1+CASH_RET_PC/100) + LIAB_CF_PL[mt]*((1+CASH_RET_PC/100)**0.5) + \
				pBOND_CF[mt] + pREALIGN_CASH*(CASH_RET_PC/100)
		return box

	def pcEQUITY_RET(pcEQUITY_REALIGN_TARGET,cEQUITY_RET_RATE):
		return pcEQUITY_REALIGN_TARGET * ((1 + cEQUITY_RET_RATE)**12)
	
# 	def cEQUITY_RET_RATE(EQUITY_RET_IDX,cEQUITY_RET_RATE):
# 		for t in range(1,EQUITY_RET_IDX.shape[-1]):
# 			if EQUITY_RET_IDX[t-1]==0:
# 				pass
# 			else:
# 				cEQUITY_RET_RATE[t] = (EQUITY_RET_IDX[t]/EQUITY_RET_IDX[t-1]) ** (1/12) - 1
	def cEQUITY_RET_RATE(EQUITY_RET_IDX,cEQUITY_RET_RATE):
		for t in range(EQUITY_RET_IDX.shape[-1]):
			if t==0:
				cEQUITY_RET_RATE[t] = EQUITY_RET_IDX[t] ** (1/12) - 1
			else:
				if EQUITY_RET_IDX[t-1]==0:
					cEQUITY_RET_RATE[t] = 0
				else:
					cEQUITY_RET_RATE[t] = (EQUITY_RET_IDX[t]/EQUITY_RET_IDX[t-1]) ** (1/12) - 1
				
			
