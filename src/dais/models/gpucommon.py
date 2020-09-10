from numba import njit, jit, vectorize, int64, int32, void, prange, cuda
from numpy import int32, int64, float64
import pandas as pd
import numpy as np
try:
	import cupy as cp
except ImportError:
	print("You need cupy installed.")
import networkx as nx
import inspect
import math
from collections import OrderedDict
import time
from operator import itemgetter
from .graphs import ModelDependencyGraph, FunctionNode, FunctionNodeBond
from .simple_dispatcher import gpu_tabledyn_helper_minmax, gpu_fast_take
from .complex_dispatcher import CompiledFunction,BondCompiledFunction,GPUCompiledFunction
from .common3 import BondModel,BaseModel,GUIBaseModel,GUIMixModel
import importlib


class GPUBondModel(BondModel):
	def __init__(self,time_periods=1200,val_time_periods=120):
		super().__init__(time_periods,val_time_periods)
		self.node_type=GPUCompiledFunction

	def execute_table(self,tablenode):
		#this is a single-threaded activity
		fnode=self.internal_graph.get_fnode(tablenode)
		assert fnode.source_type == 'TablePolicy'
		tablefields=list(fnode.attr['mapping'].keys())
		mapfields=[fnode.attr['mapping'][f] for f in tablefields]
		#Check to make sure all upstream dependencies have indeed been fulfilled.
		assert len(set(mapfields).intersection(set(self.vals.keys())))==len(set(mapfields))
		input_data={}
		for k in tablefields:
			val=self.vals[fnode.attr['mapping'][k]]
			if type(val)==cp.ndarray:
				input_data[k]=cp.asnumpy(val.ravel())
			else:
				input_data[k]=val.ravel()
		datamaps=pd.DataFrame(input_data)
		table=self.tables[fnode.attr['source']][tablefields + [tablenode]]
		self.vals[tablenode]=cp.asarray(pd.merge(datamaps
										,table
										,how='left'
										,on=tablefields
		)[tablenode].to_numpy().reshape((-1,1)))

	def execute_tabledyn(self,tablenode):
		# print(tablenode)
		fnode=self.internal_graph.get_fnode(tablenode)
		assert fnode.source_type == 'TablePolicyTime'
		tablefields=list(fnode.attr['mapping'].keys())
		mapfields=[fnode.attr['mapping'][f] for f in tablefields]
		#Check to make sure all upstream dependencies have indeed been fulfilled.
		assert len(set(mapfields).intersection(set(self.vals.keys())))==len(set(mapfields))
		#'col' is the special value given to to the colunms on a time-based lookup
		row_tablefields=[k for k in tablefields if k!= 'col']
		input_data={}
		for k in row_tablefields:
			val=self.vals[fnode.attr['mapping'][k]]
			if type(val)==cp.ndarray:
				input_data[k]=cp.asnumpy(val.ravel())
			else:
				input_data[k]=val.ravel()
		datamaps=pd.DataFrame(input_data)
		# print(datamaps)
		table=self.tables[tablenode][row_tablefields].reset_index()
		rownumbers=pd.merge(datamaps,table,how='left'
							,on=row_tablefields)['index'].to_numpy().reshape((self.mpf.shape[0],1))
		# print(rownumbers)
		colnames_list=[int(col) for col in self.tables[tablenode].columns if not ((col in row_tablefields) or (col in ['index','Description','DESCRIPTION']))]
		colnames=cp.asarray(colnames_list)
		colnames_min=int(cp.min(colnames))
		colnames_max=int(cp.max(colnames))
		assert cp.allclose(colnames,cp.arange(colnames_min,colnames_max+1))
		colval=self.vals[fnode.attr['mapping']['col']]
# 		if type(colval)==cuda.cudadrv.devicearray.DeviceNDArray:
# 			colval=colval.copy_to_host()
		col_indices=gpu_tabledyn_helper_minmax(colval.ravel(),colnames_min,colnames_max)	
		row_indices2=cp.broadcast_to(cp.asarray(rownumbers),colval.shape)
		row_indices3=row_indices2.ravel()
# 		row_indices3.flags.writeable=False
		table=cp.asarray(self.tables[tablenode][colnames_list].to_numpy())
		self.vals[tablenode]=gpu_fast_take(table,row_indices3,col_indices).reshape(colval.shape)

	def execute_summary(self,summarynode):
		fnode=self.internal_graph.get_fnode(summarynode)
		assert fnode.source_type=='Summary'
		indexes=fnode.attr['byvars']
		vals=fnode.attr['vars']
		assert len(set(indexes+vals).intersection(set(self.vals.keys())))==len(set(indexes+vals))
		fnode=self.internal_graph.get_fnode(summarynode)
		subnodes=fnode.attr['subnodes']
		byvars=fnode.attr['byvars']
# 		vals=fnode.attr['vars']
		results={}
		tempvals={}
		for var in subnodes.keys():
			if type(self.vals[var])==cp.ndarray:
				tempvals[var]=cp.asnumpy(self.vals[var])
			else:
				tempvals[var]=self.vals[var]
			if self.internal_graph.get_fnode(var).attr['shape']==2:
				columns=range(self.vals[var].shape[1])
				df=pd.DataFrame(columns=columns,data=tempvals[var])
			elif self.internal_graph.get_fnode(var).attr['shape']==1:
				df=pd.DataFrame({var:tempvals[var][:,0]})
			for byvar in byvars:
				assert self.internal_graph.get_fnode(byvar).attr['shape']==1
				if type(self.vals[byvar])==cp.ndarray:
					tempvals[byvar]=cp.asnumpy(self.vals[byvar])
				else:
					tempvals[byvar]=self.vals[byvar]
				df.insert(0,byvar,tempvals[byvar].ravel())
			self.vals[subnodes[var]]=df.groupby(byvars).agg(fnode.attr['func']).reset_index()

	def initialise(self,targets=None):
		"""Targets is a list of final variables for which to solve. None means all and is default.
		The initialise function creates input variables for T, Constants and MPF values from the input sources.
		Execution can only begin once initialised."""
		tasklist=self.create_tasklist(targets)
		overlap=set(tasklist).intersection(set(self.internal_graph.get_unmapped()))
		assert len(overlap)==0
		
		self.vals['VT']=cp.broadcast_to(cp.arange(self.val_time_periods,dtype=cp.int32).reshape((1,self.val_time_periods))
										  ,(len(self.mpf),self.val_time_periods))
# 		self.vals['VT']=cuda.to_device(self.vals['VT'].copy())
		
		self.vals['T']=cp.broadcast_to(cp.arange(self.time_periods,dtype=cp.int32).reshape((1,self.time_periods))
										  ,(len(self.mpf),self.time_periods))
# 		self.vals['T']=cuda.to_device(self.vals['T'].copy())

		for node in tasklist:
			fnode=self.internal_graph.get_fnode(node)
			if fnode.is_original_input():
				print("Initialising source:",fnode.name)
				if fnode.source_type=='MPF':
					if self.mpf[node].dtype != np.dtype('O'):
						self.vals[node]=cp.asarray(self.mpf[node].to_numpy().reshape((-1,1)))
					else:
						self.vals[node]=self.mpf[node].to_numpy().reshape((-1,1))
				elif fnode.source_type=='Constant':
					self.vals[node]=cp.broadcast_to(cp.asarray([[self.constants[node]]]).reshape((1,1))
															,(len(self.mpf),1))
# 					self.vals[node]=self.vals[node].copy()
				fnode.attr['type']=self.vals[node].dtype
# 				self.vals[node]=cuda.to_device(self.vals[node])
		self.compile_or_load(targets)
		
class GUIGPUModel(GPUBondModel,GUIBaseModel):
	pass
class GPUMixModel(GUIMixModel,GPUBondModel):
	pass

@cuda.jit
def BOND_VALUES11(VT,FX_INDEX,MV_T0,BVCF,ACCRUED_INTEREST_T0,CALENDAR_YEAR,MV_NUMPERIODS,COUPON_FREQ,COUPON_PC,COUPON_RATE_RESET,ASSET_TYPE,REF_RATE_MARGIN_PC
            ,PURCHASE_T0,YIELD_IDX,CALENDAR_MONTH,REDEMP_YEAR,REDEMP_MONTH,BOND_REMAINING_PERIOD_VT,ASSET_SCALAR
            ,YIELDS,BOOK_YIELD_NPVRATE,REDEMPTION,REDEMP_AMT,val_yr
            ,COUPON_RATE_RESET_T,MV,NPV_BY,CF
           ):
	y,t=cuda.grid(2)
#     inner_COUPON = cuda.shared.array((16,16),float64)
#     inner_COUPON_RATE = cuda.shared.array((16,16),float64)
#     yy=cuda.threadIdx.x
#     tt=cuda.threadIdx.y
	if (y<COUPON_RATE_RESET_T.shape[0]) and (t<COUPON_RATE_RESET_T.shape[1]):
		if t>min(BOND_REMAINING_PERIOD_VT[y,1+PURCHASE_T0[y,0]],MV_NUMPERIODS[y,0]):
			inner_COUPON_RATE = 0
			inner_COUPON = 0
		else:
		#     for t in range(PURCHASE_T0[0],min(BOND_REMAINING_PERIOD_VT[1+PURCHASE_T0[0]]+1,MV_NUMPERIODS[0])):
			if (COUPON_FREQ[y,0])==0:
				inner_COUPON_RATE = 0
			else:
				inner_COUPON_RATE = COUPON_PC[y,0]/(100*COUPON_FREQ[y,0])
			if t==PURCHASE_T0[y,0]:
				MV[y,t]=MV_T0[y,0]
				NPV_BY[y,t]=-(BVCF[y,0]+ACCRUED_INTEREST_T0[y,0])
				COUPON_RATE_RESET_T[y,t]=COUPON_RATE_RESET[y,0]
			else:
				for ty in range(t,min(t+BOND_REMAINING_PERIOD_VT[y,t]+1,CALENDAR_YEAR.shape[1])):
					##########
					#CALCULALATE THE COUPON AMOUNT
					if COUPON_FREQ[y,0]==0:
						inner_COUPON = 0
					else:
						if BOND_REMAINING_PERIOD_VT[y,t]-(ty-t)>=0:
							if (BOND_REMAINING_PERIOD_VT[y,t]-(ty-t))%(12/COUPON_FREQ[y,0])==0:
								inner_COUPON = FX_INDEX[y,t-PURCHASE_T0[y,0]]*inner_COUPON_RATE*REDEMP_AMT[y,0]*ASSET_SCALAR[y,0]
							else:
								inner_COUPON = 0
						else:
							inner_COUPON = 0
					#END COUPON CALC
					##########
					if ty>t:
						MV[y,t]=MV[y,t]+(inner_COUPON + FX_INDEX[y,t-PURCHASE_T0[y,0]]*REDEMPTION[y,ty]*ASSET_SCALAR[y,0])*YIELDS[YIELD_IDX[y,0]+t,ty-t]
						NPV_BY[y,t]=NPV_BY[y,t]+(inner_COUPON + FX_INDEX[y,t-PURCHASE_T0[y,0]]*REDEMPTION[y,ty]*ASSET_SCALAR[y,0])/(1+BOOK_YIELD_NPVRATE[y,0])**(ty-t)
					if ty==(t+1):
						COUPON_RATE_RESET_T[y,t]=inner_COUPON_RATE*COUPON_FREQ[y,0]
					if ty==t:
						CF[y,t]=inner_COUPON + FX_INDEX[y,t-PURCHASE_T0[y,0]]*REDEMPTION[y,ty]*ASSET_SCALAR[y,0]


def wrapped_BOND_VALUES11(VT,FX_INDEX,MV_T0,BVCF,ACCRUED_INTEREST_T0,CALENDAR_YEAR,MV_NUMPERIODS,COUPON_FREQ,COUPON_PC,COUPON_RATE_RESET,ASSET_TYPE,REF_RATE_MARGIN_PC,PURCHASE_T0,YIELD_IDX,CALENDAR_MONTH,REDEMP_YEAR,REDEMP_MONTH,BOND_REMAINING_PERIOD_VT,ASSET_SCALAR,YIELDS,BOOK_YIELD_NPVRATE,REDEMPTION,REDEMP_AMT,val_yr):
	arr_COUPON_RATE_RESET_T=cp.zeros(VT.shape,float64)
	arr_MV=cp.zeros(VT.shape,float64)
	arr_NPV_BY=cp.zeros(VT.shape,float64)
	arr_CF=cp.zeros(VT.shape,float64)

	threadsperblock=(16,16)
	blockspergrid_y=math.ceil(VT.shape[0] / threadsperblock[0])
	blockspergrid_t=math.ceil(VT.shape[1] / threadsperblock[1])
	blockspergrid=(blockspergrid_y,blockspergrid_t)
	stream=cuda.stream()
	BOND_VALUES11[blockspergrid,threadsperblock,stream](VT,FX_INDEX,MV_T0,BVCF,ACCRUED_INTEREST_T0
							,CALENDAR_YEAR,MV_NUMPERIODS,COUPON_FREQ,COUPON_PC,COUPON_RATE_RESET
							,ASSET_TYPE,REF_RATE_MARGIN_PC,PURCHASE_T0,YIELD_IDX,CALENDAR_MONTH
							,REDEMP_YEAR,REDEMP_MONTH,BOND_REMAINING_PERIOD_VT,ASSET_SCALAR,YIELDS
							,BOOK_YIELD_NPVRATE,REDEMPTION,REDEMP_AMT,val_yr
							,arr_COUPON_RATE_RESET_T,arr_MV,arr_NPV_BY,arr_CF)
	stream.synchronize()
	return {'COUPON_RATE_RESET_T': arr_COUPON_RATE_RESET_T,'MV': arr_MV,'NPV_BY': arr_NPV_BY,'CF': arr_CF}

	