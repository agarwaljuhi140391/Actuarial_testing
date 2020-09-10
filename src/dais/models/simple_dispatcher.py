import numpy as np
from numpy import int32, int64, float64
try:
	import cupy as cp
except ImportError:
	import numpy as cp
from numba import prange,njit,cuda
import pandas as pd
import math

parallel_state=True

def gpu_fast_take(table,rs,cs):
	threadsperblock=32
	blockspergrid=math.ceil(rs.shape[0] / threadsperblock)
	out=cp.zeros(rs.shape[0],table.dtype)
	stream=cuda.stream()
	gpukernel_fast_take[blockspergrid,threadsperblock,stream](table,rs,cs,out)
	stream.synchronize()
	return out

@cuda.jit
def gpukernel_fast_take(table,rs,cs,out):
	y=cuda.grid(1)
	if (y<rs.shape[0]) and (y<cs.shape[0]) and (y<out.shape[0]):
		#Possible improvement from caching table in shared memory perhaps?
		out[y]=table[rs[y],cs[y]]

def gpu_tabledyn_helper_minmax(arr,minval,maxval):
	threadsperblock=32
	blockspergrid=math.ceil(arr.shape[0] / threadsperblock)
	out=cp.zeros(arr.shape[0],arr.dtype)
	stream=cuda.stream()
	gpukernel_tabledyn_helper_minmax[blockspergrid,threadsperblock,stream](arr,minval,maxval,out)
	stream.synchronize()
	return out

@cuda.jit
def gpukernel_tabledyn_helper_minmax(arr,minval,maxval,out):
	y=cuda.grid(1)
	if y < out.shape[0]:
		out[y]=min(max(arr[y],minval),maxval)-minval

@njit(parallel=parallel_state,nogil=True,cache=True)
def tabledyn_helper_minmax(arr,minval,maxval):
	res=np.zeros(arr.shape,np.int32)
	for y in prange(arr.shape[0]):
		res[y]=min(max(arr[y],minval),maxval)-minval
	return res

@cuda.jit(device=True)
def gpu_inner_npv_fixrate(cf,r):
	npv=0
	for t in range(cf.shape[0]):
		if cf[t]==0:
			pass
		else:
			npv=npv+cf[t]/(r+1)**t
	return npv

@cuda.jit(device=True)
def gpu_irr_est(cf,guess):
	PRECISION=1e-9
	ITERATIONS=100
	xa=0
	xb=guess
	for i in range(ITERATIONS):
		r=(xa+xb)/2
		npv=gpu_inner_npv_fixrate(cf,r)
		ca=gpu_inner_npv_fixrate(cf,xa)
		if (npv*ca)>0:
			xa=r
		else:
			xb=r
		if (npv==0) or (abs(xb-xa)<=PRECISION):
			return r
	return 999 #error because it hasn't converged


@njit(nogil=True,cache=True)
def inner_npv_fixrate(cf,r):
	npv=0
	for t in range(cf.shape[0]):
		if cf[t]==0:
			pass
		else:
			npv=npv+cf[t]/(r+1)**t
	return npv
	
@njit(nogil=True,cache=True)
def irr_est(cf,guess):
	PRECISION=1e-9
	ITERATIONS=100
	xa=-guess-0.2*guess
	xb=guess
	for i in range(ITERATIONS):
		r=(xa+xb)/2
		npv=inner_npv_fixrate(cf,r)
		ca=inner_npv_fixrate(cf,xa)
		if (npv*ca)>0:
			xa=r
		else:
			xb=r
		if (npv==0) or (abs(xb-xa)<=PRECISION):
			return r
	return 999 

def zcb_master(bond_tables,economy,economy_map):
	economy=economy_map[economy]
	ZCB=bond_tables['ZCB'].query('ECONOMY == {}'.format(economy))
	ZCB_BASE_FRATE=bond_tables['ECON'].query('Economy=={}'.format(economy))['ZCB_BASE_FRATE'].iloc[0]
	zcb_index_columns=['SIMULATION','ECONOMY','TERM']
	years=[int(c) for c in ZCB.columns if c not in zcb_index_columns]
	zcbrates=ZCB[years[:-1]].to_numpy()
	available_terms=ZCB.TERM.to_numpy()
	maxterm=int(ZCB.TERM.max())
	numyears=len(years)-1
	yield_curve = zcb_inner(zcbrates=zcbrates,available_terms=available_terms,numyears=numyears,ZCB_BASE_FRATE=ZCB_BASE_FRATE,maxterm=maxterm,economy=economy)
	df=pd.DataFrame(data=yield_curve.T).reset_index().rename(columns={'index':'TERM'}).assign(**{'ECONOMY':economy})
	return df

@njit(parallel=False,cache=True)
def zcb_inner(zcbrates,available_terms,numyears,ZCB_BASE_FRATE,maxterm,economy):
	v1=math.exp(-ZCB_BASE_FRATE/12)
	vn=np.zeros((maxterm,),np.float64)
	interp_maxterm=maxterm*12+1
	v=np.zeros((interp_maxterm,),np.float64)
	vdiff=np.zeros((interp_maxterm,),np.float64)
	yield_rates=np.zeros((interp_maxterm,numyears),np.float64)

	for t in range(maxterm):
		vn[t]=1/(1-v1**((t+1)*12))
	term_index=0
	for t in range(interp_maxterm):
		if t==0:
			v[t]=vn[0]
		elif t==interp_maxterm-1:
			v[t]=v[t-1]*v1
		elif t>=available_terms[term_index+1]*12:
				term_index=term_index+1
				v[t]=vn[available_terms[term_index+1]-available_terms[term_index]-1]
		else:
			v[t]=v[t-1]*v1
		vdiff[t]=vn[available_terms[term_index+1]-available_terms[term_index]-1]-v[t]
		for y in prange(numyears):
			if t==interp_maxterm-1:
				yield_rates[t,y]=1
			else:
				yield_rates[t,y]=zcbrates[term_index,y]*(1-vdiff[t])+zcbrates[term_index+1,y]*vdiff[t]
	return yield_rates

##################       TEMPORARILY PARKED CODE     #############################
#this code has been removed from the main body because it is pre-processing that we can optimize separately from the model
#how it gets used finally is subject to discussion still, and so is parked here for reference
# def zcb_master2(zcb_info,economy, economy_map,start_month,start_year):
#     ZCB_BASE_FRATE=zcb_info['ECON'].query('Economy=={}'.format(economy))['ZCB_BASE_FRATE'].iloc[0]
#     zcb_tab=zcb_info['ZCB'].query('ECONOMY=={}'.format(economy))
#     zcb_index_columns=['SIMULATION','ECONOMY','TERM']
#     years=[int(c) for c in zcb_tab.columns if c not in zcb_index_columns]
#     zcbrates=zcb_tab[years].to_numpy()
#     available_terms=zcb_tab.TERM.to_numpy()
#     max_available_terms=np.max(available_terms)
#     # maxterm=int(zcbhkd.TERM.max())
#     maxterm=600+1
#     numyears=len(years)-1
#     numyear_periods=numyears*12+1
#     v1=math.exp(-ZCB_BASE_FRATE/12)
#     yield_curve=np.zeros((numyear_periods,maxterm),dtype=np.float64)
#     start_month=12
#     start_year=2019
#     #Call the numba function
#     inner_zcb2(zcbrates,yield_curve,start_month,start_year,v1,available_terms,numyear_periods,maxterm,max_available_terms)
#     df=pd.DataFrame(data=yield_curve).reset_index().rename(columns={'index':'TERM'})
#     df.insert(0,'ECONOMY',economy)
#     return df
# @njit(cache=True)
# def inner_zcb2(zcbrates,yield_curve,start_month,start_year,v1,available_terms,numyear_periods,maxterm,max_available_terms):
#     for y in range(numyear_periods):
#         year_idx=(start_month-1+y)//12
#         term_idx=0
#         last_zcb=zcbrates[term_idx,year_idx]
#         next_zcb=zcbrates[term_idx+1,year_idx]
#         for t in range(maxterm):
#             if t == max_available_terms:
#                 yield_curve[y,t]=next_zcb
#     #             pass
#             elif t>max_available_terms:
#                 yield_curve[y,t]=yield_curve[y,t-1]*v1
#             else:
#                 if t==available_terms[term_idx+1]:
#                     term_idx=term_idx+1
#                     last_zcb=zcbrates[term_idx,year_idx]
#                     next_zcb=zcbrates[term_idx+1,year_idx]
#                 weight=(t-available_terms[term_idx])/(available_terms[term_idx+1]-available_terms[term_idx])
#                 yield_curve[y,t]=last_zcb**(1-weight) * next_zcb**weight
# yield_rates=pd.concat([
#     zcb_master2(zcb_info,2, economy_map
#             ,start_month=CONSTANTS['val_mth'],start_year=CONSTANTS['val_yr']).assign(ECONOMY=2)
#     , zcb_master2(zcb_info,1, economy_map
#             ,start_month=CONSTANTS['val_mth'],start_year=CONSTANTS['val_yr']).assign(ECONOMY=1)
#     ],ignore_index=True)
# yield_rates.insert(0,'SIMULATION',1)
# yield_rates=yield_rates.sort_values(by=['SIMULATION','ECONOMY','TERM']).reset_index(drop=True)


@njit(parallel=parallel_state, nogil=True, cache=True)
def elementwise_min(a,b):
	c=np.empty_like(a)
	for i in prange(a.shape[0]):
		c[i]=min(a[i],b)
	return c

def chunks(l, n):
	""" Yield n successive chunks from l.
	"""
	newn = int(l / n)
	for i in range(0, n-1):
		yield (i*newn,i*newn+newn)
	yield (n*newn-newn,l)
    
@njit(parallel=parallel_state,nogil=True,cache=True)
def fast_take(table,rs,cs):
	out=np.empty(rs.shape[0],table.dtype)
	for i in prange(rs.shape[0]):
		out[i]=table[rs[i],cs[i]]
	return out
# def fast_take(table,rs,cs):
#     indexes=np.asarray(list(chunks(len(rs),5)))
#     return nfast_take(table,rs,cs,indexes)

@njit(parallel=parallel_state,nogil=True,cache=True)
def nfast_searchsorted(thing_to_search,thing_to_sort,indexes):
	out=np.empty(thing_to_sort.shape,int32)
	for m in prange(indexes.shape[0]):
		out[indexes[m,0]:indexes[m,1]]=np.searchsorted(thing_to_search,thing_to_sort[indexes[m,0]:indexes[m,1]])
	return out
def fast_searchsorted(thing_to_search,thing_to_sort):
	indexes=np.asarray(list(chunks(len(thing_to_sort),5)))
	return nfast_searchsorted(thing_to_search,thing_to_sort,indexes)


#RETIRE THIS CODE ONCE YOU'RE SURE THE NEW CODE IN COMPLEX_DISPATCHER WORKS CORRECTLY
def func_fac1(f,outdtype,arg0):
	print("func_fac1")
	if arg0.shape[1]==1:
		@njit(parallel=parallel_state,nogil=True)
		def _temp(arg0):
			out=np.empty(arg0.shape,outdtype)
			for i in prange(arg0.shape[0]):
				out[i]=f(arg0[i,0])
			return out
		return _temp
	else:
		@njit(parallel=parallel_state,nogil=True)
		def _temp(arg0):
			out=np.empty(arg0.shape,outdtype)
			for i in prange(arg0.shape[0]):
				for t in range(arg0.shape[1]):
					out[i,t]=f(arg0[i,t])
			return out
		return _temp
        
def func_fac2(f,outdtype,arg0,arg1):
	print("func_fac2")
	if arg0.shape[1]==1:
		@njit(parallel=parallel_state,nogil=True)
		def _temp(arg0,arg1):
			out=np.empty(arg0.shape,outdtype)
			for i in prange(arg0.shape[0]):
				out[i]=f(arg0[i,0],arg1[i,0])
			return out
		return _temp
	else:
		@njit(parallel=parallel_state,nogil=True)
		def _temp(arg0,arg1):
			out=np.empty(arg0.shape,outdtype)
			for i in prange(arg0.shape[0]):
				for t in range(arg0.shape[1]):
					out[i,t]=f(arg0[i,t],arg1[i,t])
			return out
		return _temp

def func_fac3(f,outdtype,arg0,arg1,arg2):
	if arg0.shape[1]==1:
		@njit(parallel=parallel_state,nogil=True)
		def _temp(arg0,arg1,arg2):
			out=np.empty(arg0.shape,outdtype)
			for i in prange(arg0.shape[0]):
				out[i]=f(arg0[i,0],arg1[i,0],arg2[i,0])
			return out
		return _temp
	else:
		@njit(parallel=parallel_state,nogil=True)
		def _temp(arg0,arg1,arg2):
			out=np.empty(arg0.shape,outdtype)
			for i in prange(arg0.shape[0]):
				for t in range(arg0.shape[1]):
					out[i,t]=f(arg0[i,t],arg1[i,t],arg2[i,t])
			return out
		return _temp

def func_fac4(f,outdtype,arg0,arg1,arg2,arg3):
	if arg0.shape[1]==1:
		@njit(parallel=parallel_state,nogil=True)
		def _temp(arg0,arg1,arg2,arg3):
			out=np.empty(arg0.shape,outdtype)
			for i in prange(arg0.shape[0]):
				out[i]=f(arg0[i,0],arg1[i,0],arg2[i,0],arg3[i,0])
			return out
		return _temp
	else:
		@njit(parallel=parallel_state,nogil=True)
		def _temp(arg0,arg1,arg2,arg3):
			out=np.empty(arg0.shape,outdtype)
			for i in prange(arg0.shape[0]):
				for t in range(arg0.shape[1]):
					out[i,t]=f(arg0[i,t],arg1[i,t],arg2[i,t],arg3[i,t])
			return out
		return _temp
		
def func_fac5(f,outdtype,arg0,arg1,arg2,arg3,arg4):
	if arg0.shape[1]==1:
		@njit(parallel=parallel_state,nogil=True)
		def _temp(arg0,arg1,arg2,arg3,arg4):
			out=np.empty(arg0.shape,outdtype)
			for i in prange(arg0.shape[0]):
				out[i]=f(arg0[i,0],arg1[i,0],arg2[i,0],arg3[i,0],arg4[i,0])
			return out
		return _temp
	else:
		@njit(parallel=parallel_state,nogil=True)
		def _temp(arg0,arg1,arg2,arg3,arg4):
			out=np.empty(arg0.shape,outdtype)
			for i in prange(arg0.shape[0]):
				for t in range(arg0.shape[1]):
					out[i,t]=f(arg0[i,t],arg1[i,t],arg2[i,t],arg3[i,t],arg4[i,t])
			return out
		return _temp

def func_fac6(f,outdtype,arg0,arg1,arg2,arg3,arg4,arg5):
	if arg0.shape[1]==1:
		@njit(parallel=parallel_state,nogil=True)
		def _temp(arg0,arg1,arg2,arg3,arg4,arg5):
			out=np.empty(arg0.shape,outdtype)
			for i in prange(arg0.shape[0]):
				out[i]=f(arg0[i,0],arg1[i,0],arg2[i,0],arg3[i,0],arg4[i,0],arg5[i,t])
			return out
		return _temp
	else:
		@njit(parallel=parallel_state,nogil=True)
		def _temp(arg0,arg1,arg2,arg3,arg4,arg5):
			out=np.empty(arg0.shape,outdtype)
			for i in prange(arg0.shape[0]):
				for t in range(arg0.shape[1]):
					out[i,t]=f(arg0[i,t],arg1[i,t],arg2[i,t],arg3[i,t],arg4[i,t],arg5[i,t])
			return out
		return _temp

def func_fac7(f,outdtype,arg0,arg1,arg2,arg3,arg4,arg5,arg6):
	if arg0.shape[1]==1:
		@njit(parallel=parallel_state,nogil=True)
		def _temp(arg0,arg1,arg2,arg3,arg4,arg5,arg6):
			out=np.empty(arg0.shape,outdtype)
			for i in prange(arg0.shape[0]):
				out[i]=f(arg0[i,0],arg1[i,0],arg2[i,0],arg3[i,0],arg4[i,0],arg5[i,0],arg6[i,0])
			return out
		return _temp
	else:
		@njit(parallel=parallel_state,nogil=True)
		def _temp(arg0,arg1,arg2,arg3,arg4,arg5,arg6):
			out=np.empty(arg0.shape,outdtype)
			for i in prange(arg0.shape[0]):
				for t in range(arg0.shape[1]):
					out[i,t]=f(arg0[i,t],arg1[i,t],arg2[i,t],arg3[i,t],arg4[i,t],arg5[i,0],arg6[i,0])
			return out
		return _temp
