import pandas as pd
import numpy as np
from numba import jit, float64, int64, void, typeof

class MortalityAssumptionSet:
	"""A class representing a set of mortality assumptions"""
	gender_map={"M":0,"F":1}
	
	def __init__(self,file_locs):
		"""file_locs is a dict of files, expecting {"M":"/path/to/file1", "F":"/path/to/file2"}"""
		self.originalFileLocs={}
		self.raw_files={}
		#check type and structure of file_locs
		#check files exist
		self.originalFileLocs=file_locs
		for key in self.originalFileLocs:
			self.raw_files[key]=self.read_file(self.originalFileLocs[key])
	
	def read_file(self,file_loc):
		"""Internal function to read a given mortality table file"""
		mortality=pd.read_csv(file_loc,
							  header=None, sep=' ', skiprows=1,
							  names=['start_age', 'var1', 'var2', 'var3', 'end_age'])
		return mortality
		
	def get_periods(self):
		#check both files have same periods
		keys=list(self.raw_files.keys())
		return [c for c in self.raw_files[keys[0]].columns if c not in ['start_age','end_age']]

	def process_mort(self,globals_run):
		MinAge=0
		MaxAge=globals_run['MAX_AGE']
		Prop_Vq2_pc=globals_run['PROP_VQ2_PC']
		Prop_Eq2_pc=globals_run['PROP_EQ2_PC']
		Prop_Valq_pc=globals_run['PROP_VALQ_PC']
		Prop_Expq_pc=globals_run['PROP_EXPQ_PC']
		periods=self.get_periods()
		q=np.zeros((2,MaxAge+1,len(periods)),dtype='float64')
		Px_Val_M=np.zeros((2,MaxAge+1,len(periods),2),dtype='float64')
		Px_Exp_M=np.zeros((2,MaxAge+1,len(periods),2),dtype='float64')
		#THERE IS SOME FIX FOR INDEXING THAT YOU NEED TO DO.
		#ALSO, HOW DO YOU CALL A STATIC METHOD INSIDE A CLASS
		
		for key in self.raw_files:
			f=self.raw_files[key].query('start_age>=@MinAge and start_age<=@MaxAge').reset_index()[periods].values
# 			print(f.shape)
			MortalityAssumptionSet._process_mort(f,
				self.gender_map[key],
				MinAge,MaxAge,Prop_Vq2_pc,Prop_Eq2_pc,Prop_Valq_pc,Prop_Expq_pc,
				q,Px_Val_M,Px_Exp_M
			)
		return {"q":q,"Px_Val_M":Px_Val_M,"Px_Exp_M":Px_Exp_M}

	@staticmethod
	@jit(nopython=True)
	def _process_mort(rawtable,gender,
					MinAge,MaxAge,Prop_Vq2_pc,Prop_Eq2_pc,Prop_Valq_pc,Prop_Expq_pc,
					q,Px_Val_M,Px_Exp_M
					):
		MaxTabAge=MaxAge
		for sct in range(rawtable.shape[1]):
			qx_temp=0
			qx_temp_prev=0
			MaxAgeTemp=MaxAge-sct
			for iage in range(MaxAgeTemp-1): #This "-1" is a bugfix from code that worked. I think by luck. Need to doublecheck...
				age=iage+sct
# 				print(age,iage,sct)
				if age<MinAge:
					qx_temp=0
				elif age>MaxTabAge or age>=MaxAgeTemp:
					qx_temp=1
				else:
					temp=rawtable[iage,sct]
					if temp<0:
						qx_temp=0
					elif temp>1:
						qx_temp=1
					else:
						qx_temp=temp
						if age>0 and qx_temp==1:
							MaxTabAge=min(MaxTabAge,age-1)
							qx_temp=1
				q[gender,age,sct]=qx_temp
				qx_temp_prev=qx_temp
				Px_Val_M[gender,age,sct,0]=(1-q[gender,age,sct]*Prop_Valq_pc)**(1/12)
				Px_Exp_M[gender,age,sct,0]=(1-q[gender,age,sct]*Prop_Expq_pc)**(1/12)
				Px_Val_M[gender,age,sct,1]=(1-q[gender,age,sct]*Prop_Vq2_pc)**(1/12)
				Px_Exp_M[gender,age,sct,1]=(1-q[gender,age,sct]*Prop_Eq2_pc)**(1/12)
	
	def set_tpx(self):
		pass
	
	@staticmethod
	@jit(nopython=True)
	def _set_tpx():
		1==1