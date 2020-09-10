import pandas as pd

class GlobalsHandler:
	
	def __init__(self,file_loc):
		#Test presence of file_loc
		self.rawfile=self._readFile(file_loc)
		self.global_vars={}
		self.runs=[]
		self._setRuns()
		self.derived_var={}
		self._parseFile()

	def _readFile(self,file_loc):
		a=pd.read_csv(file_loc,header=1,dtype=str)
		return a[a['!2']=='*'].drop(columns='!2')

	def _setRuns(self):
		self.runs = [c for c in self.rawfile.columns if c[0:3]=='RUN']

	def _parseFile(self):
		for run in self.runs:
			vals=self.rawfile[run].tolist()
			vars={}
			for i,var in enumerate(self.rawfile['VAR_NAME'].tolist()):
				if var[-2:]=='PC':
					vars[var]=float(vals[i])/100
				else:
					vars[var]=int(vals[i])
			vars['FUT_ACC_PER']=720 #missing from the file for some reason, but in the excel
			self.global_vars[run] = vars
			self._setVarsForRun(run)
	
	def _setVarsForRun(self,run_id):
		self.derived_var[run_id]={}
		self.derived_var[run_id]["m_val_int"]=(1+self.global_vars[run_id]['VAL_INT_PC'])**(1/12)
		self.derived_var[run_id]["disc_factor"]=1/self.derived_var[run_id]["m_val_int"]
		self.derived_var[run_id]["M_Re_Inflat"]=(1+self.global_vars[run_id]['CPI_GTH_PC'])**(1/12)
		self.derived_var[run_id]["Tot_Ann_Rate"]=self.global_vars[run_id]['AN_FII_PC']+self.global_vars[run_id]['AN_UFII_PC']+\
			self.global_vars[run_id]['AN_RCGCHG_PC']+self.global_vars[run_id]['AN_RCGUNC_PC']+self.global_vars[run_id]['AN_UNRCG_PC']
		if self.derived_var[run_id]["Tot_Ann_Rate"] < -1:
			self.derived_var[run_id]["M_Tot_Return"]=0
		else:
			self.derived_var[run_id]["M_Tot_Return"]=(1+self.derived_var[run_id]["Tot_Ann_Rate"])**(1/12)-1

		if self.derived_var[run_id]["Tot_Ann_Rate"]==0:
			self.derived_var[run_id]["M_FII"]=self.global_vars[run_id]['AN_FII_PC']/12
			self.derived_var[run_id]["M_UFII"]=self.global_vars[run_id]['AN_UFII_PC']/12
			self.derived_var[run_id]["M_RCG_CHG"]=self.global_vars[run_id]['AN_RCGCHG_PC']/12
			self.derived_var[run_id]["M_RCG_UNCHG"]=self.global_vars[run_id]['AN_RCGUNC_PC']/12
			self.derived_var[run_id]["M_UNRCG"]=self.global_vars[run_id]['AN_UNRCG_PC']/12
		else:
			self.derived_var[run_id]["M_FII"]=self.derived_var[run_id]["M_Tot_Return"]*self.global_vars[run_id]['AN_FII_PC']/self.derived_var[run_id]["Tot_Ann_Rate"]
			self.derived_var[run_id]["M_UFII"]=self.derived_var[run_id]["M_Tot_Return"]*self.global_vars[run_id]['AN_UFII_PC']/self.derived_var[run_id]["Tot_Ann_Rate"]
			self.derived_var[run_id]["M_RCG_CHG"]=self.derived_var[run_id]["M_Tot_Return"]*self.global_vars[run_id]['AN_RCGCHG_PC']/self.derived_var[run_id]["Tot_Ann_Rate"]
			self.derived_var[run_id]["M_RCG_UNCHG"]=self.derived_var[run_id]["M_Tot_Return"]*self.global_vars[run_id]['AN_RCGUNC_PC']/self.derived_var[run_id]["Tot_Ann_Rate"]
			self.derived_var[run_id]["M_UNRCG"]=self.derived_var[run_id]["M_Tot_Return"]*self.global_vars[run_id]['AN_UNRCG_PC']/self.derived_var[run_id]["Tot_Ann_Rate"]
		