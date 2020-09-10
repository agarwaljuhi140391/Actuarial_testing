import pandas as pd
import numpy as np
import inspect

class MPF:
	"""Class that is rather like pandas, but enables numba acceleration for actuarial calcs
	It takes a pandas DataFrame as input"""
	VALID_DTYPES=[np.int64,np.float64,np.dtype('O')]
	
	def __init__(self,df):
		for dtype in self.VALID_DTYPES:
			cols=list(df.dtypes[df.dtypes==dtype].index)
			self.mpf[dtype]=df[cols].values
		dtypes=df.dtypes

def _inspect_and_jit(func):
	args=inspect.getfullargspec(func)
	print(args)
	return func
	