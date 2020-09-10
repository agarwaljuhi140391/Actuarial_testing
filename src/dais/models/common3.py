from numba import njit, jit, vectorize, int64, int32, void, prange, cuda
import pandas as pd
import numpy as np
import networkx as nx
try:
	from ipywidgets import FileUpload,Button,Layout,HBox,VBox
	from IPython.core.display import display
except ImportError:
	pass
import io
import inspect
from collections import OrderedDict
import time
from operator import itemgetter
from .graphs import ModelDependencyGraph, FunctionNode, FunctionNodeBond
from .simple_dispatcher import tabledyn_helper_minmax,fast_searchsorted,fast_take,elementwise_min
from .complex_dispatcher import CompiledFunction,BondCompiledFunction,GPUCompiledFunction
import importlib
# importlib.reload(.complex_dispatcher)
# importlib.reload(.simple_dispatcher)
# importlib.reload(.graphs)
# from .complex_dispatcher import CompiledFunction
# from .graphs import ModelDependencyGraph, FunctionNode
# from .simple_dispatcher import func_fac1,func_fac2,func_fac3,func_fac4,func_fac5,func_fac6,func_fac7,fast_searchsorted,fast_take,elementwise_min

class BaseModel:
	"""This is an outline class from which other models can inherit common functionality"""
	debug=False
	def __init__(self,time_periods=1200):
		"""initializes the ordered dictionaries for variables"""
		self.internal_graph=ModelDependencyGraph()
		self.timings=OrderedDict()
		# self.out_vals=OrderedDict()
		# self.table_vals=OrderedDict()
		#table_mapper - a dict of dicts, showing key:field mappings for each table
		# self.table_mapper=OrderedDict()
		self.tables=OrderedDict()
		self.constants=OrderedDict()
		self.vals=OrderedDict()
		self.time_periods=time_periods
		self.executors={
			'Derived':self.execute_derived
			,'DerivedComplex':self.execute_derivedcomplex
			,'TablePolicy':self.execute_table
			,'TablePolicyTime':self.execute_tabledyn
			,'Summary':self.execute_summary
		}
		self.node_type=CompiledFunction
		self.graphnode_type=FunctionNode

	def set_targets(self,targetlist):
		"""targetlist is being initializedin the function"""
		self.targetlist=targetlist

	def create_tasklist(self,targetlist=None,executable=False):
		"""The function returns the tasklist. taskscope is created first and then from the taskscope tasklist is created
		1. If the targetlist passes is none and does not have targetlist as an attribute then taskscope is set as nodes
		2. else taskscope is set as the targetlist passed in the the function
		3. After taskscope has been created tasklist is created"""
		if targetlist is None and not hasattr(self,'targetlist'):
			taskscope=set(self.internal_graph.nodes)
		else:
			if targetlist is None:
				targetlist=self.targetlist
			taskscope=set(targetlist)
			for target in targetlist:
				taskscope.update(nx.ancestors(self.internal_graph,target))

		tasklist=[]
		for task in nx.topological_sort(self.internal_graph):
			if task in taskscope:
				if executable:
					if self.internal_graph.get_fnode(task).is_executable_or_derived():
						tasklist.append(task) 
				else:
					tasklist.append(task) 
		return tasklist

	def add_source_tables(self,df_dict):
		#accepts a dict of pandas dataframes.
		"""accepts a dictionary of pandas dataframe and prints the no of mapped and unmapped variables in tables"""
		self.tables.update(df_dict)
		unmapped=self.internal_graph.get_unmapped()
		print("Processing tables. {} unmapped variables to start with.".format(len(unmapped)))
		mapped=[]
		for node in unmapped:
			fnode=self.internal_graph.nodes[node]['funcnode']
			if fnode.source_type == 'TablePolicy':
				try:
					# print("Mapping {}, trying to find source {}".format(node,fnode.attr['source']))
					cols=[col for col in df_dict[fnode.attr['source']].columns if col not in fnode.attr['mapping'].keys()]
					if node in cols:
						# print("Mapped {}".format(node))
						fnode.set_mapped()
						mapped.append(node)
						self.internal_graph.update_color(node)
				except KeyError:
					# print("KeyError: {}".format(node))
					pass
			elif fnode.source_type == 'TablePolicyTime':
				# print("Mapping {}".format(node))
				if node in df_dict.keys():
					# print("Mapped {}".format(node))
					fnode.set_mapped()
					mapped.append(node)
					self.internal_graph.update_color(node)
		print("Processed tables. {} new variables mapped.".format(len(mapped)))

	def add_source_constants(self,const_dict):
		"""accepts a dict of name:value pairs as constants for the model to run and prints the mapped and unmapped constants"""
		#accepts a dict of name:value pairs as constants for the model to run
		self.constants.update(const_dict)
		keys=set(list(const_dict.keys()))
		unmapped=self.internal_graph.get_unmapped()
		print("Processing constants. {} unmapped variables to start with.".format(len(unmapped)))
		mapped=[]
		for node in unmapped:
			fnode=self.internal_graph.nodes[node]['funcnode']
			if node in keys:
				fnode.set_source_type('Constant')
				fnode.set_attributes({'shape':0})
				mapped.append(node)
				self.internal_graph.update_color(node)
		print("Processed tables. {} new variables mapped.".format(len(mapped)))


	def add_source_mpf(self,df):
		"""accepts a single pandas dataframe. Each record should be a model point. The function prints the no of mapped and unmapped
		variables in the mpf file
		1. the data frame passed is initialized to mpf variale and columns are also set in columns variable
		2. The columns are then assigned as fnodes , their source type and attributes are set
		3. then the nodes are then appended to mapped variables and color of the node is updated"""
		self.mpf=df
		columns=set(list(df.columns))
		unmapped=self.internal_graph.get_unmapped()
		print("Processing MPF. {} unmapped variables to start with.".format(len(unmapped)))
		mapped=[]
		for node in unmapped:
			# print("unmapped node: {}".format(node))
			if node in columns:
				# print("Mapping in {}".format(node))
				fnode=self.internal_graph.nodes[node]['funcnode']
				fnode.set_source_type('MPF')
				fnode.set_attributes({'type':self.mpf[node].to_numpy().dtype,'shape':1})
				mapped.append(node)
				self.internal_graph.update_color(node)
		print("Processed MPF. {} new variables mapped.".format(len(mapped)))

	def setfuncs(self,model_class,debug=False):
		"""Functional class is set in this function which contains different members like single_derivations,complex_derivations,summaries
		mappings and various functions then with the _inspect() we are adding nodes and edges for those varios members 
		if 'T' is present in internal graph nodes then value of funcnode is assigned to fnode , source type is set to 'T' , attrinutes are 
		set and the color of the graph is also updated"""
		self.functions_class=model_class
		self._inspect(debug)
		if 'T' in self.internal_graph.nodes:
			fnode=self.internal_graph.nodes['T']['funcnode']
			fnode.set_source_type('T')
			fnode.set_attributes({'type':np.int32,'shape':2})
			self.internal_graph.update_color('T')

	def _derive_linkages(self):
		"""In this function we are first assigning values to args that is the node values and then edges are being for linkages between 
		the nodes. If the nodes are not present as args then first nodes are added and then the edges
		1. First we assign the node values to the fnode function 
		2.  a.if the node source type is Derived'or 'DerivedComplex' and does not have parentname attribute then for those nodes then those
			variables are set to args variable
			b.for the source type DerivedComplex if particular arg from the args list is not present in attribute list then it is assigned 
			to the args otherwise if the attr['numba'] value is 'j' then arg whose value is not equal to node is set to args
			c.In the next step the edges are added for the list of args . If arg is present in the nodes then directly edge is added 
			otherwise node is added first and then edge is added
		3. if the node source type is TablePolicy'or 'TablePolicyTime'then the node mapping values are assigned to args and then if the
			args is present as internal graph node then directly edge is added otherwise node is added first and then the edge is added
		4. For the source type is summaries ,if is_summarysubvar() is false(i.e if does not have parentname as attribute) then bywars and 
			vars are assigned to args . If the args is present in node then edges are added if not then nodes are added first and then the 
			edges"""
		for node in list(self.internal_graph.nodes):
			fnode=self.internal_graph.nodes[node]['funcnode']
			"""Considers sources which are Derived and DerivedComplex"""
			if fnode.source_type in ('Derived','DerivedComplex'):
				#"""Ignore nodes that are not the parent in complexitems and sets value of args variable"""
				if not hasattr(fnode,'parentname'):
					args=list(inspect.signature(fnode.func).parameters.keys())
					# print(node)
					if fnode.source_type=='DerivedComplex':
						args=[arg for arg in args if arg not in fnode.attr['outvars'].keys()]
					elif fnode.attr['numba']=='j':
						args=[arg for arg in args if arg != node]
					for i,arg in enumerate(args):
						if arg in self.internal_graph.nodes:
							self.internal_graph.add_edge(arg,node,order=i)
						else:
							self.internal_graph.add_node(self.graphnode_type(arg))
							self.internal_graph.add_edge(arg,node,order=i)
			#"""Considers sources which are TablePolicy and TablePolicyTime"""
			elif fnode.source_type in ('TablePolicy','TablePolicyTime'):
				args=fnode.attr['mapping'].values()
				# print("Mapping table {}".format(node))
				for arg in args:
					if arg in self.internal_graph.nodes:
						# print("{} exists, adding edge".format(arg))
						self.internal_graph.add_edge(arg,node)
					else:
						# print("{} does not exist, adding node and edge".format(arg))
						self.internal_graph.add_node(self.graphnode_type(arg))
						self.internal_graph.add_edge(arg,node)
			#"""Considers Summary sources"""            
			elif fnode.source_type in ('Summary'):
				if not fnode.is_summarysubvar():
					args=fnode.attr['byvars']+fnode.attr['vars']
					for arg in args:
						if arg in self.internal_graph.nodes:
							self.internal_graph.add_edge(arg,node)
						else:
							self.internal_graph.add_node(self.graphnode_type(arg))
							self.internal_graph.add_edge(arg,node)

	def _inspect_singles(self,functions,debug=False):
		"""The following steps take place in the function:
		1. assigns the single_derivation dictionary from Function class to a variable singles
		2. then within that dictionary searches for the keys and if debug is True then
		3. then passes the variable in graphnode_type(single,'Derived') which is the FunctionNode class (in graphs.py).
			The __init__ function then initializes value of variable names to the first parameter passed(ultimately single_derivations) and
			the second parameter passed is first checked whether it is a valid source_type and If valid then assigned to source_type and
			value of mapped variable is assigned .If the source_type  is not valid then an error message is printed 
			All these  name,source_type and mapped values are then stored in variable node
		4. In the next two steps the function class and attributes for the single_derivation are set
		5. the data type of the attributes is then converted to numpy format and finally nodes values are added for the single derivation 
		variables"""
		singles=self.functions_class.single_derivations
		for single in singles.keys():
			if debug:
				print("Processing {}".format(single))
			node=self.graphnode_type(single,'Derived')
			node.set_func(functions[single])
			node.set_attributes(singles[single])
			node.attr['type']=np.dtype(node.attr['type'])
			self.internal_graph.add_node(node)
		
	def _inspect_complexes(self,functions,debug=False):
		"""The following steps take place in the function:
		1. assigns the complex_derivations dictionary from Function class to a variable complexes
		2. then within that dictionary searches for the keys and if debug is True then
		3. then passes the variable in graphnode_type(complex_item,'DerivedComplex') which is the FunctionNode class (in graphs.py).
			The __init__ function then initializes value of variable names to the first parameter passed(ultimately complex_derivations)
			and the second parameter passed is first checked whether it is a valid source_type and If valid then assigned to source_type 
			value of mapped variable is assigned .If the source_type  is not valid then an error message is printed 
			All these  name,source_type and mapped values are then stored in variable node
		4. In the next two steps the function class and attributes for the complex_derivations are set
		5. For particular complex_items the outvars dictionary for each item are then stored in strict order in args variable and later the 		value for each outvar key is assigned to node.attr['outvars'][var]['order'] and data type from outvars dict is stored in in 
		node.attr['outvars'][var]['type']
		6. the data type of the attributes is then converted to numpy format and finally nodes are added for the complex derivation 
			variables
		7. For the variables creating the complex_item variable the parentname is set as the particular complex_item and attributes are set
		for that variable 
		8.Finally nodes and edges are added for the varibales"""
		complexes=self.functions_class.complex_derivations
		for complex_item in complexes.keys():
			if debug:
				print("Processing {}".format(complex_item))
			node=self.graphnode_type(complex_item,'DerivedComplex')
			node.set_func(functions[complex_item])
			node.set_attributes(complexes[complex_item])
			args=list(inspect.signature(node.func).parameters.keys())
			argorders={arg:i for i,arg in enumerate(args)}
			for var in node.attr['outvars'].keys():
				node.attr['outvars'][var]['order']=argorders[var]
				node.attr['outvars'][var]['type']=np.dtype(node.attr['outvars'][var]['type'])
			self.internal_graph.add_node(node)
			for outvar in complexes[complex_item]['outvars'].keys():
				node=self.graphnode_type(outvar,'DerivedComplex')
				node.set_complexparent(complex_item)
				node.set_attributes(complexes[complex_item]['outvars'][outvar])
# 				node.attr['type']=np.dtype(node.attr['type'])
				node.attr['order']=argorders[outvar]
				self.internal_graph.add_node(node)
				self.internal_graph.add_edge(complex_item,outvar,order=argorders[outvar])
			
	def _inspect_tables(self,functions,debug=False):
		"""The following steps take place in the function:
		1. assigns the mappings dictionary from Function class to a variable table_vars  
		2. then within that dictionary searches for the keys and if shape is 2
			then passes the variable in graphnode_type(single,'TablePolicyTime') and if shape is 1 then
			graphnode_type(single,'TablePolicyTime') is called which is the FunctionNode class (in graphs.py).
			The __init__ function then initializes value of variable names to the first parameter passed(ultimately mappings) and
			the second parameter passed is first checked whether it is a valid source_type and If valid then assigned to source_type and
			value of mapped variable is assigned .If the source_type  is not valid then an error message is printed 
			All these  name,source_type and mapped values are then stored in variable node
		3. In the next two steps attributes for the mappings are set and the data type of the attributes is then converted to numpy format
		and finally nodes values are added for the single derivation 
		variables"""        
		table_vars=self.functions_class.mappings
		for table_var in table_vars.keys():
			if debug:
				print("Processing {}".format(table_var))
			if table_vars[table_var]['shape']==2:
				node=self.graphnode_type(table_var,'TablePolicyTime')
			else:
				node=self.graphnode_type(table_var,'TablePolicy')
			node.set_attributes(table_vars[table_var])
			node.attr['type']=np.dtype(node.attr['type'])
			self.internal_graph.add_node(node)
		
	def _inspect_summaries(self,functions,debug=False):
		"""The following steps take place in the function:
		1. assigns the summaries dictionary from Function class to a variable summaries  
		2. then within that dictionary searches for the keys and
			then calls graphnode_type(summary,'Summary') which is the FunctionNode class (in graphs.py).
			The __init__ function then initializes value of variable names to the first parameter passed(ultimately summaries) and
			the second parameter passed is first checked whether it is a valid source_type and If valid then assigned to source_type and
			value of mapped variable is assigned .If the source_type  is not valid then an error message is printed 
			All these  name,source_type and mapped values are then stored in variable node
		3. In the next step attributes for the mappings are set
		4. For each vars of the key of the summaries and the data type of the attributes is then converted to numpy format
		and finally nodes values are added for the single derivation 
		variables"""       
		summaries=self.functions_class.summaries
		for summary in summaries.keys():
			if debug:
				print("Adding summary", summary)
			node=self.graphnode_type(summary,'Summary')
			node.set_attributes(summaries[summary])
			self.internal_graph.add_node(node)
			subnodes={}
			for summary_var in summaries[summary]['vars']:
				if debug:
					print("Adding summary vars: ",summary_var)
				new_nodename=summaries[summary]['func']+'_'+summary_var
				node=self.graphnode_type(new_nodename,'Summary')
				node.set_complexparent(summary)
				self.internal_graph.add_node(node)
				self.internal_graph.add_edge(summary,new_nodename)
				subnodes[summary_var]=new_nodename
			self.internal_graph.get_fnode(summary).attr.update({'subnodes':subnodes})


	def _inspect(self,debug=False):
		"""The function first creates an ordered dictionary of the functions that are to be executed and calls individual _inspect functions for singles,complexes,tables and summaries and finally calls _derive_linkages() where nodes and edges are being created."""
		functions=OrderedDict(inspect.getmembers(self.functions_class,inspect.isfunction))

		self._inspect_singles(functions,debug)	
		self._inspect_complexes(functions,debug)	
		self._inspect_tables(functions,debug)
		self._inspect_summaries(functions,debug)
		print(len(self.internal_graph.nodes), "variable definitions imported.")
		self._derive_linkages()

	def compile_or_load(self,targets=None):
		tasklist=self.create_tasklist(targets,executable=True)

		for task in tasklist:
			fnode=self.internal_graph.get_fnode(task)
			# print(task)
			if fnode.is_executable() and fnode.is_code_backed():
				edges=self.internal_graph.in_edges(nbunch=task,data=True)
				argnodes=[self.internal_graph.nodes[argnode]['funcnode'] for argnode,_task,_data in edges]
				edge_order={arg:data['order'] for arg,_task,data in edges}
				ordered_argnodes=sorted(argnodes,key=lambda x: edge_order[x.name])
				fnode.set_compiled_func(self.node_type(fnode,ordered_argnodes,self.functions_class.__name__ + type(self).__name__,debug=self.debug))

	def initialise(self,targets=None):
		"""Targets is a list of final variables for which to solve. None means all and is default.
		The initialise function creates input variables for T, Constants and MPF values from the input sources.
		Execution can only begin once initialised."""
		tasklist=self.create_tasklist(targets)
		overlap=set(tasklist).intersection(set(self.internal_graph.get_unmapped()))
		assert len(overlap)==0
		
		self.vals['T'],_=np.broadcast_arrays(np.arange(self.time_periods,dtype=np.int32).reshape((1,self.time_periods))
										  ,self.mpf[self.mpf.columns[0]].to_numpy().reshape((-1,1)))
		self.vals['T']=self.vals['T'].copy()

		for node in tasklist:
			fnode=self.internal_graph.nodes[node]['funcnode']
			if fnode.is_original_input():
				if fnode.source_type=='MPF':
					self.vals[node]=self.mpf[node].to_numpy().reshape((-1,1))
				elif fnode.source_type=='Constant':
					self.vals[node],_=np.broadcast_arrays(np.asarray([[self.constants[node]]]).reshape((1,1))
															,self.mpf[self.mpf.columns[0]].to_numpy().reshape((-1,1)))
					self.vals[node]=self.vals[node].copy()
				fnode.attr['type']=self.vals[node].dtype
		self.compile_or_load(targets)

# 	def execute_summary(self,summarynode):
# 		fnode=self.internal_graph.get_fnode(summarynode)
# 		assert fnode.source_type=='Summary'
# 		indexes=fnode.attr['byvars']
# 		vals=fnode.attr['vars']
# 		assert len(set(indexes+vals).intersection(set(self.vals.keys())))==len(set(indexes+vals))
# 		if len(indexes)==1:
# 			index=self.vals[indexes[0]].ravel()
# 			columns=['T{}'.format(val) for val in self.vals['T'][0]]
# 		else:
# 			print("We don't have multi-index implementation for summaries yet")
# 		subnodes=fnode.attr['subnodes']
# 		for val in subnodes.keys():
# 			df=pd.DataFrame(data=self.vals[val],columns=columns,index=index)
# 			self.vals[subnodes[val]]=df.groupby(df.index).agg(fnode.attr['func'])

	def execute_summary(self,summarynode):
		"""I think this could run faster. Needs a bit of thought. Save for another day
		1. gets the funcnode for summaries and assigns to fnode
		2. byvars are assigned to indexes and vars are assigned to vals
		3. On the basis of the attr shape columns and dataframes are defined and then aggregated at func level to get the summaries 
		information"""
	# I think this could run faster. Needs a bit of thought. Save for another day
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
		for var in subnodes.keys():
			if self.internal_graph.get_fnode(var).attr['shape']==2:
				columns=range(self.vals[var].shape[1])
				df=pd.DataFrame(columns=columns,data=self.vals[var])
			elif self.internal_graph.get_fnode(var).attr['shape']==1:
				df=pd.DataFrame({var:self.vals[var][:,0]})
			for byvar in byvars:
				assert self.internal_graph.get_fnode(byvar).attr['shape']==1
				df.insert(0,byvar,self.vals[byvar].ravel())
			self.tables[subnodes[var]]=df.groupby(byvars).agg(fnode.attr['func']).reset_index()


	def execute_table(self,tablenode):
		#this is a single-threaded activity
		"""1. gets the funcnode for tables and assigns to fnode
		2. mapping keys are assigned to tablefields and vars are assigned to vals
		3. then datamaps are created for different tablefields and - not completely sure whats happening in line for datamaps and table"""
		fnode=self.internal_graph.get_fnode(tablenode)
		assert fnode.source_type == 'TablePolicy'
		tablefields=list(fnode.attr['mapping'].keys())
		mapfields=[fnode.attr['mapping'][f] for f in tablefields]
		#Check to make sure all upstream dependencies have indeed been fulfilled.
		assert len(set(mapfields).intersection(set(self.vals.keys())))==len(set(mapfields))
		datamaps=pd.DataFrame({k:self.vals[fnode.attr['mapping'][k]].ravel() for k in tablefields})
		table=self.tables[fnode.attr['source']][tablefields + [tablenode]]
		self.vals[tablenode]=pd.merge(datamaps
										,table
										,how='left'
										,on=tablefields
		)[tablenode].to_numpy().reshape((-1,1))

	def execute_derived(self,derivednode):
		"""1. gets the funcnode for derived node and assigns to fnode
		2. fnode calls the numba_func and stores the argnode name in argnodes"""
		fnode=self.internal_graph.nodes[derivednode]['funcnode']
		# assert fnode.source_type in ['Derived']
		func=fnode.compiled_func.numba_func
		argnodes=[argnode.name for argnode in fnode.compiled_func.argnodes]
		args=[self.vals[arg] for arg in argnodes]
		self.vals[derivednode]=func(*args)

	def execute_derivedcomplex(self,derivednode):
		"""1. gets the funcnode for DerivedComplex node and assigns to fnode
		2. fnode calls the numba_func and stores the argnode name in argnodes"""
		fnode=self.internal_graph.nodes[derivednode]['funcnode']
		assert fnode.source_type in ['DerivedComplex']
		func=fnode.compiled_func.numba_func
		argnodes=[argnode.name for argnode in fnode.compiled_func.argnodes]
		args=[self.vals[arg] for arg in argnodes]
		results=func(*args)
		self.vals.update(results)

# 	def execute_tabledyn(self,tablenode):
# 		# print(tablenode)
# 		fnode=self.internal_graph.get_fnode(tablenode)
# 		assert fnode.source_type == 'TablePolicyTime'
# 		tablefields=list(fnode.attr['mapping'].keys())
# 		mapfields=[fnode.attr['mapping'][f] for f in tablefields]
# 		#Check to make sure all upstream dependencies have indeed been fulfilled.
# 		assert len(set(mapfields).intersection(set(self.vals.keys())))==len(set(mapfields))
# 		#'col' is the special value given to to the colunms on a time-based lookup
# 		row_tablefields=[k for k in tablefields if k!= 'col']
# 		datamaps=pd.DataFrame({k:self.vals[fnode.attr['mapping'][k]].ravel() for k in row_tablefields})
# 		# print(datamaps)
# 		table=self.tables[tablenode][row_tablefields].reset_index()
# 		rownumbers=pd.merge(datamaps,table,how='left'
# 							,on=row_tablefields)['index'].to_numpy().reshape((self.mpf.shape[0],1))
# 		# print(rownumbers)
# 		colnames=[int(col) for col in self.tables[tablenode].columns if not ((col in row_tablefields) or (col in ['index','Description','DESCRIPTION']))]
# 		colval=self.vals[fnode.attr['mapping']['col']]
# 		if 'autocapcol' in fnode.attr:
# 			if fnode.attr['autocapcol']:
# 				colval=elementwise_min(colval.ravel(),max(colnames))
# 		col_indices=fast_searchsorted(np.asarray(colnames),colval.ravel())
# 		row_indices2,_temp=np.broadcast_arrays(rownumbers,self.vals['T'])
# 		del _temp
# 		row_indices3=row_indices2.ravel()
# 		row_indices3.flags.writeable=False
# 		self.vals[tablenode]=fast_take(self.tables[tablenode][colnames].to_numpy(),row_indices3,col_indices).reshape((self.mpf.shape[0],self.vals['T'].shape[1]))

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
		datamaps=pd.DataFrame({k:self.vals[fnode.attr['mapping'][k]].ravel() for k in row_tablefields})
		# print(datamaps)
		table=self.tables[tablenode][row_tablefields].reset_index()
		rownumbers=pd.merge(datamaps,table,how='left'
							,on=row_tablefields)['index'].to_numpy().reshape((self.mpf.shape[0],1))
		# print(rownumbers)
		####FROM HERE IS REWRITTEN TO BE FASTER. EXPLICIT,EFFICIENT/FAST CHECK BUILT IN TO PREVENT SILLY MISTAKES IN INPUT TABLE###
		colnames=[]
		for col in self.tables[tablenode].columns:
			if type(col)==str:
				if col.isnumeric():
					colnames.append(col)
			elif type(col)==int:
				colnames.append(col)
		colnames=np.asarray(colnames)
# 		colnames=np.asarray([int(col) for col in self.tables[tablenode].columns if col.isnumeric()])
		colnames_min=np.min(colnames)
		colnames_max=np.max(colnames)
		assert np.allclose(colnames,np.arange(colnames_min,colnames_max+1))
		colval=self.vals[fnode.attr['mapping']['col']].ravel()
# 		if 'autocapcol' in fnode.attr:
# 			if fnode.attr['autocapcol']:
# 				colval=np.minimum(colval,colnames_max)
# 				colval=np.maximum(colval,colnames_min)
## 		col_indices=fast_searchsorted(np.asarray(colnames),colval)
# 		col_indices=colval-colnames_min
		col_indices=tabledyn_helper_minmax(colval,colnames_min,colnames_max)
		row_indices2=np.broadcast_to(rownumbers,self.vals['T'].shape)
# 		del _temp
		row_indices3=row_indices2.ravel()
		row_indices3.flags.writeable=False
		self.vals[tablenode]=fast_take(self.tables[tablenode][colnames].to_numpy(),row_indices3,col_indices).reshape((self.vals['T'].shape[0],self.vals['T'].shape[1]))


	def execute(self,targets=None,dokeep=None):
		"""The function first creats a task list by calling create_tasklist() and then checks whether the task is executable and if true 
		then executes the tasks by calling the executors in the complex_dispatcher.py"""
		tasklist=self.create_tasklist(targets,executable=True)
		# if targets is None:
			# tasklist = [task for task in nx.topological_sort(self.internal_graph)
				# if self.internal_graph.get_fnode(task).is_executable_or_derived()]
		# else:
			# all_ancestors=set(targets)
			# for target in targets:
				# all_ancestors.update(nx.ancestors(self.internal_graph,target))
			# tasklist = [task for task in nx.topological_sort(self.internal_graph)
				# if self.internal_graph.get_fnode(task).is_executable_or_derived() and (task in all_ancestors)]
		if self.debug:
			print(tasklist)
		if dokeep is None:
			self.dokeep=set([]).update(set(self.targetlist))
		elif dokeep=="ALL":
			self.dokeep=set(tasklist)
		else:
			self.dokeep=set(dokeep)

		done_tasks=set()
		for task in tasklist:
			if self.internal_graph.get_fnode(task).is_executable():
				start=time.time()
				fnode=self.internal_graph.get_fnode(task)
				if self.debug:
					print(fnode,fnode.source_type)
				self.executors[fnode.source_type](task)
				if self.debug:
					print("Executed ",fnode.name)
				self.timings[task]=time.time()-start
			else:
				if self.debug:
					print(task," not executable")
			if (fnode.source_type=='DerivedComplex') and not fnode.can_have_data():
				derived_nodes=[node for task,node in self.internal_graph.out_edges(task)]
				done_tasks.update(derived_nodes+[task])
			else:
				done_tasks.update([task])
			ttr=self.tasks_to_remove(task,done_tasks)
			for item in ttr:
				if item not in self.dokeep:
					try:
						del self.vals[item]
					except KeyError as e:
						print("Failed to delete: ",item)
						raise ValueError


	def tasks_to_remove(self,task,done_tasks,dokeep=[]):
		"""Removes tasks that are not eligible"""
		tasks=[]
		possibles=[ins for ins,task in self.internal_graph.in_edges(task)]
		# + [out for task,out in self.internal_graph.out_edges(task) if (len(self.internal_graph.out_edges(out))==0)]
		if len(self.internal_graph.out_edges(task))==0:
			possibles.append(task)
		eligibles=[ins for ins in possibles if not self.internal_graph.get_fnode(ins).is_original_input() and self.internal_graph.get_fnode(ins).can_have_data()]
		for arg in eligibles:
			remaining_deps=[outs for task,outs in self.internal_graph.out_edges(arg) if outs not in done_tasks]
			if len(remaining_deps)==0 and arg not in dokeep:
				# print("removing",arg)
				tasks.append(arg)
		return tasks

	def compute_size(self,tasks_in_memory):
		"""computes the size of the task"""
		length=self.vals['T'].shape[0]
		width=self.vals['T'].shape[1]
		unit_sizes={np.dtype(np.int32):4,np.dtype(np.int64):8,np.dtype(np.float64):8,np.dtype('O'):8}
		sizes=OrderedDict()
		for node in tasks_in_memory:
			fnode=self.internal_graph.nodes[node]['funcnode']
			if fnode.can_have_data():
	# 				print(node)
				dtype=fnode.attr['type']
				if fnode.attr['shape']==2:
					size=unit_sizes[dtype]*length*width
				elif fnode.attr['shape']==1:
					size=unit_sizes[dtype]*length
				else:
					size=8
				sizes[node]=size
		return sum(sizes.values())

	def maxsize(self,targets=None,dokeep=None,execute_list=None):
		"""This function returns the size of the task taking the maximum memory"""
		if dokeep is None:
			dokeep=set(list(self.internal_graph.nodes))
		if execute_list is None:
			execute_list= [task for task in nx.topological_sort(self.internal_graph)
					if self.internal_graph.get_fnode(task).is_executable_or_derived()]
		tasks_in_memory=set([fnode.name for fnode in self.internal_graph.get_fnodes() if fnode.is_original_input()])
		done_tasks=set()
		sizes=OrderedDict()
		for task in execute_list:
			fnode=self.internal_graph.get_fnode(task)
			if (fnode.source_type=='DerivedComplex') and not fnode.can_have_data():
				derived_nodes=[node for task,node in self.internal_graph.out_edges(task)]
				done_tasks.update(derived_nodes+[task])
				tasks_in_memory.update(derived_nodes+[task])
			else:
				done_tasks.update([task])
				tasks_in_memory.update([task])
			ttr=self.tasks_to_remove(task,done_tasks)
			for item in ttr:
				if item not in dokeep:
					try:
						tasks_in_memory.remove(item)
					except ValueError as e:
						print("Failed to remove: ",item)
						raise ValueError
			sizes[task]=self.compute_size(tasks_in_memory)
		return max(sizes.values())

	def visualise(self,target_node=None,targetlist=None,format='svg',update_tooltips=True):
		"""The function visualizes the tasks--more info required"""
		if update_tooltips:
			self.internal_graph.update_tooltips()
		if target_node is None:
			tasklist=self.create_tasklist(targetlist=targetlist)
			A=nx.nx_agraph.to_agraph(nx.subgraph(self.internal_graph,tasklist))
		else:	
			sub_graph_nodes=list(nx.all_neighbors(self.internal_graph,target_node))+[target_node]
			sub_graph=nx.subgraph(self.internal_graph,sub_graph_nodes)
			A=nx.nx_agraph.to_agraph(sub_graph)
		if format=='svg':
			return A.draw(format='svg',prog='dot')
		else:
			return A.draw(format='png',prog='dot')


class BondModel(BaseModel):
	"""Just like a normal model, but with a special capability to discount with yield rates
	turning 1200 datapoints of bond cashflow into annual valuation forecasts for 100 years"""
	def __init__(self,time_periods=1200,val_time_periods=120):
		super().__init__(time_periods)
		self.val_time_periods=val_time_periods
		self.node_type=BondCompiledFunction
		self.graphnode_type=FunctionNodeBond
# 		self.executors['DerivedBond']=self.execute_derived

	def setfuncs(self,model_class):
		super().setfuncs(model_class)
		if 'VT' in self.internal_graph.nodes:
			self.internal_graph.nodes['VT']['funcnode']=self.graphnode_type('VT')
			fnode=self.internal_graph.get_fnode('VT')
			fnode.set_source_type('VT')
			fnode.set_attributes({'type':np.int32,'shape':2,'2d':'VT'})
			self.internal_graph.update_color('VT')

	def initialise(self):
		self.vals['VT'],_=np.broadcast_arrays(np.arange(self.val_time_periods,dtype=np.int32).reshape((1,self.val_time_periods))
										  ,self.mpf[self.mpf.columns[0]].to_numpy().reshape((-1,1)))
		self.vals['VT']=self.vals['VT'].copy()
		super().initialise()

	def _inspect_complexes(self,functions,debug=False):
		complexes=self.functions_class.complex_derivations
		for complex_item in complexes.keys():
			if debug:
				print("Processing {}".format(complex_item))
			#THIS IS WHERE THE CHANGE IS, LINE BELOW HERE. ENABLES ENHANCED FUNCTIONALITY IN FUNCTIONNODEBOND FOR THE ALTERNATIVE TIME DIMENSION
			node=self.graphnode_type(complex_item,'DerivedComplex')
			#END OF CHANGES
			node.set_func(functions[complex_item])
			node.set_attributes(complexes[complex_item])
			args=list(inspect.signature(node.func).parameters.keys())
			argorders={arg:i for i,arg in enumerate(args)}
			for var in node.attr['outvars'].keys():
				node.attr['outvars'][var]['order']=argorders[var]
				node.attr['outvars'][var]['type']=np.dtype(node.attr['outvars'][var]['type'])
			self.internal_graph.add_node(node)
# 			print(argorders)
			for outvar in complexes[complex_item]['outvars'].keys():
				node=FunctionNode(outvar,'DerivedComplex')
				node.set_complexparent(complex_item)
				node.set_attributes(complexes[complex_item]['outvars'][outvar])
# 				node.attr['type']=np.dtype(node.attr['type'])
				node.attr['order']=argorders[outvar]
				self.internal_graph.add_node(node)
				self.internal_graph.add_edge(complex_item,outvar,order=argorders[outvar])

	def set_source_yieldrates(self,yields):
		if 'index' not in yields.columns:
			yields=yields.sort_values(by=['SIMULATION','ECONOMY','TERM']).reset_index()
		yields.rename(columns={'index':'YIELD_IDX'},inplace=True)
		self.yieldrates_terms=yields['TERM'].unique()
		self.yieldrates_sims=yields['SIMULATION'].unique()
		self.yieldrates_econs=yields['ECONOMY'].unique()
		#Sanity checks on right number of terms / economies per simulation
		np.testing.assert_array_equal(
			yields.groupby(['SIMULATION'])['SIMULATION'].agg('count').unique()
			,np.array([len(self.yieldrates_econs)*len(self.yieldrates_terms)]))
		np.testing.assert_array_equal(
			yields.groupby(['SIMULATION','ECONOMY'])['SIMULATION'].agg('count').unique()
			,np.array([len(self.yieldrates_terms)]))
		#set the table for the join in of the yield indexes
		# self.tables['YIELDS']=yields[['SIMULATION','ECONOMY','TERM','index']].query('TERM=={}'.format(min(self.yieldrates_terms)))
		self.add_source_tables({'YIELDS':yields[['SIMULATION','ECONOMY','TERM','YIELD_IDX']].query('TERM=={}'.format(min(self.yieldrates_terms)))[['SIMULATION','ECONOMY','YIELD_IDX']]})
		#Set the raw (interpolated) yield rates for fast direct access
		self.yieldyears=[c for c in yields.columns if c not in set(['ECONOMY','TERM','SIMULATION','YIELD_IDX'])]
		self.vals['YIELDS']=yields[self.yieldyears].to_numpy()
		fnode=self.graphnode_type('YIELDS',source_type='TableBond')
		fnode.set_attributes({'shape':3,'type':np.float64})
		self.internal_graph.add_node(fnode)
		self.internal_graph.update_color(fnode.name)
		
# 	def execute_derived_bonds(self,drivedbondnode):
# 		fnode=self.internal_graph.nodes[drivedbondnode]['funcnode']
# 		assert fnode.source_type in ['DerivedBond']
# 		func=fnode.compiled_func.numba_func
# 		argnodes=[argnode.name for argnode in fnode.compiled_func.argnodes]
# 		args=[self.vals[arg] for arg in argnodes]
# 		self.vals[derivednode]=func(*args)

class GUIBaseModel(BaseModel):
	def interface_inputs(self):
		mpf_input=FileUpload(description="Upload MPF Parquet file",multiple=False,layout=Layout(width="250px",height="50px"))
		tables_input=FileUpload(description="Upload Excel tables files",multiple=True,layout=Layout(width="250px",height="50px"))
		constants_input=FileUpload(description="Upload Excel constants file",multiple=False,layout=Layout(width="250px",height="50px"))
		process_files_button=Button(description="Process Files")
		hbox=HBox([mpf_input,tables_input,constants_input])
		vbox=VBox([hbox,process_files_button])
		
		def process_files(value):
			for f in mpf_input.value.keys():
				print("Processing uploaded MPF file:",f)
				self.add_source_mpf(pd.read_parquet(io.BytesIO(mpf_input.value[f]['content'])))
			for f in tables_input.value.keys():
				print("Processing uploaded tables file:",f)
				self.add_source_tables(pd.read_excel(io.BytesIO(tables_input.value[f]['content']),sheet_name=None))
			for f in constants_input.value.keys():
				df=pd.read_excel(io.BytesIO(constants_input.value[f]['content']))
				constants={}
				for row in df.itertuples():
					if int(row[2])==row[2]:
						constants[row[1]]=int(row[2])
					else:
						constants[row[1]]=row[2]
				self.add_source_constants(constants)
		
		process_files_button.on_click(process_files)
		interface=vbox
		return display(interface)
		
class RollForwardModel(BaseModel):
	pass
	#We need to have new types:
	# - consolidation
	# - alignment attribute - PCY
	# - linkages from tables to PCY, POOL,CATEGORY,PURCHASE_YEAR
	# Some grouping of nodes into a single execution template. Is this even possible?
	# What gets responsibility for what? The wrapper template has to be this class or a brand new one? The individual functions included within the loops can be another type of dispatcher?
	
		
	def SVGvisualise(self,*args,**kwargs):
		from IPython.display import SVG 
		return SVG(self.visualise(*args,**kwargs))
		
class GUIBondModel(BondModel,GUIBaseModel):
	pass
	
class GUIMixModel(GUIBondModel):
	def set_submodels(self,models):
		self.submodels=models
	def add_source_tables(self,tables):
		for model in self.submodels:
			print("SubModel: ",model.functions_class.__name__)
			model.add_source_tables(tables)
	def add_source_constants(self,constants):
		for model in self.submodels:
			print("SubModel: ",model.functions_class.__name__)
			model.add_source_constants(constants)
	def set_source_yieldrates(self,yields):
		for model in self.submodels:
			print("SubModel: ",model.functions_class.__name__)
			model.set_source_yieldrates(yields)
	def set_targets(self,targets):
		for model in self.submodels:
			print("SubModel: ",model.functions_class.__name__)
			model.set_targets(targets)
	def initialise(self):
		for model in self.submodels:
			print("SubModel: ",model.functions_class.__name__)
			model.initialise()
	def execute(self,dokeep=None):
		for model in self.submodels:
			print("SubModel: ",model.functions_class.__name__)
			model.execute(dokeep=dokeep)


	#also do for initialise, execute
	#execute should also summaries up the outputs into a single set