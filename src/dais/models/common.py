from numba import jit
import pandas as pd
import numpy as np
import networkx as nx
import inspect
from collections import OrderedDict
import time

class BaseModel:
	"""This is an outline class from which other models can inherit common functionality"""
	def __init__(self):
		self.internal_graph=nx.DiGraph()
		
		self.jit_funcs=OrderedDict()
		
		self.out_vars=OrderedDict()
		self.out_vals=OrderedDict()
		
		self.table_vars=OrderedDict()
		self.table_vals=OrderedDict()
		#table_mapper - a dict of dicts, showing key:field mappings for each table
		self.table_mapper=OrderedDict()

	def _process_mapper(self,table_mappers):
		for row in table_mappers.itertuples():
			if row.Table in self.table_mapper:
				self.table_mapper[row.Table].update({row.Key:row.Mapping})
			else:
				self.table_mapper[row.Table]={row.Key:row.Mapping}
				
	def _process_tables(self,tables,source):
		Master=tables.pop('Master')
		Keys=tables.pop('Keys')
		tablekeys=OrderedDict()
		flat_tables=OrderedDict()
		if len(Master)>0:
			for row in Keys.itertuples():
				if row.Table in tablekeys:
					tablekeys[row.Table].append(row.Key)
				else:
					tablekeys[row.Table]=[row.Key]
			for record in Master.itertuples():
				flat_tables[record.Table]={'table':tables[record.Table]
										  ,'source':source
										  ,'type':record.Type
										  ,'keys':tablekeys[record.Table]}
		return flat_tables
		
	def _flatter_tables(self,flat_tables,table_mappers,specials=['Description','DESCRIPTION']):
		flatter_tables={}
		for tabname in flat_tables.keys():
			if flat_tables[tabname]['type']=='named_var':
				out_tabs=[var for var in flat_tables[tabname]['table'].columns if var not in flat_tables[tabname]['keys']+specials]
				for out_tab in out_tabs:
					flatter_tables[out_tab]={'table': flat_tables[tabname]['table'][flat_tables[tabname]['keys']+[out_tab]],
											 'keys': flat_tables[tabname]['keys']}
					for row in table_mappers[table_mappers.Table==tabname].itertuples():
						if out_tab in self.table_mapper:
							self.table_mapper[out_tab].update({row.Key:row.Mapping})
						else:
							self.table_mapper[out_tab]={row.Key:row.Mapping}
			elif flat_tables[tabname]['type']=='dynamic':
				flatter_tables[tabname]={'table': pd.melt(flat_tables[tabname]['table'][[c for c in flat_tables[tabname]['table'].columns if c not in specials]],
													id_vars=flat_tables[tabname]['keys'],
													var_name='col',
													value_name=tabname),
										 'keys': flat_tables[tabname]['keys']+['col']}
				for row in table_mappers[table_mappers.Table==tabname].itertuples():
					if tabname in self.table_mapper:
						self.table_mapper[tabname].update({row.Key:row.Mapping})
					else:
						self.table_mapper[tabname]={row.Key:row.Mapping}
		self.table_vars.update(flatter_tables)
		
	def configure(self,constants,common_file,be_file,mappers):
		self.CONSTANTS=constants
		common_tables=pd.read_excel(common_file,sheet_name=None)
		be_tables=pd.read_excel(be_file,sheet_name=None)
		table_mappers=pd.read_excel(mappers,sheet_name='Mappers')
		a=self._process_tables(common_tables,'common')
		a.update(self._process_tables(be_tables,'be'))
		self._flatter_tables(a,table_mappers)
# 		self._process_mapper(table_mappers)
		
	def load_mpf(self,mpf_file,headers_and_size_only=False):
		if headers_and_size_only:
			mpf=pd.read_table(mpf_file,header=0,nrows=1)
			self.mpf_fields=mpf.columns
		else:
			self.mpf=pd.read_table(mpf_file,header=0)
			self.mpf_fields=self.mpf.columns
			
	def resolve_arg(self,arg):
		if arg in self.mpf_fields:
			return 'mpf'
		elif arg in self.out_vars:
			return 'derived'
		elif arg in self.CONSTANTS:
			return 'const'
		elif arg in self.table_vars.keys():
			return 'table'
		else:
			print(arg,'resolve fail')
			return 'fail'
	
	def execute(self):
		tasklist = [task for task in nx.topological_sort(self.internal_graph)
			if self.internal_graph.nodes[task]['table'] in ['derived','table']]
# 		print(tasklist)
		for task in tasklist:
			print(task)
			if self.internal_graph.nodes[task]['table'] == 'table':
				mapping=self.table_vars[task]['keys']
				#prepare source
				source_dict=OrderedDict()
				for key in mapping:
					loc=self.resolve_arg(self.table_mapper[task][key])
					if loc=='mpf':
						source_dict[key]=self.mpf[self.table_mapper[task][key]].to_numpy()
					elif loc=='const':
						source_dict[key]=self.CONSTANTS[self.table_mapper[task][key]]
					elif loc=='derived':
						source_dict[key]=self.out_vals[self.table_mapper[task][key]]
					elif loc=='table':
						source_dict[key]=self.table_vals[self.table_mapper[task][key]]
# 				print(task,source_dict)
				source=pd.DataFrame(source_dict,index=self.mpf.index)
				#do the join
				start=time.time()
				self.table_vals[task]=pd.merge(source,
						self.table_vars[task]['table'],
						how='left',
						on=self.table_vars[task]['keys'])[task].to_numpy()
				end=time.time()
# 				print(task, end-start)
			elif self.internal_graph.nodes[task]['table'] == 'derived':
				args=self.internal_graph.in_edges(nbunch=task,data=True)
				args=[arg for arg,func,attr in sorted(args, key=lambda item: item[2]['order'])]
				tables=[self.internal_graph.nodes[arg]['table'] for arg in args]
				resolved_args=[]
				for arg in args:
					source=self.resolve_arg(arg)
# 					print(task,arg,source)
					if source=='mpf':
						resolved_args.append(self.mpf[arg].to_numpy())
					elif source=='const':
						resolved_args.append(self.CONSTANTS[arg])
					elif source=='derived':
						resolved_args.append(self.out_vals[arg])
					elif source=='table':
						resolved_args.append(self.table_vals[arg])
				resolved_args.append(self.out_vals[task])
				#LAUNCH THE TASK
				start=time.time()
				self.jit_funcs[task](*resolved_args)
				end=time.time()
# 				print(task, end-start)
		
		
		
	def initialise(self):
		#check everything is set up and makes sense
		#to implement later!!
		#set up outputs
		for var in self.out_vars.keys():
			self.out_vals[var]=np.zeros(self.mpf.shape[0],dtype=self.out_vars[var])

	
	def setfuncs(self,model_class):
		self.functions_class=model_class
		self._inspect()
			
	def _inspect(self):
		colors={'mpf': 'green',
			'derived': 'red',
			'table': 'blue',
			'const': 'white'}
		functions=OrderedDict(inspect.getmembers(self.functions_class,inspect.isfunction))
		singles=self.functions_class.single_derivations
		self.out_vars.update(singles)
		for funcname in singles.keys():
			args=list(inspect.signature(functions[funcname]).parameters.keys())
			self.jit_funcs[funcname]=jit(functions[funcname],nopython=True)
			table='derived'
			self.internal_graph.add_node(funcname,table=table,style='filled',fillcolor=colors[table])
			for i, arg in enumerate(args[:-1]):
				if arg in self.internal_graph.nodes:
					pass
				else:
					table=self.resolve_arg(arg)
					self.internal_graph.add_node(arg,table=table,style='filled',fillcolor=colors[table])
				self.internal_graph.add_edge(arg,funcname,order=i)
		#now go back through the graph and resolve the mapped join keys for the tables
		for tablenode in [n for n,d in self.internal_graph.nodes(data=True) if d['table']=='table']:
			for key in self.table_mapper[tablenode].values():
				if key in self.internal_graph.nodes:
					pass
				else:
					table=self.resolve_arg(arg)
					self.internal_graph.add_node(key,table=table,style='filled',fillcolor=colors[table])
# 				print(tablenode,key)
				self.internal_graph.add_edge(key,tablenode)
		for tablenode in [n for n,d in self.internal_graph.nodes(data=True) if d['table']=='table']:
			for key in self.table_mapper[tablenode].values():
				if key in self.internal_graph.nodes:
					pass
				else:
					table=self.resolve_arg(arg)
					self.internal_graph.add_node(key,table=table,style='filled',fillcolor=colors[table])
# 				print(tablenode,key)
				self.internal_graph.add_edge(key,tablenode)
		for tablenode in [n for n,d in self.internal_graph.nodes(data=True) if d['table']=='table']:
			for key in self.table_mapper[tablenode].values():
				if key in self.internal_graph.nodes:
					pass
				else:
					table=self.resolve_arg(arg)
					self.internal_graph.add_node(key,table=table,style='filled',fillcolor=colors[table])
# 				print(tablenode,key)
				self.internal_graph.add_edge(key,tablenode)
	
	def visualise(self):
		A=nx.nx_agraph.to_agraph(self.internal_graph)
		return A.draw(format='png',prog='dot')
		
		
		