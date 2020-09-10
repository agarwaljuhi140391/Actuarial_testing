from numba import njit, jit, vectorize, int64, int32, void, prange
import pandas as pd
import numpy as np
import networkx as nx
import inspect
from collections import OrderedDict
import time
from operator import itemgetter
from .simple_dispatcher import func_fac1,func_fac2,func_fac3,func_fac4,func_fac5,func_fac6,func_fac7,fast_searchsorted,fast_take,elementwise_min

class BaseModel:
	"""This is an outline class from which other models can inherit common functionality"""
	def __init__(self):
		self.colors={
			'T': 'orange',
			'mpf': 'green',
			'derived': 'red',
			'table': 'blue',
			'const': 'white'}
		self.func_facs={
			1:func_fac1,
			2:func_fac2,
			3:func_fac3,
			4:func_fac4,
			5:func_fac5,
			6:func_fac6,
			7:func_fac7
		}
		self.internal_graph=nx.DiGraph()
		
		self.timings=OrderedDict()
		
		self.jit_funcs=OrderedDict()
		self.vec_funcs=OrderedDict()
		self.table_funcs=OrderedDict()
		self.func_facs_funcs=OrderedDict()
		
		self.out_vars=OrderedDict()
		self.out_vals=OrderedDict()
		
		self.table_vars=OrderedDict()
		self.table_vals=OrderedDict()
		#table_mapper - a dict of dicts, showing key:field mappings for each table
		self.table_mapper=OrderedDict()

	def _process_mapper(self, tabname, key, mapping):
		if tabname in self.table_mapper:
			self.table_mapper[tabname].update({key:mapping})
		else:
			self.table_mapper[tabname]={key:mapping}
				
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
				
	def configure(self,constants,common_file,be_file):
		self.CONSTANTS=constants
		common_tables=pd.read_excel(common_file,sheet_name=None)
		be_tables=pd.read_excel(be_file,sheet_name=None)
# 		table_mappers=pd.read_excel(mappers,sheet_name='Mappers')
		self.simple_tables=self._process_tables(common_tables,'common')
		self.simple_tables.update(self._process_tables(be_tables,'be'))
		
# 		self._flatter_tables(a,table_mappers)
# 		self._process_mapper(table_mappers)
		
	def load_mpf(self,mpf_file,headers_and_size_only=False,format='text'):
		if headers_and_size_only:
			if format=='text':
				mpf=pd.read_table(mpf_file,header=0,nrows=1)
				self.mpf_fields=mpf.columns
		else:
			if format=='text':
				self.mpf=pd.read_table(mpf_file,header=0)
			elif format=='parquet':
				self.mpf=pd.read_parquet(mpf_file)
			self.mpf_fields=self.mpf.columns
				
			
	def resolve_arg(self,arg):
		if arg=='T':
			return 'T'
		elif arg in self.mpf_fields:
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

	def make_blank_val(self,arg):
		if self.out_vars[arg]['shape']==1:
			return np.zeros((self.mpf.shape[0],1),dtype=self.out_vars[arg]['type'])
		elif self.out_vars[arg]['shape']==2:
			return np.zeros((self.mpf.shape[0],self.CONSTANTS['t_range']),dtype=self.out_vars[arg]['type'])

	
	def get_val(self,arg):
		loc=self.resolve_arg(arg)
		if loc=='T':
			return self.T
		elif loc=='mpf':
			return self.mpf[arg].to_numpy().reshape((self.mpf.shape[0],1))
		elif loc=='const':
			return self.CONSTANTS[arg]
		elif loc=='derived':
			if arg in self.out_vals:
				return self.out_vals[arg]
			else:
				return self.make_blank_val(arg)
		elif loc=='table':
			return self.table_vals[arg]	
	
	def execute_table(self,task):
# 		mapping=self.table_vars[task]['keys']
		#prepare source
		print("Running a table")

		source_dict=OrderedDict()
		for key in self.table_vars[task]['mapping'].keys():
			mapped=self.table_vars[task]['mapping'][key]
			source_dict[key]=self.get_val(mapped)
			if type(source_dict[key]) in [int,float]:
				pass
			else:
				 source_dict[key]=source_dict[key].ravel()
		#check the lengths are all valid
		lengths={}
		valid_lengths=[1,self.mpf.shape[0],self.mpf.shape[0]*self.T.shape[1]]
		for key in source_dict.keys():
			if type(source_dict[key]) in [int,float]:
				lengths[key]=1
			else:
				length=source_dict[key].shape[0]
				if length in valid_lengths:
					lengths[key]=length
				else:
					print('FAIL: INVALID LENGTH,',key,length)
# 		print('lengths',lengths)
		#If any of the lengths are the last type then we have a dynamic lookup 
		if max([l for l in lengths.values()]) == valid_lengths[-1]:
			start=time.time()
			col=source_dict.pop('col')
			source_rows=pd.DataFrame(source_dict)
			row_indices=pd.merge(source_rows
								,self.table_vars[task]['table']['index']
								,how='left'
								,left_on=list(source_dict.keys())
								,right_index=True)['index'].to_numpy().reshape((self.mpf.shape[0],1))
			colnames=[int(c) for c in self.table_vars[task]['table'].columns if c not in list(source_dict.keys())+['index']+['Description']]
			if 'autocapcol' in self.table_vars[task]:
				if self.table_vars[task]['autocapcol']:
					col2=np.zeros(col.shape,dtype=col.dtype)
					elementwise_min(col,max(colnames),col2)
				else:
					col2=col
			else:
				col2=col
			col_indices=fast_searchsorted(np.asarray(colnames),col2.ravel())
			row_indices2,_temp=np.broadcast_arrays(row_indices,self.T)
			del _temp
			row_indices3=row_indices2.ravel()
			row_indices3.flags.writeable=False
# 			if task=='COMM_CB_RATE':
# 				print(self.table_vars[task]['table'][colnames].to_numpy())
# 				print(row_indices3,col_indices)
			self.table_vals[task]=fast_take(self.table_vars[task]['table'][colnames].to_numpy(),row_indices3,col_indices).reshape((self.mpf.shape[0],self.T.shape[1]))
		else:
			start=time.time()
# 					print(source_dict,self.table_vars[task])
			if max([l for l in lengths.values()])==1:
				for key in source_dict.keys():
					temp=np.zeros(self.mpf.shape[0])
					temp[:]=source_dict[key]
					source_dict[key]=temp
			self.table_vals[task]=pd.merge(pd.DataFrame(source_dict)
											,self.table_vars[task]['table']
											,how='left'
											,left_on=list(source_dict.keys())
											,right_index=True)[task].to_numpy().reshape((self.mpf.shape[0],1))
		end=time.time()
		print(task, end-start)

	def execute_derived(self,task):
		args=self.internal_graph.in_edges(nbunch=task,data=True)
		args={arg:attr['order'] for arg,func,attr in args}
# 		print(args)
		if self.internal_graph.nodes[task].get('complexitem'):
			print("Executing complex task:",task)
			args2=self.internal_graph.out_edges(nbunch=task,data=True)
			args.update({func:attr['order'] for arg,func,attr in args2})
		elif self.jit_funcs[task]['numba']=='j':
			args.update({task: max(args.values())+1})
		print(args)
		args2=[arg for arg,order in sorted(args.items(), key=itemgetter(1))]
		resolved_args=[]
		for arg in args:
			if (self.resolve_arg(arg) == 'derived') and (arg not in self.out_vals):
				self.out_vals[arg]=self.get_val(arg)
				_temp=self.out_vals[arg]
			else:
				_temp=self.get_val(arg)
			resolved_args.append(_temp)
		if self.jit_funcs[task]['numba']=='v':
			resolved_args=np.broadcast_arrays(*resolved_args)
			for arg in resolved_args:
				arg.flags.writeable=False
		#LAUNCH THE TASK
		start=time.time()
		if self.jit_funcs[task]['numba']=='v':
			#THIS SEEMS TO ASSUME SOMETHING ABOUT THE SHAPE OF THE OUTPUT. IS IT MUCKING SOMETHING UP SOMEWHERE?!
# 			self.out_vals[task][:,:]=self.jit_funcs[task]['func'](*resolved_args)
			if len(resolved_args)<=7:
				if task in self.func_facs_funcs:
					pass
				else:
					self.func_facs_funcs[task]=self.func_facs[len(resolved_args)](self.jit_funcs[task]['func'],self.out_vars[task]['type'],*resolved_args)
				print("Running a func_fac")
				self.out_vals[task]=self.func_facs_funcs[task](*resolved_args)
			else:
				print("Running a vec_func")
				self.out_vals[task]=self.vec_funcs[task]['func'](*resolved_args)
		elif self.jit_funcs[task]['numba']=='j':
# 			resolved_args.append(self.out_vals[task])
# 			for arg in resolved_args:
# 				if arg not in self.out_vals:
# 					self.out_vals[arg]=res
			self.jit_funcs[task]['func'](*resolved_args)
		end=time.time()
		print(task, end-start)		
	
	def execute(self,targets=None,dokeep=None,clear=True):
		if clear:
			self.out_vals=OrderedDict()
			self.table_vals=OrderedDict()
		if targets is None:
			tasklist = [task for task in nx.topological_sort(self.internal_graph)
				if self.internal_graph.nodes[task]['table'] in ['derived','table']]
		else:
			all_ancestors=set(targets)
			for target in targets:
				all_ancestors.update(nx.ancestors(self.internal_graph,target))
			tasklist = [task for task in nx.topological_sort(self.internal_graph)
				if (self.internal_graph.nodes[task]['table'] in ['derived','table']) and (task in all_ancestors)]
		print(tasklist)
		if dokeep is None:
			self.dokeep=set([])
		elif dokeep=="ALL":
			self.dokeep=set(tasklist)
		else:
			self.dokeep=set(dokeep)
		
		done_tasks=[]
		for task in tasklist:
			print(task)
			nolog=False
			logging_start=time.time()
			if self.internal_graph.nodes[task]['table'] == 'table':
				self.execute_table(task)
			elif self.internal_graph.nodes[task]['table'] == 'derived':
				if 'complexinternal' in self.internal_graph.nodes[task].keys():
					if self.internal_graph.nodes[task]['complexinternal']:
						#do nothing
						nolog=True
					else:
						self.execute_derived(task)
				else:
					self.execute_derived(task)
			logging_end=time.time()
			if not nolog:
				self.timings[task]=logging_end-logging_start
			done_tasks.append(task)
			tasks_to_remove=self.tasks_to_remove(task,done_tasks)
			for t in tasks_to_remove:
				if self.resolve_arg(t)=='table' and t in self.table_vals:
					del self.table_vals[t]
				elif self.resolve_arg(t)=='derived' and t in self.out_vals:
					del self.out_vals[t]
			
	def tasks_to_remove(self,task,done_tasks):
		tasks=[]
		possibles=[ins for ins,task in self.internal_graph.in_edges(task)]
		eligibles=[ins for ins in possibles if self.resolve_arg(ins) in set(['table','derived'])]
		for arg in eligibles:
			remaining_deps=[outs for task,outs in self.internal_graph.out_edges(arg) if outs not in done_tasks]
			if len(remaining_deps)==0 and arg not in self.dokeep:
				print("removing",arg)
				tasks.append(arg)
		return tasks
						
	def initialise(self):
		#check everything is set up and makes sense
		#to implement later!!
		#switch out the mpf with only the first row
		#set up time with 1 row
		#execute for the single row. this initialises all the JITs.
		#then switch back the full size mpf
		#set up time for all rows
		#DO WE NEED TO SET UP THE OUTPUTS? CAN WE MOVE THIS ALL TO THE EXECUTE PHASE? WE CAN THEN MORE GRACEFULLY MOVE TO CLEARING MEMORY
		mpf_backup=self.mpf
		self.mpf=self.mpf.head(2)
		self.T=np.zeros((self.mpf.shape[0],self.CONSTANTS['t_range']),dtype=np.int32)
		self.T[:,:]=np.arange(1,self.CONSTANTS['t_range']+1,dtype=np.int32)
		print("Executing with a single row")
		self.execute()
		print("Done. Restoring full MPF")
		for arg in list(self.out_vals.keys()):
			del self.out_vals[arg]
		for arg in list(self.table_vals.keys()):
			del self.table_vals[arg]
		self.mpf=mpf_backup
		self.T=np.zeros((self.mpf.shape[0],self.CONSTANTS['t_range']),dtype=np.int32)
		self.T[:,:]=np.arange(1,self.CONSTANTS['t_range']+1,dtype=np.int32)
		


# 		for var in self.out_vars.keys():
# 			if self.out_vars[var]['shape']==1:
# 				self.out_vals[var]=np.zeros((self.mpf.shape[0],1),dtype=self.out_vars[var]['type'])
# 			elif self.out_vars[var]['shape']==2:
# 				self.out_vals[var]=np.zeros((self.mpf.shape[0],self.CONSTANTS['t_range']),dtype=self.out_vars[var]['type'])
	
	def setfuncs(self,model_class):
		self.functions_class=model_class
		#process tables
		self._flatter_tables()
		self._inspect()
		
	def _flatter_tables(self,specials=['Description','DESCRIPTION']):
		#SOMETHING IFFY GOING ON WRT KEY ORDER ON THE TABLES WHICH I DON'T QUITE UNDERSTAND YET.
		flatter_tables={}
		table_mappers=self.functions_class.mappings
		#We should iterate on the items listed out in the model class, not the full set in the tables
		for tabname in table_mappers.keys():
			if 'source' in table_mappers[tabname].keys():
				#if source exists then it's named_var
				keys=list(table_mappers[tabname]['mapping'].keys())
# 				print(tabname,self.simple_tables[table_mappers[tabname]['source']]['table'],keys)
				flatter_tables[tabname]={'table': self.simple_tables[table_mappers[tabname]['source']]['table'][keys+[tabname]].set_index(keys)
										, 'mapping': table_mappers[tabname]['mapping']
										, 'type': 'named_var'
				}
			else:
				keys=[c for c in table_mappers[tabname]['mapping'].keys() if c not in ['col']]
				print(tabname)
				if 'autocapcol' in table_mappers[tabname].keys():
					autocapcol=True
# 					print(tabname, 'autocapcol')
				else:
					autocapcol=False
				flatter_tables[tabname]={'table': self.simple_tables[tabname]['table'][sorted([c for c in self.simple_tables[tabname]['table'].columns if c not in specials+keys])+keys].reset_index().set_index(keys)
										 , 'mapping': table_mappers[tabname]['mapping']
										 , 'type': 'dynamic'
										 , 'autocapcol': autocapcol}
		self.table_vars.update(flatter_tables)

	def add_tool_tips(self):
		functions=OrderedDict(inspect.getmembers(self.functions_class,inspect.isfunction))
		for node in self.internal_graph.nodes:
			if node in functions:
				self.internal_graph.nodes[node]['tooltip']=inspect.getsource(functions[node])
				
	def _inspect(self):
		functions=OrderedDict(inspect.getmembers(self.functions_class,inspect.isfunction))
		singles=self.functions_class.single_derivations
		self.out_vars.update(singles)
		
		complexes=self.functions_class.complex_derivations
		for complex_item in complexes.keys():
			for outvar in complexes[complex_item]['outvars'].keys():
				self.out_vars[outvar]=complexes[complex_item]['outvars'][outvar]
				self.out_vars[outvar]['parent']=complex_item
				
		for funcname in singles.keys():
			print("Inspecting:",funcname)
			if singles[funcname]['numba']=='v':
				self.vec_funcs[funcname]={'func':vectorize(functions[funcname],cache=True),'numba':'v'}
				self.jit_funcs[funcname]={'func':njit(functions[funcname],nogil=True,cache=True),'numba':'v'}
				args=list(inspect.signature(functions[funcname]).parameters.keys())
# 				print(args)
			elif singles[funcname]['numba']=='j':
				self.jit_funcs[funcname]={'func':njit(functions[funcname],parallel=True,nogil=True,cache=True),'numba':'j'}
				args=[arg for arg in inspect.signature(functions[funcname]).parameters.keys() if arg not in [funcname]]
			table='derived'
			self.internal_graph.add_node(funcname,style='filled'
				,table=table,fillcolor=self.colors[table]
				,func_code=inspect.getsource(functions[funcname]))
			for i, arg in enumerate(args):
				if arg in self.internal_graph.nodes:
					pass
				else:
					table=self.resolve_arg(arg)
					self.internal_graph.add_node(arg,table=table,style='filled',fillcolor=self.colors[table])
				self.internal_graph.add_edge(arg,funcname,order=i)
				
		for complex_item in complexes.keys():
			print("Inspecting complex:",complex_item)
			self.jit_funcs[complex_item]={'func':njit(functions[complex_item],parallel=True,nogil=True,cache=True),'numba':'j'}
			args=list(inspect.signature(functions[complex_item]).parameters.keys())
			self.internal_graph.add_node(complex_item,complexitem=True,style='filled',table='derived',fillcolor=self.colors['derived'])
			for outvar in complexes[complex_item]['outvars'].keys():
				#The complexinternal flag is to help the executor know to not try and execute this one, only the parent complex_item
				self.internal_graph.add_node(outvar,complexinternal=True,style='filled',table='derived',fillcolor=self.colors['derived'])
				self.internal_graph.add_edge(complex_item,outvar,complexinternal=True)
			for i, arg in enumerate(args):
				if arg in self.internal_graph.nodes:
					nx.set_edge_attributes(self.internal_graph,{(complex_item,arg):{'order':i}})
				else:
					table=self.resolve_arg(arg)
					self.internal_graph.add_node(arg,table=table,style='filled',fillcolor=self.colors[table])
				if arg not in complexes[complex_item]['outvars'].keys():
					self.internal_graph.add_edge(arg,complex_item,order=i)
				
		
		#now go back through the graph and resolve the mapped join keys for the tables
		tables_to_fix = [n for n,d in self.internal_graph.nodes(data=True) if (d['table']=='table') and (len(list(self.internal_graph.predecessors(n)))==0)]
		while(len(tables_to_fix)>0):
			self._resolve_table(tables_to_fix)
			tables_to_fix = [n for n,d in self.internal_graph.nodes(data=True) if (d['table']=='table') and (len(list(self.internal_graph.predecessors(n)))==0)]
		
	def _resolve_table(self,tables_to_fix):
		print('Fixing tables: ', tables_to_fix)
		functions=OrderedDict(inspect.getmembers(self.functions_class,inspect.isfunction))
		for tablenode in tables_to_fix:
# 			self.table_funcs[tablenode]=functions[tablenode]
			join_info=self.functions_class.mappings[tablenode]
			for key in join_info['mapping'].values():
				if key in self.internal_graph.nodes:
					pass
				else:
					table=self.resolve_arg(key)
					self.internal_graph.add_node(key,table=table,style='filled',fillcolor=self.colors[table])
# 				print(tablenode,key)
				self.internal_graph.add_edge(key,tablenode)

	
	def visualise(self,target_node=None,format='svg'):
		#put something in to reduce scope of visualisation
		if target_node is None:
			A=nx.nx_agraph.to_agraph(self.internal_graph)
		else:	
			sub_graph_nodes=list(nx.all_neighbors(self.internal_graph,target_node))+[target_node]
			sub_graph=nx.subgraph(self.internal_graph,sub_graph_nodes)
			A=nx.nx_agraph.to_agraph(sub_graph)
		if format=='svg':
			return A.draw(format='svg',prog='dot')
		else:
			return A.draw(format='png',prog='dot')
		
		