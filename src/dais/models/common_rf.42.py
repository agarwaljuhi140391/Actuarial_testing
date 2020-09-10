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
import itertools
from .graphs import ModelDependencyGraph, FunctionNode, FunctionNodeBond, FunctionNodeRollForward
from .simple_dispatcher import tabledyn_helper_minmax,fast_searchsorted,fast_take,elementwise_min
from .complex_dispatcher import CompiledFunction,BondCompiledFunction,GPUCompiledFunction
from .complex_dispatcher_rf import CompiledFunctionRF
import importlib
from .common3 import BaseModel
from pathlib import Path
import os
import dais

class RollForwardModel(BaseModel):

	def __init__(self,time_periods=100,val_yr=2019):
		super().__init__(time_periods)
		self.val_yr=val_yr
		self.alignment_sizes={}
		self.node_type=CompiledFunctionRF
		self.graphnode_type=FunctionNodeRollForward

	def add_source_mpf(self,dict_dfs):
		self.alignment_mpfs=dict_dfs
		for alignment in self.alignments.keys():
			self.alignment_sizes[alignment]=len(self.alignment_mpfs[self.alignments[alignment]])
		unmapped=self.internal_graph.get_unmapped()
		print("Processing MPF. {} unmapped variables to start with.".format(len(unmapped)))
		mapped=[]
		indexes={node:i for i,node in enumerate(self.alignments.keys())}
		reverse={self.alignments[node]:node for node in self.alignments.keys()}
		# print(indexes,reverse)
		for node in unmapped:
			# print("unmapped node: {}".format(node))
			if node in self.alignment_mpfs.keys():
				# print("Mapping in {}".format(node))
				fnode=self.internal_graph.nodes[node]['funcnode']
				fnode.set_source_type('MPF')
				fnode.set_attributes({'type':self.alignment_mpfs[node].to_numpy().dtype,'shape':1,'alignment':reverse[node]})
				mapped.append(node)
				self.internal_graph.update_color(node)
				# self.vals[node]=np.zeros(tuple(self.alignment_sizes.values()+[self.time_periods]),dtype=np.int32)
				shape=[self.alignment_sizes[align] if align==reverse[node] else 1 for align in self.alignments.keys()]+[1]
				self.vals[node]=self.alignment_mpfs[node].values.reshape(tuple(shape))
				shape[-1]=self.time_periods
				self.vals[node]=np.broadcast_to(self.vals[node],tuple(shape))
		print("Processed MPF. {} new variables mapped.".format(len(mapped)))

	def _inspect(self,debug=False):
		self._inspect_alignment()
		super()._inspect(debug=debug)
		self._inspect_collapsers()
		self._inspect_rollfowards()
		special_nodes=['rfMONTH','rfYEAR']
		self.internal_graph.add_node(self.graphnode_type('T'))
		for node in special_nodes:
			if debug:
				print("Adding special nodes", node)
			fnode=self.graphnode_type(node,'rfTime')
			self.internal_graph.add_node(fnode)
			self.internal_graph.add_edge('T',node)
		
	
	def _inspect_alignment(self,debug=True):
		self.alignments=self.functions_class.alignments
		for alignment in self.alignments.keys():
			if debug:
				print("Adding alignment", alignment)
			node=FunctionNode(self.alignments[alignment])
			node.set_attributes({'alignment_source':alignment})
			self.internal_graph.add_node(node)
		#special values - rfMONTH,rfYEAR
	
	def _inspect_rollfowards(self,debug=True):
		self.rollforwards=self.functions_class.rollforwards
		for rf in self.rollforwards.keys():
			if debug:
				print("Adding rollforward", rf)
			node=self.graphnode_type(rf,'RollForward')
			node.set_attributes(self.rollforwards[rf])
			node.set_attributes({'type':'float64'})
			self.internal_graph.add_node(node)
			self.internal_graph.add_edge(self.rollforwards[rf]['initial'],rf)
			# node_to_update = self.internal_graph.get_fnode(self.rollforwards[rf]['loop_var'])
			# node.set_attributes({'rollforwards':rf})
			# node.set_attributes(self.rollforwards[rf])
		###THIS IS NOT YET COMPLETE###
		
	def _inspect_collapsers(self,debug=True):
		self.collapsers=self.functions_class.collapsers
		for collapser in self.collapsers.keys():
			if debug:
				print("Adding collapser", collapser)
			node=self.graphnode_type(collapser,'Collapser')
			node.set_mapped()
			# print(collapser,self.internal_graph.get_fnode(self.collapsers[collapser]['source']),self.internal_graph.get_fnode(self.collapsers[collapser]['source']).attr)
			node.set_attributes({'type':'float64'})
			node.set_attributes(self.collapsers[collapser])
			self.internal_graph.add_node(node)
			self.internal_graph.add_edge(self.collapsers[collapser]['source'],collapser,order=0)
	
	def _get_alignment_sources(self):
		alignment_sources=[]
		for node in self.internal_graph.get_fnodes():
			if hasattr(node,'attr'):
				if 'alignment_source' in node.attr:
					alignment_sources.append(node)
		return alignment_sources

	def setfuncs(self,model_class,debug=False):
		super().setfuncs(model_class,debug=False)
		if 'T' in self.internal_graph.nodes:
			self.internal_graph.get_fnode('T').set_attributes({'alignment':''})
				
	def initialise_1dtable(self,tablenode):
		print("1dtable:",tablenode)
		fnode=self.internal_graph.get_fnode(tablenode)
		node_alignment=fnode.attr['alignment']
		indexes=[self.alignments[align] for align in node_alignment]
		table=self.tables[fnode.attr['source']][indexes+[tablenode]].set_index(indexes)[[tablenode]].sort_index()
		table=self.alignment_sources[node_alignment].join(table).fillna(0)
		shape=[]
		for align in self.alignments:
			if align in node_alignment:
				shape.append(self.alignment_sizes[align])
			else:
				shape.append(1)
		shape.append(-1)
		self.vals[tablenode] = table.values.reshape(tuple(shape))


	def initialise_2dtable(self,tablenode):
		print("2dtable:",tablenode)
		node_alignment=self.internal_graph.get_fnode(tablenode).attr['alignment']
		indexes=[self.alignments[align] for align in node_alignment]
		table=self.tables[tablenode].set_index(indexes)[[col for col in self.tables[tablenode].columns if isinstance(col,int)]].sort_index()
		table=self.alignment_sources[node_alignment].join(table).fillna(0)
		shape=[]
		for align in self.alignments:
			if align in node_alignment:
				shape.append(self.alignment_sizes[align])
			else:
				shape.append(1)
		shape.append(-1)
		self.vals[tablenode] = table.values.reshape(tuple(shape))
	
	def initialise(self,targets=None):
		#initialise the tables and alignments
		subnodes=[]
		for fnode in self.internal_graph.get_fnodes():
			if 'loop' not in fnode.attr.keys():
				subnodes.append(fnode.name)
		if targets:
			tasklist=self.create_tasklist(targets)
		else:
			tasklist=self.create_tasklist(subnodes)
		overlap=set(tasklist).intersection(set(self.internal_graph.get_unmapped()))
		assert len(overlap)==0
		print(tasklist)
		specials=set(['T','rfMONTH','rfYEAR','POOL','CATEGORY','PURCHASE_YEAR_CAPPED'])

		#PREP ALL THE ALIGNMENT HELPERS
		self.alignment_sources={}
		for i in range(1,len(self.alignments.keys())+1):
			combos=itertools.combinations(self.alignments.keys(),i)
			if i==1:
				for combo in combos:
					align=list(combo)[0]
					self.alignment_sources[align]=self.alignment_mpfs[self.alignments[align]].set_index(self.alignments[align])
			else:
				for combo in combos:
					listcombo=list(combo)
# 					print(listcombo)
					self.alignment_sources[''.join(listcombo)]=pd.DataFrame(
						index=pd.MultiIndex.from_product([self.alignment_mpfs[self.alignments[val]].values.flatten() for val in listcombo]
												,names=[self.alignments[name] for name in listcombo]))
		#PREP ALL THE INPUT TABLES BEFORE 
		for task in tasklist:
			if task not in specials:
				fnode=self.internal_graph.get_fnode(task)
				if fnode.source_type=='TablePolicy':
					self.initialise_1dtable(task)
				elif fnode.source_type=='TablePolicyTime':
					self.initialise_2dtable(task)
				else:
					print("Initialising something else:",task)
			elif task == 'T':
				self.vals[task]=np.arange(self.time_periods).reshape(tuple([1 for align in self.alignments.keys()]+[self.time_periods]))
		#COMPILE
		self.compile_or_load()
		if targets is None:
			self.initialise_loop(1)

# 		for node in tasklist:
# 			fnode=self.internal_graph.get_fnode(node)
# 			if fnode in alignments:
# 				pass
# 			elif fnode.is_original_input():
# 				if fnode.source_type=='Constant':
# 					self.vals[node],_=np.broadcast_arrays(np.asarray([[self.constants[node]]]).reshape((1,1))
# 															,self.mpf[self.mpf.columns[0]].to_numpy().reshape((-1,1)))
# 					self.vals[node]=self.vals[node].copy()
# 				fnode.attr['type']=self.vals[node].dtype
# 		self.compile_or_load(targets)

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
				fnode.set_compiled_func(self.node_type(fnode,ordered_argnodes,self.functions_class.__name__ + type(self).__name__
					, alignments=self.alignments,debug=self.debug))

	def initialise_loop(self,loop):
		def loop_initialise_rf(self,rfnode):
			fnode=self.internal_graph.get_fnode(rfnode)
			initial_fnode=self.internal_graph.get_fnode(fnode.attr['initial'])
			init_val=self.vals[initial_fnode.name]
			t_end=1
			monthly=1
			if initial_fnode.is_monthly():
				monthly=12
				t_end += monthly-1
			self.vals[rfnode]=np.zeros(init_val.shape[:-1]+(self.time_periods*monthly,),dtype=init_val.dtype)
			self.vals[rfnode][:,:,:,0:0+t_end]=init_val[:,:,:,0:t_end]

		def loop_initialise_nonrf(self,nonrfnode):
			fnode=self.internal_graph.get_fnode(nonrfnode)
			fnode_shape=[]
			for align in self.alignments.keys():
				fnode_shape.append(self.alignment_sizes[align] if align in fnode.attr['alignment'] else 1)
			fnode_shape.append(self.time_periods*(12 if fnode.is_monthly() else 1))
			self.vals[nonrfnode]=np.zeros(tuple(fnode_shape),dtype=fnode.attr['type'])   

		nodes_in_loop=[fnode for fnode in self.internal_graph.get_fnodes() if fnode.attr.get('loop')==loop]
		rf_nodes=[fnode for fnode in nodes_in_loop if fnode.source_type=='RollForward']
		for fnode in rf_nodes:
			if self.debug:
				print('initialising rf',fnode)
			loop_initialise_rf(self,fnode.name)
		nonrf_nodes=[fnode for fnode in nodes_in_loop if fnode not in rf_nodes]
		nonrf_nodenames=[fnode.name for fnode in nonrf_nodes]
		for fnode in nonrf_nodes:
			if self.debug:
				print('initialising nonrf',fnode)
			loop_initialise_nonrf(self,fnode.name)

	def prep_loop(self,loop):
		template="""
from numba import njit, void, prange
import numpy as np
import math
from numpy import float64, int32, int64
{imports}

# @njit(parallel=False,cache=True)
def loop_{loop}({all_args}):
	for t in range({t_range}):
		t_end=t+1
		cashflow_t_start=t*12
		cashflow_t_end=(t_end)*12
		next_cashflow_t_start=(t+1)*12
		next_cashflow_t_end=(t_end+1)*12
		prev_t=t-1
		prev_t_end=t_end-1
		prev_cashflow_t_start=max(0,(t-1)*12)
		prev_cashflow_t_end=max(0,(t_end-1)*12)
		
		if t>0:
{loop_end_function_calls}
{loop_start_function_calls}
{loop_rf_function_calls}

{function_calls}

		"""
		rfvar_template="""
			{outvar}{out_indexes}={invar}{in_indexes}
"""
		classname=self.functions_class.__name__ + type(self).__name__
# 		nodes_in_loop=[fnode for fnode in self.internal_graph.get_fnodes() if (fnode.attr.get('loop')==loop) and (fnode.source_type != 'RollForward')]
		nodes_in_loop=[self.internal_graph.get_fnode(node) for node in self.create_tasklist()  
							if (self.internal_graph.get_fnode(node).attr.get('loop')==loop)]
		imports=["from .{node} import wrapped_{node}".format(node=fnode.name) for fnode in nodes_in_loop if fnode.source_type != 'RollForward']
		start_nodes=[fnode for fnode in nodes_in_loop if fnode.attr.get('loop_location')=='start' and fnode.source_type != 'RollForward']
		end_nodes=[fnode for fnode in nodes_in_loop if fnode.attr.get('loop_location')=='end']
		rf_nodes=[fnode for fnode in nodes_in_loop if fnode.source_type=='RollForward']
		nodes_in_loop=[fnode for fnode in nodes_in_loop if fnode not in (start_nodes+end_nodes+rf_nodes)]
		function_calls=[]
		loop_start_function_calls=[]
		loop_end_function_calls=[]
		loop_rf_function_calls=[]
		all_args={}
		for fnode in rf_nodes:
			indexes=[':' for align in self.alignments.keys()]
			out_indexes=indexes+['cashflow_t_start:cashflow_t_end' if fnode.is_monthly() else 't:t_end']
			in_indexes=indexes+['prev_cashflow_t_start:prev_cashflow_t_end' if fnode.is_monthly() else 'prev_t:prev_t_end']
# 			in_indexes=out_indexes		
			loop_rf_function_calls.append(rfvar_template.format(**{'outvar':fnode.name
																,'invar':fnode.attr.get('loop_var')
																,'out_indexes':"["+",".join(out_indexes)+"]"
																,'in_indexes':"["+",".join(in_indexes)+"]"}))
			all_args.update({fnode.name:''})
			all_args.update({fnode.attr.get('loop_var'):''})
		for fnode in start_nodes+end_nodes:
			edges=self.internal_graph.in_edges(nbunch=fnode.name,data=True)
			argnodes=[self.internal_graph.get_fnode(argnode) for argnode,_task,_data in edges]
			edge_order={arg:data['order'] for arg,_task,data in edges}
			ordered_argnodes=sorted(argnodes,key=lambda x: edge_order[x.name])
			args=[]
			out_indexes=[':' for align in self.alignments]
			out_indexes.append('prev_cashflow_t_start:cashflow_t_end' if fnode.is_monthly() else 'prev_t:t_end')
			for argnode in argnodes:
				argindexes = [':' for align in self.alignments.keys()]
				if argnode.attr.get('shape')==1:
					argindexes.append('0:1')
				elif argnode.attr.get('monthly'):
					argindexes.append('prev_cashflow_t_start:cashflow_t_end')
				else:
					argindexes.append('prev_t:t_end')
				args.append(argnode.name+'['+','.join(argindexes)+']')
			args.append(fnode.name+'['+','.join(out_indexes)+']')
			if fnode in end_nodes:
				loop_end_function_calls.append("\t\t\twrapped_{node}({args})".format(node=fnode.name,args=",".join(args)))
			elif fnode in start_nodes:
				loop_start_function_calls.append("\t\t\twrapped_{node}({args})".format(node=fnode.name,args=",".join(args)))
			all_args.update({fnode.name:''})
			all_args.update({arg_fnode.name:'' for arg_fnode in argnodes})
			
		
		for fnode in nodes_in_loop:
			edges=self.internal_graph.in_edges(nbunch=fnode.name,data=True)
			# print(fnode,edges)
			argnodes=[self.internal_graph.get_fnode(argnode) for argnode,_task,_data in edges]
			edge_order={arg:data['order'] for arg,_task,data in edges}
			ordered_argnodes=sorted(argnodes,key=lambda x: edge_order[x.name])
			args=[]
# 			out_indexes=[align.lower() if align in fnode.attr['alignment'] else '0' for align in self.alignments]
			out_indexes=[':' for align in self.alignments]
			out_indexes.append('cashflow_t_start:cashflow_t_end' if fnode.is_monthly() else 't:t_end')
			for argnode in argnodes:
# 				argindexes = [align.lower() if (align in argnode.attr['alignment']) else '0' for align in self.alignments.keys()]
				argindexes = [':' for align in self.alignments.keys()]
				if argnode.attr.get('shape')==1:
					argindexes.append('0:1')
				elif argnode.attr.get('monthly'):
					argindexes.append('cashflow_t_start:cashflow_t_end')
				else:
					argindexes.append('t:t_end')
				args.append(argnode.name+'['+','.join(argindexes)+']')
			args.append(fnode.name+'['+','.join(out_indexes)+']')
			function_calls.append("\t\twrapped_{node}({args})".format(node=fnode.name,args=",".join(args)))
			all_args.update({fnode.name:''})
			all_args.update({arg_fnode.name:'' for arg_fnode in argnodes})
		filetext=template.format(**{'loop':loop
							,'all_args':",".join(all_args.keys())
							,'imports':"\n".join(imports)
							,'loop_start_function_calls':"\n".join(loop_start_function_calls)
							,'loop_end_function_calls':"\n".join(loop_end_function_calls)
							,'loop_rf_function_calls':"\n".join(loop_rf_function_calls)
# 							,'t_range':2
							,'t_range':self.time_periods
							,'function_calls':"\n".join(function_calls)
		}).encode('UTF-8')
		path=os.path.normpath(os.path.join(inspect.getfile(dais),'..','__dais_cache__',classname))
# 			print('FuncNode: ',funcnode,path)
		Path(path).mkdir(parents=True,exist_ok=True)
		self.cachepath = path
		self.modulepath='dais.__dais_cache__.{}.{}'.format(classname,'loop_{}'.format(loop))
		try:
			with open(os.path.join(self.cachepath,'loop_{}.py'.format(loop)),'rb') as f:
				cachecontent=f.read()
		except (NameError,FileNotFoundError) as e:
			cachecontent = b''
# 		return {"CacheContent": cachecontent,"FileText": filetext}
		if cachecontent==filetext:
			pass
		else:
			with open(os.path.join(self.cachepath,'loop_{}.py'.format(loop)),'wb') as f:
				f.write(filetext)
		mod=importlib.import_module(self.modulepath)
		importlib.reload(mod)
		self.loop_executor=getattr(mod,'loop_{}'.format(loop))
		self.loop_executor_argorder=all_args


	#We need to add the variables specific in the alignments attribute
	#We should parse these on model import, in the setfuncs chain
	#We should expect the constants of POOL, CATEGORY and PURCHASE YEAR to be set and when 
		#when we see those constants we should treat them specially, holding them as alignments dict values
		#initialise should then use these to set the dimensionality of these to set tensor shapes
		#table imports should ensure we have the right number of levels as per the alignment variables and the values of the alignment constants
		
