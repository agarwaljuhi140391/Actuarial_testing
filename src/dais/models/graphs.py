import networkx as nx
import inspect
from collections import OrderedDict
from operator import itemgetter

class ModelDependencyGraph(nx.DiGraph):

	def add_node(self,node):
		# print("Graph: Adding node {}".format(node))
		node_details={}
		node_details['funcnode']=node
		node_details['style']='filled'
		if node.mapped:
			node_details['fillcolor']=node.get_color()
		return super().add_node(node.name,**node_details)

	def get_unmapped(self):
		return [arg for arg in self.nodes if self.nodes[arg]['funcnode'].mapped==False]

	def update_tooltips(self):
		for fnode in self.get_fnodes():
			if fnode.is_code_backed():
				if fnode.get_func_string() is None:
					print("no func string, but it was called:",fnode)
				else:
					self.nodes[fnode.name]['tooltip']=fnode.get_func_string().replace('\n', '&#10;')

	def update_color(self,node):
		self.nodes[node]['fillcolor']=self.nodes[node]['funcnode'].get_color()

	def get_fnodes(self):
		return [self.nodes[node]['funcnode'] for node in self.nodes]

	def get_fnode(self,node):
		return self.nodes[node]['funcnode']

	def get_executable_or_derived(self):
		subgraph=self.subgraph([fnode.name for fnode in self.get_fnodes() if fnode.is_executable_or_derived()])

class FunctionNode:
	"""A simple set of information about a node on a task graph for a liability model
		Internal information:
		- name The actual function name. Arguments are represented in the edges of the DAG with which this node is linked.
		- func_string The actual source code of the underlying function. This can take 2 forms:
				1. For element-wise calculations (approach="element") this is the python code that is JIT-ed into a for-loop (default)
				2. For complex interacting calculations (approach="complex") this is the python code including the explicit for-loop that is subsequently JIT-ed
		- approach Takes one of 3 values: None (for tables or source inputs), "element" or "complex". Determines whether the for-loop needs to be automatically added or explicitly included in the source code
		- dim Represent the number of dimensions in the field.
				1. For 1D fields with no time component this value is "1"
				2. For 2D fields with a full monthly time component this value is "2".
		- mapped Boolean indicating whether the node mapping has been resolved. For resolution see source_type
		- source_type Indicates the source nature of the node. Takes one of the available values:
				1. "Unmapped" (default)
				2. "MPF"
				3. "Constant"
				4. "TablePolicy"
				5. "TablePolicyTime"
				6. "Derived"
				7. "DerivedComplex"
				8. "T"
				9. "Summary"
		- keep Boolean indicating whether this field should be kept in memory even if it is no longer required
		- style Dict of information on the style of the nodes when shown graphically
				"color" The file color of the node"""
	
	validvalues={"source_type":set(["Unmapped","MPF","Constant","TablePolicy","TablePolicyTime","Derived","DerivedComplex","T","Summary"])
				, "dim":set([1,2])
				, "keep":set([True,False])
				, "mapped":set([True,False])
	}
	colors={
		'T': 'orange',
		'MPF': 'green',
		'Derived': 'red',
		'DerivedComplex': 'red3',
		'TablePolicy': 'cornflowerblue',
		'TablePolicyTime': 'blue',
		'Constant': 'white',
		'Unmapped':'pink',
		'Summary':'darkseagreen'
		}
	
	def __init__(self,name,source_type='Unmapped'):
		self.attr={}
		if type(name) is str:
			self.name=name
		else:
			print("Name must be a string")
		if source_type in self.validvalues['source_type']:
			self.source_type=source_type
			if source_type in ("TablePolicy","TablePolicyTime"):
				#By default a table is unmapped. If you know it's mapped, set the flag explicitly afterwards. Otherwise get it updated when add the tables in a separate step.
				self.mapped=False
			elif source_type in ('Unmapped'):
				self.mapped=False
			else:
				self.mapped=True
		else:
			print("Invalid source_type: {}".format(source_type))
	def __repr__(self):
		return ('name: {}, source_type: {}, mapped: {}'.format(self.name,self.source_type,self.mapped))
	def set_mapped(self,mapped=True):
		if (mapped in self.validvalues['mapped']) and (self.source_type != "Unmapped"):
			self.mapped=mapped
		else:
			print("Invalid mapped attribute: {}".format(mappped))
	def set_compiled_func(self,compiled_func):
		self.compiled_func=compiled_func
	def get_compiled_func(self):
		assert self.source_type in ('Derived','DerivedComplex','DerivedBond')
		return self.compiled_func.numba_func
	def set_complexparent(self,parentname):
		assert self.source_type in ('DerivedComplex','Summary')
		self.parentname=parentname
	def set_attributes(self,attr):
		self.attr.update(attr)
# 		self.attr=attr
	def set_source_type(self,source_type,update_mapped=True):
		if source_type in self.validvalues['source_type']:
			self.source_type=source_type
			if update_mapped==True:
				self.mapped=True
		else:
			print("Invalid source_type: {}".format(source_type))
	def set_style(self,style):
		self.style=style
	def set_func(self,func):
		self.func=func
	def get_func_string(self):
		try:
			if self.func:
				return inspect.getsource(self.func)
		except:
			pass
	def get_color(self):
		if self.mapped:
			return self.colors[self.source_type]
		else:
			return self.colors['Unmapped']
	def can_have_data(self):
		if (not hasattr(self,'parentname')) and (self.source_type=='DerivedComplex'):
			return False
		else:
			return True
	def is_original_input(self):
		if self.source_type in ['MPF','Constant','T']:
			return True
		else:
			return False
	def is_executable_or_derived(self):
		if self.is_executable() or self.is_complexderived():
			return True
		else:
			return False
	def is_summarysubvar(self):
		if (self.source_type=='Summary') and hasattr(self,'parentname'):
			return True
		else:
			return False
	def is_complexderived(self):
		if (self.source_type =='DerivedComplex') and hasattr(self,'parentname'):
			return True
		else:
			return False
	def is_code_backed(self):
		if self.source_type in ["Derived","DerivedComplex"]:
			if hasattr(self,'parentname') and (self.source_type in ['DerivedComplex','Summary']):
				return False
			else:
				return True
		else:
			return False
	def is_executable(self):
		if (self.source_type in ["TablePolicy","TablePolicyTime","Derived","DerivedComplex","Summary"]):
			if hasattr(self,'parentname') and (self.source_type in ['DerivedComplex','Summary']):
				return False
			else:
				return True
		else:
			return False
			
	def is_loopvar(self):
		if 'loop' not in self.attr.keys():
			return False
		else:
			return True
	

class FunctionNodeBond(FunctionNode):
	validvalues=FunctionNode.validvalues
	validvalues['source_type'].add('DerivedBond')
	validvalues['source_type'].add('TableBond')
	validvalues['source_type'].add('VT')
	colors=FunctionNode.colors
	colors['DerivedBond']='red4'
	colors['TableBond']='red4'
	colors['VT']='darkorange2'
	def is_original_input(self):
		if self.source_type in set(['TableBond','VT']):
			return True
		else:
			return super().is_original_input()
	def is_code_backed(self):
		if self.source_type in ["Derived","DerivedComplex","DerivedBond"]:
			return True
		else:
			return False
	def is_executable(self):
		if (self.source_type in ["TablePolicy","TablePolicyTime","Derived","DerivedComplex","DerivedBond","Summary"]):
			if hasattr(self,'parentname') and (self.source_type in ['DerivedComplex','Summary']):
				return False
			else:
				return True
		else:
			return False
			
class FunctionNodeRollForward(FunctionNode):
	validvalues=FunctionNode.validvalues
	validvalues['source_type'].add('RollForward')
	validvalues['source_type'].add('Collapser')
	validvalues['source_type'].add('rfTime')
	colors=FunctionNode.colors
	colors['RollForward']='darkorchid4'
	colors['Collapser']='darkorchid3'
	colors['rfTime']='darkorchid2'
	
	def is_monthly(self):
		if 'monthly' in self.attr.keys():
			return self.attr['monthly']
		else:
			return False
	
	def is_code_backed(self):
		if self.source_type in ["Collapser","Derived","DerivedComplex","DerivedBond"]:
			return True
		else:
			return False

	def is_executable(self):
		if (self.source_type in ["Collapser","TablePolicy","TablePolicyTime","Derived","DerivedComplex","DerivedBond","Summary"]):
			if hasattr(self,'parentname') and (self.source_type in ['DerivedComplex','Summary']):
				return False
			else:
				return True
		else:
			return False