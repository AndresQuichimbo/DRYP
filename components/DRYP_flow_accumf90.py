"""Flow accumulator DRYP
"""
import numpy as np
import components.fortran as floss
from landlab.components.flow_accum import flow_accum_bw#, make_ordered_node_array
from landlab.core.utils import as_id_array
from landlab import RasterModelGrid
from landlab.components import FlowDirectorD8

class runoff_routing(object):
	"""Function to discharge and transmission losses discharge.
	Running run_one_step() results in the following to occur:
		1. Flow directions are updated (unless update_flow_director is set
		as False).
		2. Intermediate steps that analyse the drainage network topology
		and create datastructures for efficient drainage area and discharge
		calculations.
		3. Calculation of discharge and transmission losses.
	"""
	def __init__(self, env_state, data_in):
		#global Kloss
		#if Kloss is None:
		self.Kloss = data_in.Kloss
		# Creates numpy arrays for passing model variables
		#env_state.grid.add_zeros('node', "surface_water__discharge")#, dtype=float32)
		Create_parameter_WV(env_state.grid, data_in.Kloss)
		self.dis_dt = np.zeros(env_state.grid_size)
		self.qfl_dt = np.zeros(env_state.grid_size)
		self.tls_dt = np.zeros(env_state.grid_size)
		self.flow_tls_dt = np.zeros(env_state.grid_size)
		self.carea = None
		
		# 1. Check if flow director is needed
		if env_state.act_update_flow_director:
			fd = FlowDirectorD8(env_state.grid, 'topographic__elevation')
			fd.run_one_step()
		
		# 2. Creates drainage networks, flowpaths and id arrays
		# a value of 1 is added to change from python to forttran
		self.r = as_id_array(env_state.grid["node"]["flow__receiver_node"])
		#nd = as_id_array(flow_accum_bw._make_number_of_donors_array(self.r))
		#delta = as_id_array(flow_accum_bw._make_delta_array(nd))
		#D = as_id_array(flow_accum_bw._make_array_of_donors(self.r, delta))
		self.s = as_id_array(flow_accum_bw.make_ordered_node_array(self.r))
		self.carea = find_drainage_area(self.s, self.r,
					env_state.area_cells,
					env_state.grid.boundary_nodes)		
	
	def run_runoff_one_step(self, inf, swb, aof, env_state, data_in):
		"""Function to make FlowAccumulator calculate drainage area and
		discharge.
		Running run_one_step() results in the following to occur:
			1. Flow directions are updated (unless update_flow_director is set
			as False).
			2. Intermediate steps that analyse the drainage network topology
			and create datastructures for efficient drainage area and discharge
			calculations.
			3. Calculation of drainage area and discharge.
			4. Depression finding and mapping, which updates drainage area and
			discharge.
		"""
		
		# dis_dt:	Discharge [mm]
		# exs_dt:	Infiltration excess [mm]
		# tls_dt	Transmission losses [mm]
		# cth_area:	Cell factor area, it reduces or increases the area
		# Q_ini:	Initial available water in the channel
		# aof:		River flow abstraction [mm]
		env_state.grid.at_node['AOF'][:] = aof
		act_nodes = env_state.act_nodes
		
		env_state.grid.at_node['runoff'][act_nodes] = (
				inf.exs_dt[act_nodes]*0.001*env_state.cth_area[act_nodes]
				+ (env_state.SZgrid.at_node['discharge'][act_nodes])
				)
				
		check_dry_conditions = len(np.where(
				(env_state.grid.at_node['runoff'][act_nodes]
				+ env_state.grid.at_node['Q_ini'][act_nodes]
				) > 0.0)[0])
		
		if check_dry_conditions > 0:
			# this function return
			# discharge, QTL, Q_ini, Q_aof
			#print(env_state.area_cells[act_nodes])
			env_state.grid.at_node['runoff'][env_state.grid.boundary_nodes] = 0
			
			discharge = np.array(env_state.grid.at_node['runoff']
						*env_state.area_cells, np.float32)#Qw			
			QTL = np.array(env_state.grid.at_node['Transmission_losses'], np.float32) #QTL
			Q_ini = np.array(env_state.grid.at_node['Q_ini'], np.float32) #Q_ini
			Qaof = np.array(env_state.grid.at_node['AOF'], np.float32) #Qaof
			#print(env_state.grid.at_node['Transmission_losses'][env_state.grid.at_node['Transmission_losses']<0])
			floss.ftransloss.find_discharge_and_losses(
				self.s+1, self.r+1,
				np.array(env_state.grid.at_node['SS_loss']), #Criv
				np.array(env_state.grid.at_node['decay_flow']), #Kt
				np.array(env_state.grid.at_node['river'], np.int32), #riv
				np.array(env_state.grid.at_node['AOFT']), #Qaoft
				np.array(env_state.grid.at_node['riv_sat_deficit']), #riv_std
				np.array(env_state.grid.at_node['par_3']), #P3
				np.array(env_state.grid.at_node['par_4']), #P4
				self.Kloss,
				discharge, QTL, Q_ini, Qaof)
			
			env_state.grid.at_node["surface_water__discharge"][:] = discharge
			env_state.grid.at_node['Transmission_losses'][:] = QTL
			env_state.grid.at_node['Q_ini'][:] = Q_ini
			env_state.grid.at_node['AOF'][:] = Qaof
			
			#env_state.grid.at_node["surface_water__discharge"][:] = aux[0] #discharge
			#env_state.grid.at_node['Transmission_losses'][:] = aux[1] #QTL
			#env_state.grid.at_node['Q_ini'][:] = aux[2] #Q_ini
			#env_state.grid.at_node['AOF'][:] = aux[3] #aof
			
			self.dis_dt[act_nodes] = np.array(
					env_state.grid.at_node["surface_water__discharge"][act_nodes])
		else:
			env_state.grid.at_node['Transmission_losses'][env_state.river_ids_nodes] = 0.0				
			self.dis_dt[:] = 0				
			noflow = 0
		self.tls_dt = np.array(
				env_state.grid.at_node['Transmission_losses']
				*1000.0/env_state.area_cells)
		self.tls_flow_dt = np.array(env_state.grid.at_node['Transmission_losses'])
		self.qfl_dt[act_nodes] = np.array(
				env_state.grid.at_node['Q_ini'][act_nodes]
				*1000.0/env_state.area_cells[act_nodes])
		#print(self.tls_dt[act_nodes])

def Create_parameter_WV(grid, kKloss):
	"""additional parameters to save computational time
	parameters:
	grid:	landlab grid
	Kloss:	scale and unit-time step factor
	"""
	grid.at_node['par_4'] = 2.0*kKloss/(grid.at_node['decay_flow']*grid.at_node['river_width'])	
	grid.at_node['par_3'] = (grid.at_node['river_width']*grid.at_node['SS_loss']
							/ (grid.at_node['river_width']-2*kKloss))
	return		


def find_drainage_area(s, r, node_cell_area=1.0, boundary_nodes=None):

	"""Calculate the drainage area and water discharge at each node, permitting
	discharge to fall (or gain) as it moves downstream according to some
	function. Note that only transmission creates loss, so water sourced
	locally within a cell is always retained. The loss on each link is recorded
	in the 'surface_water__discharge_loss' link field on the grid; ensure this
	exists before running the function.
	Parameters
	----------
	s : ndarray of int
		Ordered (downstream to upstream) array of node IDs
	r : ndarray of int
		Receiver node IDs for each node
	boundary_nodes: list, optional
		Array of boundary nodes to have discharge and drainage area set to zero.
		Default value is None.
	Returns
	-------
	tuple of ndarray
		drainage area and discharge
	Notes
	-----
	-  If node_cell_area not given, the output drainage area is equivalent
	to the number of nodes/cells draining through each point, including
	the local node itself.
	-  Give node_cell_area as a scalar when using a regular raster grid.
	-  If runoff is not given, the discharge returned will be the same as
	drainage area (i.e., drainage area times unit runoff rate).
	-  If using an unstructured Landlab grid, make sure that the input
	argument for node_cell_area is the cell area at each NODE rather than
	just at each CELL. This means you need to include entries for the
	perimeter nodes too. They can be zeros.
	
	Examples
	--------
	>>> import numpy as np
	>>> from landlab import RasterModelGrid
	>>> from landlab.components.flow_accum import (
	...     find_drainage_area_and_discharge)
	>>> r = np.array([2, 5, 2, 7, 5, 5, 6, 5, 7, 8])-1
	>>> s = np.array([4, 1, 0, 2, 5, 6, 3, 8, 7, 9])
	>>> l = np.ones(10, dtype=int)  # dummy
	>>> nodes_wo_outlet = np.array([0, 1, 2, 3, 5, 6, 7, 8, 9])
	
	"""
	# Number of points
	npoint = len(s)
	
	# Initialize the drainage_area and discharge arrays. Drainage area starts
	# out as the area of the cell in question, then (unless the cell has no
	# donors) grows from there. Discharge starts out as the cell's local runoff
	# rate times the cell's surface area.
	drainage_area = np.zeros(npoint, dtype=int) + node_cell_area
	#discharge = np.zeros(npoint, dtype=int) + node_cell_area
	# note no loss occurs at a node until the water actually moves along a link
	
	# Optionally zero out drainage area and discharge at boundary nodes
	if boundary_nodes is not None:
		drainage_area[boundary_nodes] = 0
	
	# Iterate backward through the list, which means we work from upstream to
	# downstream.
	for i in range(npoint - 1, -1, -1):
		donor = s[i]
		recvr = r[donor]
		if donor != recvr:
			drainage_area[recvr] += drainage_area[donor]
			
	return drainage_area		

