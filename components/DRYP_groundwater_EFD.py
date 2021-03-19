
import os
import numpy as np
from landlab import RasterModelGrid
from landlab.grid.mappers import map_mean_of_link_nodes_to_link
from landlab.grid.mappers import map_max_of_link_nodes_to_link
from landlab.grid.mappers import map_min_of_link_nodes_to_link
from landlab.io import read_esri_ascii

#Global variables
REG_FACTOR = 0.001 #Regularisation factor
COURANT_2D = 0.25 # Courant Number 2D flow
COURANT_1D = 0.50 # Courant number 1D flow
STR_RIVER = 0.001 # Riverbed storage factor

class gwflow_EFD(object):
		
	def __init__(self, env_state, data_in):
	
		env_state.SZgrid.add_zeros('node', 'recharge', dtype=float)		
		env_state.SZgrid.add_zeros('node', 'discharge', dtype=float)		
		env_state.SZgrid.add_zeros('node', 'river_stage__elevation', dtype=float)		
		env_state.SZgrid.add_zeros('node', 'water_storage_anomaly', dtype=float)
		
		env_state.SZgrid.at_node['topographic__elevation'] = np.array(env_state.grid.at_node['topographic__elevation'])		
		#env_state.SZgrid.at_node['BOT'] =  np.array(env_state.grid.at_node['topographic__elevation'])
					
		act_links = env_state.SZgrid.active_links
				
		Kmax = map_max_of_link_nodes_to_link(env_state.SZgrid, 'Hydraulic_Conductivity')		
		Kmin = map_min_of_link_nodes_to_link(env_state.SZgrid, 'Hydraulic_Conductivity')		
		Ksl = map_mean_of_link_nodes_to_link(env_state.SZgrid, 'Hydraulic_Conductivity')
		
		self.Ksat = np.zeros(len(Ksl))		
		self.Ksat[act_links] = Kmax[act_links]*Kmin[act_links]/Ksl[act_links]
		
		self.hriv = np.array(env_state.SZgrid.at_node['water_table__elevation'])		
		self.wte_dt = np.array(env_state.SZgrid.at_node['water_table__elevation'])
		dzdl = env_state.SZgrid.calc_grad_at_link(env_state.SZgrid.at_node['topographic__elevation'])
		self.dzdl = np.where(np.abs(dzdl) > 0.10, 0.001, 1.0)
		self.zm = map_mean_of_link_nodes_to_link(env_state.SZgrid,'topographic__elevation')
		
	def add_second_layer_gw(self, env_state, thickness, Ksat, Sy, Ss):	
		# thickness:	Thickness of the deep aquifer
		# Ksat:			Saturated hydraulic conductivity of the deep aquifer
		# Sy:			Specific yield of the second layer
		# Ss:			Specific storage of the second layer	
		env_state.SZgrid.add_zeros('node', 'Ksat_2', dtype=float)		
		env_state.SZgrid.add_zeros('node', 'BOT_2', dtype=float)		
		env_state.SZgrid.add_zeros('node', 'Sy_2', dtype=float)		
		env_state.SZgrid.add_zeros('node', 'Ss_2', dtype=float)
		env_state.SZgrid.add_zeros('node', 'HEAD_2', dtype=float)
	
		env_state.SZgrid.at_node['BOT_2'] = np.array(
			env_state.SZgrid.at_node['BOT']
			+ thickness
			)
		
		self.thickness = thickness
		
		env_state.SZgrid.at_node['Sy_2'][:] = Sy		
		env_state.SZgrid.at_node['Ss_2'][:] = Ss		
		env_state.SZgrid.at_node['Ksat_2'][:] = Ksat
		
		act_links = env_state.SZgrid.active_links		
		Kmax = map_max_of_link_nodes_to_link(env_state.SZgrid, 'Ksat_2')		
		Kmin = map_min_of_link_nodes_to_link(env_state.SZgrid, 'Ksat_2')		
		Ksl = map_mean_of_link_nodes_to_link(env_state.SZgrid, 'Ksat_2')
		
		self.Ksat_2 = np.zeros(len(Ksl))		
		self.Ksat_2[act_links] = Kmax[act_links]*Kmin[act_links]/Ksl[act_links]		
		env_state.SZgrid.at_node['HEAD_2'][:] = np.array(env_state.SZgrid.at_node['water_table__elevation'])		
		
	def run_one_step_gw_R(self, env_state, dt, tht_dt, rtht_dt, Droot):				
		"""
		Function to update water table depending on the unsaturated zone.
		Parameters:
		Droot:		Rooting depth [mm]
		tht_dt:		Water content at time t [-]
		Duz:		Unsaturated zone depth
		env_state:	grid:	z:	Topograhic elevation
							h:	water table
							fs:	Saturated water content
							fc:	Field capacity
							Sy:	Specific yield
							dq:	water storage anomaly					
		Groundwater storage variation
		"""				
		dts = time_step(COURANT_2D, env_state.SZgrid.at_node['SZ_Sy'],
						env_state.SZgrid.at_node['Hydraulic_Conductivity'],
						env_state.SZgrid.at_node['water_table__elevation'],
						env_state.SZgrid.at_node['BOT'],
						env_state.SZgrid.dx,
						env_state.SZgrid.core_nodes)
		
		act_links = env_state.SZgrid.active_links
		A = np.power(env_state.SZgrid.dx,2)
		Ariv = (env_state.grid.at_node['river_width']
				* env_state.grid.at_node['river_length'])
		W =  Ariv / env_state.SZgrid.dx		
		kriv = np.where(Ariv == 0, 0, 1/Ariv)
		kaq = 1/A
		
		Tch = exponential_T(env_state.grid.at_node['SS_loss'], STR_RIVER,
				env_state.grid.at_node['river_topo_elevation'], self.hriv)
		
		dts_riv = time_step_confined(COURANT_1D, env_state.SZgrid.at_node['SZ_Sy'],
						Tch/W, W, env_state.riv_nodes)
		
		dtp = np.nanmin([dt, dts, dts_riv])
		dtsp = dtp
		env_state.SZgrid.at_node['discharge'][:] = 0.0
		wsa_aux = np.zeros(len(env_state.SZgrid.at_node['discharge'][:]))
		
		while dtp <= dt:
			# Make water table always below or equal surface elevation
			env_state.SZgrid.at_node['water_table__elevation'][:] = np.maximum(
				env_state.SZgrid.at_node['water_table__elevation'],
				env_state.SZgrid.at_node['BOT']
				)
			
			# Make water table always greater or equal to bottom elevation
			env_state.SZgrid.at_node['water_table__elevation'][:] = np.minimum(
				env_state.SZgrid.at_node['topographic__elevation'],
				env_state.SZgrid.at_node['water_table__elevation']
				)
			# Make river water table always below or equal surface elevation
			self.hriv = np.minimum(self.hriv,
				env_state.grid.at_node['river_topo_elevation']
				)
			
			# Calculate the hydraulic gradients
			dhdl = env_state.SZgrid.calc_grad_at_link(env_state.SZgrid.at_node['water_table__elevation'])
			
			# mean hydraulic head at the face of the of the patch
			qs = np.zeros(len(self.Ksat))
			
			#zm = map_mean_of_link_nodes_to_link(env_state.SZgrid,'topographic__elevation')
			hm = map_mean_of_link_nodes_to_link(env_state.SZgrid,'water_table__elevation')
			bm = map_mean_of_link_nodes_to_link(env_state.SZgrid,'BOT')
			
			# Calculate flux per unit length at each face
			qs[act_links] = -self.Ksat[act_links]*(hm[act_links]-bm[act_links])*dhdl[act_links]
		
			# Calculate the flux gradient
			dqsdxy = (-env_state.SZgrid.calc_flux_div_at_node(qs)
					+ env_state.SZgrid.at_node['recharge']/dt)
			
			# Calculate cell river flux
			Tch = exponential_T(env_state.grid.at_node['SS_loss'], STR_RIVER,
				env_state.grid.at_node['river_topo_elevation'], self.hriv)
				
			qs_riv = -(Tch*(env_state.SZgrid.at_node['water_table__elevation']
				- self.hriv)*2 / (env_state.SZgrid.dx + W))
			
			dqsdxy += kaq*qs_riv
			
			# Regularization approach for aquifer cells
			dqs = regularization(
				env_state.SZgrid.at_node['topographic__elevation'],
				env_state.SZgrid.at_node['water_table__elevation'],
				env_state.SZgrid.at_node['BOT'],
				dqsdxy, REG_FACTOR)
			
			# Regularization approach for river cells
			dqs_riv = regularization(
				env_state.grid.at_node['river_topo_elevation'],
				self.hriv, 	env_state.SZgrid.at_node['BOT'],
				-kriv*qs_riv, REG_FACTOR)
			
			# Update the head elevations
			env_state.SZgrid.at_node['water_table__elevation'] += ((dqsdxy-dqs)
				* dtsp / env_state.SZgrid.at_node['SZ_Sy'])
				
			self.hriv += (-kriv*qs_riv - dqs_riv)*dtsp/env_state.SZgrid.at_node['SZ_Sy']
			
			# Update storage change for soil-gw interactions
			env_state.SZgrid.at_node['water_storage_anomaly'][:] = (dqsdxy-dqs) *dtsp
			fun_update_UZ_SZ_depth(env_state, tht_dt, rtht_dt, Droot)
			env_state.SZgrid.at_node['discharge'][:] += (dqs + dqs_riv*Ariv/A)*dtsp
			
			dtsp = time_step(COURANT_2D, env_state.SZgrid.at_node['SZ_Sy'],
						env_state.SZgrid.at_node['Hydraulic_Conductivity'],
						env_state.SZgrid.at_node['water_table__elevation'],
						env_state.SZgrid.at_node['BOT'],
						env_state.SZgrid.dx,
						env_state.SZgrid.core_nodes)
			
			dtsp_riv = time_step_confined(COURANT_1D, env_state.SZgrid.at_node['SZ_Sy'],
						Tch/W, W, env_state.riv_nodes)
			
			dtsp = np.min([dtsp, dtsp_riv])
			
			if dtsp <= 0:
				raise Exception("invalid time step", dtsp)
			if dtp == dt:
				dtp += dtsp
			elif (dtp + dtsp) > dt:
				dtsp = dt - dtp
				dtp += dtsp
			else:
				dtp += dtsp
			
		self.wte_dt = np.array(env_state.SZgrid.at_node['water_table__elevation'])
		
		env_state.SZgrid.at_node['discharge'][:] *= (1/dt)
		
		env_state.grid.at_node['riv_sat_deficit'][:] = np.power(env_state.grid.dx,2)\
				* np.array(env_state.grid.at_node['river_topo_elevation'][:]
				- env_state.SZgrid.at_node['water_table__elevation'][:])
		
		env_state.grid.at_node['riv_sat_deficit'][env_state.grid.at_node['riv_sat_deficit'][:] < 0] = 0.0
		
		pass

	def run_one_step_gw_2Layer(self, env_state, dt, tht_dt, rtht_dt, Droot):	
				
		"""
		Function to update water table depending on the unsaturated zone.
		Parameters:
		Droot:		Rooting depth [mm]
		tht_dt:		Water content at time t [-]
		Duz:		Unsaturated zone depth
		env_state:	grid:	z:	Topograhic elevation
							h:	water table
							fs:	Saturated water content
							fc:	Field capacity
							Sy:	Specific yield
							dq:	water storage anomaly
					
		Groundwater storage variation
		"""
		dts = time_step(COURANT_2D, env_state.SZgrid.at_node['SZ_Sy'],
						env_state.SZgrid.at_node['Hydraulic_Conductivity'],
						env_state.SZgrid.at_node['water_table__elevation'],
						env_state.SZgrid.at_node['BOT'],
						env_state.SZgrid.dx,
						env_state.SZgrid.core_nodes)
		
		act_links = env_state.SZgrid.active_links
		A = np.power(env_state.SZgrid.dx,2)
		dtp = np.nanmin([dt, dts])
		dtsp = dtp
		env_state.SZgrid.at_node['discharge'][:] = 0.0
		
		while dtp <= dt:
			
			## specifiying river flow boundary conditions for groundwater flow
			env_state.SZgrid.at_node['water_table__elevation'][:] = np.maximum(
				env_state.SZgrid.at_node['water_table__elevation'],
				env_state.SZgrid.at_node['BOT']
				)
			## specifiying river flow boundary conditions for groundwater flow
			env_state.SZgrid.at_node['water_table__elevation'][:] = np.minimum(
				env_state.SZgrid.at_node['topographic__elevation'],
				env_state.SZgrid.at_node['water_table__elevation']
				)				
			# Calcualte vertical conductivity for river cells
			p = np.where((env_state.SZgrid.at_node['HEAD_2']
				-env_state.SZgrid.at_node['BOT_2']) > 0,1,0)
			
			dh_1 = (env_state.SZgrid.at_node['water_table__elevation']
				- env_state.SZgrid.at_node['BOT_2']
				)*p
						
			dh_2 = (env_state.SZgrid.at_node['HEAD_2']
				- env_state.SZgrid.at_node['BOT']
				)*(1-p)
						
			Cz = (np.where(dh_1 > 0,1/(env_state.SZgrid.at_node['Hydraulic_Conductivity']
				/(0.5*dh_1)),0) + 1/(env_state.SZgrid.at_node['Ksat_2']
				/(0.5*self.thickness)))
		
			aux_h = np.where((env_state.SZgrid.at_node['HEAD_2']
				-env_state.SZgrid.at_node['BOT_2']) > 0, env_state.SZgrid.at_node['water_table__elevation'],
				env_state.SZgrid.at_node['HEAD_2'])
			
			dhdz = np.where((dh_2+dh_1) > 0,
				(aux_h - env_state.SZgrid.at_node['HEAD_2'])/np.power(self.thickness+dh_1,2),0)
			
			dqz = - (1/Cz)*dhdz
			
			# Calculate the hydraulic gradients
			dhdl_1 = env_state.SZgrid.calc_grad_at_link(env_state.SZgrid.at_node['water_table__elevation'])
			dhdl_2 = env_state.SZgrid.calc_grad_at_link(env_state.SZgrid.at_node['HEAD_2'])
			
			# mean hydraulic head at the face of the of the patch
			qxy_1 = np.zeros(len(self.Ksat))
			qxy_2 = np.zeros(len(self.Ksat_2))
			
			hm = map_mean_of_link_nodes_to_link(env_state.SZgrid,'water_table__elevation')
			bm = map_mean_of_link_nodes_to_link(env_state.SZgrid,'BOT_2')
						
			# Calculate flux per unit length at each face layer 1
			qxy_1[act_links] = -self.Ksat[act_links]*(hm[act_links]-bm[act_links])*dhdl_1[act_links]
			
			# Calculate flux per unit length at each face layer 2
			hm = map_mean_of_link_nodes_to_link(env_state.SZgrid,'HEAD_2')
			bm = map_mean_of_link_nodes_to_link(env_state.SZgrid,'BOT')
			
			b2 = np.minimum(hm-bm, Self.thickness)			
			qxy_2[act_links] = -self.Ksat_2[act_links]*b2[act_links]*dhdl_2[act_links]
					
			# Calculate the flux gradient
			dqxy_1 = (-env_state.SZgrid.calc_flux_div_at_node(qxy_1) + dqz + p*env_state.SZgrid.at_node['recharge']/dt)#*dtsp
			dqxy_2 = (-env_state.SZgrid.calc_flux_div_at_node(qxy_2) - dqz + (1-p)*env_state.SZgrid.at_node['recharge']/dt)#*dtsp
						
			CRIV =  smoth_func(env_state.SZgrid.at_node['water_table__elevation'],
					env_state.grid.at_node['river_topo_elevation'],0.00001,
					dqxy_1, env_state.SZgrid.core_nodes)*(1/A)*env_state.grid.at_node['SS_loss']
			
			FSy, FSs = smoth_func_L2(env_state.SZgrid.at_node['HEAD_2'][:],
				env_state.SZgrid.at_node['BOT'][:],
				self.thickness,0.0001, env_state.SZgrid.core_nodes)
			
			alpha = 1/(env_state.SZgrid.at_node['SZ_Sy'] + dtsp*CRIV)
			
			# Update water table upper layer
			env_state.SZgrid.at_node['water_table__elevation'][env_state.SZgrid.core_nodes] = (dtsp*alpha[env_state.SZgrid.core_nodes]
				*((p*dqxy_1)[env_state.SZgrid.core_nodes] + CRIV[env_state.SZgrid.core_nodes]*env_state.grid.at_node['river_topo_elevation'][env_state.SZgrid.core_nodes])
				+ env_state.SZgrid.at_node['SZ_Sy'][env_state.SZgrid.core_nodes]*alpha[env_state.SZgrid.core_nodes]
				*env_state.SZgrid.at_node['water_table__elevation'][env_state.SZgrid.core_nodes])
			
			# Update water table lower layer
			env_state.SZgrid.at_node['HEAD_2'][env_state.SZgrid.core_nodes] += (dqxy_2[env_state.SZgrid.core_nodes]*dtsp*
				(FSy[env_state.SZgrid.core_nodes]/env_state.SZgrid.at_node['Sy_2'][env_state.SZgrid.core_nodes]+
				FSs[env_state.SZgrid.core_nodes]/env_state.SZgrid.at_node['Ss_2'][env_state.SZgrid.core_nodes])
				)
			
			env_state.SZgrid.at_node['water_table__elevation'][env_state.SZgrid.core_nodes] = np.maximum(
				env_state.SZgrid.at_node['water_table__elevation'][env_state.SZgrid.core_nodes],
				env_state.SZgrid.at_node['BOT_2'][env_state.SZgrid.core_nodes])
			
			# Update water storage anomalies
			env_state.grid.at_node['Base_flow'][env_state.SZgrid.core_nodes] += (
				CRIV[env_state.SZgrid.core_nodes]
				*np.abs(env_state.SZgrid.at_node['water_table__elevation'][env_state.SZgrid.core_nodes]
				-env_state.grid.at_node['river_topo_elevation'][env_state.SZgrid.core_nodes])
				)
						
			dtsp = time_step(COURANT_2D, env_state.SZgrid.at_node['SZ_Sy'],
						env_state.SZgrid.at_node['Hydraulic_Conductivity'],
						env_state.SZgrid.at_node['water_table__elevation'],
						env_state.SZgrid.at_node['BOT'],
						env_state.SZgrid.dx,
						env_state.SZgrid.core_nodes
						)
			
			if dtsp <= 0:
				raise Exception("invalid time step", dtsp)
			if dtp == dt:
				dtp += dtsp
			elif (dtp + dtsp) > dt:
				dtsp = dt - dtp
				dtp += dtsp
			else:
				dtp += dtsp
			
		self.wte_dt = np.array(env_state.SZgrid.at_node['water_table__elevation'])
		
		env_state.SZgrid.at_node['discharge'][:] *= (1/dt)
		
		env_state.grid.at_node['riv_sat_deficit'][:] = np.power(env_state.grid.dx,2)\
				*np.array(env_state.grid.at_node['topographic__elevation']-\
				env_state.SZgrid.at_node['water_table__elevation'][:])
		
		pass
	
	def run_one_step_gw_var_T(self, env_state, dt, tht_dt, rtht_dt, Droot, f):
		"""
		Function to update water table depending on the unsaturated zone.
		Parameters:
		Droot:		Rooting depth [mm]
		tht_dt:		Water content at time t [-]
		Duz:		Unsaturated zone depth
		env_state:	grid:	z:	Topograhic elevation
							h:	water table
							fs:	Saturated water content
							fc:	Field capacity
							Sy:	Specific yield
							dq:	water storage anomaly
							f:	effective depth of the aquifer
					
		Groundwater storage variation
		"""
		act_links = env_state.SZgrid.active_links
		
		dts = time_step_confined(COURANT_2D, env_state.SZgrid.at_node['SZ_Sy'],
			exponential_T(env_state.SZgrid.at_node['Hydraulic_Conductivity'], f,
			env_state.SZgrid.at_node['topographic__elevation'],
			env_state.SZgrid.at_node['water_table__elevation']),
			env_state.SZgrid.dx,
			env_state.SZgrid.core_nodes
			)
		
		dtp = np.nanmin([dt, dts])		
		dtsp = dtp
		
		A = np.power(env_state.SZgrid.dx,2)
		Ariv = (env_state.grid.at_node['river_width']
				* env_state.grid.at_node['river_length'])
		W =  Ariv / env_state.SZgrid.dx		
		kriv = np.where(Ariv == 0, 0, 1/Ariv)
		stage = env_state.grid.at_node['Q_ini'] * kriv
		kaq = 1/A
		
		Tch = exponential_T(env_state.grid.at_node['SS_loss'], STR_RIVER,
				env_state.grid.at_node['river_topo_elevation'], self.hriv)
		
		dts_riv = time_step_confined(COURANT_1D, env_state.SZgrid.at_node['SZ_Sy'],
						Tch/W, W, env_state.riv_nodes)
		
		dtp = np.nanmin([dt, dts, dts_riv])
		
		env_state.SZgrid.at_node['discharge'][:] = 0.0		
				
		while dtp <= dt:
			
			# specifiying river flow boundary conditions for groundwater flow		
			env_state.SZgrid.at_node['water_table__elevation'][:] = np.minimum(
				env_state.SZgrid.at_node['topographic__elevation'],
				env_state.SZgrid.at_node['water_table__elevation']
				)
			
			# Make river water table always below or equal surface elevation
			self.hriv = np.minimum(self.hriv,
				env_state.grid.at_node['river_topo_elevation']
				)
			
			# Calculate conductivity for river cells			
			hm = map_mean_of_link_nodes_to_link(env_state.SZgrid,'water_table__elevation')
			#bm = map_mean_of_link_nodes_to_link(env_state.SZgrid,'BOT')			
			
			# Calculate transmissivity
			T = exponential_T(self.Ksat*self.dzdl, f, self.zm, hm)
			#T = (1.0/ns)*np.power(1-(zm-hm)/(bm-zm),ns)
			
			# Calculate the hydraulic gradients			
			dhdl = env_state.SZgrid.calc_grad_at_link(env_state.SZgrid.at_node['water_table__elevation'])
			
			# mean hydraulic head at the face of the of the patch			
			qs = np.zeros(len(self.Ksat))
					
			# Calculate flux per unit length at each face			
			#qs[act_links] = -self.Ksat[act_links]*D[act_links]*dhdl[act_links]
			qs[act_links] = -T[act_links]*dhdl[act_links]
			
			# Calculate the flux gradient
			dqsdxy = (-env_state.SZgrid.calc_flux_div_at_node(qs)+env_state.SZgrid.at_node['recharge']/dt)#*dtsp
			
			# Calculate the flux gradient
			dqsdxy = (-env_state.SZgrid.calc_flux_div_at_node(qs)
					+ env_state.SZgrid.at_node['recharge']/dt)
			
			# Calculate cell river flux
			Tch = exponential_T(env_state.grid.at_node['SS_loss'], STR_RIVER,
				env_state.grid.at_node['river_topo_elevation'], self.hriv)
				
			qs_riv = -(Tch*(env_state.SZgrid.at_node['water_table__elevation']
				- (self.hriv+stage))*2 / (env_state.SZgrid.dx + W))
			
			dqsdxy += kaq*qs_riv
			
			# Regularization approach for aquifer cells
			dqs = regularization_T(
				env_state.SZgrid.at_node['topographic__elevation'],
				env_state.SZgrid.at_node['water_table__elevation'],
				f, dqsdxy, REG_FACTOR
				)
			
			# Regularization approach for river cells
			dqs_riv = regularization_T(
				env_state.grid.at_node['river_topo_elevation'],
				self.hriv, 	f, -kriv*qs_riv, REG_FACTOR
				)
			
			# Update the head elevations
			env_state.SZgrid.at_node['water_table__elevation'] += ((dqsdxy-dqs)
				* dtsp / env_state.SZgrid.at_node['SZ_Sy'])
				
			self.hriv += (-kriv*qs_riv - dqs_riv)*dtsp/env_state.SZgrid.at_node['SZ_Sy']
			
			# Update storage change for soil-gw interactions
			env_state.SZgrid.at_node['water_storage_anomaly'][:] = (dqsdxy-dqs) *dtsp
			fun_update_UZ_SZ_depth(env_state, tht_dt, rtht_dt, Droot)
			env_state.SZgrid.at_node['discharge'][:] += (dqs + dqs_riv*Ariv/A)*dtsp
			
			dtsp = time_step_confined(COURANT_2D, env_state.SZgrid.at_node['SZ_Sy'],
				exponential_T(env_state.SZgrid.at_node['Hydraulic_Conductivity'], f,
				env_state.SZgrid.at_node['topographic__elevation'],
				env_state.SZgrid.at_node['water_table__elevation']),
				env_state.SZgrid.dx,
				env_state.SZgrid.core_nodes
				)
			
			dtsp_riv = time_step_confined(COURANT_1D, env_state.SZgrid.at_node['SZ_Sy'],
						Tch/W, W, env_state.riv_nodes)
			
			dtsp = np.min([dtsp, dtsp_riv])			
			
			# Update time step
			if dtsp <= 0:
				raise Exception("invalid time step", dtsp)			
			if dtp == dt:			
				dtp += dtsp			
			elif (dtp + dtsp) > dt:			
				dtsp = dt - dtp				
				dtp += dtsp				
			else:			
				dtp += dtsp
			
		self.wte_dt = np.array(env_state.SZgrid.at_node['water_table__elevation'])
		
		env_state.SZgrid.at_node['discharge'][:] *= (1/dt)
		
		env_state.grid.at_node['riv_sat_deficit'][:] = (np.power(env_state.grid.dx, 2)
				* np.array(env_state.grid.at_node['topographic__elevation']
				- env_state.SZgrid.at_node['water_table__elevation'][:])
				)
				
		pass
	
	def SZ_potential_ET(self, env_state, pet_sz):	
		SZ_aet = (env_state.SZgrid.at_node['water_table__elevation']
			- env_state.grid.at_node['topographic__elevation']
			+ env_state.Droot*0.001)*1000		
		SZ_aet = np.where(SZ_aet > 0, SZ_aet,0.0)		
		f = SZ_aet/env_state.Droot
		return f*pet_sz

def fun_update_UZ_SZ_depth(env_state, tht_dt, rtht_dt, Droot):
	"""
	Function to update water table depending on both water content of
	the unsaturated zone.
	Parameters:
	Droot:		Rooting depth [mm]
	tht_dt:		Water content at time t [-]
	Duz:		Unsaturated zone depth
	env_state:	grid:	z:	Topograhic elevation
						h:	water table
						fs:	Saturated water content
						fc:	Field capacity
						Sy:	Specific yield
						dq:	water storage anomaly
	Groundwater storage variation
	"""	
	h0 = env_state.SZgrid.at_node['water_table__elevation'] - \
		env_state.SZgrid.at_node['water_storage_anomaly']/ \
		env_state.SZgrid.at_node['SZ_Sy']
	
	tht_dt = np.where(env_state.SZgrid.at_node['water_storage_anomaly'][:] < 0.0,
			env_state.fc, tht_dt)
	
	dtht = env_state.grid.at_node['saturated_water_content'] - tht_dt
		
	ruz = (env_state.grid.at_node['topographic__elevation']-Droot) - h0

	dh = np.where((h0-(env_state.grid.at_node['topographic__elevation']-Droot)) > 0.0,
		env_state.SZgrid.at_node['water_storage_anomaly']/dtht,
		env_state.SZgrid.at_node['water_storage_anomaly']/
		env_state.SZgrid.at_node['SZ_Sy'])

	h_aux = h0+dh-(env_state.grid.at_node['topographic__elevation']-Droot)
	
	dhr_aux = np.abs(ruz)-np.abs(dh)
	
	dh_aux = np.where(dh >= 0,1,-1)
	
	alpha = -1.*np.less_equal(dhr_aux, 0)*dh_aux*np.less_equal(ruz, 0)
			
	beta = np.less_equal(dhr_aux, 0)*dh_aux*np.less_equal(ruz, 0)
	
	gama = np.where(dhr_aux > 0, 1, 0)
	
	gama = np.where(dh > 0, gama, 1-gama)*np.less(h_aux, 0)
	
	gama[h_aux < 0] = 1
			
	dht = ((env_state.SZgrid.at_node['water_storage_anomaly']
		+ ruz*(alpha*env_state.SZgrid.at_node['SZ_Sy'] + beta*dtht))
		/ ((gama*env_state.SZgrid.at_node['SZ_Sy'] + (1-gama)*dtht)))
					
	env_state.SZgrid.at_node['water_table__elevation'] = h0 + dht
	
	env_state.SZgrid.at_node['discharge'][:] += np.where((
		env_state.SZgrid.at_node['topographic__elevation']
		- env_state.SZgrid.at_node['water_table__elevation']) <= 0.0,
		- (env_state.SZgrid.at_node['topographic__elevation']
		- env_state.SZgrid.at_node['water_table__elevation'])*dtht, 0.0)
			
	env_state.SZgrid.at_node['water_table__elevation'][:] = np.minimum(
		env_state.SZgrid.at_node['water_table__elevation'],
		env_state.SZgrid.at_node['topographic__elevation'])
	
	pass

def storage(env_state):
	storage = np.sum((env_state.SZgrid.at_node['water_table__elevation'][env_state.SZgrid.core_nodes] -\
		env_state.SZgrid.at_node['BOT'][env_state.SZgrid.core_nodes])*\
		env_state.SZgrid.at_node['SZ_Sy'][env_state.SZgrid.core_nodes])
	#discharge = np.sum(env_state.SZgrid.at_node['discharge'][env_state.SZgrid.core_nodes])
	#recharge = np.sum(env_state.SZgrid.at_node['recharge'][env_state.grid.core_nodes])
	return storage#+discharge#+recharge
	
def storage_2layer(env_state):

	head_Sy = (np.maximum(env_state.SZgrid.at_node['water_table__elevation'],
		env_state.SZgrid.at_node['BOT_2'])
		- env_state.SZgrid.at_node['BOT_2'])

	storage_1 = np.sum(head_Ss[env_state.SZgrid.core_nodes]
		*env_state.SZgrid.at_node['SZ_Sy'][env_state.SZgrid.core_nodes])
	#discharge = np.sum(env_state.SZgrid.at_node['discharge'][env_state.SZgrid.core_nodes])
	#recharge = np.sum(env_state.SZgrid.at_node['recharge'][env_state.grid.core_nodes])
	head_Sy = (np.minimun(env_state.SZgrid.at_node['HEAD_2'],
		env_state.SZgrid.at_node['BOT_2'])
		- env_state.SZgrid.at_node['BOT'])
	
	head_Ss = (np.maximum(env_state.SZgrid.at_node['HEAD_2'],
		env_state.SZgrid.at_node['BOT_2'])
		- env_state.SZgrid.at_node['BOT_2'])
	
	storage_2 = np.sum((head_Sy*env_state.SZgrid.at_node['Sy_2']
		+head_Ss*env_state.SZgrid.at_node['Ss_2'])[env_state.SZgrid.core_nodes]
		)
	return storage_1+storage_2#+discharge#+recharge

def storage_uz_sz(env_state):
	""" Total storage in the saturated zone
	Parameters:
		env_state:	state variables and model parameters			
	Output:
		total:		Volume of water stored in the saturated zone
	"""
	str_uz = (env_state.SZgrid.at_node['water_table__elevation']
		- (env_state.grid.at_node['topographic__elevation']
		- env_state.Droot*0.001)
		)
	
	str_uz = np.where(str_uz < 0,0.0, str_uz)		
	
	str_sz = (env_state.SZgrid.at_node['water_table__elevation']
		- env_state.SZgrid.at_node['BOT'])
	
	str_sz = str_sz - str_uz
	
	str_uz = str_uz*(env_state.grid.at_node['saturated_water_content']
		- env_state.fc)
	
	str_sz = str_sz*env_state.SZgrid.at_node['SZ_Sy']
			
	total = np.sum(str_sz[env_state.SZgrid.core_nodes]
		+ str_uz[env_state.SZgrid.core_nodes]
		#+ env_state.SZgrid.at_node['discharge'][env_state.SZgrid.core_nodes]
		)
	return total

def smoth_func_L1(h, hr, r, dq, *nodes):

	aux = np.power(h-hr, 3)/r	
	aux = np.where(aux > 0, aux, 0)
	aux = np.where(aux > 1, 1, aux)
	aux = np.where(dq > 0, 1, aux)
	
	if nodes:
		p = np.zeros(len(aux))
		p[nodes] = 1
		aux *= p
	
	return  aux
	
def smoth_func_L2(h, hr, d, r,*nodes):

	u = (h-hr)/D
	aux = (1-u)/r

	FSy = 1 - np.exp(-aux)
	FSs = 1 - np.exp(aux)

	FSy = np.where(u >= 1, 0, fSy)
	FSs = np.where(u >= 1, fSs, 0)
		
	if nodes:
		aux = np.zeros(len(aux))
		aux[nodes] = 1
		FSs *= aux
		FSy *= aux
	
	return  FSy, FSs

def river_flux(h, hriv, C, A, *nodes):
	q_riv = (h-hriv)*C/A
	if nodes:
		p = np.zeros(len(aux))
		p[nodes] = 1
		aux *= p
	return  q_riv

def smoth_func_T(h, hriv, r, f, dq, *nodes):
	SF = (h-hriv+f)/f
	SF = np.where(SF > 1, SF, 0)
	SF = np.where(SF > 0, 1 - np.exp(SF/r), 0)
	if nodes:
		p = np.zeros(len(SF))
		p[nodes] = 1
		SF *= p
	return SF

def smoth_func(h, hriv, r, dq, *nodes):
	aux = np.power(h-hriv, 3)/r	
	aux = np.where(aux > 0, aux, 0)
	aux = np.where(aux > 1, 1, aux)
	if nodes:
		p = np.zeros(len(aux))
		p[nodes] = 1
		aux *= p
	return  np.where(aux <= 0, 0, aux)
	
# regularization function for unconfined
def regularization(zm, hm, bm, dq, r):
	# zm:	surface elevation
	# hm:	hydraulic head
	# bm:	bottom elevation aquifer
	# dq:	flux per unit area
	# r:	regularization factor
	aux = (hm-bm)/(zm-bm)
	aux = np.where((aux-1) > 0, 1, aux)
	return np.exp((aux-1)/r)*dq*np.where(dq > 0, 1, 0)

# regularization function for confined aquifers
def regularization_T(zm, hm, f, dq, r):
	# zm:	surface elevation
	# hm:	hydraulic head
	# f:	e-folding depth
	# dq:	flux per unit area
	# r:	regularization factor
	aux = (hm-zm)/f+1	
	aux = np.where(aux > 0,aux,0)
	return np.exp((aux-1)/r)*dq*np.where(dq > 0,1,0)
	
def exponential_T(Ksat, f, z, h):
	return Ksat*f*np.exp(-np.maximum(z-h,0)/f)

# Maximum time step for unconfined aquifers
def time_step(D, Sy, Ksat, h, zb, dx, *nodes):
	# D:	Courant number
	T = (h - zb)*Ksat
	dt = D*Sy*np.power(dx,2)/(4*T)
	if nodes:		
		dt = np.nanmin((dt[nodes])[dt[nodes] > 0])
	else:
		dt = np.nanmin(dt[dt > 0])
	return dt

# Maximum time step for confined aquifers
def time_step_confined(D, Sy, T, dx, *nodes):
	# D:	Courant number
	dt = D*Sy*np.power(dx, 2)/(4*T)	
	if nodes:
		dt = np.nanmin((dt[nodes])[dt[nodes] > 0])
	else:
		dt = np.nanmin(dt[dt > 0])
	return dt