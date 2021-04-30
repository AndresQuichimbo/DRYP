
import os
import numpy as np
from landlab import RasterModelGrid
from landlab.grid.mappers import (
	map_mean_of_link_nodes_to_link,
	map_max_of_node_links_to_node,
	map_max_of_link_nodes_to_link,
	map_min_of_link_nodes_to_link)
from landlab.io import read_esri_ascii

#Global variables
REG_FACTOR = 0.001 #Regularisation factor
COURANT_2D = 0.25 # Courant Number 2D flow
COURANT_1D = 0.50 # Courant number 1D flow
STR_RIVER = 0.001 # Riverbed storage factor
# provisional
a_faq = 150
b_faq = 131

class gwflow_EFD(object):
		
	def __init__(self, env_state, data_in):
	
		env_state.SZgrid.add_zeros('node', 'recharge', dtype=float)		
		env_state.SZgrid.add_zeros('node', 'discharge', dtype=float)		
		env_state.SZgrid.add_zeros('node', 'river_stage__elevation', dtype=float)		
		env_state.SZgrid.add_zeros('node', 'water_storage_anomaly', dtype=float)
		
		env_state.SZgrid.at_node['topographic__elevation'] = np.array(env_state.grid.at_node['topographic__elevation'])		
					
		act_links = env_state.SZgrid.active_links
				
		Kmax = map_max_of_link_nodes_to_link(env_state.SZgrid, 'Hydraulic_Conductivity')		
		Kmin = map_min_of_link_nodes_to_link(env_state.SZgrid, 'Hydraulic_Conductivity')		
		Ksl = map_mean_of_link_nodes_to_link(env_state.SZgrid, 'Hydraulic_Conductivity')
		
		self.Ksat = np.zeros(len(Ksl))		
		self.Ksat[act_links] = Kmax[act_links]*Kmin[act_links]/Ksl[act_links]
		
		self.hriv = np.array(env_state.SZgrid.at_node['water_table__elevation'])		
		self.wte_dt = np.array(env_state.SZgrid.at_node['water_table__elevation'])
		self.faq = np.ones_like(Ksl)
		self.faq_node = np.ones_like(self.hriv)
				
		if env_state.func == 1:
			print('Transmissivity function for variable WT depth')
		elif env_state.func == 2:
			print('Constant transmissivity')
		else:
			print('Unconfined conditions: variable thickness')
		print('Change approach in setting_file: line 26')
		
		
		if env_state.func == 1 or env_state.func == 2:
			dzdl = env_state.SZgrid.calc_grad_at_link(env_state.SZgrid.at_node['topographic__elevation'])
			a_faq = map_mean_of_link_nodes_to_link(env_state.SZgrid, 'SZ_a_aq')
			b_faq = map_mean_of_link_nodes_to_link(env_state.SZgrid, 'SZ_b_aq')
			self.faq = a_faq/(1+b_faq*np.abs(dzdl))
			self.faq_node = map_max_of_node_links_to_node(env_state.SZgrid, self.faq)
		self.zm = map_mean_of_link_nodes_to_link(env_state.SZgrid,'topographic__elevation')
		self.flux_out = 0
		self.act_fix_link = 0
		
		A = np.power(env_state.SZgrid.dx,2)		
		Ariv = (env_state.grid.at_node['river_width']
				* env_state.grid.at_node['river_length'])
		self.W =  Ariv / env_state.SZgrid.dx	
		kriv = Ariv
		kriv[Ariv != 0] = 1/Ariv[Ariv != 0]
		self.kriv = kriv
		self.kaq = 1/A
		self.kAriv = Ariv*self.kaq
		self.f = 30
		self.dh = np.zeros_like(Ariv)
		
		if len(env_state.SZgrid.open_boundary_nodes) > 0:			
			self.fixed_links = env_state.SZgrid.links_at_node[
				env_state.SZgrid.open_boundary_nodes]
			self.act_fix_link = 1
		
		
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
			# Calculate vertical conductivity for river cells
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
				self.thickness, REG_FACTOR, env_state.SZgrid.core_nodes)
			
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
	
	def run_one_step_gw(self, env_state, dt, tht_dt, Droot):
		"""Function to update water table depending on the unsaturated zone.
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
							f:	effective aquifer depth
					
		Groundwater storage variation
		"""
		act_links = env_state.SZgrid.active_links
		
		T = transmissivity(env_state, self.Ksat, act_links, self.faq, self.zm)
		
		dts = time_step_confined(COURANT_2D, env_state.SZgrid.at_node['SZ_Sy'],
			map_max_of_node_links_to_node(env_state.SZgrid, T),
			env_state.SZgrid.dx, env_state.SZgrid.core_nodes
			)
	
		stage = env_state.grid.at_node['Q_ini'] * self.kriv
		
		aux_riv = np.ones(len(stage))
		
		aux_riv[stage > 0] = 0
		
		Tch = exponential_T(env_state.grid.at_node['SS_loss'], STR_RIVER,
				env_state.grid.at_node['river_topo_elevation'], self.hriv)
		
		dts_riv = time_step_confined(COURANT_1D, env_state.SZgrid.at_node['SZ_Sy'],
						Tch/self.W, 0.25*(env_state.SZgrid.dx - self.W),
						env_state.riv_nodes)
		
		#dtp = np.nanmin([dt, dts, dts_riv])
		dtp = np.nanmin([dt, dts])
		
		dtsp = dtp
		
		env_state.SZgrid.at_node['discharge'][:] = 0.0

		self.flux_out = 0

		self.dh[:] = 0
		
		while dtp <= dt:
			
			# Make water table always greater or equal to bottom elevation
			env_state.SZgrid.at_node['water_table__elevation'][:] = np.minimum(
				env_state.SZgrid.at_node['topographic__elevation'],
				env_state.SZgrid.at_node['water_table__elevation']
				)
						
			# Make water table always above the bottom elevation
			if env_state.func == 2:
				env_state.SZgrid.at_node['water_table__elevation'][:] = np.maximum(
					env_state.SZgrid.at_node['water_table__elevation'],
					env_state.SZgrid.at_node['BOT']
					)
			
			# Make river water table always below or equal surface elevation
			self.hriv = np.minimum(self.hriv,
				env_state.grid.at_node['river_topo_elevation']
				)
			
			# Calculate transmissivity
			T = transmissivity(env_state, self.Ksat, act_links, self.faq, self.zm)
						
			# Calculate the hydraulic gradients			
			dhdl = env_state.SZgrid.calc_grad_at_link(env_state.SZgrid.at_node['water_table__elevation'])
			
			# Calculate flux per unit length at each face
			qs = np.zeros(len(self.Ksat))
			qs[act_links] = -T[act_links]*dhdl[act_links]
			
			if self.act_fix_link == 1:
				self.flux_out += (np.sum(qs[self.fixed_links])/env_state.SZgrid.dx)*dtsp
				
			# Calculate flux head boundary conditions
			dfhbc = exponential_T(env_state.SZgrid.at_node['SZ_FHB'], 60,
				env_state.SZgrid.at_node['topographic__elevation'],
				env_state.SZgrid.at_node['water_table__elevation']
				)
			self.flux_out += np.sum(dfhbc)
			
			# Calculate flux gradient
			dqsdxy = (-env_state.SZgrid.calc_flux_div_at_node(qs)
					- dfhbc + env_state.SZgrid.at_node['recharge']/dt)
			
			# Calculate river cell flux
			Tch = exponential_T(env_state.grid.at_node['SS_loss'], STR_RIVER,
				env_state.grid.at_node['river_topo_elevation'], self.hriv)

			diff_stage = (env_state.SZgrid.at_node['water_table__elevation']
				- self.hriv)
			
			stage_aux = np.array(stage)
			stage_aux[diff_stage < 0] = 0
			
			qs_riv = -(Tch*(diff_stage-stage_aux)*50 / (env_state.SZgrid.dx - self.W))
			qs_riv[qs_riv < 0] = qs_riv[qs_riv < 0]*aux_riv[qs_riv < 0]
			
			dqsdxy += self.kaq*qs_riv
			
			# Regularization approach for aquifer cells
			if env_state.func == 1 or  env_state.func == 2:
				dqs = regularization_T(
					env_state.SZgrid.at_node['topographic__elevation'],
					env_state.SZgrid.at_node['water_table__elevation'],
					self.faq_node, dqsdxy, REG_FACTOR
					)
			
			else:
				dqs = regularization(
					env_state.SZgrid.at_node['topographic__elevation'],
					env_state.SZgrid.at_node['water_table__elevation'],
					env_state.SZgrid.at_node['BOT'],
					dqsdxy, REG_FACTOR)
					
			# Regularization approach for river cells
			dqs_riv = regularization_T(
				env_state.grid.at_node['river_topo_elevation'],
				self.hriv, 	self.f, -self.kriv*qs_riv, REG_FACTOR
				)
			
			# Update the head elevations
			env_state.SZgrid.at_node['water_table__elevation'] += ((dqsdxy-dqs)
				* dtsp / env_state.SZgrid.at_node['SZ_Sy'])
						
			self.hriv += (-self.kriv*qs_riv - dqs_riv)*dtsp/env_state.SZgrid.at_node['SZ_Sy']
			
			# Update storage change for soil-gw interactions
			env_state.SZgrid.at_node['water_storage_anomaly'][:] = (dqsdxy-dqs) *dtsp			
			fun_update_UZ_SZ_depth(env_state, tht_dt, Droot)
			
			# Calculate total discharge
			env_state.SZgrid.at_node['discharge'][:] += (dqs + dqs_riv*self.kAriv)*dtsp
			
			# Calculate maximum time step
			dtsp = time_step_confined(COURANT_2D, env_state.SZgrid.at_node['SZ_Sy'],
				map_max_of_node_links_to_node(env_state.SZgrid, T),
				env_state.SZgrid.dx, env_state.SZgrid.core_nodes
				)
			
			dtsp_riv = time_step_confined(COURANT_1D, env_state.SZgrid.at_node['SZ_Sy'],
						Tch/self.W, 0.02*(env_state.SZgrid.dx - self.W), env_state.riv_nodes)
			
			#dtsp = np.min([dtsp, dtsp_riv])			
			
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
		
		# Update state variables
		self.dh = np.array(env_state.SZgrid.at_node['water_table__elevation'])-self.wte_dt
		self.wte_dt = np.array(env_state.SZgrid.at_node['water_table__elevation'])
		
		env_state.SZgrid.at_node['discharge'][:] *= (1/dt)
		
		env_state.grid.at_node['riv_sat_deficit'][:] = (np.power(env_state.grid.dx,2)
				* np.array(env_state.grid.at_node['river_topo_elevation'][:]
				- env_state.SZgrid.at_node['water_table__elevation'][:])
				)
		
		env_state.grid.at_node['riv_sat_deficit'][env_state.grid.at_node['riv_sat_deficit'][:] < 0] = 0.0
		
		if self.act_fix_link == 1:
				self.flux_out *= 1/dt
		
		pass
	
	def SZ_potential_ET(self, env_state, pet_sz):
		SZ_aet = (env_state.SZgrid.at_node['water_table__elevation']
			- env_state.grid.at_node['topographic__elevation']
			+ env_state.Droot*0.001)*1000		
		SZ_aet[SZ_aet < 0] = 0		
		f = SZ_aet/env_state.Droot
		return f*pet_sz

def transmissivity(env_state, Ksat, act_links, f, zm):

	T = np.zeros(len(Ksat))
	
	if env_state.func != 2:
		hm = map_mean_of_link_nodes_to_link(env_state.SZgrid,'water_table__elevation')
	
	if env_state.func == 1: # Variable transmissivity			
		T = exponential_T(Ksat, f, zm, hm)
	elif env_state.func == 2: # Constant transmissivity				
		T[act_links] = Ksat[act_links]
	else: # Unconfined aquifer
		bm = map_mean_of_link_nodes_to_link(env_state.SZgrid,'BOT')
		T[act_links] = Ksat[act_links]*(hm[act_links]-bm[act_links])
	
	return T

def fun_update_UZ_SZ_depth(env_state, tht_dt, Droot):
	"""Function to update water table depending on both water content of
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
	return storage
	
def storage_2layer(env_state):

	head_Sy = (np.maximum(env_state.SZgrid.at_node['water_table__elevation'],
		env_state.SZgrid.at_node['BOT_2'])
		- env_state.SZgrid.at_node['BOT_2'])

	storage_1 = np.sum(head_Ss[env_state.SZgrid.core_nodes]
		*env_state.SZgrid.at_node['SZ_Sy'][env_state.SZgrid.core_nodes])
	
	head_Sy = (np.minimun(env_state.SZgrid.at_node['HEAD_2'],
		env_state.SZgrid.at_node['BOT_2'])
		- env_state.SZgrid.at_node['BOT'])
	
	head_Ss = (np.maximum(env_state.SZgrid.at_node['HEAD_2'],
		env_state.SZgrid.at_node['BOT_2'])
		- env_state.SZgrid.at_node['BOT_2'])
	
	storage_2 = np.sum((head_Sy*env_state.SZgrid.at_node['Sy_2']
		+head_Ss*env_state.SZgrid.at_node['Ss_2'])[env_state.SZgrid.core_nodes]
		)
	return storage_1+storage_2

def storage_uz_sz(env_state, tht, dh):
	""" Total storage in the saturated zone
	Parameters:
		env_state:	state variables and model parameters			
	Output:
		total:		Volume of water stored in the saturated zone [m]
	"""
	str_uz1 = (env_state.SZgrid.at_node['water_table__elevation']
		- (env_state.grid.at_node['topographic__elevation']
		- env_state.Droot*0.001)
		)
	
	str_uz1[str_uz1 < 0] = 0.0
		
	str_uz0 = (env_state.SZgrid.at_node['water_table__elevation'] - dh
		- (env_state.grid.at_node['topographic__elevation']
		- env_state.Droot*0.001)
		)
	
	str_uz0[str_uz0 < 0] = 0.0
	
	tht[dh < 0] = env_state.fc[dh < 0]
	
	dtht = env_state.grid.at_node['saturated_water_content'] - tht
	
	str_uz = dtht*(str_uz1-str_uz0)
	
	str_sz = (dh-str_uz1+str_uz0)*env_state.SZgrid.at_node['SZ_Sy']
	
	total = np.sum(str_sz[env_state.SZgrid.core_nodes]
		+ str_uz[env_state.SZgrid.core_nodes]
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