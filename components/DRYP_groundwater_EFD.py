
import os

import numpy as np

from landlab import RasterModelGrid

from landlab.grid.mappers import map_mean_of_link_nodes_to_link

from landlab.grid.mappers import map_max_of_link_nodes_to_link

from landlab.grid.mappers import map_min_of_link_nodes_to_link

from landlab.io import read_esri_ascii


class gwflow_EFD(object):
		
	def __init__(self, env_state, data_in):
	
		env_state.SZgrid.add_zeros('node', 'recharge', dtype = float)		
		env_state.SZgrid.add_zeros('node', 'discharge', dtype = float)		
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
				
		self.wte_dt = np.array(env_state.SZgrid.at_node['water_table__elevation'])
		
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
		self.hriv = np.array(env_state.SZgrid.at_node['water_table__elevation'])
	
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
		dts = time_step(env_state.SZgrid.at_node['SZ_Sy'],
						env_state.SZgrid.at_node['Hydraulic_Conductivity'],
						env_state.SZgrid.at_node['water_table__elevation'],
						env_state.SZgrid.at_node['BOT'],
						env_state.SZgrid.dx,
						env_state.SZgrid.core_nodes)
		
		act_links = env_state.SZgrid.active_links
		
		dtp = np.nanmin([dt, dts])
		dtsp = dtp
		env_state.SZgrid.at_node['discharge'][:] = 0.0
		wsa_aux = np.zeros(len(env_state.SZgrid.at_node['discharge'][:]))
		
		while dtp <= dt:
			
			env_state.SZgrid.at_node['water_table__elevation'][:] = np.maximum(
				env_state.SZgrid.at_node['water_table__elevation'],
				env_state.SZgrid.at_node['BOT']
				)
			
			## specifiying river flow boundary conditions for groundwater flow
			env_state.SZgrid.at_node['water_table__elevation'][:] = np.minimum(
				env_state.SZgrid.at_node['topographic__elevation'],
				env_state.SZgrid.at_node['water_table__elevation']
				)
						
			# Calculate the hydraulic gradients
			dhdl = env_state.SZgrid.calc_grad_at_link(env_state.SZgrid.at_node['water_table__elevation'])
			
			# mean hydraulic head at the face of the of the patch
			qs = np.zeros(len(self.Ksat))
			
			#values_at_links = mg.empty(at = 'link')
			#zm = map_mean_of_link_nodes_to_link(env_state.SZgrid,'topographic__elevation')
			hm = map_mean_of_link_nodes_to_link(env_state.SZgrid,'water_table__elevation')
			bm = map_mean_of_link_nodes_to_link(env_state.SZgrid,'BOT')
			
			# Calculate flux per unit length at each face
			qs[act_links] = -self.Ksat[act_links]*(hm[act_links]-bm[act_links])*dhdl[act_links]
		
			# Calculate the flux gradient
			dqsdxy = (-env_state.SZgrid.calc_flux_div_at_node(qs)+env_state.SZgrid.at_node['recharge']/dt)
			
			# Update water storage anomalies
			#env_state.SZgrid.at_node['water_storage_anomaly'][:] = dqsdxy
			#env_state.SZgrid.at_node['water_storage_anomaly'][:] = np.where((
			#	env_state.SZgrid.at_node['topographic__elevation']-\
			#	env_state.SZgrid.at_node['water_table__elevation']) <= 0.0,
			#	0.0, dqsdxy)
				
			#env_state.SZgrid.at_node['water_storage_anomaly'][:] = np.where((
			#	dqsdxy) <= 0.0, dqsdxy,
			#	env_state.SZgrid.at_node['water_storage_anomaly'][:])
			
			#wsa_aux += env_state.SZgrid.at_node['water_storage_anomaly'][:]
			dqs = regularization(
				env_state.SZgrid.at_node['topographic__elevation'],
				env_state.SZgrid.at_node['water_table__elevation'],
				env_state.SZgrid.at_node['BOT'],
				dqsdxy,0.001)
			
			# Update the head elevations
			#str0 = storage(env_state)
			#str0 = storage_uz_sz(env_state)
			#str0 = np.array(env_state.SZgrid.at_node['water_table__elevation'])
			env_state.SZgrid.at_node['water_table__elevation'] += (dqsdxy-dqs) * dtsp / env_state.SZgrid.at_node['SZ_Sy']
						
			env_state.SZgrid.at_node['water_storage_anomaly'][:] = (dqsdxy-dqs) *dtsp
						
			fun_update_soil_rip_depth(env_state, tht_dt, rtht_dt, Droot)
			
			env_state.SZgrid.at_node['discharge'][:] += dqs*dtsp
			
			dtsp = time_step(env_state.SZgrid.at_node['SZ_Sy'],
						env_state.SZgrid.at_node['Hydraulic_Conductivity'],
						env_state.SZgrid.at_node['water_table__elevation'],
						env_state.SZgrid.at_node['BOT'],
						env_state.SZgrid.dx,
						env_state.SZgrid.core_nodes)
			
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
				* np.array(env_state.grid.at_node['river_topo__elevation'][:]
				- env_state.SZgrid.at_node['water_table__elevation'][:])
		
		env_state.grid.at_node['riv_sat_deficit'][env_state.grid.at_node['riv_sat_deficit'][:] < 0] = 0.0
		
		pass
		
	def run_one_step_gw(self, env_state, dt, tht_dt, rtht_dt, Droot):	
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
		dts = time_step(env_state.SZgrid.at_node['SZ_Sy'],
						env_state.SZgrid.at_node['Hydraulic_Conductivity'],
						env_state.SZgrid.at_node['water_table__elevation'],
						env_state.SZgrid.at_node['BOT'],
						env_state.SZgrid.dx,
						env_state.SZgrid.core_nodes)
		
		act_links = env_state.SZgrid.active_links
		
		dtp = np.nanmin([dt, dts])
		A = np.power(env_state.SZgrid.dx,2)
		dtsp = dtp
		env_state.SZgrid.at_node['discharge'][:] = 0.0
		env_state.grid.at_node['Base_flow'][:] = 0.0
		wsa_aux = np.zeros(len(env_state.SZgrid.at_node['discharge'][:]))
		
		while dtp <= dt:
			
			# specifiying river flow boundary conditions for groundwater flow
			env_state.SZgrid.at_node['water_table__elevation'][:] = np.maximum(
				env_state.SZgrid.at_node['water_table__elevation'],
				env_state.SZgrid.at_node['BOT']
				)
				
			self.hriv = np.minimum(self.hriv,
				env_state.SZgrid.at_node['water_table__elevation']
				)
			
			# specifiying river flow boundary conditions for groundwater flow
			env_state.SZgrid.at_node['water_table__elevation'][:] = np.minimum(
				env_state.SZgrid.at_node['topographic__elevation'],
				env_state.SZgrid.at_node['water_table__elevation']
				)
				
			# Calculate the hydraulic gradients
			dhdl = env_state.SZgrid.calc_grad_at_link(env_state.SZgrid.at_node['water_table__elevation'])
			
			# mean hydraulic head at the face of the of the patch
			qs = np.zeros(len(self.Ksat))
			hm = map_mean_of_link_nodes_to_link(env_state.SZgrid,'water_table__elevation')
			bm = map_mean_of_link_nodes_to_link(env_state.SZgrid,'BOT')
			
			# Calculate flux per unit length at each face
			qs[act_links] = -self.Ksat[act_links]*(hm[act_links]-bm[act_links])*dhdl[act_links]
						
			# Calculate the flux gradient
			dqsdxy = (-env_state.SZgrid.calc_flux_div_at_node(qs)+env_state.SZgrid.at_node['recharge']/dt)#*dtsp
			
			# Calculate river conductivity
			dqs_riv = river_flux(env_state.SZgrid.at_node['water_table__elevation'],
				self.hriv,(env_state.SZgrid.at_node['water_table__elevation']+
				self.hriv)*0.05*env_state.grid.at_node['SS_loss']/env_state.SZgrid.dx,
				A)
			
			qs_riv = regularization(env_state.grid.at_node['river_topo__elevation'],
				self.hriv, env_state.SZgrid.at_node['BOT'],
				dqs_riv,0.001)
						
			#CRIV =  smoth_func(env_state.SZgrid.at_node['water_table__elevation'],
			#		env_state.grid.at_node['river_topo__elevation'],0.0001,
			#		dqsdxy, env_state.SZgrid.core_nodes)*(1/A)*env_state.grid.at_node['SS_loss']
			#			
			#alpha = 1/(env_state.SZgrid.at_node['SZ_Sy'] + dtsp*CRIV)
			#
			#env_state.SZgrid.at_node['water_table__elevation'] = (dtsp*alpha
			#	*(dqsdxy - CRIV*env_state.SZgrid.at_node['water_table__elevation'])
			#	+ env_state.SZgrid.at_node['SZ_Sy']*alpha
			#	*env_state.SZgrid.at_node['water_table__elevation'])
			
			env_state.SZgrid.at_node['water_table__elevation'] += (dqsdxy-dqs_riv) * dtsp / env_state.SZgrid.at_node['SZ_Sy']
			env_state.SZgrid.at_node['water_storage_anomaly'][:] = (dqsdxy-dqs_riv) *dtsp
						
			fun_update_soil_rip_depth(env_state, tht_dt, rtht_dt, Droot)
			
			# Aquifer discharge into stream (base flow)
			env_state.grid.at_node['Base_flow'][env_state.SZgrid.core_nodes] += qs_riv[env_state.SZgrid.core_nodes]*dtsp
			self.hriv += (dqs_riv - qs_riv)*dtsp/env_state.SZgrid.at_node['SZ_Sy']
			
			#env_state.grid.at_node['Base_flow'][env_state.SZgrid.core_nodes] += (
			#	CRIV[env_state.SZgrid.core_nodes]
			#	*np.abs(env_state.SZgrid.at_node['water_table__elevation'][env_state.SZgrid.core_nodes]
			#	-env_state.grid.at_node['river_topo__elevation'][env_state.SZgrid.core_nodes])
			#	)
			
			# Update water storage anomalies
			fun_update_UZ_SZ_depth(env_state, tht_dt, rtht_dt, Droot)
			
			dtsp = time_step(env_state.SZgrid.at_node['SZ_Sy'],
						env_state.SZgrid.at_node['Hydraulic_Conductivity'],
						env_state.SZgrid.at_node['water_table__elevation'],
						env_state.SZgrid.at_node['BOT'],
						env_state.SZgrid.dx,
						env_state.SZgrid.core_nodes)
			
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
		#print(env_state.SZgrid.at_node['Ksat_2'])		
		dts = time_step(env_state.SZgrid.at_node['SZ_Sy'],
						env_state.SZgrid.at_node['Hydraulic_Conductivity'],
						env_state.SZgrid.at_node['water_table__elevation'],
						env_state.SZgrid.at_node['BOT'],
						env_state.SZgrid.dx,
						env_state.SZgrid.core_nodes)
		
		act_links = env_state.SZgrid.active_links
		A = np.power(env_state.SZgrid.dx,2)
		dtp = np.nanmin([dt, dts])#*0.011
		dtsp = dtp
		env_state.SZgrid.at_node['discharge'][:] = 0.0
		env_state.grid.at_node['Base_flow'][:] = 0.0
		wsa_aux = np.zeros(len(env_state.SZgrid.at_node['discharge'][:]))
		
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
					env_state.grid.at_node['river_topo__elevation'],0.00001,
					dqxy_1, env_state.SZgrid.core_nodes)*(1/A)*env_state.grid.at_node['SS_loss']
			
			FSy, FSs = smoth_func_L2(env_state.SZgrid.at_node['HEAD_2'][:],
				env_state.SZgrid.at_node['BOT'][:],
				self.thickness,0.0001, env_state.SZgrid.core_nodes)
			
			alpha = 1/(env_state.SZgrid.at_node['SZ_Sy'] + dtsp*CRIV)
			
			# Update water table upper layer
			env_state.SZgrid.at_node['water_table__elevation'][env_state.SZgrid.core_nodes] = (dtsp*alpha[env_state.SZgrid.core_nodes]
				*((p*dqxy_1)[env_state.SZgrid.core_nodes] + CRIV[env_state.SZgrid.core_nodes]*env_state.grid.at_node['river_topo__elevation'][env_state.SZgrid.core_nodes])
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
				-env_state.grid.at_node['river_topo__elevation'][env_state.SZgrid.core_nodes])
				)
						
			#env_state.SZgrid.at_node['water_storage_anomaly'][:] = dqsdxy

			#fun_update_UZ_SZ_depth(env_state, tht_dt, rtht_dt, Droot)
			
			dtsp = time_step(env_state.SZgrid.at_node['SZ_Sy'],
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
		
		dts = time_step_confined(env_state.SZgrid.at_node['SZ_Sy'],
			exponential_T(env_state.SZgrid.at_node['Hydraulic_Conductivity'], f,
			env_state.SZgrid.at_node['topographic__elevation'],
			env_state.SZgrid.at_node['water_table__elevation']),
			env_state.SZgrid.dx,
			env_state.SZgrid.core_nodes
			)
		
		dtp = np.nanmin([dt, dts])		
		dtsp = dtp
		
		A = np.power(env_state.SZgrid.dx,2)
		
		env_state.SZgrid.at_node['discharge'][:] = 0.0		
		env_state.grid.at_node['Base_flow'][:] = 0.0
		
		wsa_aux = np.zeros(len(env_state.SZgrid.at_node['discharge'][:]))		
		riv_node = np.where(env_state.grid.at_node['river'] > 0,1,0)
		
		while dtp <= dt:
			
			# specifiying river flow boundary conditions for groundwater flow		
			env_state.SZgrid.at_node['water_table__elevation'][:] = np.minimum(
				env_state.SZgrid.at_node['topographic__elevation'],
				env_state.SZgrid.at_node['water_table__elevation']
				)
			
			#self.hriv = np.minimum(self.hriv,
			#	env_state.SZgrid.at_node['water_table__elevation']
			#	)
			
			# Calculate conductivity for river cells			
			hm = map_mean_of_link_nodes_to_link(env_state.SZgrid,'water_table__elevation')
			#bm = map_mean_of_link_nodes_to_link(env_state.SZgrid,'BOT')			
			zm = map_mean_of_link_nodes_to_link(env_state.SZgrid,'topographic__elevation')
			
			# Calculate transmissivity
			T = exponential_T(self.Ksat, f, zm, hm)
			#T = (1.0/ns)*np.power(1-(zm-hm)/(bm-zm),ns)
			#T = self.Ksat*100.0
			#print(np.max(T[act_links]),(zm-hm)[act_links],self.Ksat[act_links])
			# Calculate the hydraulic gradients			
			dhdl = env_state.SZgrid.calc_grad_at_link(env_state.SZgrid.at_node['water_table__elevation'])
			
			# mean hydraulic head at the face of the of the patch			
			qs = np.zeros(len(self.Ksat))
					
			# Calculate flux per unit length at each face
			
			#qs[act_links] = -self.Ksat[act_links]*D[act_links]*dhdl[act_links]
			qs[act_links] = -T[act_links]*dhdl[act_links]
			
			# Calculate the flux gradient
			dqsdxy = (-env_state.SZgrid.calc_flux_div_at_node(qs)+env_state.SZgrid.at_node['recharge']/dt)#*dtsp
			
			# Calculate river conductivity			
			CRIV =  (smoth_func(env_state.SZgrid.at_node['water_table__elevation'],
					env_state.grid.at_node['river_topo__elevation'], 0.001,# f,
					dqsdxy, env_state.SZgrid.core_nodes)
					* (1/A) * env_state.grid.at_node['SS_loss']
					)
						
			alpha = 1/(env_state.SZgrid.at_node['SZ_Sy'] + dtsp*CRIV)
			
			env_state.SZgrid.at_node['water_table__elevation'][env_state.SZgrid.core_nodes] = (
				dtsp*alpha[env_state.SZgrid.core_nodes]
				* (dqsdxy[env_state.SZgrid.core_nodes] + CRIV[env_state.SZgrid.core_nodes]
				* env_state.grid.at_node['river_topo__elevation'][env_state.SZgrid.core_nodes])
				+ env_state.SZgrid.at_node['SZ_Sy'][env_state.SZgrid.core_nodes]
				* alpha[env_state.SZgrid.core_nodes]
				* env_state.SZgrid.at_node['water_table__elevation'][env_state.SZgrid.core_nodes])
					
			#env_state.SZgrid.at_node['water_storage_anomaly'][:] = dqsdxy
			
			# Aquifer discharge into stream (base flow)
			dqs_riv = (env_state.grid.at_node['river'] * CRIV
				* (env_state.SZgrid.at_node['water_table__elevation']
				- env_state.grid.at_node['river_topo__elevation'])
				)
			
			env_state.grid.at_node['Base_flow'][:] += dqs_riv * dtsp
			
			#env_state.SZgrid.at_node['water_table__elevation'] += (dqsdxy-dqs_riv) * dtsp / env_state.SZgrid.at_node['SZ_Sy']
			#			
			env_state.SZgrid.at_node['water_storage_anomaly'][:] = (dqsdxy-dqs_riv) * dtsp
						
			fun_update_soil_rip_depth(env_state, tht_dt, rtht_dt, Droot)
			
			# Aquifer discharge into stream (base flow)			
			#env_state.grid.at_node['Base_flow'][env_state.SZgrid.core_nodes] += qs_riv[env_state.SZgrid.core_nodes]*dtsp
			#self.hriv += qs_riv*dtsp/env_state.SZgrid.at_node['SZ_Sy']
			
			# Update the head elevations
			#env_state.SZgrid.at_node['water_table__elevation'] += dqsdxy / env_state.SZgrid.at_node['SZ_Sy']
			fun_update_UZ_SZ_depth(env_state, tht_dt, rtht_dt, Droot)
			
			#wsa_aux += env_state.SZgrid.at_node['discharge'][:]
			#print(env_state.SZgrid.at_node['Hydraulic_Conductivity'][env_state.SZgrid.core_nodes])			
			dtsp = time_step_confined(env_state.SZgrid.at_node['SZ_Sy'],
				exponential_T(env_state.SZgrid.at_node['Hydraulic_Conductivity'], f,
				env_state.SZgrid.at_node['topographic__elevation'],
				env_state.SZgrid.at_node['water_table__elevation']),
				env_state.SZgrid.dx,
				env_state.SZgrid.core_nodes
				)
			
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

	# Modified version of Businesq eq to solve channel flow
	def run_one_step_gw_var_T_mod(self, env_state, dt, tht_dt, rtht_dt, Droot, f):				
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
		
		dts = time_step_confined(env_state.SZgrid.at_node['SZ_Sy'],
			exponential_T(env_state.SZgrid.at_node['Hydraulic_Conductivity'], f,
			env_state.SZgrid.at_node['water_table__elevation'],
			env_state.SZgrid.at_node['topographic__elevation']),
			env_state.SZgrid.dx,
			env_state.SZgrid.core_nodes
			)		
		dtp = np.nanmin([dt, dts])		
		dtsp = dtp		
		A = np.power(env_state.SZgrid.dx,2)		
		env_state.SZgrid.at_node['discharge'][:] = 0.0		
		env_state.grid.at_node['Base_flow'][:] = 0.0		
		wsa_aux = np.zeros(len(env_state.SZgrid.at_node['discharge'][:]))		
		riv_node = np.where(env_state.grid.at_node['river'] > 0,1,0)
		
		while dtp <= dt:			
			# specifiying river flow boundary conditions for groundwater flow		
			env_state.SZgrid.at_node['water_table__elevation'][:] = np.minimum(
				env_state.SZgrid.at_node['topographic__elevation'],
				env_state.SZgrid.at_node['water_table__elevation']
				)			
			self.hriv = np.minimum(self.hriv,
				env_state.SZgrid.at_node['water_table__elevation']
				)			
			# Calculate conductivity for river cells			
			hm = map_mean_of_link_nodes_to_link(env_state.SZgrid,'water_table__elevation')			
			#bm = map_mean_of_link_nodes_to_link(env_state.SZgrid,'BOT')			
			zm = map_mean_of_link_nodes_to_link(env_state.SZgrid,'topographic__elevation')
						
			D = exponential_T(self.Ksat, f, zm, hm)
			#D = (1.0/ns)*np.power(1-(zm-hm)/(bm-zm),ns)
			#D = self.Ksat*100.0
			# Calculate the hydraulic gradients			
			dhdl = env_state.SZgrid.calc_grad_at_link(env_state.SZgrid.at_node['water_table__elevation'])
			
			# mean hydraulic head at the face of the of the patch			
			qs = np.zeros(len(self.Ksat))
					
			# Calculate flux per unit length at each face			
			#qs[act_links] = -self.Ksat[act_links]*D[act_links]*dhdl[act_links]
			qs[act_links] = -D[act_links]*dhdl[act_links]
			#print(D[act_links])
			# Calculate the flux gradient			
			dqsdxy = (-env_state.SZgrid.calc_flux_div_at_node(qs)+env_state.SZgrid.at_node['recharge']/dt)#*dtsp
			
			# Calculate river conductivity			
			dqs_riv = river_flux(env_state.SZgrid.at_node['water_table__elevation'],
				self.hriv, env_state.grid.at_node['SS_loss']
				#(env_state.SZgrid.at_node['water_table__elevation']+self.hriv)
				*0.1/env_state.SZgrid.dx,
				A)*riv_node
			
			qs_riv = regularization(env_state.grid.at_node['river_topo__elevation'],
				self.hriv, f, dqs_riv,0.001)
			#CRIV =  smoth_func(env_state.SZgrid.at_node['water_table__elevation'],
			#		env_state.grid.at_node['river_topo__elevation'],0.00001,
			#		dqsdxy, env_state.SZgrid.core_nodes)*(1/A)*env_state.grid.at_node['SS_loss']
			#alpha = 1/(env_state.SZgrid.at_node['SZ_Sy'] + dtsp*CRIV)
			#
			#env_state.SZgrid.at_node['water_table__elevation'][env_state.SZgrid.core_nodes] = (dtsp
			#	*alpha[env_state.SZgrid.core_nodes]
			#	*(dqsdxy[env_state.SZgrid.core_nodes] + CRIV[env_state.SZgrid.core_nodes]
			#	*env_state.grid.at_node['river_topo__elevation'][env_state.SZgrid.core_nodes])
			#	+ env_state.SZgrid.at_node['SZ_Sy'][env_state.SZgrid.core_nodes]
			#	*alpha[env_state.SZgrid.core_nodes]
			#	*env_state.SZgrid.at_node['water_table__elevation'][env_state.SZgrid.core_nodes])
			#		
			##env_state.SZgrid.at_node['water_storage_anomaly'][:] = dqsdxy
			#
			## Aquifer discharge into stream (base flow)
			#
			#env_state.grid.at_node['Base_flow'][env_state.SZgrid.core_nodes] += (
			#	CRIV[env_state.SZgrid.core_nodes]
			#	*np.abs(env_state.SZgrid.at_node['water_table__elevation'][env_state.SZgrid.core_nodes]
			#	-env_state.grid.at_node['river_topo__elevation'][env_state.SZgrid.core_nodes])
			#	)
			
			env_state.SZgrid.at_node['water_table__elevation'] += (dqsdxy-dqs_riv) * dtsp / env_state.SZgrid.at_node['SZ_Sy']
						
			env_state.SZgrid.at_node['water_storage_anomaly'][:] = (dqsdxy-dqs_riv) *dtsp
						
			fun_update_soil_rip_depth(env_state, tht_dt, rtht_dt, Droot)
			
			
			# Aquifer discharge into stream (base flow)			
			env_state.grid.at_node['Base_flow'][env_state.SZgrid.core_nodes] += qs_riv[env_state.SZgrid.core_nodes]*dtsp
			
			self.hriv += qs_riv*dtsp/env_state.SZgrid.at_node['SZ_Sy']
			
			# Update the head elevations
			
			#env_state.SZgrid.at_node['water_table__elevation'] += dqsdxy / env_state.SZgrid.at_node['SZ_Sy']
			
			#fun_update_UZ_SZ_depth(env_state, tht_dt, rtht_dt, Droot)
			
			#wsa_aux += env_state.SZgrid.at_node['discharge'][:]
							
			dtsp = time_step_confined(env_state.SZgrid.at_node['SZ_Sy'],
				exponential_T(env_state.SZgrid.at_node['Hydraulic_Conductivity'], f,
				env_state.SZgrid.at_node['water_table__elevation'],
				env_state.SZgrid.at_node['topographic__elevation']),
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
	
	def run_one_step_gw_var_T_R(self, env_state, dt, tht_dt, rtht_dt, Droot, f):				
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
		dts = time_step_confined(env_state.SZgrid.at_node['SZ_Sy'],
			exponential_T(env_state.SZgrid.at_node['Hydraulic_Conductivity'], f,
			env_state.SZgrid.at_node['water_table__elevation'],
			env_state.SZgrid.at_node['topographic__elevation']),
			env_state.SZgrid.dx,
			env_state.SZgrid.core_nodes
			)				
		dtp = np.nanmin([dt, dts])
		dtsp = dtp		
		env_state.SZgrid.at_node['discharge'][:] = 0.0
		wsa_aux = np.zeros(len(env_state.SZgrid.at_node['discharge'][:]))
		
		while dtp <= dt:			
			# specifiying river flow boundary conditions for groundwater flow
			env_state.SZgrid.at_node['water_table__elevation'][:] = np.minimum(
				env_state.SZgrid.at_node['topographic__elevation'],
				env_state.SZgrid.at_node['water_table__elevation']
				)		
			# Calculate conductivity for river cells
			hm = map_mean_of_link_nodes_to_link(env_state.SZgrid,'water_table__elevation')
			zm = map_mean_of_link_nodes_to_link(env_state.SZgrid,'topographic__elevation')
						
			D = exponential_T(self.Ksat, f, zm, hm)
			#D = (1.0/ns)*np.power(1-(zm-hm)/(bm-zm),ns)
			#D = self.Ksat*100.0
			#print(np.max(D[act_links]))
			# Calculate the hydraulic gradients			
			dhdl = env_state.SZgrid.calc_grad_at_link(env_state.SZgrid.at_node['water_table__elevation'])
			
			# mean hydraulic head at the face of the of the patch
			qs = np.zeros(len(self.Ksat))
					
			# Calculate flux per unit length at each face
			#qs[act_links] = -self.Ksat[act_links]*D[act_links]*dhdl[act_links]
			qs[act_links] = -D[act_links]*dhdl[act_links]
			
			# Calculate the flux gradient
			dqsdxy = (-env_state.SZgrid.calc_flux_div_at_node(qs)+env_state.SZgrid.at_node['recharge']/dt)*dtsp
		
			# Update water storage anomalies
			dqs = regularization_T(
				env_state.SZgrid.at_node['topographic__elevation'],
				env_state.SZgrid.at_node['water_table__elevation'],
				f, dqsdxy,0.001
				)		
			env_state.SZgrid.at_node['water_table__elevation'] += (dqsdxy-dqs) * dtsp / env_state.SZgrid.at_node['SZ_Sy']
			env_state.SZgrid.at_node['water_storage_anomaly'][:] = (dqsdxy-dqs) *dtsp
			
			fun_update_soil_rip_depth(env_state, tht_dt, rtht_dt, Droot)
			
			env_state.SZgrid.at_node['discharge'][:] += dqs*dtsp
			#wsa_aux += env_state.SZgrid.at_node['discharge'][:]
							
			dtsp = time_step_confined(env_state.SZgrid.at_node['SZ_Sy'],
				exponential_T(env_state.SZgrid.at_node['Hydraulic_Conductivity'], f,
				env_state.SZgrid.at_node['water_table__elevation'],
				env_state.SZgrid.at_node['topographic__elevation']),
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
		
		env_state.grid.at_node['riv_sat_deficit'][:] = (
				env_state.SZgrid.at_node['SZ_Sy']
				*np.power(env_state.grid.dx,2)
				*np.array(env_state.grid.at_node['topographic__elevation']
				-env_state.SZgrid.at_node['water_table__elevation'][:]
				))		
		pass
	
	def run_one_step_gw_var_T_R_s(self, env_state, dt, tht_dt, rtht_dt, Droot, f):				
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
		
		dts = time_step_confined(env_state.SZgrid.at_node['SZ_Sy'],
			exponential_T(env_state.SZgrid.at_node['Hydraulic_Conductivity'], f,
			env_state.SZgrid.at_node['water_table__elevation'],
			env_state.SZgrid.at_node['topographic__elevation']),
			env_state.SZgrid.dx,
			env_state.SZgrid.core_nodes
			)
		dtp = np.nanmin([dt, dts])
		dtsp = dtp
		env_state.SZgrid.at_node['discharge'][:] = 0.0
		wsa_aux = np.zeros(len(env_state.SZgrid.at_node['discharge'][:]))
		
		while dtp <= dt:			
			# specifiying river flow boundary conditions for groundwater flow		
			env_state.SZgrid.at_node['water_table__elevation'][:] = np.minimum(
				env_state.SZgrid.at_node['topographic__elevation'],
				env_state.SZgrid.at_node['water_table__elevation']
				)		
			# Calculate conductivity for river cells			
			hm = map_mean_of_link_nodes_to_link(env_state.SZgrid,'water_table__elevation')			
			#bm = map_mean_of_link_nodes_to_link(env_state.SZgrid,'BOT')			
			zm = map_mean_of_link_nodes_to_link(env_state.SZgrid,'topographic__elevation')
						
			D = exponential_T(self.Ksat, f, zm, hm)
			#D = (1.0/ns)*np.power(1-(zm-hm)/(bm-zm),ns)
			#D = self.Ksat*100.0
			#print(np.max(D[act_links]))
			# Calculate the hydraulic gradients			
			dhdl = env_state.SZgrid.calc_grad_at_link(env_state.SZgrid.at_node['water_table__elevation'])
			
			# mean hydraulic head at the face of the of the patch			
			qs = np.zeros(len(self.Ksat))
					
			# Calculate flux per unit length at each face			
			#qs[act_links] = -self.Ksat[act_links]*D[act_links]*dhdl[act_links]
			qs[act_links] = -D[act_links]*dhdl[act_links]
			
			# Calculate the flux gradient			
			dqsdxy = (-env_state.SZgrid.calc_flux_div_at_node(qs)+env_state.SZgrid.at_node['recharge']/dt)*dtsp
		
			# Update water storage anomalies			
			dqs = regularization_T(
				env_state.SZgrid.at_node['topographic__elevation'],
				env_state.SZgrid.at_node['water_table__elevation'],
				f, dqsdxy,0.001
				)
			
			env_state.SZgrid.at_node['water_table__elevation'] += (dqsdxy-dqs) * dtsp / env_state.SZgrid.at_node['SZ_Sy']
			env_state.SZgrid.at_node['water_storage_anomaly'][:] = (dqsdxy-dqs) *dtsp
			
			fun_update_soil_rip_depth(env_state, tht_dt, rtht_dt, Droot)
			
			env_state.SZgrid.at_node['discharge'][:] += dqs*dtsp
			
			#wsa_aux += env_state.SZgrid.at_node['discharge'][:]
							
			dtsp = time_step_confined(env_state.SZgrid.at_node['SZ_Sy'],
				exponential_T(env_state.SZgrid.at_node['Hydraulic_Conductivity'], f,
				env_state.SZgrid.at_node['water_table__elevation'],
				env_state.SZgrid.at_node['topographic__elevation']),
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
	
	def SZ_potential_ET(self, env_state,pet_sz):	
		SZ_aet = (env_state.SZgrid.at_node['water_table__elevation'] - \
			env_state.grid.at_node['topographic__elevation'] + \
			env_state.Droot*0.001)*1000		
		SZ_aet = np.where(SZ_aet > 0, SZ_aet,0.0)		
		f = SZ_aet/env_state.Droot
		return f*pet_sz

def cal_alpha_conduc(gridOF,gridSZ, dt, b):
	return 1-dt*gridOF.at_node['SS_loss']/(gridSZ.at_node['SZ_Sy']*b)

def conductance(self,grid):
	act_links = grid.active_links
	Kmax = map_max_of_link_nodes_to_link(grid, 'Mod_Hydraulic_Conductivity')
	Kmin = map_min_of_link_nodes_to_link(grid, 'Mod_Hydraulic_Conductivity')
	Ksl = map_mean_of_link_nodes_to_link(grid,'Mod_Hydraulic_Conductivity')
	Ksat = np.zeros(len(Ksl))
	Ksat[act_links] = Kmax[act_links]*Kmin[act_links]/Ksl[act_links]
	return Ksat

def cal_Ksat_conductance(grid, depth):	
	Ksat = np.where((grid.at_node['water_table__elevation']
		-grid.at_node['topographic__elevation']+depth) >= 0.0,
		grid.at_node['Hydraulic_Conductivity']*np.exp(-0.05/
		(grid.at_node['water_table__elevation']
		-grid.at_node['topographic__elevation']+depth)),
		grid.at_node['Hydraulic_Conductivity']
		)
	grid.at_node['Mod_Hydraulic_Conductivity'][:] = Ksat	
	pass

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
	
	alpha = np.where((np.abs(ruz)-np.abs(dh)) > 0,0,-1)*np.where(dh >= 0,1,-1)*np.where(ruz > 0,0,1)
			
	beta = np.where((np.abs(ruz)-np.abs(dh)) > 0,0,1)*np.where(dh >= 0,1,-1)*np.where(ruz > 0,0,1)
	
	gama = np.where((np.abs(ruz)-np.abs(dh)) > 0,1,0)
	gama = np.where(dh > 0, gama, 1-gama)*np.where(h_aux >= 0,0,1)
	gama = np.where(h_aux >= 0,gama,1)
			
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


def fun_update_soil_rip_depth(env_state, tht_dt, rtht_dt, Droot):
	"""
	Function to update water table depending on both soil and riparian
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
	h0 = (env_state.SZgrid.at_node['water_table__elevation']
		- env_state.SZgrid.at_node['water_storage_anomaly']
		/ env_state.SZgrid.at_node['SZ_Sy'])
	
	tht_dt = np.where(env_state.SZgrid.at_node['water_storage_anomaly'][:] < 0.0,
			env_state.fc, tht_dt)
	
	dtht = env_state.grid.at_node['saturated_water_content'] - tht_dt
		
	dh = np.where((h0-(env_state.grid.at_node['topographic__elevation']-Droot)) > 0.0,
		env_state.SZgrid.at_node['water_storage_anomaly']/dtht,
		env_state.SZgrid.at_node['water_storage_anomaly']/
		env_state.SZgrid.at_node['SZ_Sy'])
		
	h_aux = h0+dh-(env_state.grid.at_node['topographic__elevation']-Droot)
	ruz = (env_state.grid.at_node['topographic__elevation'] - Droot) - h0
	alpha = np.where((np.abs(ruz)-np.abs(dh)) > 0,0,-1)*np.where(dh >= 0,1,-1)*np.where(ruz > 0,0,1)
	beta = np.where((np.abs(ruz)-np.abs(dh)) > 0,0,1)*np.where(dh >= 0,1,-1)*np.where(ruz > 0,0,1)

	gama = np.where((np.abs(ruz)-np.abs(dh)) > 0,1,0)
	gama = np.where(dh > 0, gama, 1-gama)*np.where(h_aux >= 0,0,1)
	gama = np.where(h_aux >= 0,gama,1)
	
	dht = ((env_state.SZgrid.at_node['water_storage_anomaly']
		+ ruz*(alpha*env_state.SZgrid.at_node['SZ_Sy'] + beta*dtht))
		/ ((gama*env_state.SZgrid.at_node['SZ_Sy'] + (1-gama)*dtht)))
		
	env_state.SZgrid.at_node['water_table__elevation'] = h0 + dht

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
	aux = np.where(aux > 0, aux, 0)#(aux+1)*q)
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
	return Ksat*f*np.exp(-(z-h)/f)

# Maximum time step for unconfined aquifers
def time_step(Sy, Ksat, h, zb, dx, *nodes):
	D = 0.25
	T = (h - zb)*Ksat
	dt = D*Sy*np.power(dx,2)/(4*T)
	if nodes:
		dt = np.nanmin((dt[nodes])[dt[nodes] > 0])
	else:
		dt = np.nanmin(dt[dt > 0])
	return dt

# Maximum time step for confined aquifers
def time_step_confined(Sy, T, dx, *nodes):
	D = 0.25
	dt = D*Sy*np.power(dx, 2)/(4*T)	
	if nodes:
		dt = np.nanmin((dt[nodes])[dt[nodes] > 0])
	else:
		dt = np.nanmin(dt[dt > 0])
	return dt