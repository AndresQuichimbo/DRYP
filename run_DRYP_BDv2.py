# -*- coding: utf-8 -*-
"""DRYP: Dryland WAter Partitioning Model for sequence input files
"""
import numpy as np
import pandas as pd
from components.DRYP_io import inputfile, model_environment_status
from components.DRYP_infiltration import infiltration
from components.DRYP_rainfall import input_datasets_bigfiles
from components.DRYP_ABM_connector import ABMconnector
from components.DRYP_routing import runoff_routing
from components.DRYP_soil_layer import swbm
from components.DRYP_Gen_Func import (GlobalTimeVarPts,
							GlobalTimeVarAvg, GlobalGridVar,
							save_map_to_rastergrid, check_mass_balance)
from components.DRYP_groundwater_EFD import gwflow_EFD, storage, storage_uz_sz


# Structure and model components ---------------------------------------
# data_in:	Input variables 
# env_state:Model state and fluxes
# rf:		Precipitation
# abc:		Anthropic boundary conditions
# inf:		Infiltration 
# swbm:		Soil water balance
# ro:		Routing - Flow accumulator
# gw:		Groundwater flow
#@profile
def run_DRYP(filename_input):

	data_in = inputfile(filename_input)
	
	# setting model fluxes and state variables
	env_state = model_environment_status(data_in)
	env_state.set_output_dir(data_in)
	env_state.points_output(data_in)
	
	# setting model components
	rf = input_datasets_bigfiles(data_in, env_state)
	abc = ABMconnector(data_in, env_state)
	inf = infiltration(env_state, data_in)
	swb = swbm(env_state, data_in)
	swb_rip = swbm(env_state, data_in)
	ro = runoff_routing(env_state, data_in)
	gw = gwflow_EFD(env_state, data_in)
	
	# Output variables and location
	outavg = GlobalTimeVarAvg(env_state.area_catch_factor)
	outavg_rip = GlobalTimeVarAvg(env_state.area_river_factor)
	outpts = GlobalTimeVarPts()
	state_var = GlobalGridVar(env_state, data_in)
	
	t = 0	
	t_eto = 0	
	t_pre = 0
	
	gw_level = []
	pre_mb = []
	exs_mb = []
	tls_mb = []			
	gws_mb = []
	uzs_mb = []
	dis_mb = []
	rch_mb = []
	aet_mb = []
	egw_mb = []

	rch_agg = np.zeros(len(swb.L_0))
	etg_agg = np.zeros(len(swb.L_0))
	dt_GW = np.int(data_in.dt)
	
	while t < rf.t_end:
	
		for UZ_ti in range(data_in.dt_hourly):
			
			for dt_pre_sub in range(data_in.dt_sub_hourly):
				#print('out',np.argwhere(np.isnan(env_state.SZgrid.at_node['water_table__elevation'])))
				swb.run_soil_aquifer_one_step(env_state,
					env_state.grid.at_node['topographic__elevation'],
					env_state.SZgrid.at_node['water_table__elevation'],
					env_state.Duz,
					swb.tht_dt)
				
				env_state.Duz = swb.Duz
				
				rf.run_dataset_one_step(t_pre, env_state, data_in)
				
				abc.run_ABM_one_step(t_pre, env_state,
					rf.rain, env_state.Duz, swb.tht_dt, env_state.fc,
					env_state.grid.at_node['wilting_point'],
					env_state.SZgrid.at_node['water_table__elevation'],
					)
					
				rf.rain += abc.auz
				
				inf.run_infiltration_one_step(rf, env_state, data_in)				
				
				aux_usz = np.sum((swb.L_0*env_state.hill_factor)[env_state.act_nodes])
				aux_usp = np.sum((swb_rip.L_0*env_state.riv_factor)[env_state.act_nodes])
				
				swb.run_swbm_one_step(inf.inf_dt, rf.PET, env_state.Kc,
					env_state.grid.at_node['Ksat_soil'], env_state, data_in)
				
				env_state.grid.at_node['riv_sat_deficit'][:] *= (
					env_state.grid.at_node['saturated_water_content'][:]
					-swb_rip.tht_dt)
				
				ro.run_runoff_one_step(inf, swb, abc.aof, env_state, data_in)
				
				tls_aux = ro.tls_flow_dt*env_state.rip_factor
				
				rip_inf_dt = inf.inf_dt + tls_aux									 
				#print('in',swb.Duz[np.argwhere(np.isnan(env_state.SZgrid.at_node['water_table__elevation']))])															
				swb_rip.run_swbm_one_step(rip_inf_dt, rf.PET, env_state.Kc,
						env_state.grid.at_node['Ksat_ch'], env_state,
						data_in, env_state.river_ids_nodes)
	  
				swb_rip.pcl_dt *= env_state.riv_factor
				swb_rip.aet_dt *= env_state.riv_factor
				swb.pcl_dt *= env_state.hill_factor
				swb.aet_dt *= env_state.hill_factor
				rech = swb.pcl_dt + swb_rip.pcl_dt - abc.asz# [mm/dt]
				etg_dt = gw.SZ_potential_ET(env_state, swb.gwe_dt)									  
				etg_agg += np.array(etg_dt) # [mm/h]											
				rch_agg += np.array(rech) # [mm/dt]
				
				# Water balance storage and flow
				pre_mb.append(np.sum(rf.rain[env_state.act_nodes]))
				exs_mb.append(np.sum(inf.exs_dt[env_state.act_nodes]))
				tls_mb.append(np.sum(tls_aux[env_state.act_nodes]))
				aux_usz1 = np.sum((swb.L_0*env_state.hill_factor)[env_state.act_nodes])
				aux_usp1 = np.sum((swb_rip.L_0*env_state.riv_factor)[env_state.act_nodes])
				
				uzs_mb.append(aux_usp1+aux_usz1-aux_usp-aux_usz)
				aet_mb.append(np.sum((swb_rip.aet_dt+swb.aet_dt)[env_state.act_nodes]))
				egw_mb.append(np.sum(etg_dt[env_state.act_nodes]))
				rch_mb.append(np.sum(rech[env_state.act_nodes]))				
				#aux_ssz = storage_uz_sz(env_state)				
				
				if dt_GW == data_in.dtSZ:
					# Change units to m/h
					env_state.SZgrid.at_node['discharge'][:] = 0.0
					env_state.SZgrid.at_node['recharge'][:] = (rch_agg - etg_agg)*0.001 #[m/dt]
					gw.recharge(env_state, data_in.dtSZ/60)
					gw.run_one_step_gw(env_state, data_in.dtSZ/60, swb.tht_dt,
							env_state.Droot*0.001)
					rch_agg = np.zeros_like(swb.L_0)
					etg_agg = np.zeros_like(swb.L_0)
		  
					dt_GW = 0
				
				dt_GW += np.int(data_in.dt)				
				
				gws_mb.append(storage_uz_sz(env_state, np.array(swb.tht_dt), gw.dh))#-aux_ssz)
				dis_mb.append(np.sum(env_state.SZgrid.at_node['discharge'][env_state.act_nodes])-gw.flux_out)								   
				
				#Extract average state and fluxes				
				outavg.extract_avg_var_pre(env_state.basin_nodes,rf)				
				outavg.extract_avg_var_UZ_inf(env_state.basin_nodes,inf)
				outavg.extract_avg_var_UZ_swb(env_state.basin_nodes,swb)
				outavg_rip.extract_avg_var_UZ_swb(env_state.basin_nodes,swb_rip)
				outavg.extract_avg_var_OF(env_state.basin_nodes,ro)
				outavg.extract_avg_var_SZ(env_state.basin_nodes,gw)
				
				#Extract states and fluxes at point locations
				outpts.extract_point_var_UZ_inf(env_state.gaugeidUZ,inf)
				outpts.extract_point_var_UZ_swb(env_state.gaugeidUZ,swb)
				outpts.extract_point_var_OF(env_state.gaugeidOF,ro)
				outpts.extract_point_var_SZ(env_state.gaugeidGW,gw)
				state_var.get_env_state(t_pre, rf, inf, swb,
									ro, gw, swb_rip, env_state)							
								
				env_state.L_0 = np.array(swb.L_0)				
				t_pre += 1
			t_eto += 1		
		t += 1
	
	mb = [pre_mb, exs_mb, tls_mb, rch_mb, gws_mb,
		uzs_mb, dis_mb, aet_mb, egw_mb]
	
	outavg.save_avg_var(env_state.fnameTS_avg+'.csv', rf.date_sim_dt)
	outavg_rip.save_avg_var(env_state.fnameTS_avg+'rip.csv', rf.date_sim_dt)
	outpts.save_point_var(env_state.fnameTS_OF, rf.date_sim_dt,
			ro.carea[env_state.gaugeidOF],
			env_state.rarea[env_state.gaugeidOF])	
	state_var.save_netCDF_var(env_state.fnameTS_avg+'.nc')
	check_mass_balance(env_state.fnameTS_avg, outavg, outpts,
			outavg_rip, mb, rf.date_sim_dt,
			ro.carea[env_state.gaugeidOF[0]])
	
	# Save water table for initial conditions
	fname_out = env_state.fnameTS_avg + '_wte_ini.asc'	
	save_map_to_rastergrid(env_state.SZgrid, 'water_table__elevation', fname_out)

if __name__ == '__main__':
	run_DRYP(filename_input)