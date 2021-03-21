# -*- coding: utf-8 -*-
"""
DRYP: Dryland WAter Partitioning Model for sequence input files
"""
import numpy as np
import pandas as pd
from components.DRYP_io import inputfile, model_environment_status
from components.DRYP_infiltration import infiltration
from components.DRYP_rainfall import input_datasets_bigfiles
from components.DRYP_ABM_connector import ABMconnector
from components.DRYP_routing import runoff_routing
from components.DRYP_soil_layer import swbm
from components.DRYP_Gen_Func import GlobalTimeVarPts, GlobalTimeVarAvg, GlobalGridVar
from components.DRYP_Gen_Func import save_map_to_rastergrid, check_mass_balance
from components.DRYP_groundwater_EFD import gwflow_EFD, storage, storage_uz_sz
import components.DRYP_plot_fun as dryp_plot
import matplotlib.pyplot as plt
from landlab.plot.imshow import imshow_grid
from datetime import datetime, timedelta

# Structure and model components ---------------------------------------
# data_in:	Input variables 
# env_state:Model state and fluxes
# rf:		Precipitation
# abc:		Anthropic boundary conditions
# inf:		Infiltration 
# swbm:		Soil water balance
# ro:		Routing - Flow accumulator
# gw:		Groundwater flow

def run_DRYP(filename_input):

	data_in = inputfile(filename_input)
	daily = 1
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
	rip_env_state = GlobalGridVar(env_state.grid)
	state_month = GlobalGridVar(env_state.grid)
	
	t = 0	
	t_eto = 0	
	t_pre = 0
	
	gw_level = []
	pre_mb = []
	gws_mb = []
	uzs_mb = []
	dis_mb = []
	rch_mb = []
	aet_mb = []

	rch_agg = np.zeros(len(swb.L_0))
	dt_GW = np.int(data_in.dtUZ)
	
	while t < rf.t_end:
	
		for UZ_ti in range(data_in.dt_hourly):
			
			for dt_pre_sub in range(data_in.dt_sub_hourly):
				
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
				
				swb.run_swbm_one_step(inf.inf_dt, rf.PET, env_state.Kc,
					env_state.grid.at_node['Ksat_soil'], env_state, data_in)
				
				env_state.grid.at_node['riv_sat_deficit'][:] *= (swb.tht_dt)
				
				ro.run_runoff_one_step(inf, swb, abc.aof, env_state, data_in)
				
				rip_inf_dt = (inf.inf_dt+ro.tls_flow_dt*1000
					/ np.where(env_state.area_cells_banks <= 0, 1,
					env_state.area_cells_banks)
					)
								
				swb_rip.run_swbm_one_step(rip_inf_dt, rf.PET, env_state.Kc,
						env_state.grid.at_node['Ksat_ch'], env_state,
						data_in, env_state.river_ids_nodes)
				swb_rip.pcl_dt *= env_state.area_cells_banks/env_state.area_cells
				swb_rip.aet_dt *= env_state.area_cells_banks/env_state.area_cells
				rip_env_state.pcl_dt = swb_rip.pcl_dt
				rip_env_state.aet_dt = swb_rip.aet_dt
				swb.PCL_dt = swb.pcl_dt * env_state.area_cells_hills/env_state.area_cells
				swb.AET_dt = swb.aet_dt * env_state.area_cells_hills/env_state.area_cells
				rech = swb.PCL_dt + rip_env_state.pcl_dt - abc.asz# [mm/dt]
				swb.gwe_dt = gw.SZ_potential_ET(env_state, swb.gwe_dt) #[mm/dt]								
				rch_agg += (np.array(rech-swb.gwe_dt)*0.001) #[m/dt]
				
				# Water balance storage and flow
				str_gw_t = storage_uz_sz(env_state)
				pre_mb.append(np.sum(rf.rain[env_state.grid.core_nodes]))
				uzs_mb.append(np.sum(swb.L_0[env_state.grid.core_nodes]))
				aet_mb.append(np.sum((rip_env_state.aet_dt+swb.AET_dt)[env_state.grid.core_nodes]))
				rch_mb.append(np.sum(env_state.SZgrid.at_node['recharge'][env_state.grid.core_nodes]))				
				dis_mb.append(np.sum(env_state.SZgrid.at_node['discharge'][env_state.grid.core_nodes]))
				gws_mb.append(str_gw_t)				
				
				if dt_GW == data_in.dtSZ:
					env_state.SZgrid.at_node['discharge'][:] = 0.0
					env_state.SZgrid.at_node['recharge'][:] = rch_agg
					#gw.run_one_step_gw(env_state,24.0,swb.tht_dt,swb_rip.tht_dt,env_state.Droot*0.001)
					gw.run_one_step_gw_var_T(env_state,data_in.dtSZ/60,swb.tht_dt,swb_rip.tht_dt,env_state.Droot*0.001,50)
					rch_agg = np.zeros(len(swb.L_0))
					dt_GW = 0
					
				dt_GW += np.int(data_in.dtUZ)				
				
				#Extract average state and fluxes				
				outavg.extract_avg_var_pre(env_state.basin_nodes,rf)				
				outavg.extract_avg_var_UZ_inf(env_state.basin_nodes,inf)
				outavg.extract_avg_var_UZ_swb(env_state.basin_nodes,swb)
				outavg_rip.extract_avg_var_UZ_swb(env_state.basin_nodes,swb_rip)
				outavg.extract_avg_var_OF(env_state.basin_nodes,ro)
				outavg.extract_avg_var_SZ(env_state.basin_nodes,gw)
				
				#Extract point state and fluxes
				outpts.extract_point_var_UZ_inf(env_state.gaugeidUZ,inf)
				outpts.extract_point_var_UZ_swb(env_state.gaugeidUZ,swb)
				outpts.extract_point_var_OF(env_state.gaugeidOF,ro)
				outpts.extract_point_var_SZ(env_state.gaugeidGW,gw)
				
				env_state.L_0 = np.array(swb.L_0)				
				t_pre += 1
			t_eto += 1		
		t += 1
				
	outavg.save_avg_var(env_state.fnameTS_avg+'.csv', rf.date_sim_dt)
	outavg_rip.save_avg_var(env_state.fnameTS_avg+'rip.csv', rf.date_sim_dt)
	outpts.save_point_var(env_state.fnameTS_OF, rf.date_sim_dt)	
	
	check_mass_balance(outavg, outpts, outavg_rip)
	
	fname_out = env_state.fnameTS_avg + '_wte_ini.asc'	
	save_map_to_rastergrid(env_state.SZgrid, 'water_table__elevation', fname_out)
	
	df = pd.DataFrame()
	df['Date'] = rf.date_sim_dt
	df['pre'] = pre_mb
	df['rch'] = rch_mb
	df['gws'] = gws_mb
	df['uzs'] = uzs_mb
	df['dis'] = dis_mb
	df['aet'] = aet_mb
	fname_out = env_state.fnameTS_avg+'_mb.csv'
	df.to_csv(fname_out)

if __name__ == '__main__':
	run_DRYP(filename_input)