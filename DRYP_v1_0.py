# -*- coding: utf-8 -*-
"""
DRYP: Dryland WAter Partitioning Model

"""

#sys.modules[_name_]._dict_clear()
import numpy as np
import pandas as pd
from DRYP_io import inputfile, model_environment_status
from DRYP_infiltration import infiltration
from DRYP_rainfall import rainfall
from DRYP_routing import runoff_routing
from DRYP_soil_layer import swbm
from DRYP_Gen_Func import GlobalTimeVarPts, GlobalTimeVarAvg, GlobalGridVar
from DRYP_Gen_Func import save_map_to_rastergrid, check_mass_balance
from DRYP_groundwater_EFD import gwflow_EFD, storage, storage_uz_sz
import DRYP_plot_fun as dryp_plot
import matplotlib.pyplot as plt
from landlab.plot.imshow import imshow_grid
from datetime import datetime, timedelta

# Structure and model components ---------------------------------------
# data_in:	Input variables 
# env_state:Model state and fluxes
# rf:		Precipitation
# inf:		Infiltration 
# swbm:		Soil water balance
# ro:		Routing - Flow accumulator
# gw:		Groundwater flow

# Read input file names
#rootname = 'v322_1km_60m_4p'
local_drive = ''

#local_drive = 'C:/Users/Edisson/Simulation/'
#filename_list = 'Dryland_models_'+rootname+'_MC_list.txt'
filename_list = 'DRYP_model_list.txt'
fmodels = pd.read_csv(filename_list)

nmodel = 23
daily = 1
first_read = 1
kkk = 0
#for nmodel in [24,25,26]:#range(1):
for nmodelk in range(1):
	# reading input variables
	filename_inputs = fmodels['Model'][nmodel]
	data_in = inputfile(filename_inputs)
	
	# setting model fluxes and state variables
	env_state = model_environment_status(data_in)
	env_state.set_output_dir(data_in)
	env_state.points_output(data_in)
	
	# setting model components
	rf = rainfall(data_in, env_state)
	inf = infiltration(env_state, data_in)
	swb = swbm(env_state, data_in)
	swb_rip = swbm(env_state, data_in)
	ro = runoff_routing(env_state, data_in)
	gw = gwflow_EFD(env_state, data_in)
	
	# Output variables and location
	outavg = GlobalTimeVarAvg(env_state.area_catch_factor)
	outpts = GlobalTimeVarPts()
	rip_env_state = GlobalGridVar(env_state.grid)
	state_month = GlobalGridVar(env_state.grid)
		
	t = 0	
	t_eto = 0	
	t_pre = 0
	
	gw_level = []
	pre_mb = []
	gws_mb = []
	dis_mb = []
	rch_mb = []
	aet_mb = []

	rch_agg = np.zeros(len(swb.L_0))
	day = rf.date_sim_dt[0].day+1
	
	while t < rf.t_end:
	
		for UZ_ti in range(data_in.dt_hourly):
			
			for dt_pre_sub in range(data_in.dt_sub_hourly):
				
				swb.run_soil_aquifer_one_step(env_state,
					env_state.grid.at_node['topographic__elevation'],
					env_state.SZgrid.at_node['water_table__elevation'],
					env_state.Duz,
					swb.tht_dt)
															
				env_state.Duz = swb.Duz
				
				rf.run_rainfall_one_step(t_pre, t_eto, env_state, data_in)
					
				inf.run_infiltration_one_step(rf, env_state, data_in)
				
				swb.run_swbm_one_step(inf.inf_dt, rf.PET, env_state.Kc,
					env_state.grid.at_node['Ksat_soil'], env_state, data_in)
				
				env_state.grid.at_node['riv_sat_deficit'][:] *= (swb.tht_dt)
				
				ro.run_runoff_one_step(inf, swb, env_state, data_in)
				
				rip_inf_dt = (inf.inf_dt+ro.tls_dt
					/ np.where(env_state.area_cells_banks <= 0, 1,
					env_state.area_cells_banks)
					)
								
				swb_rip.run_swbm_one_step(rip_inf_dt, rf.PET, env_state.Kc,
						env_state.grid.at_node['Ksat_ch'], env_state,
						data_in, env_state.river_ids_nodes)
				
				rip_env_state.pcl_dt = swb_rip.pcl_dt * env_state.area_cells_banks/env_state.area_cells
				rip_env_state.aet_dt = swb_rip.aet_dt * env_state.area_cells_banks/env_state.area_cells
				swb.PCL_dt = swb.pcl_dt * env_state.area_cells_hills/env_state.area_cells
				swb.AET_dt = swb.aet_dt * env_state.area_cells_hills/env_state.area_cells
				rech = swb.PCL_dt + rip_env_state.pcl_dt # [mm/dt]
				swb.gwe_dt = gw.SZ_potential_ET(env_state,swb.gwe_dt) #[mm/dt]								
				rch_agg += (np.array(rech-swb.gwe_dt)*0.001) #[m/dt]
				
				# Water balance storage and flow
				pre_mb.append(np.sum(rf.rain[env_state.grid.core_nodes]))
				aet_mb.append(np.sum((rip_env_state.aet_dt+swb.AET_dt)[env_state.grid.core_nodes]))
				rch_mb.append(np.sum(env_state.SZgrid.at_node['recharge'][env_state.grid.core_nodes]))				
				
				str_gw_t = storage_uz_sz(env_state)
				#str_gw_t = storage(env_state)
				
				if daily == 1:
					env_state.SZgrid.at_node['discharge'][:] = 0.0
					env_state.SZgrid.at_node['recharge'][:] = (rech-swb.gwe_dt)*0.001
					#gw.run_one_step_gw(env_state,1.0,swb.tht_dt,swb_rip.tht_dt,env_state.Droot*0.001)
					gw.run_one_step_gw_var_T(env_state,1.0,swb.tht_dt,swb_rip.tht_dt,env_state.Droot*0.001,20)
					
				else:
					
					if day == rf.date_sim_dt[t_pre].day:
						env_state.SZgrid.at_node['discharge'][:] = 0.0
						env_state.SZgrid.at_node['recharge'][:] = rch_agg
						#gw.run_one_step_gw(env_state,24.0,swb.tht_dt,swb_rip.tht_dt,env_state.Droot*0.001)
						gw.run_one_step_gw_var_T(env_state,24.0,swb.tht_dt,swb_rip.tht_dt,env_state.Droot*0.001,50)
						rch_agg = np.zeros(len(swb.L_0))
						day = (rf.date_sim_dt[t_pre] + timedelta(days = 1)).day
						
				#str_gw_t = storage_uz_sz(env_state)
				
				gws_mb.append(str_gw_t)				
				dis_mb.append(np.sum(env_state.SZgrid.at_node['discharge'][env_state.grid.core_nodes]))
								
				#state_month.append_grid_var(rf,inf,swb,ro)
				
				#Extract average state and fluxes				
				outavg.extract_avg_var_pre(env_state.basin_nodes,rf)				
				outavg.extract_avg_var_UZ_inf(env_state.basin_nodes,inf)
				outavg.extract_avg_var_UZ_swb(env_state.basin_nodes,swb)
				#outavg.extract_avg_var_UZ_swb(env_state.basin_nodes,swb_rip)
				outavg.extract_avg_var_OF(env_state.basin_nodes,ro)
				outavg.extract_avg_var_SZ(env_state.basin_nodes,gw)
				
				#Extract point state and fluxes
				outpts.extract_point_var_UZ_inf(env_state.gaugeidUZ,inf)
				outpts.extract_point_var_UZ_swb(env_state.gaugeidUZ,swb)
				outpts.extract_point_var_OF(env_state.gaugeidOF,ro)
				outpts.extract_point_var_SZ(env_state.gaugeidGW,gw)
				#outpts.extract_point_var_SZ_L2(env_state.gaugeidGW,env_state.SZgrid)
				
				#outUZ.extract_point_var_UZ(env_state.gaugeidUZ,rf,inf,env_state)
				#outSZ.extract_point_var_SZ(env_state.gaugeidSZ,rf,inf,env_state)
				env_state.L_0 = np.array(swb.L_0)				
				
				t_pre += 1
			t_eto += 1		
		t += 1
				
	
	#fnameCDF = '../Kenya/SR_dataset'+str(nmodel)+'.nc'	
	#state_month.save_monthly_netCDF_var(rf.date_sim_dt,fnameCDF)
	
	outavg.save_avg_var(env_state.fnameTS_avg+'.csv',rf.date_sim_dt)	
	outpts.save_point_var(env_state.fnameTS_OF,rf.date_sim_dt)	
	#outUZ.save_avg_var(env_state.fnameTS_UZ,rf.date_sim_dt)	
	#outSZ.save_avg_var(env_state.fnameTS_SZ,rf.date_sim_dt)	
	
	check_mass_balance(outavg,outpts)
	
	fname_out = env_state.fnameTS_avg + '_wte_ini.asc'	
	save_map_to_rastergrid(env_state.SZgrid, 'water_table__elevation',fname_out)
	
	#dryp_plot.DRYP_plot.plot_avg_var(env_state.fnameTS_avg+'.csv',fname_out)
	
	#fname_out = '../Kenya/SR_gw_level_'+str(nmodel)+'.csv'
	
	df = pd.DataFrame()
	df['pre'] = pre_mb
	df['rch'] = rch_mb
	df['gws'] = gws_mb
	df['dis'] = dis_mb
	df['aet'] = aet_mb
	fname_out = env_state.fnameTS_avg+'_mb.csv'
	#df['head'] = np.array(gw_level)
	
	df.to_csv(fname_out)