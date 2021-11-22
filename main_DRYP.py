# -*- coding: utf-8 -*-
"""
DRYP: Dryland WAter Partitioning Model
"""
import numpy as np
import pandas as pd
from components.DRYP_io import inputfile, model_environment_status
from components.DRYP_infiltration import infiltration
from components.DRYP_rainfall import rainfall
from components.DRYP_rainfall import input_datasets_bigfiles
from components.DRYP_ABM_connector import ABMconnector
#from components.DRYP_routing import runoff_routing
#from components.DRYP_flow_accum import runoff_routing
from components.DRYP_flow_accumf90 import runoff_routing
from components.DRYP_soil_layer import swbm
from components.DRYP_groundwater_EFD import (
	gwflow_EFD,	storage, storage_uz_sz)
from components.DRYP_Gen_Func import (
	GlobalTimeVarPts, GlobalTimeVarAvg, GlobalGridVar,
	save_map_to_rastergrid, check_mass_balance)


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
	
	# read model paramters and model setting file
	data_in = inputfile(filename_input)
	
	# setting model fluxes and state variables
	env_state = model_environment_status(data_in)
	env_state.set_output_dir(data_in)
	env_state.points_output(data_in)
	
	# setting model components
	if data_in.nfiles == 0:
		rf = rainfall(data_in, env_state)
	else:
		rf = input_datasets_bigfiles(data_in, env_state)
	abc = ABMconnector(data_in, env_state)
	inf = infiltration(env_state, data_in)
	swb = swbm(env_state, env_state.Duz, env_state.tht, data_in)
	swb_rip = swbm(env_state, env_state.Droot, env_state.ptht, data_in)
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
		
	pre_mb = []
	exs_mb = []
	tls_mb = []
	gws_mb = []
	uzs_mb = []
	gbf_mb = []
	rch_mb = []
	aet_mb = []
	egw_mb = []
	chs_mb = []
	rzs_mb = []
	
	rch_agg = np.zeros(len(swb.L_0))
	etg_agg = np.zeros(len(swb.L_0))
	dt_GW = np.int(data_in.dt)
	
	while t < rf.t_end:
	
		for UZ_ti in range(data_in.dt_hourly):
			
			for dt_pre_sub in range(data_in.dt_sub_hourly):
				
				# get rainfall				
				if data_in.nfiles == 0:
					rf.run_rainfall_one_step(t_pre, t_eto,
							env_state, data_in)
				else:
					rf.run_dataset_one_step(t_pre,
							env_state, data_in)
				
				# estimate abstractions
				abc.run_ABM_one_step(t_pre, env_state,
					rf.rain, env_state.Duz, swb.tht_dt, env_state.fc,
					env_state.grid.at_node['wilting_point'],
					env_state.SZgrid.at_node['water_table__elevation'],
					)
				
				# add abstraction as rain
				rf.rain += abc.auz
				
				# estimate infiltration
				inf.run_infiltration_one_step(rf, env_state, data_in)
				
				# subsurface storage [mm]
				if data_in.run_GW == 1:
					aux_ssz = storage_uz_sz(env_state, np.array(swb.tht_dt))
				
				# soil storage at time t0 [mm]
				aux_usz = np.mean((swb.L_0[env_state.act_nodes]))
						#* env_state.hill_factor)[env_state.act_nodes])
				
				# riparian storage at time t0[mm]			
				aux_usp = np.mean((swb_rip.L_0
						* env_state.riv_factor)[env_state.act_nodes])
				
				# estimate soil water balance
				swb.run_swbm_one_step(inf.inf_dt, rf.PET, env_state.Kc,
					env_state.grid.at_node['Ksat_soil'],
					env_state, data_in)
				
				# calculate riparial pet
				rpet_dt = rf.PET - swb.aet_dt
				#print('aa', swb_rip.smd_uz)
				# estimate available storage ar riparian zone
				# change discharge from m to mm per unit rip. area
				smd, qriv, inf_rip_dt = swb_rip.water_deficit(
					(env_state.SZgrid.at_node['discharge']*1000*
					env_state.inv_riv_factor),
					rpet_dt)
				
				env_state.grid.at_node['riv_sat_deficit'][:] += (
					smd*env_state.rarea)
				#env_state.grid.at_node['riv_sat_deficit'][:] *= np.array(
				#	swb_rip.tht_dt)
				
				# update discharge
				env_state.SZgrid.at_node['discharge'][env_state.riv_nodes] = (
					env_state.SZgrid.at_node['discharge'][env_state.riv_nodes]*
					qriv[env_state.riv_nodes]*
					env_state.riv_factor[env_state.riv_nodes]*0.001)
				#print('a',env_state.SZgrid.at_node['discharge'][env_state.riv_nodes])
				# estimate runoff
				ro.run_runoff_one_step(inf, swb, abc.aof, env_state, data_in)
				
				# change transmission losses to riparian area
				tls_aux = ro.tls_flow_dt*env_state.rip_factor
				#print('a')
				# estimate inputs to riparian zone [mm]
				rip_inf_dt = tls_aux + inf_rip_dt
				#print('a', inf_rip_dt[env_state.riv_nodes])
				# estimate riparian water balance
				swb_rip.run_swbm_one_step(rip_inf_dt, rpet_dt, env_state.Kc,
						env_state.grid.at_node['Ksat_ch'], env_state,
						data_in, env_state.river_ids_nodes)
				#print('b')#,env_state.SZgrid.at_node['discharge'][env_state.riv_nodes])
				# change riparian fluxes to cell area
				swb_rip.pcl_dt *= env_state.riv_factor
				swb_rip.aet_dt *= env_state.riv_factor
				
				# correct hill slop fluxes to grid cells
				#swb.pcl_dt *= env_state.hill_factor
				#swb.aet_dt *= env_state.hill_factor
				
				# estimate total groundwater recharge
				rech = swb.pcl_dt + swb_rip.pcl_dt - abc.asz# [mm/dt]
				
				# estimate capillary rise
				gwe_dt = rf.PET - (swb.aet_dt + swb_rip.aet_dt)
				gwe_dt[gwe_dt < 0] = 0
				etg_dt = gw.SZ_potential_ET(env_state, gwe_dt)
				
				# temporal aggregation of fluxes for groundwater
				etg_agg += np.array(etg_dt) # [mm/h]
				rch_agg += np.array(rech) # [mm/dt]
				
				# save total catchment fluxes for water balance
				pre_mb.append(np.mean(rf.rain[env_state.act_nodes]))
				exs_mb.append(np.mean(inf.exs_dt[env_state.act_nodes]))
				tls_mb.append(np.mean(ro.tls_dt[env_state.act_nodes]))
				
				# save soil and riparian AET for water balance
				aet_mb.append(np.mean((swb_rip.aet_dt
					+swb.aet_dt)[env_state.act_nodes]))
				egw_mb.append(np.mean(etg_dt[env_state.act_nodes]))
				
				# save diffuse and focussed recharge for water balance
				rch_mb.append(np.mean(rech[env_state.act_nodes]))
				
				# soil storage at time t1 [mm]
				aux_usz1 = np.mean((swb.L_0[env_state.act_nodes]))
					#*env_state.hill_factor)[env_state.act_nodes])
				
				# riparian storage at time t1 [mm]
				aux_usp1 = np.mean((swb_rip.L_0
					*env_state.riv_factor)[env_state.act_nodes])
				
				# channel storage at delta time [mm]
				chs_mb.append(np.mean(ro.qfl_dt[env_state.act_nodes]))
				
				# change in soil storage at delta time
				uzs_mb.append(aux_usz1-aux_usz) #[mm]
				
				# change in riparian storage at delta time
				rzs_mb.append(aux_usp1-aux_usp) #[mm]
				
				# activate groundwater component (gw)
				if data_in.run_GW == 1:
					if dt_GW == data_in.dtSZ:
						# empty discharge array
						env_state.SZgrid.at_node['discharge'][:] = 0.0
						
						# estimate and change recharge units [mm/h --> m/h]
						env_state.SZgrid.at_node['recharge'][:] = (
								rch_agg - etg_agg)*0.001 #[mm/dt]
						
						# run groundwater component
						gw.run_one_step_gw(env_state, data_in.dtSZ/60,
							swb.tht_dt,	env_state.Droot*0.001)
						
						# empty array
						rch_agg = np.zeros(len(swb.L_0))
						etg_agg = np.zeros(len(swb.L_0))
						dt_GW = 0
					
					# time accumulator for gw	
					dt_GW += np.int(data_in.dt)
				
				# update soil moisture
				if data_in.run_GW == 1:
					swb.run_soil_aquifer_one_step(env_state,
						env_state.grid.at_node['topographic__elevation'],
						env_state.SZgrid.at_node['water_table__elevation'],
						env_state.Duz,
						swb.tht_dt)
				
				# update rooting depth
				if data_in.run_GW == 1:
					env_state.Duz = swb.Duz
				
				# estimate groundwater storage change for delta t
				if data_in.run_GW == 1:
					gws_mb.append(storage_uz_sz(env_state,
							np.array(swb.tht_dt))
							-aux_ssz)
				else:
					gws_mb.append(0.0)
				
				# save groundwater discharge at outlet for water balance
				gbf_mb.append(np.mean(
					env_state.SZgrid.at_node['discharge'][env_state.act_nodes]*1000)
					+ gw.flux_out)
				
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
				state_var.get_env_state(t_pre, rf, inf, swb,
									ro, gw, swb_rip, env_state)
				
				# update soil water content for next iteration
				env_state.L_0 = np.array(swb.L_0)
				t_pre += 1
			t_eto += 1		
		t += 1
	
	mb = [pre_mb, exs_mb, tls_mb, rch_mb, gws_mb,
		uzs_mb, gbf_mb, aet_mb, egw_mb, chs_mb, rzs_mb]	
	
	# save catchment and riparian average results
	outavg.save_avg_var(env_state.fnameTS_avg+'.csv',
		rf.date_sim_dt)
	outavg_rip.save_avg_var(env_state.fnameTS_avg+'rip.csv',
		rf.date_sim_dt)
	
	# save point results
	outpts.save_point_var(env_state.fnameTS_OF, rf.date_sim_dt,
			ro.carea[env_state.gaugeidOF],
			env_state.rarea[env_state.gaugeidOF])	
	
	# save grided model result datasets 
	state_var.save_netCDF_var(env_state.fnameTS_avg+'.nc')
	
	# check mass balance
	check_mass_balance(env_state.fnameTS_avg, outavg, outpts,
			outavg_rip, mb, rf.date_sim_dt,
			ro.carea[env_state.gaugeidOF[0]])
	
	# Save water table for initial conditions
	save_map_to_rastergrid(env_state.SZgrid,
			'water_table__elevation',
			env_state.fnameTS_avg + '_wte_ini.asc')
	
	# Save soil moisture for initial conditions
	save_map_to_rastergrid(env_state.grid,
			'Soil_Moisture',
			env_state.fnameTS_avg + '_tht_ini.asc')
	
if __name__ == '__main__':
	run_DRYP(filename_input)