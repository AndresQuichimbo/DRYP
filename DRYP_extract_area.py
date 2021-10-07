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
from components.DRYP_Gen_Func import GlobalTimeVarPts, GlobalTimeVarAvg, GlobalGridVar
from components.DRYP_Gen_Func import save_map_to_rastergrid, check_mass_balance
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
	abc = ABMconnector(data_in, env_state)
	swb = swbm(env_state, data_in)
	# setting model components
	ro = runoff_routing(env_state, data_in)
	inf.exs_dt[:] = 1.0
	ro.run_runoff_one_step(inf, swb, abc.aof, env_state, data_in)
	fname = env_state.fnameTS_OF + 'areas.csv'
	#fOF = pd.read_csv(data_in.fname_DISpoints)
	df = pd.Dataframe()
	df['IDnode'] = env_state.gaugeidOF
	df['Area'] = ro.carea
	df.to_csv(fname)