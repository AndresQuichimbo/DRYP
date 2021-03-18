import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date, date2num
from datetime import datetime
# Global parameters
ABC_RIVER = 0.2 # River abstraction parameter

class ABMconnector(object):
	def __init__(self, inputfile, env_state):
		if inputfile:
			# here add any additional parameter required to run ABM
			# which are constants over the entire simulation
			# a function can be imported
			# grids can be added here if necesary:
			# for surface grids:
			# env_state.grid.add_zeros('node', 'name_variable', dtype=float)
			# e.g. maximun fraction of streamflow taken from river,
			# for now it is to avoid streams to become dry
			env_state.grid.at_node['AOFT'][:] = ABC_RIVER
			# For groundwater grids:
			# env_state.SZgrid.add_zeros('node', 'name_variable', dtype=float)
						 
			print('Add any additional parameter for running ABM')
			self.aof = np.zeros(env_state.grid_size)
			self.auz = np.zeros(env_state.grid_size)
			self.asz = np.zeros(env_state.grid_size)
			self.data_provided = 1 # Change to one for running ABM-conector
		else:
			self.data_provided = 0
			print('Not available water extractions file')
		self.netcdf_file = int(inputfile.netcf_ABC)
	# find water abstraction for specific time steps
	def run_ABM_one_step(self, i_ABM, env_state, rain, Duz, theta, theta_fc, theta_wp, wte):
		"""
		Call this to execute a step in the model.
				
		Parameters:
			i_ABM:		it is counter in case it is needed
			env_state:	all state parameters of the model
			rain:		to check if irrigation is needed (mm)
			Duz:		Soil depth (mm)
			theta:		water content (-)
			theta_fc:	water at field capacity (-)
			theta_wp:	water at wilting point (-)
		
		Outputs:
			aof:	fluw rate of stream flow abstracted [m3/dt] (*dt is model timestep)
			auz:	fluw rate of irrigation [mm/dt]
			asz:	fluw rate of groundwater abstraction [mm/dt]
		"""
		if self.data_provided == 0:
			self.aof = np.zeros(env_state.grid_size)
			self.auz = np.zeros(env_state.grid_size)
			self.asz = np.zeros(env_state.grid_size)
		else:
			# add any additional function to modifed any axx states
			# or import a function
			
			# Soil moisture deficit
			#dtheta = (theta_fc-theta)
			#dtheta[dtheta < 0] = 0
			#SMD = Duz*dtheta
			
			#aof, auz, asz = ABM()
			#to pass variables to other components
			#self.aof = aof
			#self.auz = auz
			#self.asz = asz
			self.data_provided = 0		
		
	def work_out_stable_timestep(self,):
		"""
		Something like this might be needed
		"""
		pass

def ABM():
	pass #aof, auz, asz