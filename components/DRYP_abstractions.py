import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date, date2num
from datetime import datetime

class abstractions(object):
	def __init__(self, inputfile, env_state):
		if inputfile.first_read == 1:
			t_end = inputfile.Sim_period.days
			if inputfile.dtUZ_pre != 60:
				date_sim_m = pd.date_range(inputfile.ini_date, periods = t_end*inputfile.dt_hourly*inputfile.dt_sub_hourly, freq = str(np.int(inputfile.dtUZ_pre))+'min')
			freq_dt = 'H'
			if inputfile.dtUZ_pre > 60:
				freq_dt = str(np.int(inputfile.dtUZ_pre/60))+'H'
			date_sim_h = pd.date_range(inputfile.ini_date, periods = t_end*inputfile.dt_hourly, freq = freq_dt)
			date_sim_d = pd.date_range(inputfile.ini_date, periods = t_end, freq = 'd')
			if t_end <= 0:
				sys.exit("End of the simulation period should be later than initial date")
				
		if os.path.exists(inputfile.fname_TSABC):
			if inputfile.netcf_ABC == 1:
				fabc = Dataset(inputfile.fname_TSABC, 'r')
				time_pre_aux = num2date(fabc['time'][:-1], units = fabc['time'].units, calendar = fabc['time'].calendar)
				time_pre = []
				for iidate in time_pre_aux:
					if not iidate == None:
						time_pre.append(datetime(iidate.year,iidate.month,iidate.day,iidate.hour,iidate.minute))
					else:
						time_pre.append(None)
				time_pre = np.array(time_pre)
			else:# Read time series of water abstractions	
				fabc = pd.read_csv(inputfile.fname_TSABC)
				fabc["Date"] = pd.to_datetime(fabc['Date'])#,format = '%d/%m/%Y %H:%M')
				time_pre = fabc["Date"]
						
			# Time periods-----------------------------------------------------------------------
			idate_aux = np.where((time_pre < inputfile.end_datet) & (time_pre >= inputfile.ini_date))[0]
			if inputfile.dtUZ_pre != 60:
				idateabc = np.zeros(len(date_sim_m))
				idate_pre = date_sim_m.isin(time_pre)
			else:
				idateabc = np.zeros(len(date_sim_h))
				idate_pre = date_sim_h.isin(time_pre)
			idateabc[idate_pre == True] = idate_aux
			idateabc[idate_pre == False] = np.nan
			if inputfile.netcf_ABC == 1:
				t_end = (time_pre[-1] - inputfile.ini_date).days
			else:
				t_end = (time_pre.iloc[-1] - inputfile.ini_date).days
			date_sim_h = pd.date_range(inputfile.ini_date, periods = t_end*24, freq = 'H')
			date_sim_d = pd.date_range(inputfile.ini_date, periods = t_end, freq = 'd')	
			#self.date_sim_d = date_sim_d
			#self.date_sim_h = date_sim_h
			#self.date_sim_dt = pd.Series(pd.date_range(inputfile.ini_date, periods = t_end*inputfile.dt_hourly*inputfile.dt_sub_hourly, freq = inputfile.Agg_method))
			self.fabc = fabc
			self.idateabc = idateabc
			#self.t_end = t_end
			self.data_provided = 1
		else:
			self.data_provided = 0
			print('Not available water extractions file')
		self.netcdf_file = int(inputfile.netcf_ABC)
	# find water abstraction for specific time steps
	def run_abstractions_one_step(self, t_abc, env_state, inputfile):
		"""
		Call this to execute a step in the model.
		"""
		if self.data_provided == 0:
			self.aof = np.zeros(env_state.grid_size)
			self.auz = np.zeros(env_state.grid_size)
			self.asz = np.zeros(env_state.grid_size)
		else:
			if not np.isnan(self.idateabc[t_abc]):
				if self.netcdf_file == 1:
					self.aof = (self.fabc.variables['AOF'][self.idateabc[t_abc]][:]).T.flatten()
					self.auz = (self.fabc.variables['AUZ'][self.idateabc[t_abc]][:]).T.flatten()
					self.asz = (self.fabc.variables['ASZ'][self.idateabc[t_abc]][:]).T.flatten()
				else: # Uniform precipitation over the whole catchement
					self.aof = np.ones(env_state.grid_size)*self.fabc['AOF'][self.idateabc[t_abc]]
					self.auz = np.ones(env_state.grid_size)*self.fabc['AUZ'][self.idateabc[t_abc]]
					self.asz = np.ones(env_state.grid_size)*self.fabc['ASZ'][self.idateabc[t_abc]]
			self.rain_day_before = 1
				
	
	def work_out_stable_timestep(self,):
		"""
		Something like this might be needed
		"""
		pass
