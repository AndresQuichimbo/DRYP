import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date, date2num
from datetime import datetime, timedelta

class rainfall(object):
	"""Read input dataset
	"""
	def __init__(self, inputfile,env_state):
		if inputfile.first_read == 1:
			t_end = inputfile.Sim_period.days
			if inputfile.dt != 60:
				date_sim_m = pd.date_range(inputfile.ini_date, periods = t_end*inputfile.dt_hourly*inputfile.dt_sub_hourly, freq = str(np.int(inputfile.dt))+'min')
			freq_dt = 'H'
			if inputfile.dt > 60:
				freq_dt = str(np.int(inputfile.dt/60))+'H'
			date_sim_h = pd.date_range(inputfile.ini_date, periods = t_end*inputfile.dt_hourly, freq = freq_dt)
			date_sim_d = pd.date_range(inputfile.ini_date, periods = t_end, freq = 'd')
			if t_end <= 0:
				sys.exit("End of the simulation period should be later than initial date")
				
		if inputfile.first_read == 1:
			if inputfile.netcf_pre == 1:
				fpre = Dataset(inputfile.fname_TSPre, 'r')
				time_pre_aux = num2date(fpre['time'][:-1], units = fpre['time'].units, calendar = fpre['time'].calendar)
				time_pre = []
				for iidate in time_pre_aux:
					if not iidate == None:
						time_pre.append(datetime(iidate.year,iidate.month,iidate.day,iidate.hour,iidate.minute))
					else:
						time_pre.append(None)
				time_pre = np.array(time_pre)
			else:# Read time series of precipitation	
				fpre = pd.read_csv(inputfile.fname_TSPre)
				fpre["Date"] = pd.to_datetime(fpre['Date'])#,format = '%d/%m/%Y %H:%M')
				time_pre = fpre["Date"]
			
			# Read time series of evapotranspiration
			if inputfile.netcf_ETo == 1:
				dataETo = Dataset(inputfile.fname_TSMeteo, 'r')
				time_ETo_aux = num2date(dataETo['time'][:-1], units = dataETo['time'].units, calendar = dataETo['time'].calendar)
				time_ETo = []
				for iidate in time_ETo_aux:
					if not iidate == None:
						time_ETo.append(datetime(iidate.year,iidate.month,iidate.day,iidate.hour,iidate.minute))
					else:
						time_ETo.append(None)
				time_ETo = np.array(time_ETo)
			else:
				dataETo = pd.read_csv(inputfile.fname_TSMeteo)
				dataETo["Date"] = pd.to_datetime(dataETo['Date'])
				if inputfile.dt > 60:
					dataETo.index = pd.DatetimeIndex(dataETo['Date'])
					dataETo = (dataETo.resample(inputfile.Agg_method).sum()).reset_index()
				time_ETo = dataETo["Date"]
			
			# Time periods-----------------------------------------------------------------------
			idate_aux = np.where((time_pre < inputfile.end_datet) & (time_pre >= inputfile.ini_date))[0]
			if inputfile.dt != 60:
				idatepre = np.zeros(len(date_sim_m))
				idate_pre = date_sim_m.isin(time_pre)
			else:
				idatepre = np.zeros(len(date_sim_h))
				idate_pre = date_sim_h.isin(time_pre)
			idatepre[idate_pre == True] = idate_aux
			idatepre[idate_pre == False] = np.nan
			idateETo = np.where((time_ETo < inputfile.end_datet) & (time_ETo >= inputfile.ini_date))[0]
			if len(idateETo) < t_end*inputfile.dt_hourly:
				if inputfile.netcf_pre == 1:
					t_end = (time_ETo[-1] - inputfile.ini_date).days
				else:
					t_end = (time_ETo.iloc[-1] - inputfile.ini_date).days
				date_sim_h = pd.date_range(inputfile.ini_date, periods = t_end*24, freq = 'H')
				date_sim_d = pd.date_range(inputfile.ini_date, periods = t_end, freq = 'd')	
		self.date_sim_d = date_sim_d
		self.date_sim_h = date_sim_h
		self.date_sim_dt = pd.Series(pd.date_range(inputfile.ini_date, periods = t_end*inputfile.dt_hourly*inputfile.dt_sub_hourly, freq = inputfile.Agg_method))
		self.fpre = fpre
		self.dataETo = dataETo
		self.idateETo = idateETo
		self.idatepre = idatepre
		self.PET = np.zeros(env_state.grid_size)
		self.PETr = np.zeros(env_state.grid_size)
		self.t_end = t_end

	# find precipitation and PET for an specific time step
	def run_rainfall_one_step(self, j_tp, j_te, env_state, inputfile):
		"""
		Call this to execute a step in the model.
		"""
		self.rain_day_before = 0
		self.rain = np.zeros(env_state.grid_size)
		if not np.isnan(self.idatepre[j_tp]):
			if inputfile.netcf_pre == 1:
				self.rain = (self.fpre.variables['pre'][self.idatepre[j_tp]][:]).T.flatten()
			else: # Uniform precipitation over the whole catchement
				self.rain = np.ones(env_state.grid_size)*self.fpre['pre'][self.idatepre[j_tp]]
			self.rain_day_before = 1
			
		if not np.isnan(self.idateETo[j_te]):
			if inputfile.netcf_ETo == 1:
				self.PET = (self.dataETo.variables['pet'][self.idateETo[j_te]][:]).T.flatten()*inputfile.unit_sim
				self.PETr += self.dataETo.variables['pet'][self.idateETo[j_te]][:].flatten()*inputfile.unit_sim
			else: # Uniform precipitation over the whole catchement
				self.PET[:] = self.dataETo['ETo'][self.idateETo[j_te]]*inputfile.unit_sim
				#Cummulative value of ETp for daily stimation of AET in river cells
				self.PETr += self.dataETo['ETo'][self.idateETo[j_te]]*inputfile.unit_sim
		# Do the calc, using e.g. self.riv as you have it locally now
		# Update model_environment_status
	
	def work_out_stable_timestep(self,):
		"""
		Something like this might be needed
		"""
		pass

class input_datasets_bigfiles(object):
	"""Read input datasets from different sequential files
	Parameters:
	Outputs:
	
	"""
	def __init__(self, inputfile, env_state):
		dt = np.min([inputfile.dtOF, inputfile.dtUZ, inputfile.dtSZ])
		if inputfile.first_read == 1:
			t_end = inputfile.Sim_period.days
			self.date_sim_dt = pd.date_range(inputfile.ini_date,
				periods = t_end*inputfile.dt_hourly*inputfile.dt_sub_hourly,
				freq = str(np.int(dt))+'min')
			if t_end <= 0:
				sys.exit("End of the simulation period should be later than initial date")
		self.t_end = t_end
		self.read_before_pre = 1
		self.read_before_pet = 1
		self.year_pre = int(inputfile.ini_date.year)
		self.year_pet = int(inputfile.ini_date.year)
		self.dt = dt
		self.nsteps_pre = int(inputfile.dt/inputfile.dt_pre)
		self.nsteps_pet = int(inputfile.dt/inputfile.dt_pet)
		self.nsteps_day_pre = int(1440/inputfile.dt_pre)
		self.nsteps_day_pet = int(1440/inputfile.dt_pet)
		self.nsteps_hour_pre = int(inputfile.dt_pre/60)
		self.nsteps_hour_pet = int(inputfile.dt_pet/60)
	# find precipitation and PET for an specific time step
	def run_dataset_one_step(self, j, env_state, inputfile):
		"""
		Call this to execute a step in the model.
		Parameters:
			j:	Counter for time
		Outputs:
			rain:	precipitation for the actual time step
			pet:	potential evapotranspiration for current timestep
		"""
		date = self.date_sim_dt[j]
		# Precipitation
		idate_pre = self.date_sim_dt[j] - timedelta(hours=(self.nsteps_pre-1))
		self.rain = np.zeros(env_state.grid_size)
		for i in range(self.nsteps_pre):			
			if self.year_pre == idate_pre.year:
				hour_pre = int(int(idate_pre.strftime('%-H'))/self.nsteps_hour_pre)
				j_tp = (int(idate_pre.strftime('%-j'))-1)*self.nsteps_day_pre + hour_pre
				if self.read_before_pre == 1:
					fname_pre = inputfile.fname_TSPre + '_' + str(idate_pre.year) + '.nc'
					self.fpre = Dataset(fname_pre, 'r')
					rain = (self.fpre.variables['pre'][j_tp][:]).flatten()
					self.read_before_pre = 0
				else:
					rain = (self.fpre.variables['pre'][j_tp][:]).flatten()
					self.read_before_pre = 0
			else:
				hour_pre = int(int(idate_pre.strftime('%-H'))/self.nsteps_hour_pre) 
				j_tp = (int(idate_pre.strftime('%-j'))-1)*self.nsteps_day_pre + hour_pre
				fname_pre = inputfile.fname_TSPre + '_' + str(idate_pre.year) + '.nc'
				self.fpre = Dataset(fname_pre, 'r')
				rain = (self.fpre.variables['pre'][j_tp][:]).T.flatten()
				self.read_before_pre = 0
			self.rain += rain
			self.year_pre = int(idate_pre.year)
			idate_pre += timedelta(hours=1)
		
		# Evapotranspiration
		idate_pet = self.date_sim_dt[j] - timedelta(hours=(self.nsteps_pet-1))
		self.PET = np.zeros(env_state.grid_size)
		for i in range(self.nsteps_pet):			
			if self.year_pet == idate_pet.year:				
				hour_pet = int(int(idate_pet.strftime('%-H'))/self.nsteps_hour_pet)
				j_te = (int(idate_pet.strftime('%-j'))-1)*self.nsteps_day_pet + hour_pet
				if self.read_before_pet == 1:
					fname_pet = inputfile.fname_TSMeteo + '_' + str(idate_pet.year) + '.nc'
					self.fpet = Dataset(fname_pet, 'r')
					PET = (self.fpet.variables['pet'][j_tp][:]).flatten()
					self.read_before_pet = 0
				else:
					PET = (self.fpet.variables['pet'][j_tp][:]).flatten()
					self.read_before_pet = 0
			else:
				hour_pet = int(int(idate_pet.strftime('%-H'))/self.nsteps_hour_pet)
				j_te = (int(idate_pet.strftime('%-j'))-1)*self.nsteps_day_pet + hour_pet 
				fname_pet = inputfile.fname_TSMeteo + '_' + str(idate_pet.year) + '.nc'
				self.fpet = Dataset(fname_pet, 'r')
				PET = (self.fpet.variables['pet'][j_tp][:]).flatten()
				self.read_before_pet = 0
			self.PET += PET			
			self.year_pet = int(idate_pet.year)
			idate_pet += timedelta(hours=1)