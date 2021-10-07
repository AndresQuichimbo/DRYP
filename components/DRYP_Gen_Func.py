import os
import numpy as np
import pandas as pd
from calendar import monthrange
import datetime
from datetime import timedelta
from netCDF4 import Dataset, num2date, date2num
from landlab.io import write_esri_ascii

class GlobalTimeVarPts:

	def __init__(self):
		""" Create a list with model states and variables
			states at selected location will be added to the list
		RCH		: Recharge
		SMD_t	: Soil moisture deficit
		s_t		: Accumulated water depth [mm]
		PET_t	: Potential evapotranspiration [mm/day]
		AETr_t	: Actual evapotranspiration for river[mm/day]
		AET_t	: Actual evapotranspiration [mm/day]
		INF_t	: Infiltration rate [mm/day]
		PCL_t	: Discharge [mm/day]
		BFL_t	: Baseflow [mm/day]		
		"""
		self.PRE = []
		self.PET = []
		self.AET = []
		self.INF = []	# Infiltration rate [mm/day]
		self.EXS = []	# Runoff [mm/day]
		self.OFL = []	# Overlandflow [mm/day]
		self.QFL = []
		self.TLS = []
		self.PCL = []
		self.THT = []
		self.AER = []	# Actual evapotranspiration at rivers
		self.FRC = []
		self.WTE = []
		self.HEAD = []	# hydraulic head/water table bottom layer
		self.WRSI = []
	
	def extract_point_var_pre(self,nodes,pre):
		self.PRE.append(pre.rain[nodes])	
		self.PET.append(pre.PET[nodes])
		
	def extract_point_var_OF(self,nodes,ro):
		self.OFL.append(ro.dis_dt[nodes])
		self.QFL.append(ro.qfl_dt[nodes])
		self.TLS.append(ro.tls_dt[nodes])
	
	def extract_point_var_UZ_inf(self,nodes,inf):
		self.INF.append(inf.inf_dt[nodes])		
		self.EXS.append(inf.exs_dt[nodes])
	
	def extract_point_var_UZ_swb(self,nodes,swb):	
		self.AET.append(swb.aet_dt[nodes])		
		self.THT.append(swb.tht_dt[nodes])		
		self.PCL.append(swb.pcl_dt[nodes])
		self.WRSI.append(swb.WRSI[nodes])
	
	def extract_point_var_SZ(self, nodes, gw):
		self.WTE.append(gw.wte_dt[nodes])
			
	def extract_point_var_SZ_L2(self, nodes, grid):
		self.HEAD.append(grid.at_node['HEAD_2'][nodes])
	
	def save_point_var(self, fname, time, area, rarea):
		# precipitation
		if self.INF:
			df = pd.DataFrame()
			df['Date'] = time
			data = np.transpose(self.PRE)
			
			for i in range(len(data)):			
				field = 'PRE_'+str(i)				
				df[field] = data[i,:]
			
			fname_out = fname+'_PRE.csv'			
			os.remove(fname_out) if os.path.exists(fname_out) else None			
			df.to_csv(fname_out,index = False)
			
			df = pd.DataFrame()
			df['Date'] = time
			data = np.transpose(self.PET)
			
			for i in range(len(data)):			
				field = 'PET_'+str(i)				
				df[field] = data[i,:]
			
			fname_out = fname+'_PET.csv'			
			os.remove(fname_out) if os.path.exists(fname_out) else None			
			df.to_csv(fname_out,index = False)
		
		# infiltration
		if self.INF:
			df = pd.DataFrame()
			df['Date'] = time
			data = np.transpose(self.INF)
			
			for i in range(len(data)):			
				field = 'INF_'+str(i)				
				df[field] = data[i,:]
			
			fname_out = fname+'_INF.csv'			
			os.remove(fname_out) if os.path.exists(fname_out) else None			
			df.to_csv(fname_out,index = False)
			
			# infiltration excess				
			df = pd.DataFrame()			
			df['Date'] = time			
			data = np.transpose(self.EXS)
						
			for i in range(len(data)):			
				field = 'INF_'+str(i)				
				df[field] = data[i,:]
			
			fname_out = fname+'_EXS.csv'			
			os.remove(fname_out) if os.path.exists(fname_out) else None			
			df.to_csv(fname_out,index = False)
		
		if self.AET:
			
			# Actual evapotranspiration			
			df = pd.DataFrame()			
			df['Date'] = time			
			data = np.transpose(self.AET)
						
			for i in range(len(data)):			
				field = 'AET_'+str(i)				
				df[field] = data[i,:]
			
			fname_out = fname+'_AET.csv'			
			os.remove(fname_out) if os.path.exists(fname_out) else None			
			df.to_csv(fname_out,index = False)
			
			# Soil moisture			
			df = pd.DataFrame()			
			df['Date'] = time			
			data = np.transpose(self.THT)
			
			for i in range(len(data)):			
				field = 'SM_'+str(i)				
				df[field] = data[i,:]
				
			fname_out = fname+'_SM.csv'			
			os.remove(fname_out) if os.path.exists(fname_out) else None			
			df.to_csv(fname_out,index = False)
			
			# Soil percolation			
			df = pd.DataFrame()			
			df['Date'] = time			
			data = np.transpose(self.PCL)
			
			for i in range(len(data)):			
				field = 'RCH_'+str(i)				
				df[field] = data[i,:]
			
			fname_out = fname+'_RCH.csv'			
			os.remove(fname_out) if os.path.exists(fname_out) else None			
			df.to_csv(fname_out,index = False)
			
			# Ratio of seasonal actual crop evapotranspiration			
			df = pd.DataFrame()			
			df['Date'] = time			
			data = np.transpose(self.WRSI)
			
			for i in range(len(data)):			
				field = 'WRSI_'+str(i)				
				df[field] = data[i,:]
			
			fname_out = fname+'_WRSI.csv'			
			os.remove(fname_out) if os.path.exists(fname_out) else None			
			df.to_csv(fname_out,index = False)
		
		if self.OFL:
				
			df = pd.DataFrame()			
			df['Date'] = time			
			data = np.transpose(self.OFL)
			
			for i, iarea in enumerate(area):			
				field = 'OF_'+str(i)				
				df[field] = data[i,:]*1000./iarea
			
			fname_dis = fname+'Dis.csv'			
			os.remove(fname_dis) if os.path.exists(fname_dis) else None			
			df.to_csv(fname_dis,index = False)
			
			#df = pd.DataFrame()			
			#df['Date'] = time			
			#data = np.transpose(self.TLS)
			#
			#for i in range(len(data)):			
			#	field = 'OF_'+str(i)				
			#	df[field] = data[i,:]
			#
			#fname_dis = fname+'Run.csv'			
			#os.remove(fname_dis) if os.path.exists(fname_dis) else None			
			#df.to_csv(fname_dis,index = False)
			
			df = pd.DataFrame()			
			df['Date'] = time			
			data = np.transpose(self.TLS)
			
			for i in range(len(data)):			
				field = 'TL_'+str(i)				
				df[field] = data[i,:]
			
			fname_dis = fname+'TLS.csv'			
			os.remove(fname_dis) if os.path.exists(fname_dis) else None			
			df.to_csv(fname_dis,index = False)
			
			df = pd.DataFrame()			
			df['Date'] = time			
			data = np.transpose(self.QFL)#+self.OFL)
			#print(rarea, data)
			for i, irarea in enumerate(rarea):#range(len(data)):			
				field = 'QFL_'+str(i)				
				df[field] = data[i,:]/irarea
			
			fname_dis = fname+'QFL.csv'			
			os.remove(fname_dis) if os.path.exists(fname_dis) else None			
			df.to_csv(fname_dis,index = False)
			
		if self.WTE:
			
			df = pd.DataFrame()			
			df['Date'] = time			
			data = np.transpose(self.WTE)
			
			for i in range(len(data)):			
				field = 'WT_'+str(i)				
				df[field] = data[i,:]
			
			fname_dis = fname+'wte.csv'			
			os.remove(fname_dis) if os.path.exists(fname_dis) else None			
			df.to_csv(fname_dis,index = False)
			
		if self.HEAD:
			
			df = pd.DataFrame()			
			df['Date'] = time			
			data = np.transpose(self.HEAD)
			
			for i in range(len(data)):			
				field = 'WT_'+str(i)				
				df[field] = data[i,:]
				
			fname_dis = fname+'head.csv'			
			os.remove(fname_dis) if os.path.exists(fname_dis) else None			
			df.to_csv(fname_dis,index = False)
	
	
class GlobalTimeVarAvg:

	def __init__(self,*factor):
		""" Create a list with model states and variables
			states at selected location will be added to the list
		RCH		: Recharge
		SMD_t	: Soil moisture deficit
		s_t		: Accumulated water depth [mm]
		PET_t	: Potential evapotranspiration [mm/day]
		AETr_t	: Actual evapotranspiration for river[mm/day]
		AET_t	: Actual evapotranspiration [mm/day]
		INF_t	: Infiltration rate [mm/day]
		PCL_t	: Discharge [mm/day]
		BFL_t	: Baseflow [mm/day]
		
		*factor	: Catchment area factor for fluxes estimation
		
		"""
		self.PRE = []
		self.PET = []
		self.AET = []
		self.INF = []	# Infiltration rate [mm/day]
		self.EXS = []	# Runoff [mm/day]
		self.OFL = []	# Overlandflow [mm/day]
		self.QFL = []
		self.TLS = []
		self.PCL = []
		self.THT = []
		self.AER = []	# Actual evapotranspiration at rivers
		self.FRC = []
		self.WTE = []
		
		if factor:
			
			self.cth_factor = factor[0]
					
		else:
		
			self.cth_factor = []
		
	def extract_avg_var_pre(self,nodes,pre):		
		if len(self.cth_factor):			
			area_catch_factor = np.array(self.cth_factor[nodes])		
		else:		
			area_catch_factor = 1.0

		self.PRE.append(np.sum(area_catch_factor*pre.rain[nodes]))	
		self.PET.append(np.sum(area_catch_factor*pre.PET[nodes]))
		
	def extract_avg_var_OF(self,nodes,ro):
		if len(self.cth_factor):		
			area_catch_factor = np.array(self.cth_factor[nodes])		
		else:		
			area_catch_factor = 1.0
		
		if ro.carea is None:
			factor = 1
		else:
			factor = (1000/ro.carea[nodes])
		self.OFL.append(np.sum(area_catch_factor*factor*ro.dis_dt[nodes]))		
		self.TLS.append(np.sum(area_catch_factor*ro.tls_dt[nodes]))
	
	def extract_avg_var_UZ_inf(self,nodes,inf):	
		if len(self.cth_factor):		
			area_catch_factor = np.array(self.cth_factor[nodes])		
		else:		
			area_catch_factor = 1.0

		self.INF.append(np.sum(area_catch_factor*inf.inf_dt[nodes]))		
		self.EXS.append(np.sum(area_catch_factor*inf.exs_dt[nodes]))
	
	def extract_avg_var_UZ_swb(self,nodes,swb):	
		if len(self.cth_factor):		
			area_catch_factor = np.array(self.cth_factor[nodes])		
		else:		
			area_catch_factor = 1.0
			
		self.AET.append(np.sum(area_catch_factor*swb.aet_dt[nodes]))		
		self.THT.append(np.sum(area_catch_factor*swb.tht_dt[nodes]))		
		self.PCL.append(np.sum(area_catch_factor*swb.pcl_dt[nodes]))
	
	def extract_avg_var_SZ(self,nodes,gw):
		self.WTE.append(np.mean(gw.wte_dt[nodes]))
				
	def save_avg_var(self,fname,time):	
		df = pd.DataFrame()		
		df['Date'] = time
		
		if self.PRE:		
			df['PET'] = self.PET			
			df['Pre'] = self.PRE		
		if self.AET:		
			df['AET'] = self.AET			
			df['PCL'] = self.PCL			
			df['THT'] = self.THT		
		if self.EXS:		
			df['EXS'] = self.EXS			
			df['INF'] = self.INF		
		if self.TLS:		
			df['TLS'] = self.TLS			
			df['DIS'] = self.OFL		
		if self.WTE:		
			df['WTE'] = self.WTE		
		#df.index = pd.DatetimeIndex(df['Date'])
		#agg_step = 'D'
		#df = df.resample(agg_step).sum().reset_index()
		
		os.remove(fname) if os.path.exists(fname) else None
		df.to_csv(fname,index = False)		
		
		
class GlobalGridVar:

	def __init__(self, env_state, inputfile):
		"""Create a set of 2D arrays to store spatio-temporal datasets
		"""
		self.save_results = int(inputfile.save_results)
		if self.save_results == 1:
			self.lat = env_state.grid.node_y
			self.lon = env_state.grid.node_x
			self.grid_shape = np.array(env_state.grid.shape)
			self.xaxis = env_state.grid.node_x[:self.grid_shape[1]]
			self.yaxis = np.linspace(np.min(env_state.grid.node_y),
					np.max(env_state.grid.node_y),
					num=self.grid_shape[0])
					
			# temporal accumulation
			self.PRE_dt = np.zeros_like(self.lat)
			self.PET_dt = np.zeros_like(self.lat)
			self.AET_dt = np.zeros_like(self.lat)
			self.THT_dt = np.zeros_like(self.lat)
			self.INF_dt = np.zeros_like(self.lat)
			self.OFL_dt = np.zeros_like(self.lat)
			self.QFL_dt = np.zeros_like(self.lat)
			self.TLS_dt = np.zeros_like(self.lat)
			self.FRH_dt = np.zeros_like(self.lat)
			self.DRH_dt = np.zeros_like(self.lat)
			self.GDH_dt = np.zeros_like(self.lat)
			self.WTE_dt = np.zeros_like(self.lat)
			# Calculate time delta
			self.delta = str2timedelta(inputfile.ini_date,
						inputfile.dt_results)
			# Calculate text time step			
			self.idate = addtime(inputfile.ini_date, self.delta)
			self.pdate = inputfile.ini_date
			# Store variables
			self.PRE = []
			self.PET = []
			self.AET = []
			self.THT = []
			self.INF = []
			self.OFL = []
			self.QFL = []
			self.TLS = []
			self.FRH = []
			self.DRH = []
			self.GDH = []
			self.WTE = []
			self.time_grid = []
			self.nsteps = 0
			
	def get_env_state(self, t_date, pre, inf, swb, ro, gw, swb_rip, env_state):
		if self.save_results == 1:
			if t_date == len(pre.date_sim_dt)-1:
				self.idate = pre.date_sim_dt[t_date]
				date = pre.date_sim_dt[t_date]
			else:
				date = pre.date_sim_dt[t_date+1]
			self.PRE_dt += pre.rain
			self.PET_dt += pre.PET
			self.AET_dt += swb.aet_dt
			self.THT_dt += swb.tht_dt
			self.INF_dt += inf.inf_dt
			self.OFL_dt += ro.dis_dt
			self.QFL_dt += ro.qfl_dt
			self.TLS_dt += ro.tls_dt
			self.FRH_dt += swb_rip.pcl_dt
			self.DRH_dt += swb.pcl_dt
			self.GDH_dt += env_state.SZgrid.at_node['discharge'][:]
			self.WTE_dt += gw.wte_dt		
			self.nsteps += 1
			if self.idate == date:
				self.PRE.append(self.PRE_dt)
				self.PET.append(self.PET_dt)
				self.AET.append(self.AET_dt)
				self.THT.append(self.THT_dt*1/self.nsteps)
				self.INF.append(self.INF_dt)
				self.OFL.append(self.OFL_dt)
				self.QFL.append(self.QFL_dt)
				self.TLS.append(self.TLS_dt)
				self.FRH.append(self.FRH_dt)
				self.DRH.append(self.DRH_dt)
				self.GDH.append(self.GDH_dt)
				self.WTE.append(self.WTE_dt*1/self.nsteps)
				# temporal accumulation
				self.PRE_dt = np.zeros_like(self.lat)
				self.PET_dt = np.zeros_like(self.lat)
				self.AET_dt = np.zeros_like(self.lat)
				self.THT_dt = np.zeros_like(self.lat)
				self.INF_dt = np.zeros_like(self.lat)
				self.OFL_dt = np.zeros_like(self.lat)
				self.QFL_dt = np.zeros_like(self.lat)
				self.TLS_dt = np.zeros_like(self.lat)
				self.FRH_dt = np.zeros_like(self.lat)
				self.DRH_dt = np.zeros_like(self.lat)
				self.GDH_dt = np.zeros_like(self.lat)
				self.WTE_dt = np.zeros_like(self.lat)		
							
				self.time_grid.append(self.pdate)
				self.pdate = self.idate
				self.idate = addtime(self.idate, self.delta)
				self.nsteps = 0
			
	def	save_netCDF_var(self, fname):		
		if self.save_results == 1:
			dataset = Dataset(fname, 'w', format='NETCDF4_CLASSIC')
			dataset.createDimension('time', None)		
			dataset.createDimension('lon', self.grid_shape[1])		
			dataset.createDimension('lat', self.grid_shape[0])		
			# Create coordinate variables for 4-dimensions		
			lat = dataset.createVariable('lat', np.float32, ('lat',))		
			lon = dataset.createVariable('lon', np.float32, ('lon',))		
			time = dataset.createVariable('time', np.float32, ('time',))		
			# Create the actual 4-d variable			
			npre = dataset.createVariable('pre', np.float32, ('time', 'lat', 'lon'))
			npet = dataset.createVariable('pet', np.float32, ('time', 'lat', 'lon'))
			naet = dataset.createVariable('aet', np.float32, ('time', 'lat', 'lon'))
			ntht = dataset.createVariable('tht', np.float32, ('time', 'lat', 'lon'))
			ninf = dataset.createVariable('inf', np.float32, ('time', 'lat', 'lon'))
			ndis = dataset.createVariable('run', np.float32, ('time', 'lat', 'lon'))
			nqfl = dataset.createVariable('qfl', np.float32, ('time', 'lat', 'lon'))
			ntls = dataset.createVariable('tls', np.float32, ('time', 'lat', 'lon'))
			nfch = dataset.createVariable('fch', np.float32, ('time', 'lat', 'lon'))				
			ndch = dataset.createVariable('dch', np.float32, ('time', 'lat', 'lon'))
			ngdh = dataset.createVariable('gdh', np.float32, ('time', 'lat', 'lon'))
			nwte = dataset.createVariable('wte', np.float32, ('time', 'lat', 'lon'))
			
			time.units = 'hours since 1980-01-01 00:00:00'		
			time.calendar = 'gregorian'
			lat[:] = self.yaxis
			lon[:] = self.xaxis
			
			for j, idate in enumerate(self.time_grid):		
				time[j] = date2num(idate, units=time.units, calendar=time.calendar)
				npre[j,:,:] = (self.PRE[j][:]).reshape(self.grid_shape)
				npet[j,:,:] = (self.PET[j][:]).reshape(self.grid_shape)
				naet[j,:,:] = (self.AET[j][:]).reshape(self.grid_shape)
				ntht[j,:,:] = (self.THT[j][:]).reshape(self.grid_shape)
				ninf[j,:,:] = (self.INF[j][:]).reshape(self.grid_shape)
				ndis[j,:,:] = (self.OFL[j][:]).reshape(self.grid_shape)
				nqfl[j,:,:] = (self.QFL[j][:]).reshape(self.grid_shape)
				ntls[j,:,:] = (self.TLS[j][:]).reshape(self.grid_shape)
				nfch[j,:,:] = (self.FRH[j][:]).reshape(self.grid_shape)
				ndch[j,:,:] = (self.DRH[j][:]).reshape(self.grid_shape)
				ngdh[j,:,:] = (self.GDH[j][:]).reshape(self.grid_shape)
				nwte[j,:,:] = (self.WTE[j][:]).reshape(self.grid_shape)
				
			dataset.close()
		
	def env_state_grid(self):
		self.inf_dt = []

	def empty_daily_grid(self):
		array_aux = np.zeros(array_size)
		self.INF = array_aux
		self.EXS = array_aux
		self.AET = array_aux
		self.OFT = array_aux
		self.THT_d = array_aux
		self.SMD = array_aux
		self.PRE_aux_d = array_aux
		self.PCL_d = array_aux
	
	def empty_subdaily_grid(self, array_size):
		array_aux = np.zeros(array_size)
		self.INF_h = []
		self.EXS_h = array_aux
		self.RIN_h = array_aux
		self.Dh = array_aux
		self.AETh = array_aux
		self.INF_dt = array_aux
		self.DIS_grid_h = np.zeros_like(z)
		self.DIS_dt = np.zeros_like(z)
		self.SMD_h = array_aux
		self.THT_h = array_aux

	def extract_point_average_var(time_var, grid_var, nodes):
		for var_time_name, var_grid_name in zip(grid_var, time_var):
			var_time_name.append(var_grid_name[nodes])


def check_mass_balance(fname, outavg, outOF, outavg_AER, data, dt, area):	
	data_summary = []	
	per_PRE = 100*np.sum(outavg.PRE)/np.sum(outavg.PRE)
	per_AET = 100*np.sum(outavg.AET)/np.sum(outavg.PRE)
	per_INF = 100*np.sum(outavg.INF)/np.sum(outavg.PRE)
	per_EXS = 100*np.sum(outavg.EXS)/np.sum(outavg.PRE)
	per_TLS = 100*np.sum(outavg.TLS)/np.sum(outavg.PRE)
	per_PCL = 100*np.sum(outavg.PCL)/np.sum(outavg.PRE)
	
	if outavg_AER:
		per_AER = 100*np.sum(outavg_AER.AET)/np.sum(outavg.PRE)
		per_FRC = 100*np.sum(outavg_AER.PCL)/np.sum(outavg.PRE)
	per_OFL = 100*(1000*np.sum(np.transpose(outOF.OFL)[0])/area)/np.sum(outavg.PRE)
	
	data_summary.append('Total Precipitation.......{:.3f} mm ({:.2f}%)'.format(np.sum(outavg.PRE),per_PRE))
	data_summary.append('Total Infiltration........{:.3f} mm ({:.2f}%)'.format(np.sum(outavg.INF),per_INF))
	data_summary.append('Total Excess..............{:.3f} mm ({:.2f}%)'.format(np.sum(outavg.EXS),per_EXS))
	data_summary.append('Total Transmission Losses.{:.3f} mm ({:.2f}%)'.format(np.sum(outavg.TLS),per_TLS))
	data_summary.append('Total Evapot. Hills Slop..{:.3f} mm ({:.2f}%)'.format(np.sum(outavg.AET),per_AET))
	if  outavg_AER:
		data_summary.append('Total Riparean AET .......{:.3f} mm ({:.2f}%)'.format(np.sum(outavg_AER.AET), per_AER))
		data_summary.append('Total Focused Recharge....{:.3f} mm ({:.2f}%)'.format(np.sum(outavg_AER.PCL), per_FRC))
	data_summary.append('Total Diffuse Recharge....{:.3f} mm ({:.2f}%)'.format(np.sum(outavg.PCL), per_PCL))
	data_summary.append('Total Discharge...........{:.3f} mm ({:.2f}%)'.format((1000*np.sum(np.transpose(outOF.OFL)[0])/area) ,per_OFL))
	
	df_summary = pd.DataFrame(data=data_summary, index=range(0,len(data_summary)), columns=['DrylandModel'])
	fname_out = fname+'_summary.txt'
	df_summary.to_csv(fname_out, index=False)
	print(df_summary)
	
	df = pd.DataFrame()
	df['Date'] = dt
	df['pre'] = np.array(data[0])
	df['exs'] = np.array(data[1])
	df['tls'] = np.array(data[2])
	df['rch'] = np.array(data[3])
	df['gws'] = np.array(data[4])*1000.
	df['uzs'] = np.array(data[5])
	df['dis'] = np.array(data[6])*1000.
	df['aet'] = np.array(data[7])
	df['egw'] = np.array(data[8])
	fname_out = fname+'_mb.csv'
	df.to_csv(fname_out, index = False)
	
	check_mass_balance_error(df)
	
	pass	

def check_mass_balance_error(df):
	"""Dataframe
	"""	
	IN_UZ = df['pre'] - df['exs'] + df['tls']	
	OUT_UZ = df['rch'] + df['aet']
	
	IN_SZ = df['rch']	
	OUT_SZ = df['dis'] + df['egw']
	
	ERROR_UZ = IN_UZ - OUT_UZ - df['uzs']	
	ERROR_SZ = IN_SZ - OUT_SZ - df['gws']
	
	TOTAL = np.sum(IN_UZ+OUT_UZ) + np.sum(IN_SZ+OUT_SZ)
	
	#print('Mass balance Error: ', np.sum(ERROR_UZ+ERROR_SZ)/TOTAL)
	#print('Mass balance unsaturated zone:', np.sum(ERROR_UZ)/np.sum(IN_UZ+OUT_UZ))
	#print('Mass balance saturated zone:', np.sum(ERROR_SZ)/np.sum(IN_SZ+OUT_SZ))


def str2timedelta(date, delta):
	# It will assume a year time step if a only a number is provided	
	# format Y M D H
	
	dt = [0, 0, 0, 0]
	if len(delta) > 1:
		dt_t = int(delta[:-1])
		dt_delta = delta[-1]
		print(delta, type(dt_t), dt_delta)
		if dt_delta == 'H':
			dt[3] = dt_t
		elif dt_delta == 'D':
			dt[2] = dt_t
		elif dt_delta == 'M':
			dt[1] = dt_t
		else:
			dt[0] = dt_t
	else:
		if delta == 'H':
			dt[3] = 1
		elif delta == 'D':
			dt[2] = 1
		elif delta == 'M':
			dt[1] = 1
		else:
			dt[0] = 1
	return dt

def addtime(date,dt):
	#print(type(date),dt)
	if dt[0] > 0:
		date = datetime.date(date.year+1,1,1)
	if dt[1] > 0:
		if date.month == 12:
			#date.month = 1
			#date.year += 1
			date = datetime.datetime(date.year+1,1,1)
		else:
			#date.month += dt[1]
			date = datetime.datetime(date.year, date.month+1, 1)
	#print(date)
	#if dt[0] == 0:
	#	date = date + timedelta(days=date[2], hours=dt[3])
	#if dt[1] == 0:
	#	date = date + timedelta(days=date[2], hours=dt[3])
	return date

def save_map_to_rastergrid(grid, field, fname):
	os.remove(fname) if os.path.exists(fname) else None
	files = write_esri_ascii(fname, grid,field)


# Modify model input files ans model setting files	
def write_sim_file(filename_input, parameter):
	""" modify the parematers of the model input file and
		model setting file.
		WARNING: it will reeplace the original file, so
		make a copy of the original files
	parameters:
		filename_input:	model inputfile, including path
		parameter:		1D array of model paramters
	"""
	if not os.path.exists(filename_input):
		raise ValueError("File not availble")
	
	f = pd.read_csv(filename_input)
	f.drylandmodel[1] = f.drylandmodel[1] + str(int(parameter[0]))
	
	filename_simpar = f.drylandmodel[87]
	fsimpar = pd.read_csv(filename_simpar)	
	
	# Simulation parameters
	fsimpar['DWAPM_SET'][46] = ('%.5f' % parameter[1]) # kdt
	fsimpar['DWAPM_SET'][48] = ('%.5f' % parameter[2]) # kDroot
	fsimpar['DWAPM_SET'][50] = ('%.2f' % parameter[3]) # kAWC
	fsimpar['DWAPM_SET'][52] = ('%.5f' % parameter[4]) # kKsat
	fsimpar['DWAPM_SET'][54] = ('%.5f' % parameter[5]) # kSigma
	fsimpar['DWAPM_SET'][56] = ('%.5f' % parameter[6]) # kKch
	fsimpar['DWAPM_SET'][58] = ('%.5f' % parameter[7]) # T
	fsimpar['DWAPM_SET'][60] = ('%.5f' % parameter[8]) # kW
	fsimpar['DWAPM_SET'][62] = ('%.5f' % parameter[9]) # kKaq
	fsimpar['DWAPM_SET'][64] = ('%.5f' % parameter[10])# kSy
	
	os.remove(filename_input) if os.path.exists(filename_input) else None
	os.remove(filename_simpar) if os.path.exists(filename_simpar) else None
	
	f.to_csv(filename_input, index=False)
	fsimpar.to_csv(filename_simpar, index=False)