#General functions

import os
import numpy as np
import pandas as pd
from calendar import monthrange
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
		self.TLS.append(ro.tls_dt[nodes])
	
	def extract_point_var_UZ_inf(self,nodes,inf):
		self.INF.append(inf.inf_dt[nodes])		
		self.EXS.append(inf.exs_dt[nodes])
	
	def extract_point_var_UZ_swb(self,nodes,swb):	
		self.AET.append(swb.aet_dt[nodes])		
		self.THT.append(swb.tht_dt[nodes])		
		self.PCL.append(swb.pcl_dt[nodes])
		self.WRSI.append(swb.WRSI[nodes])
	
	def extract_point_var_SZ(self,nodes,gw):
		self.WTE.append(gw.wte_dt[nodes])
			
	def extract_point_var_SZ_L2(self,nodes,grid):
		self.HEAD.append(grid.at_node['HEAD_2'][nodes])
	
	def save_point_var(self,fname,time):
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
			
			for i in range(len(data)):			
				field = 'OF_'+str(i)				
				df[field] = data[i,:]
			
			fname_dis = fname+'Dis.csv'			
			os.remove(fname_dis) if os.path.exists(fname_dis) else None			
			df.to_csv(fname_dis,index = False)
			
			df = pd.DataFrame()			
			df['Date'] = time			
			data = np.transpose(self.TLS)
			
			for i in range(len(data)):			
				field = 'TL_'+str(i)				
				df[field] = data[i,:]
			
			fname_dis = fname+'TLS.csv'			
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

		self.OFL.append(np.sum(area_catch_factor*ro.dis_dt[nodes]))		
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

	def __init__(self,grid):
		self.latitude = grid.node_y
		self.longitude =  grid.node_x
		self.grid_shape = grid.shape
		self.xaxis = grid.node_x[:self.grid_shape[1]]
		self.yaxis = np.linspace(np.min(grid.node_y), np.max(grid.node_y), num=self.grid_shape[0])
					
	def add_grid_var(self):
		self.PRE = []
		self.AET = []
		self.PET = []
		self.THT = []
		self.INF = []
		self.OFL = []
		self.PCL = []
		
	def append_array_var(self, pre, inf, swb, ro):
		self.PRE.append(np.array(pre.rain))
		self.PET.append(np.array(pre.PET))
		self.AET.append(np.array(swb.AET_dt))
		self.THT.append(np.array(swb.THT_dt))
		self.INF.append(np.array(inf.inf_dt))
		self.OFL.append(np.array(ro.dis_dt))
		self.PCL.append(np.array(swb.PCL_dt))
		
	def append_array_var_drylandmodel(self, rain, PET, AET_dt, THT_dt, inf_dt, dis_dt, PCL_dt):
		self.PRE.append(rain[:])
		self.PET.append(PET[:])
		self.AET.append(AET_dt[:])
		self.THT.append(THT_dt[:])
		self.INF.append(inf_dt[:])
		self.OFL.append(dis_dt[:])
		self.PCL.append(PCL_dt[:])
		
	def append_grid_var(self, pre, inf, swb, ro):
		self.PRE.append(np.array(pre.rain.reshape(self.grid_shape)))
		self.PET.append(np.array(pre.PET.reshape(self.grid_shape)))
		self.AET.append(np.array(swb.AET_dt.reshape(self.grid_shape)))
		self.THT.append(np.array(swb.THT_dt.reshape(self.grid_shape)))
		self.INF.append(np.array(inf.inf_dt.reshape(self.grid_shape)))
		self.OFL.append(np.array(ro.dis_dt.reshape(self.grid_shape)))
		self.PCL.append(np.array(swb.PCL_dt.reshape(self.grid_shape)))
		
	def create_zero_grid_var(self, grid):
		self.latitude = grid.dy
		self.longitude =  grid.dx
		array_aux = np.zeros(len(latitude))
		self.PRE = array_aux
		self.AET = array_aux
		self.PET = array_aux
		self.THT = array_aux
		self.INF = array_aux
		self.OFL = array_aux
		self.PCL = array_aux
		
	def empty_all_grid_var(self):
		self.PRE = []
		self.AET = []
		self.PET = []
		self.THT = []
		self.INF = []
		self.OFL = []
		self.PCL = []
	
	def	save_netCDF_var(self, date, fname):		
		dataset = Dataset(fname, 'w', format='NETCDF4_CLASSIC')
		dataset.createDimension('time', None)		
		dataset.createDimension('lon', self.grid_shape[1])		
		dataset.createDimension('lat', self.grid_shape[0])		
		# Create coordinate variables for 4-dimensions		
		lat = dataset.createVariable('lat', np.float64, ('lat',))		
		lon = dataset.createVariable('lon', np.float64, ('lon',))		
		time = dataset.createVariable('time', np.float64, ('time',))		
		# Create the actual 4-d variable			
		npre = dataset.createVariable('precipitation', np.float32, ('time', 'lon', 'lat'))
		npet = dataset.createVariable('pevaporation', np.float32, ('time', 'lon', 'lat'))
		naet = dataset.createVariable('aevaporation', np.float32, ('time', 'lon', 'lat'))
		ntht = dataset.createVariable('soil_moisture', np.float32, ('time', 'lon', 'lat'))
		ninf = dataset.createVariable('infitration', np.float32, ('time', 'lon', 'lat'))
		ndis = dataset.createVariable('runoff', np.float32, ('time', 'lon', 'lat'))
		nrch = dataset.createVariable('recharge', np.float32, ('time', 'lon', 'lat'))				
		time.units = 'hours since 1980-01-01 00:00:00'		
		time.calendar = 'gregorian'
		lat[:] = self.yaxis
		lon[:] = self.xaxis
		j = 0
		for idate in date:		
			time[j] = date2num(idate, units=time.units, calendar=time.calendar)
			npre[j,:,:] = self.PRE[j][:,:]
			naet[j,:,:] = self.AET[j][:,:]
			npet[j,:,:] = self.PET[j][:,:]
			ntht[j,:,:] = self.THT[j][:,:]
			ninf[j,:,:] = self.INF[j][:,:]
			ndis[j,:,:] = self.OFL[j][:,:]
			nrch[j,:,:] = self.PCL[j][:,:]
			j += 1
		dataset.close()
		
	def	save_monthly_netCDF_var(self, date, fname):		
		dataset = Dataset(fname, 'w', format='NETCDF4_CLASSIC')
		dataset.createDimension('time', None)		
		dataset.createDimension('lon', self.grid_shape[1])		
		dataset.createDimension('lat', self.grid_shape[0])		
		# Create coordinate variables for 4-dimensions		
		lat = dataset.createVariable('lat', np.float64, ('lat',))		
		lon = dataset.createVariable('lon', np.float64, ('lon',))		
		time = dataset.createVariable('time', np.float64, ('time',))		
		# Create the actual 4-d variable			
		npre = dataset.createVariable('precipitation', np.float32, ('time', 'lat', 'lon'))
		npet = dataset.createVariable('pevaporation', np.float32, ('time', 'lat', 'lon'))
		naet = dataset.createVariable('aevaporation', np.float32, ('time', 'lat', 'lon'))
		ntht = dataset.createVariable('soil_moisture', np.float32, ('time', 'lat', 'lon'))
		ninf = dataset.createVariable('infiltration', np.float32, ('time', 'lat', 'lon'))
		ndis = dataset.createVariable('runoff', np.float32, ('time', 'lat', 'lon'))
		nrch = dataset.createVariable('recharge', np.float32, ('time', 'lat', 'lon'))
		time.units = 'hours since 1980-01-01 00:00:00'
		time.calendar = 'gregorian'
		lat[:] = self.yaxis
		lon[:] = self.xaxis
		j = 0
		k = 0
		M = date[0].month
		time[0] = date2num(date[0], units=time.units, calendar=time.calendar)
				
		aux_pre = np.zeros(self.grid_shape)
		aux_aet = np.zeros(self.grid_shape)
		aux_pet = np.zeros(self.grid_shape)
		aux_tht = np.zeros(self.grid_shape)
		aux_inf = np.zeros(self.grid_shape)
		aux_dis = np.zeros(self.grid_shape)
		aux_rch = np.zeros(self.grid_shape)
		
		for idate in date:
			aux_pre += np.array(self.PRE[j])
			aux_aet += np.array(self.AET[j])
			aux_pet += np.array(self.PET[j])
			aux_tht += np.array(self.THT[j])
			aux_inf += np.array(self.INF[j])
			aux_dis += np.array(self.OFL[j])
			aux_rch += np.array(self.PCL[j])
							
			j += 1
			
			if j == len(date):			
				npre[k,:,:] = np.array(aux_pre)#.reshape(self.grid_shape[0],self.grid_shape[1])
				naet[k,:,:] = np.array(aux_aet)#.reshape(self.grid_shape[0],self.grid_shape[1])
				npet[k,:,:] = np.array(aux_pet)#.reshape(self.grid_shape[0],self.grid_shape[1])
				ninf[k,:,:] = np.array(aux_inf)#.reshape(self.grid_shape[0],self.grid_shape[1])
				ndis[k,:,:] = np.array(aux_dis)#.reshape(self.grid_shape[0],self.grid_shape[1])
				nrch[k,:,:] = np.array(aux_rch)#.reshape(self.grid_shape[0],self.grid_shape[1])
				ntht[k,:,:] = np.array(aux_tht)/(24*monthrange(idate.year, idate.month)[1])
			
			if j < len(date):			
				if date[j].month > M:					
					npre[k,:,:] = np.array(aux_pre)#.reshape(self.grid_shape[0],self.grid_shape[1])
					naet[k,:,:] = np.array(aux_aet)#.reshape(self.grid_shape[0],self.grid_shape[1])
					npet[k,:,:] = np.array(aux_pet)#.reshape(self.grid_shape[0],self.grid_shape[1])
					ntht[k,:,:] = np.array(aux_tht)#.reshape(self.grid_shape[0],self.grid_shape[1])/(24*monthrange(idate.year, idate.month)[1])
					ninf[k,:,:] = np.array(aux_inf)#.reshape(self.grid_shape[0],self.grid_shape[1])
					ndis[k,:,:] = np.array(aux_dis)#.reshape(self.grid_shape[0],self.grid_shape[1])
					nrch[k,:,:] = np.array(aux_rch)#.reshape(self.grid_shape[0],self.grid_shape[1])					
					ntht[k,:,:] = np.array(aux_tht)/(24*monthrange(idate.year, idate.month)[1])
					
					aux_pre = np.zeros(self.grid_shape)
					aux_aet = np.zeros(self.grid_shape)
					aux_pet = np.zeros(self.grid_shape)
					aux_tht = np.zeros(self.grid_shape)
					aux_inf = np.zeros(self.grid_shape)
					aux_dis = np.zeros(self.grid_shape)
					aux_rch = np.zeros(self.grid_shape)					
					
					k += 1
					time[k] = date2num(date[j],units = time.units, calendar = time.calendar)
					M = date[j].month
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
		self.DIS_grid_h = np.zeros(len(z))
		self.DIS_dt = np.zeros(len(z))
		self.SMD_h = array_aux
		self.THT_h = array_aux

	def extract_point_average_var(time_var, grid_var, nodes):
		for var_time_name, var_grid_name in zip(grid_var, time_var):
			var_time_name.append(var_grid_name[nodes])


def check_mass_balance(outavg, outOF, outavg_AER):	
	data_summary = []	
	per_PRE = 100*np.sum(outavg.PRE)/np.sum(outavg.PRE)
	per_AET = 100*np.sum(outavg.AET)/np.sum(outavg.PRE)
	per_INF = 100*np.sum(outavg.INF)/np.sum(outavg.PRE)
	per_EXS = 100*np.sum(outavg.EXS)/np.sum(outavg.PRE)
	per_TLS = 100*np.sum(outavg.TLS)/np.sum(outavg.PRE)
	per_PCL = 100*np.sum(outavg.PCL)/np.sum(outavg.PRE)
	
	if  outavg_AER:
		per_AER = 100*np.sum(outavg_AER.AET)/np.sum(outavg.PRE)
		per_FRC = 100*np.sum(outavg_AER.PCL)/np.sum(outavg.PRE)
	per_OFL = 100*np.sum(np.transpose(outOF.OFL)[0]) /np.sum(outavg.PRE)
	
	data_summary.append('Total Precipitation.......{:.3f} mm ({:.2f}%)'.format(np.sum(outavg.PRE),per_PRE))
	data_summary.append('Total Infiltration........{:.3f} mm ({:.2f}%)'.format(np.sum(outavg.INF),per_INF))
	data_summary.append('Total Excess..............{:.3f} mm ({:.2f}%)'.format(np.sum(outavg.EXS),per_EXS))
	data_summary.append('Total Transmission Losses.{:.3f} mm ({:.2f}%)'.format(np.sum(outavg.TLS),per_TLS))
	data_summary.append('Total Evapot. Hills Slop..{:.3f} mm ({:.2f}%)'.format(np.sum(outavg.AET),per_AET))
	if  outavg_AER:
		data_summary.append('Total Riparean AET .......{:.3f} mm ({:.2f}%)'.format(np.sum(outavg_AER.AET),per_AER))
		data_summary.append('Total Focused Recharge....{:.3f} mm ({:.2f}%)'.format(np.sum(outavg_AER.PCL),per_FRC))
	data_summary.append('Total Diffuse Recharge....{:.3f} mm ({:.2f}%)'.format(np.sum(outavg.PCL),per_PCL))
	data_summary.append('Total Discharge...........{:.3f} mm ({:.2f}%)'.format(np.sum(np.transpose(outOF.OFL)[0]) ,per_OFL))
	
	df_summary = pd.DataFrame(data = data_summary, index = range(0,len(data_summary)),columns = ['DrylandModel'])
	print(df_summary)	

def save_map_to_rastergrid(grid,field,fname):
	os.remove(fname) if os.path.exists(fname) else None
	files = write_esri_ascii(fname, grid,field)

def save_map_to_figure(time_var,grid_var,nodes):
	titlePre = 'Rainfall [mm] - '+date_sim_h[j_t-1].strftime("%m/%d/%Y, %H:%M")
	ax = imshow_grid(rg, rainfall, cmap = 'YlGnBu',show_elements = True,shrink = 0.5, color_for_closed = None,plot_name = titlePre,limits = (0, rainfall.max()))
	outputfilenameRainTimeSeries = outputcirnetcdf+'/WG_RR_'+date_sim_h[j_t-1].strftime("%d%m%Y_%H%M")+'.png'
	plt.savefig(outputfilenameRainTimeSeries,dpi = 100, bbox_inches = 'tight')
	plt.close() 

	titlePre = 'Soil Moisture [%] - '+date_sim_h[j_t-1].strftime("%m/%d/%Y, %H:%M")
	rg.at_node['Soil_Moisture'][act_nodes] = theta_h
	ax = imshow_grid(rg, 'Soil_Moisture', cmap = 'YlGnBu',show_elements = True,shrink = 0.5, color_for_closed = None,plot_name = titlePre,limits = (0, theta.max()))
	outputfilenameThetaTimeSeries = outputcirnetcdf+'/WG_SM_'+date_sim_h[j_t-1].strftime("%d%m%Y_%H%M")+'.png'
	plt.savefig(outputfilenameThetaTimeSeries,dpi = 100, bbox_inches = 'tight')
	plt.close() 
	
	if daily_model == 0: # if the daily model is disable, discharge and transmission losses are saved
		titlePre = 'Discharge [m3/h] - '+date_sim_h[j_t-1].strftime("%m/%d/%Y, %H:%M")
		ax = imshow_grid(rg, 'surface_water__discharge', cmap = 'YlGnBu',show_elements = True,shrink = 0.5, color_for_closed = None,plot_name = titlePre,limits = (0, rg['node']['surface_water__discharge'].max()))
		outputfilenamegridTimeSeries = outputcirnetcdf+'/WG_OF_'+date_sim_h[j_t-1].strftime("%d%m%Y_%H%M")+'.png'
		plt.savefig(outputfilenamegridTimeSeries,dpi = 100, bbox_inches = 'tight')
		plt.close() 

		titlePre = 'Transmission Losses[m3/h] - '+date_sim_h[j_t-1].strftime("%m/%d/%Y, %H:%M")
		ax = imshow_grid(rg, TL, cmap = 'YlGnBu',show_elements = True,shrink = 0.5, color_for_closed = None,plot_name = titlePre,limits = (0, TL.max()))
		outputfilenameThetaTimeSeries = outputcirnetcdf+'/WG_TL_'+date_sim_h[j_t-1].strftime("%d%m%Y_%H%M")+'.png'
		plt.savefig(outputfilenameThetaTimeSeries,dpi = 100, bbox_inches = 'tight')
		plt.close() 
		#plt.show()
		
		outputfilenamenetcdf = outputcirnetcdf+'DrylandModel_'+date_sim_h[j_t-1].strftime("%d%m%Y_%H%M")+'.nc'
		#write_netcdf(outputfilenamenetcdf, rg, format = 'NETCDF3_64BIT',names = ['Precipitation','Soil_Moisture','Transmission_losses','surface_water__discharge'])
	else: # if the daily model is active, discharge and transmission losses are not saved
		outputfilenamenetcdf = outputcirnetcdf+'DrylandModel_'+date_sim_h[j_t-1].strftime("%d%m%Y_%H%M")+'.nc'
		#write_netcdf(outputfilenamenetcdf, rg, format = 'NETCDF3_64BIT',names = ['Precipitation','Soil_Moisture'])
	
	#outputfilenamegridTimeSeries = outputcirnetcdf+'/WG_Res_'+date_sim_h[j_t-1].strftime("%d%m%Y_%H%M")+'.png'
	#plt.savefig(outputfilenamegridTimeSeries,dpi = 100, bbox_inches = 'tight')
	#plt.close()
	step_num += 1
	titlePre = 'Water table [m] - '+ini_date.strftime("%m/%d/%Y, %H:%M")
	ax = imshow_grid(rg, h, cmap = 'YlGnBu',show_elements = True,shrink = 0.5, color_for_closed = None,plot_name = titlePre,limits = (0, h.max()))
	outputfilenamewtTimeSeries = 'results_hourly/pic/WG_water_table_t_'+str(step_num)+'.png'
	plt.savefig(outputfilenamewtTimeSeries,dpi = 100, bbox_inches = 'tight')
	#imshow_grid(rg, 'Precipitation', cmap = 'hsv',show_elements = True)
	outputfilenamenetcdf_GW = outputcirnetcdf+'DrylandModel_GW_'+str(i)+'.nc'
	write_netcdf(outputfilenamenetcdf_GW, rg, format = 'NETCDF3_64BIT',names = 'Pressure_head')
	plt.show()	