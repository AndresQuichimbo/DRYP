import os
import time
import datetime
import numpy as np
import pandas as pd
from landlab import RasterModelGrid
from landlab.io import read_esri_ascii, write_esri_ascii
from datetime import timedelta, datetime
from netCDF4 import Dataset, num2date, date2num
from landlab.components import FlowDirectorSteepest, FlowAccumulator
# Global parameters
ABC_RIVER = 0.2 # River abstraction parameter

class inputfile(object):
	"""
	"""
	def __init__(self, filename_inputs, first_read=1):
		"""Model paramter settings and input file namens and location
		"""
		self.first_read = first_read
		# =================================================================
		f = pd.read_csv(filename_inputs)
		self.Mname = f.drylandmodel[1]
		# Surface components ======================================== SZ = 
		self.fname_DEM = f.drylandmodel[4]
		self.fname_Area = f.drylandmodel[6]
		self.fname_FlowDir = f.drylandmodel[8]
		self.fname_Mask = f.drylandmodel[12]
		self.fname_River = f.drylandmodel[14]
		self.fname_RiverWidth = f.drylandmodel[16]
		self.fname_RiverElev = f.drylandmodel[18]		
		# Subsurface components ===================================== UZ = 
		self.fname_n = f.drylandmodel[28]		# porosity (n)
		self.fname_theta_r = f.drylandmodel[30]	# Saturated infiltration rate (a-Ks)
		self.fname_AWC = f.drylandmodel[32]		# Available water content (AWC)
		self.fname_wp = f.drylandmodel[34]		# wilting point (wp)
		self.fname_SoilDepth = f.drylandmodel[36] # root zone (D)
		self.fname_b_SOIL = f.drylandmodel[38]	# Soil parameter alpha (b)
		self.fname_PSI = f.drylandmodel[40]		# Soil parameter alpha (alpha)
		self.fname_Ksat_soil = f.drylandmodel[42] # Saturated infiltration rate (a-Ks)
		self.fname_sigma_ks = f.drylandmodel[44]
		self.fname_theta = f.drylandmodel[46]	# Initial water content [-]
		self.fname_Ksat_ch = f.drylandmodel[48] # Channel Saturated hydraulic conductivity (Ks)
		
		# Groundwater components ==================================== GW = 
		self.fname_GWdomain = f.drylandmodel[51]# GW Boundary conditions
		self.fname_SZ_Ksat = f.drylandmodel[53] # Saturated hydraulic conductivity (Ks)
		self.fname_SZ_Sy = f.drylandmodel[55] 	# Specific yield
		self.fname_GWini = f.drylandmodel[57] 	# Initial water table
		self.fname_FHB = f.drylandmodel[59]		# flux head boundary
		self.fname_CHB = f.drylandmodel[61]		# Constant flux boundary
		self.fname_SZ_bot = f.drylandmodel[63]	# Aquifer bottom elevation
		
		if len(f) < 89:
			self.fname_a_aq = 'None'
		else:
			self.fname_a_aq = f.drylandmodel[89]
		
		if len(f) < 91:
			self.fname_b_aq = 'None'
		else:
			self.fname_b_aq = f.drylandmodel[91]
		# Meterological data ======================================== ET = 
		self.fname_TSPre = f.drylandmodel[66]	# Precipitation file
		self.fname_TSMeteo = f.drylandmodel[68]	# Evapotranspiration file
		self.fname_TSABC = f.drylandmodel[70]	# Abstraction file: AOF, AUZ, ASZ
		# Vegetation parameters ==========================================
		self.fname_TSKc = f.drylandmodel[21]	# Vegetation parameter Kc
		self.fname_Rip_width = f.drylandmodel[23]#Available
		self.fname_Rip_init = f.drylandmodel[25] #Available
		# Output files maps ===================================== Print = 
		self.DirOutput = f.drylandmodel[81]		# Output directory
		#reading output points
		self.fname_DISpoints = f.drylandmodel[75]	# Discharge points
		self.fname_SMDpoints = f.drylandmodel[77]	# Soil moisture points
		self.fname_GWpoints = f.drylandmodel[79]	# Groundwater observation points
		
		#==================================================================		
		# Reading simulation and printing parameters		
		filename_simpar = f.drylandmodel[87]
		fsimpar = pd.read_csv(filename_simpar)
		
		self.ini_date = datetime.strptime(fsimpar.DWAPM_SET[2], '%Y %m %d')
		self.end_datet = datetime.strptime(fsimpar.DWAPM_SET[4], '%Y %m %d')
		self.dtOF = int(fsimpar.DWAPM_SET[6])
		self.dtUZ = float(fsimpar.DWAPM_SET[8])
		self.dtSZ = float(fsimpar.DWAPM_SET[10])
		
		aux_dt_pre = fsimpar.DWAPM_SET[13].split()
		aux_dt_pet = fsimpar.DWAPM_SET[15].split()
		aux_dt_ABC = fsimpar.DWAPM_SET[17].split()
		
		# Datasets format
		self.netcf_pre = int(aux_dt_pre[0])
		self.netcf_ETo = int(aux_dt_pet[0])
		self.netcf_ABC = int(aux_dt_ABC[0])
		
		# default time step of data sets
		self.dt_pre = 60
		self.dt_pet = 60
		self.dt_ABC = 60
		
		if len(aux_dt_pre) > 1:
			self.dt_pre = int(aux_dt_pre[1])
		if len(aux_dt_pet) > 1:
			self.dt_pet = int(aux_dt_pet[1])
		if len(aux_dt_ABC) > 1:
			self.dt_ABC = int(aux_dt_ABC[1])
		
		# read separeted files for all inputs
		self.nfiles = int(fsimpar.DWAPM_SET[31])
						
		self.inf_method = int(fsimpar.DWAPM_SET[20])
		
		if self.inf_method > 3:
			self.inf_method = 0
		
		aux_run_GW = fsimpar.DWAPM_SET[24].split()
		
		self.run_GW = int(aux_run_GW[0])
		if len(aux_run_GW) > 1:
			self.gw_func = int(aux_run_GW[1])
		else:
			self.gw_func = 0
		
		self.run_OF_lr = int(fsimpar.DWAPM_SET[22])
				
		self.print_sim_ti = int(fsimpar.DWAPM_SET[31])
		self.print_t = int(fsimpar.DWAPM_SET[43])
		self.save_results = int(fsimpar.DWAPM_SET[33])
		self.dt_results = fsimpar.DWAPM_SET[35]
		self.print_maps_tn = int(fsimpar.DWAPM_SET[39])		
		# Unsaturated zone factors =========================================
		self.kdt_r = float(fsimpar.DWAPM_SET[46])
		self.kDroot = float(fsimpar.DWAPM_SET[48])	# k for soil depth
		self.kAWC = float(fsimpar.DWAPM_SET[50])	# k for AWC
		self.kKs = float(fsimpar.DWAPM_SET[52])		# k for soil infiltration
		self.k_sigma_ks = float(fsimpar.DWAPM_SET[54])
		# River routing factors ============================================
		self.Kloss = float(fsimpar.DWAPM_SET[56])	# infiltration on channel
		self.T_loss = float(fsimpar.DWAPM_SET[58])	# Runoff decay flow
		self.kpe = float(fsimpar.DWAPM_SET[60])
		# Saturated zone factors ===========================================
		self.kKsat_gw = float(fsimpar.DWAPM_SET[62])# Runoff decay flow
		self.kSy_gw = float(fsimpar.DWAPM_SET[64])
		if len(fsimpar.DWAPM_SET) == 66:
			self.GW_Cond_factor = float(fsimpar.DWAPM_SET[66])
		else:
			self.GW_Cond_factor = 50
		#self.kTr_ini_par = float(fsimpar.DWAPM_SET[51])
		#self.kpKloss = float(fsimpar.DWAPM_SET[51])
		#self.kpLoss = float(fsimpar.DWAPM_SET[51])
		#self.Ktr = float(fsimpar.DWAPM_SET[51])
		self.dt = np.min([self.dtOF, self.dtUZ, self.dtSZ])
		if self.dt > 60:
			self.dt_sub_hourly = 1
			self.dt_hourly = np.int(1440/self.dt)
			self.unit_sim = self.dt/1440		#change mm/d -> mm/dt
			self.unit_sim_k = self.dt*24/1440	#change mm/h -> mm/dt
			self.kT_units = self.dt/60
		else:
			self.dt_sub_hourly = np.int(60/self.dt)
			self.dt_hourly = 24
			self.unit_sim = self.dt/60			#change mm/d -> mm/dt
			self.unit_sim_k = self.dt/60		#change mm/h -> mm/dt
			self.kT_units = self.dt/60
		self.unit_change_manning = (1/(self.dt*60))**(3/5)
		self.Agg_method = str(self.dt)+'T'
		self.kpKloss = 1.0							# initial Kloss increase for TL
		self.T_str_channel = 0.0			# duration of initial Kloss increase for TL
		self.Kloss = self.Kloss*self.unit_sim_k
		self.river_banks = 30.0 					# Riparian zone with [m]
		self.run_FAc = 1
		self.dt_OF = 1
		self.Sim_period = self.end_datet - self.ini_date
		
		print("Model Name: ",self.Mname)
		
class model_environment_status(object):
	"""Setting model input varables and environmental states
	"""
	def __init__(self, inputfile):
	
		# build the data classes
		# ================ Reading surface water model inputs ==============
		
		print('******************* Reading Input Files ********************')
		
		# Reading digital elevation model
		if os.path.exists(inputfile.fname_DEM):
			(rg, z) = read_esri_ascii(inputfile.fname_DEM,
				name='topographic__elevation')
		else:
			raise Exception("A digital elevation model map must be supplied")
			
		# Reading the raster file of river network
		if os.path.exists(inputfile.fname_River):     
			read_esri_ascii(inputfile.fname_River,
				name='river_length', grid=rg)[1]
			
		else:
			rg.add_ones('node', 'river_length', dtype=float)
			rg.at_node['river_length'][:] *= rg.dx
			print('River network................. not provided as raster')
			print('All cells are considered rivers with length of grid size')
		
		riv = rg.add_ones('node', 'river', dtype=int)
		riv[np.where(rg.at_node['river_length'][:] <= 0)[0]] = 0
		self.riv_nodes = np.where(riv > 0)[0] # River nodes for domain arrays
		
		# Reading the raster file of river width		
		if not os.path.exists(inputfile.fname_RiverWidth):
			riv_width = rg.add_zeros('node', 'river_width', dtype=float)
			rg.at_node['river_width'][:] = 10.0
			rg.at_node['river_width'][rg.at_node['river_width'] > rg.dx] = np.array(rg.dx)
			print('River width................... not provided as raster. Global default applied of W = 10 m')
		else:
			riv_width = read_esri_ascii(inputfile.fname_RiverWidth,
				name = 'river_width', grid = rg)[1]
		
		# Reading the raster file of river elevation		
		if not os.path.exists(inputfile.fname_RiverElev):
			rg.add_zeros('node', 'river_topo_elevation', dtype=float)
			rg.at_node['river_topo_elevation'][:] = np.array(z) #[m]
			print('River bottom.................. not provided as raster')
			print('River bottom elevation: surface elevation')
		else:
			read_esri_ascii(inputfile.fname_RiverElev,
				name='river_topo_elevation',
				grid=rg)[1]
		
		# Reading a raster file of flow direction in LandLab format (receiving node ID)
		if os.path.exists(inputfile.fname_FlowDir):
			fd = read_esri_ascii(inputfile.fname_FlowDir,
				name='flow__receiver_node', grid=rg)[1]
			self.act_update_flow_director = False
		else:
			print('Flow direction................ not provided as raster')
			self.act_update_flow_director = True

		# Reading catchment cell areas
		if os.path.exists(inputfile.fname_Area):
			cth_area = read_esri_ascii(inputfile.fname_Area,
				name='cth_area_k', grid=rg)[1]
			
			area_aux = FlowAccumulator(rg, 'topographic__elevation',
										flow_director='D8',
										runoff_rate='cth_area_k')
					
			area_aux.accumulate_flow(
				update_flow_director=self.act_update_flow_director)	
			self.area_discharge = np.array(
				rg.at_node['surface_water__discharge'])
			rg.at_node['surface_water__discharge'][:] = 0
			self.run_flow_accum_areas = 1
		else:
			self.run_flow_accum_areas = 0
			cth_area = rg.add_ones('node', 'cth_area_k', dtype=float)
			rg.add_zeros('node', 'surface_water__discharge', dtype=float)
			#self.cth_area = np.array(cth_area)
			print('Cells factor area............. not provided as raster')
		
		self.cth_area = np.array(cth_area)
		
		# Catchment area: raster file of ceros and ones: ones represent the main cathment
		# The area can be the model domain or any area inside the model domain
		if not os.path.exists(inputfile.fname_Mask):
			mask = read_esri_ascii(inputfile.fname_DEM,
				name='basin', grid=rg)[1]
			print('Basin boundary................ not provided')
		else:
			mask = read_esri_ascii(inputfile.fname_Mask,
				name='basin', grid=rg)[1]
		
		# Reading Soil saturated hydraulic conductivity
		if not os.path.exists(inputfile.fname_Ksat_soil):
			rg.add_ones('node', 'Ksat_soil', dtype=float)
			print('Hydraulic conductivity........ not provided as raster. Global default applied of 1.0 mm/h')
		else:
			read_esri_ascii(inputfile.fname_Ksat_soil,
				name = 'Ksat_soil', grid = rg)[1]
		# Change units and applying scale factor kKs
		rg.at_node['Ksat_soil'] = (rg.at_node['Ksat_soil']*
				inputfile.unit_sim_k*inputfile.kKs)
				
		# Reading soil depth map: raster file [mm]
		if not os.path.exists(inputfile.fname_SoilDepth):
			rg.add_ones('node', 'Soil_depth', dtype=float)
			rg.at_node['Soil_depth'] *= 1000.0	# default value 1000 mm
			print('Rooting depth................. not provided as raster. Global default applied of 1000mm')
		else:
			read_esri_ascii(inputfile.fname_SoilDepth,
				name='Soil_depth', grid=rg)[1]
		# Applying scale factor kDroot
		rg.at_node['Soil_depth'] *= inputfile.kDroot
		self.Droot = np.array(rg.at_node['Soil_depth'])
		
		# Reading residual water content
		if not os.path.exists(inputfile.fname_theta_r): 			
			rg.add_zeros('node', 'theta_r', dtype=float)
			rg.at_node['theta_r'][:] += 0.025
			print('Residual moisture content..... not provided as raster. Global default applied of 0.025')
		else:
			read_esri_ascii(inputfile.fname_theta_r,
				name='theta_r', grid=rg)[1]

		
		# Reading Wilting point field
		if not os.path.exists(inputfile.fname_wp):
			rg.add_ones('node', 'wilting_point', dtype=float)
			rg.at_node['wilting_point'][:] = 0.05
			print('Wilting point................. not provided as raster. Global default applied of 0.05')
		else:
			read_esri_ascii(inputfile.fname_wp,
				name='wilting_point', grid=rg)[1]
				
		# Read Saturated water content (porosity)
		if not os.path.exists(inputfile.fname_n): 
			rg.add_ones('node', 'saturated_water_content', dtype=float) # Under unsaturated conditions [mm]
			rg.at_node['saturated_water_content'][:] = 0.40
			print('Porosity...................... not provided as raster. Global default applied of 0.4')
		else:
			read_esri_ascii(inputfile.fname_n, name='saturated_water_content', grid=rg)[1]
		
		# Reading available water content: raster file		
		if not os.path.exists(inputfile.fname_AWC):
			rg.add_zeros('node', 'AWC', dtype=float)
			rg.at_node['AWC'][:] = 0.10
			print('Available Water Content....... not provided as raster. Global default applied of 0.10')
		else:
			read_esri_ascii(inputfile.fname_AWC, name='AWC', grid=rg)[1]
		# Applying scale factor kAWC
		rg.at_node['AWC'] = rg.at_node['AWC']*inputfile.kAWC
					
		# Exponent for soil moisture - matrix potential relation
		# Rawls (1982), and Clapp and Hornberger (1978)
		if not os.path.exists(inputfile.fname_b_SOIL):			
			rg.add_zeros('node', 'b_SOIL', dtype=float)
			rg.at_node['b_SOIL'][:] = 10.05
			print('Soi particle distribution parameter not provided as raster. Global default applied of 10.5')
		else:
			read_esri_ascii(inputfile.fname_b_SOIL,
				name='b_SOIL', grid=rg)[1]
	
		# air-entry/saturated capillary potential, [mm]
		if not os.path.exists(inputfile.fname_PSI):
			rg.add_ones('node', 'PSI', dtype=float)
			rg.at_node['PSI'] = (rg.at_node['PSI'] * (
							rg.at_node['b_SOIL']*2+2.5)
							/ (rg.at_node['b_SOIL']+2.5))
			print('Suction head.................. not provided as raster. Global default applied of 1 mm')
		else:
			read_esri_ascii(inputfile.fname_PSI, name='PSI', grid=rg)[1]
		
		# Read Kc for transforming RET to PET
		if not os.path.exists(inputfile.fname_TSKc): 
			Kc = rg.add_ones('node', 'Kc', dtype=float)
			self.Kc = np.array(Kc)
			print('Kc vegetation................. not provided as raster. Global default applied of 1')
		else:
			Kc = read_esri_ascii(inputfile.fname_TSKc, name='Kc', grid=rg)[1]
			self.Kc = np.array(Kc)
		
		# Water bastraction component
		rg.add_zeros('node', 'AOF', dtype=float)
		rg.add_ones('node', 'AOFT', dtype=float)
		rg.at_node['AOFT'][:] = ABC_RIVER # additional threshold variable
		
		# ======================== GW - groundwater parameters ===================
		#read model domain of the GW model and merge it with the surface model
		if inputfile.run_GW == 1:
			if not os.path.exists(inputfile.fname_GWdomain):
				if os.path.exists(inputfile.fname_Mask):
					(gw, gwz) = read_esri_ascii(inputfile.fname_Mask,
								name='model_domain')
					print('GW domain ................ not provided, default surface basin')
					
				else:
					(gw, gwz) = read_esri_ascii(inputfile.fname_DEM,
								name='model_domain')
					print('GW domain ................ not provided, default surface basin')
				
			else:
				(gw, gwz) = read_esri_ascii(inputfile.fname_GWdomain, name='model_domain')
		else:
			if os.path.exists(inputfile.fname_Mask):
				(gw, gwz) = read_esri_ascii(inputfile.fname_Mask,
							name='model_domain')
				#print('Basin boundary................ not provided')
					
			else:
				(gw, gwz) = read_esri_ascii(inputfile.fname_DEM,
								name='model_domain')
			#(gw, gwz) = read_esri_ascii(inputfile.fname_Mask, name='model_domain')
		
		# Setting boundary conditions
		rg.status_at_node[rg.status_at_node == rg.BC_NODE_IS_FIXED_VALUE] = rg.BC_NODE_IS_CLOSED
		gw.status_at_node[gw.status_at_node == gw.BC_NODE_IS_FIXED_VALUE] = gw.BC_NODE_IS_CLOSED
		gw.status_at_node[np.where(gwz <= 0)[0]] = gw.BC_NODE_IS_CLOSED # Model domain of the GW
		rg.status_at_node[np.where(gwz <= 0)[0]] = rg.BC_NODE_IS_CLOSED # Model domain of the OF
		
		# ========================== Creating landlab fields =====================
		# Soil parameters fields
		rg.add_zeros('node', 'runoff', dtype=float)
		rg.add_zeros('node', 'percolation', dtype=float)
		rg.add_zeros('node', 'recharge', dtype=float)
		rg.add_zeros('node', 'PET', dtype=float)
		rg.add_zeros('node', 'Sorptivity', dtype=float)
	
		# Transmission losses parameters
		rg.add_zeros('node', 'Transmission_losses', dtype=float) # river transmission losses
		rg.add_zeros('node', 'Base_flow', dtype=float) # river transmission losses
		rg.add_zeros('node', 'decay_flow', dtype=float) # river decay flow parameter
		rg.add_zeros('node', 'AETp_riv', dtype=float) # Actual ET river
		rg.add_zeros('node', 'ETp_riv', dtype=float) # PET river
		rg.add_zeros('node', 'Q_ini', dtype=float)
		rg.add_ones('node', 'kTr_ini', dtype=float)		

		# Setting groundwater component
		if inputfile.run_GW == 1:
			print("Running Groundwater component")
			# Reading specific yield
			if not os.path.exists(inputfile.fname_SZ_Sy):				
				gw.add_zeros('node', 'SZ_Sy', dtype=float) # Water table elevation
				gw.at_node['SZ_Sy'] += 0.01
				print('Specific yield................ not provided as raster. Global default applied of 0.01')
			else:
				read_esri_ascii(inputfile.fname_SZ_Sy,
					name='SZ_Sy', grid=gw)[1]
			# Applying scale factor kSy
			gw.at_node['SZ_Sy'] *= inputfile.kSy_gw
			
			# Aquifer bottom
			if not os.path.exists(inputfile.fname_SZ_bot): 
				gw.add_zeros('node', 'BOT', dtype = float)
				#gw.at_node['BOT'] = z*0.0#+1450.0 # Defailt values of aquifer bottom 
				print('Aquifer bottom elevation...... not provided as raster. Global default applied of 0.0 m')
			else:
				SZ_bot = read_esri_ascii(inputfile.fname_SZ_bot,
					name='BOT', grid=gw)[1]
			
			# Aquifer Saturated hydraulic conductivity
			if not os.path.exists(inputfile.fname_SZ_Ksat): 
				gw.add_ones('node', 'Hydraulic_Conductivity', dtype=float)
				#gw.at_node['Hydraulic_Conductivity'] += 1.0 # Defailt values of aquifer bottom 
				print('Aquifer Ksat.................. not provided as raster. Global default applied of 1.0 m/h')
			else:
				read_esri_ascii(inputfile.fname_SZ_Ksat, name = 'Hydraulic_Conductivity', grid=gw)[1]
			# Applying scale factor kKsat_gw
			gw.at_node['Hydraulic_Conductivity'] *= inputfile.kKsat_gw
							
			# Check if flux boundary is provided m/h
			if not os.path.exists(inputfile.fname_FHB):
				gw.add_zeros('node', 'SZ_FHB', dtype=float)
				print('Flux boundary conditions. not provided as raster')
			else:
				SZ_CHBa = read_esri_ascii(inputfile.fname_FHB,
					name='SZ_FHB', grid=gw)[1]
				gw.at_node['SZ_FHB'][gw.at_node['SZ_FHB'] == -9999] = 0				
				#gw.at_node['SZ_FHB'][:] = gw.at_node['SZ_FHB'][:]*2/(np.power(rg.dx, 2))
			gw.at_node['SZ_FHB'][gw.status_at_node[gw.status_at_node == gw.BC_NODE_IS_CLOSED]] = 0
			
			# Read parameters for calulating  effective thickness
			# a: numerator, and b: denominator
			# Read aquifer paramter a for calculating effective thickness
			if not os.path.exists(inputfile.fname_a_aq):
				gw.add_zeros('node', 'SZ_a_aq', dtype=float)
				gw.at_node['SZ_a_aq'][:] = 50.0
				print('Not available a parameter aquifer, a=50m')
			else:
				SZ_CHBa = read_esri_ascii(inputfile.fname_a_aq,
					name='SZ_a_aq', grid=gw)[1]
				
			# Read aquifer parameter b for calculating effective thickness
			if not os.path.exists(inputfile.fname_b_aq):
				gw.add_zeros('node', 'SZ_b_aq', dtype=float)
				print('Not available b parameter aquifer, b=0')
			else:
				SZ_CHBa = read_esri_ascii(inputfile.fname_b_aq,
					name='SZ_b_aq', grid=gw)[1]
						
			# Initial water table depth
			print("reading initial water table elevation...")
			if not os.path.exists(inputfile.fname_GWini): 
				h = gw.add_zeros('node', 'water_table__elevation', dtype=float)
				gw.at_node['water_table__elevation'] = z - rg.at_node['Soil_depth']*0.001
				print('Initial water table elevation. not provided as raster')
				print('Initial water table elevation assumed equal to root depth elevation')
			else:
				h = read_esri_ascii(inputfile.fname_GWini,
					name='water_table__elevation', grid=gw)[1]

			#qs = gw.add_zeros('link', 'unit_flux', dtype = float) # Unit flux for GW model
			# Check if constant head boundary is provided
			if not os.path.exists(inputfile.fname_CHB):				
				print('Constant head boundary conditions not provided as raster')
			else:
				SZ_CHBa = read_esri_ascii(inputfile.fname_CHB,
					name='SZ_CHB', grid=gw)[1]
				id_CHB = np.where(gw.at_node['SZ_CHB'] != -9999)[0]
				gw.at_node['water_table__elevation'][id_CHB] = gw.at_node['SZ_CHB'][id_CHB]
				gw.status_at_node[id_CHB] = gw.BC_NODE_IS_FIXED_VALUE

			
			# Calculating intial groundwater river deficit
			rg.add_ones('node', 'riv_sat_deficit', dtype=float)
			root_aux = np.array(z - h)*1000
			root_aux = np.where(root_aux > np.array(rg.at_node['Soil_depth']),
					np.array(rg.at_node['Soil_depth']), root_aux)
			rg.at_node['riv_sat_deficit'] = (z - h)*np.power(rg.dx, 2)
			self.Duz = np.array(root_aux)
			
		else:
			print("Groundwater component not used")
			h = z - rg.at_node['Soil_depth']*0.001
			gw.add_field('water_table__elevation', np.array(h), at="node")
			self.Duz = np.array(z-h)*1000.0#rg.at_node['Soil_depth'])
			rg.add_ones('node', 'riv_sat_deficit', dtype=float)
			rg.at_node['riv_sat_deficit'][:] = 100*np.power(rg.dx, 2)
			
		self.SZgrid = gw		
		
	# ======================== Defining core cells ========================
		# defining nodes to reduce the number of operation
		act_nodes = rg.core_nodes
		aux_mask = np.zeros_like(z)
		aux_mask[act_nodes] = 1
		mask = np.where(mask > 0, 1, 0)
		self.mask = np.where(mask > 0, aux_mask, mask)
		
		# nodes inside the catchement for domain arrays
		self.basin_nodes = np.where(self.mask > 0)[0]
		
		# Nodes inside the catchment for active node arrays
		self.basin_ids_core = np.where(self.mask[act_nodes] > 0)[0]
		
		# river nodes inside the basin for active node arrays
		self.river_ids_core = np.where(riv[act_nodes] == 1)[0]
		
		# river nodes inside the basin for domain arrays
		self.river_ids_nodes = act_nodes[self.river_ids_core]		
		# Nodes inside the catchment for active node arrays
		#self.river_ids_cath = np.where((riv[act_nodes])[self.basin_ids_core] == 1)[0]
		# river nodes inside the catchment for domain arrays
		#self.river_ids_cath_node = self.basin_nodes[np.where(riv[self.basin_nodes] > 0)[0]]
		
		# Field capacity (m3/m3))
		fc = np.array(rg.at_node['wilting_point']+rg.at_node['AWC']) 
		
		# Saturated water content [mm]
		self.Lsat = np.array(rg.at_node['Soil_depth']
			*rg.at_node['saturated_water_content'])
		
		#Water content at wilting point (mm)
		Lwp = np.array(rg.at_node['wilting_point']
			*rg.at_node['Soil_depth'])
		
		#Water content at field capacity (mm)
		Lfc = np.array(fc*rg.at_node['Soil_depth'])
		
		# Exponent c for Rawls (1982), and Clapp and Hornberger (1978)
		# c_SOIL = np.array(rg.at_node['b_SOIL']2+2.5
		# Campbell (1974)
		c_SOIL = 2/np.array(rg.at_node['b_SOIL']) + 3
		
		# Channel hydraulic parameters
		# Assuming a flow velocity of 1 m/s => 3600 m/h
		rg.at_node['decay_flow'][:] = (inputfile.kT_units*3600.0
									*inputfile.T_loss/rg.dx)
		#river_banks = 30.0 # It is hard coded for now and will be pass as a raster grid
		
		if not os.path.exists(inputfile.fname_Ksat_ch):			
			rg.add_ones('node', 'Ksat_ch', dtype=float)
			print('Channel Ksat.................. not provided')
			print('Assumed equal to soil Ksat')
		else:		
			read_esri_ascii(inputfile.fname_Ksat_ch, name='Ksat_ch', grid=rg)[1]
		
		# Applying unit factor
		rg.at_node['Ksat_ch'] = (rg.at_node['Ksat_ch']*0.001
								*inputfile.Kloss
								*self.mask)		#-> m/h
		rg.at_node['SS_loss'] = (rg.at_node['river_width']
								*rg.at_node['Ksat_ch']
								*rg.at_node['river_length']) # m3/h
		
		# INITIAL CONDITIONS ---------------------------------------------------------------
		# Initial water content as volumetric fraction
		print("Reading initial conditons")
		print("Soil moisture...")
		if not os.path.exists(inputfile.fname_theta):			
			rg.add_zeros('node','Soil_Moisture', dtype=float)			
			rg.at_node['Soil_Moisture'][:] = np.array(rg.at_node['wilting_point'])*1.01			
			print('Initial soil moisture...... not provided as raster. Global default applied of 1.01*wp')			
		else:		
			read_esri_ascii(inputfile.fname_theta, name='Soil_Moisture', grid=rg)[1]
		
		print("Riparian soil moisture...")
		if not os.path.exists(inputfile.fname_theta):			
			rg.add_zeros('node','pSoil_Moisture', dtype=float)			
			rg.at_node['pSoil_Moisture'][:] = np.array(rg.at_node['wilting_point'])			
			print('Initial riparian moisture...... not provided as raster. Global default applied of 1.01*wp')			
		else:		
			read_esri_ascii(inputfile.fname_theta, name='pSoil_Moisture', grid=rg)[1]
		
		
		# Water content soil[mm]
		self.tht = np.array(rg.at_node['Soil_Moisture'][:])	
		self.L_0 = np.array(rg.at_node['Soil_Moisture'][:])*self.Duz		
		self.t_0 = np.zeros_like(self.L_0)		
		self.Ft_0 = np.zeros_like(self.L_0)		
		self.SORP0 = np.zeros_like(self.L_0)
		
		# Water content riparian area []
		self.ptht = np.array(rg.at_node['pSoil_Moisture'][:])	
		self.ptht = np.where(h >= rg.at_node['river_topo_elevation' ],
				rg.at_node['wilting_point']+rg.at_node['AWC'],
				self.ptht)
		
		ds = np.array(rg.at_node['Soil_depth'][act_nodes])
		
		# threshold capillary rise
		self.gwet_lim = ((rg.at_node['wilting_point']+0.5*rg.at_node['AWC'])/
				rg.at_node['saturated_water_content'])
		
		# Initial soil moisture deficit
		SMD_0 = np.array(Lfc-self.L_0)		
		self.SMD_0 = np.where(SMD_0 < 0.0, 0.0, SMD_0)		
		self.SMDh = SMD_0		
		#SMD_riv_0 = np.array(Lfc[self.river_ids_nodes]-self.L_0[self.river_ids_nodes])		
		#self.SMD_riv_0 = np.where(SMD_riv_0 < 0.0, 0.0, SMD_riv_0)
						
		self.L0_riv = np.array(self.L_0[self.river_ids_nodes])
		self.grid = rg
		self.act_nodes = act_nodes
		self.grid_size = len(z)
		self.c_SOIL = c_SOIL
		self.fc = fc
		
		# calculating cells area
		self.area_cells = rg.dx*rg.dy*rg.at_node['cth_area_k']
		
		self.area_cells_hills = rg.dx*rg.dy*rg.at_node['cth_area_k']
		self.area_cells_banks = np.zeros_like(z)
		
		self.area_catch_factor = (rg.at_node['cth_area_k']
			/ np.sum(rg.at_node['cth_area_k'][self.basin_nodes]))
		
		self.area_river_factor = np.zeros_like(z)
		self.area_river_factor[self.river_ids_nodes] = 1 / np.sum(rg.at_node['cth_area_k'][self.basin_nodes])
		
		if rg.dx > inputfile.river_banks:		
			self.area_cells_hills[self.riv_nodes] = (
				self.area_cells_hills[self.riv_nodes]
				- rg.at_node['river_length'][self.riv_nodes]
				* (riv_width[self.riv_nodes]+2*inputfile.river_banks)
				)
	
			self.area_cells_banks = self.area_cells-self.area_cells_hills
		else:
			self.area_cells_hills[self.riv_nodes] = 0.0
			self.area_cells_banks[self.riv_nodes] = (rg.dx
				*rg.dy*rg.at_node['cth_area_k'][self.riv_nodes]
				)
		
		# catchment cells
		self.mask_grid = np.ones(len(aux_mask), dtype=int) - aux_mask
		
		# riparian area factor [-]
		# to pass from rip_cell to model_cell
		self.riv_factor = aux_mask*self.area_cells_banks/self.area_cells
		self.inv_riv_factor = np.zeros_like(self.riv_factor)
		self.inv_riv_factor[self.area_cells_banks > 0] =(
			self.area_cells[self.area_cells_banks > 0]/
			self.area_cells_banks[self.area_cells_banks > 0])
		
		# hillslope area cell factor [-]
		self.hill_factor = aux_mask*self.area_cells_hills/self.area_cells
		
		# proportion of riparian area [-]
		# to pass from [m3] -> [mm]
		self.rip_factor = np.zeros_like(z)
		self.rip_factor[self.area_cells_banks != 0] = (
			1000./self.area_cells_banks[self.area_cells_banks != 0])
				
		# riparian area [m2]
		self.rarea = np.array(rg.at_node['river_width']*rg.at_node['river_length'])
		
		# Groundwater selection function parameter
		self.func = int(inputfile.gw_func)
		
		
	# Create directories for saving results	
	def set_output_dir(self, inputfile):
		"""
		Directory *DirOutput* Created
		Directory *outputcirnetcdf* already exists
		"""
		#print('************************************************************')
		print('********************* Output Directory **********************')
		if not os.path.exists(inputfile.DirOutput):
			os.mkdir(inputfile.DirOutput)
			print("Directory ", inputfile.DirOutput, " Created ")
		else:
			print("Directory ", inputfile.DirOutput, " already exists")
		
		output_dir_nc = inputfile.DirOutput+'/TimeFrames'
		
		if not os.path.exists(output_dir_nc):
			os.mkdir(output_dir_nc)
			print("Directory ", output_dir_nc, " Created ")
		else:
			print("Directory ", output_dir_nc, " already exists")
		
		# Output filenames
		self.fnameTS_avg = inputfile.DirOutput+'/' + inputfile.Mname + '_avg'
		self.fnameTS_OF  = inputfile.DirOutput+'/' + inputfile.Mname + '_OF_'
		self.fnameTS_UZ  = inputfile.DirOutput+'/' + inputfile.Mname + '_UZ_'
		self.fnameTS_GW  = inputfile.DirOutput+'/' + inputfile.Mname + '_GW_'
		
	# Find coordinates of points in model components
	def points_output(self, inputfile):
		"""
		Read data output points for model results		
		"""				
		# -------------Reading output points and creating variables for storing outputs-----------
		fOF = pd.read_csv(inputfile.fname_DISpoints)
		fUZ = pd.read_csv(inputfile.fname_SMDpoints)
		fGW = pd.read_csv(inputfile.fname_GWpoints)
		
		npointsOF = len(fOF['North'])
		npointsUZ = len(fUZ['North'])
		npointsGW = len(fGW['North'])

		gaugeidOF = []
		gaugeidUZ = []
		gaugeidGW = []
		
		OF_label = []
		UZ_label = []
		GW_label = []
		
		#	Overland component points and variables
		for ndis in range(npointsOF):
			gaugeidOF.append(self.grid.find_nearest_node([fOF['East'][ndis], fOF['North'][ndis]]))
			OF_label.append('OF_'+str(ndis))
		
		#	Unsaturated component points and variables
		for nUZ in range(npointsUZ):
			gaugeidUZ.append(self.grid.find_nearest_node([fUZ['East'][nUZ], fUZ['North'][nUZ],]))
			UZ_label.append('UZ_'+str(nUZ))
		
		#	Saturated components points and variables
		for nGW in range(npointsGW):
			gaugeidGW.append(self.grid.find_nearest_node([fGW['East'][nGW], fGW['North'][nGW]]))
			GW_label.append('SZ_'+str(nGW))
		
		ncell_a = np.int(700/self.grid.dx)+1 # number of cells around station
		
		# Finding core cells, river cells and basin cells
		act_nodes = self.grid.core_nodes # Nodes inside the model domain
		
		if inputfile.first_read == 1:
		
			#create a range of cell for cosmos probe			
			cosmos_ids = []			
			cosmos_ids_core = []
			
			if ncell_a == 0:				
				cosmos_ids_core = np.where(self.grid.core_nodes == gaugeidUZ[0])[0]				
				cosmos_ids_core = cosmos_ids_core.astype(int)				
				cosmos_ids = gaugeidUZ[0].astype(int)			
			else:				
				for cos_id in range(gaugeidUZ[0]-ncell_a*self.grid.shape[1], gaugeidUZ[0]+ncell_a*self.grid.shape[1],self.grid.shape[1]):				
					for x in range(cos_id-ncell_a, cos_id+ncell_a):					
						cosmos_ids.append(x)
					
				self.cosmos_ids_core_mask = np.isin(act_nodes, cosmos_ids)							
				self.cosmos_ids = act_nodes[self.cosmos_ids_core_mask]
				cosmos_ids_core_aux = np.array(range(len(act_nodes)))				
				self.cosmos_ids_core_mask[self.cosmos_ids_core_mask == True] = 1				
				self.cosmos_ids_core = np.where(self.cosmos_ids_core_mask == 1)[0]
			
		gaugeidUZ_act = []
		
		for dUZgi in range(len(gaugeidUZ)):		
			gaugeidUZ_act.append(np.where(self.grid.core_nodes == gaugeidUZ[dUZgi])[0])
		
		self.gaugeidOF = gaugeidOF
		self.gaugeidUZ = gaugeidUZ
		self.gaugeidGW = gaugeidGW
		
		#print(self.gaugeidOF)
		#print(self.gaugeidUZ)
		#print(self.gaugeidGW)
	
	def save_data_to_file(inputfile, model_environment_status):
		"""
		Do any saving here
		"""
		pass