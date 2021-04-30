import numpy as np

class swbm(object):

	def __init__(self, env_state, data_in):
		# AET_dt:	Actual evapotranspiration
		# THT_dt:	Soil water content [-]
		# PCL_dt:	Percolation [mm]
		
		self.aet_dt = np.zeros(env_state.grid_size)		
		self.gwe_dt = np.zeros(env_state.grid_size)		
		self.sro_dt = np.zeros(env_state.grid_size)		
		self.tht_dt = np.array(env_state.grid.at_node['Soil_Moisture'][:])		
		self.pcl_dt = np.zeros(env_state.grid_size)
		self.WRSI = np.zeros(env_state.grid_size)
		self.Duz = env_state.Duz		
		self.L_0 = env_state.Duz*self.tht_dt
	
	def run_soil_aquifer_one_step(self, env_state, sur_elev, wt_elev, Duz0, tht_dt):
		"""	Update depth of the unsaturated soil depending on the
		water table elevation.
		tht_dt:		Soil water content at time t [-]
		env_state:	grid:	z:		Topograhic elevation
							h:		water table
							fs:		Saturated water content
							fc:		Field capacity
							Sy:		Specific yield
							dq:		water storage anomaly
							Droot:	Rooting depth [mm]		
		Return:
		Duz:	Unsaturated zone depth
		tht:	Updated soil ater content		
		"""
		Duz = variable_soil_depth(sur_elev,wt_elev,env_state.Droot)
		tht = np.where((Duz0-Duz) > 0.0,tht_dt*Duz,(Duz0*tht_dt-(Duz0-Duz)*env_state.fc))
		tht[Duz > 0] = tht[Duz > 0]/Duz[Duz > 0]
		tht[Duz <= 0] = env_state.grid.at_node['wilting_point'][Duz <= 0]
		self.tht = tht
		self.L_0 = tht*Duz
		self.Duz = Duz
			
	def run_swbm_one_step(self, inf, pet, Kc, Ksat, env_state, data_in, *nodes):
		"""Run soil water balance
		"""
		if nodes:		
			act_nodes = nodes[0]		
		else:		
			act_nodes = env_state.act_nodes
		nodes_aux = np.where(env_state.Duz[act_nodes] > 0.0)[0]		
		act_nodes = act_nodes[nodes_aux]
		del nodes_aux #to reduce memory use
		
		inf_dt = inf[act_nodes]		
		Kc = Kc[act_nodes]		
		Ks = Ksat[act_nodes]		
		PET = np.array(pet[act_nodes])		
		L_0 = self.L_0[act_nodes]		
		SMD_0 = env_state.SMD_0[act_nodes]		
		ds = self.Duz[act_nodes]
		fs = env_state.grid.at_node['saturated_water_content'][act_nodes]		
		fc = env_state.fc[act_nodes]		
		wp = env_state.grid.at_node['wilting_point'][act_nodes]		
		c = env_state.c_SOIL[act_nodes]
		
		self.sro_dt *= 0.0		
		self.pcl_dt *= 0.0
		
		if data_in.dt >= 1440:				
			AET, SMD_dt, D, L, RO = SWBM(inf_dt,PET,Kc,L_0,SMD_0,ds,fs,fc,wp)				
		else:				
			AET, SMD_dt, D, L, RO = SWBMh(inf_dt,PET,Kc,L_0,SMD_0,ds,fs,fc,wp,c,Ks)
		
		self.aet_dt[act_nodes] = np.array(AET)		
		self.pcl_dt[act_nodes] = np.array(D)		
		self.tht_dt[act_nodes] = np.array(L/ds)		
		self.sro_dt[act_nodes] = RO
		WSRI = np.zeros(len(act_nodes))
		WSRI[PET != 0] = AET[PET != 0]/PET[PET != 0]
		self.WRSI[act_nodes] = WSRI
		self.gwe_dt = pet - self.aet_dt		
		self.L_0[act_nodes] = np.array(L)
	
	def run_swb_lat_flow_one_step(self, env_state_soil, env_state_rip):
		""" This function estimate lateral flow in river cell, it is
		assumed that channel width is less than river cell size
		env_state_soil:	Soil water content at river cell
		env_state_rip:	Water content at riparian river cell		
		"""
		river_cell = env_state_soil.river_ids_nodes
		L_riv = env_state_rip.L_0[river_cell]
		L_soil = env_state_soil.L_0[river_cell]
		K_soil = np.array(env_state_soil.grid.at_node['Ksat_soil'][river_cell])
		K_riv = np.array(env_state_soil.grid.at_node['Ksat_soil'][river_cell])
		c_SOIL = env_state_soil.c_SOIL[river_cell]
		Ks_theta = K_s*np.power(theta/theta_sat,c_SOIL)
		L = lateral_flow_river_cell(L_riv,L_soil,Ks_theta,dx)
		env_state_rip.L_0[river_cell] += -L 
		env_state_soil.L_0[river_cell] += L
		
	def storage(self, inf_dt, env_state, *nodes):
		"""Calculate storage
		parameters:
			inf_dt:	infiltration rate
			*nodes:	nodes to consider in total storage
		Outputs:
			Total water storage at unsaturated zone
		"""
		if nodes:
			act_nodes = nodes[0]
		else:
			act_nodes = env_state.act_nodes
		str_uz = self.L_0 - (inf_dt -(self.pcl_dt + self.aet_dt + self.sro_dt))
		return np.sum(str_uz[act_nodes])
	
	def storage_ini(self, env_state, *nodes):
		if nodes:
			act_nodes = nodes[0]
		else:
			act_nodes = env_state.act_nodes
		return np.sum(str_uz[act_nodes])
		
# Soil water balance model for daily time steps - constant Kc
# This model assumes that percolation occurrs immediately after precipitation 
def SWBM(I, PET, Kc, L0, SMD_0, z_soil, fs, fc, wp):
	"""Soil water balance
	Parameters:
		I:	Infiltration
		PET:	Potential evapotranspiration
		L0:	initial water content
		SMD_0:Initial soil moisture deficit
		z_soil:Soil depth variation
		fs:	Soil moisture at saturated conditions
		fc:	Soil moisture at field capacity
		wp:	Soil moisture at wilting point
		c:	Exponential recession term estimated as
		d:	Soil parameter par_swb for reducing time steps
		z:    Surface elevation
		h:    Water table elevation
		Droot:Soil rooting depth
		Lsat:	Water content at saturated condition	
	Outouts:
		AET:	Actual evpotranspiration
		D:		Drainage
		L:		Water content
	"""
	PET = Kc*PET
	Pc = 0.5
	Lsat = z_soil*fs
	Lwp = z_soil*wp
	Lfc =  z_soil*fc
	TAW =  Lfc - Lwp
	RAW = Pc * TAW
	L_RAW = Lfc-RAW
	L_TAW = Lfc-TAW
	
	beta = (L0-L_TAW) / (L_RAW-L_TAW)
	beta[beta > 1] = 1
	beta[beta < 0] = 0
	
	I_AET = np.where(I > PET,PET,I)
	AET = I_AET*(1-beta)+beta*PET
	AET[AET < 0] = 0
	L = L0+I-AET
	AET = np.where((L-Lwp) < 0,L0+I-Lwp,AET)
	L = L0+I-AET
	D = np.where(L-Lfc > 0.0,L-Lfc,0.0)
	L = L-D
	SMD = np.where(L < Lfc,SMD_0+AET-I,0)
	RO = np.where(L > Lsat,L-Lsat,0)
	return AET, SMD, D, L, RO

# Soil moisture water balance model for subhourly time steps - Variable Kc
def SWBMh(I, PET, Kc, L0, SMD_0, z_soil, fs, fc, wp, c, Ksat):
	"""Soil water balance
	Parameters:
		I:	Infiltration
		PET:	Potential evapotranspiration
		L0:	initial water content
		SMD_0:Initial soil moisture deficit
		z_soil:Soil depth variation
		fs:	Soil moisture at saturated conditions
		fc:	Soil moisture at field capacity
		wp:	Soil moisture at wilting point
		c:	Exponential recession term estimated as
		d:	Soil parameter par_swb for reducing time steps
		z:    Surface elevation
		h:    Water table elevation
		Droot:Soil rooting depth
		Lsat:	Water content at saturated condition	
	Outouts:
		AET:	Actual evpotranspiration
		D:		Drainage
		L:		Water content
	"""
	kd = (c-1)*Ksat/(z_soil*np.power(fs,c))
	PET = Kc*PET
	Pc = 0.5
	Lsat = z_soil*fs
	Lwp = z_soil*wp
	Lfc = z_soil*fc
	TAW = Lfc - Lwp
	RAW = Pc * TAW
	L_RAW = Lfc - RAW
	L_TAW = Lfc - TAW
	beta = (L0-L_TAW) / (L_RAW-L_TAW)
	beta[beta > 1] = 1
	beta[beta < 0] = 0
	I_AET = np.where(I > PET, PET, I)
	AET = I_AET*(1-beta)+beta*PET
	AET[AET < 0] = 0
	L_aux = L0+I-AET
	# This will not allow the soil misture to be less than wilting_point
	AET = np.where((L_aux-Lwp) < 0, L0+I-Lwp, AET)
	L_aux = L0 + I - AET
	RO = np.where(L_aux > Lsat,L_aux-Lsat, 0)
	L_aux -= RO
	DL = np.where(L_aux > Lfc, z_soil*np.exp((-c+1)*np.log(np.power(L_aux/z_soil,-c+1)+kd))-Lfc,0.0)
	DL = np.where(DL < 0.0, L_aux-Lfc, 0.0)
	L = L_aux - DL
	D = L_aux - L + RO
	SMD = np.where(L < Lfc, SMD_0+AET-I, 0)	
	return AET, SMD, D, L, RO

# Variable soil depth
def variable_soil_depth(z, h, Droot):
	# Luz	: Soil depth variation
	# z     : Surface elevation
	# h     : Water table elevation
	# Droot	: Soil rooting depth
	# Lsat	: Water content at saturated condition
	Zs = z - Droot*0.001
	Duz = np.where((Zs - h) >= 0,Droot,(z - h)*1000.0)
	return np.where(Duz <= 0, 0, Duz)
	
# Schaake infiltration approach, Schaake et. al. (1996)
# Soil moisture water balance model for subhourly time steps
# Needs numerical integrator to work RK4Sh
def SWBSh(P, PET, L, depth, Ks, Lsat, Lfc, Lwp, Pc, c, ga_kdt):
	"""Sheeke infiltration rates	
	"""
	TAW =  Lfc - Lwp
	RAW = Pc * TAW
	L_RAW = Lfc-RAW
	L_TAW = Lfc-TAW
	D = Lsat-L
	
	I_aux = D*ga_kdt
	I = P*I_aux/(P+I_aux)
	I = np.where(P+I_aux == 0.0,0.0,I)
		
	beta = (L-L_TAW) / (L_RAW-L_TAW)
	beta[beta > 1] = 1
	
	I_AET = np.where(I > PET,PET,I)
	AET = I_AET*(1-beta)+beta*PET
	R = np.where(L-Lfc > 0.0,Ks*(L/Lsat)**c,0.0)
	dL = I-AET-R
	return dL, I, AET, R
	
# Runge Kutta apprach for solving soil moisture SWBSh
def RK4Sh(P, PET, L0, depth, Ks, Lsat, Lfc, Lwp, Pc, kdt, c, dt):
	k1, I, AET, R = SWBSh(P, PET, L0, depth, Ks, Lsat, Lfc, Lwp, Pc, c, kdt)
	k2, I, AET, R = SWBSh(P, PET, L0+k1/3, depth, Ks, Lsat, Lfc, Lwp, Pc, c, kdt)
	k3, I, AET, R = SWBSh(P, PET, L0+k2/3, depth, Ks, Lsat, Lfc, Lwp, Pc, c, kdt)
	k4, I, AET, R = SWBSh(P, PET, L0+k3, depth, Ks, Lsat, Lfc, Lwp, Pc, c, kdt)
	L = L0 + k1/6 + k2/3 + k3/3 + k4/6
	k, I, AET, R = SWBSh(P, PET, L, depth, Ks, Lsat, Lfc, Lwp, Pc, c, kdt)
	return L,I,AET,R

# Soil water balance model for subdaily time steps
# Infiltration input
# Needs numerical integrator to work RK4Ph
def SWBPh(P, PET, L, depth, Ks, Lsat, Lfc, Lwp, Pc, c, ga_b):
	TAW =  Lfc - Lwp
	RAW = Pc * TAW
	L_RAW = Lfc-RAW
	L_TAW = Lfc-TAW
	D = Lsat-L

	I_aux = D*ga_kdt
	I = P*I_aux/(P+I_aux)
	I = np.where(P+I_aux == 0.0,0.0,I)

	beta = (L-L_TAW) / (L_RAW-L_TAW)
	beta[beta > 1] = 1

	I_AET = np.where(I > PET,PET,I)
	AET = I_AET*(1-beta)+beta*PET
	R = np.where(L-Lfc > 0.0,Ks*(L/Lsat)**c,0.0)
	dL = I-AET-R
	return dL, I, AET, R

# Runge Kutta apprach for solving soil moisture SWBPh
def RK4Ph(P, PET, L0, depth, Ks, Lsat, Lfc, Lwp, Pc, kdt, c, dt):
	k1, I, AET, R = SWBPh(P, PET, L0, depth, Ks, Lsat, Lfc, Lwp, Pc, c, kdt)
	k2, I, AET, R = SWBPh(P, PET, L0+k1/3,depth, Ks, Lsat, Lfc, Lwp, Pc, c, kdt)
	k3, I, AET, R = SWBPh(P, PET, L0+k2/3,depth, Ks, Lsat, Lfc, Lwp, Pc, c, kdt)
	k4, I, AET, R = SWBPh(P, PET, L0+k3, depth, Ks, Lsat, Lfc, Lwp, Pc, c, kdt)
	L = L0 + k1/6 + k2/3 + k3/3 + k4/6
	k, I, AET, R  = SWBPh(P, PET, L, depth, Ks, Lsat, Lfc, Lwp, Pc, c, kdt)
	return L, I, AET, R
	
def lateral_flow_river_cell(L_riv,L_soil,Ktheta,dx):
	return (L_riv - L_soil)*np.exp(Ktheta/d_sr)+L_soil
