import numpy as np

class swbm(object):

	def __init__(self, env_state, Duz, tht_t0, data_in, *layer):
		
		# AET_dt:	Actual evapotranspiration
		# THT_dt:	Soil water content [-]
		# PCL_dt:	Percolation [mm]
		# layer:	up to 2 layer, default 1
		
		self.aet_dt = np.zeros(env_state.grid_size)		
		self.gwe_dt = np.zeros(env_state.grid_size)		
		self.sro_dt = np.zeros(env_state.grid_size)		
		self.tht_dt = np.array(tht_t0)#env_state.grid.at_node['Soil_Moisture'][:])		
		self.pcl_dt = np.zeros(env_state.grid_size)
		self.WRSI = np.zeros(env_state.grid_size)
		self.Duz = np.array(Duz)
		self.L_0 = np.array(Duz)*self.tht_dt
		self.smd_uz = np.array(
			env_state.grid.at_node['wilting_point']
			+ env_state.grid.at_node['AWC']
			- self.tht_dt)*self.Duz
		
		# activate second layer for soil moisture
		if layer:
			self.two_layer = 1
		else:
			self.two_layer = 0
			
		if self.two_layer == 1:
			# call the one-layer model
			self.L_0l = np.array(Duz)*self.tht_dt
			self.thtl_dt = np.array(tht_t0)
		
		
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
		# update rooting depth
		Duz = variable_soil_depth(sur_elev, wt_elev, env_state.Droot)
		
		# update soil water content
		tht = np.where((Duz0-Duz) > 0.0,
				tht_dt*Duz,
				(Duz0*tht_dt-(Duz0-Duz)*env_state.fc))
				
		tht[Duz > 0] = tht[Duz > 0]/Duz[Duz > 0]
		tht[Duz <= 0] = env_state.grid.at_node['wilting_point'][Duz <= 0]
		
		# save values
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
		
		# Pass variables only with rooting depth greater than zero
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
		
		# run two-layer model
		if self.two_layer == 0:
			# call the one-layer model
			if data_in.dt >= 1440:				
				AET, SMD_dt, D, L, RO = SWBM(inf_dt, PET, Kc,
										L_0, SMD_0, ds,
										fs, fc, wp
										)				
			else:				
				AET, SMD_dt, D, L, RO = SWBMh(inf_dt, PET, Kc,
										L_0, SMD_0, ds,
										fs, fc, wp, c, Ks
										)
			#print(PET)
		else:
			
			# water content of the lower soil layer
			L_0l = self.L_0l[act_nodes]
			#print(L_0l)
			# call two-layer model solver
			L, Ll, SMD_dt, E, T, D, RO = FAO2L(inf_dt, PET, Kc,
										L_0, L_0l,
										ds, ds,
										fs, fc, wp,
										c, Ks)
			
			self.L_0l[act_nodes] = np.array(Ll)
			self.thtl_dt[act_nodes] = np.array(Ll/ds)	
			#print(D)
			AET = E + T
		
		
		self.aet_dt[act_nodes] = np.array(AET)		
		self.pcl_dt[act_nodes] = np.array(D)		
		self.tht_dt[act_nodes] = np.array(L/ds)		
		self.sro_dt[act_nodes] = RO
		
		# percolation for cells with depth zero
		aux_pcl = np.array(inf - pet)
		aux_pcl[aux_pcl < 0] = 0
		aux_pcl[self.Duz > 0] = 0
		self.pcl_dt += aux_pcl
		
		aux_aet = np.array(inf - pet)
		aux_aet[aux_aet <= 0] = np.array(inf[aux_aet <= 0])
		aux_aet[aux_aet > 0] = np.array(pet[aux_aet > 0])
		aux_aet[self.Duz > 0] = 0
		self.aet_dt += aux_aet
		
		WSRI = np.zeros_like(act_nodes)
		WSRI[PET != 0] = AET[PET != 0]/PET[PET != 0]
		
		self.WRSI[act_nodes] = WSRI
		self.gwe_dt = pet - self.aet_dt		
		self.L_0[act_nodes] = np.array(L)
		self.smd_uz[act_nodes] = SMD_dt 
	
	def run_swb_lat_flow_one_step(self, env_state_soil, env_state_rip):
		""" This function estimate lateral flow in river cell, it is
		assumed that channel width is less than river cell size
		env_state_soil:	Soil water content at river cell
		env_state_rip:	Water content at riparian river cell		
		"""
		river_cell = env_state_soil.river_ids_nodes
		L_riv = env_state_rip.L_0[river_cell]
		L_soil = env_state_soil.L_0[river_cell]
		K_soil = np.array(
			env_state_soil.grid.at_node['Ksat_soil'][river_cell])
		K_riv = np.array(
			env_state_soil.grid.at_node['Ksat_soil'][river_cell])
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
		str_uz = self.L_0 - (inf_dt -
			(self.pcl_dt + self.aet_dt + self.sro_dt))
			
		return np.sum(str_uz[act_nodes])
	
	def storage_ini(self, env_state, *nodes):
		if nodes:
			act_nodes = nodes[0]
		else:
			act_nodes = env_state.act_nodes
		return np.sum(str_uz[act_nodes])
	
	def water_deficit(self, qsz, pet):
		"""Calculate the amount of water required to saturated
		the riparian zone
		PARAMETERS
		----------
		qsz:		groundwater discharge [mm]
		smd_uz:		soil moisture deficit riparian zone [mm]
		pet:		potential evapotranpiration riparian zone [mm]
					pet = (PET - AET)
		
		OUTPUTS
		-------
		smd:		water deficit below the streambed [mm]
		qriv:		runoff
		inf:		infiltration
		"""
		
		# water required to saturate riparian zone
		#print(self.smd_uz)
		qriv = qsz - self.smd_uz		
		self.smd_uz = np.where(qriv > 0, 0, self.smd_uz-qsz)
		qriv[qriv < 0] = 0
		#print(qsz)
		
		#print('sm',self.smd_uz)
		# water required to overcome potential ET
		smd_et = pet - qriv
		
		qriv = np.where(smd_et > 0, 0, -smd_et)
			
		smd_et[smd_et < 0] = 0
		
		# total deficit in the unsaturated zone
		smd = self.smd_uz + smd_et
		
		inf = qsz - qriv
		#print(qriv)
		#print(inf)
		return smd, qriv, inf
	
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
	
	L_RAW = Lfc - RAW
	L_TAW = Lfc - TAW
	
	den = L_RAW-L_TAW
	
	beta = (L0-L_TAW)
	beta = (beta/den)
	beta[beta > 1] = 1
	beta[beta < 0] = 0	
	
		
	#beta[denominator > 0] = (beta[denominator > 0]
	#			/denominator[denominator > 0])
	#
	I_AET = np.where(I > PET, PET, I)
	AET = I_AET*(1-beta) + beta*PET
	AET[AET < 0] = 0
	
	L = L0+I-AET
	
	AET = np.where((L-Lwp) < 0, L0+I-Lwp, AET)
	
	L = L0 + I - AET
	
	D = np.where(L-Lfc > 0, L-Lfc, 0)
	
	L = L-D
	
	RO = L-Lsat
	RO[RO < 0] = 0
	
	L = L-RO
	
	SMD = Lfc - L
	SMD[SMD < 0] = 0
	
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
	
	# reference potential evapotranspiration
	PET = Kc*PET
	Pc = 0.5
	
	# water content at different states
	Lsat = z_soil*fs
	Lwp = z_soil*wp
	Lfc = z_soil*fc
	
	# total availble water
	TAW = Lfc - Lwp
	# stress limit available water
	RAW = Pc * TAW
		
	L_RAW = Lfc - RAW
	L_TAW = Lfc - TAW
	
	# Calculate stress coeficient
	den = L_RAW - L_TAW	
	beta = (L0 - L_TAW)
	beta = beta/den
	
	beta[beta > 1] = 1
	beta[beta < 0] = 0	
	#beta[den > 0] = (beta[den > 0]
	#			/den[den > 0])
	#beta = beta/den
	
	I_AET = np.where(I > PET, PET, I)
	AET = I_AET*(1-beta) + beta*PET
	AET[AET < 0] = 0
	
	L_aux = L0 + I - AET
	
	# This will not allow the soil misture to be less than wilting_point
	AET = np.where((L_aux-Lwp) < 0, L0+I-Lwp, AET)
	L_aux = L0 + I - AET
	#print(I)
	RO = np.where(L_aux > Lsat, L_aux-Lsat, 0)
	
	L_aux -= RO
	
	DL = np.where(L_aux > Lfc,
		z_soil*np.exp((-c+1)*np.log(np.power(L_aux/z_soil,-c+1)+kd)) - Lfc,
		0)
		
	DL = np.where(DL < 0.0, L_aux-Lfc, DL)
	
	L = L_aux - DL
	D = L_aux - L
	SMD = Lfc - L
	SMD[SMD < 0] = 0
	#SMD = np.where(L < Lfc, SMD_0+AET-I, 0)	
	
	return AET, SMD, D, L, RO

# Variable soil depth
def variable_soil_depth(z, h, Droot):
	# Luz	: Soil depth variation
	# z     : Surface elevation
	# h     : Water table elevation
	# Droot	: Soil rooting depth
	# Lsat	: Water content at saturated condition
	Zs = z - Droot*0.001
	Duz = np.where((Zs - h) >= 0, Droot, (z - h)*1000.0)
	Duz[Duz < 0] = 0
	return Duz
	
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


# Two layer model using and numerical integrator
# The model considers firstevaporation from the upper layer
# If there is enough water in the 
def SWBM1L(I, PET, Kc, L0, z_soil, fs, fc, wp, c, Ksat):
	"""Soil water balance
	Parameters:
		I:	Infiltration
		PET:	Potential evapotranspiration
		L0u:	initial water content upper layer
		L0l:	initial water content bottom layer
		SMD_0:	Initial soil moisture deficit
		D_u:	Soil depth upper layer
		D_l:	Soil depth lower layer
		fs:		Soil moisture at saturated conditions
		fc:		Soil moisture at field capacity
		wp:		Soil moisture at wilting point
		c:		Exponential recession term estimated as
		d:		Soil parameter par_swb for reducing time steps
		z:		Surface elevation
		h:		Water table elevation
		Droot:	Soil rooting depth
		Lsat:	Water content at saturated condition	
	Outouts:
		AET:	Actual evpotranspiration
		D:		Drainage
		L:		Water content
	"""
	
	# reference potential evapotranspiration
	PET = Kc*PET
	Pc = 0.5
	
	# water content at different states
	Lsat = z_soil*fs
	Lwp = z_soil*wp
	Lfc = z_soil*fc
	
	# total availble water
	TAW = Lfc - Lwp
	# stress limit available water
	RAW = Pc * TAW
		
	L_RAW = Lfc - RAW
	L_TAW = Lfc - TAW
	
	# Calculate stress coeficient
	den = L_RAW - L_TAW	
	beta = (L0 - L_TAW)
	beta = beta/den
	
	beta[beta > 1] = 1
	beta[beta < 0] = 0	
		
	I_AET = np.where(I > PET, PET, I)
	AET = I_AET*(1-beta) + beta*PET
	AET[AET < 0] = 0
	
	#L_aux = L0 + I - AET
	
	# This will not allow the soil misture to be less than wilting_point
	AET = np.where((L0 + I - AET - Lwp) < 0, L0+I-Lwp, AET)
	
	L_aux = L0 + I - AET
	
	RO = np.where(L_aux > Lsat, L_aux-Lsat, 0)
	
	#L_aux -= RO
	aux = L0 - Lfc
	aux[aux < 0] = 0
	#R = np.where(aux > 0,
	R =	Ksat*np.power((aux)/Lsat, c)#,#/z_soil,
		#0)
		
	#DL = np.where(DL < 0, L_aux-Lfc, 0.0)
	R = np.where(aux - R < 0, aux, R)
	DL = I - AET - R
	
	return DL, AET, R, RO


def SWBM2L(I, PET, Kc, L0u, L0l, z_soilu, z_soill, fs, fc, wp, c, Ksat):
	"""Mass balance of the two reservoirs
	"""
	# Potential evapotranspiration upper layer
	PETu = 2*PET*z_soilu*(0.5*z_soilu + z_soill)*np.power(z_soilu + z_soill,-2)	
	
	# Upper layer mass balance
	DLu, E, RO, Ru = SWBM1L(I, PETu, Kc, L0u, z_soilu, fs, fc, wp, c, Ksat)
	
	# Potential evapotranspiration lower layer
	PETl = PET - E	
	
	# Lower layer mass balance
	DLl, T, RO, R = SWBM1L(Ru, PETl, Kc, L0l, z_soill, fs, fc, wp, c, Ksat)
	
	return DLu, DLl, E, T, R, RO
	
	

def FAO2L(I, PET, Kc, L0u, L0l, z_soilu, z_soill, fs, fc, wp, c, Ksat):
	""" Runge Kutta apprach for solving soil moisture SWBPh
	Lu:			Water content upper layer
	Ll:			Water content lower layer
	z_soilu:	Soil upper layer thickness
	z_soill:	Soil lower layer thickness
	
	"""
	
	ku1, kl1, E, T, R, RO = SWBM2L(I, PET, Kc, L0u, L0l, z_soilu, z_soill, fs, fc, wp, c, Ksat)
	ku2, kl2, E, T, R, RO = SWBM2L(I, PET, Kc, L0u+ku1/3, L0l+kl1/3, z_soilu, z_soill, fs, fc, wp, c, Ksat)
	ku3, kl3, E, T, R, RO = SWBM2L(I, PET, Kc, L0u+ku2/3, L0l+kl2/3, z_soilu, z_soill, fs, fc, wp, c, Ksat)
	ku4, kl4, E, T, R, RO = SWBM2L(I, PET, Kc, L0u+ku3, L0l+kl3, z_soilu, z_soill, fs, fc, wp, c, Ksat)
	Lu = L0u + ku1/6 + ku2/3 + ku3/3 + ku4/6
	Ll = L0l + kl1/6 + kl2/3 + kl3/3 + kl4/6
	
	DLu, DLl, E, T, R, RO  = SWBM2L(I, PET, Kc, Lu, Ll, z_soilu, z_soill, fs, fc, wp, c, Ksat)
	
	# Calculate soil moisture deficit for GW - SW interactions
	SMD_dt = z_soilu*fc - Lu
	SMD_dt[SMD_dt < 0] = 0
	
	return Lu, Ll, SMD_dt, E, T, R, RO
	

	
	
	
	
	
	
	
	