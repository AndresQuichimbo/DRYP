import numpy as np
import os
import pdb
from landlab import RasterModelGrid
from landlab.io import read_esri_ascii
import scipy.special as spy

class infiltration(object):
	def __init__(self, env_state, data_in):		
		# K_sat:	Saturated hydraulic conductivity	
		# PSI_f:	Wetting front suction head			
		act_nodes = env_state.act_nodes		
		self.rain_day_before = 0
		
		self.inf_dt = np.zeros(env_state.grid_size)		
		self.exs_dt = np.zeros(env_state.grid_size)
		self.K_sat = np.array(env_state.grid.at_node['Ksat_soil'][act_nodes])
		self.PSI_f = np.array(env_state.grid.at_node['PSI'][act_nodes])
		# Schaake infiltration approach		
		if data_in.inf_method == 0:		
			delta_time = data_in.dtUZ*60/86400			
			ks_ref = 7.2*data_in.unit_sim_k #mm/h			
			ga_kdt = 1.0-np.exp(-data_in.kdt_r*delta_time*self.K_sat/ks_ref)			
			self.args = ga_kdt
		# Read sigma_Ksat for UPSACALED_GA		
		elif data_in.inf_method == 2:
			if not os.path.exists(data_in.fname_sigma_ks):
				sigma_Ksat = np.abs(np.log(self.K_sat)/3.0) #This is an approximation, can be modify by k_sigma_ks				
				print('Not avalilable sigma_Ksat, considering 1/3 of log(Ksat)')				
			else:		
				sigma_ks = read_esri_ascii(data_in.fname_sigma_ks, name = 'sigma_ks',grid = env_state.grid)[1]				
				sigma_Ksat = np.array(env_state.grid.at_node['sigma_ks'][act_nodes]*data_in.k_sigma_ks)
			mu_log_Ksat = np.log(self.K_sat) - 0.5*(sigma_Ksat**2)
			self.args = (mu_log_Ksat, sigma_Ksat)
		else:
			self.args = ()
	
	def run_infiltration_one_step(self, rf, env_state, data_in):
		# L_0:	Initial soil water content [mm]
		# Lsat:	Saturated water content [mm]
		# Ft0:	Cummulative infiltration [mm]
		# t_0:	Initial time for infiltration rate [h]
		# SORP0:Sorptivity at the begining of the time step
		inf_method = data_in.inf_method
		act_nodes = env_state.act_nodes
		rainfall = np.array(rf.rain[act_nodes])
		PSI_f = np.array(self.PSI_f)
		Droot = np.array(env_state.Duz[act_nodes])
		SORP0 = np.array(env_state.SORP0[act_nodes])
		L_0 = np.array(env_state.L_0[act_nodes])
		Lsat = np.array(env_state.Lsat[act_nodes])
		t_0 = np.array(env_state.t_0[act_nodes])
		Ft0 = np.array(env_state.Ft_0[act_nodes])
		rain_day_before = self.rain_day_before
				
		Ft, SORP, inf_t, excess, t_0, rain_day_before = infiltration_model(rainfall,
																self.K_sat,
																PSI_f,
																Droot,SORP0,
																L_0,Lsat,
																np.array(t_0),
																Ft0,rain_day_before,
																inf_method,
																self.args)

		# Update environmental states
		env_state.Ft_0[act_nodes] = Ft
		env_state.t_0[act_nodes] = np.array(t_0)
		env_state.SORP0[act_nodes] = SORP
		self.rain_day_before = rain_day_before
		self.inf_dt[act_nodes] = inf_t
		self.exs_dt[act_nodes] = excess

def infiltration_model(rainfall,K_sat,PSI_f,Droot,SORP0,L_0,Lsat,t_0,Ft0,rain_day_before,inf_method,*args):
	#	grid		: Landlad grid
	#	rainfall	: Rainfall, numpy array 
	#	act_nodes	: Active nodes, array
	#	inf_method	: 0 - Shaake
	#				  1 - Philips
	#				  2 - Up-scaled GA
	#				  3 - Modified GA
	#	PSI_f		: Maximum potentiometric head (cm), numpy array
	#	Droot			: Soil depth (mm), numpy array
	#	SORP		: Sorptivity
	#	SORP0
	#	L_0			: Initial water content t = 0
	#	L			: Water content at time t
	#	ds			: Soil depth
	#	t_0			: initial t
	#	t_i			: time from the start of the precipitation event
	#	F			: Cummulative infiltration rate
	#	rain_day_before
	# Incorporate the variable saturated conditions
	# make zero precipitation for cell with saturated
	# and zero depth of UZ zone
	# Create an exs_aux to store the precipitation
	sat_excess_dt = np.where((Lsat-L_0) <= 0.0,rainfall,0.0)
	rainfall = np.where((Lsat-L_0) <= 0.0,0.0,rainfall)
	sat_excess_dt = np.where(Droot <= 0.0,rainfall,sat_excess_dt)
	rainfall = np.where(Droot <= 0.0,0.0,rainfall)
	inode_inf_aux = np.where(rainfall > 0.0)[0]
	
	if len(inode_inf_aux) > 0:
		t_i = np.zeros(len(rainfall))
		if inf_method == 0: # SCHAAKE METHOD
			ga_kdt = args[0]
			SORP = np.zeros(len(rainfall))
			Ft = np.zeros(len(rainfall))
			inf_dt, excess_dt = SCHAAKE(rainfall, ga_kdt, Lsat, L_0)
		elif inf_method == 1: # PHILIPS EQUATION
			F = np.zeros(len(rainfall))
			SORP = np.where(Droot == 0,0.0,np.sqrt(2*K_sat*PSI_f*((Lsat-L_0)/Droot)))
			if rain_day_before == 1:
				SORP[inode_inf_aux] = SORP0[inode_inf_aux]
				t_i[inode_inf_aux] = t_0[inode_inf_aux]
				F[inode_inf_aux] = Ft0[inode_inf_aux]
			Ft,inf_dt,excess_dt = Philip(rainfall, 
										0.5*K_sat,
										SORP, F,
										np.array(t_i))
		elif inf_method == 2: # UPSACALED GREEN & AMPT METHOD
			Ft = np.zeros(len(rainfall))
			mu_logks = args[0][0]
			sigma_ks = args[0][1]
			SORP = np.where(Droot == 0,0.0,PSI_f*((Lsat-L_0)/Droot))
			if rain_day_before == 1:
				SORP[inode_inf_aux] = SORP0[inode_inf_aux]
				t_i[inode_inf_aux] = t_0[inode_inf_aux]
			inf_dt,excess_dt = Upscaled_GA(rainfall,K_sat,SORP,
											np.array(t_i+1.0),
											mu_logks,
											sigma_ks)
		elif inf_method == 3: # MODIFIED GREEN AND AMPT EQUATION
			F = np.zeros(len(rainfall))
			SORP = np.where(Droot == 0,0.0,np.sqrt(2*K_sat*PSI_f*((Lsat-L_0)/Droot)))
			if rain_day_before == 1:
				SORP[inode_inf_aux] = SORP0[inode_inf_aux]
				t_i[inode_inf_aux] = t_0[inode_inf_aux]
				F[inode_inf_aux] = Ft0[inode_inf_aux]
			Ft,inf_dt,excess_dt = Mod_GA(rainfall,
										K_sat,
										np.array(SORP),
										F,
										np.array(t_i),1.)
		
		t_0[inode_inf_aux] += 1
		rain_day_before = 1
	else:
		t_0 = np.zeros(len(rainfall),dtype = float)
		rain_day_before = 0
		inf_dt = np.zeros(len(rainfall),dtype = float)
		Ft = np.zeros(len(rainfall),dtype = float)
		SORP = np.zeros(len(rainfall),dtype = float)
		excess_dt = np.zeros(len(rainfall),dtype = float)
	excess_dt += sat_excess_dt
	return Ft, SORP, inf_dt, excess_dt, t_0, rain_day_before

# Schaake infiltration approach, Schaake et. al. (1996)
def SCHAAKE(P,ga_kdt,Lsat,L):
	D = Lsat-L
	I_aux = D*ga_kdt
	I = P*I_aux/(P+I_aux)
	I = np.where(P+I_aux == 0.0,0.0,I)
	RO = P-I
	return I, RO

# Philips infiltration rate
def Philip(P,ks,Sp,F,t):
	# P:	Precipitation
	# ks:	Sat. Hydraulic Conductivity
	# Sp:	Sorptivity (keep the same sorptivity for one event)
	# F:	Cummulative infiltration
	# t:	Cummulative event time
	dt = 1
	Fp = np.where(P > 0,0.5*(Sp**2)*(P-0.5*ks)*((P-ks)**(-2)),0)
	Fp_aux = Fp-F
	dtp = np.where(P > 0.0,Fp_aux/P,0.0)
	ts = np.where(dtp > dt,t+dt,t+dtp)
	ts = np.where(dtp < 0,t,ts)
	Faux = np.where(dtp < 0,F,Fp)
	sp_aux = np.sqrt(Sp**2+4*ks*Faux)-Sp
	to_p = (1/4)*(sp_aux/ks)**2
	to = np.where(dtp < dt,ts-to_p,t+dt-ts)
	dtc = np.where(P == 0.0,0.0,t+dt-to)
	Ft_aux = np.where(dtc == 0.0,0,Sp*dtc**0.5+ks*dtc)
	Ft = np.where(F+P < Ft_aux,F+P,Ft_aux)
	I = Ft-F
	RO = P-I
	return Ft,I,RO

# Upscaled Green & Ampt infiltration approach - Craig et. al. 2010
# Required Gauss2p, epsilon, and getX
def Upscaled_GA(P,ks,Sp,t,mu_Y,sigma_Y):
	X = getX(t,Sp,P)
	A = np.where(X == 0.0,0.0,(np.log(P*X)-mu_Y)/(sigma_Y*np.sqrt(2)))
	A = np.where(sigma_Y == 0,1e99,A)
	Aaux = np.exp(mu_Y+0.5*(sigma_Y**2)) # it can be estimated offline
	I = np.where(X == 0.0,0.0,0.5*P*spy.erfc(A)+(0.5/X)*Aaux*spy.erfc((sigma_Y/np.sqrt(2))-A))
	I += np.where(X == 0.0,0.0,Gauss2p(t,Sp,P,mu_Y,sigma_Y,ks))
	RO = P-I
	return I, RO

# 2-point Gauss-Legrenge integrator	
def Gauss2p(t,Sp,P,mu_Y,sigma_Y,ks):
	X = getX(t,Sp,P)
	dk = 0.5*(P*X-np.exp(mu_Y-3.0*sigma_Y))
	km = P*X-dk
	k1 = km-0.57735*dk
	k2 = km+0.57735*dk
	return dk*P*(epsilon_fks(k1,t,Sp,P,mu_Y,sigma_Y,ks)+epsilon_fks(k1,t,Sp,P,mu_Y,sigma_Y,ks))

# Epsilon funtion for upsaceld GA infiltration	
def epsilon_fks(k,t,Sp,P,mu_Y,sigma_Y,ks):
	X = getX(t,Sp,P)
	kp = ks/(P*X)
	fks = np.where(k <= 0.0,0.0,(1.0/(k*sigma_Y*np.sqrt(2.0*np.pi)))*np.exp(-0.5*(np.power((np.log(k)-mu_Y)/sigma_Y,2))))
	epsilon = np.where(X == 0.0,0.0,0.36315*np.power(1-X,0.484)*np.power(1.0-kp,1.74)*np.power(kp,0.38))
	epsilon = np.where(kp >= 1.0,0.0,epsilon)
	epsilon = np.where(kp == 0.0,0.0,epsilon)
	return epsilon*fks

# Dimentionless time parameter	
def getX(t,Sp,P):
	X_aux = P*t/Sp
	return np.where(X_aux == 0.0,0.0,1/(1+1/X_aux))
	

# Modified Green & Ampt infiltration approach
# Requires solver (Newthon_Rap_Mod_GA) and F (f_GA) and F' (dF_GA)
def Mod_GA(P,ks,Sp,F,t,dt):
	tp = np.where(P > ks,Sp/(P-ks),0)
	aux_1 = tp-t
	aux_2 = t+1-tp
	aux = aux_1*aux_2
	id_error = np.where(aux >= 0)[0]
	if len(id_error) > 0: # ponding during time step
		to = Newthon_Rap_Mod_GA(P,ks,Sp,F,t,tp,0.1)
	else: # Ponding already ocurred
		to = t
	Faux = np.where(to == 0,0,ks*(t+dt-to)+Sp*np.log((t+dt)/to))

	F_tf = np.where(tp > t+1,F+P,Faux)
	F_tf = np.where(tp < to,F_tf+F,F_tf)
	F_tf = np.where(tp == 0,P+F,F_tf)

	I = F_tf - F
	RO = P-I
	return F_tf, I, RO
	
def Newthon_Rap_Mod_GA(P,ks,Sp,F,t,tp,to_0):
	# this is for allowing a matrix convergence of each cell at each time step
	# to avoid going through each cell
	error = np.zeros(len(F))
	aux_1 = tp-t
	aux_2 = t+1-tp
	aux = aux_1*aux_2
	id_error = np.where(aux > 0)[0]
	len_error = len(id_error)
	error[id_error] = 1
	to = t
	
	while len_error > 0:
		to = np.where(error < 0.001,t,to_0-f_GA(P,ks,Sp,F,t,tp,to_0)/dF_GA(ks,Sp,to_0))
		error = np.where(error <= 0.001,0.0,np.abs(to-to_0)/to)
		len_error = len(np.where(error >= 0.001)[0])
		to_0 = to	
	return to
	
# Implicit solution of Green and Ampt equation	
def f_GA(P,ks,Sp,F,t,tp,to):	
	return ks*(tp-to)+Sp*np.log(tp/to)-F-P*(tp-t)

# Derivative of the implicit GA equation
def dF_GA(ks,Sp,to):
	return -(ks+Sp/to)
	
# Modified Green & Ampt infiltration approach
# Simple version
def Mod_GA_Sim(P,ks,Sp,t,dt):
	# Approximation of Green and Ampt equation for small time steps
	tp = np.where(P > ks,Sp/(P-ks),0)
	tf = np.where(tp > t+dt,t+dt,tp)
	I  = np.where(t == 0,0,ks*(t+dt-tp)+Sp*np.log((t+dt)/tp)-P*(t-tp))
	RO = P-I
	return F, I, RO
	
def to_modGA(P,ks,Sp,t,tp,to_0):
	# Next step for 'to', it is done to avoid iteration for each cell
	F = ks*(tp-to)+Sp*np.log(tp/to_0)-P*(t-tp)
	dF = -ks-B/to_0
	to = to_0-F/dF
	return to