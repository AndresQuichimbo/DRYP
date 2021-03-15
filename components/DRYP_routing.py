import os
import numpy as np
from landlab.components import LossyFlowAccumulator

class runoff_routing(object):
		
	def __init__(self,env_state,data_in):
		Create_parameter_WV(env_state.grid,data_in.Kloss)
		self.dis_dt = np.zeros(env_state.grid_size)
		self.tls_dt = np.zeros(env_state.grid_size)
		self.flow_tls_dt = np.zeros(env_state.grid_size)
				
		fa = LossyFlowAccumulator(env_state.grid, 'topographic__elevation',
								flow_director = 'D8',
								loss_function = TransLossWV,
								runoff_rate = 'runoff')
		self.fa = fa

	def base_flow_streams(self,env_state,dt):				
		# dis_dt:	Discharge [mm]
		# exs_dt:	Infiltration excess [mm]
		# tls_dt	Transmission losses [mm]
		# cth_area:	Cell factor area, it reduces or increases the area
		# Q_ini:	Initial available water in the channel				
		dhriv = (env_state.SZgrid.at_node['water_table__elevation']
			- env_state.grid.at_node['river_topo__elevation'])
		env_state.grid.at_node['base_flow_streams'] = env_state.grid.at_node['SS_loss']*dhriv/(1.0*dt) #[m/dt]		
		pass		

	def run_runoff_one_step(self, inf, swb, aof, env_state, data_in):				
		# dis_dt:	Discharge [mm]
		# exs_dt:	Infiltration excess [mm]
		# tls_dt	Transmission losses [mm]
		# cth_area:	Cell factor area, it reduces or increases the area
		# Q_ini:	Initial available water in the channel
		# aof:		River flow abstraction [mm]
		env_state.grid.at_node['AOF'][:] = aof
		act_nodes = env_state.act_nodes
		if data_in.daily_model == 0:
			env_state.grid.at_node['runoff'][act_nodes] = (inf.exs_dt[act_nodes]*0.001
											* env_state.cth_area[act_nodes]
											+ env_state.grid.at_node['Base_flow'][act_nodes]
											+ env_state.SZgrid.at_node['discharge'][act_nodes]
											)
			check_dry_conditions = len(np.where((env_state.grid.at_node['runoff'][act_nodes]
				+ env_state.grid.at_node['Q_ini'][act_nodes]
				+ env_state.grid.at_node['Base_flow'][act_nodes]) > 0.0)[0])
			
			if check_dry_conditions > 0:
				self.fa.accumulate_flow(update_flow_director = env_state.act_update_flow_director)
				if env_state.run_flow_accum_areas == 0:			
					self.dis_dt[act_nodes] = (1000* np.array(
						env_state.grid.at_node["surface_water__discharge"][act_nodes]
						/ (env_state.grid.at_node["drainage_area"][act_nodes])))							
				else:				
					self.dis_dt[act_nodes] = (1000.*np.array(
						env_state.grid.at_node["surface_water__discharge"][act_nodes]
						/ env_state.area_discharge[act_nodes]))
			else:
				env_state.grid.at_node['Transmission_losses'][env_state.river_ids_nodes] = 0.0				
				self.dis_dt[:] = 0				
				noflow = 0
			self.tls_dt = env_state.grid.at_node['Transmission_losses']*1000.0/env_state.area_cells
			self.tls_flow_dt = env_state.grid.at_node['Transmission_losses']

# ===================================================================
# Transmission losses functions

# Exponential decay parameters
def Create_parameter_WV(grid,Kloss):
	grid.at_node['par_4'] = 2.0*Kloss/(grid.at_node['decay_flow']*grid.at_node['river_width'])	
	grid.at_node['par_3'] = (grid.at_node['river_width']*grid.at_node['SS_loss']
							/ (grid.at_node['river_width']-2*Kloss))
	grid.at_node['base_flow_streams'] = np.zeros(len(grid.at_node['par_3']))	
	return

# Transmission losses function
def TransLossWV(Qw, nodeID, linkID, grid): 
	Qout = Qw	
	if grid.at_node['river'][nodeID]  !=  0:		
		Qin = (Qw+grid.at_node['Q_ini'][nodeID])
		# abstractions
		if Qin <= grid.at_node['AOF'][nodeID]:
			grid.at_node['AOF'][nodeID] = grid.at_node['AOFT'][nodeID]*Qin
		Qin += -grid.at_node['AOF'][nodeID]
			
		grid.at_node['Q_ini'][nodeID] = 0
		grid.at_node['Transmission_losses'][nodeID] = 0
			
		if Qin > 0.0:		
			if grid.at_node['riv_sat_deficit'][nodeID] <= 0.0:		
				Qout, Qo = exp_decay_wp(Qin,
						grid.at_node['decay_flow'][nodeID]
						)

				TL = 0				
			else:		
				Qout, TL, Qo = exp_decay_loss_wp(grid.at_node['SS_loss'][nodeID],Qin,
										grid.at_node['decay_flow'][nodeID],
										grid.at_node['par_3'][nodeID],
										grid.at_node['par_4'][nodeID])
								
				if TL > grid.at_node['riv_sat_deficit'][nodeID]:					
					Qo += TL-grid.at_node['riv_sat_deficit'][nodeID]				
					TL = grid.at_node['riv_sat_deficit'][nodeID]
			
			grid.at_node['Q_ini'][nodeID] = Qo			
			grid.at_node['Transmission_losses'][nodeID] = TL
					
	return Qout

def exp_decay_loss_wp(TL,Qin,k,P3,P4):

	Qout, Qo = 0, 0	
	t = -(1/k)*np.log(P3/(Qin*k))
			
	if t > 0:
	
		if t > 1:		
			t = 1
	
		Q1 = Qin*(1-np.exp(-k*t))		
		Qtl = Qin*k*P4*(1-np.exp(-k*t))+TL*t	
		Qout = Q1-Qtl
		
	if t >= 1:	
		Qo = Qin-Q1
	else:	
		Qo = 0.0		
		Qtl = Qin-Qout
		
	return Qout, Qtl, Qo

def exp_decay_wp(Qin,k):
	Qout = Qin*(1-np.exp(-k))
	Qo = Qin-Qout
	return Qout, Qo

# ==========================================================================
# This functions are disabled in the code
# Transmission losses functions - Manning with exponential recesion
# Requires solver for manning equation
def Create_parameter_ManningTL(grid,n_manning,unit_change):

	# Estimating channel slope
	So_dh = grid.at_node['topographic__elevation']-\
		grid.at_node['topographic__elevation'][np.array(grid.at_node['flow__receiver_node']).astype(int)]
	
	So_dl = np.array(grid.at_node['river_length'])	
	So = np.where(So_dl == 0.0,0.001,So_dh/So_dl)	
	So[So <= 0.0] = 0.001	
	grid.at_node['Manning'] = unit_change*(n_manning/(grid.at_node['river_width']*(So**0.5)))**(3/5)#*(1/rg.at_node['river_width'][:])**(3/5)
	par_m = grid.add_zeros('node', 'par_m',dtype = float)
	grid.at_node['par_m'] = (grid.at_node['SS_loss']*grid.at_node['Manning'])#/grid.at_node['river_width'])
	
	return

def TranslossManning(Qw, nodeID, linkID, grid): 
	Qout = Qw	
	if grid.at_node['river'][nodeID] != 0:
		D = 1.0
		qo = (Qw+grid.at_node['Q_ini'][nodeID])*grid.at_node['decay_flow'][nodeID]		
		
		if qo > grid.at_node['SS_loss'][nodeID]:		
			t = Newthon_R_Manning(qo,grid.at_node['decay_flow'][nodeID],
								grid.at_node['SS_loss'][nodeID],D,
								grid.at_node['par_m'][nodeID],0.01)
		else:
			t = 0.0
													
		grid.at_node['Transmission_losses'][nodeID] = \
			grid.at_node['SS_loss'][nodeID]*t+\
			(5/3)*(grid.at_node['par_m'][nodeID]/grid.at_node['decay_flow'][nodeID])*\
			(qo**(3/5))*(1-np.exp(-3/5*grid.at_node['decay_flow'][nodeID]*t))
							
		Q = (qo/grid.at_node['decay_flow'][nodeID])*\
			(1-np.exp(-grid.at_node['decay_flow'][nodeID]*t))
										
		Qout = Q-grid.at_node['Transmission_losses'][nodeID]
		
		if t < 1.0:		
			Qout = 0.0			
			grid.at_node['Transmission_losses'][nodeID] = qo/grid.at_node['decay_flow'][nodeID]-Qout			
			grid.at_node['Q_ini'][nodeID] = 0.0
		else:		
			grid.at_node['Q_ini'][nodeID] = (qo/grid.at_node['decay_flow'][nodeID])-\
											Qout-grid.at_node['Transmission_losses'][nodeID]
			
	return Qout
	
def fmaning(qo,k,iloss,D,n3_5,t):
	return qo*np.exp(-k*t)-n3_5*(qo**(3/5))*np.exp(-3*k*t/5)-iloss

def dfmanning(qo,k,iloss,D,n3_5,t):
	return -k*qo*np.exp(-k*t)+(3/5)*n3_5*k*(qo**(3/5))*np.exp(-3*k*t/5)
	
def Newthon_R_Manning(qo,k,iloss,D,n3_5,t0):
	error = 100.0
	while error > 0.001:
		t  = t0-fmaning(qo,k,iloss,D,n3_5,t0)/dfmanning(qo,k,iloss,D,n3_5,t0)
		error = np.abs((t-t0)/t)
		t0 = t
	# it means greather than the time step
	if t > 1.0: 
		t = 1.0
	elif t < 0:
		t = 0.0

	return t

# Transmission losses functions - Constant loss
def TransLoss(Qw, nodeID, linkID, grid):	
	Qout = Qw	
	if grid.at_node['river'][nodeID] != 0:	
		if Qw > grid.at_node['SS_loss'][nodeID]:		
			grid.at_node['Transmission_losses'][nodeID] = grid.at_node['SS_loss'][nodeID]			
			Qout = Qw-grid.at_node['Transmission_losses'][nodeID]			
		else:		
			grid.at_node['Transmission_losses'][nodeID] = Qw			
			Qout = 0
	
	return Qout

# Transmission losses functions - exp-Log function
def TransLoss_power(Qw, nodeID, linkID, grid):
	Qout = Qw	
	if grid.at_node['river'][nodeID] != 0:
		
		Qin = (Qw+grid.at_node['Q_ini'][nodeID])
		grid.at_node['Q_ini'][nodeID] = 0
		grid.at_node['Transmission_losses'][nodeID] = 0
		TL_aux = grid.at_node['SS_loss'][nodeID]
		
		if Qin > 0.0:		
			Qout, TL, Qo = exp_decay_power(TL_aux,Qin,
									grid.at_node['decay_flow'][nodeID],0.6)
			
			grid.at_node['Q_ini'][nodeID] = Qo
			grid.at_node['Transmission_losses'][nodeID] = TL
					
	return Qout

def exp_decay_power(TL,Qin,k,a):
	
	Qout, Qo = 0, 0	
	kd = k*(1-a)	
	t = -(1/kd)*np.log(TL*np.power((Qin*k),a-1))
	
	if t > 0:	
		if t > 1:		
			t = 1			
		Q1 = Qin*(1-np.exp(-k*t))		
		Qtl = TL*np.power(Qin*k,a)*(1-np.exp(-a*k*t))/(a*k)	
		Qout = Q1-Qtl
	
	if t >= 1:	
		Qo = Qin-Q1
	else:	
		Qo = 0.0		
		Qtl = Qin-Qout

	return Qout, Qtl, Qo

# -----------------------------------------------------------------------	
# Variable flow velocity in streams
def velocity(L,W,Ss,So):	
	# Ss:	Channel storage
	# So:	River slope
	# W:	Channel width	
	B = -0.0543 * np.log10(So)	
	C = 2*Ss/(L*W)+W	
	v = 1.564*np.power(R,B)*np.power(Ss,0.573)/(np.power(L,0.573)*np.power(C,0.40))	
	return v
	
# Hydraulic radio
def h_radio(y,W):
	return (y * W)/(2*y + W)
	
def Storage(Ss,v,L,dQ,dt):
	K = dt*v/L	
	Ss = Ss*np.exp(-K)+(1-np.exp(-K))*L/v*dQ
	return Ss

# -----------------------------------------------------------------------	
# Variable flow velocity in streams, transmission losses
def flow(Qo,I,K,dt):
	return (Qo-I)*np.exp(K*dt) + I
	
def t_zero(Qo,I,K):
	return -(1/K)*np.log(I/(I-Qo))	

def Maning(y, W, S, n):
	#y:	Stream stage
	#W:	Channel width
	#S:	Channel slope
	#n:	Manning roughtness	
	R = h_radio(y,W)	
	A = W * y	
	return A * np.power(R,2/3) * np.sqrt(S)	

def dfdyManning(y, W, S, n):	
	dRdy = np.power(W,5/3)*np.power(y,2/3)*(6*y+5*W)/(3*np.power(2*y+W,5/3))	
	return (1/n) * np.sqrt(S)* dRdy	
	
def Manning_Newthon_Rapshon(W, S, n, Q, y0):
	error = 100.0
	while error > 0.001:	
		y = y0 - (Maning(y0, W, S, n) - Q)/dfdyManning(y0, W, S, n)		
		error = np.abs((y-y0)/y)
		y0 = y
	return y0

def Manning_wide_ch(W, S, n, Q, y0):
	return np.power(Q * n/(W * np.sqrt(S)),3/5)