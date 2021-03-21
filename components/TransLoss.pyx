import cython
import numpy as np

@cython.binding(True)  # this ensures it really "looks" like a Python func
cpdef TransLossWVc(pyQw, pynodeID, pylinkID, grid):
	cdef double Qw = pyQw
	cdef int nodeID = pynodeID
	cdef int linkID = pylinkID
	cdef double TL
	cdef double Qout
	cdef double Qo
	cdef double Qin
	# transmission loss:
	cdef isriver = grid.at_node['river'][nodeID]
	
	Qout = Qw	
	if isriver != 0:		
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
	cdef double t, Qout, Qo, Q1, Qtl

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