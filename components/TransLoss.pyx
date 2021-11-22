import cython
import numpy
cimport numpy

#@cython.binding(True)  # this ensures it really "looks" like a Python func
cpdef find_discharge_and_losses(s, r, runoff, Criv, Kt,
				QTL, Q_ini, riv, Q_aof, Q_aoft, riv_std, P3, P4,
				node_cell_area, boundary_nodes, Kloss):
				
	
	cdef int i, donor, recvr, ncells
	
	cdef numpy.ndarray discharge
	

	"""Calculate the drainage area and water discharge at each node, permitting
	discharge to fall (or gain) as it moves downstream according to some
	function. Note that only transmission creates loss, so water sourced
	locally within a cell is always retained. The loss on each link is recorded
	in the 'surface_water__discharge_loss' link field on the grid; ensure this
	exists before running the function.
	Parameters
	----------
	s : ndarray of int
		Ordered (downstream to upstream) array of node IDs
	r : ndarray of int
		Receiver node IDs for each node
	l : ndarray of int
		Link to receiver node IDs for each node
	loss_function : Python function(Qw, nodeID, linkID, grid)
		Function dictating how to modify the discharge as it leaves each node.
		nodeID is the current node; linkID is the downstream link, grid is a
		ModelGrid. Returns a float.
	grid : Landlab ModelGrid (or None)
		A grid to enable spatially variable parameters to be used in the loss
		function. If no spatially resolved parameters are needed, this can be
		a dummy variable, e.g., None.
	node_cell_area : float or ndarray
		Cell surface areas for each node. If it's an array, must have same
		length as s (that is, the number of nodes).
	runoff : float or ndarray
		Local runoff rate at each cell (in water depth per time). If it's an
		array, must have same length as s (that is, the number of nodes).
	boundary_nodes: list, optional
		Array of boundary nodes to have discharge and drainage area set to zero.
		Default value is None.
	Returns
	-------
	tuple of ndarray
		drainage area and discharge
	"""
	
	# Number of points
	npoint = len(s)
	
	# Initialize the drainage_area and discharge arrays. Drainage area starts
	# out as the area of the cell in question, then (unless the cell has no
	# donors) grows from there. Discharge starts out as the cell's local runoff
	# rate times the cell's surface area.
	#drainage_area = np.zeros(npoint, dtype=int) + node_cell_area
	discharge = node_cell_area * runoff
	# note no loss occurs at a node until the water actually moves along a link
	
	# Optionally zero out drainage area and discharge at boundary nodes
	if boundary_nodes is not None:
		discharge[boundary_nodes] = 0
	#print(discharge)
	# Iterate backward through the list, which means we work from upstream to
	# downstream.
	for i in range(npoint - 1, -1, -1):
		donor = s[i]
		recvr = r[donor]
		if donor != recvr:
			# this function calculate:
			discharge_remaining, Q_TLp, Q_inip, Q_aofp = TransLossWV(
					discharge[donor],
					Criv[donor],
					Kt[donor], QTL[donor],
					Q_ini[donor], riv[donor],
					Q_aof[donor], Q_aoft[donor],
					riv_std[donor], P3[donor], P4[donor],
					Kloss)
					
			discharge[recvr] += discharge_remaining
			QTL[donor] = Q_TLp
			Q_ini[donor] = Q_inip
			Q_aof[donor] = Q_aofp
			
	return discharge, QTL, Q_ini, Q_aof

cpdef TransLossWV(Qw, Criv, Kt, QTL, Q_ini,
				riv, Qaof, Qaoft, riv_std, P3, P4, Kloss):
	
	cdef double Qout, Con, TL
	
	#cdef double aux[2], aux_l[3]
	
	"""Transmission losses function
	Parameters
	----------
	Qw :		flow entering the cell
	Kloss :		Transmission losses factor
	Criv :		Conductivity
	Kt :		Decay parameter
	QTL :		Transimission losses
	Q_ini :		Initial flow
	riv :		river cell id
	Qaof :		Abstraction flux
	Qaoft :		Threshold abstraction
	riv_std :	River saturation deficit
	P3 :		Parameter
	P4 :		Parameter
	Returns
	-------
	Qout :		flow leaving the cell
	QTL :		Transimission losses
	Q_ini :		Initial flow
	Qaof :		Abstraction flux
	"""
	TL = 0
	Qout = Qw	
	if riv != 0:		
		Qin = (Qw+Q_ini)
		# abstractions, it can be modified
		if Qin <= Qaof:
			Qaof = Qaoft*Qin
		Qin += -Qaoft
			
		Q_ini = 0
		QTL = 0
			
		if Qin > 0.0:		
			if riv_std <= 0.0:		
				Qout, Qo = exp_decay_wp(Qin, Kt)
				TL = 0				
			else:
				Con = numpy.array(Criv*Kloss)
				Qout, Qtl, Qo = exp_decay_loss_wp(Con, Qin, Kt, P3, P4)
				
				if TL > riv_std:					
					Qo += TL-riv_std
					TL = riv_std
			
			Q_ini = Qo			
			QTL = TL
					
	return Qout, QTL, Q_ini, Qaof

cpdef exp_decay_loss_wp(TL, Qin, k, P3, P4):
	
	cdef double t, Qout, Qo, Q1, Qtl	
	
	Qout, Qo = 0, 0	
	
	t = -(1/k)*numpy.log(P3/(Qin*k))
	if t > 0:
	
		if t > 1:		
			t = 1
	
		Q1 = Qin*(1-numpy.exp(-k*t))		
		Qtl = Qin*k*P4*(1-numpy.exp(-k*t))+TL*t	
		Qout = Q1-Qtl
		
	if t >= 1:	
		Qo = Qin-Q1
	else:	
		Qo = 0.0		
		Qtl = Qin-Qout
		
	return Qout, Qtl, Qo

cpdef exp_decay_wp(Qin, k):
	Qout = Qin*(1-numpy.exp(-k))
	Qo = Qin - Qout
	return Qout, Qo