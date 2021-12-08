MODULE ftransloss
IMPLICIT NONE
CONTAINS
SUBROUTINE find_discharge_and_losses(s, r, Criv, Kt,riv, Q_aoft, riv_std, P3, P4, Kloss,&
discharge, Qtl, Q_ini, Q_aof)
INTEGER, INTENT(IN) :: s(:)
INTEGER, INTENT(IN) :: r(:)
INTEGER, INTENT(IN) :: riv(:)
REAL, INTENT(IN) :: Criv(:)
REAL, INTENT(IN) :: Kt(:)
REAL, INTENT(IN) :: Q_aoft(:)
REAL, INTENT(IN) :: riv_std(:)
REAL, INTENT(IN) :: P3(:)
REAL, INTENT(IN) :: P4(:)
REAL, INTENT(IN) :: Kloss
REAL, INTENT(INOUT) :: Qtl(:)
REAL, INTENT(INOUT) :: Q_ini(:)
REAL, INTENT(INOUT) :: discharge(:)
REAL, INTENT(INOUT) :: Q_aof(:)


! Other variables
INTEGER :: i, donor, recvr, npoint
REAL :: Q_out, Q_TLp, Q_inip, Q_aofp
!"""Calculate the drainage area and water discharge at each node, permitting
!discharge to fall (or gain) as it moves downstream according to some
!function. Note that only transmission creates loss, so water sourced
!locally within a cell is always retained. The loss on each link is recorded
!in the 'surface_water__discharge_loss' link field on the grid; ensure this
!exists before running the function.

!Parameters
!----------
!s : ndarray of int
!	Ordered (downstream to upstream) array of node IDs
!r : ndarray of int
!	Receiver node IDs for each node
!l : ndarray of int
!	Link to receiver node IDs for each node
!loss_function : Python function(Qw, nodeID, linkID, grid)
!	Function dictating how to modify the discharge as it leaves each node.
!	nodeID is the current node; linkID is the downstream link, grid is a
!	ModelGrid. Returns a float.
!grid : Landlab ModelGrid (or None)
!	A grid to enable spatially variable parameters to be used in the loss
!	function. If no spatially resolved parameters are needed, this can be
!	a dummy variable, e.g., None.
!node_cell_area : float or ndarray
!	Cell surface areas for each node. If it's an array, must have same
!	length as s (that is, the number of nodes).
!runoff : float or ndarray
!	Local runoff rate at each cell (in water depth per time). If it's an
!	array, must have same length as s (that is, the number of nodes).
!boundary_nodes: list, optional
!	Array of boundary nodes to have discharge and drainage area set to zero.
!	Default value is None.
!Returns
!-------
!tuple of ndarray
!	drainage area and discharge
!Notes
!-----
!-  If node_cell_area not given, the output drainage area is equivalent
!to the number of nodes/cells draining through each point, including
!the local node itself.
!-  Give node_cell_area as a scalar when using a regular raster grid.
!-  If runoff is not given, the discharge returned will be the same as
!drainage area (i.e., drainage area times unit runoff rate).
!-  If using an unstructured Landlab grid, make sure that the input
!argument for node_cell_area is the cell area at each NODE rather than
!just at each CELL. This means you need to include entries for the
!perimeter nodes too. They can be zeros.
!-  Loss cannot go negative.
!# Number of points
npoint = SIZE(s)

!# Initialize the drainage_area and discharge arrays. Drainage area starts
!# out as the area of the cell in question, THEN (unless the cell has no
!# donors) grows from there. Discharge starts out as the cell's local runoff
!# rate times the cell's surface area.
!#drainage_area = np.zeros(npoint, dtype=int) + node_cell_area
!discharge = node_cell_area * runoff
!# note no loss occurs at a node until the water actually moves along a link

!# Optionally zero out drainage area and discharge at boundary nodes
!IF boundary_nodes is not None THEN
!	discharge[boundary_nodes] = 0
!END IF
!# Iterate backward through the list, which means we work from upstream to
!# downstream.
DO i = npoint, 1, -1
donor = s(i)
recvr = r(donor)
IF (donor .ne. recvr) THEN
Q_out = discharge(donor)
Q_TLp = Qtl(donor)
Q_inip = Q_ini(donor)
Q_aofp = Q_aof(donor)


CALL TRANSLOSS(Criv(donor),Kt(donor),riv(donor),Q_aoft(donor),riv_std(donor),&
P3(donor), P4(donor), Kloss, Q_out, Q_TLp, Q_inip, Q_aofp)

discharge(recvr) = discharge(recvr) + Q_out
Qtl(donor) = Q_TLp
Q_ini(donor) = Q_inip
Q_aof(donor) = Q_aofp

!IF (Q_out .lt. 0) THEN
!PRINT*, Q_out, Q_TLp, Q_inip, discharge(donor)
!END IF

END IF
END DO
END SUBROUTINE

SUBROUTINE TRANSLOSS(Criv, Kt, riv, Qaoft, riv_std, P3, P4, Kloss, Qout, QTL, Q_ini, Qaof)
REAL, INTENT(IN) :: Criv, Kt, Qaoft, riv_std, P3, P4, Kloss
INTEGER, INTENT(IN) :: riv
REAL, INTENT(INOUT) :: Qout, QTL, Q_ini, Qaof
REAL :: Con, Qo, Qin, TL
!"""Transmission losses function
!Parameters
!----------
!Qw :		flow entering the cell
!Kloss :		Transmission losses factor
!Criv :		Conductivity
!Kt :		Decay parameter
!QTL :		Transimission losses
!Q_ini :		Initial flow
!riv :		river cell id
!Qaof :		Abstraction flux
!Qaoft :		Threshold abstraction
!riv_std :	River saturation deficit
!P3 :		Parameter
!P4 :		Parameter
!Returns
!-------
!Qout :		flow leaving the cell
!QTL :		Transimission losses
!Q_ini :		Initial flow
!Qaof :		Abstraction flux
!"""
!IF (Qout .lt. 0) THEN
!PRINT*, Qout, TL, Qo
!END IF
IF (riv .eq. 1) THEN
Qin = (Qout+Q_ini)
!# abstractions, it can be modified
IF (Qin .le. Qaof) THEN
Qaof = Qaoft*Qin*0.0
END IF
Qin = Qin-Qaoft

Q_ini = 0
QTL = 0

IF (Qin .gt. 0.0) THEN
IF (riv_std .le. 0.0) THEN
CALL EXP_DECAY(Qin, Kt, Qout, Qo)
TL = 0
ELSE
Con = Criv*Kloss
CALL EXP_DECAY_LOSS(Con, Qin, Kt, P3, P4, Qout, TL, Qo)

IF (TL .gt. riv_std) THEN
Qo = Qo+TL-riv_std
TL = riv_std
END IF

Q_ini = Qo
QTL = TL

END IF
END IF
END IF
END SUBROUTINE

SUBROUTINE EXP_DECAY_LOSS(TL, Qin, k, P3, P4, Qout, Qtl, Qo)
!eXPONENTIAL FUNCTION INCLUDING TRANSMISSION LOSSES
REAL, INTENT(IN) :: TL, Qin, k, P3, P4
REAL, INTENT(OUT) :: Qout, Qo, Qtl
REAL :: t, Q1



t = -(1/k)*log(P3/(Qin*k))

IF (t .gt. 0) THEN
IF (t .gt. 1) THEN
t = 1
END IF

Q1 = Qin*(1-exp(-k*t))
Qtl = Qin*k*P4*(1-exp(-k*t))+TL*t
Qout = Q1-Qtl
ELSE
Qout = 0
Qo = 0
END IF

IF (t .lt. 1) THEN
Qo = 0.0
Qtl = Qin-Qout
ELSE
Qo = Qin-Q1
END IF

END SUBROUTINE

SUBROUTINE EXP_DECAY(Qin, k, Qout, Qo)
!EXPONENTIAL FUNCTION WITH NO TRANSMISSION LOSSES
REAL, INTENT(IN) :: Qin, k
REAL, INTENT(OUT) :: Qout, Qo
Qout = Qin*(1-exp(-k))
Qo = Qin - Qout
END SUBROUTINE

END MODULE