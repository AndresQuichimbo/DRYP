"""Create multiple simulation files for Monte Carlo analysis
"""
import sys
import os
import numpy as np
import pandas as pd

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
	
	f = pd.read_csv(input_file)
	f.drylandmodel[1] = f.drylandmodel[1] + str(parameter[0])
	
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
