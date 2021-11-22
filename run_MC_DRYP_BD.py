from components.DRYP_Gen_Func import write_sim_file
import pandas as pd
from main_DRYP import run_DRYP
from shutil import copyfile
import time
filename_input = '../Kenya/EW_IMERG_input_3h.dmp'
#filename_input = '../Kenya/EW_MSWEAP_input_3h.dmp'
#filename_input = '../Kenya/EW_IMERG_input_v1.dmp'
#filename_input = '../Kenya/EW_MSWEAP_input_v1.dmp'
filename_paramerer = '../Kenya/KE_parameter_set_b.csv'
#filename_paramerer = '../Kenya/KE_parameter_set_MSWEP.csv'
parameter = pd.read_csv(filename_paramerer)

#Create a copy of inputfile
fname_ext = filename_input.split('.')[-1]
filename_inputs = filename_input[:-(len(fname_ext)+1)]+'m.txt'

for npar in range(0, 20):#len(parameter)):	2
	copyfile(filename_input, filename_inputs)
	write_sim_file(filename_inputs, parameter.loc[npar])
	startTime = time.time()
	
	run_DRYP(filename_inputs)
	
	print('time=', time.time() - startTime)