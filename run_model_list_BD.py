import pandas as pd
import pytest
from main_DRYP import run_DRYP
import time

filename_list = 'DRYP_model_list.txt'
fmodels = pd.read_csv(filename_list)

for nmodel in [22]:#, 1#2, 13, 14, -22 23]:

	filename_inputs = fmodels['Model'][nmodel]
	
	startTime = time.time()
	
	run_DRYP(filename_inputs)
	
	print('time=', time.time() - startTime)