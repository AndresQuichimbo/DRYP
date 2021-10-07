import pandas as pd
import pytest
from DRYP_extract_area import run_DRYP

filename_list = 'DRYP_model_list.txt'
fmodels = pd.read_csv(filename_list)

for nmodel in [14]:#, 12, 13, 14]:

	filename_inputs = fmodels['Model'][nmodel]
	
	run_DRYP(filename_inputs)
	
