import pandas as pd
import pytest
from run_DRYP import run_DRYP

filename_list = 'DRYP_model_list.txt'
fmodels = pd.read_csv(filename_list)

for nmodel in [23, 24, 25, 26]:

	filename_inputs = fmodels['Model'][nmodel]

	run_DRYP(filename_inputs)