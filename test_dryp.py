import numpy as np
import pandas as pd
from run_DRYP import run_DRYP

def test_dryp():
	"""run a test model file
	"""
	run_DRYP('test/input_test.dmp')
	
	df = pd.read_csv('test/output/test_OF_Dis.csv')
	ans = np.array(df['OF_0'])
	
	df = pd.read_csv('test/output_test/test_OF_Dis.csv')
	out = np.array(df['OF_0'])
	
	assert np.allclose(out, ans)

if __name__ == '__main__':
	test_dryp()