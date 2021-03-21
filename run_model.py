import pandas as pd
import pytest
from run_DRYP import run_DRYP

import time 

startTime = time.time()
run_DRYP('../GW_1D/GW_1D_input.dmp')
print ('time=', time.time() - startTime)

