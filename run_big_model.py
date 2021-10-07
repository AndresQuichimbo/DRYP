import pandas as pd
import pytest
from run_DRYP_BD import run_DRYP

import time 

startTime = time.time()
run_DRYP('../Kenya/test_big_model.dmp')
print ('time=', time.time() - startTime)