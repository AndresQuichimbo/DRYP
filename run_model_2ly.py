import pandas as pd
import pytest
from main_DRYP_2L import run_DRYP

import time 

startTime = time.time()
#C:\Users\km19051\OneDrive - Cardiff University\PhD\WS\LandLab\Channel\GW_1D_input_drain.dmp
#run_DRYP('../../WS/LandLab/Channel/GW_1D_input_drain.dmp')
#run_DRYP('../../WS/LandLab/GW_1D/GW_1D_input.dmp')
#run_DRYP('../GW_1D/GW_1D_input.dmp')
run_DRYP('../alluvium/AL_input.dmp')
print ('time=', time.time() - startTime)
