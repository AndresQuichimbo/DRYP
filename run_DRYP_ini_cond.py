import pandas as pd
import numpy as np
import pytest
from main_DRYP import run_DRYP
import time
import fileinput
import sys
from shutil import copyfile
import rasterio

def replaceAll(file,searchExp,replaceExp):
    for line in fileinput.input(file, inplace=1):
        if searchExp in line:
            line = line.replace(searchExp,replaceExp)
        sys.stdout.write(line)

folder = [
"../Kenya/EW_v1_1k_MSWEAP_3h/EW_",
"../Kenya/EW_v1_1k_ERA_3h/EW_",
"../Kenya/EW_v1_1k_IMERG_3h/EW_",
"../Kenya/test_big/EW_"
]

folder_out = "../Kenya/EW_input_1k/EW_1k"

Model = ['M', 'E', 'I', 'B']

filename_list = 'DRYP_model_list.txt'
fmodels = pd.read_csv(filename_list)
# Imerg i = 2 nmodel = 23
# MSWEP i = 0 nmodel = 22
i = 2
nmodel = 23
tol = 1.0
rdiff = 100.
iter = 0
while rdiff >= tol:#for nmodel in [22]:#, 1#2, 13, 14, -22 23]:

	filename_inputs = fmodels['Model'][nmodel]
	
	startTime = time.time()
	
	run_DRYP(filename_inputs)
	
	print('time=', time.time() - startTime)
	#/home/c1755103/Kenya/test_big/EW_avg_wte_ini.asc
	
	# simulated file name
	fname_in = folder[i] + "avg_wte_ini.asc"
	# initial condition
	fname_out = folder_out + '_wte_t0_' + Model[i] +'.asc'
	
	# read raster data
	raster_t0 = rasterio.open(fname_in).read(1)
	raster_t1 = rasterio.open(fname_out).read(1)
	
	# compare data
	rerror = (np.abs(raster_t1 - raster_t0))/raster_t1
	rdiff_wt = np.nanmax(rerror)
	
	#if rdiff_wt*100 < tol:
	# copy water table
	copyfile(fname_in, fname_out)
	
	#replaceAll(fname_out,
	#			"cellsize 925.145414826",
	#			"cellsize 925.145414826388")
	
	# copy soil moisture
	fname_in = folder[i] + 'avg_tht_ini.asc'
	fname_out = folder_out + '_tht_t0_' + Model[i] +'.asc'
	
	# read raster data
	raster_t0 = rasterio.open(fname_in).read(1)
	raster_t1 = rasterio.open(fname_out).read(1)
	
	# compare data
	rerror = (np.abs(raster_t1 - raster_t0))/raster_t1
	rdiff_tht = np.nanmax(rerror)
	
	#if rdiff_tht*100 > tol:
	# copy water table
	copyfile(fname_in, fname_out)
	
	#replaceAll(fname_out,
	#			"cellsize 925.145414826",
	#			"cellsize 925.145414826388")
		
	rdiff = np.max([rdiff_wt, rdiff_tht])*100.0
	
	iter += 1
	print(rdiff, iter)