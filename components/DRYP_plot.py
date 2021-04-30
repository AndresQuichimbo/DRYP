from netCDF4 import Dataset, num2date, date2num
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

filed = ['pre', 'pet', 'aet', 'tht', 'inf',
		'run', 'tls', 'fch', 'dch', 'gdh', 'wte']
fname = '../test/output/test_avg.nc'		
#def plot_DRYP(fname):
	
field = ['pre', 'pet', 'aet', 'tht', 'inf',
	'run', 'tls', 'fch', 'dch', 'gdh', 'wte']

fpre = Dataset(fname, 'r')


	
#for ifield in enumerate(field):
ifield = field[-1]	
for i in range(len(fpre['time'][:])):

	plt.imshow(fpre[ifield][i,:,:], animated=True)
	
	plt.show()
	
#import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation
#
#fig = plt.figure()
#
#
#def f(x, y):
#    return np.sin(x) + np.cos(y)
#
#x = np.linspace(0, 2 * np.pi, 120)
#y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
#
#im = plt.imshow(f(x, y), animated=True)
#
#
#def updatefig(*args):
#    global x, y
#    x += np.pi / 15.
#    y += np.pi / 20.
#    im.set_array(f(x, y))
#    return im,
#
#ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
#plt.show()