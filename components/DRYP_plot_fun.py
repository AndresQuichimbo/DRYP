import matplotlib.pyplot as plt

import pandas as pd

class DRYP_plot(object):

	def __init__(self,):		
		pass
		
	def plot_avg_var(fname, fname_out, delta_t='D'):
		df = pd.read_csv(fname)		
		head = list(df)[1:]		
		df["Date"] = pd.to_datetime(df['Date'])		
		df.index = pd.DatetimeIndex(df['Date'])		
		df2 = df.resample(delta_t).mean().reset_index()		
		df = df.resample(delta_t).sum().reset_index()		
		fig, (ax) = plt.subplots(len(head),1, sharex = True)		
		fig.set_size_inches(9,len(head)*1.25)
		
		for ilabel, iax in zip(head,fig.axes):			
			if ilabel == 'THT':				
				iax.plot(df2['Date'],df2[ilabel])				
			else:			
				iax.plot(df['Date'],df[ilabel])			
			iax.set_ylabel(ilabel)
			
		plt.legend(frameon = False)		
		iax.set_xlabel('Date')		
		fig.tight_layout()		
		plt.savefig(fname_out,dpi = 100)
		
	def plot_point_var(fname, fname_out, delta_t = 'D'):	
		df = pd.read_csv(fname)		
		head = list(df)[1:]		
		df["Date"] = pd.to_datetime(df['Date'])		
		df.index = pd.DatetimeIndex(df['Date'])		
		df = df.resample(delta_t).mean().reset_index()		
		fig, (ax) = plt.subplots(1,1, sharex = True)		
		fig.set_size_inches(9,2.5)		
		
		for ilabel in head:			
			ax.plot(df['Date'],df[ilabel], label  = ilabel)
			
		plt.legend(frameon = False)		
		ax.set_xlabel('Date')		
		fig.tight_layout()		
		plt.savefig(fname_out,dpi = 100)