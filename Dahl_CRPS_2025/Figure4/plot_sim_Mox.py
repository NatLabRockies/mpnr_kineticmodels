#!/Users/pdahl/anaconda3/bin/python
import numpy as np
from Master import MoxMEq
from matplotlib import pyplot as plt
import matplotlib.font_manager as font_manager
import csv


def plot_sim(params,dat_array=[]):

	# Read in experimental data
#	with open('estate_populations.csv',newline='\n') as csvfile:
#	        data = csv.reader(csvfile,delimiter=',',quotechar='|')
#	        dat_array = np.array([[float(i) for i in row] for row in data])


	# set some initial guesses for the rates
	#[ko0,ko1,kro,khp2,khp3,khp4,k08,k10,k21,k32,k43,k54,k65,k76,k87,kre,koa] = params[:17] 
	

	[ko0,kro,ko1,khp2,khp3,khp4,kf,kre,koa,kto] = params[:10]
	
	
	k08=kto
	k10=kf
	k21=kf
	k32=kf
	k43=kf
	k54=kf
	k65=kf
	k76=kf
	k87=kf
	
	N2=1.0
	
	
	
	
	
	#A = MoxMEq([ko0,ko1,kro,khp2,khp3,khp4,k08,k10,k21,k32,k43,k54,k65,k76,k87,kre,koa,N2,H2])



	#rates = params[:17]

	Pfit = params[10:21] 

	[y0,N2,H2] = params[21:]
	N2=1.0; H2=0.0
	
	A = MoxMEq([ko0,ko1,kro,khp2,khp3,khp4,k08,k10,k21,k32,k43,k54,k65,k76,k87,kre,koa,N2,H2])
	
	# Define the initial condition
	P = np.array(Pfit)
	P[1] += -1*y0
	
	
	# Now integrate

	if len(dat_array) == 0:
		ni = 5000
		dt = 0.1 # min
	else:
		ni = int(dat_array[-1,0]*20)
		dt = int(dat_array[-1,0]) / ni # minutes

	i = 0
	time_val = 0
	population = np.zeros([len(P),ni+1])
	population[:,0] = P
	H2 = 0.0
	while i < ni:
		F1 = dt*(khp2*P[3] + khp3*P[4] + (khp4+N2*kre)*P[5] - koa*P[6]*H2)
		F2 = dt*(khp2*P[3] + khp3*P[4] + (khp4+N2*kre)*P[5] - koa*P[6]*(H2+0.5*F1))
		F3 = dt*(khp2*P[3] + khp3*P[4] + (khp4+N2*kre)*P[5] - koa*P[6]*(H2+0.5*F2))
		F4 = dt*(khp2*P[3] + khp3*P[4] + (khp4+N2*kre)*P[5] - koa*P[6]*(H2+F3))
		
		H2 = H2 + (1/6)*F1 + (1/3)*F2 + (1/3)*F3 + (1/6)*F4
		
		F1_2 = dt*np.matmul(A.MEq,P)
		F2_2 = dt*np.matmul(A.MEq,(P+0.5*F1_2))
		F3_2 = dt*np.matmul(A.MEq,(P+0.5*F2_2))
		F4_2 = dt*np.matmul(A.MEq,(P+F3_2))

		P = P + (1/6)*F1_2 + (1/3)*F2_2 + (1/3)*F3_2 + (1/6)*F4_2

		A = MoxMEq([ko0,ko1,kro,khp2,khp3,khp4,k08,k10,k21,k32,k43,k54,k65,k76,k87,kre,koa,N2,H2])
		
		i += 1; time_val += dt
		population[:,i] = P
	
	#print(H2)
	#population = population * (1.0 - y0)  
	population[1,:] += y0
	

	time_array = np.linspace(0,time_val,ni+1)

	hfont = {'fontname':'Helvetica'}
	leg_font = font_manager.FontProperties(family='Helvetica',style='normal',size=14)
	
	fig = plt.figure(figsize=(8,7))
	plt.grid(color=[0.9, 0.9, 0.9],linestyle='--')
	ax = fig.gca()
	ax.set_axisbelow(True)
	
	fig2 = plt.figure(figsize=(8,7))
	plt.grid(color=[0.9, 0.9, 0.9],linestyle='--')
	ax2 = fig2.gca()
	ax2.set_axisbelow(True)

	fig3 = plt.figure(figsize=(8,7))
	plt.grid(color=[0.9,0.9,0.9],linestyle='--')
	ax3 = fig3.gca()
	ax3.set_axisbelow(True)
	
	if len(dat_array) != 0:
		ax.scatter(dat_array[:,0],dat_array[:,1],marker='o',s=105,color=[0.88,0.88,0.88],edgecolors='k',linewidths=2)
		ax2.scatter(dat_array[:,0],dat_array[:,2],marker='o',s=105,color=[0.68,0.85,0.90],edgecolors='k',linewidths=2)

		ax3.scatter(dat_array[:,0],dat_array[:,4],marker='o',s=105,color=[0.68,0.85,0.90],edgecolors='k',linewidths=2)
		ax3.scatter(dat_array[:,0],dat_array[:,3],marker='o',s=105,color=[0.76,0.85,0.72],edgecolors='k',linewidths=2)
		ax3.scatter(dat_array[:,0],dat_array[:,5],marker='o',s=105,color=[0.69,0.61,0.85],edgecolors='k',linewidths=2)
		#ax2.scatter(dat_array[:,0],dat_array[:,6],marker='o',s=105,color=[1.00,0.71,0.76],edgecolors='k',linewidths=2)
	
	ax.plot(time_array,population[1,:],linewidth=2,color='k',label=r'E$_{0}$')
	ax2.plot(time_array,population[3,:],linewidth=2,color='b',label=r'E$_{2}$(2H)')

	ax3.plot(time_array,population[5,:],linewidth=2,color='g',label=r'E$_{4}$(4H)')
	ax3.plot(time_array,population[6,:],linewidth=2,color='c',label=r'E$_{4}$(2N2H)')
	ax3.plot(time_array,population[8,:],linewidth=2,color=[0.5,0.0,0.5],label=r'E$_{6}$')
	#ax2.plot(time_array,population[9,:],linewidth=2,color='m',label=r'E$_{8}$')

	#ax3.plot(time_array,population[2,:],linewidth=2,linestyle='--',color=[0.5,0.5,0.5],label=r'E$_{1}$')
	#ax3.plot(time_array,population[0,:],linewidth=2,linestyle='--',color=[0.93,0.90,0.0],label=r'M$^{OX}$')
	#ax3.plot(time_array,population[7,:],linewidth=2,linestyle='--',color='c',label=r'E$_{5}$')
	#ax3.plot(time_array,population[9,:],linewidth=2,linestyle='--',color='m',label=r'E$_{7}$')
	
	ax.set_xlabel('time (min)',fontsize=16,**hfont)
	ax2.set_xlabel('time (min)',fontsize=16,**hfont)
	ax3.set_xlabel('time (min)',fontsize=16,**hfont)

	xlab1 = 'concentration\n'; xlab2 = r'($\mu$M)'

	ax.set_ylabel(xlab1+xlab2,fontsize=16,**hfont)
	ax2.set_ylabel(xlab1+xlab2,fontsize=16,**hfont)
	ax3.set_ylabel(xlab1+xlab2,fontsize=16,**hfont)

	ax.set_xlim([0,time_array[-1]])
	ax2.set_xlim([0,time_array[-1]]); #ax2.set_ylim([0,0.05])
	ax3.set_xlim([0,time_array[-1]]); #ax3.set_ylim([0,0.2])

	plt.setp(ax.get_xticklabels(),fontsize=16,**hfont)
	plt.setp(ax.get_yticklabels(),fontsize=16,**hfont)
	plt.setp(ax2.get_xticklabels(),fontsize=16,**hfont)
	plt.setp(ax2.get_yticklabels(),fontsize=16,**hfont)
	plt.setp(ax3.get_xticklabels(),fontsize=16,**hfont)
	plt.setp(ax3.get_yticklabels(),fontsize=16,**hfont)

	ax.legend(prop=leg_font); ax2.legend(prop=leg_font); ax3.legend(prop=leg_font)
	plt.show()

	# Print results
	fOUT = open('populations.txt','w')
	for i in range(len(time_array)):
		print('%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f' % (time_array[i],population[0,i],population[1,i],population[2,i],population[3,i],population[4,i],population[5,i],population[6,i],population[7,i],population[8,i],population[9,i],population[10,i]),file=fOUT)
	
	fOUT.close()

	return 0
