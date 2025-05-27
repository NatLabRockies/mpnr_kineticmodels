#!/bin/python

import numpy as np
from Master import MoxMEq
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import csv
import plot_sim_Mox as ps
import sys


def residuals(rates,y0,expdat,init_pop,elements=[1,3,5,6,8]):
	global fOUT, rmsd
	exp_times = expdat[:,0]


	[ko0,kro,ko1,khp2,khp3,khp4,kf,kre,koa,kto,N2,H2] = rates

	
	k08=kto
	k10=kf
	k21=kf
	k32=kf
	k43=kf
	k54=kf
	k65=kf
	k76=kf
	k87=kf

	#koa=0.0

	N2=1.0
	H2=0.0




	
	A = MoxMEq([ko0,ko1,kro,khp2,khp3,khp4,k08,k10,k21,k32,k43,k54,k65,k76,k87,kre,koa,N2,H2])

	#print(np.shape(A.MEq)); 
	init_pop_set = init_pop

	init_pop[1] += -1*y0

	ni = int(exp_times[-1]*20)
	dt = exp_times[-1] / ni # minutes
	i = 0; j = 1
	time_val = 0
	population = np.zeros([len(init_pop),len(exp_times)])
	population[:,0] = init_pop
	P = population[:,0]
	#print(np.shape(P))

	# Integration block
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
		

		rates[-1] = H2;
		A = MoxMEq([ko0,ko1,kro,khp2,khp3,khp4,k08,k10,k21,k32,k43,k54,k65,k76,k87,kre,koa,N2,H2])

		time_val += dt
		if abs(time_val-exp_times[j]) < 1e-5:
			population[:,j] = P
			j += 1
		i += 1
	population[1,:] += y0       	
	init_pop[1] += y0
		
#	poperr = np.mean(abs(np.sum(population,axis=0) - np.ones(len(exp_times))*0.97))

	population = population[elements,:]

	deviations = population.transpose() - expdat[:,[1,2,3,4,5]]

	deviations = deviations * np.array([1.0, 1.0, 1.0, 1.0, 0.0])
	deviations = np.reshape(deviations,(1,np.shape(deviations)[0]*np.shape(deviations)[1]))
	#print(deviations)
	deviations = [m for m in deviations[0]]
	pop_sum = np.sum(np.array(init_pop_set));
	pop_diff = 2.0-pop_sum
	deviations=deviations+[pop_diff]
	#print(deviations)
	deviations = np.array(deviations)

	#deviations = deviations + [2.0-np.sum(np.array(init_pop))]

	rmsd = np.sqrt(np.mean(deviations**2))#+poperr*0.11
	#print('%.5f' % rmsd)
	
	return rmsd

def function_wrapper(x):
	global y0,dat_array,N2,H2,khp,khp3,kf,kre,kto
	x = list(x)
	#print(x)
	gas = [N2,H2]
	#fixed1 = [k08]; fixed2 = [k65,k76,k87]

	r = x[:10]+gas
	P0 = x[10:len(x)-1]
	y0 = x[-1]
	return residuals(r,y0,dat_array,P0,elements=[1,3,5,6,8])
		

# Read in command line options
try:
	args = sys.argv[1:]
	for i in range(len(args)):
		if args[i] == '-plot':
			plot_dat = args[i+1]
		elif args[i] == '-expdat':
			data_fname = args[i+1]
		elif args[i] == '-opt_mthd':
			mthd = args[i+1]
		elif args[i] == '-expfmt':
			input_fmt = args[i+1]
		elif args[i] == '-atten':
			y0 = float(args[i+1])

except:
	do = None

# Set defaults
try: mthd
except NameError: mthd = 'nelder-mead'

try: data_fname
except NameError: data_fname = 'estate_populations.csv'

try: y0
except NameError: y0 = 0.0

try: input_fmt
except NameError: input_fmt = 'csv'

# Read in experimental data
if input_fmt == 'csv':
	with open(data_fname,newline='\n') as csvfile:
        	data = csv.reader(csvfile,delimiter=',',quotechar='|')
        	dat_array = np.array([[float(i) for i in row] for row in data])


elif input_fmt == 'txt':
#	try:
#		fileIN = open(data_fname,'r').readlines()
#		dat_array = [[] for i in range(len(fileIN))]
#		i = 0
#		for line in fileIN:
#			dat = line.split('\t')
#			dat_entry = [[] for k in range(len(dat))]
#			for j in range(len(dat)):
#				try:
#					dat_entry[j] = float(dat[j])
#				except:
#					value = dat[j].split('\n')
#					dat_entry[j] = float(value[0])
#			dat_array[i] = dat_entry
#			i += 1
#		dat_array = np.array(dat_array)
#
#	except:
	fileIN = open(data_fname,'r').read()
	fileIN = fileIN.split('\n')
	dat_array = [[] for i in range(len(fileIN))]
	i = 0
	for line in fileIN:
		#print(line)
		dat = line.split('\t')
		# check if last entry is not empty
		if len(dat[-1]) == 0:
			dat_entry = [[] for k in range(len(dat)-1)]
			looprange = len(dat)-1
		else:
			dat_entry = [[] for k in range(len(dat))]
			looprange = len(dat)
		for j in range(looprange):
			try:
				dat_entry[j] = float(dat[j])
			except:
				value = dat[j].split('\n')
				#print(value)
				dat_entry[j] = float(value[0])
		dat_array[i] = dat_entry
		i += 1
		#if i == len(fileIN)-1: break
	dat_array = np.array(dat_array)	


# set some initial guesses for the rates
N2 = 1.0
khp3 = 6.605E-08;            kre = 3.90047305e-02;     H2 = 0.0
koa = 4.34226806e-02;
khp2 = 5.20705061e-02;        kto = 2.13353936e-06;
khp4 = 7.376E-02;
kf  = 7.54386956e-02;        #k65 = 0.01;
#k21  = 0.032510;        k76 = 0.01;
#k32  = 0.028920;        k87 = 0.01;
#k43  = 0.054250;        k08 = 0.0004;

ko0 = 3.05777851e-02;
ko1 = 3.05777851e-02;
kro = 2.692E-03;

#[khp2,khp3,khp4,k10,k21,k32,k43] = [0.04757267, 0.00801429, 0.0100385,  0.05905873, 0.0896198,  0.14231328, 0.01000211]
#[khp2,khp3,khp4,k08,k10,k21,k32,k43,k54,kre,koa] = [0.1101133222026994, 0.143460587365964262, 0.15080031675960786, 0.00004, 0.06030548933476439, 0.09278409613261365, 0.0926639219461276, 0.02561486712544109, 0.02162648695447293, 0.06894680187341373, 0.018806257440490683]


# initialize the rates you wish to fit
#init_rates = [ko0,ko1,kro,khp2,khp3,khp4,k08,k10,k21,k32,k43,k54,k65,k76,k87,kre,koa]
#rate_bnds = [(0.0,0.0),(0.0,0.0),(0.0,0.0),(0.0001,0.35),(0.0001,0.35),(0.0001,0.35),(0.0,0.003),(0.01,0.17),(0.01,0.16),(0.02,0.22),(0.005,0.18),(0.005,0.18),(0.005,0.7),(0.005,0.7),(0.005,0.7),(0.005,0.7),(0.00001,0.7)]

init_rates = [ko0,kro,ko1,khp2,khp3,khp4,kf,kre,koa,kto]
rate_bnds = [(0.0,2.0),(0.0,2.0),(0.0,2.0),(0.0,2.0),(0.0,2.0),(0.0,2.0),(0.0,2.0),(0.0,2.0),(0.0,2.0),(0.0,2.0)]

P0 =       [2.86286799e-02,     1.74329333e+00,       8.25760393e-09,       6.49989026e-02,       1.61978534e-01,      0.00,        0.000,         0.0,        0.0,       0.0,        0.0]
pop_bnds = [(0.0,2.0),(0.0,2.0),(0.0,2.0),(0.0,2.0),(0.0,2.0),(0.0,2.0),(0.0,2.0),(0.0,0.01),(0.00,0.01),(0.0,0.01),(0.0,0.01)]

x0 = init_rates+P0+[y0]
y0_bnds = [(0.00,0.3)]

#bnds = [(0.08,0.3),(0.08,0.3),(0.0001,0.8),(0.0001,0.8),(0.0001,0.8),(0.0001,0.8),(0.0001,0.8),(0.0001,0.8),(0.0001,0.8),(0.0001,0.8),(0.0001,0.8),(0.0001,0.8),(0.0001,0.8),(0.0001,0.8)]
#bnds = [(0.01,0.15),(0.008,0.05),(0.01,0.1),(0.0,0.003),(0.01,0.17),(0.01,0.16),(0.02,0.22),(0.005,0.18),(0.005,0.18),(0.005,0.7),(0.005,0.7),(0.005,0.7),(0.005,0.7),(0.005,0.7),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0)]

#bnds = [(0.01,0.15),(0.008,0.05),(0.01,0.1),(0.0,0.003),(0.01,0.17),(0.01,0.16),(0.02,0.22),(0.005,0.18),(0.005,0.18),(0.005,0.7),(0.005,0.7),(0.005,0.7),(0.005,0.7),(0.005,0.7),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0)]

bnds = rate_bnds+pop_bnds+y0_bnds



mthd = 'nelder-mead'
print('\nPerforming least-squares fitting using the %s method...\n' % mthd)

res = minimize(function_wrapper,x0,method=mthd,bounds=bnds,tol=1e-6,options={'adaptive':True,'maxiter':5.0e2,'disp':True})

results = res.x; 

print(res)
print('\n\nThe final values for the parameters:\n')
print('[kinact,kreact,vox,khp2,khp3,khp4,vcat,kre,koa,kto,X,E0,E1,E2,E3,E4,E42n2h,E5,E6,E7,E8,y0] = ')
print(res.x)

#param_list = ['ko0','ko1','kro','kHP2','kHP3','kPH4','k80','k10','k21','k32','k43','k54','k65','k76','k87','kre','koa','y0']
param_list = ['ko0','kro','kHP','khp3','khp4','kf','kre','kto']
state_list = ['X','E0','E1','E2(2H)','E3','E4(4H)','E4(2N2H)','E5','E6','E7','E8']

print('\nFitting concluded.\n\n\nFinal RMSD: %.3E\n' % (rmsd))

print('\n\nFinal fit parameters:\n')
print('\n\nWARNING: THIS CODE IS UNDER CONSTRUCTION, THE RATE CONSTANTS ARE PRINTED OUT OF ORDER. REFER TO LIST OF PARAMETERS ABOVE\n')
print('--------------------------------------------------------------------------')
print('rate constant\tvalue (min-1)\t|\tE-state\t\tP0 (spin mol-1)')
print('--------------------------------------------------------------------------')

for i in range(len(param_list)):
	try:
		state = state_list[i]
		pop = list(res.x)[i+len(init_rates)]
		fmt = 'flt'
	except:
		state = ' '; pop = ' '; fmt = 'stg'

	if i == len(param_list)-1: ix = -1
	else: ix = i

	if fmt == 'flt':
		print('%s\t\t%.3E\t|\t%s\t\t%.4f' % (param_list[i],list(res.x)[ix],state,pop))
	elif fmt == 'stg':
		print('%s\t\t%.3E\t|\t%s\t\t%s' % (param_list[i],list(res.x)[ix],state,pop))

print('--------------------------------------------------------------------------')
print('\n\n')

params = list(res.x)+[N2,H2]


#try: 
plot_dat
if plot_dat:
	ps.plot_sim(params,dat_array)
#except:
#	do = None



