import numpy as np
from Master import MoxMEq
from scipy.optimize import minimize
from scipy.linalg import solve
from matplotlib import pyplot as plt
import matplotlib.font_manager as font_manager


# initial guesses
#kf_0 = 0.015 
#khp_0 = 0.03
#kre_0 = 0.04
#koa_0 = 0.01
#ko0_0 = 0.02
#kro_0 = 0.04
#kto_0 = 0.2

kf_0 = 0.05
khp_0 = 0.05
kre_0 = 0.05
koa_0 = 0.05
ko0_0 = 0.05
ko1_0 = 0.05
kro_0 = 0.05
kto_0 = 0.05


# Read experimental data



def resid_calc(rates,expdat):
	global epr_pop2,populations,rmsd,tot_pop

	times = [200.0,225.0,200.0,200.0,200.0]
	
	diff = np.zeros([len(expdat),len(expdat[0])])
	epr_pop2 = np.zeros([5,5])
	tot_pop = np.zeros([5,11])
	for k in range(5):

		[kf,khp2,khp3,khp4,kre,koa,ko0,kro,ko1,kto] = rates[k]
		
		
		k10 =kf 
		k21 =kf 
		k32 =kf 
		k43 =kf 
		k54 =kf 
		k65 =kf 
		k76 =kf 
		k87 =kf
		
		k08 = kto
		#koa = 0;
		

		#ko0 = 0.0; kro = 0.0
		
		
		#rates = [ko0,ko1,kro,khp,khp,khp,k08,k10,k21,k32,k43,k54,k65,k76,k87,kre,koa]
		N2 = 1.0; H2 = 0.0;
		
		
		
		A = MoxMEq([ko0,ko1,kro,khp2,khp3,khp4,k08,k10,k21,k32,k43,k54,k65,k76,k87,kre,koa,N2,H2])

		ni = times[k] / 0.1
		dt = 0.1 # minutes
		i = 0; j = 1
		time_val = 0
		population = np.zeros(11)
		#population[:,0] = init_pop
		P = np.zeros(11); P[1] = 2.0;
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
			
			
			#rates[-1] = H2;
			A = MoxMEq([ko0,ko1,kro,khp2,khp3,khp4,k08,k10,k21,k32,k43,k54,k65,k76,k87,kre,koa,N2,H2])
			
			time_val += dt
		#	if abs(time_val-exp_times[j]) < 1e-5:
		#		population[:,j] = P
		#	        j += 1
			i += 1


		
		
		populations = P; #print(populations); print(np.sum(populations))
		epr_pop = [populations[l] for l in (1,3,5,6,8,10)]
		silent_pop = np.sum(np.array(populations)) - np.sum(np.array(epr_pop))
		#print(epr_pop)
		#print(len(epr_pop))
		
		#epr_pop2 = np.zeros(len(epr_pop)-1)
		#print(len(epr_pop2))
		epr_pop_comp = [epr_pop[l] for l in range(len(epr_pop)-2)]
		epr_pop2[k,:] = np.array(epr_pop_comp + [np.sum(epr_pop[-2:])])
		tot_pop[k,:] = populations

		#print(epr_pop2)

		diff_pop = list(epr_pop2[k,:]) + [silent_pop]
		
		diff[k,:] = np.array(expdat[k]) - np.array(diff_pop)
	rmsd = np.sqrt(np.mean(diff**2))

	return rmsd

def function_wrapper(x):
	global dat_array

	[kf1,kf2,kf3,kf4,kf5,khp1,khp2,khp3,kre,koa,ko01,ko02,ko03,ko04,ko05,kro,ko1,kto] = x
	rates1 = [kf1,khp1,khp2,khp3,kre,koa,ko01,kro,ko1,kto]
	rates2 = [kf2,khp1,khp2,khp3,kre,koa,ko02,kro,ko1,kto]
	rates3 = [kf3,khp1,khp2,khp3,kre,koa,ko03,kro,ko1,kto]
	rates4 = [kf4,khp1,khp2,khp3,kre,koa,ko04,kro,ko1,kto]; rates5 = [kf5,khp1,khp2,khp3,kre,koa,ko05,kro,ko1,kto]
	rates = [rates1,rates2,rates3,rates4,rates5]
	return resid_calc(rates,dat_array)


# Read experimental data
fIN = open('sample0mMDT_populations.txt').readlines()
dat_array1 = [float(fIN[i]) for i in range(len(fIN))]
silent = 2.0 - np.sum(np.array(dat_array1));
dat_array1 = dat_array1 + [silent]

fIN = open('sample5mMDT_populations.txt').readlines()
dat_array2 = [float(fIN[i]) for i in range(len(fIN))]
silent = 2.0 - np.sum(np.array(dat_array1));
dat_array2 = dat_array2 + [silent]

fIN = open('sample12mMDT_populations.txt').readlines()
dat_array3 = [float(fIN[i]) for i in range(len(fIN))]
silent = 2.0 - np.sum(np.array(dat_array2));
dat_array3 = dat_array3 + [silent]

fIN = open('sample20mMDT_populations.txt').readlines()
dat_array4 = [float(fIN[i]) for i in range(len(fIN))]
silent = 2.0 - np.sum(np.array(dat_array3));
dat_array4 = dat_array4 + [silent]

fIN = open('sample40mMDT_populations.txt').readlines()
dat_array5 = [float(fIN[i]) for i in range(len(fIN))]
silent = 2.0 - np.sum(np.array(dat_array4))
dat_array5 = dat_array5 + [silent]

dat_array = [dat_array1,dat_array2,dat_array3,dat_array4,dat_array5]



x0 = [kf_0,kf_0,kf_0,kf_0,kf_0,khp_0,khp_0,khp_0,kre_0,koa_0,ko0_0,ko0_0,ko0_0,ko0_0,ko0_0,kro_0,ko1_0,kto_0]
bnds = [(0.0,5.0),(0.0,5.0),(0.0,5.0),(0.0,5.0),(0.0,5.0),(0.0,5.0),(0.0,5.0),(0.0,5.0),(0.0,5.0),(0.0,5.0),(0.0,5.0),(0.0,5.0),(0.0,5.0),(0.0,5.0),(0.0,5.0),(0.0,5.0),(0.0,5.0),(0.0,5.0)]

mthd = 'nelder-mead'

res = minimize(function_wrapper,x0,method=mthd,bounds=bnds,tol=1e-8,options={'adaptive':True,'maxiter':1.0e6,'disp':True})
results = res.x

print('%.5f\n' % rmsd)
print(res.x)


fig = plt.figure()
ax = fig.gca()
ax.bar(np.arange(len(epr_pop2[0,:])),epr_pop2[0,:],width=0.7,color=[0.5,0.5,0.5],edgecolor='k',tick_label=[r'E$_0$',r'E$_2$(2H)',r'E$_4$(4H)',r'E$_4$(2N2H)',r'E$_{6/8}$'])
#ax.bar(np.arange(len(dat_array1)),dat_array1,width=0.7,color=[0.2,0.5,0.5],edgecolor='k')
ax.set_ylabel(r'population (spin mol$^{-1}$ MoFe protein)')

fig2 = plt.figure()
ax2 = fig2.gca()
ax2.bar(np.arange(len(epr_pop2[1,:])),epr_pop2[1,:],width=0.7,color=[0.5,0.5,0.5],edgecolor='k',tick_label=[r'E$_0$',r'E$_2$(2H)',r'E$_4$(4H)',r'E$_4$(2N2H)',r'E$_{6/8}$'])
ax2.set_ylabel(r'population (spin mol$^{-1}$ MoFe protein)')

fig3 = plt.figure()
ax3 = fig3.gca()
ax3.bar(np.arange(len(epr_pop2[2,:])),epr_pop2[2,:],width=0.7,color=[0.5,0.5,0.5],edgecolor='k',tick_label=[r'E$_0$',r'E$_2$(2H)',r'E$_4$(4H)',r'E$_4$(2N2H)',r'E$_{6/8}$'])
ax3.set_ylabel(r'population (spin mol$^{-1}$ MoFe protein)')

fig4 = plt.figure()
ax4 = fig4.gca()
ax4.bar(np.arange(len(epr_pop2[3,:])),epr_pop2[3,:],width=0.7,color=[0.5,0.5,0.5],edgecolor='k',tick_label=[r'E$_0$',r'E$_2$(2H)',r'E$_4$(4H)',r'E$_4$(2N2H)',r'E$_{6/8}$'])
ax4.set_ylabel(r'population (spin mol$^{-1}$ MoFe protein)')

plt.show()

fOUT = open('opt_populations.txt','w')
for i in range(np.shape(epr_pop2)[0]):
	print('%.5f\t%.5f\t%.5f\t%.5f\t%.5f' % (epr_pop2[i,0],epr_pop2[i,1],epr_pop2[i,2],epr_pop2[i,3],epr_pop2[i,4]),file=fOUT)

fOUT.close()

fOUT = open('tot_populations.txt','w')
for i in range(np.shape(tot_pop)[0]):
	print('%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f' % (tot_pop[i,0],tot_pop[i,1],tot_pop[i,2],tot_pop[i,3],tot_pop[i,4],tot_pop[i,5],tot_pop[i,6],tot_pop[i,7],tot_pop[i,8],tot_pop[i,9],tot_pop[i,10]),file=fOUT)

fOUT.close()

#params = ra`tes+list(pop)+[0.0,N2,H2];
#print(params[28:])

#ps.plot_sim(params)
