import numpy as np
from matplotlib import pyplot as plt
from Master import MoxMEq
import photochem_turnover_test3 as hv

def product_formation(khp2,khp3,khp4,kre,koa,ttot,readrates=None,tol=0.005,maxdt=1.0,eval_scale=10.0,N2=1.0,H2atm=0.0,kto=1.00,NH3=0.0,secondary=None,paramfile='photochem_params_test.txt'):
	
	# initialize products and fixed rates
	H2 = H2atm*670e-6/(293*0.082057366)*1e9
	NH3 = 0.0
	H2scale = 1e-9 * 0.082057366 * 293 / 670e-6


	# Set kf and kb for dononr acceptor dynamics
	kf = 0.0; kb = 0.0

	# compute the photochemical rates at the initial condition
	if readrates==None:
		[hv_rates,inputparam] = hv.run_rates(paramfile)
	else:
		[hv_rates,inputparam] = hv.run_rates(paramfile,readrates)
	# CAT1(), CAT2(), OX1(), OX2(), HS3(), HS4()
	[vcat1,vcat2,vox1,vox2,vHS3,vHS4] = hv_rates[1]
	vcat_1 = vcat1+vcat2; vox_1 = vox1+vox2
	if secondary != None:
		if readrates==None:
			[hv_rates2,inputparam2] = hv.run_rates(paramfile,secondary)
		else:
			secrates = [[] for i in range(len(secondary))]
			for z in range(len(secondary)):
				inputs = readrates + secondary[z]
				[hv_rates2,inputparam2] = hv.run_rates(paramfile,inputs)
				secrates[z] = hv_rates2[1]
		vcat_2 = secrates[0][0]+secrates[0][1]; vox_2 = secrates[0][2]+secrates[0][3];
		vcat_3 = secrates[1][0]+secrates[1][1]; vox_3 = secrates[1][2]+secrates[1][3];
		vcat_4 = secrates[2][0]+secrates[2][1]; vox_4 = secrates[2][2]+secrates[2][3];
		vcat_5 = secrates[3][0]+secrates[3][1]; vox_5 = secrates[3][2]+secrates[3][3];
		vcat_N = secrates[4][0]+secrates[4][1]; vox_N = secrates[4][2]+secrates[4][3];

	#  kET, kHT, kHT2, ktr, kAET, kAET2, kBET, k0, KA, KD, KA2, KD2, KL, nA, nD, nA2, nD2, nL, Ntr, L, D, A, D2, A2, PWR, wl, V, pl, eps

	[KL,nL,L,D,A,D2,A2,V] = [inputparam[i] for i in [12,17,19,20,21,22,23,26]]
	
	# set the time parameters for integration
	dt = 0.001 # sec
	ni = int(ttot / dt)
	next_eval = eval_scale * dt
	next_print = 60
	
	# initialize population
	[Pnone,Pbound] = hv.binomial_ligand(nL,KL,L)
	init_pop = L*Pbound * V * 1e9 # nmol
	D = D*V*1e9; Dinit = D
	A = A*V*1e9
	D2 = D2*V*1e9; D2init = D2
	A2 = A2*V*1e9
	P = np.zeros(12)
	P[2] = init_pop;

	pop_conc = lambda pop: pop*1e-9 / V  # M  
	conc_pop = lambda conc: conc*V*1e9 # nmol

	if secondary == None:
		vcat_2 = vcat; vox_2 = vox; vcat1_2 = vcat1; vcat2_2 = vcat2
		vox1_2 = vox1; vox2_2 = vox2; vHS3_2 = vHS3; v_HS4_2 = vHS4

	B = MoxMEq([vcat_1,vcat_2,vcat_3,vcat_4,vcat_5,vcat_N,vox_1,vox_2,vox_3,vox_4,vox_5,vox_N,khp2,khp3,khp4,kre,koa,kto,N2,H2])
	
	i = 0
	time_val = 0
	population = np.zeros([len(P),ni+1]); time_array = np.zeros(ni+1)
	NH3_array = np.zeros(ni+1); H2_array = np.zeros(ni+1)

	ein = 0;
	D_array = np.zeros(ni+1); A_array = np.zeros(ni+1); D2_array = np.zeros(ni+1); A2_array = np.zeros(ni+1)
	D_array[0] = pop_conc(D)*1000; A_array[0] = pop_conc(A)*1000;  D2_array[0] = pop_conc(D2)*1000; A2_array[0] = pop_conc(A2)*1000
	population[:,0] = P
	vcat_vals = np.zeros(ni+1);
	vcat_array = np.array([vcat_1]); vox_array = np.array([vox_1]); t2 = np.array([0])
	vcat_array2 = np.array([vcat_2]); vox_array2 = np.array([vox_2])
	while time_val < ttot:
		##### INTEGRATION OF THE COUPLED DIFFERENTIAL EQUATIONS

		# this is the population of states which can accept an electron (not E4(4H) and not E8)
		#     0       1      2     3      4     5      6        7       8   9    10   11
		#   INACT    MOX    E0     E1    E2     E3    E44H    E42n2h   E5   E6   E7   E8
		elec_pop1 = np.sum(np.array([P[i] for i in [1,2,3,4,5]]))
		hole_pop1 = np.sum(np.array([P[i] for i in [2,3,4,5,6]]))
		elec_pop2 = np.sum(np.array([P[i] for i in [7,8,9,10]]))
		hole_pop2 = np.sum(np.array([P[i] for i in [8,9,11]]))
	

		## Term 1: Euler
		F1 = dt*(khp2*P[4] + khp3*P[5] + (khp4+N2*kre)*P[6] - koa*P[7]*H2scale*H2)
		F1_P = dt*np.matmul(B.MEq,P)
		F1_N = dt*(vcat_N*P[9] + kto*P[11])   
		F1_D = dt*(0.5*P[2]*vox1 + 0.5*P[3]*secrates[0][2] + 0.5*P[4]*secrates[1][2] + 0.5*P[5]*secrates[2][2] + 0.5*P[6]*secrates[3][2] + 0.5*hole_pop2*secrates[4][2] - 0.5*elec_pop2*secrates[4][0] - 0.5*P[1]*vcat1 - 0.5*P[2]*secrates[0][0] - 0.5*P[3]*secrates[1][0] - 0.5*P[4]*secrates[2][0] - 0.5*P[5]*secrates[3][0] + 0.5*L*vHS4 - 0.5*L*vHS3 - kf*D*A2**2)
		F1_A = dt*(-0.5*P[2]*vox1 - 0.5*P[3]*secrates[0][2] - 0.5*P[4]*secrates[1][2] - 0.5*P[5]*secrates[2][2] - 0.5*P[6]*secrates[3][2] - 0.5*hole_pop2*secrates[4][2] + 0.5*elec_pop2*secrates[4][0] + 0.5*P[1]*vcat1 + 0.5*P[2]*secrates[0][0] + 0.5*P[3]*secrates[1][0] + 0.5*P[4]*secrates[2][0] -0.5*P[5]*secrates[3][0] - 0.5*L*vHS4 + 0.5*L*vHS3 + kf*D*A2**2)
		F1_D2 = dt*(P[2]*vox2 + P[3]*secrates[0][3] + P[4]*secrates[1][3] + P[5]*secrates[2][3] + P[6]*secrates[3][3] + hole_pop2*secrates[4][3] - elec_pop2*secrates[4][1] - P[1]*vcat2 - P[2]*secrates[0][1] - P[3]*secrates[1][1] - P[4]*secrates[2][1] - P[5]*secrates[3][1] - L*vHS4 + L*vHS3 - 2*kf*D*A2**2)
		F1_A2 = dt*(-1*P[2]*vox2 - P[3]*secrates[0][3] - P[4]*secrates[1][3] - P[5]*secrates[2][3] - P[6]*secrates[3][3] - hole_pop2*secrates[4][3] + elec_pop2*secrates[4][1] + P[1]*vcat2 + P[2]*secrates[0][1] + P[3]*secrates[1][1] + P[4]*secrates[2][1] + P[5]*secrates[3][1] + L*vHS4 - L*vHS3 - 2*kf*D*A2**2)
	

		# Compute the rates and populations, evaluated at the F2 condition: this is the most accurate, but is massively intensive
		P_F2 = P + 0.5*F1_P
		D_F2 = D + 0.5*F1_D; D2_F2 = D2 + 0.5*F1_D2
		A_F2 = A + 0.5*F1_A; A2_F2 = A2 + 0.5*F1_A2
		elec_pop1_F2 = np.sum(np.array([P_F2[i] for i in [1,2,3,4,5]]))
		hole_pop1_F2 = np.sum(np.array([P_F2[i] for i in [2,3,4,5,6]]))
		elec_pop2_F2 = np.sum(np.array([P_F2[i] for i in [7,8,9,10]]))
		hole_pop2_F2 = np.sum(np.array([P_F2[i] for i in [8,9,11]]))


		# Term 2
		F2 = dt*(khp2*P_F2[4] + khp3*P_F2[5] + (khp4+N2*kre)*P_F2[6] - koa*H2scale*P_F2[7]*(H2+0.5*F1))
		B2 = MoxMEq([vcat_1,vcat_2,vcat_3,vcat_4,vcat_5,vcat_N,vox_1,vox_2,vox_3,vox_4,vox_5,vox_N,khp2,khp3,khp4,kre,koa,kto,N2,H2+0.5*F1])
		F2_P = dt*np.matmul(B2.MEq,P_F2)
		F2_N = dt*(vcat_N*P_F2[9] + kto*P_F2[11])
		F2_D = dt*(0.5*P_F2[2]*vox1 + 0.5*P_F2[3]*secrates[0][2] + 0.5*P_F2[4]*secrates[1][2] + 0.5*P_F2[5]*secrates[2][2] + 0.5*P_F2[6]*secrates[3][2] + 0.5*hole_pop2_F2*secrates[4][2] - 0.5*elec_pop2_F2*secrates[4][0] - 0.5*P_F2[1]*vcat1 - 0.5*P_F2[2]*secrates[0][0] - 0.5*P_F2[3]*secrates[1][0] - 0.5*P_F2[4]*secrates[2][0] - 0.5*P_F2[5]*secrates[3][0] + 0.5*L*vHS4 - 0.5*L*vHS3 - kf*D_F2*A2_F2**2)
		F2_A = dt*(-0.5*P_F2[2]*vox1 - 0.5*P_F2[3]*secrates[0][2] - 0.5*P_F2[4]*secrates[1][2] - 0.5*P_F2[5]*secrates[2][2] - 0.5*P_F2[6]*secrates[3][2] - 0.5*hole_pop2_F2*secrates[4][2] + 0.5*elec_pop2_F2*secrates[4][0] + 0.5*P_F2[1]*vcat1 + 0.5*P_F2[2]*secrates[0][0] + 0.5*P_F2[3]*secrates[1][0] + 0.5*P_F2[4]*secrates[2][0] + 0.5*P_F2[5]*secrates[3][0] - 0.5*L*vHS4 + 0.5*L*vHS3 + kf*D_F2*A2_F2**2)
		F2_D2 = dt*(P_F2[2]*vox2 + P_F2[3]*secrates[0][3] + P_F2[4]*secrates[1][3] + P_F2[5]*secrates[2][3] + P_F2[6]*secrates[3][3] + hole_pop2_F2*secrates[4][3] - elec_pop2_F2*secrates[4][1] - P_F2[1]*vcat2 - P_F2[2]*secrates[0][1] - P_F2[3]*secrates[1][1] - P_F2[4]*secrates[2][1] - P_F2[5]*secrates[3][1] - L*vHS4 + L*vHS3 - 2*kf*D_F2*A2_F2**2)
		F2_A2 = dt*(-1*P_F2[2]*vox2 - P_F2[3]*secrates[0][3] - P_F2[4]*secrates[1][3] - P_F2[5]*secrates[2][3] - P_F2[6]*secrates[3][3] - hole_pop2_F2*secrates[4][3] + elec_pop2_F2*secrates[4][1] + P_F2[1]*vcat2 + P_F2[2]*secrates[0][1] + P_F2[3]*secrates[1][1] + P_F2[4]*secrates[2][1] + P_F2[5]*secrates[3][1] + L*vHS4 - L*vHS3 - 2*kf*D_F2*A2_F2**2)

		
		# compute the rates and populations, evaluated at the F3 condition: this is the most accurate, but is massively intensive
		P_F3 = P + 0.5*F2_P
		D_F3 = D + 0.5*F2_D; D2_F3 = D2 + 0.5*F2_D2
		A_F3 = A + 0.5*F2_A; A2_F3 = A2 + 0.5*F2_A2
		elec_pop1_F3 = np.sum(np.array([P_F3[i] for i in [1,2,3,4,5]]))
		hole_pop1_F3 = np.sum(np.array([P_F3[i] for i in [2,3,4,5,6]]))
		elec_pop2_F3 = np.sum(np.array([P_F3[i] for i in [7,8,9,10]]))
		hole_pop2_F3 = np.sum(np.array([P_F3[i] for i in [8,9,11]]))

	
		# Term 3
		F3 = dt*(khp2*P_F3[4] + khp3*P_F3[5] + (khp4+N2*kre)*P_F3[6] - koa*H2scale*P_F3[7]*(H2+0.5*F2))
		B3 = MoxMEq([vcat_1,vcat_2,vcat_3,vcat_4,vcat_5,vcat_N,vox_1,vox_2,vox_3,vox_4,vox_5,vox_N,khp2,khp3,khp4,kre,koa,kto,N2,H2+0.5*F2])
		F3_P = dt*np.matmul(B3.MEq,P_F3)
		F3_N = dt*(vcat_N*P_F3[9] + kto*P_F3[11])
		F3_D = dt*(0.5*P_F3[2]*vox1 + 0.5*P_F3[3]*secrates[0][2] + 0.5*P_F3[4]*secrates[1][2] + 0.5*P_F3[5]*secrates[2][2] + 0.5*P_F3[6]*secrates[3][2] + 0.5*hole_pop2_F3*secrates[4][2] - 0.5*elec_pop2_F3*secrates[4][0] - 0.5*P_F3[1]*vcat1 - 0.5*P_F3[2]*secrates[0][0] - 0.5*P_F3[3]*secrates[1][0] - 0.5*P_F3[4]*secrates[2][0] - 0.5*P_F3[5]*secrates[3][0] + 0.5*L*vHS4 - 0.5*L*vHS3 - kf*D_F3*A2_F3**2)
		F3_A = dt*(-0.5*P_F3[2]*vox1 - 0.5*P_F3[3]*secrates[0][2] - 0.5*P_F3[4]*secrates[1][2] - 0.5*P_F3[5]*secrates[2][2] - 0.5*P_F3[6]*secrates[3][2] - 0.5*hole_pop2_F3*secrates[4][2] + 0.5*elec_pop2_F3*secrates[4][0] + 0.5*P_F3[1]*vcat1 + 0.5*P_F3[2]*secrates[0][0] + 0.5*P_F3[3]*secrates[1][0] + 0.5*P_F3[4]*secrates[2][0] + 0.5*P_F3[5]*secrates[3][0] - 0.5*L*vHS4 + 0.5*L*vHS3 + kf*D_F3*A2_F3**2)
		F3_D2 = dt*(P_F3[2]*vox2 + P_F3[3]*secrates[0][3] + P_F3[4]*secrates[1][3] + P_F3[5]*secrates[2][3] + P_F3[6]*secrates[3][3] + hole_pop2_F3*secrates[4][3] - elec_pop2_F3*secrates[4][1] - P_F3[1]*vcat2 - P_F3[2]*secrates[0][1] - P_F3[3]*secrates[1][1] - P_F3[4]*secrates[2][1] - P_F3[5]*secrates[3][1] - L*vHS4 + L*vHS3 - 2*kf*D_F3*A2_F3**2)
		F3_A2 = dt*(-1*P_F3[2]*vox2 - P_F3[3]*secrates[0][3] - P_F3[4]*secrates[1][3] - P_F3[5]*secrates[2][3] - P_F3[6]*secrates[3][3] - hole_pop2_F3*secrates[4][3] + elec_pop2_F3*secrates[4][1] + P_F3[1]*vcat2 + P_F3[2]*secrates[0][1] + P_F3[3]*secrates[1][1] + P_F3[4]*secrates[2][1] + P_F3[5]*secrates[3][1] + L*vHS4 - L*vHS3 - 2*kf*D_F3*A2_F3**2)
		
		
		# compute the rates and populations, evaluated at the F4 conditions: this is the most accurate, but is massively intensive
		P_F4 = P + F3_P
		D_F4 = D + F3_D; D2_F4 = D2 + F3_D2
		A_F4 = A + F3_A; A2_F4 = A2 + F3_A2
		elec_pop1_F4 = np.sum(np.array([P_F4[i] for i in [1,2,3,4,5]]))
		hole_pop1_F4 = np.sum(np.array([P_F4[i] for i in [2,3,4,5,6]]))
		elec_pop2_F4 = np.sum(np.array([P_F4[i] for i in [7,8,9,10]]))
		hole_pop2_F4 = np.sum(np.array([P_F4[i] for i in [8,9,11]]))
	
		# Term 4
		F4 = dt*(khp2*P_F4[4] + khp3*P_F4[5] + (khp4+N2*kre)*P_F4[6] - koa*H2scale*P_F4[7]*(H2+F3))
		B4 = MoxMEq([vcat_1,vcat_2,vcat_3,vcat_4,vcat_5,vcat_N,vox_1,vox_2,vox_3,vox_4,vox_5,vox_N,khp2,khp3,khp4,kre,koa,kto,N2,H2+F3])
		F4_P = dt*np.matmul(B4.MEq,P_F3)
		F4_N = dt*(vcat_N*P_F4[9] + kto*P_F4[11])
		F4_D = dt*(0.5*P_F4[2]*vox1 + 0.5*P_F4[3]*secrates[0][2] + 0.5*P_F4[4]*secrates[1][2] + 0.5*P_F4[5]*secrates[2][2] + 0.5*P_F4[6]*secrates[3][2] + 0.5*hole_pop2_F4*secrates[4][2] - 0.5*elec_pop2_F4*secrates[4][0] - 0.5*P_F4[1]*vcat1 - 0.5*P_F4[2]*secrates[0][0] - 0.5*P_F4[3]*secrates[1][0] - 0.5*P_F4[4]*secrates[2][0] - 0.5*P_F4[5]*secrates[3][0] + 0.5*L*vHS4 - 0.5*L*vHS3 - kf*D_F4*A2_F4**2)
		F4_A = dt*(-0.5*P_F4[2]*vox1 - 0.5*P_F4[3]*secrates[0][2] - 0.5*P_F4[4]*secrates[1][2] - 0.5*P_F4[5]*secrates[2][2] - 0.5*P_F4[6]*secrates[3][2] - 0.5*hole_pop2_F4*secrates[4][2] + 0.5*elec_pop2_F4*secrates[4][0] + 0.5*P_F4[1]*vcat1 + 0.5*P_F4[2]*secrates[0][0] + 0.5*P_F4[3]*secrates[1][0] + 0.5*P_F4[4]*secrates[2][0] + 0.5*P_F4[5]*secrates[3][0] - 0.5*L*vHS4 + 0.5*L*vHS3 + kf*D_F4*A2_F4**2)
		F4_D2 = dt*(P_F4[2]*vox2 + P_F4[3]*secrates[0][3] + P_F4[4]*secrates[1][3] + P_F4[5]*secrates[2][3] + P_F4[6]*secrates[3][3] + hole_pop2_F4*secrates[4][3] - elec_pop2_F4*secrates[4][1] - P_F4[1]*vcat2 - P_F4[2]*secrates[0][1] - P_F4[3]*secrates[1][1] - P_F4[4]*secrates[2][1] - P_F4[5]*secrates[3][1] - L*vHS4 + L*vHS3 - 2*kf*D_F4*A2_F4**2)
		F4_A2 = dt*(-1*P_F4[2]*vox2 - P_F4[3]*secrates[0][3] - P_F4[4]*secrates[1][3] - P_F4[5]*secrates[2][3] - P_F4[6]*secrates[3][3] - hole_pop2_F4*secrates[4][3] + elec_pop2_F4*secrates[4][1] + P_F4[1]*vcat2 + P_F4[2]*secrates[0][1] + P_F4[3]*secrates[1][1] + P_F4[4]*secrates[2][1] + P_F4[5]*secrates[3][1] + L*vHS4 - L*vHS3 - 2*kf*D_F4*A2_F4**2)

		# compute the difference in the populations
		dH2 = (1/6)*F1 + (1/3)*F2 + (1/3)*F3 + (1/6)*F4
		dNH3 = (1/6)*F1_N + (1/3)*F2_N + (1/3)*F3_N + (1/6)*F4_N
		dP = (1/6)*F1_P + (1/3)*F2_P + (1/3)*F3_P + (1/6)*F4_P
		dD = (1/6)*F1_D + (1/3)*F2_D + (1/3)*F3_D + (1/6)*F4_D
		dA = (1/6)*F1_A + (1/3)*F2_A + (1/3)*F3_A + (1/6)*F4_A
		dD2 = (1/6)*F1_D2 + (1/3)*F2_D2 + (1/3)*F3_D2 + (1/6)*F4_D2
		dA2 = (1/6)*F1_A2 + (1/3)*F2_A2 + (1/3)*F3_A2 + (1/6)*F4_A2
		#din = (1/6)*F1_in + (1/3)*F2_in + (1/3)*F3_in + (1/6)*F4_in
		#dout = (1/6)*F1_out + (1/3)*F2_out + (1/3)*F3_out + (1/6)*F4_out
	
		
		# Now evaluate the populations at t+dt
		H2 = H2 + dH2
		NH3 = NH3 + dNH3
		P = P + dP
		D = D + dD
		A = A + dA
		D2 = D2 + dD2
		A2 = A2 + dA2

		population[:,i] = P
		time_array[i] = time_val
		NH3_array[i] = NH3
		H2_array[i] = H2
		D_array[i] = pop_conc(D)*1000; D2_array[i] = pop_conc(D2)*1000
		A_array[i] = pop_conc(A)*1000; A2_array[i] = pop_conc(A2)*1000
		

		# Evaluate step size
		dAll = np.array(list(dP)+[dH2]+[dNH3]+[dD]+[dA]+[dD2]+[dA2])
		abs_dy = np.abs(dAll)
		max_abs_dy = np.max(abs_dy)

		dAD = np.array([dD]+[dA])
		abs_dAD = np.abs(dAD)
		max_abs_dAD = np.max(abs_dAD)

		if max_abs_dy < tol:
			dt = dt*1.1
			if dt > maxdt:
				dt = maxdt
		elif max_abs_dy > tol:
			i = i-1
			dt = dt/1.05


		# Evaluate systme parameters and recompute matrix
		if time_val >= next_eval:
			if readrates==None:
				[hv_rates,inputparam] = hv.run_rates(paramfile,[['D',pop_conc(D)],['A',pop_conc(A)],['D2',pop_conc(D2)],['A2',pop_conc(A2)]])
				if secondary != None:
					inputs = secondary + [['D',pop_conc(D)],['A',pop_conc(A)],['D2',pop_conc(D2)],['A2',pop_conc(A2)]]
					[hv_rates2,inputparam2] = hv.run_rates(paramfile,inputs)
			else:
				inputs = readrates + [['D',pop_conc(D)],['A',pop_conc(A)],['D2',pop_conc(D2)],['A2',pop_conc(A2)]]
				[hv_rates,inputparam] = hv.run_rates(paramfile,inputs)
				if secondary != None:
					secrates = [[] for i in range(len(secondary))]
					for z in range(len(secondary)):
						inputs = readrates + secondary[z] + [['D',pop_conc(D)],['A',pop_conc(A)],['D2',pop_conc(D2)],['A2',pop_conc(A2)]]
						[hv_rates2,inputparam2] = hv.run_rates(paramfile,inputs)
						secrates[z] = hv_rates2[1]

			[vcat1,vcat2,vox1,vox2,vHS3,vHS4] = hv_rates[1]
			vcat_1 = vcat1+vcat2; vox_1 = vox1+vox2
			
			if secondary == None:
				vcat_2 = vcat; vox_2 = vox; vcat1_2 = vcat1; vcat2_2 = vcat2
				vox1_2 = vox1; vox2_2 = vox2; vHS3_2 = vHS3; v_HS4_2 = vHS4

			vcat_array = np.append(vcat_array,vcat_1)
			vox_array = np.append(vox_array,vox_1)

			if secondary != None:
				vcat_2 = secrates[0][0]+secrates[0][1]; vox_2 = secrates[0][2]+secrates[0][3];
				vcat_3 = secrates[1][0]+secrates[1][1]; vox_3 = secrates[1][2]+secrates[1][3];
				vcat_4 = secrates[2][0]+secrates[2][1]; vox_4 = secrates[2][2]+secrates[2][3];
				vcat_5 = secrates[3][0]+secrates[3][1]; vox_5 = secrates[3][2]+secrates[3][3];
				vcat_N = secrates[4][0]+secrates[4][1]; vox_N = secrates[4][2]+secrates[4][3];

				vcat_array2 = np.append(vcat_array2,vcat_2); vox_array2 = np.append(vox_array2,vox_2)
			
			t2 = np.append(t2,time_val)
			next_eval = time_val+dt*eval_scale

		
		B = MoxMEq([vcat_1,vcat_2,vcat_3,vcat_4,vcat_5,vcat_N,vox_1,vox_2,vox_3,vox_4,vox_5,vox_N,khp2,khp3,khp4,kre,koa,kto,N2,H2])
		
		if time_val >= next_print:
			print('time = %.1f min; e in prod = %.3f nmol; h in A = %.3f nmol; h in A2 = %.3f nmol;  D  = %.3f nmol; D2 = %.3f nmol' % (time_val/60,2*H2+3*NH3,2*A,A2,2*D,D2))

			print('vcat = %.3f s-1' % vcat_1)
			print('vcat_2 = %.3f s-1' % vcat_2)
			print('\n')
			next_print = time_val + 60

		i += 1; time_val += dt

	population = population[:,:i]
	time_array = time_array[:i]
	NH3_array = NH3_array[:i]
	H2_array = H2_array[:i]
	D_array = D_array[:i]; D2_array = D2_array[:i]
	A_array = A_array[:i]; A2_array = A2_array[:i]

	#fOUT = open('vcats.txt','w')
	#for i in range(len(t2)):
	#	print('%.6f\t%.6f' % (t2[i],vcat_array[i]),file=fOUT)
	#fOUT.close()

	population[2,:] = population[2,:] + Pnone*L
	return [time_array,t2,vcat_array,vcat_array2,vox_array,vox_array2,H2_array,NH3_array,D_array,A_array,D2_array,A2_array,population]

