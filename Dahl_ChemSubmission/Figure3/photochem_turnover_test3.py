import numpy as np
import math

#READ IN ALL GLOBAL VARIABLES
def get_params(param_file,param_spec=None):
	global kET, kHT, kHT2, ktr, kAET, kAET2, kBET, k0, KA, KD, KA2, KD2, KL, nA, nD, nA2, nD2, nL, Ntr, L, D, A, D2, A2, PWR, wl, V, pl, eps
	fIN = open(param_file).readlines()
	nparams = 0
	for line in fIN:
		vals = line.split()
		if vals[0] == 'kET':
			kET = float(vals[1]); nparams+=1
		elif vals[0] == 'kHT':
			kHT = float(vals[1]); nparams+=1
		elif vals[0] == 'kHT2':
			kHT2 = float(vals[1]); nparams+=1
		elif vals[0] == 'ktr':
			ktr = float(vals[1]); nparams+=1
		elif vals[0] == 'kAET':
			kAET = float(vals[1]); nparams+=1
		elif vals[0] == 'kAET2':
			kAET2 = float(vals[1]); nparams+=1
		elif vals[0] == 'kBET':
			kBET = float(vals[1]); nparams+=1
		elif vals[0] == 'k0':
			k0 = float(vals[1]); nparams+=1
		elif vals[0] == 'KA':
			KA = float(vals[1]); nparams+=1
		elif vals[0] == 'KD':
			KD = float(vals[1]); nparams+=1
		elif vals[0] == 'KA2':
			KA2 = float(vals[1]); nparams+=1
		elif vals[0] == 'KD2':
			KD2 = float(vals[1]); nparams+=1
		elif vals[0] == 'KL':
			KL = float(vals[1]); nparams+=1
		elif vals[0] == 'nA':
			nA = int(vals[1]); nparams+=1
		elif vals[0] == 'nD':
			nD = int(vals[1]); nparams+=1
		elif vals[0] == 'nD2':
			nD2 = int(vals[1]); nparams+=1
		elif vals[0] == 'nA2':
			nA2 = int(vals[1]); nparams+=1
		elif vals[0] == 'nL':
			nL = int(vals[1]); nparams+=1
		elif vals[0] == 'NTr':
			Ntr = float(vals[1]); nparams+=1
		elif vals[0] == 'L':
			L = float(vals[1]); nparams+=1
		elif vals[0] == 'D':
			D = float(vals[1]); nparams+=1
		elif vals[0] == 'A':
			A = float(vals[1]); nparams+=1
		elif vals[0] == 'D2':
			D2 = float(vals[1]); nparams+=1
		elif vals[0] == 'A2':
			A2 = float(vals[1]); nparams+=1
		elif vals[0] == 'PWR':
			PWR = float(vals[1]); nparams+=1
		elif vals[0] == 'wl':
			wl = float(vals[1]); nparams+=1
		elif vals[0] == 'V':
			V = float(vals[1]); nparams+=1
		elif vals[0] == 'pl':
			pl = float(vals[1]); nparams+=1
		elif vals[0] == 'eps':
			eps = float(vals[1]); nparams+=1
	if nparams != 29:
		print('ERROR: Total number of parameters should be 29. %s parameters were read' % nparams)
	else:
		do = None #cat = CAT(); ox = OX(); aet = AET(); tr = TR(); hs = HS()
	if param_spec != None:
		for par in param_spec:
			[name,pval] = par
			if name == 'kET':
				kET = float(pval)
			elif name == 'kHT':
				kHT = float(pval)
			elif name =='kHT2':
				kHT2 = float(pval)
			elif name == 'ktr':
				ktr = float(pval)
			elif name == 'kAET':
				kAET = float(pval)
			elif name == 'kAET2':
				kAET2 = float(pval)
			elif name == 'kBET':
				kBET = float(pval)
			elif name == 'k0':
				k0 = float(pval)
			elif name == 'KA':
				KA = float(pval)
			elif name == 'KD':
				KD = float(pval)
			elif name == 'KD2':
				KD2 = float(pval)
			elif name =='KA2':
				KA2 = float(pval)
			elif name == 'KL':
				KL = float(pval)
			elif name == 'nA':
				nA = int(pval)
			elif name == 'nD':
				nD = int(pval)
			elif name == 'nD2':
				nD2 = int(pval)
			elif name == 'nA2':
				nA2 = int(pval)
			elif name == 'nL':
				nL = int(pval)
			elif name == 'NTr':
				Ntr = float(pval)
			elif name == 'L':
				L = float(pval)
			elif name == 'D':
				D = float(pval)
			elif name == 'A':
				A = float(pval)
			elif name == 'A2':
				A2 = float(pval)
			elif name == 'D2':
				D2 = float(pval)
			elif name == 'PWR':
				PWR = float(pval)
			elif name == 'wl':
				wl = float(pval)
			elif name == 'V':
				V = float(pval)
			elif name == 'pl':
				pl = float(pval)
			elif name == 'eps':
				eps = float(pval)
	return 0
	

def binomial(i,n,K,L):
	p = K * L / (1 + K*L)
	B = math.comb(n,i) * p**i * (1 - p)**(n-i)
	return B

def binomial_ligand(n,K,L):
	Pone = np.sum(np.array([i*binomial(i,n,K,L) for i in range(1,n+1)]))
	Ptwo = 1 + np.sum(np.array([binomial(i,n,K,L) for i in range(1,n+1)]))
	Ptot = Pone / Ptwo
	prob = [1-Ptot,Ptot]
	return prob

def poisson(avgN,i):
	P = avgN**i / math.factorial(i) * math.exp(-1*avgN)
	return P

def QEET(cond=None):
	global z, BL
	# nL,KL,L,Ntr,nA,KA,A are global variables
	# We need to compute the ensemble w.r.t. the number of enzymes, electron traps, and acceptors
	# evaluate poisson distribution to find where P < 1e-5
	try: z
	except:
		z = 0; y = 1; P = 1
		while P >= 1e-5 and y > 0:
			P = poisson(Ntr,z); y = poisson(Ntr,z+1) - poisson(Ntr,z)
			z+=1
	# precalculate binomial ligand
	try: BL
	except:
		BL = binomial_ligand(nL,KL,L)
	if cond=='HT':
		QE = lambda i,j,k,l: i*kET / (i*kET + j*ktr + (k + 1)*kAET + l*kAET2)
		values = [BL[i]*poisson(Ntr,j)*binomial(k,nA-1,KA,A)*binomial(l,nA2,KA2,A2)*QE(i,j,k,l) for i in range(2) for j in range(z) for k in range(nA) for l in range(nA2+1)] 
	elif cond=='HT2':
		QE = QE = lambda i,j,k,l: i*kET / (i*kET + j*ktr + k*kAET + (l+1)*kAET2)
		values = [BL[i]*poisson(Ntr,j)*binomial(k,nA,KA,A)*binomial(l,nA2-1,KA2,A2)*QE(i,j,k,l) for i in range(2) for j in range(z) for k in range(nA+1) for l in range(nA2)]
	elif cond==None:
		QE = lambda i,j,k,l: i*kET / (i*kET + j*ktr + k*kAET + l*kAET2 + k0)
		values = [BL[i]*poisson(Ntr,j)*binomial(k,nA,KA,A)*binomial(l,nA2,KA2,A2)*QE(i,j,k,l) for i in range(2) for j in range(z) for k in range(nA+1) for l in range(nA2+1)]
	EnQEET = np.sum(np.array(values))
	return EnQEET 

def QEHT(cond=None):
	global BL
	# nD, KD, D, nL, KL, L are global variables
	# We need to compute the ensemble w.r.t. the number of donors, and enzymes
	# precalculate binomial ligand
	try: BL
	except:
		BL = binomial_ligand(nL,KL,L)
	if cond=='ET':
		QE = lambda i,j: i*kHT / (i*kHT + j*kHT2 + kBET)
		values = [binomial(i,nD,KD,D)*binomial(j,nD2,KD2,D2)*QE(i,j) for i in range(nD+1) for j in range(nD2+1)]
	#elif cond =='AET':
	#	QE = lambda i,j,k: 2*i*kHT / (2*i*kHT + j*kBET + k*kHT2)
	#	values = [binomial(i-1,nD-1,KD,D)*BL[j]*binomial(k,nD2,KD2,D)*QE(i,j,k) for i in range(1,nD+1) for j in range(2) for k in range(nD2+1)]
	elif cond=='AET2':
		QE = lambda i,j,k: i*kHT / (i*kHT + j*kBET + (k+1)*kHT2)
		values = [binomial(i,nD,KD,D)*BL[j]*binomial(k,nD2-1,KD2,D)*QE(i,j,k) for i in range(nD+1) for j in range(2) for k in range(nD2)]
	#elif cond=='Tr':
	#	QE = lambda i,j,k: 2*i*kHT / (2*i*kHT + j*kBET + k*kHT2 + k0) ### I am approximating HT to a Trap as recombination
	#	values = [binomial(i,nD,KD,D)*BL[j]*binomial(k,nD2,KD2,D)*QE(i,j,k) for i in range(nD+1) for j in range(2) for k in range(nD2+1)]
	elif cond==None:
		QE = lambda i,j,k: i*kHT / (i*kHT + j*kBET + k*kHT2 + k0)
		values = [binomial(i,nD,KD,D)*BL[j]*binomial(k,nD2,KD2,D2)*QE(i,j,k) for i in range(nD+1) for j in range(2) for k in range(nD2+1)]
	EnQEHT = np.sum(np.array(values))
	return EnQEHT

def QEHT2(cond=None):
	global BL
	# nD, KD, D, nD2, KD2, D2, nL, KL, L are global variables
	# We need to compute the ensemble w.r.t. the number of donors and enzymes
	# precalculate binomial ligand
	try: BL
	except:
		BL = binomial_ligand(nL,KL,L)
	if cond=='ET':
		QE = lambda i,j: j*kHT2 / (i*kHT + j*kHT2 + kBET)
		values = [binomial(i,nD,KD,D)*binomial(j,nD2,KD2,D2)*QE(i,j) for i in range(nD+1) for j in range(nD2+1)]
	elif cond=='AET':
		QE = lambda i,j,k: k*kHT2 / ((i + 1)*kHT + j*kBET + k*kHT2)
		values = [binomial(i,nD-1,KD,D)*BL[j]*binomial(k,nD2,KD2,D2)*QE(i,j,k) for i in range(nD) for j in range(2) for k in range(nD2+1)]
	#elif cond=='AET2':
	#	QE = lambda i,j,k: k*kHT2 / (2*i*kHT + j*kBET + (k+1)*kHT2)
	#	values = [binomial(i,nD,KD,D)*BL[j]*binomial(k,nD2-1,KD2,D2)*QE(i,j,k) for i in range(nD+1) for j in range(2) for k in range(nD2)]
	#elif cond=='Tr':
	#	QE = lambda i,j,k: k*kHT2 / (2*i*kHT + j*kBET + k*kHT2 + k0) ### I am approximating HT to a Trap as recombination
	#	values = [binomial(i,nD,KD,D)*BL[j]*binomial(k,nD2,KD2,D2)*QE(i,j,k) for i in range(nD+1) for j in range(2) for k in range(nD2+1)]
	elif cond==None:
		QE = lambda i,j,k: k*kHT2 / (i*kHT + j*kBET + k*kHT2 + k0)
		values = [binomial(i,nD,KD,D)*BL[j]*binomial(k,nD2,KD2,D2)*QE(i,j,k) for i in range(nD+1) for j in range(2) for k in range(nD2+1)]
	EnQEHT2 = np.sum(np.array(values))
	return EnQEHT2

def QEAET(cond=None):
	global z, BL
	# nL,KL,L,Ntr,nA,KA,A are global variables
	# We need to compute the ensemble w.r.t. the number of enzymes, electron traps, and acceptors
	# evaluate poisson distribution to find where P < 1e-5
	try: z
	except:
		z = 0; y = 1; P = 1
		while P >= 1e-5 and y > 0:
			P = poisson(Ntr,z); y = poisson(Ntr,z+1) - poisson(Ntr,z)
			z+=1
	# precalculate binomial ligand
	try: BL
	except:
		BL = binomial_ligand(nL,KL,L)
	#if cond=='HT':
	#	QE = lambda i,j,k,l: 2*i*kAET / (k*kET + j*ktr + 2*i*kAET + l*kAET2)
	#	values = [binomial(i-1,nA-1,KA,A)*poisson(Ntr,j)*BL[k]*binomial(l,nA2,KA2,A2)*QE(i,j,k,l) for i in range(1,nA+1) for j in range(z) for k in range(2) for l in range(nA2+1)]
	if cond=='HT2':
		QE = lambda i,j,k,l: i*kAET / (k*kET + j*ktr + i*kAET + (l+1)*kAET2)
		values = [binomial(i,nA,KA,A)*poisson(Ntr,j)*BL[k]*binomial(l,nA2-1,KA2,A2)*QE(i,j,k,l) for i in range(nA+1) for j in range(z) for k in range(2) for l in range(nA2)]
	if cond=='BET':
		QE = lambda i,j,k: i*kAET / (kET + j*ktr + i*kAET + k*kAET2)   
		values = [binomial(i,nA,KA,A)*poisson(Ntr,j)*binomial(k,nA2,KA2,A2)*QE(i,j,k) for i in range(nA+1) for j in range(z) for k in range(nA2+1)] # PROBABILITY OF ENZYME BOUND COLLAPSES TO 1
	elif cond==None:
		QE = lambda i,j,k,l: i*kAET / (k*kET + j*ktr + i*kAET + l*kAET2 + k0)
		values = [binomial(i,nA,KA,A)*poisson(Ntr,j)*BL[k]*binomial(l,nA2,KA2,A2)*QE(i,j,k,l) for i in range(nA+1) for j in range(z) for k in range(2) for l in range(nA2+1)]
	EnQEAET = np.sum(np.array(values))
	return EnQEAET

def QEAET2(cond=None):
	global z, BL
	# nL,KL,L,Ntr,nA,KA,A are global variables
	# We need to compute the ensemble w.r.t. the number of enzymes, electron traps, and acceptors
	# evaluate poisson distribution to find where P < 1e-5
	try: z
	except:
		z = 0; y = 1; P = 1
		while P >= 1e-5 and y > 0:
			P = poisson(Ntr,z); y = poisson(Ntr,z+1) - poisson(Ntr,z)
			z+=1
	# precalculate binomial ligand
	try: BL
	except:
		BL = binomial_ligand(nL,KL,L)
	if cond=='HT':
		QE = lambda i,j,k,l: l*kAET2 / (k*kET + j*ktr + (i + 1)*kAET + l*kAET2)
		values = [binomial(i,nA-1,KA,A)*poisson(Ntr,j)*BL[k]*binomial(l,nA2,KA2,A2)*QE(i,j,k,l) for i in range(nA) for j in range(z) for k in range(2) for l in range(nA2+1)]
	#elif cond=='HT2':
	#	QE = lambda i,j,k,l: l*kAET2 / (k*kET + j*ktr + 2*i*kAET + l*kAET2)
	#	values = [binomial(i,nA,KA,A)*poisson(Ntr,j)*BL[k]*binomial(l-1,nA2-1,KA2,A2)*QE(i,j,k,l) for i in range(nA+1) for j in range(z) for k in range(2) for l in range(1,nA2+1)]
	elif cond=='BET':
		QE = lambda i,j,k: k*kAET2 / (kET + j*ktr + i*kAET + k*kAET2)
		values = [binomial(i,nA,KA,A)*poisson(Ntr,j)*binomial(k,nA2,KA2,A2)*QE(i,j,k) for i in range(nA+1) for j in range(z) for k in range(nA2+1)] # PROBABILITY OF ENZYME BOUND COLLAPSES TO 1
	elif cond==None:
		QE = lambda i,j,k,l: l*kAET2 / (k*kET + j*ktr + i*kAET + l*kAET2 + k0)
		values = [binomial(i,nA,KA,A)*poisson(Ntr,j)*BL[k]*binomial(l,nA2,KA2,A2)*QE(i,j,k,l) for i in range(nA+1) for j in range(z) for k in range(2) for l in range(nA2+1)]
	EnQEAET2 = np.sum(np.array(values))
	return EnQEAET2

def QEBET(cond=None):
	global BL
	# nL, KL, L, nD, KD, D are global variables
	# precalculate binomial ligand
	try: BL
	except:
		BL = binomial_ligand(nL,KL,L)
	if cond=='AET':
		QE = lambda i,j,k: i*kBET / (i*kBET + (j + 1)*kHT + k*kHT2)
		values = [BL[i]*binomial(j,nD-1,KD,D)*binomial(k,nD2,KD2,D2)*QE(i,j,k) for i in range(2) for j in range(nD) for k in range(nD2+1)]
	elif cond=='AET2':
		QE = lambda i,j,k: i*kBET / (i*kBET + j*kHT + (k+1)*kHT2)
		values = [BL[i]*binomial(j,nD,KD,D)*binomial(k,nD2-1,KD2,D2)*QE(i,j,k) for i in range(2) for j in range(nD+1) for k in range(nD2)]
	#elif cond=='Tr':
	#	QE = lambda i,j,k: i*kBET / (i*kBET + 2*j*kHT + k*kHT2 + k0)
	#	values = [BL[i]*binomial(i,nD,KD,D)*binomial(k,nD2,KD2,D2)*QE(i,j,k) for i in range(2) for j in range(nD+1) for k in range(nD2+1)] ### I am approximating HT to a Trap as recombination
	elif cond==None:
		QE = lambda i,j,k: i*kBET / (i*kBET + j*kHT + k*kHT2 + k0)
		values = [BL[i]*binomial(j,nD,KD,D)*binomial(k,nD2,KD2,D2)*QE(i,j,k) for i in range(2) for j in range(nD+1) for k in range(nD2+1)]
	EnQEBET = np.sum(np.array(values))
	return EnQEBET

def QETr(cond=None):
	global z, BL
	# nL,KL,L,Ntr,nA,KA,A are global variables
	# We need to compute the ensemble w.r.t. the number of enzymes, electron traps, and acceptors
	# evaluate poisson distribution to find where P < 1e-5
	try: z
	except:
		z = 0; y = 1; P = 1
		while P >= 1e-5 and y > 0:
			P = poisson(Ntr,z); y = poisson(Ntr,z+1) - poisson(Ntr,z)
			z+=1
	# precalculate binomial ligand
	try: BL
	except:
		BL = binomial_ligand(nL,KL,L)
	if cond=='HT':
		QE = lambda i,j,k,l: i*ktr /  (k*kET + i*ktr + 2*j*kAET + l*kAET2)
		values = [poisson(Ntr,i)*binomial(j-1,nA-1,KA,A)*BL[k]*binomial(l,nA2,KA2,A2)*QE(i,j,k,l) for i in range(z) for j in range(1,nA+1) for k in range(2) for l in range(nA2+1)]
	elif cond=='HT2':
		QE = lambda i,j,k,l: i*ktr /  (k*kET + i*ktr + 2*j*kAET + l*kAET2)
		values = [poisson(Ntr,i)*binomial(j,nA,KA,A)*BL[k]*binomial(l-1,nA2-1,KA2,A2)*QE(i,j,k,l) for i in range(z) for j in range(nA+1) for k in range(2) for l in range(1,nA2+1)]
	elif cond=='BET':
		QE = lambda i,j,k: i*ktr /  (kET + i*ktr + 2*j*kAET + k*kAET2)
		values = [poisson(Ntr,i)*binomial(j,nA,KA,A)*binomial(k,nA2,KA2,A2)*QE(i,j,k) for i in range(z) for j in range(nA+1) for k in range(nA2+1)]
	elif cond==None:
		QE = lambda i,j,k,l: i*ktr /  (k*kET + i*ktr + 2*j*kAET + l*kAET2 + k0)
		values = [poisson(Ntr,i)*binomial(j,nA,KA,A)*BL[k]*binomial(l,nA2,KA2,A2)*QE(i,j,k,l) for i in range(z) for j in range(nA+1) for k in range(2) for l in range(nA2+1)]
	EnQETr = np.sum(np.array(values))
	return EnQETr

def get_rates():
	### Use this block to compute all of the quantum efficiencies you will need to compute all rates
	QEETu = QEET(); QEETcHT = QEET(cond='HT'); QEETcHT2 = QEET(cond='HT2')
	#print('start')
	#print(QEETu)
	#print(QEETcHT)
	QEHTu = QEHT(); QEHTcET = QEHT(cond='ET'); QEHTcAET2 = QEHT(cond='AET2'); #QEHTcTr = QEHT(cond='Tr')
	#print(QEHTu)
	#print(QEHTcET)
	#print('end')
	QEHT2u = QEHT2(); QEHT2cET = QEHT2(cond='ET'); QEHT2cAET = QEHT2(cond='AET'); #QEHT2cTr = QEHT2(cond='Tr')
	QEBETu = QEBET(); QEBETcAET = QEBET(cond='AET'); QEBETcAET2 = QEBET(cond='AET2'); #QEBETcTr = QEBET(cond='Tr') 
	QEAETu = QEAET(); QEAETcHT2 = QEAET(cond='HT2'); QEAETcBET = QEAET(cond='BET')
	#print(QEAETu)
	QEAET2u = QEAET2(); QEAET2cHT = QEAET2(cond='HT'); QEAET2cBET = QEAET2(cond='BET')
	#QETru = QETr(); QETrcHT = QETr(cond='HT'); QETrcHT2=QETr(cond='HT2'); QETrcBET = QETr(cond='BET')

	#print('QEBET\t%.3E' % QEBETu)
	#print('QEAET\t%.3E' % QEAETu)
	#print('QEAET|BET\t%.3E' % QEAETcBET)
	#print('QEBET|AET\t%.3E' % QEBETcAET)
	#print('QETr\t%.3E' % QETru)
	#print('QEBET|Tr\t%.3E' % QEBETcTr)
	#print('QETr|BET\t%.3E' % QETrcBET)
	
	def EXT():
		return (PWR / (6.022e23 * (6.62607e-34*2.9979e8/wl) * L * V)) * (1 - 10**(-1*eps*pl*L))
	
	def CAT1():
		# This is the probability that the enzyme is reduced by the QD and the donor is oxidized
		P = QEETu*QEHTu + QEETu*QEHTcET + QEHTu*QEETcHT
		return [P,EXT()*P]

	def CAT2():
		# This is the probability that the enzyme is reduced by the QD and the second donor is oxidized 
		P = QEETu*QEHT2u + QEETu*QEHT2cET + QEHT2u*QEETcHT2
		return [P,EXT()*P]
	
	def OX1():
		# This is the probability that the enzyme is oxidized by the QD and the acceptor is reduced
		#term1 = QEBETu*QEAETu; term2 = QEBETu*QEAETcBET; term3 = QEAETu*QEBETcAET
		#print('\noxidation probabilities')
		#print('%.5f\t%.5f\t%.5f\n' % (term1,term2,term3))
		P = QEBETu*QEAETu + QEBETu*QEAETcBET + QEAETu*QEBETcAET 
		return [P,EXT()*P]

	def OX2():
		# This is the probability that the enzyme is oxidized by the QD and the second acceptor is reduced
		P = QEBETu*QEAET2u + QEBETu*QEAET2cBET + QEAET2u*QEBETcAET2
		return [P,EXT()*P]

	def HS3():
		# This is the probability that the donor is oxidized and the second acceptor is reduced
		P = QEHTu*QEAET2u + QEHTu*QEAET2cHT + QEAET2u*QEHTcAET2
		return [P,EXT()*P]

	def HS4():
		# This is the probability that the second donor is oxidized and the acceptor is reduced
		P = QEAETu*QEHT2u + QEAETu*QEHT2cAET + QEHT2u*QEAETcHT2
		return [P,EXT()*P]
	
	def AET():
		# This is the probability that the acceptor is reduced by the QD
		P = QEBETu*QEAETu + QEBETu*QEAETcBET + QEAETu*QEBETcAET + QEAETu*QEHT2u + QEAETu*QEHT2cAET + QEHT2u*QEAETcHT2
		return [P,EXT()*P]

	def AET2():
		# This is the probability that the second acceptor is reduced by the QD
		P = QEAET2u*QEHTu + QEAET2u*QEHTcAET2 + QEHTu*QEAET2cHT + QEAET2u*QEBETu + QEAET2u*QEBETcAET2 + QEBETu*QEAET2cBET 
		return [P,EXT()*P]
	
	def TR():
		# This is the probability that the electron is trapped
		P = QETru*QEHTu + QETru*QEHTcTr + QEHTu*QETrcHT + QETru*QEBETu + QETru*QEBETcTr + QEBETu*QETrcBET + QETru*QEHT2u + QETru*QEHT2cTr + QEHT2u*QETrcHT2
		return [P,EXT()*P]
	
	def HS():
		# This is the probability that the donor is oxidized
		P = QEHTu*QEETu + QEHTu*QEETcHT + QEETu*QEHTcET + QEHTu*QETru + QEHTu*QETrcHT + QETru*QEHTcTr + QEHTu*QEAET2u + QEHTu*QEAET2cHT + QEAET2u*QEHTcAET2
		return [P,EXT()*P]

	def HS2():
		# This is the probability that the second donor is oxidized
		P = QEHT2u*QEAETu + QEHT2u*QEAETcHT2 + QEAETu*QEHT2cAET + QEHT2u*QEETu + QEHT2u*QEETcHT2 + QEETu*QEHT2cET + QEHT2u*QETru + QEHT2u*QETrcHT2 + QETru*QEHT2cTr
		return [P,EXT()*P]
	#print(CAT1())
		
	return list(np.array([CAT1(), CAT2(), OX1(), OX2(), HS3(), HS4()]).transpose())

def run_rates(param_file,param_spec=None):
	get_params(param_file,param_spec)
	rates = get_rates()
	return [rates,[kET, kHT, kHT2, ktr, kAET, kAET2, kBET, k0, KA, KD, KA2, KD2, KL, nA, nD, nA2, nD2, nL, Ntr, L, D, A, D2, A2, PWR, wl, V, pl, eps]]
