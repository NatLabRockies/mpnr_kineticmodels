import numpy as np

class ArMEq:
	def __init__(self,rates):
		[khp1,khp2,khp3,k10,k21,k32,k43] = rates
		self.MEq = np.array([[ -1*k10,       0,          khp1,              0,            0    ],
                                     [    k10,    -1*k21,         0,               khp2,          0    ],
                                     [     0,       k21,     -1*(khp1+k32),         0,           khp3  ],
                                     [     0,        0,          k32,         -1*(khp2+k43),      0    ],
                                     [     0,        0,           0,               k43,        -1*khp3 ]])

class LTMEq:
	def __init__(self,rates):
		[khp2,khp3,khp4,k0,k1,k2,k3,k4,k5,k6,k7,k8,kre,koa,pN2,pH2] = rates
		self.MEq = np.array([[   -1*k1,         0,            khp2,               0,              0,                0,           0,           0,            0,         k0    ],
				     [    k1,         -1*k2,           0,                khp3,            0,                0,           0,           0,            0,         0     ],
				     [     0,           k2,      -1*(k2+khp2),            0,             khp4,              0,           0,           0,            0,         0     ],
				     [     0,           0,             k3,          -1*(k4+khp3),         0,                0,           0,           0,            0,         0     ],
				     [     0,           0,             0,                 k4,     -1*(khp4+kre*pN2),     koa*pH2,        0,           0,            0,         0     ],
				     [     0,           0,             0,                 0,           kre*pN2,      -1*(k5+koa*pH2),    0,           0,            0,         0     ],
				     [     0,           0,             0,                 0,              0,                k5,        -1*k6,         0,            0,         0     ],
				     [     0,           0,             0,                 0,              0,                0,           k6,        -1*k7,          0,         0     ],
				     [     0,           0,             0,                 0,              0,                0,           0,           k7,         -1*k8,       0     ],
				     [     0,           0,             0,                 0,              0,                0,           0,           0,            k8,      -1*k0   ]])


class HarrisMEq:
	def __init__(self,rates):
		[khp2,khp4,k0,k1,k2,k3,kre,koa,N2,H2] = rates
		self.MEq = np.array([[    -1*k1,       khp2,                0,                 0,               k0    ],
				     [      k1,    -1*(k2+khp2),          khp4,                0,                0    ],
				     [      0,          k2,        -1*(khp4+kre*N2),     -1*(koa*H2),            0    ],
				     [      0,          0,               kre*N2,        -1*(koa*H2+k3),          0    ],
				     [      0,          0,                  0,                 k3,             -1*k0  ]])



class MoxMEq:
	def __init__(self,rates):
		[ko0,ko1,kro,khp2,khp3,khp4,k0,k1,k2,k3,k4,k5,k6,k7,k8,kre,koa,pN2,pH2] = rates
                #                           MOX        E0           E1              E2                 E3             E4(4H)          E4(2n2h)       E5            E6           E7          E8
		self.MEq = np.array([[     -1*kro,     ko0,          0,             0,                 0,              0,                0,           0,           0,            0,         0     ],        
				     [      kro,   -1*(k1+ko0),     ko1,           khp2,               0,              0,                0,           0,           0,            0,         k0    ],
		                     [       0,        k1,      -1*(k2+ko1),       ko1,               khp3,            0,                0,           0,           0,            0,         0     ],
		                     [       0,         0,           k2,     -1*(k2+khp2+ko1),        ko1,            khp4,              0,           0,           0,            0,         0     ],
		                     [       0,         0,           0,             k3,        -1*(k4+khp3+ko1),      ko1,               0,           0,           0,            0,         0     ],
		                     [       0,         0,           0,             0,                 k4,  -1*(khp4+kre*pN2+ko1),    koa*pH2,        0,           0,            0,         0     ],
		                     [       0,         0,           0,             0,                 0,           kre*pN2,      -1*(k5+koa*pH2),   ko1,          0,            0,         0     ],
		                     [       0,         0,           0,             0,                 0,              0,                k5,     -1*(k6+ko1),     ko1,           0,         0     ],
		                     [       0,         0,           0,             0,                 0,              0,                0,           k6,    -1*(k7+ko1),       ko1,        0     ],
		                     [       0,         0,           0,             0,                 0,              0,                0,           0,           k7,      -1*(k8+ko1),    0     ],
		                     [       0,         0,           0,             0,                 0,              0,                0,           0,           0,            k8,      -1*k0   ]])




class MoxStep2MEq:
	def __init__(self,rates):
		[ko0,ko1,kro,khp2,khp3,khp4,k0,k1,k2,k3,k4,k4if,k4ir,k5,k6,k7,k8,kre,koa,pN2,pH2] = rates
				#           Mox        E0            E1             E2                 E3             E4J               E4A        E42n2h          E5           E6         E7        E8
		self.MEq = np.array([[    -1*kro,      ko0,          0,             0,                 0,              0,                0,           0,           0,            0,         0,       0     ],
				     [      kro,   -1*(k1+ko0),     ko1,           khp2,               0,              0,                0,           0,           0,            0,         0,       k0    ],
				     [       0,        k1,      -1*(k2+ko1),        0,                khp3,            0,                0,           0,           0,            0,         0,       0     ],
				     [       0,         0,           k2,      -1*(k2+khp2),            0,             khp4,              0,           0,           0,            0,         0,       0     ],
				     [       0,         0,           0,             k3,          -1*(k4+khp3),         0,                0,           0,           0,            0,         0,       0     ],
				     [       0,         0,           0,             0,                 k4,     -1*(khp4+k4if),          k4ir,         0,           0,            0,         0,       0     ],
				     [       0,         0,           0,             0,                 0,            k4if,      -1*(k4ir+kre*pN2),  koa*pH2,       0,            0,         0,       0     ],
				     [       0,         0,           0,             0,                 0,              0,             kre*pN2, -1*(k5+koa*pH2),    0,            0,         0,       0     ],
				     [       0,         0,           0,             0,                 0,              0,                0,           k5,        -1*k6,          0,         0,       0     ],
				     [       0,         0,           0,             0,                 0,              0,                0,           0,           k6,         -1*k7,       0,       0     ],
				     [       0,         0,           0,             0,                 0,              0,                0,           0,           0,           k7,       -1*k8,     0     ],
				     [       0,         0,           0,             0,                 0,              0,                0,           0,           0,           0,          k8,    -1*k0   ]])

