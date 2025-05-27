The code in this repository was used to model the kinetics of photochemical nitrogen reduction. Each directory represents a manuscript.


Dahl_CPRS_2025

The directory titled Dahl_CRPS_2025 contains the code used to fit the kinetic models presented in the manuscript titled, "Pre-steady-state kinetics of nanocrystal:molybdenum nitrogenase biohybrids reveals hole-scavenging efficiency is critical to achieving N2 reduction" by Peter J. Dahl, Lauren M. Pellows, Zhi-Yong Yang, Lance C. Seefeldt, John W. Peters, Gordana Dukovic, David W. Mulder, and Paul W. King.

In the directory called "Figure4", you will find the code used to generate the fits presented in Figure 4 of the mansucript. To run the code, execute the python script titled fitting_simulation.py,

python fitting_simultion.py -expdat estate_pop_20mMDT.txt -expfmt txt -plot True

The maximum number of iterations have been set to 500 such that the code can be run on a feasible timescale. Fitting for the paper was performed using a maximum number of iterations set to 5,000,000. This enabled the optimization procedure to achieve convergence, although it took upwards of 8 hours on a high performance computer.

In the directory called "Figure5", you will find the code used to fit the data presented in Figure 5 of the manuscript. To run the code, execute the python script titled int_cpld.py

python int_cpld.py

Again, the maximum number of iterations has been decreased. To achieve convergence, the code was run with the maximum number of iterations set to 1,000,000.

Both directories contain a file called Master.py. This file contains the master equation expressed as a matrix of rate constants. This matrix is read in to the other scripts which then perform the integration.
