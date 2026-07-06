This code will integrate the model and plot the results of the fit to the experimental data collected with NC1:MoFe protein biohybrids, as shown in Figure 2. As written, the code will plot the accumulation of H2 and ammonia. To utilize the model to simulate the HQ dependence, pN2 dependence, or pH2 dependence, change the parameters at the top of run_kinetics.py.

For example, to simulate the pN2 dependence, decrease the total time (ttot) to 7 minutes and alter the N2 partial pressure (pN2) to the desired value in units of atm. 

To run the simulations, run

python run_kinetics.py
