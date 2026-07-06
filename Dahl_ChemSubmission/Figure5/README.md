This code corresponds to the fit in Figure 5 of our submitted mansucript. The file fit_parameters.txt contains the parameters output from the fit to the experimentally determined En-state populations. The parameters are formatted in the following order:

kET(Mox-->E0)

kET(E0-->E1)

kET(E1-->E2)

kET(E2-->E3)

kET(E3-->E4(4H))

kET(N)



kHT(HQ)

kET(BQ)


kBET(E0-->Mox)

kBET(E1-->E0)

kBET(E2-->E1)

kBET(E3-->E2)

kBET(E4(4H)-->E3)

kBET(N)


The remainder of the model parameters are specified in the file photochem_params.txt


To integrate the model, and plot the results, run,

python run_kinetics.py
