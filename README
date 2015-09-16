### README of eda-suite on github ###

### Overview ###
1. What is EDA suite
2. How can I use it

### Contents ####
##1. What is EDA suite##
EDA suite is a tool I used during my time at the University of Mainz, Germany.
It is meant as a tool to test differnent probabilistic models in the 
context of estimation of distribution algorithms (EDA) (see [1]).

It comprises
 - a main EDA file (eda.m) with the basic loop
 - a set of benchmark problems (fitnessXXX.m)
 - a number of probabilistic models, currently:
    * a Restricted Boltzmann Machine (RBM)
    * the Bayesian Optimization Algorithm (BOA)
    * Population-based incremental learning  (PBIL)
    * Denoising Autoencoders (DAE)
    * Deep Boltzmann Machine (DBM) * with code form R. Salakhutdinov
 - some basic selection functions
 
##2. How can I use it?##
 * I used GNU/Octave for all experiments. 
 * Start Octave, go to the ./src/ directory
 * adjust the EDA parameters (including the choice of problem)
    in the file set_eda_params.m
 * adjust the selection parameters in the file set_selection_params.m
 * choose a model which you want to use, and change the appropriate 
   .m-file, e.g. set_rbm_params.m
 * load all settings into the ocatave-workspace, and start the EDA,
   as follows:
   
 octave> set_eda_params;
 octave> set_selection_params;
 octave> set_ae_params;  %in case you want to use the denoising autoencoder
 octave> [mean_fitness, best_fitness, found_in_iteration,cputime_until_best]=eda(eda_params,selection_params,model_params)
