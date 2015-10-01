README of eda-suite on github

# Overview
1. What is EDA suite
2. How can I use it
3. Publications and configurations
4. Sources and further reading

# Contents
##1. What is EDA suite##
EDA suite is a tool I use for my research at the University of Mainz, Germany.
It is meant as a tool to test different probabilistic models in the 
context of estimation of distribution algorithms (EDA) (see [1]).

It comprises
* a main EDA file (eda.m) with the basic loop
* a set of benchmark problems (fitnessXXX.m)
* a number of probabilistic models, currently:
 * a Restricted Boltzmann Machine (RBM) [2]
 * the Bayesian Optimization Algorithm (BOA) [3]
 * Population-based incremental learning  (PBIL) [4]
 * Denoising Autoencoders (DAE) [5]
 * Deep Boltzmann Machine (DBM) [6]  * with code form R. Salakhutdinov
 * Generative Adversarial Network (GAN) [7]
* some basic selection functions
 
##2. How can I use it?##
 * I used GNU/Octave for all experiments, mostly v3.2.4 and 3.8.1
   (you might need to install some extra packages, especially for BOA) 
 * Start Octave, go to the ./src/ directory
 * adjust the EDA parameters (including the choice of problem)
    in the file set_eda_params.m
 * adjust the selection parameters in the file set_selection_params.m
 * choose a model which you want to use, and change the appropriate 
   .m-file, e.g. set_rbm_params.m
 * load all settings into the ocatave-workspace, and start the EDA,
   as follows:

```
set_eda_params;
set_selection_params;
set_ae_params;  %in case you want to use the denoising autoencoder
[mean_fitness, best_fitness, found_in_iteration,cputime_until_best]=eda(eda_params,selection_params,model_params)
```
##3. Publications and configurations
The following publications include experiments performed with this code (see individual comments).

The corresponding configuration files can be found in the directory '.configs-papers/'

Title|Comment|Config subfolder|URL
--- | --- | --- | --- 
Denoising Autoencoders for Fast Combinatorial Black Box Optimization (GECCO'15 + submission on arXiv)|slightly different autoencoder code was used|`gecco15-dae-eda-fast/`|[ACM](http://dl.acm.org/citation.cfm?doid=2739482.2764691) and [arXiv](http://arxiv.org/abs/1503.01954)
Deep Boltzmann Machines in Estimation of Distribution Algorithms for Combinatorial Optimization||`dbm-eda/`|[arXiv](http://arxiv.org/abs/1509.06535)
Generative Adversarial Networks in Estimation of Distribution Algorithms for Combinatorial Optimizatoin||`gan-eda/`|[arXiv](http://arxiv.org/abs/1509.09235)

##4. Sources and further reading
[1] P. Larranaga and J.A. Lozano. Estimation of Distribution Algorithms: A New Tool for Evolutionary Computation, Kluwer Academic Pub, 2002

[2] Hinton, G. E. A Practical Guide to Training Restricted Boltzmann Machines, 2006 Techical Report 

[3] Pelikan, M., Hierarchical Bayesian Optimization Algorithm, Springer 2005

[4] Baluja, S. Population-Based Incremental Learning: A Method for Integrating Genetic Search Based Function Optimization and Competitive Learning Carnegie Mellon University, 1994

[5] Vincent, P.; Larochelle, H.; Bengio, Y. & Manzagol, P.-A. Extracting and Composing Robust Features with Denoising Autoencoders Proceedings of the 25th international conference on Machine learning, 2008, 1096-1103

[6] Salakhutdinov, R. & Hinton, G. E. A Better Way to Pretrain Deep Boltzmann Machines Advances in Neural Information Processing Systems, 2012, 2447-2455

[7] Goodfellow, Ian and Pouget-Abadie, Jean and Mirza, Mehdi and Xu, Bing and Warde-Farley, David and Ozair, Sherjil and Courville, Aaron and Bengio, Yoshua; Generative Adversarial Nets; In: Advances in Neural Information Processing Systems (NIPS 2014), 2672-2680
