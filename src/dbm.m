% EDA suite - boa.m
% Copyright (C) <2010-2015> Malte Probst
% This file is part of the EDA suite.
%    The EDA suite is free software under the terms of the GNU General
%    Public License, see eda.m for full licence text, or <http://www.gnu.org/licenses/>.


%  Deep Boltzmann Machine. A main part of this code is derived from 
%  Rulsan Salakutdinovs Code at http://www.cs.toronto.edu/~rsalakhu/DBM.html
% 

%initialization of model parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function model=initializeDBM(model_params,from_scratch=true)
    model.foo="bar";
    model.myrun=mat2str(floor(rand*1000000) );
    numdim=model_params.problem_size;
    numhid=model_params.numhid;
    numpen=model_params.numpen;
    
    
    vishid     = 0.001*randn(numdim, numhid);
    hidpen     = 0.001*randn(numhid, numpen);
    visbiases  = zeros(1,numdim);   
    hidbiases  = zeros(1,numhid);
    penbiases  = zeros(1,numpen);
    
    save (strcat("./dbm/fullmnist_dbm",model.myrun), "hidpen", "penbiases", "vishid", "hidbiases", "visbiases");

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Trains an DBM on the given training data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function model=trainDBM(model_params,training_data,model,validation_data=false,fitness=false)
    source("./dbm/calcSmoothedErr.m");
    source("./dbm/checkTerminateTraining.m");  
    %%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%% convert params  to original code's format#####
    %%%%%%%%%%%%%%%%%%%%%%%%%
    batchdata=zeros(model_params.batchsize,model_params.problem_size,floor(size(training_data,1)/model_params.batchsize));
    [numcases numdims numbatches]=size(batchdata);
    batchsize=numcases;
    for b=1:numbatches
        batchdata(:,:,b) = training_data(1+(b-1)*batchsize:b*batchsize, :);
        %batchtargets(:,:,b) = targets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
    end;
    
    cd "dbm"
    
    ####### from here: Code by Salakhutdinov, cleared of all elements concerned with supervised learning
    % Version 1.000
    %
    % Code provided by Ruslan Salakhutdinov
    %
    % Permission is granted for anyone to copy, use, modify, or distribute this
    % program and accompanying programs and documents for any purpose, provided
    % this copyright notice is retained and prominently displayed, along with
    % a note saying that the original programs are available from our
    % web page.
    % The programs and documents are distributed without any warranty, express or
    % implied.  As the programs were written for research purposes only, they have
    % not been tested to the degree that would be advisable in any important
    % application.  All use of these programs is entirely at the user's own risk.

    %Note, Malte Probst, 2015: some small changes in the code (removed everything
    % to do with classes, to make this entirely unsupervised)

    %%%%%% Training 1st layer %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %numhid=500; maxepoch=100;
    
    numhid=model_params.numhid;
    maxepoch=model_params.n_epochs_rbm1;
    
    if model_params.loglevel>1
        fprintf(1,'Pretraining Layer 1 with RBM: %d-%d \n',numdims,numhid);
    end
    restart=1;
    rbm

    %%%%%% Training 2st layer %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %close all 
    numpen = model_params.numpen; 
    maxepoch=model_params.n_epochs_rbm2;
    if model_params.loglevel>1
        fprintf(1,'\nPretraining Layer 2 with RBM: %d-%d \n',numhid,numpen);
    end
    restart=1;
    %makebatches;
    for b=1:numbatches
        batchdata(:,:,b) = training_data(1+(b-1)*batchsize:b*batchsize, :);
        %batchtargets(:,:,b) = targets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
    end; 
    rbm_l2


    %%%%%% Training two-layer Boltzmann machine %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %close all 
    
    numhid = model_params.numhid; 
    numpen = model_params.numpen; 
    maxepoch=model_params.n_epochs; %To get results in the paper I used maxepoch=500, which took over 2 days or so. 
      
    if model_params.loglevel>1
        fprintf(1,'Learning a Deep Bolztamnn Machine. \n');
    end
    restart=1;
    %makebatches;
    for b=1:numbatches
        batchdata(:,:,b) = training_data(1+(b-1)*batchsize:b*batchsize, :);
        %batchtargets(:,:,b) = targets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
    end; 
    dbm_mf

    ####### until here: Code by Ruslan Salakhutdinov (slightly modified)
    cd ".."
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% returns n samples from the model, including metadata (such as corresponding hidden states, energies...)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [samples,metadata]=sampleDBM(model_params,model,n_samples,population)
    ####convert params  to original code's format#####
    numcases=size(population,1);
    numdim=model_params.problem_size;
    numhid=model_params.numhid;
    numpen=model_params.numpen;

    % Initializing symmetric weights and biases. 
    file=strcat("dbm/fullmnist_dbm",model.myrun);
    load(file)
    bias_vis  = repmat(visbiases,numcases,1);   
    bias_hid  = repmat(hidbiases,numcases,1);
    bias_pen  = repmat(penbiases,numcases,1);
    
    %#### initialize fields
    neghidprobs=population;
    %neghidprobs=rand(numcases,numhid)>0.5;
   
    %#### original code ### (temperature added)
    temperature=1;
    for iter=1:25
        %temperature=max(1,2-(iter/200));

        neghidstates = neghidprobs > rand(numcases,numhid);
      
        negpenprobs = 1./(1 + exp((-neghidstates*hidpen - bias_pen)/temperature));
        negpenstates = negpenprobs > rand(numcases,numpen);

        negdataprobs = 1./(1 + exp((-neghidstates*vishid' - bias_vis)/temperature));
        negdata = negdataprobs > rand(numcases,numdim);


        totin = negdata*vishid + bias_hid + negpenstates*hidpen';
        neghidprobs = 1./(1 + exp(-totin/temperature));

    end 
    samples=negdata;
    metadata.hiddens=neghidstates;
    metadata.visprobs=negdataprobs;
    
    %%delete temporary files
    delete(strcat("./dbm/fullmnist_dbm",model.myrun))
end
