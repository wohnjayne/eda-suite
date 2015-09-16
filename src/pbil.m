% EDA suite - pbil.m
% Copyright (C) <2010-2015> Malte Probst
% This file is part of the EDA suite.
%    The EDA suite is free software under the terms of the GNU General
%    Public License, see eda.m for full licence text, or <http://www.gnu.org/licenses/>.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%initialization of model parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function model=initializePBIL(model_params,from_scratch=true,model=false,keep_old_model=0)
    if from_scratch
        clear model;
        model.probs=zeros(1,model_params.problem_size)+0.5;
    end
end

function model=trainPBIL(model_params,training_data,model,validation_data=false,fitness=false)

    %%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%% update model
    %%%%%%%%%%%%%%%%%%%%%%%%%
    model.probs=model.probs+model_params.alpha*(training_data(1,:)-model.probs);
    model.probs=max(0,model.probs);
    model.probs=min(1,model.probs);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%  Print Graphs, stats, ...
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if model_params.loglevel>1 && model_params.show_stats
        
        %%%%%%%%%%%%%%%%%%%
        figure(21300,'name','PBIL Probs Histogram','Position',[300,400,300,120]);
        hist(model.probs);
        title("Probs Histogram");
        %%%%%%%%%%%%%%%%%%%
        figure(21302,'name','PBIL Probs','Position',[300,400,300,120])
        bar([1:model_params.problem_size],model.probs)
        title("Probs");
        
        refresh();
        if model_params.stopAfterPlot
            input("---Press Enter to continue---");
        end
    end

    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% returns n samples from the model, including metadata (such as corresponding hidden states, energies...)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [samples,metadata]=samplePBIL(model_params,model,n_samples,population)
    %initialize randomly
    metadata=false;
    samples=repmat(model.probs,n_samples,1)>rand(n_samples,model_params.problem_size);
end
