% EDA suite - main file (eda.m)
% Copyright (C) <2010-2015> Malte Probst
%
% This file is part of the EDA suite.
%    The EDA suite is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%    The EDA suite is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with the EDA suite.  If not, see <http://www.gnu.org/licenses/>.
function [mean_fitness, best_fitness, found_in_iteration, cputime_until_best, numEvalsUntilBest, evalsLookedUp] = eda(...
    eda_params, ...         %parameters for EDA 
    selection_params, ...   %parameters for selection mechanism
    model_params   ...     %parameters for probabilistic model
    )
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% initialization of variables
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    global savePath;
    fitnessEvaluated=[];
    numEvalsLookedUp=[];
    more off;
    if eda_params.loglevel>0
        eda_params
        selection_params
        model_params
    end
    unlink ("/tmp/stopOctave");%workaround for "octave not stopping" bug
    times=zeros(1,eda_params.max_generations);
    
    %use XOR bit flip mask to avoid model-related biases (e.g. towards ones) influcencing the EDA
    if(eda_params.bitFlipMask)
        bitFlipMask=rand(1,eda_params.problem_size)>0.5;
    else
        bitFlipMask=0;
    end
    % propagate some information to other params
    model_params.loglevel=eda_params.loglevel;
    model_params.eda_params=eda_params;
    
    %initialize model (using the configured function)
    model=model_params.finitializeModel(model_params,true);

    %initialize population with random values
    population=rand(eda_params.popsize,eda_params.problem_size)>0.5;
    %calcuate initial fitness
    fitness=eda_params.ffunction(xor(population,repmat(bitFlipMask,size(population,1),1)),eda_params.fitness_param1);
    %keep track of number of fitness evals
    numEvals=size(population,1);
    %apply pre-EDA local search, if configured
    if eda_params.localSearch
       [population,fitness,additionalEvals]=localSearch(population,fitness,eda_params,bitFlipMask);
       numEvals+=additionalEvals;
    end
    %if configured, keep track of actual number of fitness evals
    %note: as all benchmark functions are quick to evaluate, we just
    %keep track of the number of double evaluations, and evaluate nontheless
    if eda_params.useFitnessTable
            fitnessEvaluated=unique(bin2dec(num2str(population)));
    end

    %fields for stats
    fitness_max=[];
    fitness_mean=[];
    found_in_iteration=1;
    generationsWithoutImprovement=0;
    best_fitness=0;

    %%diversity measures (some experiment for a DAE paper)
    PDISelected=[];
    PDISampled=[];
    InertiaSelected=[];
    InertiaSampled=[];
    identityFunctionStrength=[];
    numBasinsOfAttraction=[];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% main loop
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    stopnow=0
    start=cputime;
    
    for g=1:eda_params.max_generations
        if eda_params.loglevel>0
            fprintf("Generation %i\n",g);
        end
        %%% Octave bug: cannot reliably stop exectution on CTRL-C when using multithreaded BLAS library
        %workaround: check existence of file /tmp/stopOctave
        if exist("/tmp/stopOctave","file")==2
            break
            fprintf("Aborting....\n");
        end
        
        %propagate some information to the model
        model_params.generation=g;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%  make selection
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %apply selection function
        winners=selection_params.sfunction(eda_params.popsize, ...
            eda_params.popsize/2,fitness);
        kept=winners;
        %sort kept population by fitness (easier to handle later on)
        [order,index]=sort(fitness(kept),'descend');
        kept=kept(index);
        
        %%optional/if configured collect stats on populationDiversity
        if eda_params.computeDiversityMeasures
            PDISelected=[PDISelected,populationDiversityIndex(population(kept,:))];
            InertiaSelected=[InertiaSelected,populationInertia(population(kept,:))];
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%% decide if to break or not
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%% Stop, if optimum was found 
        offset=1000000;
        if eda_params.stopOnOptimum && round(offset*best_fitness)/offset>=round(offset*eda_params.fitness_max)/offset
            if eda_params.loglevel>0
                fprintf("Best fitness found! --> Breaking\n");
            end
            break
        end
        
        %%%%% Stop, if maximum number of generations without improvements is reached
        if generationsWithoutImprovement>=eda_params.stopAfterNGenerationsWithoutImprovement
            if eda_params.loglevel>0
                fprintf("Maximum number of generations without improvement reached! --> Breaking\n");
            end
            break
        end
        
        
        %%%%% Stop, if population has converged
        numberOfUniqueIndividuums=size(unique(population(kept,:),'rows'),1);
        if generationsWithoutImprovement>=1 ...
            && size(unique(population(kept,:),'rows'),1)==1% && strcmp(model_params.name,"PBIL")==0;
            if eda_params.loglevel>0
                fprintf("Converged! --> Breaking\n");
            end
            break
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%% destroy or keep old model
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %unused
        if g==1
            fromScratch=true;
        else
            fromScratch=false;
        end
        
        model=model_params.finitializeModel(model_params,fromScratch,model);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%% train model
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %usually, the training function only needs the hyper parameters and the population,
        %but some experimental code uses the other stuff as well
        model=model_params.ftrainModel(model_params,population(kept,:),model,false,fitness(kept));
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%% sample new candidate solutions
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %sample candidates
        candidate_solutions=model_params.fsampleModel(model_params,model,eda_params.candidate_count,population(kept,:));
        %evaluate fitness of candidates
        fitness_candidates=eda_params.ffunction(xor(candidate_solutions,repmat(bitFlipMask,size(candidate_solutions,1),1)),eda_params.fitness_param1);
        
        numEvals=[numEvals;eda_params.candidate_count];


        
        %%optional/if configured collect stats on populationDiversity
        if eda_params.computeDiversityMeasures
            %calculate population diversity index as in Smit et al 2011, Population diversity index
            PDISampled=[PDISampled,populationDiversityIndex(candidate_solutions)];
            %calulate inertia, as in Morrison and De Jong 2002, Measurement of population diversity
            InertiaSampled=[InertiaSampled,populationInertia(candidate_solutions)];
            %run local search to determine number of optima (can take a looooong time)
            [candidatesOptimized,fitnessCandidatesOptimized,additionalEvals]=localSearch(candidate_solutions,fitness_candidates,eda_params,bitFlipMask);
            %calculate number of basins of attraction (number of distict local optima a simple hillclimber can find)
            candUnique=unique(candidatesOptimized,"rows");
            numBasinsOfAttraction=[numBasinsOfAttraction,size(candUnique,1)];
        end
          
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%  include candidate solutions in population
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %sort kept population by fitness (easier to handle later on)
        [garbage,index]=sort(fitness(kept),'descend');
        population=[population(kept,:)(index,:);candidate_solutions];
        fitness=[fitness(kept)(index,:);fitness_candidates];

        %%%%  ---- end of meaningful code, rest is stats & graphs ----
        %remember time required until this generation
        times(g)=cputime-start;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%  Update stats, ...
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %keep track of stuff
        generationsWithoutImprovement+=1;
        best_fitness=max(fitness);
        if best_fitness>max(fitness_max)
            found_in_iteration=g;
            generationsWithoutImprovement=0;
        end
        fitness_max=[fitness_max,best_fitness];
        mean_fitness=mean(fitness);
        fitness_mean=[fitness_mean,mean_fitness];
                
        %special for autoencoder, calculate how strong identity function of AE has been learned
        if eda_params.computeDiversityMeasures && strcmp(model_params.name,"AE")

            tmpStrength=getSparsityDirect(model,model_params,numExamples=100);
            identityFunctionStrength=[identityFunctionStrength,tmpStrength];
        end
        
        %if configured, save number of fitness evals that could have been evaluated using a cache
        if eda_params.useFitnessTable
            tmpIndexLookedUp=bin2dec(num2str(candidate_solutions));
            numEvalsLookedUp=[numEvalsLookedUp,sum(ismember(tmpIndexLookedUp,fitnessEvaluated))];
            fitnessEvaluated=unique([fitnessEvaluated;tmpIndexLookedUp]);
            clear tmpIndexLookedUp;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%  Print Graphs, stats, ...
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        if eda_params.loglevel>0 && mod(g,eda_params.show_stats_each_n_generations)==0
            fprintf(" ....best fitness %f\n",max(fitness));
        end
        %plot stuff to visualize progress etc.
        if eda_params.loglevel>1 && mod(g,eda_params.show_stats_each_n_generations)==0
            %%%%%%%%%%%%%%%%%%%
            figure(1040,'name','Population')
            tmp=xor([population(1:50,:);population(end-49:end,:)],bitFlipMask);
            img=zeros(size(tmp,1)+2,size(tmp,2)+2);
            img(2:end-1,2:end-1)=tmp;
            imshow(img)
            title("Population (50 best selected, 50 candidates)");
            %
            %%%%%%%%%%%%%%%%%%%
            figure(1042,'name','Fitness values')
            clf
            hold on
            plot([1:eda_params.popsize],fitness);
            plot(1:eda_params.popsize,repmat(eda_params.fitness_max,eda_params.popsize,1),'g')
            hold off
            %%%%%%%%%%%%%%%%%%%
            %figure(43,'name','Fitness histogram','Position',[300,400,300,120])
            %hist(fitness,50);
            %title("Fitness Histogram");
            %%%%%%%%%%%%%%%%%%%
            figure(1044,'name','Fitness / Generation')%,'Position',[300,400,400,300])
            clf
            hold on
            axis([0 eda_params.max_generations 0 eda_params.fitness_max+1], "autoy")
            plot(1:g,fitness_max(:),'r')
            plot(1:g,fitness_mean(:),'.')
            plot(1:g,repmat(eda_params.fitness_max,g,1),'g')
            hold off     
            %%%%%%%%%%%%%%%%%%%
            figure(1050,'name','Mean kept population')%,'Position',[300,400,300,120])
            bar([1:eda_params.problem_size],mean((xor(population(kept,:),repmat(bitFlipMask,size(population(kept,:),1),1))-0.5)))
            title("mean value of kept population");
            
            if eda_params.computeDiversityMeasures
                figure(10447)
                clf
                hold on;
                plot(1:g,PDISelected,"r")
                plot(1:g,PDISampled,"g")
                hold off;
                title("Population Diversity Index, green: selected red: sampled")
                
                figure(10448)
                clf
                hold on;
                plot(1:g,InertiaSelected,"r")
                plot(1:g,InertiaSampled,"g")
                hold off;
                title("Population Inertia, green: selected red: sampled")
                
                figure(10449)
                clf
                hold on;
                plot(1:g,numBasinsOfAttraction,"r")
                hold off;
                title("Number of Basins of Attraction (local search)")
                
                if strcmp(model_params.name,"AE")
                    %show identityFunctionStrength of AE
                    figure(10779);
                    plot(1:g,identityFunctionStrength,'r');
                    title("Sparsity / Generations")
                end
            end
            
            if eda_params.useFitnessTable
                figure(10549)
                clf
                hold on;
                plot(1:g,numEvalsLookedUp/eda_params.candidate_count,"r")
                hold off;
                title("Percentage of of Looked Up Evaluations")
            end
            
            refresh();
            if eda_params.stopAfterPlot
                input("---Press Enter to continue---");
            end
        end
        
        
        
    end
    
    %keep track of cpu times/evals
    cputime_until_best=times(found_in_iteration);
    numEvalsUntilBest=sum(numEvals(1:found_in_iteration+1));
    fprintf("Evals: %i\n",numEvalsUntilBest)

    if eda_params.useFitnessTable
        evalsLookedUp=sum(numEvalsLookedUp(1:found_in_iteration));
        ratioEvalsLookedUp=evalsLookedUp/numEvalsUntilBest;
        fprintf("Evals Looked Up: %i (%i percent) \n",floor(evalsLookedUp),ratioEvalsLookedUp*100)
    else
        ratioEvalsLookedUp=0;
        evalsLookedUp=0;
    end

    %if configured, save the calculated information on diversity/basins/fitness/... to global save path
    if eda_params.computeDiversityMeasures
        global savePath
        %save diversity measures to temp directory
        if ! exist("savePath","var") || strcmp(savePath,"") || length(savePath)<1
            savePath="/tmp/";
        end
        filename=strcat(savePath,"/DM-",model_params.name,"-",eda_params.problem_name,"-size",mat2str(eda_params.problem_size),"-pop",mat2str(eda_params.popsize),"-run-",mat2str(round(rand*10000)),mat2str(round(rand*10000)),".m")
        diversityResults.PDISelected=PDISelected(1:found_in_iteration);
        diversityResults.PDISampled=PDISampled(1:found_in_iteration);
        diversityResults.InertiaSelected=InertiaSelected(1:found_in_iteration);
        diversityResults.InertiaSampled=InertiaSampled(1:found_in_iteration);
        save(filename,"diversityResults")
        
        filename=strcat(savePath,"/BASINS-",model_params.name,"-",eda_params.problem_name,"-size",mat2str(eda_params.problem_size),"-pop",mat2str(eda_params.popsize),"-run-",mat2str(round(rand*10000)),mat2str(round(rand*10000)),".m")
        basinsResults=numBasinsOfAttraction(1:found_in_iteration);
        save(filename,"basinsResults")

        filename=strcat(savePath,"/FITNESS-",model_params.name,"-",eda_params.problem_name,"-size",mat2str(eda_params.problem_size),"-pop",mat2str(eda_params.popsize),"-run-",mat2str(round(rand*10000)),mat2str(round(rand*10000)),".m")
        fitnessResults.fitness_mean=fitness_mean(1:found_in_iteration);
        fitnessResults.fitness_max=fitness_max(1:found_in_iteration);
        save(filename,"fitnessResults")
        
        if eda_params.useFitnessTable
                filename=strcat(savePath,"/EFFECTEVAL-",model_params.name,"-",eda_params.problem_name,"-size",mat2str(eda_params.problem_size),"-pop",mat2str(eda_params.popsize),"-run-",mat2str(round(rand*10000)),mat2str(round(rand*10000)),".m")
                effectiveEvalResults=numEvalsLookedUp(1:found_in_iteration);
                save(filename,"effectiveEvalResults")
        end
        
        if strcmp(model_params.name,"AE")
            %save identityFunctionStrength measures to temp directory
            filename=strcat(savePath,"/SM-",model_params.name,"-",eda_params.problem_name,"-size",mat2str(eda_params.problem_size),"-pop",mat2str(eda_params.popsize),"-run-",mat2str(round(rand*10000)),mat2str(round(rand*10000)),".m")
            sparsityResults.sparsity=identityFunctionStrength(1:found_in_iteration); %used to something else :)
            save(filename,"sparsityResults")
        end
    end
end
