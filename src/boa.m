% EDA suite - boa.m
% Copyright (C) <2010-2015> Malte Probst
% This file is part of the EDA suite.
%    The EDA suite is free software under the terms of the GNU General
%    Public License, see eda.m for full licence text, or <http://www.gnu.org/licenses/>.
source bic.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Initialize BOA model 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function model=initializeBOA(model_params,from_scratch=true,model=false,keep_old_model=0)
    clear model
    model.structure_matrix=zeros(model_params.problem_size,model_params.problem_size);
    model.inheritance_matrix=zeros(model_params.problem_size,model_params.problem_size);
    model.cpts={};
    for i=1:model_params.problem_size
        model.cpts{i}=[0.5];
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Runs BOA on the given training data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function model=trainBOA(model_params,training_data,model,validation_data=false,fitness=false)
    %%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%% prepare stuff
    %%%%%%%%%%%%%%%%%%%%%%%%%
    %initialize node scores
    node_scores=zeros(model_params.problem_size,1);
    for i=1:model_params.problem_size
        parents=find((model.structure_matrix)(i,:));
        node_scores(i)=node_score(i,parents,training_data);
        %fprintf(" Initial score for node %i is %f\n",i,node_scores(i));
    end
    min_gain=0;
    if model_params.loglevel>1
        fprintf(" BOA:calculate initial gains\n");
    end
    %calculate initial gains for candidate edges
    candidate_gains=zeros(model_params.problem_size)+min_gain;
    for i=1:model_params.problem_size
        if model_params.loglevel>1 && mod(floor(10*i/model_params.problem_size),1)==0
                fprintf(' %i%% ',floor(10*i/model_params.problem_size)*10);
        end
            
        %we only have to calculate half the array, as
        %    entropy(a)+conditional_entropy(b|a) = entropy(b)+conditional_entropy(a|b)
        for j=i:model_params.problem_size
            if i==j
                candidate_gains(i,j)=min_gain;
                continue
            end
            %candidate_model.structure_matrix=model.structure_matrix;
            %candidate_model.structure_matrix(i,j)=1;
            parents_i=[find((model.structure_matrix)(i,:)),j];
            new_score=node_score(i,parents_i,training_data);
            candidate_gains(i,j)=new_score-node_scores(i);
            candidate_gains(j,i)=candidate_gains(i,j);
            %fprintf(" New score for node %i when adding edge from  %i is %f, (gain %f) \n",i,j,new_score,candidate_gains(i,j));
        end
    end
    %node_scores
    %candidate_gains
    %input("STOP")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%  Build graph structure with greedy algorithm
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if model_params.loglevel>1
        fprintf(" BOA:build graph structure\n");
    end
    done=false;
    while(not(done))
        %fprintf("next iteration\n")
        %%%pick edge with largest gain
        [child,parent] = find(candidate_gains==max(max(candidate_gains)));
        child=child(1);
        parent=parent(1);
        
        if candidate_gains(child,parent)==min_gain
            done=true;
            if model_params.loglevel>1
                fprintf( "Done building graph, no more good edges \n")
            end
            continue
        end
        
        if model_params.loglevel>1
            fprintf(" best candidate: edge from %i to %i, gain %f \n",parent,child,candidate_gains(child,parent))
        end
        %%%add edge to model, remove gain
        model.structure_matrix(child,parent)=1;
        node_scores(child)+=candidate_gains(child,parent);
        candidate_gains(child,parent)=min_gain;
        candidate_gains(parent,child)=min_gain;
        
        %%%%%%%%%%%%%%
        %%%update inheritance_matrix (for cycle detection)
        model.inheritance_matrix(child,parent)=1;
        
        %%%make node a descendent of it's parents' ancenstors
        new_ancestors=[parent];
        
        for i=1:model_params.problem_size
            if model.inheritance_matrix(parent,i)==1 && model.inheritance_matrix(child,i)==0
                %fprintf(" inheritance: new parent of %i, %i, has parent %i, adding new inheritance\n",child,parent,i)
                new_ancestors=[new_ancestors,i];
                model.inheritance_matrix(child,i)=1;
                %set gain to new ancestor to min gain (as it would lead to a cycle)
                candidate_gains(i,child)=min_gain;
            end
        end
        
        %%%update descendents of node with new ancestors
        descendents=find(model.inheritance_matrix(:,child)==1);
        while length(descendents)>0
            %take first descendent in list
            node=descendents(1);
            if length(descendents)>1
                descendents=descendents(2:end);
            else
                descendents=[];
            end
            
            %update it, if necessary
            node_updated=false;
            for i=1:length(new_ancestors)
                if model.inheritance_matrix(node,new_ancestors(i))==0
                    model.inheritance_matrix(node,new_ancestors(i))=1;
                    %set gain from new ancestor to node to min gain (cycle)
                    candidate_gains(new_ancestors(i),node)=min_gain;
                    node_updated=true;
                end
            end
            %if node was updated, add its children to descendents list
            if node_updated
                descendents=unique([descendents;find(model.inheritance_matrix(:,node)==1)]);
            end

        end

        %%%update candidate gains for affected child node
        for i=1:model_params.problem_size
            if model.structure_matrix(child,i)==0 ...
                && not(model.inheritance_matrix(i,child)) ...
                && not(child==i)
                %candidate_model.structure_matrix=model.structure_matrix;
                %candidate_model.structure_matrix(child,i)=1;
                parents_i=[find((model.structure_matrix)(child,:)),i];
                new_score=node_score(child,parents_i,training_data);
                candidate_gains(child,i)=new_score-node_scores(child);
                %fprintf(" New score for node %i when adding edge from  %i is %f, (gain %f) \n",child,i,new_score,candidate_gains(child,i));
            end
        end
        
        if model_params.loglevel>1
            percent_done=length(find(candidate_gains<=min_gain))/(size(candidate_gains,1)*size(candidate_gains,2));
            fprintf("done %i ",percent_done*100);
        end
        
    end

    %input("test")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%  Calculate CPTs for each node
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if model_params.loglevel>1
        fprintf(" BOA:calculate CPTs\n");
    
    end
    model.cpts={};
    for i=1:model_params.problem_size
        %fprintf("Calculating CPT for node %i. Parents are:\n",i)
        parents=find(model.structure_matrix(i,:)==1);

        nodedata=(training_data(:,i));
        parentdata=training_data(:,parents);
        
        model.cpts{i}=calculate_cpt(nodedata,parentdata);
        
    end
    if model_params.loglevel>2
        model.cpts 
        cmd="python ./drawgraph.py "
        system(cstrcat(cmd,int2str(length(model.structure_matrix))," ",int2str(model.structure_matrix(:)'))," & > /dev/null 2>&1")
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%  Print Graphs, stats, ...
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if model_params.loglevel>1 && mod(e,model_params.show_stats_each_n_epochs)==0
        
        
        replot();
        if model_params.stopAfterPlot
            input("---Press Enter to continue---");
        end
    end    

    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% returns n samples from the model, including metadata
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [samples,metadata]=sampleBOA(model_params,model,n_samples,population)
    metatata=0;
    %samples=rand(n_samples,model_params.problem_size)>0.5;
    samples=zeros(n_samples,model_params.problem_size);
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%Sort graph topological (Kahn 1962, first algorithm in http://en.wikipedia.org/wiki/Topological_sorting)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if model_params.loglevel>1
        fprintf(" BOA:Sampling: topsort\n");
    end

    tmp_structure=model.structure_matrix;
    L=[];   %list with sorted elements
    S=find(sum(tmp_structure,2)==0);   %set of nodes with no incoming edges
    %S
    %input("check S")
    while length(S)>0
        
        %remove a node n from S
        n=S(1);
        if length(S)>1
            S=S(2:end);
        else
            S=[];
        end
        %insert n into L
        L=[L;n];
        
        %for each node m with an edge e from n to m 
        candidates=find(tmp_structure(:,n)==1);
        for i=1:length(candidates)
            m=candidates(i);
            %remove edge e
            tmp_structure(m,n)=0;
            %if m has no other incoming edges
            if sum(tmp_structure(m,:))==0
                %then insert m into S
                S=[S;m];
            end
        end
    end
    if sum(sum(tmp_structure))>0
        fprintf("ERROR graph has cycles\n");
        return
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%% Generate samples according to CPTs
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if model_params.loglevel>1
        fprintf(" BOA:sampling!\n");
    end
    if model_params.loglevel>2
        fprintf(" Order of nodes:");
        L'
    end

    %L

    for i=1:n_samples
        %iterate 
        for j=1:length(L)
            parents=find(model.structure_matrix(L(j),:)==1);
            cpt_entry_number=bi2de(samples(i,parents),"left-msb");
            if length(cpt_entry_number)==0
                cpt_entry_number=1;
            else
                cpt_entry_number+=1;
            end
            model.cpts;
            model.cpts{L(j)};
            samples(i,L(j))=model.cpts{L(j)}(cpt_entry_number)>rand();

        end
    
    end
end
