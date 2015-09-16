% EDA suite - rbm.m
% Copyright (C) <2010-2015> Malte Probst
% This file is part of the EDA suite.
%    The EDA suite is free software under the terms of the GNU General
%    Public License, see eda.m for full licence text, or <http://www.gnu.org/licenses/>.

%Implements a restricted boltzmann machine
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%initialization of model parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function model=initializeRBM(model_params,from_scratch=true,model=false,keep_old_model=0)
    clear model
    model.W=randn(model_params.problem_size,model_params.numhid)*model_params.std_W;
    model.visbias=zeros(1,model_params.problem_size);
    model.hidbias=zeros(1,model_params.numhid);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Trains an RBM on the given training data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function model=trainRBM(model_params,training_data,model,validation_data=false,fitness=false)

    %%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%% prepare stuff
    %%%%%%%%%%%%%%%%%%%%%%%%%

    m=size(training_data,1);
    %training_data=unique(training_data,'rows');
    %m=size(training_data,1)
    n_updates=0; %number of fitness updates
    %shuffle training data
    n = rand(m,1);
    [garbage index] = sort(n);
    training_data = training_data(index,:); 
    
    %calculate some helper variables
    
    v=0;
    if model_params.use_validation_data>0
        m=m-floor(model_params.use_validation_data*m);
        v=floor(model_params.use_validation_data*size(training_data,1));
    end
    
    %% Set batch size intelligently
    if model_params.batchsize==0 || m<model_params.variablebatches*model_params.minbatchsize
        model_params.batchsize=m;
    elseif m<model_params.variablebatchuntil
        model_params.batchsize=max(model_params.minbatchsize,floor(m/model_params.variablebatches));
    else
        %set batch size around desired value (minimize loss of training data)
        sizes=[model_params.batchsize,model_params.batchsize*0.9,model_params.batchsize*0.8,model_params.batchsize*0.7,model_params.batchsize*1.1,model_params.batchsize*1.2,model_params.batchsize*1.3];
        loss=mod(m,sizes);
        [garbage,index]=min(loss);
        model_params.batchsize=sizes(index);
        
    end
    
    %set sampling temperature of hiddens automatically, accoding to epoch
    model_params.temperature_hiddens=min(model_params.temperature_hiddens_max, ...
            model_params.temperature_hiddens*power(model_params.increase_hidden_temperature, ...
            model_params.generation-1));
    
            
    lr_reduced=0;
    momentum_increased=0;
    W_inc=0;
    
    %%%% initialize persistent chain, if used
    if model_params.use_pcd
        neghid=+(rand(model_params.batchsize,model_params.numhid)>0.5);  
    end
    
    %%%% normalize training data (zero mean), see Yichuan Tang and Ilya Sutskever 2011
    
    vis_mean=mean(training_data(1:m,:),1);
    if model_params.useNormalization
        normalized_training_data=training_data(1:m,:)-repmat(vis_mean,m,1);
        if model_params.use_validation_data>0
            normalized_validation_data=training_data(m+1:end,:)-repmat(vis_mean,v,1);
        end
    else
        normalized_training_data=training_data;
    end
    
    mean_free_energy_training_data=[];
    mean_free_energy_validation_data=[];
    
    reconstruction_error=[];
    validation_error=[]
    smoothed_reconstruction_error=[];
    smoothed_validation_error=[];
    slope_smoothed_reconstruction_error=[];
    relativeDifference=0;
    
    %Initialize vis biases to reflect mean value directly, see Hinton 2006, A practial guide to training RBMs
    smooth_mean=min(0.995,max(0.005,vis_mean));
    model.visbias=log(smooth_mean./(1-smooth_mean));

    model.mean_free_energy_difference=[];
    
    nUpdates=0;
    stopIt=false;
    
    %%use 25% of data for measuring recon error
    num_examples_for_reconErr=floor(m/model_params.batchsize)/4*model_params.batchsize;
    
    %extra_lr_factor=sqrt((m/model_params.batchsize)/model_params.variablebatches);
    extra_lr_factor=1;
    %%%%%%%%%%%%%%%%%%%%%%%%%
    %%%% Run n epochs of training with dataset
    %%%%%%%%%%%%%%%%%%%%%%%%%
    for e=1:model_params.n_epochs
            if e==model_params.n_epochs
                fprintf("WARNING: Maximum number of epochs reached. Increase model_params.n_epochs (check for diverging optimization objective!)\n")
            end
            if model_params.data_not_binary
                fprintf("TODO: code structure incorrect. each epoch needs new sampling and subsequent mean subtraction\n")
                normalized_training_data=+(training_data(1:m,:)>rand(size(training_data(1:m,:))));
            end
            %%%% Train all batches within dataset
            for b=1:floor(m/model_params.batchsize)
                %determine batch bounds, set visibles to batch
                from=(b-1)*model_params.batchsize+1;
                to=b*model_params.batchsize;
                
                vis=normalized_training_data(from:to,:);
                
                %%%% go up
                %go up
                hidnet=(vis)*model.W+repmat(model.hidbias,model_params.batchsize,1);
                hidprobs=sigmoid(hidnet,model_params.temperature);
                hid=+(hidprobs>rand(model_params.batchsize,model_params.numhid));
                
                
                %%%% take positive statistics
                %TODO: check if it is correct to calculate gradients from mean-corrected vis values
                dW=vis'*hid;
                dvisbias=sum(vis);
                dhidbias=sum(hid);
                                
                
                %if non-persistent chains use hid from positive phase
                if not(model_params.use_pcd)

                    neghid=hid;

                end
                
                %perform n gibbs iterations
                for i=0:model_params.n_gibbs_training
                        
                        %go down
                        negvisprobs=sigmoid(neghid*model.W'+repmat(model.visbias,model_params.batchsize,1), ...
                            model_params.temperature);
                        negvis=+(negvisprobs>rand(model_params.batchsize,model_params.problem_size));
                        
                        %%%% subtract mean from vis, artifact from data normalization (see above)
                        if model_params.useNormalization
                            negvis-=repmat(vis_mean,model_params.batchsize,1);
                        end
                        %if not pcd, save probs of first batch to calculate reconstruction error
                        if not(model_params.use_pcd) && i==0 ...
                                && mod(e,model_params.calculate_free_energy_each_n_epochs)==0 ...
                                && b==1
                            reconstructions=negvis;
                        end
                                                
                        %go up
                        hidnet=(negvis)*model.W+repmat(model.hidbias,model_params.batchsize,1);
                        neghidprobs=sigmoid(hidnet,model_params.temperature);
                        neghid=+(neghidprobs>rand(model_params.batchsize,model_params.numhid));
                        
                end
                
                %%%% take negative statistics
                dW-=negvis'*neghidprobs;
                dvisbias-=sum(negvis);
                dhidbias-=sum(neghidprobs);
                
                
                %%%% calculate gradients
                W_inc=model_params.momentum*W_inc+((model_params.alpha/model_params.batchsize)/extra_lr_factor)*dW;
                visbias_inc=((model_params.visbias_factor*model_params.alpha/model_params.batchsize)/extra_lr_factor)*dvisbias;
                hidbias_inc=((model_params.hidbias_factor*model_params.alpha/model_params.batchsize)/extra_lr_factor)*dhidbias;
                
                %%%% update model
                model.W+=W_inc;
                model.visbias+=visbias_inc;
                model.hidbias+=hidbias_inc;
                
                nUpdates+=1;
            end %batches
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%  Update stats, ...
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        if mod(e,model_params.calculate_free_energy_each_n_epochs)==0 && model_params.loglevel>1
            fe=0;
            vis_term=-sum(mean(training_data(1:m,:).*repmat(model.visbias,m,1)));
            %calculate hid input (with modified hidbias due to mean-normalization, as in final model)
            hid_input=training_data(1:m,:)*model.W+repmat(model.hidbias-vis_mean*model.W,m,1);
            hid_term=-sum(mean(log(1+exp(hid_input))));
            fe=vis_term+hid_term;
            mean_free_energy_training_data=[mean_free_energy_training_data,fe];
        
            if model_params.use_validation_data>0
                fe=0;
                vis_term=-sum(mean(training_data(m+1:end,:).*repmat(model.visbias,v,1)));
                %calculate hid input (with modified hidbias due to mean-normalization, as in final model)
                hid_input=training_data(m+1:end,:)*model.W+repmat(model.hidbias-vis_mean*model.W,v,1);
                hid_term=-sum(mean(log(1+exp(hid_input))));
                fe=vis_term+hid_term;
                mean_free_energy_validation_data=[mean_free_energy_validation_data,fe];
                
                %relativeDifference=0.5*relativeDifference+0.5*(mean_free_energy_training_data(end)-mean_free_energy_validation_data(end))./mean_free_energy_training_data(end);

                model.mean_free_energy_difference=(mean_free_energy_training_data-mean_free_energy_validation_data)./mean_free_energy_training_data;

            else
                mean_free_energy_validation_data=mean_free_energy_training_data;
            end
        end
        if mod(e,model_params.calculate_free_energy_each_n_epochs)==0    
            %save reconstruction error
            if not(model_params.use_pcd)
                tmp_hidnet=(normalized_training_data(1:num_examples_for_reconErr,:))*model.W+repmat(model.hidbias,num_examples_for_reconErr,1);
                tmp_hidprobs=sigmoid(tmp_hidnet,model_params.temperature);
                tmp_hid=+(tmp_hidprobs>rand(num_examples_for_reconErr,model_params.numhid));
                tmp_negvisprobs=sigmoid(tmp_hid*model.W'+repmat(model.visbias,num_examples_for_reconErr,1), ...
                    model_params.temperature);
                tmp_negvis=+(tmp_negvisprobs>rand(num_examples_for_reconErr,model_params.problem_size));
                %%%% subtract mean from vis, artifact from data normalization (see above)
                if model_params.useNormalization
                    tmp_negvis-=repmat(vis_mean,num_examples_for_reconErr,1);
                end
                recon_err=mean(mean(abs(normalized_training_data(1:num_examples_for_reconErr,:)-tmp_negvis)));
                
                
                %recon_err=mean(mean(abs(normalized_training_data(1:model_params.batchsize,:)-reconstructions)));
                
                reconstruction_error=[reconstruction_error,recon_err];
                
                if model_params.use_validation_data>0
                    tmp_hidnet=((normalized_validation_data)*model.W)+repmat(model.hidbias,v,1);
                    tmp_hidprobs=sigmoid(tmp_hidnet,model_params.temperature);
                    tmp_hid=+(tmp_hidprobs>rand(v,model_params.numhid));
                    tmp_negvisprobs=sigmoid(tmp_hid*model.W'+repmat(model.visbias,v,1), ...
                        model_params.temperature);
                    tmp_negvis=+(tmp_negvisprobs>rand(v,model_params.problem_size));
                    %%%% subtract mean from vis, artifact from data normalization (see above)
                    if model_params.useNormalization
                        tmp_negvis-=repmat(vis_mean,v,1);
                    end
                    validation_err=mean(mean(abs(normalized_validation_data-tmp_negvis)));
                                        
                    validation_error=[validation_error,validation_err];
                end
                
                if e==model_params.calculate_free_energy_each_n_epochs
                    smoothed_reconstruction_error=recon_err;
                    #smoothed_reconstruction_error=0.5-(0.5-mean_err)
                else
                    smoothed_reconstruction_error=[smoothed_reconstruction_error,0.6*smoothed_reconstruction_error(end)+0.4*recon_err];
                    #smoothed_reconstruction_error=[smoothed_reconstruction_error,0.9*smoothed_reconstruction_error(end)+0.1*(recon_err-(0.5-mean_err))];
                end
                
                if model_params.use_validation_data>0
                    if e==model_params.calculate_free_energy_each_n_epochs
                        smoothed_validation_error=validation_error;
                    else
                        smoothed_validation_error=[smoothed_validation_error,0.6*smoothed_validation_error(end)+0.4*validation_error];
                    end
                
                end
                
                if e==model_params.calculate_free_energy_each_n_epochs
                    slope_smoothed_reconstruction_error=1;
                else
                    %calculate number of epochs that make up last quarter
                    x_orig=((e)/4)/model_params.calculate_free_energy_each_n_epochs;
                    x=max(1,floor(x_orig)); %minimum 1
                    
                    %calculate total gain
                    total=max((smoothed_reconstruction_error(1)-smoothed_reconstruction_error(end)),1e-10);
                    
                    %calculate gain of last quarter (making it a bit smoother)
                    if x_orig>1
                        tmp=x_orig-x;
                        %fprintf("Taking %f times value of %i-th last point  and %f times value of %i-th last point",(1-tmp),(x),tmp,(x+1))
                        last=((smoothed_reconstruction_error(end-x)*(1-tmp)+tmp*smoothed_reconstruction_error(end-(x+1)))-smoothed_reconstruction_error(end));
                    else
                        last=(smoothed_reconstruction_error(end-x)-smoothed_reconstruction_error(end));
                    end
                    ratio=max(0,min(1,last/total));
                    
                    
                    slope_smoothed_reconstruction_error=[slope_smoothed_reconstruction_error,ratio];
                    
                    slope_smoothed_reconstruction_error(end);
                end
                
                if model_params.loglevel>2
                    fprintf("---epoch %i  (%i updates)---\n",e,nUpdates)
                    fprintf("last  gain ratio: %f\n",slope_smoothed_reconstruction_error(end))
                    fprintf("total gain      : %f\n",smoothed_reconstruction_error(end)/(smoothed_reconstruction_error(1)+1e-15))
                end
                if slope_smoothed_reconstruction_error(end)<0.1 && not(momentum_increased) && min(smoothed_reconstruction_error)/(smoothed_reconstruction_error(1)+1e-15)<0.99
                    model_params.momentum=model_params.momentum2;
                    if model_params.loglevel>2
                        fprintf("  Epoch %i:Gain in last fourth of training was less than 10 percent of total gain, and total gain was more than 5 percent of initial RE. Setting momentum to rate to %f\n",e,model_params.momentum)
                    end
                    momentum_increased=1;
                end
                if slope_smoothed_reconstruction_error(end)<0.05 && not(lr_reduced) && min(smoothed_reconstruction_error)/(smoothed_reconstruction_error(1)+1e-15)<0.99
                    model_params.alpha/=2;
                    if model_params.loglevel>2
                        fprintf("  Epoch %i:Gain in last fourth of training was less than 5 percent of total gain, and total gain was more than 5 percent of initial RE. Reducing learning rate to %f\n",e,model_params.alpha)
                    end
                    lr_reduced=1;
                end
                if slope_smoothed_reconstruction_error(end)<0.01 && min(smoothed_reconstruction_error)/(smoothed_reconstruction_error(1)+1e-15)<0.99
                    if model_params.loglevel>2
                        fprintf("  Epoch %i:Gain in last fourth of training was less than 1 percent of total gain, and total gain was more than 5 percent of initial RE. Stopping now\n",e)
                    end
                    stopIt=true;
                elseif smoothed_reconstruction_error(end)<0.01
                    if model_params.loglevel>2
                        fprintf("  Epoch %i:Total reconstruction Error less than 1 percent. Stopping now\n",e)
                    end
                    stopIt=true;
                end
                if slope_smoothed_reconstruction_error(end)==0 ...
                    && (smoothed_reconstruction_error(end)/min(smoothed_reconstruction_error)>1.02) ...
                    && ((nUpdates>2000 && e>30) || smoothed_reconstruction_error(end)/(smoothed_reconstruction_error(1)+1e-15)<0.9)
                    if model_params.loglevel>2
                        fprintf("  Epoch %i:Reconstruction error grows! Stopping\n",e)
                    end
                    stopIt=true;
                end
            end
            
        end
        
        if e>50 && nUpdates>3000 && 0
            model_params.momentum=model_params.momentum2;
            momentum_increased=1;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%  Print Graphs, stats, ...
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        if model_params.loglevel>1 && mod(e,model_params.show_stats_each_n_epochs)==0
            
            
            %%%%%%%%%%%%%%%%%%%
            if model_params.loglevel>2
            figure(212102,'name','Weight Increase','Position',[300,400,300,300])
            mx=max(max(W_inc));
            mn=min(min(W_inc));
            mx=max(abs(mn),mx);
            img=zeros(size(W_inc',1)+4,size(W_inc',2)+4)+0.5;
            img(2:end-1,2:end-1)=1;
            img(3:end-2,3:end-2)=W_inc'/(2*mx)+0.5;
            imshow(img)
            temp=strcat("Winc, max=",mat2str(mx,2));
            title(temp)
            %%%%%%%%%%%%%%%%%%%
            figure(212110,'name','Hidbias Histogram','Position',[300,400,300,120]);
            hist(model.hidbias);
            title("Hidbias Histogram");
            %%%%%%%%%%%%%%%%%%%
            figure(212104,'name','Hidden Probs Histogram (last pos batch)','Position',[300,400,300,120])
            hist(hidprobs(:));
            title("Hidden Probs Histogram (last pos batch)");
            %%%%%%%%%%%%%%%%%%%
            figure(108,'name','Visbias Histogram','Position',[300,400,300,120]);
            hist(model.visbias);
            title("Visbias Histogram");
            %%%%%%%%%%%%%%%%%%%
            figure(212112,'name','Visprobs Histogram (last neg batch)','Position',[300,400,300,120]);
            hist(negvisprobs(:),10);
            title("Negvisprobs Histogram (last neg batch)");
            %%%%%%%%%%%%%%%%%%%
            figure(212120,'name','Hidden Probs (last pos batch)','Position',[300,400,300,120]);
            img=zeros(size(hid,1)+2,size(hid,2)+2);
            img(2:end-1,2:end-1)=hidprobs;
            imshow(img)
            title("Hidden Activity (last pos batch)");
            end
            %%%%%%%%%%%%%%%%%%%
            figure(212130,'name','Average Free Energy (red: training data, green: validation data)','Position',[300,400,300,250])
            plot([1:model_params.calculate_free_energy_each_n_epochs:e],mean_free_energy_validation_data,"2",[1:model_params.calculate_free_energy_each_n_epochs:e],mean_free_energy_training_data,"1");
            title("Average Free Energy (red: training data, green: validation data)");
            %%%%%%%%%%%%%%%%%%%
            if model_params.use_validation_data>0
                figure(212132,'name','Difference FE(training)-FE(validation)','Position',[300,400,300,250])
                plot([1:model_params.calculate_free_energy_each_n_epochs:e],mean_free_energy_validation_data-mean_free_energy_training_data);
                title("Difference FE(training)-FE(validation))");
                
                figure(212133,'name','Relative Difference FE(validation)-FE(training)/FE(training)','Position',[300,400,300,250])
                plot([1:model_params.calculate_free_energy_each_n_epochs:e],(mean_free_energy_training_data-mean_free_energy_validation_data)./mean_free_energy_training_data);
                title("Relative Difference FE(validation)-FE(training)/FE(training)");
            end
            %%%%%%%%%%%%%%%%%%%
            figure(212134,'name','Mean Reconstruction Error','Position',[300,400,300,250])
            plot([1:model_params.calculate_free_energy_each_n_epochs:e],reconstruction_error);
            title("Mean reconstruction error");
            %%%%%%%%%%%%%%%%%%%
            %for i=1:model_params.calculate_free_energy_each_n_epochs:e
            %    if i==1
            %        reconstruction_error_smooth=(0.5+reconstruction_error(i))/2;
            %    else
            %        reconstruction_error_smooth=[reconstruction_error_smooth,(reconstruction_error_smooth(end)+reconstruction_error(floor(i/model_params.calculate_free_energy_each_n_epochs)))/2];
            %    end
            %end
            figure(212136,'name','Smoothed Mean Reconstruction Error','Position',[300,400,300,250])
            plot([1:model_params.calculate_free_energy_each_n_epochs:e],smoothed_reconstruction_error);
            title("Smoothed Mean reconstruction error");
            %%%%%%%%%%%%%%%%%%%
            figure(212138,'name','Slope of Smoothed MRE','Position',[300,400,300,250])
            plot([1:model_params.calculate_free_energy_each_n_epochs:e],slope_smoothed_reconstruction_error);
            title("Slope of smoothed MRE");
            %%%%%%%%%%%%%%%%%%
            figure(212122,'name','Weights Histogram','Position',[300,400,300,100])
            hist(model.W(:),50);
            title("Weights Histogram");
            %%%%%%%%%%%%%%%%%%%
            
            
            
            refresh();
            if model_params.stopAfterPlot
                input("---Press Enter to continue---");
            end
        end  
        if(stopIt)
            break
        end  
    end
    %fprintf("   trained epochs:                 %i \n",e)
    %%%% modify hid bias, artifact from data normalization (see above)
    if model_params.useNormalization
        model.hidbias-=vis_mean*model.W;
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% returns n samples from the model, including metadata (such as corresponding hidden states, energies...)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [samples,metadata]=sampleRBM(model_params,model,n_samples,population)
    
    %initialize randomly, according to mean of current population, or with current pop
    if model_params.initialize_sample_from_average
        vis=(repmat(mean(population),n_samples,1));
    elseif model_params.initialize_sample_from_population
        vis=population;
    else
        vis=+(rand(n_samples,model_params.problem_size)>0.5);
    end
    
    %perform n gibbs iterations
    for i=0:model_params.n_gibbs_sampling
            %go up
            hidnet=(vis)*model.W+repmat(model.hidbias,n_samples,1);
            hidprobs=sigmoid(hidnet,model_params.temperature_hiddens);
            hid=+(hidprobs>rand(n_samples,model_params.numhid));
            %go down
            visprobs=sigmoid(hid*model.W'+repmat(model.visbias,n_samples,1),model_params.temperature);
            vis=+(visprobs>rand(n_samples,model_params.problem_size));
            
    end

    metadata.hiddens=hid;
    metadata.visprobs=visprobs;
    samples=vis;
    

end
