%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Observations, todos, ideas
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%TODO: check how the sampling run should be initialized (random? mean of last pop? ...?)
%TODO: check: when sampling from the model, the following observation was made:
%            samples that are close to the bias distribution are chosen much more often than
%            their proportion in the training set. in case of the eda use case, this biases the
%            whole thing the mean winnign population, i.e. the trap in case of deceptive traps
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%initialization of model parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function model=initialize3wayRBM(model_params,from_scratch=true,model=false,keep_old_model=0)
    if from_scratch || keep_old_model==0
        clear model
        model.vis1fac=randn(model_params.numvis,model_params.numfac)*model_params.std_W;
        model.vis2fac=randn(model_params.numvis,model_params.numfac)*model_params.std_W;
        model.hidfac=randn(model_params.numhid,model_params.numfac)*model_params.std_W;
        model.vis2bias=zeros(1,model_params.numvis);
        model.hidbias=zeros(1,model_params.numhid);
    else
        model.vis1fac*=keep_old_model;
        model.vis2fac*=keep_old_model;
        model.hidfac*=keep_old_model;
        model.vis2bias*=keep_old_model;
        model.hidbias*=keep_old_model;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Trains a 3-way-RBM on the given training data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function model=train3wayRBM(model_params,training_data,model,validation_data=false,fitness)
    %%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%% prepare stuff
    %%%%%%%%%%%%%%%%%%%%%%%%%
    
    training_data=unique(training_data,"rows");
    m=size(training_data,1);
	if model_params.loglevel>1
		fprintf("Unique Training cases: %i  (!)\n",m)
	end
		%figure(95547,'name','Unique Training Data','Position',[300,400,300,300])
        %tmp=training_data;
        %img=zeros(size(tmp,1)+2,size(tmp,2)+2);
        %img(2:end-1,2:end-1)=tmp;
        %imshow(img)
        %title("Unique Training Data");
	
    if m<10
		return 
	end
    %%%%%%%%%%%%%%Pretrain model
    if model_params.loglevel>1
		fprintf("Pretraining params with RBM\n")
	end
	pretrainmodel=model_params.pretrain.finitializeModel(model_params.pretrain);
    model_params.pretrain.generation=model_params.generation;
    model_params.pretrain.loglevel=model_params.loglevel;
    pretrainmodel=model_params.pretrain.ftrainModel(model_params.pretrain,training_data,pretrainmodel);
    model.vis1fac+=pretrainmodel.W*0.5;
    
    %%%%initialize vis2 bias biases to reflect mean value directly
    vis2_mean=mean(training_data(1:m,:),1);
    smooth_mean=min(0.9995,max(0.0005,vis2_mean));
    model.vis2bias=log(smooth_mean./(1-smooth_mean));
    
    %build an extended list of training data with all fitnesss comparisons
    training_data_vis1=[];
    training_data_vis2=[];
    if model_params.loglevel>1
            fprintf("Preparing training data: kmeans...\n");
    end
    %numClusters=max(1,ceil(model_params.numClusters-(0.5*model_params.generation-1)));
    numClusters=1;
    numClustersFound=false;
    while !numClustersFound
        if numClusters*power(m/numClusters,2)<model_params.numDesiredExamples
            numClustersFound=true;
        else
            numClusters+=1;
        end
    end
    if model_params.loglevel>1
            fprintf("Number of clusters %i\n",numClusters);
    end

    %numClusters=model_params.numClusters
    found=false;
    if numClusters==1
        found=true
        idx=ones(size(training_data,1),1);
    end
    attempts=0;
    while !found
        try
            attempts+=1;
            [idx, centers] = kmeans (training_data*1.0, numClusters);
            found=true;
            if model_params.loglevel>1
                figure(5555);
                hist(idx);
                title("Cluster histogram");
                replot();
            end
        catch
            msg = lasterror.message;
            fprintf(msg)
            if attempts==5
                found=true;
                idx=ones(size(training_data,1),1);
            else 
                fprintf("...trying again width fewer clusters\n")
                numClusters=max(1,numClusters-1);
            end
            
        end_try_catch
    end
    count=0;
    if model_params.loglevel>1
            fprintf("Preparing training data: generating sets ...");
    end
    
    %if numClusters==1
    %    list=[1:m,1:m]';
    %    for i=2:m
    %        training_data_vis1=[training_data_vis1;list(i:m+i-1)];
    %        training_data_vis2=[training_data_vis2;list(1:m)];
    %         
    %    end
    %else
    
    list=[1:m,1:m]';
    for i=2:m
            training_data_vis1=[training_data_vis1;list(i:m+i-1)];
            training_data_vis2=[training_data_vis2;list(1:m)];
             
    end
    keep=zeros(size(training_data_vis1));
    for i=1:numClusters
		i
		partnerCluster=mod(i,numClusters)+1
		keep=keep|all([idx(training_data_vis1),idx(training_data_vis2)]==repmat([i,partnerCluster],length(training_data_vis1),1),2);
    
    end
    %for i=1:length(training_data_vis1)
	%	if idx(training_data_vis1)!=idx(training_data_vis2)
	%		keep=[keep,i];
	%	end
    %end
    training_data_vis1=training_data_vis1(keep);
    training_data_vis2=training_data_vis2(keep);
    count=length(training_data_vis1)
    %lists={};
    %for i=1:numClusters
    %    lists{i}=[];
    %    for j=1:m
    %        if(idx(j)==i)
    %            lists(i)=[lists{i};j];
    %        end
    %    end
    %    lists(i)=[lists{i};lists{i}];
    %    m2=length(lists{i})/2;
    %    for j=2:m2
    %        training_data_vis1=[training_data_vis1;lists{i}(j:m2+j-1)];
    %        training_data_vis2=[training_data_vis2;lists{i}(1:m2)];
    %    end
    %
    %    count+=power(m2,2)-m2;
    %
    %end
    
    
    %[training_data_vis1,training_data_vis2]
    %    for i=1:m
    %        for j=1:m
    %            if (idx(i)==idx(j) && i!=j)% &&  !all (training_data(i,:)==training_data(j,:)))
    %                    training_data_vis1=[training_data_vis1;j];
    %                    training_data_vis2=[training_data_vis2;i];
    %                    %training_data_vis1=[training_data_vis2;i];
    %                    %training_data_vis2=[training_data_vis1;j];
    %                    count+=1;
    %                
    %            end
    %        end
    if model_params.loglevel>1 && mod(i,floor(m/10))==0
        fprintf(" %i ",100*i/m);
    end
            %for j=i:m
            %    if (idx(i)==idx(j))
            %        if(fitness(i)>fitness(j))
            %            training_data_vis1=[training_data_vis1;j];
            %            training_data_vis2=[training_data_vis2;i];
            %            count+=1;
            %        elseif(fitness(i)<fitness(j))
            %            training_data_vis1=[training_data_vis2;i];
            %            training_data_vis2=[training_data_vis1;j];
            %            count+=1;
            %        end
            %end
            
    %    end
    %end
    if model_params.loglevel>1
            fprintf("done (%i cases)\n",count);
    end
    while 5*model_params.batchsize>count && model_params.batchsize>2
        model_params.batchsize=max(2,model_params.batchsize/2);
    end
    
    %shuffle training data
    n = rand(count,1);
    [garbage index] = sort(n);
    training_data_vis1 = training_data_vis1(index,:); 
    training_data_vis2 = training_data_vis2(index,:); 
    
    
    if model_params.batchsize==0
        model_params.batchsize=count;
    end

    
    vis1fac_inc=0;
    vis2fac_inc=0;
    hidfac_inc=0;
    

    
    reconErr=[];
    %%%%%%%%%%%%%%%%%%%%%%%%%
    %%%% Run n epochs of training with dataset
    %%%%%%%%%%%%%%%%%%%%%%%%%
    learningRate=model_params.alpha;
    for e=1:model_params.n_epochs
            %%%% Train all batches within dataset
            for b=1:floor(count/model_params.batchsize)
				learningRate=model_params.alpha*(20000.0/(20000.0+e*floor(count/model_params.batchsize)+b));  %exponentially decaying learning rate
				%determine batch bounds, set visibles to batch
                from=(b-1)*model_params.batchsize+1;
                to=b*model_params.batchsize;
                model_params.from=from;
                model_params.to=to;
                vis1=training_data(training_data_vis1(from:to),:);
                vis2=training_data(training_data_vis2(from:to),:);
                
                
                %calculate stable input to factors from vis1 and vis2 (independent from hid)
                factorinputfromvis1=vis1*model.vis1fac;
                posfactorinputfromvis2=vis2*model.vis2fac;
                
                %%%% go up
                hidnet=repmat(model.hidbias,model_params.batchsize,1); %bias term
                hidnet+=(factorinputfromvis1.*posfactorinputfromvis2)*model.hidfac';
                hidprobs=sigmoid(hidnet,model_params.temperature);
                hid=+(hidprobs>rand(model_params.batchsize,model_params.numhid));
                
                posfactorinputfromhid=hidprobs*model.hidfac;

                %%%% take positive statistics
                dvis1fac=vis1'*(posfactorinputfromvis2.*posfactorinputfromhid);
                dvis2fac=vis2'*(factorinputfromvis1.*posfactorinputfromhid);
                dhidfac=hidprobs'*(factorinputfromvis1.*posfactorinputfromvis2);
                dvis2bias=sum(vis2);
                dhidbias=sum(hid);
                
                neghid=hid;
                %perform n gibbs iterations
                nGibbs=model_params.n_gibbs_training;
                for i=0:nGibbs
                        
                        %go down
                        negvis2net=repmat(model.vis2bias,model_params.batchsize,1);
                        negvis2net+=(factorinputfromvis1.*(neghid*model.hidfac))*model.vis2fac';
                        negvis2probs=sigmoid(negvis2net,model_params.temperature);
                        negvis2=+(negvis2probs>rand(model_params.batchsize,model_params.numvis));                        
                        
                        %go up
                        neghidnet=repmat(model.hidbias,model_params.batchsize,1); %bias term
                        neghidnet+=(factorinputfromvis1.*(negvis2*model.vis2fac))*model.hidfac';
                        neghidprobs=sigmoid(neghidnet,model_params.temperature);
                        neghid=+(neghidprobs>rand(model_params.batchsize,model_params.numhid));
                
                        
                end
                
                %%%% take negative statistics
                negfactorinputfromhid=neghidprobs*model.hidfac;
                negfactorinputfromvis2=negvis2probs*model.vis2fac;
                
                dvis1fac-=vis1'*(negfactorinputfromvis2.*negfactorinputfromhid);
                dvis2fac-=negvis2probs'*(factorinputfromvis1.*negfactorinputfromhid);
                dhidfac-=neghidprobs'*(factorinputfromvis1.*negfactorinputfromvis2);
                dvis2bias-=sum(negvis2);
                dhidbias-=sum(neghidprobs);
                
                %%%% add regularization
                if model_params.l2_penalty>0
                    dvis1fac-=model_params.l2_penalty*model.vis1fac;
                    dvis2fac-=model_params.l2_penalty*model.vis2fac;
                    dhidfac-=model_params.l2_penalty*model.hidfac;
                end
                
                %%%% calculate gradients
                vis1fac_inc=model_params.momentum*vis1fac_inc+(learningRate/model_params.batchsize)*dvis1fac;
                vis2fac_inc=model_params.momentum*vis2fac_inc+(learningRate/model_params.batchsize)*dvis2fac;
                hidfac_inc=model_params.momentum*hidfac_inc+(learningRate/model_params.batchsize)*dhidfac;
                vis2bias_inc=(model_params.visbias_factor*learningRate/model_params.batchsize)*dvis2bias;
                hidbias_inc=(model_params.hidbias_factor*learningRate/model_params.batchsize)*dhidbias;
                
                %%%% update model
                model.vis1fac+=vis1fac_inc;
                model.vis2fac+=vis2fac_inc;
                model.hidfac+=hidfac_inc;
                model.vis2bias+=vis2bias_inc;
                model.hidbias+=hidbias_inc;
                

            end %batches

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%  Update stats, ...
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if model_params.loglevel>1 && mod(e,model_params.calculate_stats_each_n_epochs)==0
            %calculate reconstruction error on last batch
            if model_params.n_gibbs_training>1
                fprintf("warning cd-n for n>1 not supported for stats!\n")
            end
            diff=(training_data(training_data_vis2(from:to),:)-negvis2);
            diff=(mean(mean(abs(diff))));
            reconErr=[reconErr,diff];
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%  Print Graphs, stats, ...
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        if model_params.loglevel>1 && mod(e,model_params.show_stats_each_n_epochs)==0
            fprintf("Epoch %i\n",e);
			fprintf("LR: %f \n",learningRate)

            %%%%%%%%%%%%%%%%%%%
            figure(1010,'name','vis1fac','Position',[300,400,300,300])
            mx=max(max(model.vis1fac));
            mn=min(min(model.vis1fac));
            mx=max(abs(mn),mx);
            img=zeros(size(model.vis1fac',1)+4,size(model.vis1fac',2)+4);
            img(2:end-1,2:end-1)=mx;
            img(3:end-2,3:end-2)=model.vis1fac';
            imshow(img,[-mx,mx])
            temp=strcat("vis1fac, max=",mat2str(mx,2));
            title(temp)
            %%%%%%%%%%%%%%%%%%%
            figure(1020,'name','vis2fac','Position',[300,400,300,300])
            mx=max(max(model.vis2fac));
            mn=min(min(model.vis2fac));
            mx=max(abs(mn),mx);
            img=zeros(size(model.vis2fac',1)+4,size(model.vis2fac',2)+4);
            img(2:end-1,2:end-1)=mx;
            img(3:end-2,3:end-2)=model.vis2fac';
            imshow(img,[-mx,mx])
            temp=strcat("vis2fac, max=",mat2str(mx,2));
            title(temp)   
            %%%%%%%%%%%%%%%%%%%
            figure(1030,'name','hidfac','Position',[300,400,300,300])
            mx=max(max(model.hidfac));
            mn=min(min(model.hidfac));
            mx=max(abs(mn),mx);
            img=zeros(size(model.hidfac',1)+4,size(model.hidfac',2)+4);
            img(2:end-1,2:end-1)=mx;
            img(3:end-2,3:end-2)=model.hidfac';
            imshow(img,[-mx,mx])
            temp=strcat("hidfac, max=",mat2str(mx,2));
            title(temp)     
            
            %%%%%%%%%%%%%%%%%%%
            figure(1040,'name','vis2fac_inc','Position',[300,400,300,300])
            mx=max(max(vis2fac_inc));
            mn=min(min(vis2fac_inc));
            mx=max(abs(mn),mx);
            img=zeros(size(vis2fac_inc',1)+4,size(vis2fac_inc',2)+4);
            img(2:end-1,2:end-1)=mx;
            img(3:end-2,3:end-2)=vis2fac_inc';
            imshow(img,[-mx,mx])
            temp=strcat("vis2fac_inc, max=",mat2str(mx,2));
            title(temp) 
            
             %%%%%%%%%%%%%%%%%%%
            figure(1080,'name','Visbias Histogram','Position',[300,400,300,120]);
            hist(model.vis2bias);
            title("Visbias Histogram");
            %%%%%%%%%%%%%%%%%%%
            figure(1120,'name','Visprobs Histogram (last neg batch)','Position',[300,400,300,120]);
            hist(negvis2probs(:),10);
            title("Negvisprobs Histogram (last neg batch)");  
            %%%%%%%%%%%%%%%%%%%
            
             %%%%%%%%%%%%%%%%%%%
            figure(1300,'name','Reconstruction error on last batch','Position',[300,400,300,250])
            plot([1:model_params.calculate_stats_each_n_epochs:e],reconErr);
            title("Reconstruction error on last batch)");
            replot()
        end    
        if reconErr(end)<model_params.stopAtReconErr
            if model_params.loglevel>0
                fprintf("Converged to < %f per cent reconstruction error. Breaking\n",model_params.stopAtReconErr*100);
            end
            break
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% returns n samples from the model, including metadata (such as corresponding hidden states, energies...)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [samples,metadata]=sample3wayRBM(model_params,model,n_samples,vis1)
    vis2=vis1;
    vis2=rand(size(vis1))>0.5;
    %calculate stable input to factors from vis1
    factorinputfromvis1=vis1*model.vis1fac;
    %perform n gibbs iterations
    for r=0:model_params.numberSamplingCycles-1
        if model_params.loglevel>1
                fprintf("Sampling: Transformation step %i\n",r+1);
        end
        for i=0:model_params.n_gibbs_sampling
                %go up
                hidnet=repmat(model.hidbias,n_samples,1); %bias term
                hidnet+=(factorinputfromvis1.*(vis2*model.vis2fac))*model.hidfac';
                hidprobs=sigmoid(hidnet,model_params.temperature);
                hid=+(hidprobs>rand(n_samples,model_params.numhid));
                %go down
                vis2net=repmat(model.vis2bias,n_samples,1);
                vis2net+=(factorinputfromvis1.*(hid*model.hidfac))*model.vis2fac';
                vis2probs=sigmoid(vis2net,model_params.temperature);
                vis2=+(vis2probs>rand(n_samples,model_params.numvis));
                
        end
        vis1=vis2;
        %vis2=rand(size(vis1))>0.5;
    end
    
    metadata.hiddens=hid;
    metadata.visprobs=vis2probs;
    samples=vis2;
end
