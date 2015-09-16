% EDA suite - ae.m
% Copyright (C) <2010-2015> Malte Probst
% This file is part of the EDA suite.
%    The EDA suite is free software under the terms of the GNU General
%    Public License, see eda.m for full licence text, or <http://www.gnu.org/licenses/>.

%(Denoising) Autoencoder
%%%%helpers
if strcmp(version,"3.2.4")!=1
    graphics_toolkit ("gnuplot")
end
function y=sigmoid(x,temp=1.0)
    y=1.0 ./(1.0+exp(-x./temp));
end
%calculate cross entropy error
function error=calculateCost(hp,reconstructions,targets)
    error=-sum(mean((targets.*log(reconstructions)+(1-targets).*log(1-reconstructions))));
end
function showDigit(flatData)
    imshow(reshape(flatData,28,28))
end

function params=initializeAE(hp,fromScratch,paramsOld,keepModel)
    %check some values
    if mod(length(hp.layerSizes),2)==0
        error("Number of layers must be odd (i.e., even number of weight matricies)\n");
    end 
    params.weights={};
    params.biases={};
    %experimental: creativity sampling (extra class-conditional bias)
    if hp.useCreativitySampling
        params.classBiases={};
    end
    %weight sharing, i.e., weights for decoding function are transposed encoding weights
    for layer=1:(length(hp.layerSizes)-1)/2
        numPieces=1;
        if(strcmp(hp.unitTypes{layer},'maxout'))
            numPieces=hp.maxoutPieces;
        end     
        if(strcmp(hp.initializationNoiseType,'uniform'))
            params.weights{layer}=(rand(hp.layerSizes(layer),hp.layerSizes(layer+1),numPieces)-0.5)*hp.randomWeightMultiplier;
        elseif(strcmp(hp.initializationNoiseType,'normal'))
            params.weights{layer}=(randn(hp.layerSizes(layer),hp.layerSizes(layer+1),numPieces))*hp.randomWeightMultiplier;
        else
            error("Noise type for weight initialization not implemented\n")
        end     
        
    end
    %initialize biases (no weight sharing, i.e., separate biases for all layers
    for layer=1:(length(hp.layerSizes)-1)
        if(strcmp(hp.initializationNoiseType,'uniform'))
            params.biases{layer}=(rand(1,hp.layerSizes(layer+1),numPieces)-0.5)*hp.randomWeightMultiplier;
        elseif(strcmp(hp.initializationNoiseType,'normal'))
            params.biases{layer}=(randn(1,hp.layerSizes(layer+1),numPieces))*hp.randomWeightMultiplier;
            
        else
            error("Noise type for weight initialization not implemented\n")
        end 
    end
    %experimental: creativity sampling (extra class-conditional bias)
    if hp.useCreativitySampling
        for layer=1:(length(hp.layerSizes)-2)
            if(strcmp(hp.initializationNoiseType,'uniform'))
                params.classBiases{layer}=(rand(hp.creativitySamplingNumClasses,hp.layerSizes(layer+1),numPieces)-0.5)*hp.randomWeightMultiplier;
            elseif(strcmp(hp.initializationNoiseType,'normal'))
                params.classBiases{layer}=(randn(hp.creativitySamplingNumClasses,hp.layerSizes(layer+1),numPieces))*hp.randomWeightMultiplier;
                
            else
                error("Noise type for weight initialization not implemented\n")
            end 
        
        end    
    end
end

%backpropagate errors to get gradients
function [gradient,biasGradient,classBiasGradient]=backProp(hp,params,predictions,targets,classes=false)
    errorDerivatives={};
    gradient={};
    biasGradient={};
    classBiasGradient={};
    %last layer uses given cost as error
    for layer=length(hp.layerSizes)-1:-1:1
        %We use weight sharing for encoding and decoding functions
        %i.e., in layers that are part of decoding function, we use transposed encoding weights
        encode=true;
        weightLayer=layer;
        if layer>(length(hp.layerSizes)-1)/2
            encode=false;
            weightLayer=1+(length(hp.layerSizes)-1)-layer;
        end
        
        
        %last layer: calculate error gradient
        if(layer==length(hp.layerSizes)-1)
            if strcmp(hp.unitTypes{layer+1},'sigmoid')
                dCostdz=predictions{layer+1}-targets;
            else
                error('ERROR: last layer must be sigmoid units, and log(y) or log(1-y) error function.\n')
            end
    
        else
            %other layers use back-propagated cost
            %regardless of neuron type: 
            %   dcost/da=errorDerivatives{layer}
            if (strcmp(hp.unitTypes{layer+1},'relu'))
                %da/dz = 1 if z>0
                %        0 else
                dCostdz=errorDerivatives{layer+1}.*(predictions{layer+1}>0);
            elseif (strcmp(hp.unitTypes{layer+1},'sigmoid'))
                %da/dz = a*(1-a)
                dCostdz=errorDerivatives{layer+1}.*(predictions{layer+1}.*(1-predictions{layer+1}));
            else
                error('ERROR: Unit type %s not implemented yet\n',hp.unitTypes{layer});
            end
        end
        %calculate total gradient for weights + biases
        %just calculate outer product lowerLayer x cost@higherLayer
        gradient{layer}=predictions{layer}'*dCostdz; %'
        biasGradient{layer}=ones(1,size(predictions{length(hp.layerSizes)-1},1))*dCostdz;
        
        %experimental: creativity sampling (extra class-conditional bias)
        if hp.useCreativitySampling && layer<length(hp.layerSizes)-1
            classBiasGradient{layer}=classes'*dCostdz; %';
        end
        
        %propagate error downwards to previous layer (=dCost/da for previous layer)
        
        if encode
            errorDerivatives{layer}=dCostdz*params.weights{layer}'; %'
        else %decoding function, use transposed encoding weights
            errorDerivatives{layer}=dCostdz*params.weights{weightLayer};
        end

    end
    
end

%%%forward propagate data through autoencoder (encoding + decoding function)
function activations=forwardProp(hp,params,data,classes=false)
    activations={};
    activations{1}=data;
    #go through all layers
    for layer=1:length(hp.layerSizes)-1
        %We use weight sharing for encoding and decoding functions
        %i.e., in layers that are part of decoding function, we use transposed encoding weights
        encode=true;
        weightLayer=layer;
        if layer>(length(hp.layerSizes)-1)/2
            encode=false;
            weightLayer=1+(length(hp.layerSizes)-1)-layer;
        end
        if hp.loglevel>3
            fprintf("Forward prop layer %i (weights from layer %i), unit type %s\n",layer,weightLayer,hp.unitTypes{layer+1});
        end
        
        if encode
            activations{layer+1}=activations{layer}*params.weights{layer}+repmat(params.biases{layer},size(data,1),1);
        else %decoding, use transposed encoding weights (but decoding biases)
            activations{layer+1}=activations{layer}*params.weights{weightLayer}'+repmat(params.biases{layer},size(data,1),1); %'
        end
        
        %experimental: creativity sampling (extra class-conditional bias)
        if hp.useCreativitySampling && layer<length(hp.layerSizes)-1
            activations{layer+1}+=classes*params.classBiases{layer};
        end
        
        
        %in case of unit with activation function, apply it!
        if(strcmp(hp.unitTypes{layer+1},'relu'))
            activations{layer+1}=max(0,activations{layer+1});
        elseif (strcmp(hp.unitTypes{layer+1},'sigmoid'))
            activations{layer+1}=sigmoid(activations{layer+1});
        %elseif (strcmp(hp.unitTypes{layer+1},'maxout'))
        %    %dont do anything here
        else
            error('ERROR: Unit type %s not implemented yet\n',hp.unitTypes{layer});
        end
    end
    
end

function data=corruptData(hp,data)
    #add salt&pepper noise
    [index,garbage]=find((rand(size(data,1)*size(data,2),1)<hp.corruptInput));
    data(index)=rand(size(index,1),1)>0.5;
end
%helper function for experimental creativity sampling 
function matrix=createOneHotRepresentationForClasses(hp,classes)
    matrix=zeros(size(classes,1),hp.creativitySamplingNumClasses);
    for i=1:size(matrix,1)
        matrix(i,classes(i)+1)=1;
    end
end

function params=trainAE(hp, data,params,validationData,fitness,classes=0)
    % Trains an autoencoder 
    % input:
    %   hp: hyperparams
    %   data: training data, matrix of size (numExamples,exampleSize)
    unlink ("/tmp/stopOctave")%workaround for "octave not stopping bug"
    hp.samplingTemp=1.0;
    hp.sampleHiddensWithHigherTemperature=0;
    
    %experimental feature creative sampling
    if hp.useCreativitySampling
        if length(classes)==1
            found=false;attempts=0;
            numClusters=hp.creativitySamplingNumClasses;
            while !found
                try
                    attempts+=1;
                    fprintf("Trying clustering with %i clusters\n",numClusters)
                    [idx, centers] = kmeans (data*1.0, numClusters);
                    found=true;
                    if hp.loglevel>1
                        figure(5555);
                        hist(idx,numClusters);
                        title("Cluster histogram");
                        refresh();
                    end
                catch
                    msg = lasterror.message;
                    fprintf(msg)
                    if numClusters==1
                        found=true;
                        idx=ones(size(data,1),1);
                    else 
                        fprintf("...trying again width fewer clusters\n")
                        numClusters=max(1,numClusters-1);
                    end
                    
                end_try_catch
            end
            %reset maximum number of classes according to clustering
            hp.creativitySamplingNumClasses=numClusters;
            
            %re-initilize model with correct number of classes
            params=initializeAE(hp);
            
            %a bit hacky: store number of clusters in params field (hp will not be available to sampling)
            params.classesAfterClustering=numClusters;
            
            %calculate + store activation rate of clusters (again, way of storing is hacky)
            params.classesActivationProportion=sum(repmat(idx,1,numClusters)==repmat(1:numClusters,size(idx,1),1))/size(idx,1);
            
            classes=idx-1; %next steps expect classes from 0...i (not 1...i+1)
            %error("Creativitvy sampling is on, but no classes given. Todo: implement some clustering\n")
        end
        classes=createOneHotRepresentationForClasses(hp,classes);
        
        %some checks
        if hp.uniqueData
            error("Unique + Creativity does not work together")
        end 

    else
        classes=zeros(size(data,1),1);
    end
    
    %make unique dataset 
    if hp.uniqueData
        oldsize=size(data,1);
        data=unique(data,'rows');
        fprintf("Deleted %i duplicates in dataset,  %i examples remaining.\n",oldsize-size(data,1),size(data,1));
    end
    
    %split off validation set, if required
    if hp.validationSet>0.0
        %shuffle data first
        [garbage,order]=sort(rand(1,size(data,1)));
        data=data(order,:);
        numItemsValidationSet=min(floor(size(data,1)*hp.validationSet),5*hp.batchSize);
        validationData=data(1:numItemsValidationSet,:);
        if hp.corruptInput>0
            validationDataCorrupted=corruptData(hp,validationData);
        else
            validationDataCorrupted=validationData;
        end
        data=data(numItemsValidationSet+1:end,:);
        
        if hp.useCreativitySampling
            validationClasses=classes(1:numItemsValidationSet,:);
            classes=classes(numItemsValidationSet+1:end,:);
        else
            validationClasses=false;
        end
    end
    
    %use configured batch size, or 20% of data
    hp.batchSize=min(hp.batchSize,floor(size(data,1)/5.0));
    %learning hyper parameters
    currentAlpha=hp.alphaInitial;
    
    %backwards compatibility: set some control variables, if not configured
    if !isfield(hp,"trackIdentityFunction")
        hp.trackIdentityFunction=0;
    end
    if !isfield(hp,"minUpdates")
        hp.minUpdates=0;
    end
    %main loop control variables
    converged=0;
    numEpochs=0;
    numUpdates=0;
    bestParams=false;
    currentMomentum=hp.momentumInitial;
    
    %stats, data fields
    error=[];
    smoothedError=[];
    errorValidationSet=[];
    smoothedErrorValidationSet=[];
    lowestReconErr=Inf;
    lowestReconErrValidationSet=Inf;
    identityFunctionIntensity=[];
    
    %fields to remember last updates (for momentum)
    lastUpdateWeights={};lastUpdateBiases={};
    for layer=1:(length(hp.layerSizes)-1)/2
        lastUpdateWeights{layer}=zeros(size(params.weights{layer}));
    end
    for layer=1:(length(hp.layerSizes)-1)
        lastUpdateBiases{layer}=zeros(size(params.biases{layer}));
    end
    if hp.useCreativitySampling
        for layer=1:(length(hp.layerSizes)-2)
            lastUpdateClassBiases{layer}=zeros(size(params.classBiases{layer}));
        end
    end
    %how many batches shall be used to calculate the reconstruction error
    numBatchesReconErr=3;
    dataForReconErr=data(1:numBatchesReconErr*hp.batchSize,:);
    dataCorruptedForReconErr=corruptData(hp,dataForReconErr);
    classesForReconErr=classes(1:numBatchesReconErr*hp.batchSize,:);
    %main loop, epochs
    while ((!converged) && ( numEpochs < hp.maxEpochs ))
        numEpochs+=1;
        
        if (hp.loglevel>1)
            fprintf("Epoch: %i\n",numEpochs)
        end
        
        %%% Octave bug: cannot stop exectution on CTRL-C when using multithreaded BLAS library (probably specific to work pc)
        %workaround: check existence of file /tmp/stopOctave
        if exist("/tmp/stopOctave","file")==2
            converged=1;
            fprintf("Aborting....\n");
        end
        
        %shuffle batches
        [garbage,shuffleindex]=sort(rand(1,size(data,1)));
        data=data(shuffleindex,:);
        if hp.useCreativitySampling
            classes=classes(shuffleindex,:);
        end
        
        %add noise, if using denoising autoencoder 
        if hp.corruptInput>0
            dataCorrupted=corruptData(hp,data);
        else
            dataCorrupted=data;
        end
        
        %%loop batches
        for batch=1:floor(size(data,1)/hp.batchSize)
            numUpdates+=1;
            batchStart=(batch-1)*hp.batchSize+1;
            batchEnd=batchStart+hp.batchSize-1;
        
            %forward prop through AE (i.e., encode + decode)
            hp.dropoutmode="train";
            activations=forwardProp(hp,params,dataCorrupted(batchStart:batchEnd,:),classes(batchStart:batchEnd,:));
            
            %caluclate gradient (backprop)
            [weightGradient,biasGradient,classBiasGradient]=backProp(hp,params,activations,data(batchStart:batchEnd,:),classes(batchStart:batchEnd,:));
            
            %combine gradients for encoding and decoding functions
            gradient={};
            for layer=1:length(hp.layerSizes)-1
                if layer<=(length(hp.layerSizes)-1)/2
                    gradient{layer}=weightGradient{layer};
                else
                    weightLayer=1+(length(hp.layerSizes)-1)-layer;
                    gradient{weightLayer}+=weightGradient{layer}'; %'
                end
            end
            
            %apply gradient update 
            for layer=1:(length(hp.layerSizes)-1)/2
                if(strcmp(hp.unitTypes{layer},'maxout'))
                    pieces=hp.maxoutPieces;
                else
                    pieces=1;
                end
                for piece=1:pieces
                    %calulate update for weights (including momentum)
                    lastUpdateWeights{layer}(:,:,piece)=currentMomentum*lastUpdateWeights{layer}(:,:,piece)+(1.0/(hp.batchSize))*gradient{layer}(:,:,piece);
                    
                    if hp.weightDecay>0.0
                        lastUpdateWeights{layer}(:,:,piece)-=hp.weightDecay*params.weights{layer}(:,:,piece);
                    end
                    
                    %apply update for weights
                    params.weights{layer}(:,:,piece)-=currentAlpha.*lastUpdateWeights{layer}(:,:,piece);
                                     
                end
            end  %weight udpate   
            %apply bias update  
            for layer=1:(length(hp.layerSizes)-1)
                if(strcmp(hp.unitTypes{layer},'maxout'))
                    pieces=hp.maxoutPieces;
                else
                    pieces=1;
                end
                for piece=1:pieces
                    %calculate update for biases
                    lastUpdateBiases{layer}(:,:,piece)=(currentMomentum*hp.momentumForBiases)*lastUpdateBiases{layer}(:,:,piece)+biasGradient{layer}(:,:,piece)*(1.0/(hp.batchSize));
                    
                    %apply update for biases
                    params.biases{layer}(:,:,piece)-=(hp.alphaBiasMultiplier*currentAlpha).*lastUpdateBiases{layer}(:,:,piece);
                end
            end  %bias udpate
            
            %experimental
            if hp.useCreativitySampling 
                %apply creativity bias update  
                for layer=1:(length(hp.layerSizes)-2)
                    %calculate update for biases
                    lastUpdateClassBiases{layer}=(currentMomentum*hp.momentumForBiases)*lastUpdateClassBiases{layer}+classBiasGradient{layer}*(1.0/(hp.batchSize));
                    
                    %apply update for biases
                    params.classBiases{layer}-=(hp.alphaBiasMultiplier*currentAlpha).*lastUpdateClassBiases{layer};
                end  %creativity bias udpate
            end %creativity sampling
            
        end %end batches
        
        %calculate stats
        if mod(numEpochs,hp.calculateStatsEachNthEpoch)==0
        
            if hp.useCreativitySampling && hp.loglevel>2
                for layer=1:(length(hp.layerSizes)-2)
                    figure(211+layer)
                    hist(params.classBiases{layer}(:))
                    tmpMin=((round(min(params.classBiases{layer}(:))*1000))/1000.0);
                    tmpMax=((round(max(params.classBiases{layer}(:))*1000))/1000.0);
                    tmpMean=((round(mean(params.classBiases{layer}(:))*1000))/1000.0);        
                    name=strcat("Creativity Biases layer ",mat2str(layer),"min:",mat2str(tmpMin)," mean:",mat2str(tmpMean)," max:",mat2str(tmpMax));
                    title(name);
                end
                refresh()
            end
        
            %calculate identityFunctionIntensity
            if hp.trackIdentityFunction
                %tmpSparsity=0;
                %for layer=1:floor(length(hp.layerSizes)/2.0)
                %    tmpSparsity+=mean(getSparsity(params.weights{layer}')); %'
                %end
                tmpSparsity=getSparsityDirect(params,hp,numExamples=100);
                identityFunctionIntensity=[identityFunctionIntensity,tmpSparsity];
            end
            
            %calculate error
            hp.dropoutmode="test";
            reconstructions=forwardProp(hp,params,dataCorruptedForReconErr,classesForReconErr){length(hp.layerSizes)};
            errorR=sum(calculateCost(hp, reconstructions,dataForReconErr));
            
            errorTotal=errorR;
            if hp.loglevel>1
                fprintf("  current loss: %f\n",errorTotal);
            end
            %save params of best epoch for later use
            if errorTotal<lowestReconErr
                bestParams=params;
                lowestReconErr=errorTotal;
                bestEpoch=numEpochs;
            end
            error=[error,errorTotal];
            
            %visualize reconstructions
            if hp.loglevel>1 &&  mod(numEpochs,hp.showStatsEachNthEpoch)==0
                if hp.mnistFlag
                    for(fig=1:5)    
                        figure(211+fig)
                        showDigit(reconstructions(fig,:))
                        name=strcat("Reconstruction: ",mat2str(fig));
                        title(name);
                    end
                else
                    figure(21101)
                    imshow(reconstructions(1:min(100,hp.batchSize),:))
                    name=strcat("Reconstructions ");
                    title(name);
                
                end
                refresh();
            end
            
            %calcualte validation set error, if configured
            if hp.validationSet>0.0
                reconstructions=forwardProp(hp,params,validationDataCorrupted,validationClasses){length(hp.layerSizes)};
                errorR=sum(calculateCost(hp, reconstructions,validationData));
                
                errorTotalValidationSet=errorR;
                errorValidationSet=[errorValidationSet,errorTotalValidationSet];
                if errorTotalValidationSet<lowestReconErrValidationSet
                    bestParams=params;
                    lowestReconErrValidationSet=errorTotalValidationSet;
                    bestEpoch=numEpochs;
                end
            
            end
            
            
            %calculate smoothed objectives
            smoothedError=[];
            smoothedErrorValidationSet=[];
                
            if floor(numEpochs/hp.calculateStatsEachNthEpoch)<20
                numSmooths=2;
            elseif floor(numEpochs/hp.calculateStatsEachNthEpoch)<30
                numSmooths=3;
            else
                numSmooths=4;
            end
            if floor(numEpochs/hp.calculateStatsEachNthEpoch)>numSmooths
                for e=1:floor(numEpochs/hp.calculateStatsEachNthEpoch)-numSmooths
                    smoothE=0;
                    smoothEValidationSet=0;
                    for s=-numSmooths:1:numSmooths
                        if e+s<1 %left default value
                            smoothE+=error(1);
                            if hp.validationSet>0.0
                                smoothEValidationSet+=errorValidationSet(1);
                            end
                        else
                            smoothE+=error(e+s);
                            if hp.validationSet>0.0
                                smoothEValidationSet+=errorValidationSet(e+s);
                            end
                        end
                    end
                    smoothedError=[smoothedError,smoothE/(2*numSmooths+1)];
                    if hp.validationSet>0.0
                        smoothedErrorValidationSet=[smoothedErrorValidationSet,smoothEValidationSet/(2*numSmooths+1)];
                    end
                    
                end
            end
        
        end
        
        %%%check for convergence
        if (mod(numEpochs,hp.calculateStatsEachNthEpoch)==0
            && (numEpochs>hp.minEpochs && numUpdates>hp.minUpdates))
            numPoints=length(smoothedError);
            totalGain=smoothedError(1)-smoothedError(end);
            if totalGain>0
                lastGain=smoothedError(ceil(2/3*numPoints))-smoothedError(end);
                relativeGain=lastGain/totalGain;
                threshold=hp.breakOnRelativeGain;
                if(relativeGain<threshold)
                    converged=1;
                    if hp.loglevel>0
                        fprintf("Relative gain of last interval: %.2f, Stopping...\n",relativeGain)
                    end
                end
            else
                if hp.loglevel>0
                    fprintf("Training seems to diverge, Stopping...\n")
                end
                converged=1;
            end
            if hp.validationSet>0.0
                totalGainValidationSet=smoothedErrorValidationSet(1)-smoothedErrorValidationSet(end);
                if totalGainValidationSet>0
                    lastGainValidationSet=smoothedErrorValidationSet(ceil(2/3*numPoints))-smoothedErrorValidationSet(end);
                    relativeGainValidationSet=lastGainValidationSet/totalGainValidationSet;
                    threshold=hp.breakOnRelativeGain;
                    if(relativeGainValidationSet<threshold)
                        converged=1;
                        if hp.loglevel>0
                            fprintf("ValidationSet: Relative gain of last 3rd: %.2f, Stopping...\n",relativeGainValidationSet)
                        end
                    end
                else
                    if hp.loglevel>0
                        fprintf("Training seems to diverge, Stopping...\n")
                    end
                    converged=1;
                end
            end
            reconErrPerBit=errorR/size(params.weights,1);
            if reconErrPerBit<0.01
                fprintf("Reconstruction error very small, stopping...\n")
                converged=1;
            end
            
            if hp.validationSet>0.0
                if ((smoothedErrorValidationSet(end)-smoothedError(end))/smoothedError(end))>0.1
                    fprintf("Gap of error / validation error too big, stopping...\n")
                    converged=1;
                end
            end
        end
        
        
        
        %%%display stats etc
        if mod(numEpochs,hp.showStatsEachNthEpoch)==0  && hp.loglevel>1
            
            hold off;
            figure(2115);
            plot(1:hp.calculateStatsEachNthEpoch:numEpochs,error);
            title("Loss")
            
            %show smoothed objective
            if floor(numEpochs/hp.calculateStatsEachNthEpoch)>numSmooths
                figure(2115);
                hold on;
                plot(1:hp.calculateStatsEachNthEpoch:numEpochs-numSmooths*hp.calculateStatsEachNthEpoch,smoothedError,'r');
                hold off;
            end
            
            %show validation set objective for D
            if hp.validationSet>0
                figure(2118);
                hold on;
                plot(1:hp.calculateStatsEachNthEpoch:numEpochs-numSmooths*hp.calculateStatsEachNthEpoch,smoothedErrorValidationSet,'g');
                hold off;
            end
            
            %show identityFunctionIntensity
            if hp.trackIdentityFunction
                figure(21171);
                plot(1:hp.calculateStatsEachNthEpoch:numEpochs,identityFunctionIntensity,'r');
                title("Identity function Intensity (input/output correlation)")
            end
            showFeatures(hp,params);
            
            %%%% Produce some samples for visual check
            if hp.loglevel>2
                if hp.mnistFlag
                    if hp.useCreativitySampling
                        if hp.creativitySamplingMixNClasses==2
                            %create classes for all combinations
                            allClasses=zeros(100,10);
                            for i=1:100
                                allClasses(i,ceil(i/10.0))=1;
                                allClasses(i,mod(i-1,10)+1)=1;
                            end
                            %create samples with all class combinations
                            hp2=hp;
                            hp2.batchSize=100;
                            samples=sampleAE(hp2,params,hp2.batchSize,false,allClasses);
                            figure(211221)
                            imshow(buildTenByTenMnistCanvas(samples))
                            title("Mixed-Class samples (row i, col j -> classes i+j mixed)")
                            
                            %create classes for all combinations
                            allClasses=zeros(100,10);
                            for i=1:100
                                allClasses(i,ceil(i/10.0))=1;
                            end
                            %create samples with all class combinations
                            hp2=hp;
                            hp2.batchSize=100;
                            samples=sampleAE(hp2,params,hp2.batchSize,false,allClasses);
                            figure(211222)
                            imshow(buildTenByTenMnistCanvas(samples))
                            title("Single-Class samples (rows i -> class i)")
                        end
                    else
                    
                        %create samples with all class combinations
                        hp2=hp;
                        hp2.batchSize=100;
                        samples=sampleAE(hp2,params,hp2.batchSize,false);
                        figure(211222)
                        imshow(buildTenByTenMnistCanvas(samples))
                        title("Random Samples")
                        %for(fig=1:5)                            
                        %    figure(1000+fig)
                        %    showDigit(samples(fig,:))
                        %    name=strcat("Generator Sample ",mat2str(fig));
                        %    title(name);
                        %end
                    end
                else
                    samples=sampleAE(hp,params,hp.batchSize,data(1:hp.batchSize,:));
                    figure(211001)
                    imshow(samples(1:min(100,hp.batchSize),:))
                    name=strcat("Generator Samples ");
                    title(name);
                
                end
                refresh();
            end %show samples
        end %stats
        
        %use best seen params
        if converged
            fprintf("Using best params (epoch %d)",bestEpoch)
            params=bestParams;
        end
    end %main loop
    
    %save identity function intensity
    if hp.trackIdentityFunction
        filename=strcat("/tmp/","/SP-",mat2str(hp.corruptInput),"-",mat2str(round(rand*10000)),mat2str(round(rand*10000)),".m")
        save(filename,"identityFunctionIntensity")
    end
end
function canvas=buildTenByTenMnistCanvas(samples)
    %canvas: 10 rows, 10 cols, each cell 28*28px. Spacing: 1px around each image (i.e, 30x30px)
    sl=30;%side length including borders
    canvas=ones(10*sl,10*sl);
    for i=1:100
        yfrom=sl*floor((i-1)/10.0)+2;
        yto=sl*ceil(i/10.0)-1;
        xfrom=sl*mod(i-1,10)+2;
        xto=sl*(mod(i-1,10)+1)-1;
        %fprintf("img %i, position: x:%i:%i y:%i:%i\n",i,xfrom,xto,yfrom,yto)
        canvas(yfrom:yto,xfrom:xto)=reshape(samples(i,:),28,28);
    end
end
%gradient checking for autoencoder
function ok=checkGradient(hp,params,data,classes=0)
    data=data(1:min(size(data,1),400),:);
    if hp.useCreativitySampling
        classes=classes(1:min(size(classes,1),400),:);
        classes=createOneHotRepresentationForClasses(hp,classes);
    end
    %% calculates the numerical gradient and compares it to the analytical gradient
    %same steps as in training, see there for details
    epsilon=0.00001;
    ratioLimit=1.1;
    paramsOrig=params;
    numberWeights=5;
    ok=1;
    fprintf("==== Gradient checking ====\n");
    %prepare data
    
    dataCorrupted=corruptData(hp,data);
    %encode + decode
    activations=forwardProp(hp,params,dataCorrupted,classes);
            
    %caluclate analytical gradient
    [weightGradient,biasGradient,classBiasGradient]=backProp(hp,params,activations,data,classes);
            
    %combine analytical gradients for encoding and decoding functions
    gradient={};
    for layer=1:length(hp.layerSizes)-1
        if layer<=(length(hp.layerSizes)-1)/2
            gradient{layer}=weightGradient{layer};
        else
            weightLayer=1+(length(hp.layerSizes)-1)-layer;
            gradient{weightLayer}+=weightGradient{layer}'; %'
        end
    end
    
    
    %calculate original error
    errorOriginal=sum(calculateCost(hp, activations{length(hp.layerSizes)},data));
    
    %check weights gradient for all layers    
    fprintf("===Weights===\n")
    for layer=1:(length(hp.layerSizes)-1)/2
        fprintf("--- Layer %i --\n",layer);   
        whichWeights=ceil(rand(1,numberWeights)*length(params.weights{layer}(:)));
        
            
        for weightindex=whichWeights
        
            fprintf(" -- weight %i --\n",weightindex);
            
            params=paramsOrig;
            %modified params
            tmpParams=params.weights{layer};
            tmpParams(weightindex)+=epsilon;
            params.weights{layer}=tmpParams;
            
            %encode + decode
            activations=forwardProp(hp,params,dataCorrupted,classes);
    
            
            %calculate new error
            errorNumerical=sum(calculateCost(hp, activations{length(hp.layerSizes)},data));
            
            numericalGradient=((errorNumerical-errorOriginal)/epsilon);
            
            analyticalGradient=gradient{layer}(weightindex)/min(size(data,1),400);
            
            ratio=numericalGradient/analyticalGradient;
            
            
            fprintf("     R: numerical gradient: %f analytical gradient %f (%f)",numericalGradient,analyticalGradient,ratio)
            if (abs(numericalGradient)<0.00001)
                fprintf("(cant decide, gradient very small)\n");
            elseif (ratio>ratioLimit || ratio<(1/ratioLimit))
                fprintf("       ERROR, this one seems off!\n");
                ok=0;
            else
                fprintf(" looks OK\n")
            end
            
            
        end %weights loop
    end %weights gradients all layers loop
    %check bias gradient   
    fprintf("===Biases===\n")
    for layer=1:(length(hp.layerSizes)-1)
        fprintf("--- Layer %i --\n",layer);   
        whichWeights=ceil(rand(1,numberWeights)*length(params.biases{layer}(:)));
        
        for weightindex=whichWeights
        
            fprintf(" -- bias %i --\n",weightindex);
            
            params=paramsOrig;
            %modified params
            tmpParams=params.biases{layer};
            tmpParams(weightindex)+=epsilon;
            params.biases{layer}=tmpParams;
            
            %encode + decode
            activations=forwardProp(hp,params,dataCorrupted,classes);
    
            
            %calculate new error
            errorNumerical=sum(calculateCost(hp, activations{length(hp.layerSizes)},data));
            
            numericalGradient=((errorNumerical-errorOriginal)/epsilon);
            
            analyticalGradient=biasGradient{layer}(weightindex)/min(size(data,1),400);
            
            ratio=numericalGradient/analyticalGradient;
            
            
            fprintf("     R: numerical gradient: %f analytical gradient %f (%f)",numericalGradient,analyticalGradient,ratio)
            if (abs(numericalGradient)<0.00001)
                fprintf("(cant decide, gradient very small)\n");
            elseif (ratio>ratioLimit || ratio<(1/ratioLimit))
                fprintf("       ERROR, this one seems off!\n");
                ok=0;
            else
                fprintf(" looks OK\n")
            end
            
            
        end %weights loop
    end %bias gadients all layers loop
    
    
    %check creativity bias gradient 
    if hp.useCreativitySampling  
        fprintf("===Creativity Biases===\n")
        for layer=1:(length(hp.layerSizes)-2)
            fprintf("--- Layer %i --\n",layer);   
            whichWeights=ceil(rand(1,numberWeights)*(length(params.classBiases{layer}(:)-1)));
            
            for weightindex=whichWeights
            
                fprintf(" -- bias %i --\n",weightindex);
                
                params=paramsOrig;
                %modified params
                tmpParams=params.classBiases{layer};
                tmpParams(weightindex)+=epsilon;
                params.classBiases{layer}=tmpParams;
                
                %encode + decode
                activations=forwardProp(hp,params,dataCorrupted,classes);
        
                
                %calculate new error
                errorNumerical=sum(calculateCost(hp, activations{length(hp.layerSizes)},data));
                
                numericalGradient=((errorNumerical-errorOriginal)/epsilon);
                
                analyticalGradient=classBiasGradient{layer}(weightindex)/min(size(data,1),400);
                
                ratio=numericalGradient/analyticalGradient;
                
                
                fprintf("     R: numerical gradient: %f analytical gradient %f (%f)",numericalGradient,analyticalGradient,ratio)
                if (abs(numericalGradient)<0.00001)
                    fprintf("(cant decide, gradient very small)\n");
                elseif (ratio>ratioLimit || ratio<(1/ratioLimit))
                    fprintf("       ERROR, this one seems off!\n");
                    ok=0;
                else
                    fprintf(" looks OK\n")
                end
                
                
            end %weights loop
        end %bias gadients all layers loop
    end
end
function showSamples(hp,params,fignum=0 ) 
    noise=sampleNoiseForGenerator(hp);
    generatorActivations=forwardPropG(hp,params,noise);
    figure(1000)
    for fig=1:hp.batchSize
        showDigit(generatorActivations{length(hp.generator.layerSizes)}(fig+fignum,:))
        name=strcat("Generator Sample ",mat2str(fig));
        title(name);    
        refresh()
        sleep(1)
    end
end
function showFeatures(hp,params)
    if hp.mnistFlag
        fprintf("Todo: print mnist features <---\n")
    else
        for layer=1:(length(hp.layerSizes)-1)/2
            figure(9900+layer)
            tmpMin=((round(min(params.weights{layer}(:))*1000))/1000.0);
            tmpMax=((round(max(params.weights{layer}(:))*1000))/1000.0);
            tmpMean=((round(mean(params.weights{layer}(:))*1000))/1000.0);
            tmpWeights=(params.weights{layer}.*(0.5/max(tmpMax,abs(tmpMin))).+0.5);
            imshow(tmpWeights');
            
            name=strcat("Features, min: ",mat2str(tmpMin)," max:",mat2str(tmpMax), " mean:",mat2str(tmpMean));
            title(name);
            
            %figure(9900+layer);
            %hist(tmpWeights(:),20)
        end
     end
end
function samples=sampleAE(hp,params,numExamples,startingExamples,classes=false)
    if hp.loglevel>0
        fprintf("Sampling Autoencoder...\n")
    end
    hp.batchSize=numExamples;
    samples=startingExamples;
    
    if  hp.sampleFromRandom
        samples=rand(numExamples,hp.layerSizes(1))>0.5;
    end
    
    %experimental
    if hp.useCreativitySampling
        if(isfield(params,"classesAfterClustering")) %re-insert information into hyperparams
            %actual number of classes used for clustering during training
            hp.creativitySamplingNumClasses=params.classesAfterClustering;
            %number of classes to mix
            hp.creativitySamplingMixNClasses=max(min(hp.creativitySamplingMixNClasses,params.classesAfterClustering-1),1);
            %actual proportion of classes
            hp.creativitySamplingClassesActivationProportion=params.classesActivationProportion;
        else
            hp.creativitySamplingClassesActivationProportion=repmat(1/hp.creativitySamplingNumClasses,1,hp.creativitySamplingNumClasses);
        end
        if length(classes)==1
            %initializes classes matrix
            classes=zeros(numExamples,hp.creativitySamplingNumClasses);
            %use single class for <hp.creativitySamplingMixThisProportionOfSamples> per cent of samples
            %activate according to class actvation proportions in clustering
            examplesPerClass=floor(hp.creativitySamplingClassesActivationProportion*numExamples*(1-hp.creativitySamplingMixThisProportionOfSamples));
            for i=1:hp.creativitySamplingNumClasses
                from=1+sum(examplesPerClass(1:i))-examplesPerClass(1);
                to=sum(examplesPerClass(1:i));
                classes(from:to,i)=1;
            end
            
            %use mixed classes for other <hp.creativitySamplingMixThisProportionOfSamples> per cent of samples
            count=floor(numExamples*hp.creativitySamplingMixThisProportionOfSamples);
            from=floor((1-hp.creativitySamplingMixThisProportionOfSamples)*numExamples)+1;
            for i=1:hp.creativitySamplingMixNClasses
                newactivations=repmat(hp.creativitySamplingClassesActivationProportion,count,1)>rand(count,hp.creativitySamplingNumClasses);
                classes(from:end,:)=or(classes(from:end,:),newactivations);
            end

        end
    end
    
    for round=1:hp.numSamplingRounds
        samples=forwardProp(hp,params,corruptData(hp,samples),classes){length(hp.layerSizes)};
    end %sampling rounds corruption sampling
end


function [samples,metadata]=sampleAEBinary(hp,params,numExamples,population,classes=0)
    samples=sampleAE(hp,params,numExamples,population,classes)>rand(numExamples,hp.layerSizes(1));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% script part
function model=testAE()
    more off;
    %load the mnist data
    images=loadMNISTImages("/home/malte/CodeTex/code/data/mnist/train-images-idx3-ubyte")';
    labels=loadMNISTLabels("/home/malte/CodeTex/code/data/mnist/train-labels-idx1-ubyte");
    hp.mnistFlag=1;
    
    numExamples=3000;
    data=images(1:min(size(images,1),numExamples),:);
    classes=labels(1:min(size(labels,1),numExamples),:);
    % ----- Hyper Params----- 
    %process
    doGradientCheck=0;
    %stats
    if (exist("instance", "var") == 0)
        instance="1";
    end
    hp.instance=instance;
    if (exist("savePath", "var") == 0)
        savePath="/tmp/";
    end
    if (exist("seed", "var") == 0)
        seed=-1; %-1=don't used fixed seed, anything else: use as seed
    end
    if seed!=-1
        rand("state",seed);
    end
    hp.savePath=savePath;
    %general hyperparams
    hp.maxEpochs=300;
    hp.minEpochs=20;
    hp.calculateStatsEachNthEpoch=1;
    hp.randomWeightMultiplier=0.01;
    hp.alphaInitial=.01;  
    hp.batchSize=128;
    hp.loglevel=3;
    hp.momentumInitial=0.25;
    hp.momentumForBiases=0;
    hp.weightDecay=0.00001;
    hp.initializationNoiseType="normal";
    hp.alphaBiasMultiplier=2.0;
    hp.validationSet=0.0;
    hp.numSamplingRounds=50;
    hp.uniqueData=false;
    hp.corruptInput=0.5;%0.5 works good with mnist
    hp.useDropout=false;
    hp.showStatsEachNthEpoch=1;
    hp.sampleFromRandom=true;
    hp.breakOnRelativeGain=0.05;
    #number of layers must be odd, sizes must be symmetric to middle layer,size of first (+last) layer must match data size
    hp.layerSizes=[784,500,500,500,784];
    #hp.layerSizes=[4,2,4];
    #number of layers must be odd, types must be symmetric to middle layer, type of first (+last) layer must match data type
    hp.unitTypes={'sigmoid';'relu';'relu';'relu';'sigmoid'};
    
    
    hp.useCreativitySampling=0;
    hp.creativitySamplingMixThisProportionOfSamples=0.5;
    hp.creativitySamplingMixNClasses=2; #sample from N classes (instead of 1)
    hp.creativitySamplingNumClasses=max(classes)+1; #number of classes
    classes=0
    % ------ Initial Params -----
    params=initializeAE(hp,false,false,false);
    if doGradientCheck
        if !checkGradient(hp,params, data,classes)
            error("Gradient check failed\n")
        end
    end
    %creativity sampling check
    if ! hp.useCreativitySampling
        classes=0;
    end
    %
    % Train
    hp
    params=trainAE(hp, data,params,false,false,classes);
    
    %samples=sampleCAE(hp,params,hp.batchSize,rand(hp.batchSize,hp.layerSizes(1))>0.5);
    %showSamples(hp,params);
    %sleep(2)
    model.hp=hp;
    model.params=params;
    
end
