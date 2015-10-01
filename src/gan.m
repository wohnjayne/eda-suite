% EDA suite - gan.m
% Copyright (C) <2010-2015> Malte Probst
% This file is part of the EDA suite.
%    The EDA suite is free software under the terms of the GNU General
%    Public License, see eda.m for full licence text, or <http://www.gnu.org/licenses/>.


%Matlab / Octave file
%%%%%%%%%%%%%%%%%%%%%%%%%% function definitions
%%%%helpers
if strcmp(version,"3.2.4")!=1
    graphics_toolkit ("gnuplot")
end
function y=sigmoid(x,temp=1.0)
    y=1.0 ./(1.0+exp(-x/temp));
end
function y=rectified(x)
    y=max(0,x);
end
function showDigit(flatData)
    imshow(reshape(flatData,28,28))
end
function showLayer(fignum,data,mytitle)
    figure(fignum)
    imshow(data)
    title(mytitle)
end
function metaData=calcuateStatsForActivations(hp,metaData,whichModel,layerNum,activations)
    if strcmp(whichModel,"discriminator")
        offset=100;
        layerOffset=-1;
    else
        offset=0;
        layerOffset=0;
    end
    figure(730+offset+layerNum)
    clf(730+offset+layerNum)
    [yinput,xinput]=hist(activations,25);
    unitType="";
    if(strcmp(eval(strcat("hp.",whichModel,".unitTypes{layerNum+layerOffset}")),'relu'))
        unitType="relu";
        xact=0:3;
        yact=xact;
    elseif (strcmp(eval(strcat("hp.",whichModel,".unitTypes{layerNum+layerOffset}")),'sigmoid'))
        unitType="sigmoid";
        xact=-5:0.5:5;
        yact=sigmoid(xact);
    elseif (strcmp(eval(strcat("hp.",whichModel,".unitTypes{layerNum+layerOffset}")),'stochastic'))
        unitType="sigmoid (stochastic)";
        xact=-5:0.5:5;
        yact=sigmoid(xact);
    end
    hold on;
    plot(xact,yact,"r");
    plot(xinput,yinput/max(yinput),"g");
    hold off;
    title(sprintf("Distribution of inputs to %s layer %i (before %s,\n red:nonlinearity, green: distribution of inputs)",whichModel,layerNum+layerOffset,unitType));
    meanInput=mean(activations(:),1);
    stdInput=std(activations(:),1);
    
    metaData.meanInput(layerNum-1)=meanInput;
    metaData.stdInput(layerNum-1)=stdInput;
            
end


%%% generative network
function [activations,metaData]=forwardPropG(hp,params,noise)
    % forward pass through generator
    activations={};
    activations{1}=noise;
    metaData.meanInput=[];
    metaData.stdInput=[];
    for layer=1:length(hp.generator.layerSizes)-1
        activations{layer+1}=activations{layer}*params.generator.weights{layer}+repmat(params.generator.biases{layer},hp.batchSize,1);
        
        
        %weight initialization phase: check distribution of inputs to non-linearity!
        if isfield(hp,"captureStatsForActivations")
            metaData=calcuateStatsForActivations(hp,metaData,"generator",layer+1,activations{layer+1}(:));
        end
        
        if(strcmp(hp.generator.unitTypes{layer+1},'relu'))
            activations{layer+1}=max(0,activations{layer+1});
        elseif (strcmp(hp.generator.unitTypes{layer+1},'sigmoid'))
            activations{layer+1}=sigmoid(activations{layer+1},hp.temperature);
        elseif (strcmp(hp.generator.unitTypes{layer+1},'stochastic') && layer==length(hp.generator.layerSizes)-1) 
            activations{layer+1}=sigmoid(activations{layer+1});
            activations{layer+2}=activations{layer+1}; %this is a hack. we store the probabilities here
            activations{layer+1}=activations{layer+1}>rand(size(activations{layer+1}));
            
        else
            error('ERROR: Unit type %s not implemented yet\n',hp.generator.unitTypes(layer+1));
        end
    end
end



%%%discrimitative network
function [activations,metaData]=forwardPropD(hp,params,data)
    % forward pass through discriminator
    activations={};
    activations{1}=data;
    metaData.meanInput=[];
    metaData.stdInput=[];
    for layer=1:length(hp.discriminator.layerSizes)-1
        %droput
        %dropout is used, applied to this layerdropoutmode in particular, and  we're in training mode
        if hp.discriminator.useDropout && hp.discriminator.dropout(layer)<1.0 && strcmp(hp.discriminator.dropoutmode,"train")
            %drop inputs from layer below according to dropout rate
            %hack: store dropout mask in activations (offset: 2*number of layers)
            % we need it when backpropagating the error to units with maxout or tanh activation function, see http://metaoptimize.com/qa/questions/13923/dropout-why-doesnt-backprop-change
            activations{layer+2*length(hp.discriminator.unitTypes)}=(rand(size(activations{layer}))>(1-hp.discriminator.dropout(layer)));
            activations{layer}=activations{layer}.*activations{layer+2*length(hp.discriminator.unitTypes)};
        end
        tmp=[];
        if(strcmp(hp.discriminator.unitTypes{layer},'maxout'))
            for piece=1:hp.discriminator.maxoutPieces
                tmp(:,:,piece)=activations{layer}*params.discriminator.weights{layer}(:,:,piece)+repmat(params.discriminator.biases{layer}(:,:,piece),hp.batchSize,1);
            end
            %store max values as activations, as well as corresponding indices (which maxout piece was used)
            %hack: store affected piece number in activation matrix (offset: number of layers)
            [activations{layer+1},activations{layer+1+length(hp.discriminator.unitTypes)}]=max(tmp,[],3);
        else
            %standard propagation without maxout
            activations{layer+1}=activations{layer}*params.discriminator.weights{layer}+repmat(params.discriminator.biases{layer},hp.batchSize,1);
        
        end
        
        
        %dropout ist used, applied, to this layer in particular, and we're in test mode
        if  hp.discriminator.useDropout && hp.discriminator.dropout(layer)!=1.0  && strcmp(hp.discriminator.dropoutmode,"test")
            %use all units, but scale down results
            activations{layer+1}*=hp.discriminator.dropout(layer);
        end
        
        %weight initialization phase: check distribution of inputs to non-linearity!
        if isfield(hp,"captureStatsForActivations") && hp.captureStatsForActivations
            %size(activations{layer+1})
            %figure(44111)
            %hist(activations{layer}(:))
            %activations{layer}(:)(1:3)
            %figure(44112)
            %hist(activations{layer+1}(:))
            metaData=calcuateStatsForActivations(hp,metaData,"discriminator",layer+1,activations{layer+1}(:));
        end
        
        %in case of unit with activation function, apply it!
        if(strcmp(hp.discriminator.unitTypes{layer},'relu'))
            activations{layer+1}=max(0,activations{layer+1});
        elseif (strcmp(hp.discriminator.unitTypes{layer},'sigmoid'))
            activations{layer+1}=sigmoid(activations{layer+1});
        elseif (strcmp(hp.discriminator.unitTypes{layer},'maxout'))
            %dont do anything here
        else
            error('ERROR: Unit type %s not implemented yet\n',hp.discriminator.unitTypes{layer});
        end
    end
end

function error=calculateCost(hp,data,target,errorType)
    %calculates the discriminative error (per training example) 
    %target is either 0 (noise samples) or 1 (training data)
    if(strcmp(errorType,'classification'))
        error=sum(abs(target-data));
    elseif(strcmp(errorType,'log(1-y)')) %y are noise-driven samples (maximize log(1-y))
        error=log(1-data);
    elseif(strcmp(errorType,'log(y)')) %y are real samples (maximize log(y))
        error=log(data);
    end
    
end

function [gradient,biasGradient,errorDerivatives]=backpropD(hp,params,targets,predictions,errorType)
    %propagates error backward & calculates the gradient 
    %   targets = vector with targets
    %   predictions=dict with all layers' forward prop activations
    errorDerivatives={};
    gradient={};
    biasGradient={};
    %last layer uses given cost as error
    for layer=length(hp.discriminator.layerSizes):-1:2
        %---cost function:----
        %cost=errorFunction(targets,a);
        %a=activation(z)
        %z=lastLayer*weights
        %---partial derivative for weights:---- 
        %dCost/dWeight=dCost/da * da/dz * dz/dlastLayer
        %---partial derivative for lastLayer errors:---- 
        %dCost/dlastLayer=dCost/da * da/dz * dz/dweights
        
        if(layer==length(hp.discriminator.layerSizes))
            %top layer: calculate error gradient
            if (strcmp(errorType,'log(1-y)') 
                &&strcmp(hp.discriminator.unitTypes{layer-1},'sigmoid'))
                %derivative of error function for  noise-driven samples
                %Cost=log(1-y), dcost/dy=1/(1-y)
                %y=1-sigm(z), dy/dz=(-1)*sigm(z)*(1-sigm(z))
                %--> dCost/dz=1/(1-sigm(z) * (-1)*sigm(z) * (1-sigm(z)) = -sigm(z)
                dCostdz=-predictions{layer}; %"predictions" are actually sigm(z)
            elseif (strcmp(errorType,'log(y)') %y are real samples
                &&strcmp(hp.discriminator.unitTypes{layer-1},'sigmoid'))
                %derivative of error function for  real samples in discriminator, or noise driven samples when training generator
                %Cost=log(y), dcost/dy=1/y
                %y=sigm(z), dy/dz=sigm(z)*(1-sigm(z))
                %--> dCost/dz=1/(sigm(z) * sigm(z) * (1-sigm(z)) = (1-sigm(z))
                dCostdz=1-predictions{layer}; %"predictions" are actually sigm(z)
            else
                error('ERROR: last layer must be sigmoid units, and log(y) or log(1-y) error function.\n')
            end
    
        else
            %other layers use back-propagated cost
            %regardless of neuron type: 
            %   dcost/da=errorDerivatives{layer}
            if (strcmp(hp.discriminator.unitTypes{layer-1},'relu'))
                %da/dz = 1 if z>0
                %        0 else
                dCostdz=errorDerivatives{layer}.*(predictions{layer}>0);
            elseif (strcmp(hp.discriminator.unitTypes{layer-1},'sigmoid'))
                %da/dz = a*(1-a)
                dCostdz=errorDerivatives{layer}.*(predictions{layer}.*(1-predictions{layer}));
            elseif (strcmp(hp.discriminator.unitTypes{layer-1},'maxout'))
                %da/dz = 1 (for affected piece)
                dCostdz=errorDerivatives{layer};
                %when using dropout with certain activation functions such as maxout or tanh, we need to apply the dropout mask, see http://metaoptimize.com/qa/questions/13923/dropout-why-doesnt-backprop-change
                if  hp.discriminator.useDropout && hp.discriminator.dropout(layer)<1.0  && strcmp(hp.discriminator.dropoutmode,"train")
                    dCostdz=dCostdz.*predictions{layer+2*length(hp.discriminator.unitTypes)};
                end
            else
                error('ERROR: Unit type %s not implemented yet\n',hp.discriminator.unitTypes{layer-1});
            end
        end
        %calculate total gradient for weights + biases
        if layer<length(hp.discriminator.layerSizes) && (strcmp(hp.discriminator.unitTypes{layer-1},'maxout'))
            %for maxout units, gradient must be attributed to the correct weight piece
            for piece=1:hp.discriminator.maxoutPieces
                gradient{layer-1}(:,:,piece)=predictions{layer-1}'*(dCostdz.*((predictions{layer+length(hp.discriminator.unitTypes)}==piece))); %'
                biasGradient{layer-1}(:,:,piece)=ones(1,hp.batchSize)*(dCostdz.*((predictions{layer+length(hp.discriminator.unitTypes)}==piece)));
            end
        else
            %for all other units just calculate outer product lowerLayer x cost@higherLayer
            gradient{layer-1}=predictions{layer-1}'*dCostdz; %'
            biasGradient{layer-1}=ones(1,hp.batchSize)*dCostdz;
        end
        %propagate error downwards to previous layer (=dCost/da for previous layer)
        if layer<length(hp.discriminator.layerSizes) && (strcmp(hp.discriminator.unitTypes{layer-1},'maxout'))
            %again, for maxout, error derivatives must be composed by using the correct weight piece
            errorDerivatives{layer-1}=0;
            for piece=1:hp.discriminator.maxoutPieces
                errorDerivatives{layer-1}+=((predictions{layer+length(hp.discriminator.unitTypes)}==piece).*dCostdz)*params.discriminator.weights{layer-1}(:,:,piece)';
            end
        else
            errorDerivatives{layer-1}=dCostdz*params.discriminator.weights{layer-1}'; %'
        end
        


    end
    
    
end

function [gradient,biasGradient]=backpropG(hp,params,targets,predictions)
    %propagates error backward & calculates the gradient 
    %   targets = vector with targets
    %   predictions=dict with all layers' forward prop activations
    errorDerivatives={};
    gradient={};
    biasGradient={};
    
    %top layer: error derivative given as input (from discriminator)
    errorDerivatives{length(hp.generator.layerSizes)}=targets;
    
    for layer=length(hp.generator.layerSizes):-1:2
        %---cost function:----
        %cost=errorFunction(targets,a);
        %a=activation(z)
        %z=lastLayer*weights
        %---partial derivative for weights:---- 
        %dCost/dWeight=dCost/da * da/dz * dz/dlastLayer
        %---partial derivative for lastLayer errors:---- 
        %dCost/dlastLayer=dCost/da * da/dz * dz/dweights
        
        %regardless of neuron type: 
        %   dcost/da=errorDerivatives{layer}
        if (strcmp(hp.generator.unitTypes{layer},'relu'))
            %da/dz = 1 if z>0
            %        0 else
            dCostdz=errorDerivatives{layer}.*(predictions{layer}>0);
        elseif (strcmp(hp.generator.unitTypes{layer},'sigmoid'))
            %da/dz = a*(1-a)
            dCostdz=errorDerivatives{layer}.*(predictions{layer}.*(1-predictions{layer}));
        elseif (strcmp(hp.generator.unitTypes{layer},'stochastic') && layer==length(hp.generator.layerSizes))
            %da/dz = a*(1-a)
            dCostdz=errorDerivatives{layer}.*(predictions{layer+1}.*(1-predictions{layer+1}));%according to hack in forwardPropG
        else
            error("ERROR: Unit type %s not implemented yet\n",hp.generator.unitTypes{layer-1});
        end
        
        %calculate total gradient for weights + biases
        gradient{layer-1}=predictions{layer-1}'*dCostdz; %'
        biasGradient{layer-1}=ones(1,hp.batchSize)*dCostdz;
        %propagate error downwards to previous layer (=dCost/da for previous layer)
        errorDerivatives{layer-1}=dCostdz*params.generator.weights{layer-1}'; %'

    end
    
end


function noise=sampleNoiseForGenerator(hp)
    %samples a batch of noise
    if(strcmp(hp.generator.unitTypes{1},'uniform'))
        noise=rand(hp.batchSize,hp.generator.layerSizes(1))*1; 
        %noise=rand(hp.batchSize,hp.generator.layerSizes(1))*6-3; %as in Goodfellow et al 2014
    else
        error("ERROR: noise distribution not implemented yet\n");
    end
                
end

function params=initializeGAN(hp,fromScratch,paramsOld,keepModel)
    #initializes a GAN
    #   -resets all weights
    %check some values
    if (hp.generator.useL2Norm && hp.generator.weightDecay)|| (hp.discriminator.useL2Norm && hp.discriminator.weightDecay)
        error("Cannot use L2 norm and Weight decay at the same time\n");
    end
    
    if hp.tuneInitialWeights
        hp.captureStatsForActivations=1;
    end
     
    #generator
    params.generator.weights={};
    params.generator.biases={};
    for layer=1:length(hp.generator.layerSizes)-1
        fprintf(" ----- Initializing generator layer %i ------\n",layer)
        weightsNotInitialized=1;
        dynamicOffset=0;
        dynamicMultiplier=1;
        while weightsNotInitialized
            if(strcmp(hp.initializationNoiseType,'uniform'))
                params.generator.weights{layer}=(rand(hp.generator.layerSizes(layer),hp.generator.layerSizes(layer+1))-0.5)*hp.generator.randomWeightMultiplier*dynamicMultiplier;
                params.generator.biases{layer}=zeros(1,hp.generator.layerSizes(layer+1))+dynamicOffset;
                %(rand(1,hp.generator.layerSizes(layer+1))-0.5)*hp.generator.randomWeightMultiplier+dynamicOffset;
            elseif(strcmp(hp.initializationNoiseType,'normal'))
                params.generator.weights{layer}=(randn(hp.generator.layerSizes(layer),hp.generator.layerSizes(layer+1)))*hp.generator.randomWeightMultiplier*dynamicMultiplier;
                params.generator.biases{layer}=zeros(1,hp.generator.layerSizes(layer+1))+dynamicOffset;
                %(randn(1,hp.generator.layerSizes(layer+1)))*hp.generator.randomWeightMultiplier+dynamicOffset;
            else
                error("Noise type for weight initialization not implemented\n")
            end
            if hp.generator.sparseInit
                fprintf("Inifializing Generator Biases for sparseness\n");
                params.generator.biases{layer}=ones(size(params.generator.biases{layer}))*hp.generator.sparseInit;
            end
            
            %% Tune weight initialization such that average activation of neurons allows for
            %% proper gradients (see e.g. Leon Bottou's talk, around 1h:20min
            %% http://videolectures.net/deeplearning2015_bottou_neural_networks/)
            if hp.tuneInitialWeights
                %clone hyperparameters
                hp2=hp;
                hp2.batchSize=500;
                %modify hp: reduce number of layers
                hp2.generator.layerSizes=hp2.generator.layerSizes(1:layer+1);
                %sample noise
                noise=sampleNoiseForGenerator(hp2);
                %forward propagate noise samples through generator G(z)
                [generatorActivations,metaData]=forwardPropG(hp2,params,noise);
                changesNeeded=0;
                if(strcmp(hp.generator.unitTypes{layer+1},'relu'))
                    %For rectified linear units, mode should be in linear part (not at the nonlinearity at 0), i.e., zero should be between mean-1.5*std and mean-0.66*std
                    if metaData.meanInput(layer)-1.5*metaData.stdInput(layer)>0
                        dynamicOffset-=0.1;
                        changesNeeded=1;
                    elseif metaData.meanInput(layer)-0.66*metaData.stdInput(layer)<0
                        dynamicOffset+=0.1;
                        changesNeeded=1;
                    end
                    if !changesNeeded
                        weightsNotInitialized=0;
                        fprintf("Initialization seems OK for reLu at generator Layer %i\n  (weight multiplier: %.2f bias offset: %.2f)\n",layer, dynamicMultiplier, dynamicOffset);
                    end
                elseif (strcmp(hp.generator.unitTypes{layer+1},'sigmoid')) || (strcmp(hp.generator.unitTypes{layer+1},'stochastic'))
                    %For sigmoid linear units, mean should be in linear part, distribution should not be too narrow/wide
                    %if metaData.meanInput(layer)-1.5*metaData.stdInput(layer)>-2
                    %    dynamicOffset-=0.1;
                    %    changesNeeded=1;
                    %elseif metaData.meanInput(layer)-0.66*metaData.stdInput(layer)<-2
                    %    dynamicOffset+=0.1;
                    %    changesNeeded=1;
                    %end
                    
                    if metaData.stdInput(layer)<.7
                        dynamicMultiplier*=1.1;
                        changesNeeded=1;
                    elseif metaData.stdInput(layer)>1
                        dynamicMultiplier/=1.1;
                        changesNeeded=1;
                    end
                    refresh();
                    if !changesNeeded
                        weightsNotInitialized=0;
                        fprintf("Initialization seems OK for sigmoid at generator Layer %i\n  (weight multiplier: %.2f bias offset: %.2f)\n",layer, dynamicMultiplier, dynamicOffset)
                    end
                end
            
            else
                weightsNotInitialized=0;
            end
        end
        
    end

    #discriminator
    params.discriminator.weights={};
    params.discriminator.biases={};
    for layer=1:length(hp.discriminator.layerSizes)-1
        fprintf(" ----- Initializing discriminator layer %i ------\n",layer)
        numPieces=1;
        if(strcmp(hp.discriminator.unitTypes{layer},'maxout'))
            numPieces=hp.discriminator.maxoutPieces;
        end
        weightsNotInitialized=1;
        dynamicOffset=0;
        dynamicMultiplier=1;
        while weightsNotInitialized
            %don't do initial weight scaling in last layer (only one output) and on maxout layers
            hp.discriminator.unitTypes{layer}
            if layer==length(hp.discriminator.layerSizes)-1 || strcmp(hp.discriminator.unitTypes{layer},'maxout')
                weightsNotInitialized=0;
                hp.captureStatsForActivations=0; %actually, this turns auto tuning off alltogether
            end
            weightsNotInitialized
            if(strcmp(hp.initializationNoiseType,'uniform'))
                params.discriminator.weights{layer}=(rand(hp.discriminator.layerSizes(layer),hp.discriminator.layerSizes(layer+1),numPieces)-0.5)*hp.discriminator.randomWeightMultiplier*dynamicMultiplier;
                params.discriminator.biases{layer}=zeros(1,hp.discriminator.layerSizes(layer+1),numPieces)+dynamicOffset;%(rand(1,hp.discriminator.layerSizes(layer+1),numPieces)-0.5)*hp.discriminator.randomWeightMultiplier+dynamicOffset;
            elseif(strcmp(hp.initializationNoiseType,'normal'))
                params.discriminator.weights{layer}=(randn(hp.discriminator.layerSizes(layer),hp.discriminator.layerSizes(layer+1),numPieces))*hp.discriminator.randomWeightMultiplier*dynamicMultiplier;
                params.discriminator.biases{layer}=zeros(1,hp.discriminator.layerSizes(layer+1),numPieces)+dynamicOffset;
                %(randn(1,hp.discriminator.layerSizes(layer+1),numPieces))*hp.discriminator.randomWeightMultiplier+dynamicOffset;
            else
                error("Noise type for weight initialization not implemented\n")
            end     
            
             if hp.tuneInitialWeights && hp.captureStatsForActivations
                %clone hyperparameters
                hp2=hp;
                hp2.batchSize=1000;
                %modify hp: reduce number of layers
                hp2.discriminator.layerSizes=hp2.discriminator.layerSizes(1:layer+1);
                %sample noise
                noise=sampleNoiseForGenerator(hp2);
                %figure(113322)
                %hist(noise(:),30)
                %dataExampleIndices=ceil(rand(hp.batchSize,1)*size(data,1));
                
                %forward propagate noise samples through generator G(z)
                generatorActivations=forwardPropG(hp2,params,noise);
                hp2.discriminator.dropoutmode="train";

                [discriminatorStatesNoiseSamples,metaData]=forwardPropD(hp2,params,generatorActivations{length(hp2.generator.layerSizes)});
                %discriminatorStatesRealSamples=forwardPropD(hp2,params,data(dataExampleIndices,:));
                refresh();

                changesNeeded=0;
                if(strcmp(hp.discriminator.unitTypes{layer},'relu'))
                    %For rectified linear units, mode should be in linear part (not at the nonlinearity at 0), i.e., zero should be between mean-1.5*std and mean-0.66*std
                    if metaData.meanInput(layer)-1.5*metaData.stdInput(layer)>0
                        dynamicOffset-=0.1;
                        changesNeeded=1;
                    elseif metaData.meanInput(layer)-0.66*metaData.stdInput(layer)<0
                        dynamicOffset+=0.1;
                        changesNeeded=1;
                    end
                    if !changesNeeded
                        weightsNotInitialized=0;
                        fprintf("Initialization seems OK for reLu at discriminator Layer %i\n  (weight multiplier: %.2f bias offset: %.2f)\n",layer, dynamicMultiplier, dynamicOffset)
                    end
                elseif (strcmp(hp.discriminator.unitTypes{layer},'sigmoid'))
                    %For sigmoid linear units, mean should be in linear part, distribution should not be too narrow/wide
                    %if metaData.meanInput(layer)-1.5*metaData.stdInput(layer)>-2
                    %    dynamicOffset-=0.1
                    %    changesNeeded=1;
                    %elseif metaData.meanInput(layer)-0.66*metaData.stdInput(layer)<-2
                    %    dynamicOffset+=0.1
                    %    changesNeeded=1;
                    %end
                    
                    if metaData.stdInput(layer)<.7
                        dynamicMultiplier*=1.1;
                        changesNeeded=1;
                    elseif metaData.stdInput(layer)>1.1
                        dynamicMultiplier/=1.1;
                        changesNeeded=1;
                    end
                    
                    if !changesNeeded
                        weightsNotInitialized=0;
                        fprintf("Initialization seems OK for sigmoid at discriminator Layer %i\n  (weight multiplier: %.2f bias offset: %.2f)\n",layer, dynamicMultiplier, dynamicOffset)
                    end
                end
                
                %input("stop\n")
            else
                weightsNotInitialized=0;
            end
            
            
        end
    end
    
        
end

function params=initializeGeneratorBiases(hp,params,data)
    #initialize biases of generator sample layer to
    # log(P(x/(1-x))
    #(i.e., biases already reflect fully factorized distribution)
    probs=sum(data)/size(data,1);
    params.generator.biases{length(hp.generator.layerSizes)-1}=log(probs./(1-probs));
end

function error=testGParams(hp,params,noise)
    generatorActivations=forwardPropG(hp,params,noise);
    hp.discriminator.dropoutmode="test";
    discriminatorStates=forwardPropD(hp,params,generatorActivations{length(hp.generator.layerSizes)});    
    error=sum(calculateCost(hp, discriminatorStates{length(hp.discriminator.layerSizes)}, target=1,errorType='log(y)'));
end

function showFeatures(hp,params)
    if hp.mnistFlag
        fprintf("Todo: print mnist features <---\n")
    else
        figure(99001)
        tmpMin=((round(min(params.generator.weights{length(hp.generator.layerSizes)-1}(:))*1000))/1000.0);
        tmpMax=((round(max(params.generator.weights{length(hp.generator.layerSizes)-1}(:))*1000))/1000.0);
        tmpMean=((round(mean(params.generator.weights{length(hp.generator.layerSizes)-1}(:))*1000))/1000.0);
        tmpWeights=(params.generator.weights{length(hp.generator.layerSizes)-1}.*(0.5/max(tmpMax,abs(tmpMin))).+0.5);
        imshow(tmpWeights);
        
        name=strcat("Generator Features, min: ",mat2str(tmpMin)," max:",mat2str(tmpMax), " mean:",mat2str(tmpMean));
        title(name);
        
        figure(99002);
        hist(tmpWeights(:),20)
    end
end

function params=trainGAN(hp, data,params,validationData,fitness)
    % Trains a generative adversarial network 
    % input:
    %   hp: hyperparams
    %   data: training data, matrix of size (numExamples,exampleSize)
    unlink ("/tmp/stopOctave")%workaround for "octave not stopping bug"
    
    %make unique dataset 
    if hp.uniqueData
        oldsize=size(data,1);
        data=unique(data,'rows');
        fprintf("Deleted %i duplicates in dataset,  %i examples remaining.\n",oldsize-size(data,1),size(data,1));
    end
    
    %split off validation set, if required
    if hp.discriminator.validationSet>0.0
        %shuffle data first
        [garbage,order]=sort(rand(1,size(data,1)));
        data=data(order,:);
        numItemsValidationSet=floor(size(data,1)*hp.discriminator.validationSet);
        validationData=data(1:numItemsValidationSet,:);
        data=data(numItemsValidationSet+1:end,:);
    end
    
    %set biases
    if hp.generator.initBiasToData
        params=initializeGeneratorBiases(hp,params,data);
    end

    %fields to remember last updates (for momentum)
    discriminatorUpdate={};discriminatorBiasUpdate={};
    for layer=1:length(hp.discriminator.layerSizes)-1
        discriminatorUpdate{layer}=zeros(size(params.discriminator.weights{layer}));
        discriminatorBiasUpdate{layer}=zeros(size(params.discriminator.biases{layer}));
    end
    generatorUpdate={};generatorBiasUpdate={};
    for layer=1:length(hp.generator.layerSizes)-1
        generatorUpdate{layer}=zeros(size(params.generator.weights{layer}));
        generatorBiasUpdate{layer}=zeros(size(params.generator.biases{layer}));
    end
    
    %set decay of learning rate such that, after the maximum number of epochs,
    % or the maximum number of updates, the learning rate is X percent of the initial rate
    % and use this point to stop training
    maxEffectiveUpdates=min(hp.maxUpdates,(size(data,1)/hp.batchSize)*hp.maxEpochs);
    effectiveAlphaDecay=nthroot(hp.alphaMin/hp.alphaInitial,maxEffectiveUpdates);
    effectiveMomentumGrowth=nthroot(hp.momentumMax/hp.momentumInitial,maxEffectiveUpdates);
    hp.maxUpdates=maxEffectiveUpdates;
    hp.alphaDecay=effectiveAlphaDecay;
    hp.momentumGrowth=effectiveMomentumGrowth;
    %alphaDecay
    
    %learning hyper parameters
    currentAlpha=hp.alphaInitial;
    currentMomentum=hp.momentumInitial;
    hp.temperature=hp.temperatureInitial;
    %alphaAdjust.generator=ones(1,length(hp.generator.layerSizes)-1);
    %alphaAdjust.discriminator=ones(1,length(hp.discriminator.layerSizes)-1);
    
    %gradientWeightRatio.generator=hp.generator.targetGradientWeightRatio*ones(1,length(hp.generator.layerSizes)-1);
    %gradientWeightRatio.discriminator=hp.discriminator.targetGradientWeightRatio*ones(1,length(hp.discriminator.layerSizes)-1);
    
    %main loop control variables
    converged=0;
    numUpdates=0;
    regularizationNotChangedCounter=0;
    checkRegularizationEachNUpdates=1000;
    
    %stats
    errorGeneratorUpdates=[];
    errorDiscriminatorUpdates=[];
    errorDiscriminatorUpdatesValidationSet=[];
    averageAbsGradient=[];
    averageGeneratorLearningSignal=[];
    learningRateUpdates=[];
    momentumUpdates=[];
    numConvergenceChecks=0;
    improvementRate=[1];
    improvementRateSmooth=[1];
    latestImprovementRelative=[];
    %main loop
    while ((!converged) && ( numUpdates < hp.maxUpdates ))
        numUpdates+=1;
        regularizationNotChangedCounter+=1;
        if (hp.loglevel>0 && mod(numUpdates,hp.showStatsEachNthUpdate)==0)
            fprintf("Gradient Updates (G): %i\n",numUpdates)
            fprintf("  i.e. Examples seen: %i (%.1f times training set [Expectation])\n",numUpdates*hp.batchSize,(numUpdates*hp.batchSize)/size(data,1))
        end
        tmpErrorGenerator=0;
        tmpErrorDiscriminator=0;
        tmpErrorDiscriminatorValidationSet=0;
        tmpAverageAbsGradient=0;
        
        %shuffle batches
        %[garbage,shuffleindex]=sort(rand(1,size(data,1)));
        %data=data(shuffleindex,:);
        
        %%% Octave bug: cannot stop exectution on CTRL-C when using multithreaded BLAS library (probably specific to work pc)
        %workaround: check existence of file /tmp/stopOctave
        if exist("/tmp/stopOctave","file")==2
            converged=1;
            fprintf("Aborting....\n");
        end
        
        %%%%%%%%%%%%%%%%%%%% train discriminator %%%%%%%%%%%%%%%%%%%%
        kMax=hp.discriminator.numTrainingSteps;
        if(hp.discriminator.headStart>0 && numUpdates==1)
            fprintf("Head start for discriminator (%i updates)!\n",hp.discriminator.headStart);
            kMax=hp.discriminator.headStart;
        end
        
        %Perform k updates on discriminator
        for(k=1:kMax)
            
            %sample minibatch of examples from noise prior p(z)
            noise=sampleNoiseForGenerator(hp);
            
            %sample examples from true data
            dataExampleIndices=ceil(rand(hp.batchSize,1)*size(data,1));
            
            %forward propagate noise samples through generator G(z)
            generatorActivations=forwardPropG(hp,params,noise);
            
            %propagate both sample batches forward through discriminator D(x)
            hp.discriminator.dropoutmode="train";
            discriminatorStatesNoiseSamples=forwardPropD(hp,params,generatorActivations{length(hp.generator.layerSizes)});
            discriminatorStatesRealSamples=forwardPropD(hp,params,data(dataExampleIndices,:));

            if(k==kMax && mod(numUpdates,hp.showStatsEachNthUpdate)==0)
                %calculate discriminative error
                errorNoiseSamples=sum(calculateCost(hp, discriminatorStatesNoiseSamples{length(hp.discriminator.layerSizes)}, target=0, errorType='log(1-y)'));
                errorRealSamples=sum(calculateCost(hp,  discriminatorStatesRealSamples{length(hp.discriminator.layerSizes)}, target=1, errorType='log(y)'));
            end
            %calculate discriminator gradient (using discrimintative error)  
            %noise sample batch
            [gradientNoiseSamples,biasGradientNoiseSamples,garbage]=backpropD(hp,params,0,discriminatorStatesNoiseSamples,errorType='log(1-y)');
            %real sample batch
            [gradientRealSamples,biasGradientRealSamples,garbage]=backpropD(hp,params,1,discriminatorStatesRealSamples,errorType='log(y)');
        
            %combine gradients of both batches
            gradient={};
            biasGradient={};
            for layer=1:length(hp.discriminator.layerSizes)-1
                gradient{layer}=gradientNoiseSamples{layer}+gradientRealSamples{layer};
                biasGradient{layer}=biasGradientNoiseSamples{layer}+biasGradientRealSamples{layer};
            end 
            
            %track average gradient
            %if(k==kMax && mod(numUpdates,hp.showStatsEachNthUpdate)==0)
            %   for layer=1:length(hp.discriminator.layerSizes)-1
            %       tmpAverageAbsGradient+=mean(abs(gradient{layer}(:)))/(length(hp.discriminator.layerSizes)-1);
            %   end             
            %end
            %apply gradient update to discrimintator
            for layer=1:length(hp.discriminator.layerSizes)-1
                if(strcmp(hp.discriminator.unitTypes{layer},'maxout'))
                    pieces=hp.discriminator.maxoutPieces;
                else
                    pieces=1;
                end
                for piece=1:pieces
                    %calulate update for weights (including momentum)
                    discriminatorUpdate{layer}(:,:,piece)=currentMomentum*discriminatorUpdate{layer}(:,:,piece)+(1.0/(2*hp.batchSize))*gradient{layer}(:,:,piece);
                    
                    %if hp.discriminator.weightDecay>0.0
                    %    discriminatorUpdate{layer}(:,:,piece)-=hp.discriminator.weightDecay*params.discriminator.weights{layer}(:,:,piece);
                    %end
                    
                    %apply update for weights
                    params.discriminator.weights{layer}(:,:,piece)+=currentAlpha.*(discriminatorUpdate{layer}(:,:,piece)-hp.discriminator.weightDecay*params.discriminator.weights{layer}(:,:,piece));
                    
                    if hp.discriminator.useL2Norm
                        %
                        fprintf("not sure if this works, line 477 (or around)\n")
                        params.discriminator.weights{layer}(:,:,piece)=params.discriminator.weights{layer}(:,:,piece).*repmat(min(1,hp.discriminator.useL2Norm./max(abs(params.discriminator.weights{layer}(:,:,piece)))),size(params.discriminator.weights{layer}(:,:,piece),1),1);
                    end
                    
                    %calculate update for biases
                    discriminatorBiasUpdate{layer}(:,:,piece)=currentMomentum*discriminatorBiasUpdate{layer}(:,:,piece)+biasGradient{layer}(:,:,piece)*(1.0/(2*hp.batchSize));
                    
                    %if hp.discriminator.weightDecay>0.0
                    %    discriminatorBiasUpdate{layer}(:,:,piece)-=hp.discriminator.weightDecay*params.discriminator.biases{layer}(:,:,piece);
                    %end
                    
                    %apply update for biases
                    params.discriminator.biases{layer}(:,:,piece)+=(hp.alphaBiasMultiplier*currentAlpha).*discriminatorBiasUpdate{layer}(:,:,piece);
                end
            end  %discriminator weight udpate   
            
        end %discriminator loop
        
        
        %stats and validation set
        if(mod(numUpdates,hp.showStatsEachNthUpdate)==0)
            %save last discriminator error
            tmpErrorDiscriminator+=(errorNoiseSamples+errorRealSamples)/(2*hp.batchSize);
            
            %calculate error for validation set (if used)
            if(hp.discriminator.validationSet>0)
                %sample minibatch of examples from noise prior p(z)
                noise=sampleNoiseForGenerator(hp);
                %sample examples from true data
                dataExampleIndices=ceil(rand(hp.batchSize,1)*size(validationData,1));
                %forward propagate noise samples through generator G(z)
                generatorActivations=forwardPropG(hp,params,noise);

                %propagate both sample batches forward through discriminator D(x)
                hp.discriminator.dropoutmode="train";
                discriminatorStatesNoiseSamples=forwardPropD(hp,params,generatorActivations{length(hp.generator.layerSizes)});
                discriminatorStatesRealSamples=forwardPropD(hp,params,validationData(dataExampleIndices,:));
                %calculate discriminative error
                errorNoiseSamples=sum(calculateCost(hp, discriminatorStatesNoiseSamples{length(hp.discriminator.layerSizes)}, target=0, errorType='log(1-y)'));
                errorRealSamples=sum(calculateCost(hp,  discriminatorStatesRealSamples{length(hp.discriminator.layerSizes)}, target=1, errorType='log(y)'));
                
                tmpErrorDiscriminatorValidationSet+=(errorNoiseSamples+errorRealSamples)/(2*hp.batchSize);
            end
            
        end
        %%%%%%%%%%%%%%%%%%%% train generator %%%%%%%%%%%%%%%%%%%%
        
        %sample minibatch of examples from noise prior p(z)
        noise=sampleNoiseForGenerator(hp);
        
        #forward propagate noise samples through G(z)
        generatorActivations=forwardPropG(hp,params,noise);
        
        

        samplesForDiscriminator=generatorActivations{length(hp.generator.layerSizes)};
            
        %forward propagate noise batch through discriminator D(x)
        hp.discriminator.dropoutmode="test";
        discriminatorStates=forwardPropD(hp,params,samplesForDiscriminator);
        
        if(mod(numUpdates,hp.showStatsEachNthUpdate)==0)
            %calculate error on discriminator (from generator perspective)
            errorGeneratorSamples=sum(calculateCost(hp, discriminatorStates{length(hp.discriminator.layerSizes)}, target=1,errorType='log(y)'));
            tmpErrorGenerator+=(errorGeneratorSamples)/(hp.batchSize);
        end
            
        %calculate generator gradient (maximize log(D(G(z))) )  
        %part1: discriminator part
        [garbage1,garbage2,errorDerivativesDiscriminator]=backpropD(hp,params,1,discriminatorStates,errorType='log(y)');
        
        %part2: generator part, use error derivatives from first layer of discriminator
        [gradient,biasGradient]=backpropG(hp,params,errorDerivativesDiscriminator{1},generatorActivations);
        
        %track average gradient
        %if(mod(numUpdates,hp.showStatsEachNthUpdate)==0)
        %       for layer=1:length(hp.generator.layerSizes)-1
        %           tmpAverageAbsGradient+=mean(abs(gradient{layer}(:)))/(length(hp.generator.layerSizes)-1);
        %       end             
        %end

        
        %when using Dropout, the incoming gradient from the discriminator is too strong (backprop does not properly reduce the gradient according to the dropout rate - so we do it here)
        dropoutMultiplier=1.0;
        if hp.discriminator.useDropout
            for layer=1:length(hp.discriminator.layerSizes)-1
                dropoutMultiplier*=hp.discriminator.dropout(layer);
            end
        end

        %apply gradient update to generator
        for layer=1:length(hp.generator.layerSizes)-1
                %calulate update for weights (including momentum + learning rate alpha)
                generatorUpdate{layer}=currentMomentum*generatorUpdate{layer}+(dropoutMultiplier*1.0/hp.batchSize)*gradient{layer};
                
                %if hp.generator.weightDecay>0.0
                %    generatorUpdate{layer}-=hp.generator.weightDecay*params.generator.weights{layer};
                %end
                

                %apply update for weights
                params.generator.weights{layer}+=(hp.generator.alphaScale*currentAlpha).*(generatorUpdate{layer}-hp.generator.weightDecay*params.generator.weights{layer});

                if hp.generator.useL2Norm
                    %
                    params.generator.weights{layer}=params.generator.weights{layer}.*repmat(min(1,hp.generator.useL2Norm./max(abs(params.generator.weights{layer}))),size(params.generator.weights{layer},1),1);
                end
                
                %calulate update for biases (including momentum + learning rate alpha)
                generatorBiasUpdate{layer}=currentMomentum*generatorBiasUpdate{layer}+biasGradient{layer}*(dropoutMultiplier*1.0/hp.batchSize);
                
                %if hp.generator.weightDecay>0.0
                %    generatorBiasUpdate{layer}-=hp.generator.weightDecay*params.generator.biases{layer};
                %end
                
                %apply update for biases
                params.generator.biases{layer}+=(hp.alphaBiasMultiplier*currentAlpha*hp.generator.alphaScale).*generatorBiasUpdate{layer};
                
        end %generatior weight update
        
        %update momentum + learning rate
        currentAlpha=max(currentAlpha*hp.alphaDecay,hp.alphaMin);
        currentMomentum=min(currentMomentum*hp.momentumGrowth,hp.momentumMax);    
        
        %update temperature for sigmoid (decreasing, t=1 at hp.temperatureNumUpdatesTillOne updates)
        if numUpdates<hp.temperatureNumUpdatesTillOne && hp.temperatureInitial!=1.0
            stepSize=(hp.temperatureInitial-1)/hp.temperatureNumUpdatesTillOne;
            hp.temperature=hp.temperatureInitial-(numUpdates*stepSize);
        else
            hp.temperature=1.0;
        end
        
        %capture stats
        if (mod(numUpdates,hp.showStatsEachNthUpdate)==0)
            errorGeneratorUpdates=[errorGeneratorUpdates,tmpErrorGenerator];
            errorDiscriminatorUpdates=[errorDiscriminatorUpdates,tmpErrorDiscriminator];
            errorDiscriminatorUpdatesValidationSet=[errorDiscriminatorUpdatesValidationSet,tmpErrorDiscriminatorValidationSet];
            
            learningRateUpdates=[learningRateUpdates,currentAlpha];
            momentumUpdates=[momentumUpdates,currentMomentum];
            averageAbsGradient=[averageAbsGradient,tmpAverageAbsGradient/2.0];
        end


        %calculate smoothed objectives
        smoothedObjectiveG=[];
        smoothedObjectiveD=[];
        smoothedObjectiveDValidationSet=[];
            
        if (mod(numUpdates,hp.showStatsEachNthUpdate)==0)
            if floor(numUpdates/hp.showStatsEachNthUpdate)<10
                numSmooths=2;
            elseif floor(numUpdates/hp.showStatsEachNthUpdate)<20
                numSmooths=3;
            elseif floor(numUpdates/hp.showStatsEachNthUpdate)<40
                numSmooths=4;
            else
                numSmooths=5;               
            end
            if floor(numUpdates/hp.showStatsEachNthUpdate)>numSmooths
                for e=1:floor(numUpdates/hp.showStatsEachNthUpdate)-numSmooths
                    smoothG=0;
                    smoothD=0;
                    smoothDValidationSet=0;
                    for s=-numSmooths:1:numSmooths
                        if e+s<1 %left default value
                            smoothG+=-0.693;
                            smoothD+=-0.693;
                            smoothDValidationSet+=-0.693;
                        else
                            smoothG+=errorGeneratorUpdates(e+s);
                            smoothD+=errorDiscriminatorUpdates(e+s);
                            smoothDValidationSet+=errorDiscriminatorUpdatesValidationSet(e+s);
                        end
                    end
                    smoothedObjectiveG=[smoothedObjectiveG,smoothG/(2*numSmooths+1)];
                    smoothedObjectiveD=[smoothedObjectiveD,smoothD/(2*numSmooths+1)];
                    smoothedObjectiveDValidationSet=[smoothedObjectiveDValidationSet,smoothDValidationSet/(2*numSmooths+1)];
                end
            end
        end 
        
        %update weight decay for discriminator, if validation set is used
        if  (mod(numUpdates,hp.showStatsEachNthUpdate)==0) && hp.discriminator.validationSet>0 && numUpdates>hp.minUpdates
            gapValiErr=(smoothedObjectiveD(end)/smoothedObjectiveDValidationSet(end));
            if gapValiErr< hp.discriminator.maxValiDifference
                converged=1;
                fprintf("Gap Training / validation erro too big: %.2f, Stopping...\n",gapValiErr)
            end
            
            %if we haven't messed with the regularization recently
            if 0 && regularizationNotChangedCounter>=checkRegularizationEachNUpdates
                %check if difference between validation set and training set is _really_ large 
                if (smoothedObjectiveD(end)-smoothedObjectiveDValidationSet(end))>0.05
                    %increase the current L2 regularitation, or decrease the L2 norm
                    if hp.discriminator.weightDecay>0
                        hp.discriminator.weightDecay=min(0.01,hp.discriminator.weightDecay*1.2);
                        fprintf("Discriminator seems to be overfitting heavily. Increaseing D's L2 weight decay to %f\n",hp.discriminator.weightDecay);
                    elseif hp.discriminator.useL2Norm>0
                        hp.discriminator.useL2Norm=max(1.0,hp.discriminator.useL2Norm/1.2);
                        fprintf("Discriminator seems to be overfitting heavily. Decreasing D's L2 weight norm to %f\n",hp.discriminator.useL2Norm);
                    end
                    if hp.generator.weightDecay>0
                        hp.generator.weightDecay=min(0.005,hp.generator.weightDecay*1.1);
                        fprintf("... and increaseing G's L2 weight decay to %f\n",hp.generator.weightDecay);
                    elseif hp.generator.useL2Norm>0
                        hp.generator.useL2Norm=max(1.5,hp.generator.useL2Norm/1.1);
                        fprintf("... and decreasing G's  L2 weight norm to %f\n",hp.generator.useL2Norm);

                    end                 
                    regularizationNotChangedCounter=0;
                    checkRegularizationEachNUpdates*=1.2;
                end
            end
        end
        
        #check convergence
        if (mod(numUpdates,hp.checkConvergenceEachNUpdates)==0)
            %Some thoughts why it is tough to nail done some convergence conditions. 
            % 1) If generator objective is very good, i.e., the discriminator cannot distinguish G's output from real samples, then it might just be the case that D is particularly bad. On the other hand, if G's objective stays at the same level, G might still be improving a lot, because D improves at the same time!
            %   --> it doesnt make sense to monitor G's objective
            % 2) The same logic applies to D's objective.
            %   --> it neither makes sense to monitor D's objective
            %Hard breaking condition:
            % - number of updates > minUpdates 
            %   and number of "epochs" >= minEpochs -> break
            %     (no real epochs here, but expected passes through training set)
            % - "total convergence", i.e, data was so simply distributed that G can model it perfectly (i.e. in later phases of EDA)
            
            numConvergenceChecks+=1;
            
            numEpochs=(numUpdates*hp.batchSize)/size(data,1);
            %hard breaking after some time
            %if(numUpdates>=hp.minUpdates &&  numEpochs>=hp.minEpochs)           
            %   converged=true;
            %   fprintf("Minimum number of [epochs] (%i) and updates (%i) reached. Stopping training...\n",hp.minEpochs,hp.minUpdates);
            %end
            
            %dont stop before minumum number of updates & epochs
            if (numUpdates>=hp.minUpdates &&  numEpochs>=hp.minEpochs)          
                %"total convergence", ln(0.52)~=-0.65 --> G produces darn good samples, and D cant deal with them
                tmp=length(errorDiscriminatorUpdates);
                if (floor(numUpdates/hp.checkConvergenceEachNUpdates)>4)
                    tmp=[tmp-2,tmp-1,tmp];
                end
                meanLossD=mean(errorDiscriminatorUpdates(tmp));
                meanLossG=mean(errorGeneratorUpdates(tmp));
                if (meanLossD<-0.65 && meanLossG>-0.65)
                    converged=true;
                    fprintf("MeanLossD=%f, MeanLossG=%f, seems good enough. Stopping training...\n",meanLossD,meanLossG);
                end
                
                %once there are some points in validation set
                if(length(smoothedObjectiveDValidationSet)>5)
                    std(smoothedObjectiveDValidationSet(end-5:end))/std(smoothedObjectiveDValidationSet)
                    if std(smoothedObjectiveDValidationSet(end-5:end))/std(smoothedObjectiveDValidationSet) <0.1
                        converged=true;
                        fprintf("Standard deviation of validation error very small in last epochs")
                    end
                    
                                        

                end
                
                %check, if G's worst objective was in first 1/3rds of updates, i.e., G gets better, or, at least not worse
                numPoints=length(smoothedObjectiveG);
                [garbage,indexSmallest]=min(smoothedObjectiveG);
                %fprintf("Worst G at %.1f percent of updates\n",(indexSmallest/numPoints)*100);
                %if indexSmallest<(1/3)*numPoints
                %   converged=1;
                %   fprintf("Worst Objective for G was in first 1rd of training. Stopping... \n");
                %end
                
                totalObjective=smoothedObjectiveG-smoothedObjectiveD;
                [garbage,indexSmallest]=min(totalObjective);
                totalGain=totalObjective(end)-totalObjective(indexSmallest);
                if totalGain>0
                    totalGain;
                    lastGain=totalObjective(end)-totalObjective(ceil(2/3*numPoints));
                    relativeGain=lastGain/totalGain;
                    threshold=0.05;
                    if(relativeGain<threshold) && indexSmallest<(2/3)*numPoints && false
                        converged=1;
                        fprintf("Relative gain of last 3rd: %.2f, Stopping...\n",relativeGain)
                    end
                end
            end
                        
            
            %save current params as old params
            oldParams=params;
            
        end
        
        %show stats
        if (mod(numUpdates,hp.showStatsEachNthUpdate)==0 && hp.loglevel>1)
            %averageGeneratorLearningSignal=[averageGeneratorLearningSignal,mean(abs(errorDerivativesDiscriminator{1}(:)))];
            

            hold off;
            figure(1);
            plot(1:hp.showStatsEachNthUpdate:numUpdates,errorDiscriminatorUpdates);
            title("Discriminator Objective")
            figure(2);
            plot(1:hp.showStatsEachNthUpdate:numUpdates,errorGeneratorUpdates);
            title("Generator Objective")
            %figure(4);
            %plot(1:hp.showStatsEachNthUpdate:numUpdates,averageAbsGradient);
            %title("Average Abs Gradient (D+G)")
            %figure(5);
            %plot(1:hp.showStatsEachNthUpdate:numUpdates,averageGeneratorLearningSignal);
            %title("Average Learning Signal Strength (G)")

            %show smoothed objectives
            if floor(numUpdates/hp.showStatsEachNthUpdate)>numSmooths
                figure(1);
                hold on;
                plot(1:hp.showStatsEachNthUpdate:numUpdates-numSmooths*hp.showStatsEachNthUpdate,smoothedObjectiveD,'r');
                hold off;
                refresh(1);
                figure(2);
                hold on;
                plot(1:hp.showStatsEachNthUpdate:numUpdates-numSmooths*hp.showStatsEachNthUpdate,smoothedObjectiveG,'r');
                hold off;
                refresh(2);
            end
            
            %show validation set objective for D
            if hp.discriminator.validationSet>0
                figure(1);
                hold on;
                plot(1:hp.showStatsEachNthUpdate:numUpdates-numSmooths*hp.showStatsEachNthUpdate,smoothedObjectiveDValidationSet,'g');
                hold off;
                refresh(1);
            end
            

            
            %%%%%TEST PLOTS
            %figure(22);
            %plot(1:hp.showStatsEachNthUpdate:numUpdates,errorGeneratorUpdates+errorDiscriminatorUpdates);
            %title("Generator Objective + Discriminiator Objective")
            %if floor(numUpdates/hp.showStatsEachNthUpdate)>numSmooths
            %   figure(22);
            %   hold on;
            %   plot(1:hp.showStatsEachNthUpdate:numUpdates-numSmooths*hp.showStatsEachNthUpdate,smoothedObjectiveD+smoothedObjectiveG,'r');
            %   hold off;
            %end
            %refresh(22);
            
            
            %figure(23);
            %plot(1:hp.showStatsEachNthUpdate:numUpdates,errorGeneratorUpdates-errorDiscriminatorUpdates);
            %title("Generator Objective - Discriminiator Objective")
            %if floor(numUpdates/hp.showStatsEachNthUpdate)>numSmooths
            %    figure(23);
            %    hold on;
            %    plot(1:hp.showStatsEachNthUpdate:numUpdates-numSmooths*hp.showStatsEachNthUpdate,smoothedObjectiveG-smoothedObjectiveD,'r');
            %    hold off;
            %end
            %refresh(23);
            
            
            %figure(24);
            %plot(1:hp.showStatsEachNthUpdate:numUpdates,errorGeneratorUpdates./errorDiscriminatorUpdates);
            %title("Generator Objective / Discriminiator Objective")
            %if floor(numUpdates/hp.showStatsEachNthUpdate)>numSmooths
            %   figure(24);
            %   hold on;
            %   plot(1:hp.showStatsEachNthUpdate:numUpdates-numSmooths*hp.showStatsEachNthUpdate,smoothedObjectiveG./smoothedObjectiveD,'r');
            %   hold off;
            %end
            %refresh(24);
            %%%%end test plots
            
            figure(11);
            plot(10:hp.showStatsEachNthUpdate:numUpdates,learningRateUpdates);
            title("Learning Rate")
            
            %if strcmp(hp.generator.unitTypes(end),'stochastic')
            %    figure(12);
            %    hist(generatorActivations{length(hp.generator.layerSizes)+1}(:));
            %    title("Histogram of Activations")
            %end
            
            %figure(15);
            %hist(params.generator.biases{length(hp.generator.layerSizes)-1}(:));
            %title("Histogram G's biases (last layer)")
            
            
            showFeatures(hp,params);
            
            figure(13);
            plot(1:hp.showStatsEachNthUpdate:numUpdates,momentumUpdates);
            title("Momentum")
            %figure(101);
            %hist(params.discriminator.weights{1}(:))
            %title("Discriminator weights1")
            %figure(102);
            %hist(params.discriminator.weights{1}(:))
            %title("Discriminator weights2")
            %figure(103);
            %hist(params.discriminator.weights{1}(:))
            %title("Discriminator weights3")
            
            %figure(201);
            %hist(discriminatorUpdate{1}(:))
            %title("Discriminator gradient1")
            %figure(202);
            %hist(discriminatorUpdate{2}(:))
            %title("Discriminator gradient2")
            %figure(203);
            %hist(discriminatorUpdate{3}(:))
            %title("Discriminator gradient3")
            
            %figure(301);
            %hist(params.generator.weights{1}(:))
            %title("generator weights1")
            %figure(302);
            %hist(params.generator.weights{1}(:))
            %title("generator weights2")
            %figure(303);
            %hist(params.generator.weights{1}(:))
            %title("generator weights3")
            
            %figure(401);
            %hist(generatorUpdate{1}(:))
            %title("generator gradient1")
            %figure(402);
            %hist(generatorUpdate{2}(:))
            %title("generator gradient2")
            %figure(403);
            %hist(generatorUpdate{3}(:))
            %title("generator gradient3")
            
            
            %%%%% Produce some samples for visual check
            %save random generator state
            state=rand("state");
            %use always the same noise for demo samples
            rand("state",0);
            %sample minibatch of examples from noise prior p(z)
            noise=sampleNoiseForGenerator(hp);
            %reset RNG state  to what it was
            rand("state",state);
            #forward propagate noise samples through G(z)
            generatorActivations=forwardPropG(hp,params,noise);
            if hp.mnistFlag
                for(fig=1:5)    
                    figure(1000+fig)
                    showDigit(generatorActivations{length(hp.generator.layerSizes)}(fig,:))
                    name=strcat("Generator Sample ",mat2str(fig));
                    title(name);
                end
            else
                figure(1001)
                tmp=generatorActivations{length(hp.generator.layerSizes)}(1:min(100,hp.batchSize),:);
                imshow(tmp)
                name=sprintf("Generator Samples min: %.2f max:%.2f",min(min(tmp)),max(max(tmp)));
                title(name);
            
            end
            
            refresh();

        end %show stats

    end %main loop
end

function ok=checkGradient(hp,params,data)
    %% calculates the numerical gradient and compares it to the analytical gradient
    %same steps as in training, see there for details
    hp.temperature=1.0;
    epsilon=0.0000001;
    ratioLimit=1.1;
    paramsOrig=params;
    numberWeights=3;
    ok=1;
    fprintf("==== Gradient checking ====\n");
    fprintf("---- Discriminator ----\n");
    
    discriminatorGradient={};
    discriminatorBiasGradient={};
    
        
    for layer=1:length(hp.discriminator.layerSizes)-1       
        %when using maxout, choose a random piece to be checked, use 1st (only) piece otherwise
        if (strcmp(hp.discriminator.unitTypes{layer},'maxout'))
            piece=ceil(rand()*hp.discriminator.maxoutPieces)
        else
            piece=1; 
        end
        fprintf("Layer %i  (Unit Type: %s, Dropout:%1.2f)\n",layer,hp.discriminator.unitTypes{layer},hp.discriminator.dropout(layer)*hp.discriminator.useDropout)
        rand("state","reset")
        whichWeights=ceil(rand(1,numberWeights)*length(params.discriminator.weights{layer}(:,:,piece)(:)));
        whichBiases=ceil(rand(1,numberWeights)*length(params.discriminator.biases{layer}(:,:,piece)(:)));
        %original params
        params=paramsOrig;
        noise=sampleNoiseForGenerator(hp);
        generatorActivations=forwardPropG(hp,params,noise);
        rand('state',999);%always use same seed, in case dropout with random mask is used
        hp.discriminator.dropoutmode="train";
        discriminatorStatesRealSamples=forwardPropD(hp,params,data(1:hp.batchSize,:));
        discriminatorStatesNoiseSamples=forwardPropD(hp,params,generatorActivations{length(hp.generator.layerSizes)});
        errorRealSamples=sum(calculateCost(hp, discriminatorStatesRealSamples{length(hp.discriminator.layerSizes)}, target=1,errorType='log(y)'));
        errorNoiseSamples=sum(calculateCost(hp, discriminatorStatesNoiseSamples{length(hp.discriminator.layerSizes)}, target=0,errorType='log(1-y)'));
        [gradientRealSamples,biasGradientRealSamples,errorDerivativesRealSamples]=backpropD(hp,params,1,discriminatorStatesRealSamples,errorType='log(y)');
        [gradientNoiseSamples,biasGradientNoiseSamples,errorDerivativesNoiseSamples]=backpropD(hp,params,1,discriminatorStatesNoiseSamples,errorType='log(1-y)');
        
        
        discriminatorGradient{layer}=gradientNoiseSamples{layer}+gradientRealSamples{layer};
        discriminatorBiasGradient{layer}=biasGradientNoiseSamples{layer}+biasGradientRealSamples{layer};
        
        originalError=errorRealSamples+errorNoiseSamples;
        
        for weightindex=whichWeights
            params=paramsOrig;
            %modified params
            tmpParams=params.discriminator.weights{layer}(:,:,piece);
            tmpParams(weightindex)+=epsilon;
            params.discriminator.weights{layer}(:,:,piece)=tmpParams;
            rand('state',999);%always use same seed, in case dropout with random mask is used
            discriminatorStatesRealSamples=forwardPropD(hp,params,data(1:hp.batchSize,:));
            discriminatorStatesNoiseSamples=forwardPropD(hp,params,generatorActivations{length(hp.generator.layerSizes)});
            errorRealSamples=sum(calculateCost(hp, discriminatorStatesRealSamples{length(hp.discriminator.layerSizes)}, target=1,errorType='log(y)'));
            errorNoiseSamples=sum(calculateCost(hp, discriminatorStatesNoiseSamples{length(hp.discriminator.layerSizes)}, target=0,errorType='log(1-y)'));
            numericalGradient=(((errorRealSamples+errorNoiseSamples)-originalError)/epsilon);
            analyticalGradient=(discriminatorGradient{layer}(:,:,piece)(weightindex));
            ratio=numericalGradient/analyticalGradient;
            fprintf("   weight %i: numerical gradient: %f analytical gradient %f (%f)",weightindex,numericalGradient,analyticalGradient,ratio)
            if (abs(numericalGradient)<0.00001)
                fprintf("(cant decide, gradient very small)\n");
            elseif (ratio>ratioLimit || ratio<(1/ratioLimit))
                fprintf("       ERROR, this one seems off!\n");
                ok=0;
            else
                fprintf(" looks OK\n")
            end
        end
        
        for biasindex=whichBiases
            params=paramsOrig;
            %modified params
            tmpParams=params.discriminator.biases{layer}(:,:,piece);
            tmpParams(biasindex)+=epsilon;
            params.discriminator.biases{layer}(:,:,piece)=tmpParams;
            rand('state',999);%always use same seed, in case dropout with random mask is used
            discriminatorStatesRealSamples=forwardPropD(hp,params,data(1:hp.batchSize,:));
            discriminatorStatesNoiseSamples=forwardPropD(hp,params,generatorActivations{length(hp.generator.layerSizes)});
            errorRealSamples=sum(calculateCost(hp, discriminatorStatesRealSamples{length(hp.discriminator.layerSizes)}, target=1,errorType='log(y)'));
            errorNoiseSamples=sum(calculateCost(hp, discriminatorStatesNoiseSamples{length(hp.discriminator.layerSizes)}, target=0,errorType='log(1-y)'));
        
            numericalGradient=(((errorRealSamples+errorNoiseSamples)-originalError)/epsilon);
            analyticalGradient=(discriminatorBiasGradient{layer}(:,:,piece)(biasindex));
            ratio=numericalGradient/analyticalGradient;
            fprintf("   biases %i: numerical gradient: %f analytical gradient %f (%f)",biasindex,numericalGradient,analyticalGradient,ratio)
            if (abs(numericalGradient)<0.00001)
                fprintf("(cant decide, gradient very small)\n");
            elseif(ratio>ratioLimit || ratio<(1/ratioLimit))
                fprintf("       ERROR, this one seems off!\n");
                ok=0;
            else
                fprintf(" looks OK\n")
            end
        end
        if !ok
            break
        end
    end
    fprintf("Check that errors are not off by far\n");
    fprintf("---- Generator ----\n");
    noise=sampleNoiseForGenerator(hp);
    hp.discriminator.dropoutmode="test";
    
    for layer=1:length(hp.generator.layerSizes)-1       
        fprintf("Layer %i\n",layer)
        whichWeights=ceil(rand(1,numberWeights)*length(params.generator.weights{layer}(:)));
        whichBiases=ceil(rand(1,numberWeights)*length(params.generator.biases{layer}(:)));
    
        %original params
        params=paramsOrig;
        generatorActivations=forwardPropG(hp,params,noise);    
        discriminatorStatesNoiseSamples=forwardPropD(hp,params,generatorActivations{length(hp.generator.layerSizes)});
        errorGeneratorSamples=sum(calculateCost(hp, discriminatorStatesNoiseSamples{length(hp.discriminator.layerSizes)}, target=1,errorType='log(y)'));
        [garbage1,garbage2,errorDerivativesDiscriminator]=backpropD(hp,params,1,discriminatorStatesNoiseSamples,errorType='log(y)');
        [gradient,biasGradient]=backpropG(hp,params,errorDerivativesDiscriminator{1},generatorActivations);
        originalError=errorGeneratorSamples;
        
        dropoutMultiplier=1.0;
        if hp.discriminator.useDropout
            for ll=1:length(hp.discriminator.layerSizes)-1
                dropoutMultiplier*=hp.discriminator.dropout(ll);
            end
        end

        
        for weightindex=whichWeights
            params=paramsOrig;
            %modified params
            params.generator.weights{layer}(weightindex)+=epsilon;
            generatorActivations=forwardPropG(hp,params,noise);    
            discriminatorStatesNoiseSamples=forwardPropD(hp,params,generatorActivations{length(hp.generator.layerSizes)});
            errorGeneratorSamples=sum(calculateCost(hp, discriminatorStatesNoiseSamples{length(hp.discriminator.layerSizes)}, target=1,errorType='log(y)'));
            gradient{layer}(weightindex)*=dropoutMultiplier;
            ratio=((errorGeneratorSamples-originalError)/epsilon)/(gradient{layer}(weightindex));
            fprintf("   weight %i: numerical gradient: %f analytical gradient %f (%f)",weightindex,(errorGeneratorSamples-originalError)/epsilon,gradient{layer}(weightindex),ratio)
            if (abs((errorGeneratorSamples-originalError)/epsilon)<0.00001)
                fprintf("(cant decide, gradient very small)\n");
            elseif(ratio>ratioLimit || ratio<(1/ratioLimit))
                fprintf("       ERROR, this one seems off!\n");
                ok=0;
            else
                fprintf(" looks OK\n")
            end
        end
        
        for biasindex=whichBiases
            params=paramsOrig;
            
            %modified params
            params.generator.biases{layer}(biasindex)+=epsilon;
            generatorActivations=forwardPropG(hp,params,noise);    
            discriminatorStatesNoiseSamples=forwardPropD(hp,params,generatorActivations{length(hp.generator.layerSizes)});
            errorGeneratorSamples=sum(calculateCost(hp, discriminatorStatesNoiseSamples{length(hp.discriminator.layerSizes)}, target=1,errorType='log(y)'));
            biasGradient{layer}(biasindex)*=dropoutMultiplier;
            ratio=((errorGeneratorSamples-originalError)/epsilon)/(biasGradient{layer}(biasindex));
            fprintf("   bias %i: numerical gradient: %f analytical gradient %f (%f)",biasindex,(errorGeneratorSamples-originalError)/epsilon,biasGradient{layer}(biasindex),ratio)
            if (abs((errorGeneratorSamples-originalError)/epsilon)<0.00001)
                fprintf("(cant decide, gradient very small)\n");
            elseif(ratio>ratioLimit || ratio<(1/ratioLimit))
                fprintf("       ERROR, this one seems off!\n");
                ok=0;
            else
                fprintf(" looks OK\n")
            end
        end
        if !ok
            break
        end
    end
                
end
function showSamples(hp,params) 
    noise=sampleNoiseForGenerator(hp);
    generatorActivations=forwardPropG(hp,params,noise);
    figure(1000)
    for fig=1:hp.batchSize
        showDigit(generatorActivations{length(hp.generator.layerSizes)}(fig,:))
        name=strcat("Generator Sample ",mat2str(fig));
        title(name);    
        refresh()
        sleep(1)
    end
end
function samples=sampleG(hp,params,numExamples)
    hp2=hp;
    hp2.batchSize=numExamples;
    noise=sampleNoiseForGenerator(hp2);
    generatorActivations=forwardPropG(hp2,params,noise);
    samples=generatorActivations{length(hp2.generator.layerSizes)};
end

function [samples,metadata]=sampleGANBinary(hp,params,numExamples,population)
    samples=sampleG(hp,params,numExamples)>rand(numExamples,hp.generator.layerSizes(end));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% script part
function model=testGAN()
    more off;
    %load the mnist data
    images=loadMNISTImages("/home/malte/CodeTex/code/data/mnist/train-images-idx3-ubyte")';
    hp.mnistFlag=1;
    %x=images(1:10,:);
    %figure(999)
    %showDigit(x(1,:))

    data=[
        1,1,0,0;
        1,1,0,0;
        1,0,0,0;
        0,0,0,1;
        0,0,1,1;
        0,0,1,1;
        1,0,1,1;
        1,1,1,1;
    ];

    data=images(1:50000,:);
    % ----- Hyper Params----- 
    %process
    doGradientCheck=1;
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
    %general
    hp.maxUpdates=50000;
    hp.minUpdates=10000;
    hp.minEpochs=50;
    hp.maxEpochs=1000;
    hp.showStatsEachNthUpdate=100;
    hp.alphaInitial=.1;  
    hp.alphaMin=0.01;
    hp.alphaDecay=1/1.000025;
    hp.momentumInitial=0.5;
    hp.momentumMax=0.7;
    hp.momentumGrowth=1.000005;
    hp.batchSize=128;
    hp.loglevel=2;
    hp.alphaBiasMultiplier=1;
    hp.initializationNoiseType="normal";
    hp.checkConvergenceEachNUpdates=500;
    hp.uniqueData=0;
    hp.temperatureInitial=1.0;
    hp.temperatureNumUpdatesTillOne=hp.minUpdates;
    hp.temperature=hp.temperatureInitial;
    hp.tuneInitialWeights=0;
    
    %generator specific
    hp.generator.layerSizes=[100,1201,1200,784]; #size of last layer must match data size
    hp.generator.unitTypes={'uniform';'relu';'relu';'sigmoid'};
    hp.generator.initBiasToData=1;
    hp.generator.randomWeightMultiplier=0.1;
    hp.generator.sparseInit=0;
    hp.generator.weightDecay=0.0;
    hp.generator.useL2Norm=0.0;
    hp.generator.alphaScale=0.5; %make LR of generator smaller or bigger

    #discriminator specific
    hp.discriminator.layerSizes=[784,241,240,1]; #size of first layer must match data size
    hp.discriminator.unitTypes={'maxout';'maxout';'sigmoid'};
    hp.discriminator.numTrainingSteps=1;
    hp.discriminator.randomWeightMultiplier=0.01;
    hp.discriminator.headStart=0;
    hp.discriminator.weightDecay=0.0;
    hp.discriminator.useL2Norm=0.0;
    hp.discriminator.useDropout=1;
    hp.discriminator.dropout=[0.8;0.5;0.5]; %rate of remaining units, 1.0=no dropout
    hp.discriminator.maxoutPieces=5;
    hp.discriminator.validationSet=0.2; %use this ratio of training set as valiation data (e.g. 0.2->20%)
    
    % ------ Initial Params -----
    params=initializeGAN(hp,false,false,false);
    if hp.generator.initBiasToData
        params=initializeGeneratorBiases(hp,params,data);
    end
    if doGradientCheck
        if !checkGradient(hp,params, data)
            error("Gradient check failed\n")
        end
    end
    %
    % Train
    params=trainGAN(hp, data,params);
    %showSamples(hp,params);
    %sleep(2)
    model.hp=hp;
    model.params=params;
end
