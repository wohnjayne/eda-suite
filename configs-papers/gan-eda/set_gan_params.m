%%%%%parameter presets file
source 'gan.m'
model_params.name='GAN';
model_params.finitializeModel=@initializeGAN;
model_params.ftrainModel=@trainGAN;
model_params.fsampleModel=@sampleGANBinary;

if (exist('instance', 'var') == 0)
    instance='1';
end
hp.instance=instance;
if (exist('savePath', 'var') == 0)
    savePath='/tmp/';
end
if (exist('seed', 'var') == 0)
    seed=-1; %-1=don't used fixed seed, anything else: use as seed
end
if ~ seed==-1
    rand('state',seed);
end
if strcmp(version,"3.2.4")!=1
    graphics_toolkit ("gnuplot")
end

%========general settings==========
%convergence checks are done only after max(minUpdates+minEpochs)
%learning rate and momentum are decreased/increased s.t. they reach their final values after min(maxUpdats,maxEpochs)
model_params.maxUpdates=15000; 
model_params.minUpdates=1500;
model_params.maxEpochs=500;
model_params.minEpochs=10;
model_params.showStatsEachNthUpdate=200;
model_params.alphaInitial=0.025;  
model_params.alphaMin=model_params.alphaInitial*0.2;
model_params.momentumInitial=0.5;
model_params.momentumMax=0.8;
model_params.batchSize=64;
model_params.loglevel=2;
model_params.alphaBiasMultiplier=1; %bias updates are usually stronger
model_params.initializationNoiseType='uniform'; %"normal" or "uniform"
model_params.checkConvergenceEachNUpdates=200;
model_params.uniqueData=0; %delete copies of data points in training data
model_params.mnistFlag=0;
%adjust initial weights, s.t. input to non-linearity is in a reasonable area (individually, for each layer)
model_params.tuneInitialWeights=0;

%experimental, dont use (sigmoid temperatures)
model_params.temperatureInitial=1.0;
model_params.temperatureNumUpdatesTillOne=model_params.minUpdates;
model_params.temperature=model_params.temperatureInitial;

%========generator specific settings==========
model_params.generator.layerSizes=[eda_params.problem_size,eda_params.problem_size,eda_params.problem_size]; %size of last layer must match data size
model_params.generator.unitTypes={'uniform';'relu';'sigmoid'}; % "sigmoid|relu|stochastic", first layer must be 'uniform' (noise distribution)
model_params.generator.initBiasToData=0; %initialize biases to mean values
model_params.generator.randomWeightMultiplier=.2;
model_params.generator.weightDecay=0.001;
model_params.generator.sparseInit=0;
model_params.generator.alphaScale=.25; %make LR of generator smaller or bigger

model_params.generator.alphaAutoAdjust=0;%not implemented
model_params.generator.targetGradientWeightRatio=0.05;%not implemented
model_params.generator.useL2Norm=0.0;%not implemented properly

%========discriminator specific settings==========
model_params.discriminator.layerSizes=[floor(eda_params.problem_size*1),floor(eda_params.problem_size*.5),1]; %size of first layer must match data size, size of last layer must be 1
model_params.discriminator.unitTypes={'relu';'sigmoid'}; %sigmoid, relu or maxout, type of last layer must be sigmoid
model_params.discriminator.numTrainingSteps=1; %number of discr. updates per gen. update
model_params.discriminator.randomWeightMultiplier=.1;
model_params.discriminator.weightDecay=0.001;
model_params.discriminator.headStart=0;
model_params.discriminator.validationSet=0.1; %0.1 = 10% of data for validation set
model_params.discriminator.maxValiDifference=0.95; %break if loss(train) is at 0.xx of loss(vali) 

%dropout
model_params.discriminator.useDropout=0;
model_params.discriminator.dropout=[1.0;1.0;1.0]; %rate of remaining units, 1.0=no dropout

%maxout (must be configured in unit types as well
model_params.discriminator.maxoutPieces=1; %1: no maxout

model_params.discriminator.alphaAutoAdjust=1;%not implemented
model_params.discriminator.targetGradientWeightRatio=0.05;%not implemented
model_params.discriminator.useL2Norm=0.0; %not implemented properly

model_params

