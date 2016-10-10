%%%%%parameter presets file
source 'ae.m'
model_params.name='AE';
model_params.finitializeModel=@initializeAE;
model_params.ftrainModel=@trainAE;
model_params.fsampleModel=@sampleAEBinary;

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
#corruption level, layer size, sample from random

%general
model_params.maxEpochs=2000;
model_params.minEpochs=50;
model_params.calculateStatsEachNthEpoch=5;
model_params.showStatsEachNthEpoch=50;
model_params.randomWeightMultiplier=0.005;
model_params.alphaInitial=.1;  
model_params.alphaBiasMultiplier=5;
model_params.momentumInitial=0.3;
model_params.momentumForBiases=0;
model_params.weightDecay=0.001; %WD could be necesary if corruption level is low
model_params.batchSize=100;
model_params.loglevel=0;
model_params.initializationNoiseType="normal";
model_params.validationSet=0;
model_params.numSamplingRounds=10; 
model_params.layerSizes=[eda_params.problem_size,eda_params.problem_size,eda_params.problem_size];
model_params.unitTypes={'sigmoid';'sigmoid';'sigmoid'}; 

model_params.useCreativitySampling=0;
model_params.creativitySamplingMixNClasses=2; #sample from N classes (instead of 1)
model_params.creativitySamplingNumClasses=10; #number of classes
model_params.creativitySamplingMixThisProportionOfSamples=0.5;

model_params.mnistFlag=0;
model_params.breakOnRelativeGain=0.05;
model_params.sampleFromRandom=1;
model_params.corruptInput=0.2; %Denoising autoencoder, more corruption seems to lead to faster convergence of EDA (perhaps, lower corruption leads more often to identiy function being learned...?)
model_params.uniqueData=0;
model_params.dropoutRate=0;
model_params

