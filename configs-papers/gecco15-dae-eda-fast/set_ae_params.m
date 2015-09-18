%%%%%parameter presets file
source 'ae.m'
model_params.name='AE';
model_params.finitializeModel=@initializeAE;
model_params.ftrainModel=@trainAE;
model_params.fsampleModel=@sampleAEBinary;

%maxiumum number of epochs to learn
model_params.maxEpochs=2000;
%minumum number of epochs/gradient updates, before checking for convergence (or rising error)
model_params.minEpochs=20;
model_params.minUpdates=0; 

%initialize random weights  "normal" or "uniform" -ly, and scale down factor
model_params.initializationNoiseType="normal";
model_params.randomWeightMultiplier=0.01;

%learning rate (currently constant)
model_params.alphaInitial=.2;  
%learning rate for biases is multiplied by this factor
model_params.alphaBiasMultiplier=5;
%momentum, for weights, and switch for biases
model_params.momentumInitial=0;
model_params.momentumForBiases=0;
%L2 regularization aka weight decay
model_params.weightDecay=0.001; %not too critical

%batch size for mini batch learning
model_params.batchSize=100;
%Denoising autoencoder, how much data shall be corrupted with salt+pepper noise? 0.1 -> 10%
model_params.corruptInput=0.1; 

%use validation set
model_params.validationSet=0.1;

%number of sampling rounds when drawing samples from the model
model_params.numSamplingRounds=10; 
%initialize samples randomly?
model_params.sampleFromRandom=1;

%layer sizes (number of layers must be odd)
model_params.layerSizes=[eda_params.problem_size,eda_params.problem_size,eda_params.problem_size];
%layer types. last layer must be sigmoid, others can be sigmoid or relu (first one is actually irrelevant)
model_params.unitTypes={'sigmoid';'sigmoid';'sigmoid'}; 

%how often to calculate/show the stats
model_params.calculateStatsEachNthEpoch=5;
model_params.showStatsEachNthEpoch=10;

%when to stop training (0.05-> break, if less then 5% of gain in recon err was achieved in last 25% epochs)
model_params.breakOnRelativeGain=0.05;



%%%%%%%experimental settings

%clusters the data before training with k-means. 
%then assigns a class-condidional bias while training. When sampling, multiple
%classes can be activated, hoping to mix features of these classes. Not properly tested.
% (might actually work better with gated autoencoder)
model_params.useCreativitySampling=0;
model_params.creativitySamplingMixNClasses=2; #sample from N classes (instead of 1)
model_params.creativitySamplingNumClasses=10; #number of classes
model_params.creativitySamplingMixThisProportionOfSamples=0.5;

%data is mnist
model_params.mnistFlag=0;

%only train on unique data
model_params.uniqueData=0;
%use dropout (not fully implemented!)
model_params.dropoutRate=0;
%track, how strong the identity function is being learned
model_params.trackIdentityFunction=0;


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

model_params

