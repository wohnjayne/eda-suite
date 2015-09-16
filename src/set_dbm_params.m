%%%%%parameter presets file
source dbm.m
model_params.name="DBM";
model_params.finitializeModel=@initializeDBM;
model_params.ftrainModel=@trainDBM;
model_params.fsampleModel=@sampleDBM;

%maximum number of epochs for DBM
model_params.n_epochs=1000;     
%maximum number of epochs for RBM1  
model_params.n_epochs_rbm1=1000;
%maximum number of epochs for RBM2
model_params.n_epochs_rbm2=1000;

model_params.problem_size=eda_params.problem_size;
model_params.numhid=model_params.problem_size;
model_params.numpen=model_params.problem_size;
model_params.problem_size=eda_params.problem_size;

model_params.batchsize=100;             %batchsize, 0=> full batch learning

%how to smooth the reconstruction error
model_params.smoothNEpchs=20;
model_params.minGenerations=20;
%break, if in <lastInterval> of epochs, gain was
% less than <minGainInLastInterval> of total gain in reconstruction error
model_params.lastInterval=0.33;
model_params.minGainInLastInterval=0.1;

