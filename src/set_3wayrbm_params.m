%%%%%parameter presets file
source 3wayrbm.m
model_params.finitializeModel=@initialize3wayRBM;
model_params.ftrainModel=@train3wayRBM;
model_params.fsampleModel=@sample3wayRBM;

model_params.problem_size=eda_params.problem_size;
model_params.numvis=eda_params.problem_size;
model_params.numhid=model_params.numvis*1;
model_params.numfac=model_params.numvis*1;

model_params.numberSamplingCycles=2; 
model_params.numDesiredExamples=20000; 
%%%%%old

%epochs_baseline_at500_samples=1000;
%model_params.n_epochs=epochs_baseline_at500_samples*(500/eda_params.candidate_count) 
model_params.n_epochs=100
model_params.stopAtReconErr=0.01
model_params.alpha=0.001;                               %static learning rate. baseline 0.1
model_params.weight_factor=1;                         %multiply weight gradient 
model_params.visbias_factor=5;      %multiply visbias gradient
model_params.hidbias_factor=5;      %multiply hidbias gradient

model_params.n_gibbs_training=1;               %number of gibbs sampling runs during training (CD-N) baseline 1
model_params.n_gibbs_sampling=100;               %number of gibbs sampling runs for sampling. baseline 50
model_params.use_pcd=false;           %baseline: false (regular CD-maxruns)

model_params.l2_penalty=0.0001;        %L2 regularization, baseline 0
model_params.l2_penalty_biases=0;     %L2 regularization for biases, baseline 0
model_params.l1_penalty=0;            %L1 regularization, baseline 0

model_params.std_W=0.01;            %standard deviation for initial weights W. baseline 0.01

model_params.momentum=0.9;            %momentum for learning weights
model_params.useadaptiveLR=true;              
model_params.batchsize=1024;             %batchsize, 0=> full batch learning
model_params.temperature=1;           %sampling temperature=1

model_params.show_stats_each_n_epochs=1;
model_params.stopAfterPlot=false;

model_params.initialize_sample_from_average=false;
model_params.initialize_sample_from_population=true;

model_params.calculate_stats_each_n_epochs=max(1,floor(model_params.show_stats_each_n_epochs/5));  
 
model_params.use_validation_data=0;
model_params.data_not_binary=false;
model_params
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
%%%%%parameter presets file
source rbm.m
model_params.pretrain.useNormalization=false;
model_params.pretrain.name="RBM";
model_params.pretrain.finitializeModel=@initializeRBM;
model_params.pretrain.ftrainModel=@trainRBM;
model_params.pretrain.fsampleModel=@sampleRBM;
%model_params.pretrain.epochs_baseline_at500_samples=6000; %baseline 500
%model_params.pretrain.n_epochs=model_params.pretrain.epochs_baseline_at500_samples*(500/eda_params.candidate_count) 
model_params.pretrain.n_epochs=2000;       %maximum number of epochs
model_params.pretrain.problem_size=eda_params.problem_size;
model_params.pretrain.numhid=model_params.numfac;
%model_params.pretrain.numhid=floor(model_params.pretrain.problem_size/2);
model_params.pretrain.problem_size=eda_params.problem_size;
model_params.pretrain.alpha=0.1;                               %static learning rate. baseline 0.1
model_params.pretrain.weight_factor=1;                         %multiply weight gradient 
model_params.pretrain.visbias_factor=5;      %multiply visbias gradient
model_params.pretrain.hidbias_factor=5;      %multiply hidbias gradient
model_params.pretrain.n_gibbs_training=1;               %number of gibbs sampling runs during training (CD-N) baseline 1
model_params.pretrain.n_gibbs_sampling=100;               %number of gibbs sampling runs for sampling. baseline 300
model_params.pretrain.use_pcd=false;           %baseline: false (regular CD-maxruns)
model_params.pretrain.l2_penalty=0.005;        %L2 regularization, baseline 0.0001
model_params.pretrain.l2_penalty_biases=0;     %L2 regularization for biases, baseline 0
model_params.pretrain.l1_penalty=0;            %L1 regularization, baseline 0
model_params.pretrain.std_W=0.001;            %standard deviation for initial weights W. baseline 0.01
model_params.pretrain.momentum=0.5;            %momentum for learning weights
model_params.pretrain.momentum2=0.9;            %momentum for learning weights
model_params.pretrain.batchsize=200;             %batchsize, 0=> full batch learning
model_params.pretrain.variablebatches=4;        %4 batches when learning variable batch sizes
model_params.pretrain.variablebatchuntil=800;        %full batch learning up to N samples
model_params.pretrain.minbatchsize=10;           %when using variable batch sizes
model_params.pretrain.temperature=1;           %sampling temperature=1
model_params.pretrain.temperature_hiddens=1; %sampling temperature for hiddens when sampling candidate solutions
model_params.pretrain.increase_hidden_temperature=1; %factor to increase hidden temperature each epoch
model_params.pretrain.temperature_hiddens_max=1; %maximum hidden temp
model_params.pretrain.show_stats_each_n_epochs=50;
model_params.pretrain.stopAfterPlot=false;
model_params.pretrain.initialize_sample_from_average=false;
model_params.pretrain.initialize_sample_from_population=true;
model_params.pretrain.calculate_free_energy_each_n_epochs=25;  
model_params.pretrain.use_validation_data=0;
model_params.pretrain.data_not_binary=false;
model_params.pretrain.name=cstrcat(model_params.pretrain.name,"_T",mat2str(model_params.pretrain.temperature_hiddens),"I",mat2str(model_params.pretrain.increase_hidden_temperature),"M",mat2str(model_params.pretrain.temperature_hiddens_max));

