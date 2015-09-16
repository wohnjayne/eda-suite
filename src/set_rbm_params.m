%%%%%parameter presets file for Restricted Boltzmann Machine
source rbm.m
%set name, and link according functions of model
model_params.name='RBM';
model_params.finitializeModel=@initializeRBM;
model_params.ftrainModel=@trainRBM;
model_params.fsampleModel=@sampleRBM;

%maximum number of epochs
model_params.n_epochs=1000;       

%number of visible neurons
model_params.problem_size=eda_params.problem_size;
%number of hidden neurons
model_params.numhid=model_params.problem_size/2;
%use normalization as in Yichuan Tang and Ilya Sutskever 2011
model_params.useNormalization=true;
%static learning rate. baseline 0.05
model_params.alpha=0.05;
%multiply weight gradient (unused, leave 1)                               
model_params.weight_factor=1;        
%multiply visbias gradient (large fan-in -> larger value)                
model_params.visbias_factor=10;      
%multiply hidbias gradient (large fan-in -> larger value)
model_params.hidbias_factor=10;      

%number of gibbs sampling runs during training (CD-N) baseline 1
model_params.n_gibbs_training=1;               
%use persistent contrastive divergence 
model_params.use_pcd=false;           

%L2 regularization aka weight decay
model_params.l2_penalty=0.0001;      
%L2 for biases? (nope!)
model_params.l2_penalty_biases=0;    
%L1 regularization, baseline 0
model_params.l1_penalty=0;            

%standard deviation for initial weights W. baseline 0.001
model_params.std_W=0.001;

%use validation data? 0.1=10% validation data
model_params.use_validation_data=0.1;

%momentum for learning weights (momentum2: will be used after some training)
model_params.momentum=0.5;
model_params.momentum2=0.8;
%batch size for mini batch learning
model_params.batchsize=64;


%number of gibbs sampling runs for sampling
model_params.n_gibbs_sampling=25;     

%how to initialize the visibles when sampling
model_params.initialize_sample_from_average=false;
model_params.initialize_sample_from_population=true;

%when to show stats
model_params.show_stats_each_n_epochs=50;
model_params.calculate_free_energy_each_n_epochs=2;  %also for other stats
model_params.stopAfterPlot=false;


%some more variables controlling mini batch size (for small samples, see code for details)
model_params.variablebatches=4;        %4 batches when learning variable batch sizes
model_params.variablebatchuntil=800;        %full batch learning up to N samples
model_params.minbatchsize=10;           %when using variable batch sizes
%experimental: settings controlling sampling temperature
model_params.temperature=1;           %sampling temperature=1
model_params.temperature_hiddens=1; %sampling temperature for hiddens when sampling candidate solutions
model_params.increase_hidden_temperature=1; %factor to increase hidden temperature each epoch
model_params.temperature_hiddens_max=1; %maximum hidden temp
%experimental
model_params.data_not_binary=false;

 
model_params.name=cstrcat(model_params.name,"_T",mat2str(model_params.temperature_hiddens),"I",mat2str(model_params.increase_hidden_temperature),"M",mat2str(model_params.temperature_hiddens_max));

