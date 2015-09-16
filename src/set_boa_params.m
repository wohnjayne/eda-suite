%%%%%parameter presets file
source boa.m
source bic.m

model_params.name="BOA";
model_params.finitializeModel=@initializeBOA;
model_params.ftrainModel=@trainBOA;
model_params.fsampleModel=@sampleBOA;

model_params.problem_size=eda_params.problem_size;

model_params.show_stats_each_n_epochs=50;
model_params.stopAfterPlot=false;
