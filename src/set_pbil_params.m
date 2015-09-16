%%%%%parameter presets file
source pbil.m
model_params.name="PBIL";
model_params.finitializeModel=@initializePBIL;
model_params.ftrainModel=@trainPBIL;
model_params.fsampleModel=@samplePBIL;

model_params.problem_size=eda_params.problem_size;

%static learning rate. baseline 0.02
model_params.alpha=.02;                               

model_params.show_stats=true;
model_params.stopAfterPlot=false;

%more time for pbil (and less frequent stats)
eda_params.max_generations=eda_params.max_generations*20; 
eda_params.stopAfterNGenerationsWithoutImprovement=eda_params.stopAfterNGenerationsWithoutImprovement*20;
eda_params.show_stats_each_n_generations=eda_params.show_stats_each_n_generations*20;


