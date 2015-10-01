if strcmp(version,"3.2.4")!=1
    graphics_toolkit ("gnuplot")
end
%%%%%parameter presets file
eda_params.max_generations=150;     %maximum number of generations in EDA
eda_params.popsize=500;              %total size of population
eda_params.candidate_count=eda_params.popsize/2;      %number of new candidate solutions per generation
eda_params.problem_size=75;          %length of individual solution in bits
eda_params.localSearch=0;
eda_params.localOptimaMode=0;
eda_params.bitFlipMask=1; 
eda_params.computeDiversityMeasures=0;
eda_params.useFitnessTable=1;
%%%%%%%%%%%%%%%%%% TRAPS %%%%%%%%% Uncomment to use, comment all other sections
%eda_params.ffunction=@fitnessTrapOpt1;  %fitness function to be used
%eda_params.fitness_param1=5;            %blocklength, in case of traps
%eda_params.fitness_max=eda_params.problem_size; 
%eda_params.problem_name=strcat(mat2str(eda_params.fitness_param1),'-TRAPS')

%%%%%%%%%%%%%%%%%% Hierarchical TRAPS %%%%%%%%% Uncomment to use, comment all other sections
%eda_params.ffunction=@fitnessHierarchicalTrap;  %fitness function to be used
%eda_params.fitness_param1=3;            %blocklength, in case of traps
%eda_params.fitness_max=eda_params.problem_size*log(eda_params.problem_size)/log(eda_params.fitness_param1=3); 
%eda_params.problem_name=cstrcat(mat2str(eda_params.fitness_param1),"-HIERARCHICALTRAPS")

%%%%%%%%%%%%%%%%%% OneMax %%%%%%%%% Uncomment to use, comment all other sections
eda_params.ffunction=@fitnessOneMax;  %fitness function to be used
eda_params.fitness_max=eda_params.problem_size;
eda_params.fitness_param1=0;
eda_params.problem_name="ONEMAX";

%%%%%%%%%%%%%%%%%% 3-deceptive %%%%%%%%% Uncomment to use, comment all other sections
%eda_params.ffunction=@fitnessDec3;  %fitness function to be used
%eda_params.fitness_max=eda_params.problem_size/3;
%eda_params.fitness_param1=0;
%eda_params.problem_name="3-DEC";

%%%%%%%%%%%%%%%%%% MaxSat %%%%%%%%% Uncomment to use, comment all other sections
#files
% ../../data/dimacs-cnf/max-sat2012/ms_random/max3sat/80v/s3v80c1000-1.cnf   best 956
%cnf=readDIMACS("../../data/dimacs-cnf/max-sat2012/ms_random/max3sat/80v/s3v80c1000-1.cnf");
%eda_params.ffunction=@fitnessCNF;
%eda_params.fitness_param1=cnf;            %the cnf, must be defined
%eda_params.fitness_max=956;#size(cnf,1);

%%%%%%%%%%%%%%%%% NK landscapes %%%%%%%%%%%%%%%%
%nkDef=readNK("../nk-instances/nk_30_4.1");
%eda_params.ffunction=@fitnessNK;
%eda_params.fitness_param1=nkDef;            %the NK def, must be defined
%eda_params.fitness_max=nkDef.maxFitness;
%eda_params.problem_name='NK'

%%%%%%%%%%%%%%%%%% HIFF %%%%%%%%% Uncomment to use, comment all other sections
%eda_params.ffunction=@fitnessHIFF;  %fitness function to be used
%eda_params.fitness_param1=0;            %unused
%eda_params.fitness_max=eda_params.problem_size*(log2(eda_params.problem_size)+1); 
%eda_params.problem_name='HIFF';

eda_params.loglevel=0;
eda_params.show_stats_each_n_generations=1;
eda_params.stopAfterPlot=false;

eda_params.keepModelParamsAfterSelectionRate=0; %1=keep existing model, 0=unlearn all parameters

eda_params.stopOnOptimum=true; %stop, if optimum is reached (to minimize cpu time for testing. in "real" scenario, method won't realize it found the optimum)
eda_params.stopAfterNGenerationsWithoutImprovement=50;
