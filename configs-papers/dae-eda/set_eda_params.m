%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters for the EDA (everything not model- or selection specific)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%maximum number of generations in EDA
eda_params.max_generations=150;     
eda_params.stopAfterNGenerationsWithoutImprovement=50;
%total size of population (before selection, i.e., |parents+candidates|)
eda_params.popsize=2000;             
%number of new candidate solutions per generation (use with caution, might break if other than 1/2)
eda_params.candidate_count=eda_params.popsize/2;      
%use local search before starting the EDA (all individuals will be in local optimum)
eda_params.localSearch=0;
%use XOR bit flip mask, to avoid any model-induced bias towards zeros or ones
eda_params.bitFlipMask=1; 
%count the number of unique fitness evaluations instead of all fitness evaluations
eda_params.useFitnessTable=1;
%log level. 0 gives nothing, 1 gives basic info, 2 plots lots of stuffs, including graphics (setting is passed to model)
eda_params.loglevel=2;
eda_params.show_stats_each_n_generations=1;
eda_params.stopAfterPlot=false;
%stop, if optimum is reached (to minimize cpu time for testing. in real scenario, method won't realize it found the optimum)
eda_params.stopOnOptimum=true; 
%retrieve some very specific measures we needed for a paper, leave=0
eda_params.computeDiversityMeasures=0;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters for test problem 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%note: 
% these parameters should be kept separately (not in eda_params)
% however, for backwards compatibility we kept it like this
%  

%problem size (number of bits)
eda_params.problem_size=64;

% which benchmark problem to solve, must be one of:
% TRAPS ONEMAX MAXSAT NK HIFF
%  (and, if applicable, the corresponding instance has to be configured below)
eda_params.problem_name="HIFF";

if strcmp(eda_params.problem_name, "TRAPS")
    %%%%%%%%%%%%%%%%%% DECEPTIVE TRAPS %%%%%%%%% 
    eda_params.ffunction=@fitnessTrapOpt1;  %fitness function to be used
    eda_params.fitness_param1=5;            %blocklength, in case of traps
    eda_params.fitness_max=eda_params.problem_size; 
    eda_params.problem_name=strcat(mat2str(eda_params.fitness_param1),'-TRAPS')
elseif strcmp(eda_params.problem_name, "ONEMAX")
    %%%%%%%%%%%%%%%%%% OneMax %%%%%%%%% 
    eda_params.ffunction=@fitnessOneMax;  %fitness function to be used
    eda_params.fitness_max=eda_params.problem_size;
    eda_params.fitness_param1=0;
elseif strcmp(eda_params.problem_name, "MAXSAT")
    %%%%%%%%%%%%%%%%%% MaxSat %%%%%%%%% 
    %a maxsat definition file in DIMACS format, we used those from 
    % http://www.maxsat.udl.cat/14/benchmarks/ms_random.tgz
    cnf=readDIMACS("../resources/s3v80c1000-1.cnf"); 
    eda_params.ffunction=@fitnessCNF;
    eda_params.fitness_param1=cnf;            %the cnf, must be defined
    %the best achievable fitness
    eda_params.fitness_max=956;
elseif strcmp(eda_params.problem_name, "NK")
    %%%%%%%%%%%%%%%%% NK landscapes %%%%%%%%%%%%%%%%
    %nk instance definition file, we used those from http://medal-lab.org/files/nk-instances.tar.gz
    nkDef=readNK("../resources/nk_30_4.1");
    eda_params.ffunction=@fitnessNK;
    eda_params.fitness_param1=nkDef;            %the NK def, must be defined
    eda_params.fitness_max=nkDef.maxFitness;
elseif strcmp(eda_params.problem_name, "HIFF")
    %%%%%%%%%%%%%%%%%% HIFF %%%%%%%%% 
    if power(2,floor(log(eda_params.problem_size)/log(2)))!=eda_params.problem_size
        error("Population size for HIFF must be a power of 2")
    end
    eda_params.ffunction=@fitnessHIFF;  %fitness function to be used
    eda_params.fitness_param1=0;            %unused
    eda_params.fitness_max=eda_params.problem_size*(log2(eda_params.problem_size)+1); 
else
    error("Unknown problem type")
end

%for backwards compatibility
if strcmp(version,"3.2.4")!=1
    graphics_toolkit ("gnuplot")
end
