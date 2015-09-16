% Version 1.000 
%
% Code provided by Ruslan Salakhutdinov 
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

% This program trains Restricted Boltzmann Machine in which
% visible, binary, stochastic pixels are connected to
% hidden, binary, stochastic feature detectors using symmetrically
% weighted connections. Learning is done with 1-step Contrastive Divergence.   
% The program assumes that the following variables are set externally:
% maxepoch  -- maximum number of epochs
% numhid    -- number of hidden units 
% batchdata -- the data that is divided into batches (numcases numdims numbatches)
% restart   -- set to 1 if learning starts from beginning 


%Note, Malte Probst, 2015: some small changes in the code (removed everything
% to do with classes, to make this entirely unsupervised)
if restart ==1,

epsilonw_0      = 0.02;   % Learning rate for weights 
epsilonvb_0     = 0.02;   % Learning rate for biases of visible units 
epsilonhb_0     = 0.02;   % Learning rate for biases of hidden units 

weightcost  = 0.001;   
initialmomentum  = 0.5;
finalmomentum    = 0.9;

file=strcat("fullmnistvh",model.myrun);
load(file)
vishid_l0 = vishid;
hidbiases_l0 = hidbiases;
visbiases_l0 = visbiases;
[numcases numdims numbatches]=size(batchdata);
numdims_l0 = numdims;

numdims = numhid;
numhid = numpen;

  restart=0;
  epoch=1;

% Initializing symmetric weights and biases. 
  vishid     = 0.01*randn(numdims, numhid);
  hidbiases  = zeros(1,numhid);
  visbiases  = zeros(1,numdims);

  poshidprobs = zeros(numcases,numhid);
  neghidprobs = zeros(numcases,numhid);
  posprods    = zeros(numdims,numhid);
  negprods    = zeros(numdims,numhid);
  vishidinc  = zeros(numdims,numhid);
  hidbiasinc = zeros(1,numhid);
  visbiasinc = zeros(1,numdims);

  numlab=10; 
  labhid = 0.01*randn(numlab,numhid);
  labbiases  = zeros(1,numlab);
  labhidinc =  zeros(numlab,numhid);
  labbiasinc =  zeros(1,numlab);

epoch=1;


end

epocherrors=[];
epocherrors_smooth=[];
for epoch = epoch:maxepoch
 epocherrors=[epocherrors,0];
 fprintf(1,'epoch %d\r',epoch); 

  CD = ceil(epoch/20);

  epsilonw = epsilonw_0/(1*CD);
  epsilonvb = epsilonvb_0/(1*CD);
  epsilonhb = epsilonhb_0/(1*CD);
 
 errsum=0;
 for batch = 1:numbatches,
 if model_params.loglevel >1
    fprintf(1,'epoch %d batch %d\r',epoch,batch); 
 end

%%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  data_l0 = batchdata(:,:,batch);
  poshidprobs_l0 = 1./(1 + exp(-data_l0*(2*vishid_l0) - repmat(2*hidbiases_l0,numcases,1)));
  data = poshidprobs_l0 > rand(numcases,numdims);
  %targets = batchtargets(:,:,batch); 

  bias_hid= repmat(hidbiases,numcases,1);
  bias_vis = repmat(2*visbiases,numcases,1);
  %bias_lab = repmat(labbiases,numcases,1);
  
  poshidprobs = 1./(1 + exp(-data*(vishid) -  bias_hid));    
  posprods    = data' * poshidprobs;
  %posprodslabhid = targets'*poshidprobs;

  poshidact   = sum(poshidprobs);
  posvisact = sum(data);
  %poslabact   = sum(targets);

%%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  poshidprobs_temp = poshidprobs;

%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  for cditer=1:CD
    poshidstates = poshidprobs_temp > rand(numcases,numhid);

    %totin = poshidstates*labhid' + bias_lab;
    %neglabprobs = exp(totin);
    %neglabprobs = neglabprobs./(sum(neglabprobs,2)*ones(1,numlab));

    %xx = cumsum(neglabprobs,2);
    %xx1 = rand(numcases,1);
    %neglabstates = neglabprobs*0;
    %for jj=1:numcases
    %   index = min(find(xx1(jj) <= xx(jj,:)));
    %   neglabstates(jj,index) = 1;
    %end
    %xxx = sum(sum(neglabstates)) ;

    negdata = 1./(1 + exp(-poshidstates*(2*vishid)' - bias_vis));
    negdata = negdata > rand(numcases,numdims); 
    poshidprobs_temp = 1./(1 + exp(-negdata*(vishid) -  bias_hid));
 end 
  neghidprobs = poshidprobs_temp;     

  negprods  = negdata'*neghidprobs;
  neghidact = sum(neghidprobs);
  negvisact = sum(negdata); 
  %neglabact = sum(neglabstates);
  %negprodslabhid = neglabstates'*neghidprobs;

%%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  err= sum(sum( (data-negdata).^2 ));
  errsum = err + errsum;
  epocherrors(epoch)+=err;
  
   if epoch>5,
     momentum=finalmomentum;
   else
     momentum=initialmomentum;
   end;

%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    vishidinc = momentum*vishidinc + ...
                epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);
    %labhidinc = momentum*labhidinc + ...
    %        epsilonw*( (posprodslabhid-negprodslabhid)/numcases - weightcost*labhid); 


    visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
    hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);
    %labbiasinc = momentum*labbiasinc + (epsilonvb/numcases)*(poslabact-neglabact);



    vishid = vishid + vishidinc;
    %labhid = labhid + labhidinc;

    visbiases = visbiases + visbiasinc;
    hidbiases = hidbiases + hidbiasinc;
    %labbiases = labbiases + labbiasinc;

 end 
%%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
  
  tmpParams=model_params;
  tmpParams.smoothNEpchs=10;
  tmpParams.minGenerations=20;
  epocherrors_smooth=[epocherrors_smooth,calcSmoothedError(epocherrors/(numcases*numbatches*numdims),tmpParams)];
  
  if model_params.loglevel>1 && mod(epoch,10)==0
  fprintf(1, 'epoch %4i error %2.4f  \n', epoch, errsum/(numcases*numbatches*numdims)); 
     figure(888);
     plot(1:epoch,epocherrors./(numcases*numbatches*numdims));
     title("RBM2 recon err");
    refresh();
  end
  
  stopTraining=checkTerminateTraining(epocherrors_smooth,tmpParams);
  
  
  if stopTraining
    break;
  end
  
%%%% Look at the test scores %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %if rem(epoch,10)==0
  %  err =   testerr(testbatchdata,testbatchtargets,vishid_l0,hidbiases_l0,...
  %      vishid,visbiases,hidbiases,labhid,labbiases);
  %  fprintf(1,'Number of misclassified test examples: %d out of 10000 \n',err);
  %end

end;
save ( strcat("fullmnistpo",model.myrun),  "vishid", "hidbiases", "visbiases" )
