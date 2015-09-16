% EDA suite - bic.m
% Copyright (C) <2010-2015> Malte Probst
% This file is part of the EDA suite.
%    The EDA suite is free software under the terms of the GNU General
%    Public License, see eda.m for full licence text, or <http://www.gnu.org/licenses/>.

%returns a score value according to the bayesian information criterion BIC, given a node
% score of a node is defined as
%   - H(X|P) * N  -  2^{|P|} * (log(N)/2)
%     ----------          -----------------
%    conditional entropy   regularization term
%  X:node X P: parents of node X
function score=node_score(node,parents,training_data)
    N=size(training_data,1);
    conditional_entropy_term=conditional_entropy(training_data(:,node),training_data(:,parents))*N;
    regularization_term=power(2,length(parents))*(log2(N)/2);
    score=-conditional_entropy_term-regularization_term;
end

%returns entropy of the data
function e=entropy(nodedata)
    px=mean(nodedata);
    e=-(px*log2(px)+(1-px)*log2(1-px));
end
%returns the conditional entropy of the data, assuming that
%the first column of the data is X and the rest of the columns are X's parents
%note: returns the entropy if parent data is empty, i.e. size=(m,0)
function ce=conditional_entropy(nodedata,parentdata)
    ce=0;
    m=length(nodedata);
    tiny=1e-16;
    %initiate first configuration of parents
    lastitem=zeros(1,m);
    
    %sort data of parents and nodes 
    [parentdata,index]=sortrows(parentdata);
    nodedata=nodedata(index);
    
    probs_ones=[];
    probs_zeros=[];
    cond_probs_ones=[];
    cond_probs_zeros=[];
    counts=[];
    %go through all parent combinations in dataset (not: all possible ones!)
    for i=1:size(parentdata,1)
        %fprintf("------Parent i=%i\n",i)
        if i>1
            if all(parentdata(i,:)==lastitem)
                %fprintf("  Same as last parent!\n")
                counts(end)+=1;
                if nodedata(i)==1
                    probs_ones(end)+=1/m;
                    cond_probs_ones(end)+=1;
                else
                    probs_zeros(end)+=1/m;
                    %cond_probs_zeros(end)+=1;
                end
                
                continue
            end
        end
        
        counts=[counts,1];
        if nodedata(i)==1
            probs_ones=[probs_ones,1/m];
            probs_zeros=[probs_zeros,0];
            cond_probs_ones=[cond_probs_ones,1];
        else
            probs_ones=[probs_ones,0];
            probs_zeros=[probs_zeros,1/m];
            cond_probs_ones=[cond_probs_ones,0];
        end
        
        lastitem=parentdata(i,:);
        
    end
    
    
    cond_probs_ones=min(1-tiny,max(tiny,cond_probs_ones./counts));
    cond_probs_zeros=1-cond_probs_ones;

    ce=-sum(probs_ones.*log2(cond_probs_ones) ...
            +probs_zeros.*log2(cond_probs_zeros));
            

end


function cpt=calculate_cpt(nodedata,parentdata)
    m=size(nodedata,1);
    pos_rows=find(nodedata);
    parentdata_pos=parentdata(pos_rows,:);
    neg_rows=find(not(nodedata));
    parentdata_neg=parentdata(neg_rows,:);
    
    tiny=1e-20;
    cpt=zeros(power(2,size(parentdata,2)),1); % calculate only positive probs p, rest is 1-p
    

    %iterate through all parent instances, calculate conditional probability of child being on
    for i=1:power(2,size(parentdata,2))
        %fprintf("calculating entry %i of cpt\n",i)
        %build entry
        entry=de2bi(i-1,size(parentdata,2),"left-msb");
        
        matches_pos=max(tiny,sum(all(parentdata_pos==repmat(entry,size(parentdata_pos,1),1),2)));
        matches_neg=max(tiny,sum(all(parentdata_neg==repmat(entry,size(parentdata_neg,1),1),2)));
        cpt(i)=matches_pos/(matches_pos+matches_neg);
        if matches_pos==tiny && matches_neg==tiny && 0
            cpt(i)=sum(nodedata)/m;
        end
    end

end

