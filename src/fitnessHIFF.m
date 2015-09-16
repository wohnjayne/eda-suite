% EDA suite - fitnessHIFF.m
% Copyright (C) <2010-2015> Malte Probst
% This file is part of the EDA suite.
%    The EDA suite is free software under the terms of the GNU General
%    Public License, see eda.m for full licence text, or <http://www.gnu.org/licenses/>.

function fitness=fitnessHIFF(population)
% returns the fitness
% 
    fitness=zeros(size(population,1),1);
    workingSet=population;
    for level=1:log2(size(population,2))+1
        #evaluate fitness in this level
        fitness+=sum(workingSet>=0,2)*power(2,level-1);
        
        #create representation for next level
        if level<log2(size(population,2))+1
            tmp=zeros(size(workingSet,1),size(workingSet,2)*0.5)-1;
            for i=1:size(tmp,2)
                tmp(find(all([workingSet(:,2*i-1),workingSet(:,2*i)]==1,2)),i)=1;
                tmp(find(all([workingSet(:,2*i-1),workingSet(:,2*i)]==0,2)),i)=0;
            end
            workingSet=tmp;
        end
    end
end


