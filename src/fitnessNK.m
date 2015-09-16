% EDA suite - fitnessNK.m
% Copyright (C) <2010-2015> Malte Probst
% This file is part of the EDA suite.
%    The EDA suite is free software under the terms of the GNU General
%    Public License, see eda.m for full licence text, or <http://www.gnu.org/licenses/>.

function fitness=fitnessNK(population,nkDef)
% returns the fitness wrt given definition of NK landscape
% 
fitness=zeros(size(population,1),1);
for i = 1:size(population,2)
    fitness+=nkDef.fitness(i,bi2de(population(:,nkDef.neighbors(i,:)),"left-msb")+1)';
end


