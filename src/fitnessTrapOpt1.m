% EDA suite - fitnessTrapOpt1.m
% Copyright (C) <2010-2015> Malte Probst
% This file is part of the EDA suite.
%    The EDA suite is free software under the terms of the GNU General
%    Public License, see eda.m for full licence text, or <http://www.gnu.org/licenses/>.

function fitness=fitnessTrapOpt1(population,k)
% k=blocklength
%computes the fitness of all individuums in the population
% each individuum consists of a number of equally sized blocks
% the fitness is the sum of the fitness of all blocks
% within each block, the fitness is deceptive, with all zeros
% being the optimum, valued k, all other combinations are valued by
% number of ones in the block (max k-1)
fitness=zeros(size(population,1),1);
for block = 1:(size(population,2)/k)
    %all ones great, deceptive towards zeros
    opts=all(population(:,(block-1)*k+1:block*k)');
    fitness=fitness+(k-1)-sum(population(:,(block-1)*k+1:block*k),2)+opts'*(k+1);
    
end


