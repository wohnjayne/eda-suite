% EDA suite - fitnessCNF.m
% Copyright (C) <2010-2015> Malte Probst
% This file is part of the EDA suite.
%    The EDA suite is free software under the terms of the GNU General
%    Public License, see eda.m for full licence text, or <http://www.gnu.org/licenses/>.

function fitness=fitnessCNF(population,cnf)
%returns the fitness of the population with respect to the given cnf 
numclauses=size(cnf,1);
%last item in clause is clause weight
clauselen=size(cnf,2)-1;
fitness=zeros(size(population,1),1);
for i=1:numclauses
    clause_satisfied=ones(size(population,1),clauselen);
    for j=1:clauselen
        if cnf(i,j)==0
            continue
        elseif cnf(i,j)>0
            clause_satisfied(:,j)=population(:,cnf(i,j));
        else 
            clause_satisfied(:,j)=not(population(:,-cnf(i,j)));
        end
    end
    fitness+=any(clause_satisfied,2);
end
