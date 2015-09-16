% EDA suite - selectionTournament.m
% Copyright (C) <2010-2015> Malte Probst
% This file is part of the EDA suite.
%    The EDA suite is free software under the terms of the GNU General
%    Public License, see eda.m for full licence text, or <http://www.gnu.org/licenses/>.

function winners=selectionTournament(popsize,nWinners,fitness)
%----tournament selection with all individuals----
popsize=floor(popsize/2)*2;
[garbage,index]=sort(rand(popsize,1));
nWinners=floor(nWinners);
firsthalfwinners=fitness(index(1:popsize/2))>fitness(index(popsize/2+1:popsize));
secondhalfwinners=not(firsthalfwinners);
winners=cat(1,index(find(firsthalfwinners==1)),index(find(secondhalfwinners==1)+popsize/2))(1:nWinners);
%----end tournament
