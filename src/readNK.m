% EDA suite - readNK.m
% Copyright (C) <2010-2015> Malte Probst
% This file is part of the EDA suite.
%    The EDA suite is free software under the terms of the GNU General
%    Public License, see eda.m for full licence text, or <http://www.gnu.org/licenses/>.

function nkDef=readNK(filename)
    %reads a nk definition

    fid=fopen(filename);

    #get first line, contains n and k
    line=strtrim(fgets(fid));
    elements=strsplit(line,' ');
    n=str2num(elements{1})
    k=str2num(elements{2})
    
    %read all k+1 neighbors for each of the n bits (one per line)
    neighbors=[];
    for i=1:n
        for j=1:k+1
            neighbors(i,j)=str2num(strtrim(fgets(fid)))+1;
        end
    end
    
    %read fitness table (one number per line) (2^(k+1) values per bit)
    fitness=[];
    for i=1:n
        for j=1:power(2,k+1)
            fitness(i,j)=str2num(strtrim(fgets(fid)));
        end
    end
    nkDef.n=n;
    nkDef.k=k;
    nkDef.neighbors=neighbors;
    nkDef.fitness=fitness;
    %last line contains max fitness
    line=strtrim(fgets(fid));
    elements=strsplit(line,' ');
    nkDef.maxFitness=str2num(elements{2});
end
