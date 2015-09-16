% EDA suite - readDIMACS.m
% Copyright (C) <2010-2015> Malte Probst
% This file is part of the EDA suite.
%    The EDA suite is free software under the terms of the GNU General
%    Public License, see eda.m for full licence text, or <http://www.gnu.org/licenses/>.

function cnf=readDIMACS(filename)
%reads a cnf from a in DIMACS format, returns matrix of clauses (one row per clause)
%sample file: filename="/home/malte/Dropbox/Malte/Uni\\ Mainz/code/data/dimacs-cnf/test/uf20-01.cnf"

command=cstrcat("cat ",filename,"|grep -v c |grep -v %|grep -v ^0> /tmp/.octave_eda_cnf.tmp");
[output,status]=system(command);

fid=fopen("/tmp/.octave_eda_cnf.tmp");
n=0;
while true
    if  fgets(fid)!=-1
        n+=1;
    else
        break
    end
end
%n=fskipl(fid,Inf);
frewind(fid);
maxTerms=1;
cnf=zeros(n,maxTerms);
for i=1:n
    line=strtrim(fgets(fid));
    elements=strsplit(line,' ');
    if length(elements)>maxTerms
        cnf_old=cnf;
        cnf=zeros(n,length(elements));
        cnf(1:i-1,1:maxTerms)=cnf_old(1:i-1,:);
        maxTerms=length(elements);
    end
    for j=1:length(elements)
        cnf(i,j)=str2num(elements{j});
    end
end
if all(cnf(end,:)==0)
    cnf=cnf(1:end-1,:);
end
