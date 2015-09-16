% EDA suite - selectionRandom.m
% Copyright (C) <2010-2015> Malte Probst
% This file is part of the EDA suite.
%    The EDA suite is free software under the terms of the GNU General
%    Public License, see eda.m for full licence text, or <http://www.gnu.org/licenses/>.

function [winners,loosers,kept]=selectionRandom(p,s,fitness,selectionPressure)
[garbage,index]=sort(rand(p,1));
winners=index(1:(p/2));
loosers=index((p/2)+1:p);
kept=winners;
