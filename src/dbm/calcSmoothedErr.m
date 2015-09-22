% EDA suite - calcSmoothedErr.m
% Copyright (C) <2010-2015> Malte Probst
% This file is part of the EDA suite.
%    The EDA suite is free software under the terms of the GNU General
%    Public License, see eda.m for full licence text, or <http://www.gnu.org/licenses/>.

function smoothedError=calcSmoothedError(errors,hp)
    epochs=size(errors,2);
    smoothedError=0;
    i=0;
    for i=0:min(epochs,hp.smoothNEpchs)-1
        smoothedError+=errors(end-i);
    end
    smoothedError/=(i+1);
end
