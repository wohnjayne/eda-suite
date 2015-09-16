function smoothedError=calcSmoothedError(errors,hp)
    epochs=size(errors,2);
    smoothedError=0;
    i=0;
    for i=0:min(epochs,hp.smoothNEpchs)-1
        smoothedError+=errors(end-i);
    end
    smoothedError/=(i+1);
end
