function [terminate,reason]=checkTerminateTraining(smoothedError,hp)
    terminate=0;
    reason="none";
    if size(smoothedError,2)>=hp.minGenerations 
        %check if reconErr grows
        if smoothedError(end)>smoothedError(end-1) && smoothedError(end)>smoothedError(end-2)
            terminate=1;
            reason=" Recon Err grows, stopping training\n";
        end
        
        %check if last interval gave significant gain
        totalGain=smoothedError(1)-smoothedError(end);
        
        intervalMark=floor(size(smoothedError,2)*(1-hp.lastInterval));
        lastGain=smoothedError(intervalMark)-smoothedError(end);
        
        lastGainRelative=lastGain/totalGain;
        
        if lastGainRelative<hp.minGainInLastInterval
            terminate=1;
            reason=" Last relative gain was enough\n";
        end
    end
        
    %check if last error was too small 
    if smoothedError(end)<0.01
        terminate=1;
        reason=" Last error was small enough\n";
    
    end
    
end
