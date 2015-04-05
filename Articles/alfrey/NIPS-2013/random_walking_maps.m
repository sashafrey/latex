	% as part of GPUTests.cs
    for i=1:2
        if (i==1) pTransition = 1; else pTransition = 0.5; end;
        maxAlgs = 2000;
        [W_rw, isSource_rw] = GPU_RunRandomWalking(sessionId, w0, maxAlgs, 10*maxIters, maxErrors, pTransition, 0, 1);
        [~, ev_rw, ec_rw, hashes_rw] = GPU_CalcAlgs(sessionId, W_rw);
        evd_rw = calcDist(ev_rw, 0, 0);
        subplot(4,2,[0 2 4]+i);
        imagesc(evd_rw);            
        colormap(flipud(gray));
        subplot(4,2,6+i);
        plot(ec_rw)
    end