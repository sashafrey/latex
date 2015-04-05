function RulesOverfittingPlot(pTrain, nTrain, pTest, nTest, class, color, fromCenter)
    nIters = size(pTrain, 1);
    
    p0 = mean(pTest(:, class));
    n0 = mean(nTest(:, class));
    pAvg = mean(pTest(:, class) - pTrain(:, class));
    nAvg = mean(nTest(:, class) - nTrain(:, class));
    label = sprintf('(p, n) = (%.3f %.3f), delta = (%.3f %.3f)', p0, n0, pAvg, nAvg);

    if (fromCenter)
        pTest = pTest - pTrain;
        pTrain = pTrain .* 0;
        
        nTest = nTest - nTrain;
        nTrain = nTrain .* 0;        
    end

    
    pMax = max([pTrain(:, class); pTest(:, class)]);
    nMax = max([nTrain(:, class); nTest(:, class)]);
    pMin = min([pTrain(:, class); pTest(:, class)]);
    nMin = min([nTrain(:, class); nTest(:, class)]);

    pAvg = (pAvg - pMin) / (pMax - pMin);
    nAvg = (nAvg - nMin) / (nMax - nMin);
        
    pTrain = (pTrain - pMin) ./ (pMax - pMin);
    pTest = (pTest - pMin) ./ (pMax - pMin);
    nTrain = (nTrain - nMin) ./ (nMax - nMin);
    nTest = (nTest - nMin) ./ (nMax - nMin);
   
    for i = 1:nIters
        annotation('arrow',[pTrain(i, class), pTest(i, class)],[nTrain(i, class), nTest(i, class)],'Color',color);
    end
    
    if (fromCenter)
        annotation('arrow',[-pMin / (pMax - pMin), pAvg],[-nMin / (nMax - nMin), nAvg],'Color','b');
        annotation('textarrow',[0.95, 0.99],[0.95, 0.99],'String',label);
    end
end