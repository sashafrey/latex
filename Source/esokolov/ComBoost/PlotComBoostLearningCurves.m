function PlotComBoostLearningCurves(cvLog, cvIdx, funcNames)
    styles = {'b', 'r', 'g', 'm', 'k'};

    h = maximizeFigure(figure);
    hold on;
    
    for iFunc = 1:size(cvLog.ensembleWeights, 1)
        ensembleWeights = cvLog.ensembleWeights{iFunc, cvIdx};
        currCurve = nan(size(ensembleWeights, 1), 1);
        
        for i = 1:length(currCurve)
            Y_pred = comBoost_classify(cvLog.testSamples{cvIdx}.X, ...
                ensembleWeights(1:i, :));
            currCurve(i) = mean(Y_pred == cvLog.testSamples{cvIdx}.Y);
        end
        
        plot(currCurve, styles{iFunc});
    end
    
    grid on;
    legend(funcNames);
end