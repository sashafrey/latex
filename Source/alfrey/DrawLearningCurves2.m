function DrawLearningCurves2()
    nRepeats = 100;
    filename_base = 'results';
    aggr = [];
    output_path = 'C:\Storage\vft11ccas\Articles\alfrey\NIPS-2013\eps\';
    
    for iRepeat=1:nRepeats
        filename = sprintf('%s%i.mat', filename_base, iRepeat);
        if (~exist(filename, 'file'))
            continue;
        end
        
        load(filename);
        
        if (isempty(aggr))
            % setup all structures
            for iTask = 1:length(params.tasknames)
                aggr.(params.tasknames{iTask}).trainError = [];
                aggr.(params.tasknames{iTask}).testError = [];
                aggr.(params.tasknames{iTask}).trainErrorMCC = [];
                aggr.(params.tasknames{iTask}).testErrorMCC = [];
                aggr.(params.tasknames{iTask}).count = zeros(length(params.ratios), 1);
            end
        end
        
        for iResult = 1:results.Count
            result = results.Data{iResult};
            if (result.failed) 
                fprintf('failed')
                continue;
            end
            
            iRatio = find(params.ratios == result.ratio);
            aggr.(result.taskname).count(iRatio) = aggr.(result.taskname).count(iRatio) + 1;
            ind = aggr.(result.taskname).count(iRatio);
            aggr.(result.taskname).trainError(iRatio, ind) = result.trainError;
            aggr.(result.taskname).testError(iRatio, ind) = result.testError;
            aggr.(result.taskname).trainErrorMCC(iRatio, ind) = result.trainErrorMCC;
            aggr.(result.taskname).testErrorMCC(iRatio, ind) = result.testErrorMCC;
        end
    end
    
    for iTask = 1:length(params.tasknames)
        if (iTask == 15) continue; end;
            
        taskname = params.tasknames{iTask};
        for iRatio = 1:size(aggr.(taskname).trainError, 1)
            trainError(iRatio) = mean(aggr.(taskname).trainError(iRatio, :));
            testError(iRatio) = mean(aggr.(taskname).testError(iRatio, :));
            trainErrorMCC(iRatio) = mean(aggr.(taskname).trainErrorMCC(iRatio, :));
            testErrorMCC(iRatio) = mean(aggr.(taskname).testErrorMCC(iRatio, :));
        end
        
        chartType = 1;
        len = size(aggr.(taskname).trainError, 1);
        clf
        switch chartType
            case 1
                % both train and test learning curves
                hold on
                plot(params.ratios(1:len), trainErrorMCC, 'g-v');
                plot(params.ratios(1:len), testErrorMCC, 'b-s');
                plot(params.ratios(1:len), trainError, 'r-d')
                plot(params.ratios(1:len), testError, 'k-x');
                axis tight
                legend boxon
                legend('ERM train error', 'ERM test error', 'LR train error', 'LR test error', 'location', 'Best')
                legend boxoff
                hold off
            case 2         
                % overfitting curves
                plot(params.ratios(1:len), testErrorMCC - trainErrorMCC, 'b:', ...
                     params.ratios(1:len), testError - trainError, 'b');
                V = axis;
                V(3) = 0;
                axis(V);
            case 3
                % only test curves
                plot(params.ratios(1:len), testErrorMCC, 'b:', ...
                     params.ratios(1:len), testError, 'b');
        end

        % Reduce margins
        % [left bottom width height] are normalized to [0, 1]
        % Use negative values for left and bottom to trim off left/bottom margins
        % Use &lt;1 values for width and height to trim off right/top margins
        %set(gca, 'OuterPosition', [-0.01 -0.005 0.99 0.99]);
        set(gcf,'PaperPositionMode','auto')
        
        %taskname2 = taskname;
        %taskname2(taskname2 == '_') = ' ';
        %title(taskname2);
        
        xlabel('training ratio')
        ylabel('error')
        saveas(gcf,sprintf('%s%s.eps', output_path, taskname),'eps2c');        
    end
end