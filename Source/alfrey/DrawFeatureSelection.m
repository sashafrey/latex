function DrawFeatureSelection()
    clf
    nRepeats = 128;
    filename_base = 'results';
    aggr = [];
    
    for iRepeat=1:nRepeats
        filename = sprintf('%s%i.mat', filename_base, iRepeat);
        if (~exist(filename, 'file'))
            continue;
        end
        
        skip = false;
        try
            load(filename);
        catch
            skip = true;
        end
        
        if (skip)
            continue;
        end

        if (isempty(aggr))
            % setup all structures
            for iTask = 1:length(params.tasknames)
                taskname = params.tasknames{iTask};
                aggr.(taskname).trainError = [];
                aggr.(taskname).testError = [];
                aggr.(taskname).count = zeros(params.nFeatures(iTask), 1);
            end
        end

        for iResult = 1:results.Count
            result = results.Data{iResult};
            if (result.failed) 
                %fprintf('failed')
                continue;
            end

            aggr.(result.taskname).count(result.iFeatures) = aggr.(result.taskname).count(result.iFeatures) + 1;
            ind = aggr.(result.taskname).count(result.iFeatures);
            aggr.(result.taskname).trainError(result.iFeatures, ind) = result.trainError;
            aggr.(result.taskname).testError(result.iFeatures, ind) = result.testError;
            aggr.(result.taskname).trainErrorMCC(result.iFeatures, ind) = result.trainErrorMCC;
            aggr.(result.taskname).testErrorMCC(result.iFeatures, ind) = result.testErrorMCC;
        end
    end
    
    for iTask = 1:length(params.tasknames)
        subplot(2,4,iTask);
        taskname = params.tasknames{iTask};
        nFeatures = size(aggr.(taskname).trainError, 1);
        trainError = [];
        testError = [];
        trainErrorMCC = [];
        testErrorMCC = [];
        for iFeatures = 1:nFeatures
            trainError(iFeatures) = mean_not_isnan(aggr.(taskname).trainError(iFeatures, :));
            testError(iFeatures) = mean_not_isnan(aggr.(taskname).testError(iFeatures, :));
            trainErrorMCC(iFeatures) = mean_not_isnan(aggr.(taskname).trainErrorMCC(iFeatures, :));
            testErrorMCC(iFeatures) = mean_not_isnan(aggr.(taskname).testErrorMCC(iFeatures, :));
        end
        
        chartType = 1;
        len = size(aggr.(taskname).trainError, 1);
        %clf
        switch chartType
            case 1
                % both train and test learning curves
                hold on
                plot(1:nFeatures, trainErrorMCC, 'g-v');
                plot(1:nFeatures, testErrorMCC, 'b-s');
                plot(1:nFeatures, trainError, 'r-d')
                plot(1:nFeatures, testError, 'k-x');
                axis tight
                legend boxon
                legend('ERM train error', 'ERM test error', 'LR train error', 'LR test error', 'location', 'Best')
                legend boxoff
                hold off
            case 2
                % 'predicted' overfitting
                hold on
                plot(1:nFeatures, trainError + testErrorMCC - trainErrorMCC, 'b-s');
                plot(1:nFeatures, trainError, 'r-d')
                plot(1:nFeatures, testError, 'k-x');
                axis tight
                legend boxon
                legend('LR predicted error', 'LR train error', 'LR test error', 'location', 'Best')
                legend boxoff
                hold off
            case 3         
                % overfitting curves
                hold on
                plot(1:nFeatures, testErrorMCC - trainErrorMCC, 'b-s');
                plot(1:nFeatures, testError - trainError, 'k-x');
                axis tight
                V = axis;
                V(3) = 0;
                axis(V);
                
                legend boxon
                legend('ERM overfit.', 'LR overfit.', 'location', 'Best')
                legend boxoff
                hold off
            case 4
                % test errors only
                hold on
                plot(1:nFeatures, testErrorMCC, 'b-s');
                plot(1:nFeatures, testError, 'k-x');
                axis tight
                V = axis;
                V(3) = 0;
                axis(V);
                
                legend boxon
                legend('ERM overfit.', 'LR overfit.', 'location', 'Best')
                legend boxoff
                hold off
        end

        % Reduce margins
        % [left bottom width height] are normalized to [0, 1]
        % Use negative values for left and bottom to trim off left/bottom margins
        % Use &lt;1 values for width and height to trim off right/top margins
        %set(gca, 'OuterPosition', [-0.01 -0.005 0.99 0.99]);
        %set(gcf,'PaperPositionMode','auto')
        
        taskname2 = taskname;
        taskname2(taskname2 == '_') = ' ';
        title(taskname2);
        
        %xlabel('iFeatures')
        %ylabel('error')
        %saveas(gcf,sprintf('%s%s.eps', output_path, taskname),'eps2c');        
    end
end

function result = mean_not_isnan(vec)
    result = mean(vec(~isnan(vec)));
end