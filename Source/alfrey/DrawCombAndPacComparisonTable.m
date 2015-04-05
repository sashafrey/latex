function DrawCombAndPacComparisonTable()
    nRepeats = 100;
    pack = true;
    
    aggr = [];
    resultsFileName = 'results';
    
    for iRepeat=1:nRepeats
        filename = sprintf('%s%i.mat', resultsFileName, iRepeat);
        if (~exist(filename, 'file'))
            continue;
        end
        
        load(filename);
        
        if (isempty(aggr))
            % setup all structures
            for iTask = 1:length(params.tasknames)
                aggr.(params.tasknames{iTask}).trainError = 0;
                aggr.(params.tasknames{iTask}).testError = 0;
                aggr.(params.tasknames{iTask}).trainErrorMCC = 0;
                aggr.(params.tasknames{iTask}).testErrorMCC = 0;
                aggr.(params.tasknames{iTask}).CombVCInverse = [];
                aggr.(params.tasknames{iTask}).CombSCInverse = [];
                aggr.(params.tasknames{iTask}).CombESInverse = [];
                aggr.(params.tasknames{iTask}).CombAFInverse = [];
                aggr.(params.tasknames{iTask}).pacBayesDIbound = 0;
                aggr.(params.tasknames{iTask}).pacBayesDDbound = 0;
                aggr.(params.tasknames{iTask}).count = 0;
            end
        end
        
        for iResult = 1:results.Count
            result = results.Data{iResult};
            
            if (pack) 
                results.Data{iResult}.CombSC = '[removed]';
                results.Data{iResult}.CombVC = '[removed]';
                results.Data{iResult}.CombES = '[removed]';
                results.Data{iResult}.CombAF = '[removed]';
                results.Data{iResult}.CombEps = '[removed]';
            end
            
            if (result.failed || isempty(result.CombVCInverse) || isempty(result.CombSCInverse) || isnan(result.CombVCInverse) || isnan(result.CombSCInverse)) 
                fprintf('failed')
                continue;
            end

            aggr.(result.taskname).trainError = aggr.(result.taskname).trainError + result.trainError;
            aggr.(result.taskname).testError = aggr.(result.taskname).testError + result.testError;
            aggr.(result.taskname).trainErrorMCC = aggr.(result.taskname).trainErrorMCC + result.trainErrorMCC;
            aggr.(result.taskname).testErrorMCC = aggr.(result.taskname).testErrorMCC + result.testErrorMCC;
            
            aggr.(result.taskname).CombVCInverse(end+1) = result.CombVCInverse;
            aggr.(result.taskname).CombSCInverse(end+1) = result.CombSCInverse;
            aggr.(result.taskname).CombESInverse(end+1) = result.CombESInverse;
            aggr.(result.taskname).CombAFInverse(end+1) = result.CombAFInverse;
            aggr.(result.taskname).pacBayesDIbound = aggr.(result.taskname).pacBayesDIbound + result.pacBayesDIbound;
            aggr.(result.taskname).pacBayesDDbound = aggr.(result.taskname).pacBayesDDbound + result.pacBayesDDbound;
            
            aggr.(result.taskname).count = aggr.(result.taskname).count + 1;
        end
        
        if (pack) 
            save(filename, 'results', 'params');
        end
    end
    
    type = 1;
    fprintf('task\ttrainErr\ttestErr\toverfitt\tcomb\tcombVC\tcombSC\tcombES\tcombAF\tPAC DI\tPAC DD\n');
    for iTask = 1:length(params.tasknames)
        taskname = params.tasknames{iTask};        
        trainError = aggr.(taskname).trainError / aggr.(taskname).count;
        testError = aggr.(taskname).testError / aggr.(taskname).count;
        trainErrorMCC = aggr.(taskname).trainErrorMCC / aggr.(taskname).count;
        testErrorMCC = aggr.(taskname).testErrorMCC / aggr.(taskname).count;

        [CombVCInverse, ~, CombVCInverseCI, ~] = normfit(aggr.(taskname).CombVCInverse);
        [CombSCInverse, ~, CombSCInverseCI, ~] = normfit(aggr.(taskname).CombSCInverse);
        [CombESInverse, ~, CombESInverseCI, ~] = normfit(aggr.(taskname).CombESInverse);
        [CombAFInverse, ~, CombAFInverseCI, ~] = normfit(aggr.(taskname).CombAFInverse);
        pacBayesDIbound = aggr.(taskname).pacBayesDIbound / aggr.(taskname).count;
        pacBayesDDbound = aggr.(taskname).pacBayesDDbound / aggr.(taskname).count;
        
        type=1;
        if (type == 1)
            fprintf('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n', ...
                taskname, trainError, testError, testError - trainError, testErrorMCC - trainErrorMCC, ...
                CombVCInverse, CombSCInverse, CombESInverse, CombAFInverse, pacBayesDIbound, pacBayesDDbound);
        elseif (type == 2)
            fprintf('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%s\t%s\t%s\t%s\t%.3f\t%.3f\n', ...
                taskname, trainError, testError, testError - trainError, testErrorMCC - trainErrorMCC, ...
                CI2STR(CombVCInverseCI), CI2STR(CombSCInverseCI), CI2STR(CombESInverseCI), CI2STR(CombAFInverseCI), pacBayesDIbound, pacBayesDDbound);
        elseif (type == 3)
            fprintf('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%s\t%s\t%s\t%s\t%.3f\t%.3f\n', ...
                taskname, trainError, testError, testError - trainError, testErrorMCC - trainErrorMCC, ...
                PM2STR(CombVCInverseCI), PM2STR(CombSCInverseCI), PM2STR(CombESInverseCI), PM2STR(CombAFInverseCI), pacBayesDIbound, pacBayesDDbound);
        end
    end
end

function s = CI2STR(ci)
    left = ci(1);
    right = ci(2);
    s = sprintf('[%.4f, %.4f]', left, right);
end

function s = PM2STR(ci)
    left = ci(1);
    right = ci(2);
    s = sprintf('[%.3f, %.4f]', mean([left, right]), (right - left) / 2);
end