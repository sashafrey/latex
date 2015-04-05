function CalcOverfittingQualityMetrics(adjTables)
    names = fieldnames(adjTables);
    for name_cell = names'
        name = name_cell{1};
        adjTable = adjTables.(name);
        adjNames = fieldnames(adjTable);
        for adjName_cell = adjNames'
            adjName = adjName_cell{1};
            adj = adjTable.(adjName);
            p1 = mean(adj.pKoef(:, 1));
            p2 = mean(adj.pKoef(:, 2));
            n1 = mean(adj.nKoef(:, 1));
            n2 = mean(adj.nKoef(:, 2));
            av = mean([p1 p2 n1 n2]);
            
            fprintf('overfitting\t%s\t%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f', name, adjName, ...
                mean(adj.pKoef(:, 1)), ...
                mean(adj.pKoef(:, 2)), ...
                mean(adj.nKoef(:, 1)), ...
                mean(adj.pKoef(:, 2)), ...
                av);
            fprintf('\r');
        end
        
               
        for adjName_cell = adjNames'
            adjName = adjName_cell{1};
            adj = adjTable.(adjName);
            p1 = countOrderedPairs(adj.pKoef(:, 1), adjTable.CV.pKoef(:, 1));
            p2 = countOrderedPairs(adj.pKoef(:, 2), adjTable.CV.pKoef(:, 2));
            n1 = countOrderedPairs(adj.nKoef(:, 1), adjTable.CV.nKoef(:, 1));
            n2 = countOrderedPairs(adj.nKoef(:, 2), adjTable.CV.nKoef(:, 2));
            av = mean([p1 p2 n1 n2]);
            
            fprintf('correlation\t%s\t%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f', name, adjName, p1, p2, n1, n2, av);
            fprintf('\r');
        end
    end
end

function c = countOrderedPairs(v1, etalon)
    c = 0;
    t = 0;
    Check(length(v1) == length(etalon));
    for i = 1:length(v1)
        for j = 1:length(v1)
            if (etalon(i) ~= etalon(j))
                if (v1(i) == v1(j))
                    continue;
                end
                t = t + 1;
                
                etalonLeq = (etalon(i) < etalon(j));
                v1Leq = (v1(i) < v1(j));
                if (etalonLeq == v1Leq)
                    c = c + 1;
                end
            end
        end
    end
    
    c = c / t;
end