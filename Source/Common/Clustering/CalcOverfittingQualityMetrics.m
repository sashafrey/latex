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
                mean(adj.nKoef(:, 2)), ...
                av);
            fprintf('\r');
        end
        
               
        for adjName_cell = adjNames'
            adjName = adjName_cell{1};
            adj = adjTable.(adjName);
            %func = @countOrderedPairs;
            %func = @kendelTau;
            func = @kendelTau_b;
            p1 = func(adj.pKoef(:, 1), adjTable.CV.pKoef(:, 1));
            p2 = func(adj.pKoef(:, 2), adjTable.CV.pKoef(:, 2));
            n1 = func(adj.nKoef(:, 1), adjTable.CV.nKoef(:, 1));
            n2 = func(adj.nKoef(:, 2), adjTable.CV.nKoef(:, 2));
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

function [tau_b] = kendelTau_b(v1, etalon) 
    [~, tau_b] = kendelTau(v1, etalon);
end

function [tau, tau_b] = kendelTau(v1, etalon)
    conc = 0;
    diso = 0;
    Check(length(v1) == length(etalon));
    n = length(v1);
    for i = 1:n
        for j = i:n
            if (v1(i) == v1(j)) 
                continue;
            end
            
            if (etalon(i) == etalon(j))
                continue;
            end
            
            if ((v1(i) > v1(j)) && (etalon(i) > etalon(j)))
                conc = conc + 1;
                continue;
            end
            
            if ((v1(i) < v1(j)) && (etalon(i) < etalon(j)))
                conc = conc + 1;
                continue;
            end
            
            diso = diso + 1;
        end
    end
    n0 = 0.5 * n * (n-1);
    tau = (conc - diso) / (n0);
    
    u_v1 = unique(v1);
    u_etalon = unique(etalon);
    
    n1 = 0;
    for val=u_v1'
        tied_count = sum(v1 == val);
        n1 = n1 + tied_count * (tied_count - 1)/2;
    end
    
    n2 = 0;
    for val=u_etalon'
        tied_count = sum(etalon == val);
        n2 = n2 + tied_count * (tied_count - 1)/2;
    end
    
    tau_b = (conc - diso) / sqrt((n0 - n1) * (n0 - n2));    
end