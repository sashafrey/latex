function bound = getCombBound_OneAlg(currAlg, sourcesVects, L, l, eps, boundType)
    % boundType:
    % 'SC_sources'
    % 'SC_classic'
    % 'CCV_classic'
    % 'EOF_classic'
    
    m = currAlg.errCnt;
    u = currAlg.upperNeighsCnt;
    
    if strcmp(boundType, 'SC_sources')
        bound = Inf;
        for j = 1:size(sourcesVects, 1)
            a = sum(currAlg.errVect < sourcesVects(j, :)');
            b = sum(currAlg.errVect > sourcesVects(j, :)');
            currBound = 0;
            for t = 0:min(a, b)
                currBound = currBound + ((CnkCalc(b, t) * ...
                    CnkCalc(L - u - b, l - u - t)) / CnkCalc(L, l)) * ...
                    hhDistr(L - u - b, l - u - t, m - b, floor((l/L) * ...
                    (m - eps * (L - l))) - t);
            end
            bound = min(bound, currBound);
        end
    else
        a_all = sum(repmat(currAlg.errVect', [size(sourcesVects, 1) 1]) < sourcesVects, 2);
        bestSource = find(a_all == min(a_all), 1);
        q = sum(currAlg.errVect > sourcesVects(bestSource, :)');
        
        L_a = L - u - q;
        l_a = l - u;
        m_a = m - q;
        k = L - l;
        
        if strcmp(boundType, 'SC_classic')
            %bound = (CnkCalc(L_a, l_a, binCoef) / CnkCalc(L, l, binCoef)) * ...
            %        hhDistr(L_a, l_a, m_a, floor((l/L) * (m - eps * (L - l))), binCoef);
            bound = 0;
            a = sum(currAlg.errVect < sourcesVects(bestSource, :));
            b = sum(currAlg.errVect > sourcesVects(bestSource, :));
            for t = 0:min(a, b)
                bound = bound + ((CnkCalc(b, t) * ...
                    CnkCalc(L - u - b, l - u - t)) / CnkCalc(L, l)) * ...
                    hhDistr(L - u - b, l - u - t, m - b, floor((l/L) * ...
                    (m - eps * (L - l))) - t);
            end
        elseif strcmp(boundType, 'CCV_classic')
            bound = (1 / k) * (CnkCalc(L_a, l_a) / CnkCalc(L, l)) * ...
                (((L_a - l_a) / L_a) * m_a + q);
        elseif strcmp(boundType, 'EOF_classic')
            bound = (CnkCalc(L_a, l_a) / CnkCalc(L, l)) * ...
                (q / k + m_a * ((L_a - l_a) / (k * L_a) - l_a / (l * L_a)));
        end
    end
end
