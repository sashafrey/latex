function QEpsTheory = NoisedSetOverfittingCalc(PrepareNSO, eps)
    Check(exist('eps', 'var'));
    
    L = PrepareNSO.L;
    ell = PrepareNSO.ell;
    m1 = PrepareNSO.m1;
    m_r = PrepareNSO.m_r;
    r = PrepareNSO.r;
    d = PrepareNSO.d;
    
    s_eps = floor(ell / L * ((m1 + r) - eps * (L - ell)));    
    
    if (PrepareNSO.type == 0)
        s_eps = min(s_eps, floor(ell / L * (m1 + r)));
        if (s_eps < 0)
            QEpsTheory = 0;
        else
        	QEpsTheory = sum(PrepareNSO.data(1 : (s_eps + 1)));
        end
    elseif (PrepareNSO.type == 1)
        D = CnkCalc(m_r, r);
        d = min(d, D);
        if (D <= L)
            Check(isfield(PrepareNSO, 'dataRatio'));
        end        

        total = 0;
        for i = 0:min(m1, ell)
            for j = 0:min(m_r, ell - i)
                tau = PrepareNSO.dataTau(i + 1, j + 1);
                lowIndex = max(s_eps - i + 1, 0) + 1;
                topIndex = min(r, j) + 1;
                Ni = sum(PrepareNSO.dataNi(i + 1, j + 1, lowIndex : topIndex));
        
                if (Ni == D)
                    total = total + tau;
                else
                    if (d == D)
                        continue; % given we are now under condition "Ni < D".
                    end
                    
                    if (D <= L)
                        % using exact values
                        total = total + tau * PrepareNSO.dataRatio(Ni + 1);
                    else
                        % using some approximation.
                        total = total + tau * power(Ni / D, d);
                    end
                end
            end
        end    

        QEpsTheory = 1 - total;
    else
        throw 'Unknown type of PrepareNSO';
    end
end
