function PrepareNSO = NoisedSetOverfittingPrepare(L, ell, m1, m_r, r, d)
    if (d < 0) % special trick to use all dataset.
        d = CnkCalc(m_r, r);
    end
    
    PrepareNSO.L = L;
    PrepareNSO.ell = ell;
    PrepareNSO.m1 = m1;
    PrepareNSO.m_r = m_r;
    PrepareNSO.r = r;
    PrepareNSO.d = d;
    
    m = m1 + r;
    CLl = CnkCalc(L, ell);
    CLl = max(CLl, 1); % can't be lower than 1. ToDo: decide, how to handle ell >= L.

    if (d == 1)
        PrepareNSO.type = 0;
        sMax = floor(ell / L * m);
        PrepareNSO.data = zeros(sMax, 1);
        
        for i = 0:sMax
            PrepareNSO.data(i + 1) = CnkCalc(m, i) * CnkCalc(L - m, ell - i) / CLl;
        end
    else
        PrepareNSO.type = 1;
        dataTau = zeros(min(m1, ell), min(m_r, ell));
        dataNi  = zeros(min(m1, ell), min(m_r, ell), r);

        for i = 0:min(m1, ell)
            for j = 0:min(m_r, ell - i)
                dataTau(i + 1, j + 1) = CnkCalc(m1, i) * CnkCalc(m_r, j) * CnkCalc(L - m1 - m_r, ell - i - j) / CLl;
                for x = 0 : min(r, j)
                    dataNi (i + 1, j + 1, x + 1) = CnkCalc(j, x) * CnkCalc(m_r - j, r - x);
                end
            end
        end
        
        PrepareNSO.dataTau = dataTau;
        PrepareNSO.dataNi  = dataNi;
        
        D = CnkCalc(m_r, r);
        if (D <= L) % here L = typical size of cnk tables.
            dataRatio = zeros(L + 1, 1);
            CDd = CnkCalc(D, d);
            for Ni = d : min(D, L)
                dataRatio(Ni + 1) = CnkCalc(Ni, d) / CDd;
            end
            
            PrepareNSO.dataRatio = dataRatio;
        end
    end
end
