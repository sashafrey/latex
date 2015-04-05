function q = runExperiment(L, n, seed)
N=50;
N1 = 10;

parfor i=1:N1
    fprintf(1,'Выполняется %d шаг из %d\n',i,N1);
    p = CreateSample(L,n,seed+i,'c');
    qq = CalcQMod(p,N);
    pC10 = Get10Perc(p, qq);%добавили 10% ограничители
    
    p = CreateSample(L,n,seed+i,'m');
    qq = CalcQMod(p,N);
    pN10 = Get10Perc(p, qq);

    pCAdd = CreateSample(L,n,seed+i+1,'c');%создаем контрольные выборки
    pC2 = AddSample(pC10, pCAdd);
    %pNAdd = CreateSample(L,n,seed+i+1,'m');
    pN2 = AddSample(pN10, pCAdd);
    qC10(i) = CalcOneStep(pC2, 1:L, L+1:2*L);
    qN10(i) = CalcOneStep(pN2, 1:L, L+1:2*L);
    pC(i) = pC2;
    pN(i) = pN2;
    pC2.nAdd = 0; pC2.pAdd = 0;
    pN2.nAdd = 0; pN2.pAdd = 0;
    
    qC(i) = CalcOneStep(pC2, 1:L, L+1:2*L);
    qN(i) = CalcOneStep(pN2, 1:L, L+1:2*L);
end;
q.qC10 = qC10;
q.qN10 = qN10;
q.pC = pC;
q.pN = pN;
q.qC = qC;
q.qN = qN;

end

function pDest = AddSample(pSource, pAdd)
    pDest = pSource;
    pDest.L = pSource.L+pAdd.L;
    pDest.sample = [pSource.sample; pAdd.sample];
end

function pDest = Get10Perc(pSource, qSource)
    pDest = pSource;
    sum=0;
    for j=qSource.Qp'
        sum = sum+j(2);
        if (sum>=0.1)
            pDest.pAdd = j(1)*pSource.L/2;
            break;
        end;
    end;
    
    sum=0;
    for j=qSource.Qn'
        sum = sum+j(2);
        if (sum>=0.9)
            pDest.nAdd = j(1)*pSource.L/2;
            break;
        end;
    end;
end