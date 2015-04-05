function [averageEps, QEps] = BallSliceSubsetOverfitting(L, ell, m, rho, d, eps)
    %avgEmpirical = Empirical(L, ell, m, rho, d);
    %maxEps = floor ( rho / 2) * L / ell / (L - ell);

    if (exist('eps', 'var'))
        if (d > 0)
            QEps = ByFreyFormula(L, ell, m, rho, d, eps);
        else
            QEps = ByFreyFormulaFullSet(L, ell, m, rho, eps);
        end
    else
        QEps = 0;
    end  
        
    if (d == 1)
        averageEps = 0;
        QEps = 0.5;
        return;
    end
    
    if (d > 0)
        averageEps = bsearch(@(eps)ByFreyFormula(L, ell, m, rho, d, eps), 0.5, 1, 0.001);
    else
        averageEps = bsearch(@(eps)ByFreyFormulaFullSet(L, ell, m, rho, eps), 0.5, 1, 0.001);
    end
end

function QEps = ByFreyFormulaFullSet(L, ell, m, r, eps) 
    k = L - ell;
    s_eps = floor(ell / L * (m - eps * k));
    rr = floor( r / 2);
    
    if (m < (eps * k))
        QEps = 0;
        return;
    end
    
    QEps = 0;
    for i = 0 : min([m, ell, s_eps + rr])
        QEps = QEps + hhProb(L, ell, m, i);
    end 
end

function QEps = ByFreyFormula(L, ell, m, r, d, eps)
    k = L - ell;
    s_eps = floor(ell / L * (m - eps * k));
    
    rr = floor( r / 2);
    
    D = 0;
    for b = 0 : rr
        D = D + CnkCalc(m, b) * CnkCalc(L - m, b);
    end
    
    total = 0;
    for i = 0 : min(m, ell)
        h = hhProb(L, ell, m, i);
        Ni = 0;
        for x = 0 : min(rr, i)
            for xx = 0 : (rr - x)
                for y = 0 : min(rr, ell - i)
                    yy = x + xx - y;
                    if (yy < 0) 
                        continue;
                    end
                    
                    if ((i - x + y) <= s_eps)
                        continue;
                    end
                    
                    Ni = Ni + ...
                        CnkCalc(i, x) * ...
                        CnkCalc(m - i, xx) * ...
                        CnkCalc(ell - i, y) * ...
                        CnkCalc(L - m - ell + i, yy);
                end
            end
        end
        
        total = total + h * power(Ni / D, d);
    end
    
    QEps = 1 - total;
end

function QEps = ByIlyaFormula(L, ell, m, r, d, eps)
    k = L - ell;
    s_eps = floor(ell / L * (m - eps * k));
   % if (s_eps < 0) 
        %QEps = 0;
        %return;
   % end
    
    D = 0;
    for b = 0 : floor(r / 2)
        D = D + CnkCalc(m, b) * CnkCalc(L - m, r - b);
    end
    
    total = 0;
    for i = 0:ell
        h = hhProb(L, ell, m, i);
        Ni = 0;
        for b = floor(m - r / 2) : m
            for p = 0 : i
                for q = (s_eps - p + 1) : (ell - i)
                   % if (q < 0) continue; end;
                    Ni = Ni + ...
                        CnkCalc(i, p) + ...
                        CnkCalc(m - i, b - p) + ...
                        CnkCalc(ell - i, q) + ...
                        CnkCalc(L - m - ell + i, m - b - q);
                end
            end
        end
        
        total = total + h * power(Ni / D, d);
    end
    
    QEps = 1 - total;
end

function averageEps = Empirical(L, ell, m, rho, d)
    % empirical approach
    % a0 := 00...0111...1, #1 = m, #0 = (L-m).

    % ToDo : figure out how to handle these cases gracefully.
    Check(rho <= m);
    Check(rho <= L - m);
    
    A = false(d, L);

    % prob(i) will contain the fraction of algorithms $a$ with rho(a0, a) = 2 * (i + 1).
    rr = floor(rho / 2);
    prob = zeros(rr + 1, 1);
    for i = 0 : rr
        prob(i + 1) = CnkCalc(m, i) * CnkCalc(L - m, i);        
    end
    prob = prob / sum(prob);
    
    % probD = cumulative distribution function of the prob.
    probD = zeros(rr + 1, 1);
    for i = 1 : (rr + 1)
        probD(i) = sum(prob(1:i));
    end
    
    for i = 1:d
        val = rand;
        r = find(probD > val, 1 ) - 1;      
        
        a0 = false(1, L);
        a0((L - m + 1): L) = true;
        a0(randsample(L-m, r)) = true;
        a0(L - m + randsample(m, r)) = false;
        A(i, :) = a0;
    end
    
    [~, averageEps] = CalcOverfitting(AlgsetCreate(A), ell / L, 1000, 0.05, 0.5, 0.01);
end