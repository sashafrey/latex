function info = HInfoD(P, N, p, n)
    mlock();
    persistent logfact_HInfoD;
    
    minL = max(P) + max(N) + 10;
    if (length(logfact_HInfoD) < minL)
        logfact_HInfoD = CreateLogFactorial(minL);
    end
 
    info =   logCnk(P+N, p+n, logfact_HInfoD) ...
           - logCnk(P, p, logfact_HInfoD) ...
           - logCnk(N, n, logfact_HInfoD);
    info(p.*N <= P.*n) = 0;
end

function logcnk = logCnk(n, k, logfact)
    logcnk = logfactF(n, logfact) - logfactF(k, logfact) - logfactF(n-k, logfact);
end

function logf = logfactF(n, logfact)
    nF = floor(n);
    
    %Use +1 as an argument to logfact because logfact(1) = log(0!) = 0.
    fLeft = logfact(nF + 1);
    fRight = logfact(nF + 2);
    logf = fLeft + (fRight - fLeft) .* (n - nF);
end

function logfact = CreateLogFactorial(L)
    logfact = [0, log(1:L)];
    
    for i=2:(L+1)
        logfact(i) = logfact(i-1) + logfact(i);
    end
    
    logfact = logfact';
end
