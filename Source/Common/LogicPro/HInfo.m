function info = HInfo(P, N, p, n)
    mlock();
    persistent logfact_HInfo;
    
    minL = max(P) + max(N) + 10;
    if (length(logfact_HInfo) < minL)
        logfact_HInfo = CreateLogFactorial(minL);
    end

    info =   logCnk(P+N, p+n, logfact_HInfo) ...
           - logCnk(P, p, logfact_HInfo) ...
           - logCnk(N, n, logfact_HInfo);
    info(p.*N <= P.*n) = 0;
end

function logcnk = logCnk(n, k, logfact)
    %Use +1 as an argument to logfact because logfact(1) = log(0!) = 0.
    logcnk = logfact(n+1) - logfact(k+1) - logfact(n-k+1);
end

function logfact = CreateLogFactorial(L)
    logfact = [0, log(1:L)];
    
    for i=2:(L+1)
        logfact(i) = logfact(i-1) + logfact(i);
    end
    
    logfact = logfact';
end
