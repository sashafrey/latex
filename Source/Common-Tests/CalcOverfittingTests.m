function CalcOverfittingTests
    % It is well-known that SC-bound (or, more generally, Hasse graph bound) 
    % is precice for monotonic net. This test uses this fact to checks QEps 
    % calculation for both monte-carlo calculator "CalcOverfitting",
    % and CalcQEpsHasse method.
    
    algset = GenerateMonotonicSet(50, 2, 3, 2);
    edges = BuildHasseGraph(algset);
    [algset, edges] = BuildInternalClosure(algset, edges);
    ell = floor(algset.L / 2);
    k = algset.L - ell;
    QEps = CalcOverfitting(algset, 0.5, 1000, 1, 0.5, 0.1);
    QEpsH.X = []; QEpsH.Y = [];
    for eps = 0:0.1:0.5
        [Q, P] = CalcQEpsHasse(algset, edges, eps, ell, k );
        QEpsH.X = [QEpsH.X; eps];
        QEpsH.Y = [QEpsH.Y; Q];    
    end
    
    Check(all(abs(QEps.Y - QEpsH.Y) < (0.01 + 0.1 * QEps.Y)))
    Check(all(abs(QEps.Y - QEpsH.Y) < (0.01 + 0.1 * QEpsH.Y)))
    
    % Calculate other bounds to see if they are working
    for eps = 0:0.1:0.5
        [~, ~] = CalcQEpsBool(algset, edges, eps, ell, k);
        [~, ~] = CalcQEpsHasseSimple(algset, edges, eps, ell, k );
    end    
end