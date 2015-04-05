function DistributePointsOnSphereTests
    runTest(2, 5);
    runTest(2, 10);
    runTest(2, 20);
    runTest(3, 5);
    runTest(3, 10);
    runTest(3, 20);    
    runTest(5, 5);
    runTest(5, 10);
    runTest(5, 20);    
    runTest(10, 5);
    runTest(10, 10);
    runTest(10, 20);    
end

function runTest(dim, N)
    x1 = DistributePointsOnSphere(dim, N, params('dT', 0.1));
    x2 = eq_point_set(dim - 1, N)';
    
    eps = 0.001;
    minDist1 = CalcMinDistance(x1);
    minDist2 = CalcMinDistance(x2);
    
    %Check(minDist1 > minDist2 - eps);
    %fprintf('Dim=%d, N = %d, symulation = %.3f, math = %.3f\n', dim, N, minDist1, minDist2);
end

function minDistance = CalcMinDistance(x)
    N = size(x, 1);
    minDistance = +Inf;
    for i=1:N
        for j=(i+1):N
            x1 = x(i, :);
            x2 = x(j, :);
            a = x1 - x2;
            aDist = sqrt(a * a');
            if (aDist < minDistance)
                minDistance = aDist;
            end
        end
    end    
end