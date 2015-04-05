function convertLinearAlgToStructure_test
    L = 200;
    d = 2;
    
    for i = 1:10
        X = randn(L, d);
        X = [X, ones(L, 1)];
        Y = ones(L, 1);
        Y(randn(L, 1) < 0.5) = -1;

        for j = 1:10
            w = rand(d + 1, 1);
            alg = convertLinearAlgToStructure(w, X, Y);
            
            % не изменился ли вектор ошибок?
            w_errVect = (sign(X * w) ~= Y);
            Check(sum(w_errVect ~= alg.errVect) == 0, ...
                'convertLinearAlgToStructure() returned different alg!');
        end
    end
end