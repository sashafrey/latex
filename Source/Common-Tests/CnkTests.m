function CnkTests
    % Cnk Tests
    for i = 0:10
        for j = 0:10
            Check(abs(cnk(i, j) - CnkCalc(i, j)) < 0.0001, 'Binomial coefficient is miscalculated.');
        end
    end
end
