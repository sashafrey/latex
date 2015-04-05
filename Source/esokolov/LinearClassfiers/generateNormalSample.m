function [sample answers noiseCnt] = generateNormalSample(L, p, sigma)
    sample = zeros(L, p);
    answers = zeros(L, 1);
    for i = 1:L
        for j = 1:p
            if (i <= L / 2)
                sample(i, j) = -3 + sigma*randn(1);
                %sample(i, j) = -3 + 2.5*randn(1);
                %sample(i, j) = -3 + 1*randn(1);
                answers(i) = -1;
            else
                sample(i, j) = 3 + sigma*randn(1);
                %sample(i, j) = 3 + 2.5*randn(1);
                %sample(i, j) = 3 + 1*randn(1);
                answers(i) = 1;
            end
        end
    end
    noiseCnt = sum(sample(:, 1) + sample(:, 2) < 0 & answers == 1) + ...
        sum(sample(:, 1) + sample(:, 2) > 0 & answers == 0);
end