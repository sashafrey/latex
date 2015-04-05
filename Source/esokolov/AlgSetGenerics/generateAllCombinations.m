function Combinations = generateAllCombinations(n, k)
    Combinations = (1:(n - k + 1))';
    for z = 2:k
        %NewCombs = [];
        NewCombs = zeros(nchoosek(n, k), z);
        cnt = 0;
        for i = 1:size(Combinations, 1)
            for j = (Combinations(i, z - 1) + 1):(n - k + z)
                %NewCombs = [NewCombs; Combinations(i, :) j];
                NewCombs(cnt + 1, :) = [Combinations(i, :) j];
                cnt = cnt + 1;
            end
        end
        %Combinations = NewCombs;
        Combinations = NewCombs(1:cnt, :);
    end
end