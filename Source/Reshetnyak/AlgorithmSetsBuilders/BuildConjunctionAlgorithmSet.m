function [algs] = BuildConjunctionAlgorithmSet(sample, sampleClasses, invertAlgorithms)

    %Перебор всех возможных наборов термов
    function Rec(termNumber)
        if (termNumber == dim + 1)
            count = count + 1;
            algs(count, :) = (curAlg == sampleClasses);
            if invertAlgorithms
                algs(count + 1, :) = ~algs(count, :);
                count = count + 1;
            end
            return ;
        end
        aOld = curAlg; 
        for i = sampleSize : -1 : 1
            if (i == sampleSize) || (sortedSample(i, termNumber) < sortedSample(i + 1, termNumber))
                Rec(termNumber + 1);
            end
            curAlg( ind(i, termNumber) ) = 0;
        end
        curAlg = aOld;     
    end

    function [numAlgs] = EstimateNumberOfAlgorithms(sample, invertAlgorithms)
        numAlgs = 1;
        if invertAlgorithms
            numAlgs = 2;
        end
        for n = 1 : size(sample, 2)
            numAlgs = numAlgs * numel(unique(sample(:, n)));
        end
        %numAlgs
    end

    if nargin < 3
        invertAlgorithms = false
    end
    [sampleSize dim] = size(sample);
    sampleClasses(sampleClasses ~= 1) = 0;
    algs = false(EstimateNumberOfAlgorithms(sample, invertAlgorithms) , sampleSize);
    
    %Используем термы вида [fi <= ai]
    [sortedSample ind] = sort(sample, 1);
    
    count = 0;
    curAlg = ones(1, sampleSize); %Текущая конъюнкция
    Rec(1);
    %algs(L^dim + 1, :) = sampleClasses;
    %count
    algs = sortrows(algs(1 : count, :));
	n = 1;
	for s = 2:size(algs, 1)
		if all(algs(s, :) == algs(s - 1, :))
			continue;
		end
		n = n + 1;
		algs(n, :) = algs(s, :);
    end
    algs = algs(1:n, :);
    terr = sum(algs, 2);
    [terr, ind] = sort(terr);
    algs = algs(ind, :);
end