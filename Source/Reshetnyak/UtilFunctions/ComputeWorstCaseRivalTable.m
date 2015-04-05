function [probTable, newErrorRate, newDegree] = ComputeWorstCaseRivalTable(changedObjectsCount, sampleSize, trainSize, eps, maxDegree)
%changedObjectsCount - число объектов, для которых мы меняем метки классов
%sampleSize - размер полной выборки
%trainSize - размер обучающей выборки
%eps - точность
%maxDegree - максимально возможное значение верхней связности алгоритма

%
%probTable - матрица вероятностей f_{changedObjects}(m, q)

    %Вычисление вклада алгоритма с фиксированными параметрами
    function [res] = GetAdditionalOverfitProb(ell, m00, m01, m10, m11)
        res = 0;
        L = m00 + m01 + m10 + m11;
        for s01 = 0 : m01
            for s10 = s01 : m10
                for s11 = 0 :  min(threshold(m01 + m11) - s01, m11)
                    s00 = ell - s01 - s10 - s11;
                    if (s00 >= 0) && (s00 <= m00)
                        res = res + ct(m01 + 1, s01 + 1) * ct(m10 + 1, s10 + 1) * ct(m11 + 1, s11 + 1) * ct(m00 + 1, s00 + 1);
                    end
                end
            end
        end
    end

    %Поиск наихудшего алгоритма, который может быть получен из алгоритма с
    %числом ошибок m и связностью q изменением меток на r объектах.
    function [bestProb, bestM, bestQ] = FindWorstAlgorithm(L, ell, r, m, q)
        bestProb = 0;
        bestM = 0;
        bestQ = 0;
        for s = 0 : r
            t = r - s;
            if m + s - t < r
                continue; 
            end
            for newQ = max(q - s, 0) : min(q, L - m - s)
                prob = GetAdditionalOverfitProb(ell - newQ, L - m - s - newQ, m - t, t, s);
                prob = prob / ct(L + 1, ell + 1);
                if prob > bestProb
                    bestProb = prob;
                    bestM = m + s - t;
                    bestQ = newQ;
                end
            end
        end
    end

    ct = ComputeChooseTable(sampleSize); %Вычисление треугольника Паскаля
    %Вычисление пороговых значений s_m числа ошибок на обучающей выборке,
    %при которых алгоритм переобучается
    threshold = TrainErrorOverfitThreshold(sampleSize, trainSize, 1 : sampleSize, eps); 
    
    probTable = zeros(sampleSize + 1, maxDegree + 1);
    newErrorRate = zeros(sampleSize + 1, maxDegree + 1);
    newDegree = zeros(sampleSize + 1, maxDegree + 1);
    
    for err = 1 : sampleSize
        for deg = 0 : maxDegree
            [probTable(err + 1, deg + 1), newErrorRate(err + 1, deg + 1), newDegree(err + 1, deg + 1)] = ...
                FindWorstAlgorithm(sampleSize, trainSize, changedObjectsCount, err, deg);
        end
    end
end