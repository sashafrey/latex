function alg = convertLinearAlgToSimpleStructure(w, X, Y)
% Для заданного линейного классификатора с весами w формирует структуру,
% описанную в initLinearAlgSimpleStructure().

    % w должен быть столбцом
    if size(w, 1) == 1
        w = w';
    end
    
    % нормируем вектор весов
    w_norm = sqrt(sum(w .^ 2));
    if w_norm > 0
        w = w ./ w_norm;
    end

    currErrVect = (sign(X * w) ~= Y);
    
    % заполняем структуру
    alg = initLinearAlgSimpleStructure();
    alg.w = w;
    alg.errVect = currErrVect;
    alg.errCnt = sum(alg.errVect);
    alg.lowerNeighsCnt = 0;
    alg.upperNeighsCnt = 0;
    alg.lowerNeighbours = [];
    alg.upperNeighbours = [];
end
