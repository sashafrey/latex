function getNeighboursLC_test
    X = [1 2;
         1 1;
         2 1];
    X = [X, ones(3, 1)];

    Y = [-1; 1; 1];
    w = -[0; 1; -1.5] + rand(3, 1) / 1000;

    alg = convertLinearAlgToStructure(w, X, Y);
    
    algsVect = StructVectorCreate(alg);
    algsVect = StructVectorAdd(algsVect, alg);
    
    [algsVect, sourcesVect] = ...
        getNeighboursLC(algsVect, VectorCreate(), 1, X, Y);
    
    % исток должен быть один и совпадать с alg
    Check(sourcesVect.Count == 1 && ...
        sourcesVect.Data(1) == 1, ...
        'getNeighboursLC() returned wrong sources');
    
    % помимо alg, должно быть три его соседа
    Check(algsVect.Count == 4, ...
        'getNeighboursLC() found wrong number of nieghbours');
    
    % у каждого соседа должно быть по одной ошибке
    Check(algsVect.Data(2).errCnt == 1 && ...
        algsVect.Data(3).errCnt == 1 && ...
        algsVect.Data(4).errCnt == 1, ...
        'getNeighboursLC() found wrong neighbours');
end