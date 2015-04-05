function [alg, neighbours] = getNeighboursLC_rw(alg, X, Y)
% Обертка для getNeighboursLC, несколько упрощающая ее использование.
% Для алгоритма alg возвращаются все соседние с ним в графе
% расслоения-связности.
    
    algsVect = StructVectorCreate(initLinearAlgStructure());
    algsVect = StructVectorAdd(algsVect, alg);
    
    sourcesVect = VectorCreate();
    
    [algsVect, ~] = ...
        getNeighboursLC(algsVect, sourcesVect, 1, X, Y);
    
    algsVect = StructVectorTrim(algsVect);
    alg = algsVect.Data(1);
    alg.lowerNeighbours = [];
    alg.upperNeighbours = [];
    
    neighbours = StructVectorDel(algsVect, 1);
end