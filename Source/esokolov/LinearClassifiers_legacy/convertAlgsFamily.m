function A = convertAlgsFamily(algs, algsCnt, L)
    A = zeros(algsCnt, L);
    for i = 1:algsCnt
        A(i, :) = algs(i).errVect;
    end
end