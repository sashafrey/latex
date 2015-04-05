function algsCnt = computeNumberOfLinearAlgs(L, d)
    algsCnt = 0;
    for i = 0:d
        algsCnt = algsCnt + CnkCalc(L - 1, i);
    end
    algsCnt = 2 * algsCnt;
end
