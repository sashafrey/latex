function isEqual = isEqualLC(alg1, alg2)
    isEqual = ( sum(alg1.errVect ~= alg2.errVect) == 0 );
end