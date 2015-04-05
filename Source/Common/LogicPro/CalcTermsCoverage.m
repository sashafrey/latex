function coverage = CalcTermsCoverage(terms, task)
    nItems = task.nItems;
    nTerms = tsLength(terms);
    coverage = false(nTerms, nItems);
    
    for i=1:nTerms
        termCoverage = CalcOneTermCoverage(terms.left(i), terms.right(i), terms.feature(i), terms.isnot(i), task);
        coverage(i, termCoverage) = true;        
    end
end
