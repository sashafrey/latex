function GenerateNFoldCVIndexesTests
    target = [ones(300, 1);2*ones(700,1)];
    perm = randperm(length(target));
    target = target(perm);
    nfoldindexes = GenerateNFoldCVIndexes(target, 10);
    
    Check(length(nfoldindexes) == 1000);
    for i=1:10
        Check(sum(nfoldindexes == i) == 100);
        Check(sum(target(nfoldindexes == i) == 1) == 30);
        Check(sum(target(nfoldindexes == i) == 2) == 70);
    end
end