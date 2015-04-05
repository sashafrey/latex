function algset = GenerateRandomAlgset(D, L, p0)
    algset = AlgsetAdd(AlgsetCreate(), (rand(D, L) > p0));
end