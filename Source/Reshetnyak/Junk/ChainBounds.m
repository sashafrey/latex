function ChainBounds( )
    L = 400;
    nGood = floor(L / 4); %число безупречно классифицируемых объектов
    l = L/2;
    eps = 0.05;
    chain = GenerateChain(L - nGood, 0.8);
    chainError = MonteCarloChainEstimation(chain, L, l,  eps, 10000)
    chainError2 = DPChainEstimation(chain, L, l, eps)
end