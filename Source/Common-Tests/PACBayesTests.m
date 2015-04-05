function PACBayesTests
    bound = DDmargin(0.7:0.001:0.8, 15);
    Check((bound > 0.24) && (bound < 0.25));
end