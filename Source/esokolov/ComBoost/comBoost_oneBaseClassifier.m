function [ensembleWeights, learningLog] = ...
        comBoost_oneBaseClassifier(X, Y, maxEnsembleLength, ell_0, ell_1, ...
        maxFeatureSubsetSize, ...
        featureSelectionFunc, baseClassifierLearnerFunc, ...
        randomizeFeatureSelection, featureSelectionSubsetRatio)
    
    L = size(X, 1);
    d = size(X, 2);
    
    ensembleWeights = nan(1, d + 1);
    learningLog = cell(1, 1);
    
    [features, featureSelectionLog] = featureSelectionFunc(X, Y, ...
        maxFeatureSubsetSize);
    learningLog{1}.featureSelectionLog = featureSelectionLog;

    X = X(:, features);        
    features = features_curr(features);
    learningLog{1}.FeaturesSelected = features;
    
    w_subset = baseClassifierLearnerFunc(X, Y);
    w_full = zeros(1, d + 1);
    w_full([features, d+1]) = w_subset;
    if norm(w_full) > 0
        w_full = w_full / norm(w_full);
    end
    ensembleWeights(1, :) = w_full;
end