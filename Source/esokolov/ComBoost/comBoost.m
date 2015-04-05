function [ensembleWeights, learningLog] = ...
        comBoost(X, Y, maxEnsembleLength, ell_0, ell_1, maxFeatureSubsetSize, ...
        featureSelectionFunc, baseClassifierLearnerFunc, ...
        randomizeFeatureSelection, featureSelectionSubsetRatio)

    if ~exist('randomizeFeatureSelection', 'var')
        randomizeFeatureSelection = false;
    end
    if ~exist('featureSelectionSubsetRatio', 'var')
        featureSelectionSubsetRatio = 0.6;
    end
    
    L = size(X, 1);
    d = size(X, 2);
    
    activeObjects = true(L, 1); % по каким объектам обучать следующий классификатор
    ensembleWeights = nan(maxEnsembleLength, d + 1);
    
    learningLog = cell(maxEnsembleLength, 1);
    
    qual_hist = zeros(maxEnsembleLength, 1);
    
    for i = 1:maxEnsembleLength
        X_curr = X(activeObjects, :);
        Y_curr = Y(activeObjects);
        
        % если классы слишком маленькие, то заканчиваем обучение
        if (length(unique(Y_curr)) == 1)
            break;
        end        
        if (sum(Y_curr == -1) < 5 || sum(Y_curr == 1) < 5)
            break;
        end
        
        % если нужно, будем делать отбор из случайного подмножества признаков
        if randomizeFeatureSelection
            features_curr = randsample(d, floor(d * featureSelectionSubsetRatio))';
            X_curr = X_curr(:, features_curr);
        else
            features_curr = 1:d;
        end
        learningLog{i} = struct('FeaturesUsed', features_curr);
                
        [features, featureSelectionLog] = featureSelectionFunc(X_curr, Y_curr, ...
            maxFeatureSubsetSize);
        learningLog{i}.featureSelectionLog = featureSelectionLog;
        
        X_curr = X_curr(:, features);        
        features = features_curr(features);
        learningLog{i}.FeaturesSelected = features;
        
        w_subset = baseClassifierLearnerFunc(X_curr, Y_curr);
        w_full = zeros(1, d + 1);
        w_full([features, d+1]) = w_subset;
        if norm(w_full) > 0
            w_full = w_full / norm(w_full);
        end
        ensembleWeights(i, :) = w_full;
        
        [Y_our, margins] = comBoost_classify(X, ensembleWeights, Y);
        
        sortedMargins = sort(margins);
        leftBound = sortedMargins(ell_0);
        rightBound = sortedMargins(ell_1);
        activeObjects = (margins > leftBound) & (margins <= rightBound);
        
        fprintf('Features:\n');
        fprintf('%d ', features);
        fprintf('\n');
        
        qual_hist(i) = sum(Y_our ~= Y);
        
        save('./esokolov/ComBoost_Experiments/comboost_backup.mat', ...
            'ensembleWeights', 'learningLog');
    end
end
