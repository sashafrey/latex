function [features, featureSelectionLog] = ...
        selectFeaturesGreedy(X, Y, maxFeaturesCnt, sampleGradeFunc)
    
    L = size(X, 1);
    d = size(X, 2);
    
    features = [];
    grades = zeros(maxFeaturesCnt, 1);
    gradesLog = {};    
    
    for i = 1:maxFeaturesCnt
        if (length(features) == d)
            break;
        end
        
        %bestQuality = Inf;
        %bestFeature = -1;
        
        new_feature_candidates = setdiff(1:d, features);
        grades_currIter = zeros(length(new_feature_candidates), 1);
        
        features_currIter = zeros(length(new_feature_candidates), length(features) + 1);
        
        parfor feature_idx = 1:length(new_feature_candidates)
            %warning off;
            
            new_feature = new_feature_candidates(feature_idx);
            newFeatures = [features new_feature];
            
            fprintf('%d ', newFeatures);
            fprintf('\n');
            
            Xcurr = X(:, newFeatures);
            grades_currIter(feature_idx) = sampleGradeFunc(Xcurr, Y);
            
            features_currIter(feature_idx, :) = newFeatures;
        end
        
        gradesLog{i} = struct('features', features_currIter, ...
            'grades', grades_currIter);
        
        % добавляем случайный из признаков, на которых достигается минимум
        bestIdx = find(grades_currIter == min(grades_currIter));
        bestIdx = bestIdx(randi(length(bestIdx)));
        bestQuality = grades_currIter(bestIdx);
        bestFeature = new_feature_candidates(bestIdx);
        
        grades(i) = bestQuality;
        
        features = [features bestFeature];
    end
    
    featureSelectionLog = [];
    featureSelectionLog.gradesLog = gradesLog;
    featureSelectionLog.grades = grades;
    featureSelectionLog.featureSubset = features;
    
    % запрещаем выбирать подпространство из одного признака
    grades(1) = Inf;
    
    bestLength = find(grades == min(grades), 1, 'first');
    features = features(1:bestLength);
    
    featureSelectionLog.bestLength = bestLength;
    featureSelectionLog.finalSubset = features;
end