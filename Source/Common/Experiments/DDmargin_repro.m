function DDmargin_repro
    tasks = LoadTasks;
    tasknames = {
    'statlog'
    'Letter'
    'Magic04'
    'Mushrooms'
    'Optdigits'
    'pageblocks'
    'pendigits'
    'waveform'
    'Wisconsin_Breast_Cancer'
    'glass'
    'pima'
    'Wdbc'
    };

estimate = zeros(length(tasknames), 1);
for iTask=1:length(tasknames)
    task = tasks.(tasknames{iTask});
    estimate(iTask)=task.nItems * task.nFeatures;
end
[~, idx] = sort(estimate);

for iTask=idx'
    taskname=tasknames{iTask};
    fprintf('%s\n', taskname);
    DDmargin_repro1(tasks.(taskname), iTask);
end

end

function DDmargin_repro1(task, iTask)
    nanobjects = any(isnan(task.objects)')';
    task.objects(nanobjects, :) = [];
    task.target(nanobjects, :) = [];
    task.nItems = task.nItems - sum(nanobjects);

    %attempt to reproduce experiment from DDmargin paper.
    task = NormalizeFeatures(task, true, 2); 
    task.objects = task.objects(any(~isnan(task.objects'))', :);
    
    nReps = 4;
    nCVFolds = 5;
    nDegs = 10;
    
    if (matlabpool('size') == 0)
        matlabpool
    end
    
    parfor iRep = 1:nReps
        ddBounds = zeros(nCVFolds, nDegs);
        diBounds = zeros(nCVFolds, nDegs);
        trainErr = zeros(nCVFolds, nDegs);
        testErr  = zeros(nCVFolds, nDegs);

        folds = GenerateNFoldCVIndexes(task.target, nCVFolds);
        for iFold = unique(folds)'
            train = GetTaskSubsample(task, folds ~= iFold);
            test =  GetTaskSubsample(task, folds == iFold);

            for deg = 1:nDegs
                %% Tune SVM
                opt = sprintf('-t 1 -d %i -r 0 -g 1 -q', deg);
                svm = svmtrain2(train.target, train.objects, opt);
                
                trainPredict = svmpredict(train.target, train.objects, svm, '-q');
                testPredict = svmpredict(test.target, test.objects, svm, '-q');
                
                trainError = sum(trainPredict ~= train.target) / train.nItems;
                testError = sum(testPredict ~= test.target) / test.nItems;
                
                if (0)
                    opt = sprintf('-t 1 -s 1 -r 0 -d %i -c 1 -b 0 -v 0', deg);
                    svm2 = svmlearn(train.objects, train.target, opt);
                    [trainError2, trainPredicts] = svmclassify(train.objects, train.target, svm2);
                    [testError2, testPredicts] = svmclassify(test.objects, test.target, svm2);

                    % Storing support vectors in a SV_num * n matrix SV
                    SV = svm2.supvec(2:end,:);
                    
                    % Computing SV_num * SV_num Gramm matrix for support vectors in
                    % a kernel space
                    KernMat = (SV*SV').^deg;
                    % Storing the vector of SV coefficients coef(i)=lambda_i*y(i)
                    coef = svm2.alpha(2:2 + size(KernMat,1) - 1);
                end
                
                if (svm.totalSV ~= 0)
                    %SV = train.objects(svm.sv_indices, :);
                    SV = full(svm.SVs);
                    KernMat = (SV*SV').^deg;
                    coef = svm.sv_coef;

                    % Computing the norm of the weight vector in a kernel space
                    w_norm = sqrt(coef'*KernMat*coef);

                    % Computing prediction values for training set
                    values = sum((train.objects*SV').^deg .* repmat(coef', train.nItems, 1), 2) - svm.rho;
                    values = values * svm.Label(1);
                    %[trainPredict, sign(values)]

                    % Computing norms of objects
                    norms = sqrt(sum(train.objects.*train.objects, 2).^deg);

                    % calc normalized margin.
                    gamma = train.target .* (values ./ norms) / w_norm;
                    Check(mean(gamma < 0) == trainError);
                    %plot(sort(gamma));

                    ddBound = DDmargin(gamma, nchoosek(task.nFeatures + deg - 1, deg));
                    diBound = DImargin(gamma);
                else
                    ddBound = NaN;
                    diBound = NaN;
                end
                    
                ddBounds(iFold, deg) = ddBound; 
                diBounds(iFold, deg) = diBound; 
               
                trainErr(iFold, deg) = trainError;
                testErr (iFold, deg) = testError;
                
                fprintf('rep/fold/deg = %i/%i/%i, totalSVs = %i, trainErr = %.2f, testErr = %.2f, DDBound = %.2f, DIBound = %.2f\n', ...
                    iRep, iFold, deg, svm.totalSV, trainError, testError, ddBound, diBound);
            end
        end
        
        ddBoundsAll{iRep} = ddBounds;
        diBoundsAll{iRep} = diBounds;
        trainErrAll{iRep} = trainErr;
        testErrAll{iRep} = testErr;
    end
    
    ddBounds = zeros(nCVFolds, nDegs);
    diBounds = zeros(nCVFolds, nDegs);
    trainErr = zeros(nCVFolds, nDegs);
    testErr  = zeros(nCVFolds, nDegs);
    for iRep = 1:nReps
        ddBounds = ddBounds + ddBoundsAll{iRep} / nReps;
        diBounds = diBounds + diBoundsAll{iRep}/ nReps;
        trainErr = trainErr + trainErrAll{iRep}/ nReps;
        testErr  = testErr + testErrAll{iRep}/ nReps;
    end
    
    subplot(4,3,iTask);
    x = 1:nDegs;
    plot(x, mean(ddBounds), 'bo-', ...
         x, mean(diBounds), 'gs-', ...
         x, mean(trainErr), 'kx-', ...
         x, mean(testErr), 'rd-');
     
    xlabel('t');
    ylabel('error');
    task_name = task.name;
    task_name(task_name=='_') = ' ';
    title(task_name);
    drawnow;
    %legend('DD-margin', 'DI-margin', 'train error', 'test error');
    set(gcf, 'Color', 'white');
end

