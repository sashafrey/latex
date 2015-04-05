function [ task ] = LoadTask( folder )
    if (~exist(folder, 'dir'))
        DataUCI_folder = LocateDataUCI();
        folder = fullfile(DataUCI_folder, folder);
    end

    matfiles = dir(fullfile(folder, '*.mat'));
    if (~isempty(matfiles))
        for i=1:length(matfiles)
            X_full = [];
            Y_full = [];
            load(fullfile(folder, matfiles(i).name));
            if (~isempty(X_full) && ~isempty(Y_full))
                % detected tasks from Eugenii Sokolov.
                Check(length(Y_full) == size(X_full, 1));
                labels = unique(Y_full);
                Check(length(labels) == 2);
                target = Y_full;
                target(Y_full == labels(1)) = 1;
                target(Y_full == labels(2)) = 2;
                
                task.nItems = size(X_full, 1);
                task.nFeatures = size(X_full, 2);
                task.nClasses = 2;
                task.target = target;
                task.objects = X_full;
                task.isnominal = false(task.nFeatures, 1);
                
                remains = folder;
                while (~isempty(remains))
                    [task.name, remains] = strtok(remains, char(92));
                end
            end            
        end
        
        return;
    end

    objects = dlmread(fullfile(folder, 'Objects.csv'), ';');
    task.nItems = size(objects, 1);
    task.nFeatures = size(objects, 2);

    remains = folder;
    while (~isempty(remains))
        [task.name, remains] = strtok(remains, char(92));
    end
    
    ftypes = fopen(fullfile(folder, 'PropertyTypes.csv'));
    typesCell = textscan(ftypes, '%s');
    fclose(ftypes);
    typesCell = typesCell{1};
    Check(length(typesCell) == task.nFeatures, 'PropertyTypes vector have the same number of features as object matrix');
    isnominal = NaN(task.nFeatures, 1);
    isnominal(ismember(typesCell, 'ORD')) = 0;
    isnominal(ismember(typesCell, 'NUM')) = 0;
    isnominal(ismember(typesCell, 'BIN')) = 1;
    isnominal(ismember(typesCell, 'NOM')) = 1;
    Check(all(~isnan(isnominal)));
    
    ftarget = fopen(fullfile(folder, 'Target.csv'));
    targetCell = textscan(ftypes, '%s');
    fclose(ftarget);
    targetCell = targetCell{1};
    targetLabels = unique(targetCell);
    Check(length(targetCell) == task.nItems, 'Target vector have the same number of items as object matrix');
    
    task.nClasses = length(targetLabels);    
   
    target = NaN(task.nItems, 1);
    for targetLabel = 1:task.nClasses
        target(ismember(targetCell, targetLabels{targetLabel})) = targetLabel;                
    end
    
    task.target = target;
    task.objects = objects;
    task.isnominal = isnominal;    
end

