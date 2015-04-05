function GPU_Initialize(folder)
    % GPU_Initialize
    % Initialized the library for fast GPU sampling of linear classifiers.

    if (~exist('folder', 'var'))
        folder = '\\alfrey-h01\vft11ccas\Source\Common\GPU';
    end
    
    if (~libisloaded(GPU_LibName))
        original_path = pwd;
        cd(folder);
        loadlibrary(sprintf('%s.dll', GPU_LibName), 'ls_api.h');
        cd(original_path);
    end
    
    % libfunctions(GPU_LibName)
    % libfunctionsview(GPU_LibName);
end