function initialize(networkpath)
    if (~exist('networkpath', 'var'))
        networkpath = '\\alfrey-h01\vft11ccas\Source';
    end
    
    addpath(fullfile(networkpath, 'alfrey'));
    addpath(fullfile(networkpath, 'Common'));
    addpath(fullfile(networkpath, 'Common\Algorithms'));
    addpath(fullfile(networkpath, 'Common\C50'));
    addpath(fullfile(networkpath, 'Common\Clustering'));
    addpath(fullfile(networkpath, 'Common\Collections'));
    addpath(fullfile(networkpath, 'Common\DataAdapters'));
    addpath(fullfile(networkpath, 'Common\Experiments'));
    addpath(fullfile(networkpath, 'Common\GPU'));
    addpath(fullfile(networkpath, 'Common\GPU\Misc'));
    addpath(fullfile(networkpath, 'Common\GPU\Misc'));
    addpath(fullfile(networkpath, 'Common\LibSVM'));
    addpath(fullfile(networkpath, 'Common\LibSVM\matlab'));
    addpath(fullfile(networkpath, 'Common\LogicPro'));
    addpath(fullfile(networkpath, 'Common\PACBayes'));
    addpath(fullfile(networkpath, 'Common\RVM'));
    addpath(fullfile(networkpath, 'Common\RVMv2'));
    addpath(fullfile(networkpath, 'Common\SVMLight'));
    addpath(fullfile(networkpath, 'Common\SVMLight\bin'));
    addpath(fullfile(networkpath, 'Common\SVMLight\matlab'));
    addpath(fullfile(networkpath, 'Common-Tests'));
    
    svnfreepath(fullfile(networkpath,'Tools'))
    svnfreepath(fullfile(networkpath,'Tools-Tests'))
    svnfreepath(fullfile(networkpath,'esokolov'))
    
    % Compile LibSVM for matlab.
    currentFolder = pwd;
    libSVMPath = fullfile(networkpath, 'Common\LibSVM\matlab');
    cd(libSVMPath);
    make;
    
    SVMLightPath = fullfile(networkpath, 'Common\SVMLight');
    cd(SVMLightPath);
    compilemex
    
    cd(currentFolder);
end