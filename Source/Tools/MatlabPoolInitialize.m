function MatlabPoolInitialize(configname)
    if (~exist('configname', 'var'))
        configname = 'trehost'; % default config in alfrey lab
    end
    
    if (matlabpool('size') == 0)
        jm = findResource('scheduler', 'configuration', configname);
        matlabpool(jm);
    end
end