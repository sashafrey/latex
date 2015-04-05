function rawalgset = RawAlgSetLoadFromLogicProOutput( filename, mode )
    % 0 - covered and own class         (0)
    % 1 - covered and other class       (1)
    % 2 - notcovered and own clas       (this could be treated as 0 or 1)
    %                                    mode = 0 => 2 is threaded as OK
    %                                    mode = 1 => 2 is threated as error.
    % 3 - notcovered and other class    (0)
    [veryRawAlgSet, LittleP, LittleN, BigP, BigN] = textread(filename, '%s\t%d\t%d\t%d\t%d', 'bufsize', 40000);
    nAlgs = length(veryRawAlgSet);
    if (nargin == 0) 
        throw(MException('InvalidOperation', 'Unable to load file algset from logicpro data - file appears to be empty.'));
    end
    
    if (nargin == 1)
        mode = 1;
    end
    
    nItems = length(veryRawAlgSet{1});    
    rawalgset = false(nAlgs, nItems);
    if (mode == 0)
        for iAlg = 1:nAlgs
            rawalgset(iAlg, :) = veryRawAlgSet{iAlg} == '1';
        end
    else
        for iAlg = 1:nAlgs
            rawalgset(iAlg, :) = (veryRawAlgSet{iAlg} == '1') | (veryRawAlgSet{iAlg} == '2');
        end
    end
    
    %algset = AlgsetAdd(AlgsetCreate(), rawalgset);
    %rawalgset = AlgsetGet(algset, 1:algset.Count);
    rawalgset = RawAlgSetRemoveDuplicates(rawalgset);
end
