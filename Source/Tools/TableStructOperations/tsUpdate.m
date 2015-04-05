function sourcestruct  = tsUpdate(sourcestruct, substruct)
    Check(tsIsValid(sourcestruct));
    Check(tsIsValid(substruct));

    names1 = fieldnames(sourcestruct);
    names2 = fieldnames(substruct);  
    isequalnames =  all(strcmp(sort(names1), sort(names2)));
    
    if(isequalnames && ~isempty(names1))
        for names1 = names1'
            
            name = names1{1,1};
            len  = length(sourcestruct,1);
            sourcestruct.(name)(1:len,:) = struct2.(name)(1:len,:);
           
        end
    end
     
    Check(tsIsValid(sourcestruct));
end
