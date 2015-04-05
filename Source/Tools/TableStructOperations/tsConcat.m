function resstruct = tsConcat(struct1, struct2)
    %Check(tsIsValid(struct1));
    %Check(tsIsValid(struct2));
    
    if(~isstruct(struct1) || (tsLength(struct1) == 0))
        resstruct = struct2;
        return;
    elseif (~isstruct(struct2) || (tsLength(struct2) == 0))
        resstruct = struct1;
        return;
    end
   
    names1 = fieldnames(struct1);
    names2 = fieldnames(struct2);  
    isequalnames =  all(strcmp(sort(names1), sort(names2)));
    
    if(isequalnames && ~isempty(names1))
        for field = names1'
            name = field{1,1};


            if (iscell(struct1.(name)))
                Check(iscell(struct2.(name)));
                resstruct.(name) = [struct1.(name);  struct2.(name)];
            else
                resstruct.(name) = [struct1.(name);  struct2.(name)];
            end

        end
    end
    
    %Check(tsIsValid(resstruct));
end