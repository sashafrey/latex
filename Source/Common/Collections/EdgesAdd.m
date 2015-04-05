function edges = EdgesAdd(edges, a, b)
    if ((size(edges.Children) < a) | (isempty(edges.Children{a})))
        edges.Children{a} = b;
    else
        edges.Children{a} = [edges.Children{a}, b];
    end
    
    if ((size(edges.Parents) < b) | (isempty(edges.Parents{b})))
        edges.Parents{b} = a;
    else
        edges.Parents{b} = [edges.Parents{b}, a];
    end
end