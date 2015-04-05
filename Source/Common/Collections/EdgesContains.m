function contains = EdgesContains(edges, a, b)
    if (length(edges.Children) < a)
        contains = false;
    else
        contains = any(edges.Children{a} == b);
    end
end