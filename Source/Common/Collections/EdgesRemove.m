function edges = EdgesRemove(edges, a, b)
    edges.Children{a}(edges.Children{a} == b) = [];
    edges.Parents{b}(edges.Parents{b} == a) = [];
end