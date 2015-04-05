function BuildHasseGraphTest
    algset = GenerateRandomAlgset(150, 20, 0.5);
    algset01 = AlgsetGet(algset, 1:algset.Count);
    hasseSlow = BuildHasseGraphSlow(algset01);
    edges = BuildHasseGraph(algset);
    for i = 1:algset.Count
        alg = AlgsetGet(algset, i);
        children = edges.Children{i};
        parents = edges.Parents{i};
        childrenSlow = hasseSlow.V{i};
        assert(all(sort(childrenSlow) == sort(sort(children))));
        
        for child = children
            algChild = AlgsetGet(algset, child);
            assert(all(alg <= algChild) && any(alg ~= algChild));
        end
        
        for parent = parents
            algParent = AlgsetGet(algset, parent);
            assert(all(alg >= algParent) && any(alg ~= algParent));
        end
    end

    % Test GetAllChildren and GetAllParents
    for i = 1:algset.Count
        alg = AlgsetGet(algset, i);
        children = GetAllChildren(edges, i);
        parents = GetAllParents(edges, i);
        for child = children
            algChild = AlgsetGet(algset, child);
            assert(all(alg <= algChild));
        end
        
        for parent = parents
            algParent = AlgsetGet(algset, parent);
            assert(all(alg >= algParent));
        end
    end
end