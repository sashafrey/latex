function [ algorithms ] = AlgsetGet(algset, ids)
    algorithms = UnpackLogicals(algset.Data(ids, :), algset.L);
end

