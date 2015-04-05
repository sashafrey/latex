%%
GSampleSize = 1;
GSample = zeros(GSampleSize, 3);
G_CCV = zeros(GSampleSize, 1);

%%
for G_idx = 1:GSampleSize
    %%
    N = 100;
    layers = [5, 10, 20, 30, 35];

    degrees = [0, 1, 2, 3, 4];
    degreeProbs = [0.05, 0.25, 0.4, 0.25, 0.05];

    G = zeros(N, N);
    %A = zeros(N, N);
    layerNumber = zeros(N, 1);
    last = 0;
    for i = 1:length(layers)
        from = last + 1;
        to = last + layers(i);
        layerNumber(from:to) = i;
        last = last + layers(i);
    end

    for vertexIdx = 1:N
        currLayer = layerNumber(vertexIdx);
        % ребро вверх или вниз?
        for dir = -1:1
            if (currLayer == 1 && dir == -1) || ...
                    (currLayer == max(layerNumber) && dir == 1)
                continue;
            end

            currDegree = degrees(find(mnrnd(1, degreeProbs)));
            for j = 1:currDegree
                neighLayerIdx = find(layerNumber == currLayer + dir);
                currNeighIdx = randsample(neighLayerIdx, 1);
                G(vertexIdx, currNeighIdx) = 1;
                G(currNeighIdx, vertexIdx) = 1;
            end
        end
    end
    
    %% считаем CCV монтекарлой
    currCCV = 0; % вот так и живем
    
    %%
    GL = G;
    GL(GL == 1) = -1;
    for i = 1:N
        GL(i, i) = sum(G(i, :));
    end

    lambda = eig(GL);
    lambda = sort(lambda, 'descend');

    %%
    GSample(G_idx, 1) = lambda(1);
    GSample(G_idx, 2) = lambda(2);
    GSample(G_idx, 3) = norm(lambda);
end

%%
