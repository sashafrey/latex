function LayerVsSphere

    dim = 7;
    L = 100;
    l = L / 2;
    m0 = 5;
    
    cTable = zeros(L + 1, L + 1);
    for i = 1:L+1
        cTable(i, 1) = 1;
        for j = 2:i 
            cTable(i, j) = cTable(i - 1, j - 1) + cTable(i - 1, j);
    end

       
    layer = zeros(1, L - m0);
    
    for i = 1:L-m0
        layer(i) = cTable(L - m0 + 1, i + 1);
    end
    
    sphere = zeros(1, L - m0);
    
    for r = 1:L-m0
        for k = 0:dim-1
            sphere(r) = sphere(r) + 2^(dim - k) * cTable(dim + 1, k + 1) * cTable(r, dim - k);
        end
    end
    
    hold on
    k = 20;
    plot([1:k], sphere(1:k) ./ layer(1:k), 'g');
    %plot(, layer(1:k), 'r');

    
end