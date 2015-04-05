function VisualizeLinearSet(sample, sampleClasses)

    function AddNewAlgorithm(A, B, C)
      
        alg = sign(sampleClasses .* (A * sample(:, 1) + B * sample(:, 2) + C)' );
        alg(alg == 1) = 0;
        alg(alg == -1) = 1;
        isNew = true;
        for n = 1:numAlgs
            if (alg == algs(n, :) )
                isNew = false;
            end
        end
        
        if (~isNew)
            return 
        end
        
        numAlgs = numAlgs + 1;
        algs(numAlgs, :) = alg;
        d = sqrt(A^2 + B^2);
        x(numAlgs) =-C / A
        y(numAlgs) = (-C - B) / A;
        for  n = 1:numAlgs
            if ( sum( abs( algs(n, :) -  algs(numAlgs, :) ) ) == 1)
                line([x(n) x(numAlgs)], [y(n) y(numAlgs)]); %[ sum(algs(n, :) ) sum(alg) ]);
             end
         end
    end
    [L dim] = size(sample);
   
    
    %ќцениваем величину шага, позвол€ющую различать любые две точки
    numAlgs = 0;
    algs = zeros(1, L);
    x = [];
    y = [];
    figure
    hold on
    eps = 1e-6;
  
    for i = 1:L
        for j = i + 1:L
            A = sample(j, 2) - sample(i, 2);
            B = sample(i, 1) - sample(j, 1);
            C = - A * sample(i, 1) - B * sample(i, 2);
            %d = sqrt(A^2 + B^2 + C^2);
            %A = A / d;
            %B = B / d;
            %C = C / d;
            if (B == 0)
                'horizontal line'
            end
         
           AddNewAlgorithm(A - eps, B, C);
           AddNewAlgorithm(A + eps, B, C);
           AddNewAlgorithm(A, B + eps, C);
           AddNewAlgorithm(A, B - eps, C);
           
           AddNewAlgorithm(-A - eps, -B, -C);
           AddNewAlgorithm(-A + eps, -B, -C);
           AddNewAlgorithm(A, -B + eps, -C);
           AddNewAlgorithm(A, -B - eps, -C);
             
           
            
        end
    end
    
    err = sum(algs, 2);
    
    scatter(x, y,10, 'g', 'filled');
    %scatter3(x, y, err,10, 'g', 'filled');
    hold off
end