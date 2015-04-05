function [sample, sampleClasses] = GenerateRandomSample(sampleSize, dim)

    function [res] = ThreePointsOnLine(n)
        
        res = false;
        for i = 1:n - 1
            for j = i + 1:n - 1
                 if ( sum( (sample(n, :) - sample(i, :)) .* (sample(n, :) - sample(j, :)) ) == norm(sample(n, :) - sample(i, :)) *  norm(sample(n, :) - sample(j, :)) )
                     res = true;
                 end
            end
           
        end
    end

   sample = zeros(sampleSize, dim);
   sampleClasses = zeros(1, sampleSize);
   
   sample(1,  :) = rand(1, dim);
   
   for n = 2:sampleSize
       
       sample(n, :) = rand(1, dim);
     
       while (ThreePointsOnLine(n))
           sample(n, :) = rand(1, dim); 
       end
   end
   
   sampleClasses(sample(:, 1) < 0.5) = 1;
   sampleClasses(sample(:, 1) >= 0.5) = -1;
   
end