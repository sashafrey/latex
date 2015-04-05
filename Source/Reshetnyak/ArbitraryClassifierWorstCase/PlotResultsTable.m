function PlotTestResultsTable(resultsTable, xRange, yRange, caption, xLabel, yLabel)

    function [res] = Rescale(number, range, maxNumber)
        if (numel(range) == 2)
            res = number/maxNumber * (range(2) - range(1)) - range(1);
        else
            res = number;
        end    
    end

    tableSize = size(resultsTable);
  
    figure('Colormap', 1 - gray(100)); 
    for i = 1:tableSize(1)
        for j = 1:tableSize(2)
            %rectangle('Position', [Rescale(i - 1, xRange, tableSize(1)), Rescale(j - 1, yRange, tableSize(2)),...
            %Rescale(1, xRange, tableSize(1)), Rescale(1, yRange, tableSize(2)) ], ...
            rectangle('Position',[i - 1, j - 1, 1, 1] , ...
            'FaceColor', max(0 , (1 - resultsTable(i, j)) * [1 1 1]), 'EdgeColor', 'none');
        end
    end
    
    title(caption);
    xlabel(xLabel);
    ylabel(yLabel);
end