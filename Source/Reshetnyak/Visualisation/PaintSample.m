function PaintSample(sample, sampleClasses)
    dim = size(sample, 2)
    good = find(sampleClasses == 1);
    bad = find(sampleClasses ~= 1);
    for i = 1 : dim
        for j = i + 1 : dim
            figure
            hold on
            grid on
            scatter(sample(good, i), sample(good, j), 30, 'b');
            scatter(sample(bad, i), sample(bad, j), 30, 'r', '*');
            hold off
        end
    end
end