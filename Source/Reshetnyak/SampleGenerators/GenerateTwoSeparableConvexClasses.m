function [sample, sampleClasses] = GenerateTwoSeparableConvexClasses(sampleSize, angle, d)
    t = sampleSize / 2;
    sampleClasses = [repmat(-1, t, 1) repmat(1, sampleSize - t, 1)];
    angles = 0.5 * (pi - angle) + [0 : t - 1] * angle / t;
    
    sample = [cos(angles) cos(angles); sin(angles) d - sin(angles)]';
end