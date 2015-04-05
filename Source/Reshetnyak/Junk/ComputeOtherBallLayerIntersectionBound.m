function [val] = ComputeOtherBallLayerIntersectionBound(sampleSize, trainSize, errorLevel, radius, eps)
  threshold = TrainErrorOverfitThreshold(sampleSize, trainSize, errorLevel, eps);
  semiRadius = floor(radius/2);
  val = 0;
  if threshold < 0
      return 
  end
  choose = ComputeChooseTable(sampleSize);
  for h = errorLevel - semiRadius : errorLevel
      factor = 0;
      for i = max(trainSize - (sampleSize - 2 * errorLevel + h), 0) : min(trainSize, errorLevel - h)
          denominator = 0;
          for j = errorLevel - semiRadius : errorLevel
              denominator = denominator + choose(errorLevel - i + 1, j + 1) * ...
                            choose(sampleSize - trainSize - errorLevel + i + 1, errorLevel - j + 1);
          end
          factor = factor + choose(errorLevel - h + 1, i + 1) * ...
              choose(sampleSize - 2 * errorLevel + h + 1, trainSize - i + 1) / denominator;
      end
      val = val + factor * choose(errorLevel + 1, h + 1) * choose(sampleSize - errorLevel + 1, errorLevel - h + 1);
  end
  
  if trainSize < semiRadius
      return 
  end
  secondSum = 0;
  for i = semiRadius + 1 : min(trainSize, errorLevel)
      if i - semiRadius <= threshold
          secondSum = secondSum + ...
              choose(errorLevel - semiRadius + 1, errorLevel - i + 1) * ...
              choose(sampleSize - errorLevel - semiRadius + 1, trainSize - i + 1) / ...
              choose(i + 1, semiRadius + 1) / ...
              choose(sampleSize - errorLevel - trainSize + i + 1, semiRadius + 1);
      end
  end
  val = val + secondSum * choose(errorLevel + 1, semiRadius + 1) * ...
      choose(sampleSize - errorLevel + 1, semiRadius + 1);
end