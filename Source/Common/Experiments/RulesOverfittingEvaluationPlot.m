function RulesOverfittingEvaluationPlot(rules, terms, task, class)
    color = 'b';    
    fromCenter = false;
    
    [pTrain, nTrain, pTest, nTest] = RulesOverfittingEvaluation(rules, terms, task);
    RulesOverfittingPlot(pTrain, nTrain, pTest, nTest, class, color, fromCenter);
end