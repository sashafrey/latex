

function taskDesc(tasks)
printDesc(tasks.Anastomosis);
printDesc(tasks.Australian);
printDesc(tasks.Congressional_Voting);
printDesc(tasks.Echo_cardiogram);
printDesc(tasks.faults);
printDesc(tasks.German);
printDesc(tasks.glass);
printDesc(tasks.Heart_Disease);
printDesc(tasks.Hepatitis);
printDesc(tasks.Ionosphere);
printDesc(tasks.Labor_Relations);
printDesc(tasks.Letter);
printDesc(tasks.Liver_Disorders);
printDesc(tasks.Magic04);
printDesc(tasks.Molecular_Biology);
printDesc(tasks.Mushrooms);
printDesc(tasks.Optdigits);
printDesc(tasks.pageblocks);
printDesc(tasks.pendigits);
printDesc(tasks.pima);
printDesc(tasks.RipleySynth);
printDesc(tasks.Sonar);
printDesc(tasks.statlog);
printDesc(tasks.Thyroid_Disease);
printDesc(tasks.waveform);
printDesc(tasks.Wdbc);
printDesc(tasks.wine);
printDesc(tasks.Wisconsin_Breast_Cancer);
printDesc(tasks.Wpbc);
end

function printDesc(task)
nanCount = sum(sum(isnan(task.objects)))/(task.nItems*task.nFeatures);
fprintf('%s\t%d\t%d\t%d\t%g\n', task.name, task.nItems, task.nFeatures, sum(task.isnominal), nanCount);
end