function file = PublishReportToHtml(ReportFunc, InputData, OutputDir)
% Публикует отчёт по файлу ReportFile в файл в директории OutputDir
% Возвращает полный путь к файлу

% Все аргументы функции --- обязательные.
% Сигнатура ReportFunc обязана иметь хотя бы один аргумент.

opts.showCode = false;

opts.outputDir = OutputDir;
assignin('base', 'reportInput__', InputData);
opts.codeToEvaluate = [ReportFunc '(reportInput__);'];

warning('off','all');
file = publish([ReportFunc '.m'],opts);
warning('on','all');
evalin('base', 'clear(''reportInput__'')');
end