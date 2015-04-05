function file = PublishReportToHtml(ReportFunc, InputData, OutputDir)
% ��������� ����� �� ����� ReportFile � ���� � ���������� OutputDir
% ���������� ������ ���� � �����

% ��� ��������� ������� --- ������������.
% ��������� ReportFunc ������� ����� ���� �� ���� ��������.

opts.showCode = false;

opts.outputDir = OutputDir;
assignin('base', 'reportInput__', InputData);
opts.codeToEvaluate = [ReportFunc '(reportInput__);'];

warning('off','all');
file = publish([ReportFunc '.m'],opts);
warning('on','all');
evalin('base', 'clear(''reportInput__'')');
end