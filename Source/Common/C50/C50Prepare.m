function task = C50Prepare(task)
    filename = mexCreateGUID;
    filename(filename=='{' | filename=='}') = [];
    ConvertTaskToC50NamesFile(task, filename, 'C:\\temp');
    task.filename = filename;
end