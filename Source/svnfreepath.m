function svnfreepath(rootdir)
% add to matlab a path (with subfolders) free from .svn folders
%
% svnfreepath(rootdir)
%
% Example
% svnfreepath('D:\Mathnb\mvr61')
%
% http://strijov.com
% 19-Oct-2008

path(rootdir,path)

dirlist = gatherdirlist(rootdir, {});

% append path
for newpath = dirlist
    if isempty( strfind(newpath{1}, '.svn') )
        disp(newpath{1});
        path(newpath{1},path);
    end
end

% save the .svn-free path
if savepath == 1 % see doc
    disp('cannot save the new path');
else
    disp('the new path was saved');
end
return

    
function dirlist = gatherdirlist(dirnom, dirlist)
% recursive gathering of the directory list
flist = dir(dirnom);      
%search for the very first file in the list    
for f = flist';
    if all( [f.isdir, ~strcmp(f.name,'.'), ~strcmp(f.name,'..')] )      
        if (strcmp(f.name, 'private'))
            continue;
        end
        
        newdir = fullfile(dirnom, f.name);
        dirlist{end+1} = newdir;
        %fprintf(1,'%s\n',newdir);
        dirlist = gatherdirlist(newdir, dirlist);
    end
end
return