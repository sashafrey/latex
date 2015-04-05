function h = plotstruct(foo, params ,X,Y, pltopts)
% h = plotstruct(mdl,X,Y) plot 2D and 3D model data and source data 
% 
% X [m,n] indenendent values
% Y [m,1] dependent values
% foo [handle] in the format foo = eval(['@ (w,x) ' some function ';']);
% params [1, W] parameter vector
% ATTENTION! the function MUST support row-vector operation
% 
% Example
%
% One-dimensional plot 
% X = [0:0.1:1]';
% Y = X + 0.1 * rand(11,1);
% f = 'w(1)* x(:,1)  + w(2) * sin(w(3) * x(:,1)) '
% foo = eval(['@ (w,x)' f ';']);
% params = [1 0.1 8];
% pltopts.display = 'on';
% plotstruct(foo, params ,X,Y, pltopts)
%
% Two-dimensional plot
% X = rand(100,2);
% Y = rand(100,1)-0.5;
% f = 'sin(w(1) * x(:,1)) .* cos(w(2) * x(:,2));' % ATTENTION! the function MUST support row-vector operation
% foo = eval(['@ (w,x)' f ';']);
% params = [10 20];
% pltopts.display = 'on';
% plotstruct(foo, params ,X,Y, pltopts);
%
% http://strijov.com
% Strijov, 08-may-08
h=[];
if nargin < 4, error('plotstruct', 'insuffitient arguments '); end   

% if there if no external tuning, make in the body
if nargin == 4,      
    pltopts.display = 'on';             % string, 'on' or 'off'
    %pltopts.legend = {'model','data'}; % string cell, i.e. {'Model', 'Data'}
    %pltopts.xlabel = 'K';               % string, i.e. 'x'
    %pltopts.ylabel = 't';               % string, i.e. '\alpha'
    %pltopts.zlabel = '\sigma^{imp}';    % string, i.e. '\sigma^{imp}'
    %pltopts.title  = 'Plot title';     % string, i.e. 'Plot title'
    %pltopts.data   = 'plot3';          % string, 'trimesh' or 'plot3' (default)
    %pltopts.ratio  = [1 1 1];           % [x y z], plot aspect ratio
    %pltopts.axis   = [18 28 0 .3 .5 5]; % [xmin xmax ymin ymax zmin zmax]
    %pltopts.view   =  [130,14];          % [Az, El], plot rotation
    %pltopts.ftype  = {'psc2', 'emf'};   % string cell, save figures in the mentioned formats, i.e. {'psc2', 'emf'}
    %pltopts.fignum = 1;                 % integer, the fist number of the auto-increasing file name for the picture
end

if isfield(pltopts,'display'), if ~strcmp(pltopts.display,'on'), return; end; end

global PLTOPTSFIGNUM;
pltopts.fignum =  PLTOPTSFIGNUM;

switch size(X,2)
    case 1
        try
            h = plot(X,foo(params,X),'r-');
            hold on
            plot(X,Y,'k.');
            % axis, ratio
            if isfield(pltopts,'axis'), axis(pltopts.axis); end            
            if isfield(pltopts,'ratio'), set(gca,'PlotBoxAspectRatio', pltopts.ratio); end
            % title, legend, labels
            if isfield(pltopts,'legend'), legend(pltopts.legend); end %legend({'Found function','Given data'},'Location','NorthOutside','Orientation','Horizontal'); 
            if isfield(pltopts,'xlabel'), xlabel(pltopts.xlabel); end
            if isfield(pltopts,'ylabel'), ylabel(pltopts.ylabel); end
            if isfield(pltopts,'zlabel'), zlabel(pltopts.zlabel); end
            if isfield(pltopts,'title'),  title(pltopts.title); end
            hold off
        catch
            return
        end%try 
    otherwise
        try
            Tri = delaunay(X(:,1),X(:,2));
            h = trisurf(Tri,X(:,1),X(:,2),foo(params,X),'FaceColor','red','EdgeColor','none');
            hold on
            camlight left; lighting phong; alpha(.8);
            if isfield(pltopts,'data'),  
                if strcmp(pltopts.data, 'trimesh')
                   trimesh(Tri,X(:,1),X(:,2),Y,'EdgeColor','black');
                    hidden off
                else        
                   plot3(X(:,1),X(:,2),Y,'k.');
                end
            else
                plot3(X(:,1),X(:,2),Y,'k.');
            end

            if isfield(pltopts,'axis'), axis(pltopts.axis); end
            if isfield(pltopts,'view'), view(pltopts.view); end
            if isfield(pltopts,'ratio'), set(gca,'PlotBoxAspectRatio', pltopts.ratio); end
            
            % title, legend, labels
            if isfield(pltopts,'legend'), legend(pltopts.legend); end
            if isfield(pltopts,'xlabel'), xlabel(pltopts.xlabel); end
            if isfield(pltopts,'ylabel'), ylabel(pltopts.ylabel); end
            if isfield(pltopts,'title'),  title(pltopts.title);   end

            hold off
        catch
            return
        end%try 
end%switch

% if there was no cath of the try, save the pictures
drawnow
if isfield(pltopts,'ftype')
    for i=1:length(pltopts.ftype)
        if ~isfield(pltopts,'fignum'), pltopts.fignum = 1; end        
        saveas(h,fullfile(pltopts.folder, sprintf('%04d',pltopts.fignum)), pltopts.ftype{i});
    end
    pltopts.fignum = pltopts.fignum + 1;
end
% useful formats, see also doc saveas
% 'psc2' stands for color 'eps', TeX
% 'png' 
% 'emf'
% 'eps' EPS Level 1

PLTOPTSFIGNUM = pltopts.fignum;
return