% ----------------------------------------------------------------------------%
%             �������� ���� ��������� ��������������� ������ ���              %
%                       ��������� �� ���������� ����������                    %
% ----------------------------------------------------------------------------%
%            copyleft   �.�.�������, 2008   email: strijov@ccas.ru            %
% ----------------------------------------------------------------------------%
%
% ���������� ��������� ����� ccaspreprint.sty:
%
% ����������� ������������
% ����������� ����� ������
% ��������� ����� "��������"
% ��������� �������� "��������"
% ����������� ��������� ��������
% ���������� ������ ����������
% ������� ��������� ��������
%
% ������� ������������� �����:
%
% example.tex - �������� ����� ���������
% example.eps - ������ ������� � ������� PostScript
% example.pdf - ������� ����� ���������

\ProvidesClass{ccaspreprint}[2008/11/25 v0.0.0]
\def\class@name{ccaspreprint}%
\GenericInfo{}{\space
 Copyright (c) 2008 CCAS Moscow^^J
 Licensed under the LPPL:^^Jhttp://www.ctan.org/tex-archive/macros/latex/base/lppl.txt^^J
 \@gobble
}%
\LoadClass[11pt,a4paper]{article}
%\RequirePackage{a4}
\RequirePackage{a4wide}

% ���������, ������������ �� ��������� Windows
\RequirePackage[cp1251]{inputenc}
% \RequirePackage[utf8]{inputenc}

% ������� ����, ����� babel.
% ���������, ������� �� russian (�������� �������� �����) � ���������� MikTeX.
% \RequirePackage[russian]{babel}
\RequirePackage[english,russian]{babel}

% ������������ ����� �� ���������
% \renewcommand{\rmdefault}{ftm} % ���� ����� �����������, ����� ����������� ������ ����� ��������� roman
% \RequirePackage[T2A]{fontenc}
% ----------------------------------------------------------------------------
% ����������� ������������
% ----------------------------------------------------------------------------
% ����� ������������ �������������� ������
\RequirePackage[fleqn]{amsmath}
\RequirePackage{amstext}  % ����� \text{} � ��������
\RequirePackage{amsfonts} % ���������� \mathfrak � \mathbb � ��.
\RequirePackage{amsmath}  % ������������� �������
\RequirePackage{amssymb}  % �������������� �������
\RequirePackage{amsthm}   % �������
%\RequirePackage{amscd}   % ������������� ���������

% ���������� ��������� theorem (����� �������� ������ amsthm)
\newtheorem{theorem}{�������}%[section]
\newtheorem{lemm}{�����}
\newtheorem{df}{�����������}
\newtheorem{eq}{������}
\newtheorem{rem}{���������}
\newtheorem{cor}{���������}
\newtheorem{st}{�����������}
\newtheorem{task}{������}

% ������� � ������� PostScript
\usepackage{graphicx, epsfig}
%\usepackage[dvips]{epsfig}
%\usepackage{epic}        % � ��������� picture
%\usepackage{ecltree}     % ������ �������

% ������ ������� �������� ����� ������, ������ �� ��������
%\usepackage[colorlinks,urlcolor=blue, linkcolor=black,  anchorcolor=black, citecolor=black]{hyperref}

% �������� �������
%\RequirePackage{ifthen}

% ----------------------------------------------------------------------------
% ����������� ����� ������
% ----------------------------------------------------------------------------
%\newcommand{\R}{\mathbb{R}} % ����� ������������ (����. � ����)
%\newcommand{\eps}{{\varepsilon}}

% ----------------------------------------------------------------------------
% ��������� ����� �������� - "��������� �� ���������� ����������"
% ----------------------------------------------------------------------------
% ������ �������� � ������� ���� ������ ���������� � ����� ������
\RequirePackage{indentfirst}

% ����� ������ ������� ������ ������ �����
\renewcommand{\thesection}{\arabic{section}.}
\renewcommand{\thesubsection}{\thesection\arabic{subsection}.}
\renewcommand{\thesubsubsection}{\thesubsection\alph{subsubsection}.}

% ������� �������� ������ �������� ���������� �������� �������
\def\alph#1{\expandafter\@alph\csname c@#1\endcsname}
%\def\Alph#1{\expandafter\@Alph\csname c@#1\endcsname}

\def\@alph#1{%
     \ifcase#1\or �\or �\or �\or �\or �\or �\or �\or �\or �\or �\or
        �\or �\or �\or �\or �\or �\or �\or �\or �\or �\or �\or �\or
        �\or �\or �\or �\or �\or �\or �\or �\or �\or �\else\@ctrerr\fi}%
%\def\@Alph#1{%
%     \ifcase#1\or �\or �\or �\or �\or �\or �\or �\or �\or �\or �\or
%        �\or �\or �\or �\or �\or �\or �\or �\or �\or �\or �\or �\or
%        �\or �\or �\or �\or �\or �\or �\or �\or �\or �\else\@ctrerr\fi}%

% �������� �������� ����� �� ������ � ����� ��������� �����
\renewcommand\section{\@startsection {section}{1}{\z@}%
                                   {-3.5ex \@plus -1ex \@minus -.2ex}%
                                   {2.3ex \@plus.2ex}%
                                   {\centering\normalfont\Large\bfseries}}
\renewcommand\subsection{\@startsection {subsection}{1}{\z@}%
                                   {-3.5ex \@plus -1ex \@minus -.2ex}%
                                   {1.5ex \@plus .2ex}%
                                   {\centering\normalfont\large\bfseries}}
\renewcommand\subsubsection{\@startsection {subsubsection}{1}{\z@}%
                                   {-3.5ex \@plus -1ex \@minus -.2ex}%
                                   {1.5ex \@plus .2ex}%
                                   {\centering\normalfont\normalsize\bfseries}}

% ����� ������ ������� ��� ������� ����� �����.
% �������� ������� ��� ������� ��������� �� ������
\renewcommand{\@makecaption}[2]{%
    \vspace{\abovecaptionskip}%
    \centering % inserted to center the figure caption
    \sbox{\@tempboxa}{#1. #2}
    \ifdim
        \wd\@tempboxa>\hsize  #1. #2\par%
    \else
        \global\@minipagefalse
        \hbox to \hsize {\hfil #1. #2\hfil}%
    \fi%
    \vspace{\belowcaptionskip}
}

% ������� �� ����� (�� ������������)
%\newcommand{\marg}[1]{\marginpar{{\tiny{\begin{tabular}{|p{3cm}}#1\end{tabular}}}}}

% �������������� ���������� ������� �������
\renewcommand{\@evenfoot}{\hfil ---~\thepage~--- \hfil}
\renewcommand{\@oddfoot}{\hfil ---~\thepage~--- \hfil}

% ----------------------------------------------------------------------------
% ��������� �������� "��������"
% ----------------------------------------------------------------------------
\unitlength=1cm
\setlength\hoffset{0cm}             %plus one inch \hoffset=0cm
\setlength\voffset{3cm}             %\voffset=3cm
%
\setlength\textheight{16.2cm}       %\textheight=16.2cm
\setlength\textwidth{11.5cm}        %\textwidth=11.5cm
\setlength\topmargin{-0.5cm}        %\topmargin=-0.5cm
\setlength\oddsidemargin{2.3cm}     %\oddsidemargin=2.3cm %.8cm
\setlength\evensidemargin{2.3cm}    %\evensidemargin=2.3cm %.8cm
                                    %\marginparsep=0.3cm
                                    %\marginparwidth=2.0cm
                                    %\headheight=0.5cm
                                    %\headsep=0.5cm

% ����������� �������� ����������� = 1.00; 1.67 - ��� ������������ ������
% \renewcommand{\baselinestretch}{1.67} % ����� ����������� ��� �������� ��������������
% ��� ���� ������������� ���������� �� ��������, �������� ���� �
% ������������ ����� a4wide

% �� ������ �������� � ����� ���������.
\sloppy

% ----------------------------------------------------------------------------
% ����������� ����� ��������� ��������
% ----------------------------------------------------------------------------
% �������������� ���� "�����"
\gdef\@author{}
\def\author#1{\ifx\@author\@empty\else\g@addto@macro\@author\fi
    \g@addto@macro\@author{\mbox{\uppercase{#1}}}}

% �������������� ���� "��������"
\gdef\@title{}
\def\title#1{\ifx\@title\@empty\else\g@addto@macro\@title\fi
    \g@addto@macro\@title{\uppercase{#1}}}

% ���������� ���� "���"
\gdef\@udk{}
\def\udk#1{\ifx\@udk\@empty\else\g@addto@macro\@udk\fi
    \g@addto@macro\@udk{\mbox{��� #1}}}

% ���������� ���� "������������� ��������"
\gdef\@editor{}
\def\editor#1{\ifx\@editor\@empty\else\g@addto@macro\@editor\fi
    \g@addto@macro\@editor{\mbox{#1}}}

% ���������� ���� "���������"
\gdef\@anno{}
\def\anno#1{\ifx\@anno\@empty\else\g@addto@macro\@anno\fi
    \g@addto@macro\@anno{#1}}

% ���������� ���� "�������� �����"
\gdef\@keywords{}
\def\keywords#1{\ifx\@keywords\@empty\else\g@addto@macro\@keywords\fi
    \g@addto@macro\@keywords{�������� �����: \emph{#1.}}}

% ���������� ���� "����������"
\gdef\@reviewer{}
\def\reviewer[#1]#2{
    \ifx\@reviewer\@empty\else\g@addto@macro\@reviewer{,\par }\fi
    \g@addto@macro\@reviewer{\ifx\\#1\\\else\fi#2}
}

% ���������� ���� "��� ����������"
\gdef\@year{}
\def\year#1{\ifx\@year\@empty\else\g@addto@macro\@year\fi
    \g@addto@macro\@year{#1}}

% ----------------------------------------------------------------------------
% ������� ��������� ��������
% ----------------------------------------------------------------------------
\def\maketitle{
% �������
    \thispagestyle{empty}
    \begin{center}
        ���������� �������� ����\\ �������������� �����\\
        \hbox to 11.3cm{\hrulefill}~\\
        ��������� �� ���������� ����������\\[3cm]
        {\uppercase{\@author}}\\[1cm]
        {\bf \uppercase{\@title}}\\
        \vfill
        �������������� ����� ���\\
        ������, {\@year}
    \end{center}
    \clearpage
% ������
    \thispagestyle{empty}
    \noindent{\@udk}
    \begin{center}
        ������������� ��������\\
        {\@editor}
    \end{center}~\\[1cm]
    \indent{\@anno}
    \bigskip\\
    \indent{\@keywords}
    \\[1cm]
    \indent\begin{tabular}{@{}lp{7cm}}
        ����������:&
        {\@reviewer}\\
    \end{tabular}
    \\ \vfill
    \begin{center}
        ������� �������\end{center}
        \indent\begin{tabular}{@{}ll}
            $\copyright$& �������������� ����� ��.~�.\,�.~�����������\\
            ~           & ���������� �������� ����,~{\@year}\\
        \end{tabular}
    \clearpage
}

% ----------------------------------------------------------------------------
% ���������� ������ ����������
% ----------------------------------------------------------------------------
% ������� ������ ���������� �������� � ������� ������� \BibAuthor{�����~A.\,A.}
\ifx\undefined\BibAuthor\def\BibAuthor#1{\emph{#1}}\else\fi

\gdef\@langeng{} % �� ��������� ����� ������� ����
\def\langeng#1{\gdef\@langeng{#1}}

% ������ ��������� thebibliography
\renewenvironment{thebibliography}[1]
{
  % ���������� - ��� ������ ��� ���������, �������������� � ����������
  \ifx\@langeng\@empty\section*{����������}{\addcontentsline{toc}{section}{����������}}
    \else
    \section*{References}{\addcontentsline{toc}{section}{References}}
  \fi
  % ���������� ���� ������������ �������. ����� ������������ ������.
  %\list{\@biblabel{\@arabic\c@enumiv}}%
  \list{\@arabic\c@enumiv.}%
  {
    %\samepage
    \itemsep=0ex\topsep=0ex\partopsep=0ex\parskip=0ex\parsep=0ex
    \settowidth\labelwidth{\@biblabel{#1}}%
    \leftmargin\labelwidth
    \advance\leftmargin\labelsep
    \@openbib@code
    \usecounter{enumiv}%
    \let\p@enumiv\@empty
    \renewcommand\theenumiv{\@arabic\c@enumiv}}%
  %\sloppy
  %\clubpenalty4000
  \@clubpenalty \clubpenalty
  \widowpenalty4000%
  \sfcode`\.\@m}
% ������, ���� ������ ���������� ����. � ���� ������ ����� ������ ��������� thebibliography
{\def\@noitemerr
  {\@latex@warning{Empty `thebibliography' environment}}%
  \endlist}
