% ----------------------------------------------------------------------------%
%             Стилевой файл препринта Вычислительного центра РАН              %
%                       Сообщение по прикладной математике                    %
% ----------------------------------------------------------------------------%
%            copyleft   В.В.Стрижов, 2008   email: strijov@ccas.ru            %
% ----------------------------------------------------------------------------%
%
% Содержание стилевого файла ccaspreprint.sty:
%
% Подключение макропакетов
% Определение новых команд
% Настройка стиля "Препринт"
% Настройка размеров "Препринт"
% Определение титульной страницы
% Оформление списка литературы
% Верстка титульной страницы
%
% Примеры использования файла:
%
% example.tex - исходный текст препринта
% example.eps - пример рисунка в формате PostScript
% example.pdf - готовый текст припринта

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

% Кодировка, используемая по умолчанию Windows
\RequirePackage[cp1251]{inputenc}
% \RequirePackage[utf8]{inputenc}

% Русский язык, пакет babel.
% Проверить, включен ли russian (переносы русского языка) в настройках MikTeX.
% \RequirePackage[russian]{babel}
\RequirePackage[english,russian]{babel}

% Используется шрифт по умолчанию
% \renewcommand{\rmdefault}{ftm} % если снять комментарий, будет использован другой шрифт семейства roman
% \RequirePackage[T2A]{fontenc}
% ----------------------------------------------------------------------------
% Подключение макропакетов
% ----------------------------------------------------------------------------
% Часто используемые математические пакеты
\RequirePackage[fleqn]{amsmath}
\RequirePackage{amstext}  % Текст \text{} в формулах
\RequirePackage{amsfonts} % Определяет \mathfrak и \mathbb и др.
\RequirePackage{amsmath}  % Многострочные формулы
\RequirePackage{amssymb}  % Математические символы
\RequirePackage{amsthm}   % Теоремы
%\RequirePackage{amscd}   % Коммутативные диаграммы

% Определены окружения theorem (после загрузки пакета amsthm)
\newtheorem{theorem}{Теорема}%[section]
\newtheorem{lemm}{Лемма}
\newtheorem{df}{Определение}
\newtheorem{eq}{Пример}
\newtheorem{rem}{Замечание}
\newtheorem{cor}{Следствие}
\newtheorem{st}{Утверждение}
\newtheorem{task}{Задача}

% Рисунки в формате PostScript
\usepackage{graphicx, epsfig}
%\usepackage[dvips]{epsfig}
%\usepackage{epic}        % К окружению picture
%\usepackage{ecltree}     % Рисует деревья

% Ссылки внешние выделены синим цветом, прочие не выделены
%\usepackage[colorlinks,urlcolor=blue, linkcolor=black,  anchorcolor=black, citecolor=black]{hyperref}

% Условный переход
%\RequirePackage{ifthen}

% ----------------------------------------------------------------------------
% Определение новых команд
% ----------------------------------------------------------------------------
%\newcommand{\R}{\mathbb{R}} % Часто используется (напр. в вики)
%\newcommand{\eps}{{\varepsilon}}

% ----------------------------------------------------------------------------
% Настройка стиля Препринт - "Сообщение по прикладной математике"
% ----------------------------------------------------------------------------
% Первый параграф в разделе тоже должен начинаться с новой строки
\RequirePackage{indentfirst}

% После номера раздела должна стоять точка
\renewcommand{\thesection}{\arabic{section}.}
\renewcommand{\thesubsection}{\thesection\arabic{subsection}.}
\renewcommand{\thesubsubsection}{\thesubsection\alph{subsubsection}.}

% Разделы третьего уровня иерархии помечаются русскими буквами
\def\alph#1{\expandafter\@alph\csname c@#1\endcsname}
%\def\Alph#1{\expandafter\@Alph\csname c@#1\endcsname}

\def\@alph#1{%
     \ifcase#1\or а\or б\or в\or г\or д\or е\or ж\or з\or и\or й\or
        к\or л\or м\or н\or о\or п\or р\or с\or т\or у\or ф\or х\or
        ц\or ч\or ш\or щ\or ъ\or ы\or ь\or э\or ю\or я\else\@ctrerr\fi}%
%\def\@Alph#1{%
%     \ifcase#1\or А\or Б\or В\or Г\or Д\or Е\or Ж\or З\or И\or Й\or
%        К\or Л\or М\or Н\or О\or П\or Р\or С\or Т\or У\or Ф\or Х\or
%        Й\or Ч\or Ш\or Щ\or Ъ\or Ы\or Ь\or Э\or Ю\or Я\else\@ctrerr\fi}%

% Названия разделов стоят по центру и имеют следующий стиль
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

% После номера рисунка или таблицы стоит точка.
% Название рисунки или таблицы находится по центру
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

% Заметки на полях (не используются)
%\newcommand{\marg}[1]{\marginpar{{\tiny{\begin{tabular}{|p{3cm}}#1\end{tabular}}}}}

% Переопределено оформление номеров страниц
\renewcommand{\@evenfoot}{\hfil ---~\thepage~--- \hfil}
\renewcommand{\@oddfoot}{\hfil ---~\thepage~--- \hfil}

% ----------------------------------------------------------------------------
% Настройка размеров "Препринт"
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

% Межстрочный интервал стандартный = 1.00; 1.67 - для редакторских правок
% \renewcommand{\baselinestretch}{1.67} % снять комментарий для удобства редактирования
% при этом рекомендуется отказаться от размеров, принятых выше и
% использовать пакет a4wide

% Не делать переносы в конце параграфа.
\sloppy

% ----------------------------------------------------------------------------
% Определение полей титульной страницы
% ----------------------------------------------------------------------------
% Переопределяем поле "Автор"
\gdef\@author{}
\def\author#1{\ifx\@author\@empty\else\g@addto@macro\@author\fi
    \g@addto@macro\@author{\mbox{\uppercase{#1}}}}

% Переопределяем поле "Название"
\gdef\@title{}
\def\title#1{\ifx\@title\@empty\else\g@addto@macro\@title\fi
    \g@addto@macro\@title{\uppercase{#1}}}

% Определяем поле "УДК"
\gdef\@udk{}
\def\udk#1{\ifx\@udk\@empty\else\g@addto@macro\@udk\fi
    \g@addto@macro\@udk{\mbox{УДК #1}}}

% Определяем поле "Ответственный редактор"
\gdef\@editor{}
\def\editor#1{\ifx\@editor\@empty\else\g@addto@macro\@editor\fi
    \g@addto@macro\@editor{\mbox{#1}}}

% Определяем поле "Аннотация"
\gdef\@anno{}
\def\anno#1{\ifx\@anno\@empty\else\g@addto@macro\@anno\fi
    \g@addto@macro\@anno{#1}}

% Определяем поле "Ключевые слова"
\gdef\@keywords{}
\def\keywords#1{\ifx\@keywords\@empty\else\g@addto@macro\@keywords\fi
    \g@addto@macro\@keywords{Ключевые слова: \emph{#1.}}}

% Определяем поле "Рецензенты"
\gdef\@reviewer{}
\def\reviewer[#1]#2{
    \ifx\@reviewer\@empty\else\g@addto@macro\@reviewer{,\par }\fi
    \g@addto@macro\@reviewer{\ifx\\#1\\\else\fi#2}
}

% Определяем поле "Год публикации"
\gdef\@year{}
\def\year#1{\ifx\@year\@empty\else\g@addto@macro\@year\fi
    \g@addto@macro\@year{#1}}

% ----------------------------------------------------------------------------
% Верстка титульной страницы
% ----------------------------------------------------------------------------
\def\maketitle{
% Обложка
    \thispagestyle{empty}
    \begin{center}
        РОССИЙСКАЯ АКАДЕМИЯ НАУК\\ ВЫЧИСЛИТЕЛЬНЫЙ ЦЕНТР\\
        \hbox to 11.3cm{\hrulefill}~\\
        СООБЩЕНИЯ ПО ПРИКЛАДНОЙ МАТЕМАТИКЕ\\[3cm]
        {\uppercase{\@author}}\\[1cm]
        {\bf \uppercase{\@title}}\\
        \vfill
        ВЫЧИСЛИТЕЛЬНЫЙ ЦЕНТР РАН\\
        МОСКВА, {\@year}
    \end{center}
    \clearpage
% Форзац
    \thispagestyle{empty}
    \noindent{\@udk}
    \begin{center}
        Ответственный редактор\\
        {\@editor}
    \end{center}~\\[1cm]
    \indent{\@anno}
    \bigskip\\
    \indent{\@keywords}
    \\[1cm]
    \indent\begin{tabular}{@{}lp{7cm}}
        Рецензенты:&
        {\@reviewer}\\
    \end{tabular}
    \\ \vfill
    \begin{center}
        Научное издание\end{center}
        \indent\begin{tabular}{@{}ll}
            $\copyright$& Вычислительный центр им.~А.\,А.~Дородницына\\
            ~           & Российской академии наук,~{\@year}\\
        \end{tabular}
    \clearpage
}

% ----------------------------------------------------------------------------
% Оформление списка литературы
% ----------------------------------------------------------------------------
% Фамилия автора выделяется курсивом с помощью команды \BibAuthor{Автор~A.\,A.}
\ifx\undefined\BibAuthor\def\BibAuthor#1{\emph{#1}}\else\fi

\gdef\@langeng{} % По умолчанию задан русский язык
\def\langeng#1{\gdef\@langeng{#1}}

% Начало окружения thebibliography
\renewenvironment{thebibliography}[1]
{
  % Литература - это раздел без нумерации, присутствующий в оглавлении
  \ifx\@langeng\@empty\section*{Литература}{\addcontentsline{toc}{section}{Литература}}
    \else
    \section*{References}{\addcontentsline{toc}{section}{References}}
  \fi
  % Литература идет нумерованным списком. Номер оканчивается точкой.
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
% Ошибка, если список литературы пуст. В этом случае нужно убрать окружение thebibliography
{\def\@noitemerr
  {\@latex@warning{Empty `thebibliography' environment}}%
  \endlist}
