1) Python script create_proc_plot.py is using for creating gantt chart of threads in BigARTM library.
   Usage:

   > python create_proc_plot.py <input log file name> base_plot.tex plot.tex

2) To create the perlexity-time plot with perplexity_time_plot.tex you shold specify four files:
   - offline.dat
   - online_old.dat
   - online_new.dat
   - online_async.dat

   All files have the same format: each line contains pair x y, separated with one space, x -- time, y -- perlexity. 