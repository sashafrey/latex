# Usage:
# python create_proc_plot.py <input log file name> base_plot.tex plot.tex

import datetime
import time
import sys

# parse user-defined parameters
input_log_file = sys.argv[1]
input_tex_file = sys.argv[2]
output_tex_file = sys.argv[3]

# define some constants
BATCH_PROC_TASK = '@batch_proc_task'
REGULARIZING_TASK = '@regularizing_task'
NORMALIZING_TASK = '@normalizing_task'
MERGING_TASK = '@merging_task'

GLOBAL_START = 'Log file created at:'
START_BATCH_PROC = 'Processor: start processing batch'
COMPLETE_BATCH_PROC = 'Processor: complete processing batch'
START_REGULARIZING = 'MasterComponent: start regularizing'
COMPLETE_REGULARIZING = 'MasterComponent: complete regularizing'
START_NORMALIZING = 'MasterComponent: start normalizing'
COMPLETE_NORMALIZING = 'MasterComponent: complete normalizing'
START_MERGING = 'MasterComponent: start merging'
COMPLETE_MERGING = 'MasterComponent: complete merging'

START_TASK_TYPE = {
    START_BATCH_PROC: BATCH_PROC_TASK,
    START_REGULARIZING: REGULARIZING_TASK,
    START_NORMALIZING: NORMALIZING_TASK,
    START_MERGING: MERGING_TASK,
}

COMPLETE_TASK_TYPE = {
    COMPLETE_BATCH_PROC: BATCH_PROC_TASK,
    COMPLETE_REGULARIZING: REGULARIZING_TASK,
    COMPLETE_NORMALIZING: NORMALIZING_TASK,
    COMPLETE_MERGING: MERGING_TASK,
}


class Task(object):
    def __init__(self, task_type=None, start_time=None, complete_time=None, batch_name=None, model_name=None, rhs=None):
        self.task_type = task_type
        self.start_time = start_time
        self.complete_time = complete_time
        self.batch_name = batch_name
        self.model_name = model_name

        if rhs is not None:
            self.task_type = rhs.task_type
            self.start_time = rhs.start_time
            self.complete_time = rhs.complete_time
            self.batch_name = rhs.batch_name
            self.model_name = rhs.model_name
            
    def __str__(self):
        return '{0}\n {1}\n {2}\n {3}\n {4}\n'.format(self.task_type, self.start_time, self.complete_time,
                                                      self.batch_name, self.model_name)


def parse_log(log_file_name):

    def _make_comparator(less_than):
        def compare(x, y):
            if less_than(x, y):
                return -1
            elif less_than(y, x):
                return 1
            else:
                return 0
        return compare

    def _read_global_start_time(line):
        line_list = line.split()

        month, day = int(line_list[4][5: 7]), int(line_list[4][8: ])
        hour, minute, second = int(line_list[5][0: 2]), int(line_list[5][3: 5]), int(line_list[5][6: ])
        
        return time.mktime(datetime.datetime(year=2000, month=month, day=day, hour=hour,
                                             minute=minute, second=second).timetuple())

    def _read_task_info(line, global_start_time, read_batch):
        line_list = line.split()
        
        process_name = line_list[2]
        month, day, hour = int(line_list[0][1: 3]), int(line_list[0][3: ]), int(line_list[1][0: 2])
        minute, second, m_second = int(line_list[1][3: 5]), int(line_list[1][6: 8]), int(line_list[1][9: ])

        current_time = time.mktime(datetime.datetime(year=2000, month=month, day=day, hour=hour,
                                                     minute=minute, second=second).timetuple()) + float('0.{}'.format(m_second))
        batch_name = None
        if read_batch:
            batch_name = line_list[8]
        
        model_name = None
        if len(line_list) == 12:
            model_name = line_list[11]

        return process_name, current_time - global_start_time, batch_name, model_name

    max_time_stamp = -1
    min_time_stamp = None
    models_list = []
    process_info = {}  # dict, key --- process name, value --- list of tasks
    with open(log_file_name, 'r') as fin:
        incomplete_tasks = {}
        global_start_time = None
        for line in fin:
            if GLOBAL_START in line:
                global_start_time = _read_global_start_time(line)

            else:
                for task_name, task_type in START_TASK_TYPE.iteritems():
                    if task_name in line:
                        read_batch = (task_type == BATCH_PROC_TASK)
                        process_name, start_time, batch_name, model_name = _read_task_info(line,
                                                                                           global_start_time,
                                                                                           read_batch=read_batch)
                        task_name = '{0}-{1}'.format(task_type, process_name)
                        incomplete_tasks[task_name] = Task(task_type, start_time, -1, batch_name, model_name)

                        if min_time_stamp is None:
                            min_time_stamp = start_time

                        if (model_name is not None) and (model_name not in models_list):
                            models_list.append(model_name)
                        break

                for task_name, task_type in COMPLETE_TASK_TYPE.iteritems():
                    if task_name in line:
                        read_batch = (task_type == BATCH_PROC_TASK)
                        process_name, complete_time, batch_name, model_name = _read_task_info(line,
                                                                                              global_start_time,
                                                                                              read_batch=read_batch)
                        task_name = '{0}-{1}'.format(task_type, process_name)
                        if (task_name not in incomplete_tasks) or\
                             (read_batch and incomplete_tasks[task_name].batch_name != batch_name and\
                                             incomplete_tasks[task_name].model_name != model_name):
                            raise IOError('Given log-file is incorrect')

                        incomplete_tasks[task_name].complete_time = complete_time

                        if complete_time > max_time_stamp:
                            max_time_stamp = complete_time

                        if process_name not in process_info:
                            process_info[process_name] = []

                        process_info[process_name].append(Task(rhs=incomplete_tasks[task_name]))
                        del incomplete_tasks[task_name]
                        break

    for process_name in process_info.keys(): 
        process_info[process_name].sort(cmp=_make_comparator(lambda x, y: x.start_time > y.start_time), reverse=True)

    return process_info, max_time_stamp, min_time_stamp, models_list


process_info, max_time_stamp, min_time_stamp, models_list = parse_log(input_log_file)
for _, tasks in process_info.iteritems():
    for task in tasks:
        task.start_time -= min_time_stamp
        task.complete_time -= min_time_stamp
max_time_stamp -= min_time_stamp

# write info to latex file
coef = 10 / max_time_stamp
with open(input_tex_file, 'r') as fin:
    with open(output_tex_file, 'w') as fout:
        for line in fin:
            if '%%%%% START_MARKER_1 %%%%%' not in line:
                fout.write(line)
            else:
                fout.write('\\begin{{wave}}{{{}}}{{10}}\n'.format(len(process_info.keys()) + 1))
                for p_name in process_info.keys():
                    process_str = '\\nextwave{{{}}} '.format(p_name)
                    process_str += '\\Wait{{ }}{{{}}}'.format(process_info[p_name][0].start_time * coef)
                    for idx, _ in enumerate(process_info[p_name]):
                        if idx != len(process_info[p_name]) - 1:
                            val_1 = (process_info[p_name][idx].complete_time - process_info[p_name][idx].start_time) * coef
                            val_2 = (process_info[p_name][idx + 1].start_time - process_info[p_name][idx].complete_time) * coef
                            
                            if process_info[p_name][idx].task_type == BATCH_PROC_TASK:
                                index = models_list.index(process_info[p_name][idx].model_name)
                                if index % 2:
                                    process_str += '\\ProcBatchOne[ ]{{{0}}} \\Wait{{ }}{{{1}}} '.format(val_1, val_2)
                                else:
                                    process_str += '\\ProcBatchTwo[ ]{{{0}}} \\Wait{{ }}{{{1}}} '.format(val_1, val_2)
                            elif process_info[p_name][idx].task_type == REGULARIZING_TASK:
                                process_str += '\\Regularization[ ]{{{0}}} \\Wait{{ }}{{{1}}} '.format(val_1, val_2)
                            elif process_info[p_name][idx].task_type == NORMALIZING_TASK:
                                process_str += '\\Normalization[ ]{{{0}}} \\Wait{{ }}{{{1}}} '.format(val_1, val_2)
                            elif process_info[p_name][idx].task_type == MERGING_TASK:
                                process_str += '\\Merge[ ]{{{0}}} \\Wait{{ }}{{{1}}} '.format(val_1, val_2)
                        else:
                            val = (process_info[p_name][idx].complete_time - process_info[p_name][idx].start_time) * coef
                            if process_info[p_name][idx].task_type == BATCH_PROC_TASK:
                                index = models_list.index(process_info[p_name][idx].model_name)
                                if index % 2:
                                    process_str += '\\ProcBatchOne[ ]{{{0}}} '.format(val)
                                else:
                                    process_str += '\\ProcBatchTwo[ ]{{{0}}} '.format(val)
                            elif process_info[p_name][idx].task_type == REGULARIZING_TASK:
                                process_str += '\\Regularization[ ]{{{0}}} '.format(val)
                            elif process_info[p_name][idx].task_type == NORMALIZING_TASK:
                                process_str += '\\Normalization[ ]{{{0}}} '.format(val)
                            elif process_info[p_name][idx].task_type == MERGING_TASK:
                                process_str += '\\Merge[ ]{{{0}}} '.format(val)

                    fout.write('{}\n'.format(process_str))
