import os
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils import module_path


# agg backend is used to create plot as a .png file
mpl.use('agg')


def boxplot_bss_eval_metrics(eval_results, save_name):
    # create a figure instance
    fig = plt.figure(1, figsize=(9, 6))
    # create an axes instance
    ax = fig.add_subplot(111)
    ax.set_xticklabels(['vocal', 'bass', 'drums', 'other'])
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # create the boxplot
    bp = ax.boxplot(eval_results)
    # save the figure
    filename = save_name + '.png'
    fig.savefig(filename, bbox_inches='tight')


def print_metrics_mean_value(model_name):
    # get eval results path
    evaluation_path = module_path.get_evaluation_path()
    eval_results_path = os.path.join(evaluation_path, model_name)
    # to store results
    bss_metrics = ['sdr', 'sir', 'sar', 'nsdr']
    sum_values = {'sdr': [0]*4, 'sir': [0]*4, 'sar': [0]*4, 'nsdr': [0]*4}
    # calculate accumelated sum
    num_json_files = len(os.listdir(eval_results_path))
    for file in os.listdir(eval_results_path):
        filename = os.path.join(eval_results_path, file)
        with open(filename, 'r', encoding='utf-8') as fd:
            bss_eval_results = json.load(fd)
            for i in range(4):
                for metric in bss_metrics:
                    sum_values[metric][i] += bss_eval_results[metric][i]
    # calculate mean values
    mean_values = dict()
    for metric in bss_metrics:
        mean_values[metric] = [value / num_json_files for value in sum_values[metric]]
    # print results
    print('stem component order: [vocals, bass, drums, other]')
    for metric in bss_metrics:
        print('mean value of ' + metric + ' ' + mean_values[metric])


def plot_model_eval_results(metric_name, model_name):
    boxplot_data = [list(), list(), list(), list()]
    # get eval results path
    evaluation_path = module_path.get_evaluation_path()
    eval_results_path = os.path.join(evaluation_path, model_name)
    num_json_files = len(os.listdir(eval_results_file_dir))
    for file in os.listdir(eval_results_path):
        filename = os.path.join(eval_results_path, file)
        with open(filename, 'r', encoding='utf-8') as fd:
            eval_metrics = json.load(fd)
            for i in range(4):
                boxplot_data[i].append(eval_metrics[metric_name][i])
    # output boxplot figure
    figure_name = metric_name + '_' + model_name
    boxplot_bss_eval_metrics(boxplot_data, figure_name)


if __name__ == '__main__':
    model_name = 'conv_res56_denoising_unet?time=20200227_0646_l2_reg.h5'
    print_metrics_mean_value(model_name)
    for metric in ['sdr', 'sir', 'sar', 'nsdr']:
        plot_model_eval_results(metric, model_name)
