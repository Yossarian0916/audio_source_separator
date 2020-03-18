import os
import json
from collections import Iterable
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils import module_path


# agg backend is used to create plot as a .png file
mpl.use('agg')


def boxplot_bss_eval_metrics(eval_results, save_name, figure_title):
    # create a figure instance
    fig = plt.figure(1, figsize=(9, 6))
    # create an axes instance
    ax = fig.add_subplot()
    ax.title.set_text(figure_title)
    # create the boxplot
    bp = ax.boxplot(eval_results, patch_artist=True)

    # change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set(color='#7570b3', linewidth=2)
        # change fill color
        box.set(facecolor='#1b9e77')
    # change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)
    # change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)
    # change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)
    # change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)

    ax.set_xticklabels(['vocal', 'bass', 'drums', 'other'])
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # save the figure
    filename = save_name + '.png'
    fig.savefig(filename, bbox_inches='tight')


def print_metrics_mean_value(model_name):
    # get eval results path
    try:
        evaluation_path = module_path.get_evaluation_path()
        eval_results_path = os.path.join(evaluation_path, model_name)
    except FileNotFoundError:
        print("Wrong file path or wrong model file name")
    # to store results
    bss_metrics = ['sdr', 'sir', 'sar', 'nsdr']
    sum_values = {
        'sdr': [0] * 4,
        'sir': [0] * 4,
        'sar': [0] * 4,
        'nsdr': [0] * 4}
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
        mean_values[metric] = [
            value / num_json_files for value in sum_values[metric]]
    # print results
    print(model_name)
    print('stem component order: [vocals, bass, drums, other]')
    for metric in bss_metrics:
        print('average ' + metric + ': ', mean_values[metric])
    print('\n')


def plot_model_eval_results(metric_name, model_name):
    print(
        'generating boxplot of metric: ',
        metric_name,
        ', model:',
        model_name)
    boxplot_data = [list(), list(), list(), list()]
    # get eval results path
    try:
        evaluation_path = module_path.get_evaluation_path()
        eval_results_path = os.path.join(evaluation_path, model_name)
        # store bss-eval-metric data
        for file in os.listdir(eval_results_path):
            filename = os.path.join(eval_results_path, file)
            with open(filename, 'r', encoding='utf-8') as fd:
                eval_metrics = json.load(fd)
                for i in range(4):
                    boxplot_data[i].append(eval_metrics[metric_name][i])
    except FileNotFoundError:
        print("Wrong file path or wrong model file name")
    # output boxplot figure
    figure_name = metric_name + '_' + model_name
    figure_path = os.path.join(evaluation_path, 'bss_eval_plots', figure_name)
    figure_title = model_name.split('?')[0]
    boxplot_bss_eval_metrics(boxplot_data, figure_path, figure_title)


def stem_track_eval_metric_model_comparison(
        metric_name, stem_track_name, models):
    stem_tracks = {'vocals': 0, 'bass': 1, 'drums': 2, 'other': 3}
    stem_track_id = stem_tracks[stem_track_name]
    if isinstance(models, Iterable) and not isinstance(models, (str, bytes)):
        boxplot_data = list()
    try:
        evaluation_path = module_path.get_evaluation_path()
        for model in models:
            eval_data = list()
            eval_results_path = os.path.join(evaluation_path, model)
            # store bss-eval-metric data
            for file in os.listdir(eval_results_path):
                filename = os.path.join(eval_results_path, file)
                with open(filename, 'r', encoding='utf-8') as fd:
                    eval_metrics = json.load(fd)
                    eval_data.append(eval_metrics[metric_name][stem_track_id])
            boxplot_data.append(eval_data)
    except FileNotFoundError:
        print("Wrong file path or wrong model file name")

    # create a figure instance
    fig = plt.figure(1, figsize=(4, 6))
    # create an axes instance
    ax = fig.add_subplot()
    ax.title.set_text(stem_track_name)
    # create the boxplot
    bp = ax.boxplot(boxplot_data, patch_artist=True)
    # change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set(color='#7570b3', linewidth=2)
        # change fill color
        box.set(facecolor='#1b9e77')
    # change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)
    # change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)
    # change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)
    # change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)
    ax.set_xticklabels(['M1', 'M2', 'M3', 'M4'])
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # save the figure
    filename = metric_name + '_' + stem_track_name + '_models_comparison' + '.png'
    fig.savefig(filename, bbox_inches='tight')


if __name__ == '__main__':
    models = ['conv_denoising_unet?time=20200307_1423.h5',
              'conv_resblock_denoising_unet?time=20200308_1227.h5',
              'conv_encoder_denoising_decoder?time=20200308_1448.h5',
              'conv_denoising_stacked_unet.h5']
    # stem_track_eval_metric_model_comparison('nsdr', 'vocals', models)

    # output average value of the bss-eval metrics for a given model
    for model in models:
        print_metrics_mean_value(model)

    # generate boxplot for a given bss-eval metric and a given saved model
    # plot_model_eval_results('nsdr', 'conv_denoising_stacked_unet.h5')
