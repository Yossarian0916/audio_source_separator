import os
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils import module_path


# agg backend is used to create plot as a .png file
mpl.use('agg')


def plot_eval_metrics(eval_results, save_name):
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


if __name__ == '__main__':
    evaluation_path = module_path.get_evaluation_path()
    # model_name = 'conv_denoising_unet?time=20200223_0347.h5'
    # model_name = 'conv_encoder_denoising_decoder?time=20200227_0838_l2_weight_regularization.h5'
    model_name = 'conv_res56_denoising_unet?time=20200227_0646_l2_reg.h5'
    eval_results_file_dir = os.path.join(evaluation_path, model_name)
    total_nsdr = [0] * 4
    nsdr_to_plot = [list(), list(), list(), list()]
    music_pieces = len(os.listdir(eval_results_file_dir))
    for file in os.listdir(eval_results_file_dir):
        filename = os.path.join(eval_results_file_dir, file)
        with open(filename, 'r', encoding='utf-8') as fd:
            eval_metrics = json.load(fd)
            for i in range(4):
                total_nsdr[i] += eval_metrics['nsdr'][i]
                nsdr_to_plot[i].append(eval_metrics['nsdr'][i])
    average_nsdr = [nsdr / music_pieces for nsdr in total_nsdr]
    print(average_nsdr)
    plot_eval_metrics(nsdr_to_plot, model_name)
