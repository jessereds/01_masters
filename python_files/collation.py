import torch
import seaborn as sn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import pathlib
import tarfile
import shutil
import time
import copy
import sys

def make_tarfile(output_filename, source_dir):
    # since = time.time()
    with tarfile.open(output_filename, "x") as tar:
        tar.add(source_dir, arcname='.')
    # time_elapsed = time.time() - since
    # print('Archive created in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

torch.save({ # To start from scratch. Comment out if carrying on from prior run(s).
    'samples_list': [], # Indices for the training and test data sets for each random split.
    'samples_counts': [], # How many E, L, S, Irr+Misc are in training and test for each random split.
    'samples_percentages': [], # Percentages of E, L, S, Irr+Misc in training and test for each random split.
    'pretrained_best_train_acc': [], # Peak training accuracies for each random split.
    'pretrained_best_test_acc': [], # Peak test accuracies for each random split.
    'pretrained_best_train_acc_epoch': [], # Epoch when peak training accuracy occurred.
    'pretrained_best_test_acc_epoch': [], # Epoch when peak testing accuracy occurred.
    'pretrained_best_train_loss': [], # 
    'pretrained_best_test_loss': [],
    'pretrained_best_train_loss_epoch': [],
    'pretrained_best_test_loss_epoch': [],
    'non_pretrained_best_train_acc': [],
    'non_pretrained_best_test_acc': [],
    'non_pretrained_best_train_acc_epoch': [],
    'non_pretrained_best_test_acc_epoch': [],
    'non_pretrained_best_train_loss': [],
    'non_pretrained_best_test_loss': [],
    'non_pretrained_best_train_loss_epoch': [],
    'non_pretrained_best_test_loss_epoch': [],
    'pretrained_cumulative_train_acc': [],
    'pretrained_cumulative_test_acc': [],
    'pretrained_cumulative_train_loss': [],
    'pretrained_cumulative_test_loss': [],
    'non_pretrained_cumulative_train_acc': [],
    'non_pretrained_cumulative_test_acc': [],
    'non_pretrained_cumulative_train_loss': [],
    'non_pretrained_cumulative_test_loss': [],
    'pretrained_elliptical_acc': [],
    'pretrained_lenticular_acc': [],
    'pretrained_spiral_acc': [],
    'pretrained_irrmisc_acc': [],
    'non_pretrained_elliptical_acc': [],
    'non_pretrained_lenticular_acc': [],
    'non_pretrained_spiral_acc': [],
    'non_pretrained_irrmisc_acc': [],
    },
	# '/project/rrg-lelliott/jsa378/misc_output/sample_perf_history.tar')
    f'/scratch/jsa378/misc_output/sample_perf_history_master.tar')
sample_perf_history = torch.load(f'/scratch/jsa378/misc_output/sample_perf_history_master.tar')

sample_perf_history_dictionary = {}
for i in range(10):
    sample_perf_history_dictionary[f'{i}'] = torch.load(f'/scratch/jsa378/misc_output/sample_perf_history_{i}.tar', map_location=torch.device('cpu'))

for i in range(10):
    for j in range(10):
        sample_perf_history['samples_list'].append(sample_perf_history_dictionary[f'{i}']['samples_list'][j])
       	sample_perf_history['samples_counts'].append(sample_perf_history_dictionary[f'{i}']['samples_counts'][j])
       	sample_perf_history['samples_percentages'].append(sample_perf_history_dictionary[f'{i}']['samples_percentages'][j])
       	sample_perf_history['pretrained_best_train_acc'].append(sample_perf_history_dictionary[f'{i}']['pretrained_best_train_acc'][j])
       	sample_perf_history['pretrained_best_test_acc'].append(sample_perf_history_dictionary[f'{i}']['pretrained_best_test_acc'][j])
       	sample_perf_history['pretrained_best_train_acc_epoch'].append(sample_perf_history_dictionary[f'{i}']['pretrained_best_train_acc_epoch'][j])
       	sample_perf_history['pretrained_best_test_acc_epoch'].append(sample_perf_history_dictionary[f'{i}']['pretrained_best_test_acc_epoch'][j])
       	sample_perf_history['pretrained_best_train_loss'].append(sample_perf_history_dictionary[f'{i}']['pretrained_best_train_loss'][j])
       	sample_perf_history['pretrained_best_test_loss'].append(sample_perf_history_dictionary[f'{i}']['pretrained_best_test_loss'][j])
       	sample_perf_history['pretrained_best_train_loss_epoch'].append(sample_perf_history_dictionary[f'{i}']['pretrained_best_train_loss_epoch'][j])
       	sample_perf_history['pretrained_best_test_loss_epoch'].append(sample_perf_history_dictionary[f'{i}']['pretrained_best_test_loss_epoch'][j])
       	sample_perf_history['non_pretrained_best_train_acc'].append(sample_perf_history_dictionary[f'{i}']['non_pretrained_best_train_acc'][j])
       	sample_perf_history['non_pretrained_best_test_acc'].append(sample_perf_history_dictionary[f'{i}']['non_pretrained_best_test_acc'][j])
       	sample_perf_history['non_pretrained_best_train_acc_epoch'].append(sample_perf_history_dictionary[f'{i}']['non_pretrained_best_train_acc_epoch'][j])
       	sample_perf_history['non_pretrained_best_test_acc_epoch'].append(sample_perf_history_dictionary[f'{i}']['non_pretrained_best_test_acc_epoch'][j])
       	sample_perf_history['non_pretrained_best_train_loss'].append(sample_perf_history_dictionary[f'{i}']['non_pretrained_best_train_loss'][j])
       	sample_perf_history['non_pretrained_best_test_loss'].append(sample_perf_history_dictionary[f'{i}']['non_pretrained_best_test_loss'][j])
       	sample_perf_history['non_pretrained_best_train_loss_epoch'].append(sample_perf_history_dictionary[f'{i}']['non_pretrained_best_train_loss_epoch'][j])
       	sample_perf_history['non_pretrained_best_test_loss_epoch'].append(sample_perf_history_dictionary[f'{i}']['non_pretrained_best_test_loss_epoch'][j])
       	sample_perf_history['pretrained_cumulative_train_acc'].append(sample_perf_history_dictionary[f'{i}']['pretrained_cumulative_train_acc'][j])
       	sample_perf_history['pretrained_cumulative_test_acc'].append(sample_perf_history_dictionary[f'{i}']['pretrained_cumulative_test_acc'][j])
       	sample_perf_history['pretrained_cumulative_train_loss'].append(sample_perf_history_dictionary[f'{i}']['pretrained_cumulative_train_loss'][j])
       	sample_perf_history['pretrained_cumulative_test_loss'].append(sample_perf_history_dictionary[f'{i}']['pretrained_cumulative_test_loss'][j])
       	sample_perf_history['non_pretrained_cumulative_train_acc'].append(sample_perf_history_dictionary[f'{i}']['non_pretrained_cumulative_train_acc'][j])
       	sample_perf_history['non_pretrained_cumulative_test_acc'].append(sample_perf_history_dictionary[f'{i}']['non_pretrained_cumulative_test_acc'][j])
       	sample_perf_history['non_pretrained_cumulative_train_loss'].append(sample_perf_history_dictionary[f'{i}']['non_pretrained_cumulative_train_loss'][j])
       	sample_perf_history['non_pretrained_cumulative_test_loss'].append(sample_perf_history_dictionary[f'{i}']['non_pretrained_cumulative_test_loss'][j])
       	sample_perf_history['pretrained_elliptical_acc'].append(sample_perf_history_dictionary[f'{i}']['pretrained_elliptical_acc'][j])
       	sample_perf_history['pretrained_lenticular_acc'].append(sample_perf_history_dictionary[f'{i}']['pretrained_lenticular_acc'][j])
       	sample_perf_history['pretrained_spiral_acc'].append(sample_perf_history_dictionary[f'{i}']['pretrained_spiral_acc'][j])
       	sample_perf_history['pretrained_irrmisc_acc'].append(sample_perf_history_dictionary[f'{i}']['pretrained_irrmisc_acc'][j])
       	sample_perf_history['non_pretrained_elliptical_acc'].append(sample_perf_history_dictionary[f'{i}']['non_pretrained_elliptical_acc'][j])
       	sample_perf_history['non_pretrained_lenticular_acc'].append(sample_perf_history_dictionary[f'{i}']['non_pretrained_lenticular_acc'][j])
       	sample_perf_history['non_pretrained_spiral_acc'].append(sample_perf_history_dictionary[f'{i}']['non_pretrained_spiral_acc'][j])
       	sample_perf_history['non_pretrained_irrmisc_acc'].append(sample_perf_history_dictionary[f'{i}']['non_pretrained_irrmisc_acc'][j])

# And so on for every other key in the dictionary.
# Once this is done, check that it works,
# and then the summary image generation code at the end of model_1_8_ddp.py
# should basically run without a hitch, more or less.

# Make summary figures.

# sample_perf_history = torch.load(f'/project/rrg-lelliott/jsa378/misc_output/sample_perf_history.tar')
# sample_perf_history = torch.load(f'/scratch/jsa378/misc_output/sample_perf_history.tar')

torch.save(sample_perf_history, '/scratch/jsa378/misc_output/sample_perf_history_master.tar')

# Make general histograms.

color1 = []
color2 = []

for i in range(len(sample_perf_history['pretrained_best_test_acc'])):
    color1.append('tab:green')
    color2.append('tab:red')

data = torch.tensor(sample_perf_history['pretrained_best_test_acc'])
data2 = torch.tensor(sample_perf_history['non_pretrained_best_test_acc'])
#fig = plt.figure()
#ax = fig.gca()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey=True, tight_layout=True)
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
ax1.hist(data, bins=10, color=color1)
ax2.hist(data2, bins=10, color=color2)
ax1.set_title('Peak Test Accuracy: \n Pretrained AlexNet')
ax1.set_xlabel('Accuracy')
ax1.set_ylabel('Count')
ax2.set_title('Peak Test Accuracy: \n Non-Pretrained AlexNet')
ax2.set_xlabel('Accuracy')
# ax2.set_ylabel('Count')
# fig.savefig('/project/rrg-lelliott/jsa378/misc_output/accuracy_histogram.png')
fig.savefig(f'/scratch/jsa378/misc_output/accuracy_histogram.png')

data = torch.tensor(sample_perf_history['pretrained_best_test_acc_epoch'])
data2 = torch.tensor(sample_perf_history['non_pretrained_best_test_acc_epoch'])
#fig = plt.figure()
#ax = fig.gca()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey=True, tight_layout=True)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
ax1.hist(data, bins=10, color=color1)
ax2.hist(data2, bins=10, color=color2)
ax1.set_title('Epoch of Peak \n Test Accuracy: \n Pretrained AlexNet')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Count')
ax2.set_title('Epoch of Peak \n Test Accuracy: \n Non-Pretrained AlexNet')
ax2.set_xlabel('Epoch')
# ax2.set_ylabel('Count')
# fig.savefig('/project/rrg-lelliott/jsa378/misc_output/accuracy_epoch_histogram.png')
fig.savefig(f'/scratch/jsa378/misc_output/accuracy_epoch_histogram.png')

data = torch.tensor(sample_perf_history['pretrained_best_test_loss'])
data2 = torch.tensor(sample_perf_history['non_pretrained_best_test_loss'])
#fig = plt.figure()
#ax = fig.gca()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey=True, tight_layout=True)
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
ax1.hist(data, bins=10, color=color1)
ax2.hist(data2, bins=10, color=color2)
ax1.set_title('Minimum Test Loss: \n Pretrained AlexNet')
ax1.set_xlabel('Loss')
ax1.set_ylabel('Count')
ax2.set_title('Minimum Test Loss: \n Non-Pretrained AlexNet')
ax2.set_xlabel('Loss')
# ax2.set_ylabel('Count')
# fig.savefig('/project/rrg-lelliott/jsa378/misc_output/loss_histogram.png')
fig.savefig(f'/scratch/jsa378/misc_output/loss_histogram.png')

data = torch.tensor(sample_perf_history['pretrained_best_test_loss_epoch'])
data2 = torch.tensor(sample_perf_history['non_pretrained_best_test_loss_epoch'])
#fig = plt.figure()
#ax = fig.gca()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey=True, tight_layout=True)
# plt.subplots_adjust(left=0.1, right=1)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
ax1.hist(data, bins=10, color=color1)
ax2.hist(data2, bins=10, color=color2)
ax1.set_title('Epoch of Minimum \n Test Loss: \n Pretrained AlexNet')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Count')
ax2.set_title('Epoch of Minimum \n Test Loss: \n Non-Pretrained AlexNet')
ax2.set_xlabel('Epoch')
# ax2.set_ylabel('Count')
# fig.savefig('/project/rrg-lelliott/jsa378/misc_output/loss_epoch_histogram.png')
fig.savefig(f'/scratch/jsa378/misc_output/loss_epoch_histogram.png')

# Make general accuracy plots (pretrained AlexNet).

fig = plt.figure()
ax = fig.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_title(f'Pretrained AlexNet Training Accuracy, All Runs')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
for i in range(len(sample_perf_history['pretrained_cumulative_train_acc'])):
    plt.plot(torch.tensor(sample_perf_history['pretrained_cumulative_train_acc'][i]), alpha=0.25)
# fig.savefig(f'/project/rrg-lelliott/jsa378/misc_output/pretrained_train_acc.png')
fig.savefig(f'/scratch/jsa378/misc_output/pretrained_train_acc.png')

fig = plt.figure()
ax = fig.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_title(f'Pretrained AlexNet Test Accuracy, All Runs')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
for i in range(len(sample_perf_history['pretrained_cumulative_test_acc'])):
    plt.plot(torch.tensor(sample_perf_history['pretrained_cumulative_test_acc'][i]), alpha=0.25)
# fig.savefig(f'/project/rrg-lelliott/jsa378/misc_output/pretrained_test_acc.png')
fig.savefig(f'/scratch/jsa378/misc_output/pretrained_test_acc.png')

fig = plt.figure()
ax = fig.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_title(f'Pretrained AlexNet Training Loss, All Runs')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
for i in range(len(sample_perf_history['pretrained_cumulative_train_loss'])):
    plt.plot(torch.tensor(sample_perf_history['pretrained_cumulative_train_loss'][i]), alpha=0.25)
# fig.savefig(f'/project/rrg-lelliott/jsa378/misc_output/pretrained_train_loss.png')
fig.savefig(f'/scratch/jsa378/misc_output/pretrained_train_loss.png')

fig = plt.figure()
ax = fig.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_title(f'Pretrained AlexNet Test Loss, All Runs')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
for i in range(len(sample_perf_history['pretrained_cumulative_test_loss'])):
    plt.plot(torch.tensor(sample_perf_history['pretrained_cumulative_test_loss'][i]), alpha=0.25)
# fig.savefig(f'/project/rrg-lelliott/jsa378/misc_output/pretrained_test_loss.png')
fig.savefig(f'/scratch/jsa378/misc_output/pretrained_test_loss.png')

# Make general accuracy plots (non-pretrained AlexNet).

fig = plt.figure()
ax = fig.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_title(f'Non-Pretrained AlexNet Training Accuracy, All Runs')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
for i in range(len(sample_perf_history['non_pretrained_cumulative_train_acc'])):
    plt.plot(torch.tensor(sample_perf_history['non_pretrained_cumulative_train_acc'][i]), alpha=0.25)
# fig.savefig(f'/project/rrg-lelliott/jsa378/misc_output/non_pretrained_train_acc.png')
fig.savefig(f'/scratch/jsa378/misc_output/non_pretrained_train_acc.png')

fig = plt.figure()
ax = fig.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_title(f'Non-Pretrained AlexNet Test Accuracy, All Runs')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
for i in range(len(sample_perf_history['non_pretrained_cumulative_test_acc'])):
    plt.plot(torch.tensor(sample_perf_history['non_pretrained_cumulative_test_acc'][i]), alpha=0.25)
# fig.savefig(f'/project/rrg-lelliott/jsa378/misc_output/non_pretrained_test_acc.png')
fig.savefig(f'/scratch/jsa378/misc_output/non_pretrained_test_acc.png')

fig = plt.figure()
ax = fig.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_title(f'Non-Pretrained AlexNet Training Loss, All Runs')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
for i in range(len(sample_perf_history['non_pretrained_cumulative_train_loss'])):
    plt.plot(torch.tensor(sample_perf_history['non_pretrained_cumulative_train_loss'][i]), alpha=0.25)
# fig.savefig(f'/project/rrg-lelliott/jsa378/misc_output/non_pretrained_train_loss.png')
fig.savefig(f'/scratch/jsa378/misc_output/non_pretrained_train_loss.png')

fig = plt.figure()
ax = fig.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_title(f'Non-Pretrained AlexNet Test Loss, All Runs')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
for i in range(len(sample_perf_history['non_pretrained_cumulative_test_loss'])):
    plt.plot(torch.tensor(sample_perf_history['non_pretrained_cumulative_test_loss'][i]), alpha=0.25)
# fig.savefig(f'/project/rrg-lelliott/jsa378/misc_output/non_pretrained_test_loss.png')
fig.savefig(f'/scratch/jsa378/misc_output/non_pretrained_test_loss.png')

# Make histograms of per-class accuracies.

data = torch.tensor(sample_perf_history['pretrained_elliptical_acc'])
data2 = torch.tensor(sample_perf_history['pretrained_lenticular_acc'])
data3 = torch.tensor(sample_perf_history['pretrained_spiral_acc'])
data4 = torch.tensor(sample_perf_history['pretrained_irrmisc_acc'])
#fig = plt.figure()
#ax = fig.gca()
fig, axs = plt.subplots(2, 2, figsize=(12.8, 9.6), sharey=True, tight_layout=True)
axs[0, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
axs[0, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
axs[1, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
axs[1, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
axs[0, 0].hist(data, bins=10)
axs[0, 1].hist(data2, bins=10)
axs[1, 0].hist(data3, bins=10)
axs[1, 1].hist(data4, bins=10)
axs[0, 0].set_title('Elliptical Classification Accuracy: \n Pretrained AlexNet')
axs[0, 0].set_xlabel('Accuracy')
axs[0, 0].set_ylabel('Count')
axs[0, 1].set_title('Lenticular Classification Accuracy: \n Pretrained AlexNet')
axs[0, 1].set_xlabel('Accuracy')
axs[0, 1].set_ylabel('Count')
axs[1, 0].set_title('Spiral Classification Accuracy: \n Pretrained AlexNet')
axs[1, 0].set_xlabel('Accuracy')
axs[1, 0].set_ylabel('Count')
axs[1, 1].set_title('Irr+Misc Classification Accuracy: \n Pretrained AlexNet')
axs[1, 1].set_xlabel('Accuracy')
axs[1, 1].set_ylabel('Count')
# fig.savefig('/project/rrg-lelliott/jsa378/misc_output/pretrained_class_accuracy_histograms.png')
fig.savefig(f'/scratch/jsa378/misc_output/pretrained_class_accuracy_histograms.png')

data = torch.tensor(sample_perf_history['non_pretrained_elliptical_acc'])
data2 = torch.tensor(sample_perf_history['non_pretrained_lenticular_acc'])
data3 = torch.tensor(sample_perf_history['non_pretrained_spiral_acc'])
data4 = torch.tensor(sample_perf_history['non_pretrained_irrmisc_acc'])
#fig = plt.figure()
#ax = fig.gca()
fig, axs = plt.subplots(2, 2, figsize=(12.8, 9.6), sharey=True, tight_layout=True)
axs[0, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
axs[0, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
axs[1, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
axs[1, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
axs[0, 0].hist(data, bins=10)
axs[0, 1].hist(data2, bins=10)
axs[1, 0].hist(data3, bins=10)
axs[1, 1].hist(data4, bins=10)
axs[0, 0].set_title('Elliptical Classification Accuracy: \n Non-Pretrained AlexNet')
axs[0, 0].set_xlabel('Accuracy')
axs[0, 0].set_ylabel('Count')
axs[0, 1].set_title('Lenticular Classification Accuracy: \n Non-Pretrained AlexNet')
axs[0, 1].set_xlabel('Accuracy')
axs[0, 1].set_ylabel('Count')
axs[1, 0].set_title('Spiral Classification Accuracy: \n Non-Pretrained AlexNet')
axs[1, 0].set_xlabel('Accuracy')
axs[1, 0].set_ylabel('Count')
axs[1, 1].set_title('Irr+Misc Classification Accuracy: \n Non-Pretrained AlexNet')
axs[1, 1].set_xlabel('Accuracy')
axs[1, 1].set_ylabel('Count')
# fig.savefig('/project/rrg-lelliott/jsa378/misc_output/non_pretrained_class_accuracy_histograms.png')
fig.savefig(f'/scratch/jsa378/misc_output/non_pretrained_class_accuracy_histograms.png')

# Make side-by-side plots of pretrained and non-pretrained AlexNet over the same run.

for i in range(len(sample_perf_history['pretrained_cumulative_train_acc'])):
    # fig = plt.figure()
    # ax = fig.gca()
    fig, axs = plt.subplots(2, 2, figsize=(12.8, 9.6),sharey=False, tight_layout=True)
    axs[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0, 0].set_title(f'Pretrained AlexNet, run {i}')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Accuracy')
    data = torch.tensor(sample_perf_history['pretrained_cumulative_test_acc'][i])
    data2 = torch.tensor(sample_perf_history['pretrained_cumulative_train_acc'][i])
    axs[0, 0].plot(data, label='Test')
    axs[0, 0].plot(data2, label='Train')
    axs[0, 0].legend()
    axs[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0, 1].set_title(f'Non-Pretrained AlexNet, run {i}')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    data3 = torch.tensor(sample_perf_history['non_pretrained_cumulative_test_acc'][i])
    data4 = torch.tensor(sample_perf_history['non_pretrained_cumulative_train_acc'][i])
    axs[0, 1].plot(data3, label='Test')
    axs[0, 1].plot(data4, label='Train')
    axs[0, 1].legend()
    axs[1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1, 0].set_title(f'Pretrained AlexNet, run {i}')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Loss')
    data5 = torch.tensor(sample_perf_history['pretrained_cumulative_test_loss'][i])
    data6 = torch.tensor(sample_perf_history['pretrained_cumulative_train_loss'][i])
    axs[1, 0].plot(data5, label='Test')
    axs[1, 0].plot(data6, label='Train')
    axs[1, 0].legend()
    axs[1, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1, 1].set_title(f'Non-Pretrained AlexNet, run {i}')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Loss')
    data7 = torch.tensor(sample_perf_history['non_pretrained_cumulative_test_loss'][i])
    data8 = torch.tensor(sample_perf_history['non_pretrained_cumulative_train_loss'][i])
    axs[1, 1].plot(data7, label='Test')
    axs[1, 1].plot(data8, label='Train')
    axs[1, 1].legend()
    # fig.savefig(f'/project/rrg-lelliott/jsa378/misc_output/comparison_run_{i}.png')
    fig.savefig(f'/scratch/jsa378/misc_output/comparison_run_{i}.png')

make_tarfile('/project/rrg-lelliott/jsa378/misc_output.tar', '/scratch/jsa378/misc_output')
shutil.rmtree(f'/scratch/jsa378/misc_output')

###

# BONUS: Below is some code I ran after this collation job. This was done on Stenning's advice to make a histogram of the differences between peak test accuracies.

# First, some preparation in Bash:

# mkdir $SCRATCH/misc_output
# tar -xf misc_output.tar -C $SCRATCH/misc_output
# cd $SCRATCH/misc_output
# source ~/msc/bin/activate

# Now, the Python code. (To run the code, paste it into a temp.py file on Compute Canada and then type `python temp.py` in Bash to run it.)

# import torch
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator

# sample_perf_history = torch.load(f'/scratch/jsa378/misc_output/sample_perf_history_master.tar')

# pretrained = np.array(sample_perf_history['pretrained_best_test_acc'])
# print('The average peak test accuracy for the pretrained AlexNet is', np.mean(pretrained))
# print()
# non_pretrained = np.array(sample_perf_history['non_pretrained_best_test_acc'])
# print('The average peak test accuracy for the non-pretrained AlexNet is', np.mean(non_pretrained))
# print()
# neg_non_pretrained = (-1)*np.array(sample_perf_history['non_pretrained_best_test_acc'])
# difference = np.add(pretrained, neg_non_pretrained)
# average = np.mean(difference)
# print('The average gain from using the pretrained AlexNet is', average, '.')
# print()

# pretrained_epoch = np.array(sample_perf_history['pretrained_best_test_acc_epoch'])
# # non_pretrained_epoch = np.array(sample_perf_history['non_pretrained_best_test_acc_epoch'][:200])
# non_pretrained_epoch = []
# for i in range(len(sample_perf_history['non_pretrained_cumulative_test_acc'])):
#     # non_pretrained_epoch.append(sample_perf_history['non_pretrained_cumulative_test_acc'][i][:200])
#     non_pretrained_epoch.append(sample_perf_history['non_pretrained_cumulative_test_acc'][i][:200].index(max(sample_perf_history['non_pretrained_cumulative_test_acc'][i][:200])))
# non_pretrained_epoch = np.array(non_pretrained_epoch)
# non_pretrained_epoch.size
# neg_non_pretrained_epoch = (-1)*non_pretrained_epoch
# difference_epoch = np.add(pretrained_epoch, neg_non_pretrained_epoch)
# average_epoch = np.mean(difference_epoch)
# print('If you had only trained the non-pretrained AlexNet for 200 epochs each run, then it would reach peak test accuracy', (-1)*average_epoch, 'epochs later than the pretrained AlexNet, on average.')
# print()
# print('For reference, the pretrained AlexNet peak test accuracy is achieved in epoch', np.mean(np.array(sample_perf_history['pretrained_best_test_acc_epoch'])), 'on average', ', and the non-pretrained AlexNet (restricted to 200 epochs of training) achieves peak test accuracy in epoch', np.mean(non_pretrained_epoch), 'on average.')

# color1 = []
# color2 = []

# for i in range(difference.size):
#     color1.append('tab:green')
#     color2.append('tab:red')

# difference = torch.tensor(difference)
# difference_epoch = torch.tensor(difference_epoch)

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey=False, tight_layout=True)
# # plt.subplots_adjust(left=0.1, right=1)
# # ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax1.hist(difference, bins=10, color=color1)
# ax2.hist(difference_epoch, bins=10, color=color2)
# ax1.set_title('Pretrained Peak Test Accuracy Minus \n Non-Pretrained Peak Test Accuracy')
# ax1.set_xlabel('Difference in Accuracy')
# ax1.set_ylabel('Count')
# ax2.set_title('Epoch of Pretrained Peak Test Accuracy Minus \n Epoch of Non-Pretrained Peak Test Accuracy \n (Non-Pretrained Restricted to First 200 Epochs)')
# ax2.set_xlabel('Difference in Epoch')
# ax2.set_ylabel('Count')
# # fig.savefig('/project/rrg-lelliott/jsa378/misc_output/loss_epoch_histogram.png')
# fig.savefig(f'/scratch/jsa378/misc_output/difference_histogram.png')

# non_pretrained_acc_restricted = []
# for i in range(len(sample_perf_history['non_pretrained_cumulative_test_acc'])):
#     non_pretrained_acc_restricted.append(max(sample_perf_history['non_pretrained_cumulative_test_acc'][i][:200]))
# non_pretrained_acc_restricted = np.array(non_pretrained_acc_restricted)
# print()
# print('The average peak test accuracy for the non-pretrained AlexNet, if it had been restricted to 200 epochs of training, is', np.mean(non_pretrained_acc_restricted), '.')
# neg_non_pretrained_acc_restricted = (-1)*non_pretrained_acc_restricted
# difference2 = np.add(pretrained, neg_non_pretrained_acc_restricted)
# average2 = np.mean(difference2)
# print()
# print('The average gain from using the pretrained AlexNet is', average2, ', if we had restricted the non-pretrained AlexNet to train for only 200 epochs.')

# difference2 = torch.tensor(difference2)

# fig, (ax1) = plt.subplots(1, 1, figsize=(6.4, 4.8), sharey=False, tight_layout=True)
# ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax1.hist(difference2, bins=10, color=color1)
# ax1.set_title('Pretrained Peak Test Accuracy Minus \n Non-Pretrained Peak Test Accuracy \n (Non-Pretrained Restricted to First 200 Epochs)')
# ax1.set_xlabel('Difference in Accuracy')
# ax1.set_ylabel('Count')
# # fig.savefig('/project/rrg-lelliott/jsa378/misc_output/loss_epoch_histogram.png')
# fig.savefig(f'/scratch/jsa378/misc_output/difference_histogram_2.png')

# BONUS BONUS: The code above, when run on Compute Canada, makes some weird-looking histograms. I modified the code slightly and ran it locally (in a Jupyter Notebook) on the MacBook Air, and the histograms now look more conventional. I also added a couple more histograms at the bottom.

# Notice that below, I didn't use the lists of colors.

# import torch
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator

# sample_perf_history = torch.load('/Users/jesse/Downloads/meeting_temp/misc_output/sample_perf_history_master.tar')

# # color1 = []
# # color2 = []

# # for i in range(len(sample_perf_history['pretrained_best_test_acc'])):
# #     color1.append('tab:green')
# #     color2.append('tab:red')

# data = torch.tensor(sample_perf_history['pretrained_best_test_acc'])
# data2 = torch.tensor(sample_perf_history['non_pretrained_best_test_acc'])
# #fig = plt.figure()
# #ax = fig.gca()
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey=True, tight_layout=True)
# ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax1.hist(data, bins=10, color='tab:green')# color=color1)
# ax2.hist(data2, bins=10, color='tab:red')# color=color2)
# ax1.set_title('Peak Test Accuracy: \n Pretrained AlexNet')
# ax1.set_xlabel('Accuracy')
# ax1.set_ylabel('Count')
# ax2.set_title('Peak Test Accuracy: \n Non-Pretrained AlexNet')
# ax2.set_xlabel('Accuracy')
# # ax2.set_ylabel('Count')
# # fig.savefig('/project/rrg-lelliott/jsa378/misc_output/accuracy_histogram.png')
# fig.savefig('/Users/jesse/Downloads/meeting_temp/misc_output/accuracy_histogram_local.png')

# data = torch.tensor(sample_perf_history['pretrained_best_test_acc_epoch'])
# data2 = torch.tensor(sample_perf_history['non_pretrained_best_test_acc_epoch'])
# #fig = plt.figure()
# #ax = fig.gca()
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey=True, tight_layout=True)
# ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax1.hist(data, bins=10, color='tab:green')# color=color1)
# ax2.hist(data2, bins=10, color='tab:red')# color=color2)
# ax1.set_title('Epoch of Peak \n Test Accuracy: \n Pretrained AlexNet')
# ax1.set_xlabel('Epoch')
# ax1.set_ylabel('Count')
# ax2.set_title('Epoch of Peak \n Test Accuracy: \n Non-Pretrained AlexNet')
# ax2.set_xlabel('Epoch')
# # ax2.set_ylabel('Count')
# # fig.savefig('/project/rrg-lelliott/jsa378/misc_output/accuracy_epoch_histogram.png')
# fig.savefig('/Users/jesse/Downloads/meeting_temp/misc_output/accuracy_epoch_histogram_local.png')

# data = torch.tensor(sample_perf_history['pretrained_best_test_loss'])
# data2 = torch.tensor(sample_perf_history['non_pretrained_best_test_loss'])
# #fig = plt.figure()
# #ax = fig.gca()
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey=True, tight_layout=True)
# ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax1.hist(data, bins=10, color='tab:green')# color=color2)
# ax2.hist(data2, bins=10, color='tab:red')# color=color2)
# ax1.set_title('Minimum Test Loss: \n Pretrained AlexNet')
# ax1.set_xlabel('Loss')
# ax1.set_ylabel('Count')
# ax2.set_title('Minimum Test Loss: \n Non-Pretrained AlexNet')
# ax2.set_xlabel('Loss')
# # ax2.set_ylabel('Count')
# # fig.savefig('/project/rrg-lelliott/jsa378/misc_output/loss_histogram.png')
# fig.savefig('/Users/jesse/Downloads/meeting_temp/misc_output/loss_histogram_local.png')

# data = torch.tensor(sample_perf_history['pretrained_best_test_loss_epoch'])
# data2 = torch.tensor(sample_perf_history['non_pretrained_best_test_loss_epoch'])
# #fig = plt.figure()
# #ax = fig.gca()
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey=True, tight_layout=True)
# # plt.subplots_adjust(left=0.1, right=1)
# ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax1.hist(data, bins=10, color='tab:green')# color=color2)
# ax2.hist(data2, bins=10, color='tab:red')# color=color2)
# ax1.set_title('Epoch of Minimum \n Test Loss: \n Pretrained AlexNet')
# ax1.set_xlabel('Epoch')
# ax1.set_ylabel('Count')
# ax2.set_title('Epoch of Minimum \n Test Loss: \n Non-Pretrained AlexNet')
# ax2.set_xlabel('Epoch')
# # ax2.set_ylabel('Count')
# # fig.savefig('/project/rrg-lelliott/jsa378/misc_output/loss_epoch_histogram.png')
# fig.savefig('/Users/jesse/Downloads/meeting_temp/misc_output/loss_epoch_histogram_local.png')

# # Make histograms of per-class accuracies.

# data = torch.tensor(sample_perf_history['pretrained_elliptical_acc'])
# data2 = torch.tensor(sample_perf_history['pretrained_lenticular_acc'])
# data3 = torch.tensor(sample_perf_history['pretrained_spiral_acc'])
# data4 = torch.tensor(sample_perf_history['pretrained_irrmisc_acc'])
# #fig = plt.figure()
# #ax = fig.gca()
# fig, axs = plt.subplots(2, 2, figsize=(12.8, 9.6), sharey=True, tight_layout=True)
# axs[0, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
# axs[0, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
# axs[1, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
# axs[1, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
# axs[0, 0].hist(data, bins=10)
# axs[0, 1].hist(data2, bins=10)
# axs[1, 0].hist(data3, bins=10)
# axs[1, 1].hist(data4, bins=10)
# axs[0, 0].set_title('Elliptical Classification Accuracy: \n Pretrained AlexNet')
# axs[0, 0].set_xlabel('Accuracy')
# axs[0, 0].set_ylabel('Count')
# axs[0, 1].set_title('Lenticular Classification Accuracy: \n Pretrained AlexNet')
# axs[0, 1].set_xlabel('Accuracy')
# axs[0, 1].set_ylabel('Count')
# axs[1, 0].set_title('Spiral Classification Accuracy: \n Pretrained AlexNet')
# axs[1, 0].set_xlabel('Accuracy')
# axs[1, 0].set_ylabel('Count')
# axs[1, 1].set_title('Irr+Misc Classification Accuracy: \n Pretrained AlexNet')
# axs[1, 1].set_xlabel('Accuracy')
# axs[1, 1].set_ylabel('Count')
# # fig.savefig('/project/rrg-lelliott/jsa378/misc_output/pretrained_class_accuracy_histograms.png')
# fig.savefig('/Users/jesse/Downloads/meeting_temp/misc_output/pretrained_class_accuracy_histograms_local.png')

# data = torch.tensor(sample_perf_history['non_pretrained_elliptical_acc'])
# data2 = torch.tensor(sample_perf_history['non_pretrained_lenticular_acc'])
# data3 = torch.tensor(sample_perf_history['non_pretrained_spiral_acc'])
# data4 = torch.tensor(sample_perf_history['non_pretrained_irrmisc_acc'])
# #fig = plt.figure()
# #ax = fig.gca()
# fig, axs = plt.subplots(2, 2, figsize=(12.8, 9.6), sharey=True, tight_layout=True)
# axs[0, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
# axs[0, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
# axs[1, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
# axs[1, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
# axs[0, 0].hist(data, bins=10)
# axs[0, 1].hist(data2, bins=10)
# axs[1, 0].hist(data3, bins=10)
# axs[1, 1].hist(data4, bins=10)
# axs[0, 0].set_title('Elliptical Classification Accuracy: \n Non-Pretrained AlexNet')
# axs[0, 0].set_xlabel('Accuracy')
# axs[0, 0].set_ylabel('Count')
# axs[0, 1].set_title('Lenticular Classification Accuracy: \n Non-Pretrained AlexNet')
# axs[0, 1].set_xlabel('Accuracy')
# axs[0, 1].set_ylabel('Count')
# axs[1, 0].set_title('Spiral Classification Accuracy: \n Non-Pretrained AlexNet')
# axs[1, 0].set_xlabel('Accuracy')
# axs[1, 0].set_ylabel('Count')
# axs[1, 1].set_title('Irr+Misc Classification Accuracy: \n Non-Pretrained AlexNet')
# axs[1, 1].set_xlabel('Accuracy')
# axs[1, 1].set_ylabel('Count')
# # fig.savefig('/project/rrg-lelliott/jsa378/misc_output/non_pretrained_class_accuracy_histograms.png')
# fig.savefig('/Users/jesse/Downloads/meeting_temp/misc_output/non_pretrained_class_accuracy_histograms_local.png')

# pretrained = np.array(sample_perf_history['pretrained_best_test_acc'])
# print('The average peak test accuracy for the pretrained AlexNet is', np.mean(pretrained))
# print()
# non_pretrained = np.array(sample_perf_history['non_pretrained_best_test_acc'])
# print('The average peak test accuracy for the non-pretrained AlexNet is', np.mean(non_pretrained))
# print()
# neg_non_pretrained = (-1)*np.array(sample_perf_history['non_pretrained_best_test_acc'])
# difference = np.add(pretrained, neg_non_pretrained)
# average = np.mean(difference)
# print('The average gain from using the pretrained AlexNet is', average, '.')
# print()

# pretrained_epoch = np.array(sample_perf_history['pretrained_best_test_acc_epoch'])
# # non_pretrained_epoch = np.array(sample_perf_history['non_pretrained_best_test_acc_epoch'][:200])
# non_pretrained_epoch = []
# for i in range(len(sample_perf_history['non_pretrained_cumulative_test_acc'])):
#     # non_pretrained_epoch.append(sample_perf_history['non_pretrained_cumulative_test_acc'][i][:200])
#     non_pretrained_epoch.append(sample_perf_history['non_pretrained_cumulative_test_acc'][i][:200].index(max(sample_perf_history['non_pretrained_cumulative_test_acc'][i][:200])))
# non_pretrained_epoch = np.array(non_pretrained_epoch)
# non_pretrained_epoch.size
# neg_non_pretrained_epoch = (-1)*non_pretrained_epoch
# difference_epoch = np.add(pretrained_epoch, neg_non_pretrained_epoch)
# average_epoch = np.mean(difference_epoch)
# print('If you had only trained the non-pretrained AlexNet for 200 epochs each run, then it would reach peak test accuracy', (-1)*average_epoch, 'epochs later than the pretrained AlexNet, on average.')
# print()
# print('For reference, the pretrained AlexNet peak test accuracy is achieved in epoch', np.mean(np.array(sample_perf_history['pretrained_best_test_acc_epoch'])), 'on average', ', and the non-pretrained AlexNet (restricted to 200 epochs of training) achieves peak test accuracy in epoch', np.mean(non_pretrained_epoch), 'on average.')

# # color1 = []
# # color2 = []

# # for i in range(difference.size):
# #     color1.append('tab:green')
# #     color2.append('tab:red')

# difference = torch.tensor(difference)
# difference_epoch = torch.tensor(difference_epoch)

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey=False, tight_layout=True)
# # plt.subplots_adjust(left=0.1, right=1)
# # ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax1.hist(difference, bins=10, color='tab:green')# color=color2)
# ax2.hist(difference_epoch, bins=10, color='tab:red')# color=color2)
# ax1.set_title('Pretrained Peak Test Accuracy Minus \n Non-Pretrained Peak Test Accuracy')
# ax1.set_xlabel('Difference in Accuracy')
# ax1.set_ylabel('Count')
# ax2.set_title('Epoch of Pretrained Peak Test Accuracy Minus \n Epoch of Non-Pretrained Peak Test Accuracy \n (Non-Pretrained Restricted to First 200 Epochs)')
# ax2.set_xlabel('Difference in Epoch')
# ax2.set_ylabel('Count')
# # fig.savefig('/project/rrg-lelliott/jsa378/misc_output/loss_epoch_histogram.png')
# fig.savefig('/Users/jesse/Downloads/meeting_temp/misc_output/difference_histogram_local.png')

# non_pretrained_acc_restricted = []
# for i in range(len(sample_perf_history['non_pretrained_cumulative_test_acc'])):
#     non_pretrained_acc_restricted.append(max(sample_perf_history['non_pretrained_cumulative_test_acc'][i][:200]))
# non_pretrained_acc_restricted = np.array(non_pretrained_acc_restricted)
# print()
# print('The average peak test accuracy for the non-pretrained AlexNet, if it had been restricted to 200 epochs of training, is', np.mean(non_pretrained_acc_restricted), '.')
# neg_non_pretrained_acc_restricted = (-1)*non_pretrained_acc_restricted
# difference2 = np.add(pretrained, neg_non_pretrained_acc_restricted)
# average2 = np.mean(difference2)
# print()
# print('The average gain from using the pretrained AlexNet is', average2, ', if we had restricted the non-pretrained AlexNet to train for only 200 epochs.')

# difference2 = torch.tensor(difference2)

# fig, (ax1) = plt.subplots(1, 1, figsize=(6.4, 4.8), sharey=False, tight_layout=True)
# ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax1.hist(difference2, bins=10, color='tab:green')# color=color2)
# ax1.set_title('Pretrained Peak Test Accuracy Minus \n Non-Pretrained Peak Test Accuracy \n (Non-Pretrained Restricted to First 200 Epochs)')
# ax1.set_xlabel('Difference in Accuracy')
# ax1.set_ylabel('Count')
# # fig.savefig('/project/rrg-lelliott/jsa378/misc_output/loss_epoch_histogram.png')
# fig.savefig('/Users/jesse/Downloads/meeting_temp/misc_output/difference_histogram_2_local.png')

# n = 20
# pretrained_acc_restricted = []
# for i in range(len(sample_perf_history['pretrained_cumulative_test_acc'])):
#     pretrained_acc_restricted.append(max(sample_perf_history['pretrained_cumulative_test_acc'][i][:n]))
# pretrained_acc_restricted = np.array(pretrained_acc_restricted)
# print()
# print('The average peak test accuracy for the pretrained AlexNet, if it had been restricted to', n, 'epochs of training, is', np.mean(pretrained_acc_restricted), '.')
# neg_pretrained_acc_restricted = (-1)*pretrained_acc_restricted
# difference3 = np.add(pretrained, neg_pretrained_acc_restricted)
# average3 = np.mean(difference3)
# print()
# print('The average gain from letting the pretrained AlexNet train for 200 epochs instead of', n, 'epochs is', average3, '.')

# fig, (ax1) = plt.subplots(1, 1, figsize=(6.4, 4.8), sharey=False, tight_layout=True)
# ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax1.hist(difference3, bins=10, color='tab:red')# color=color2)
# ax1.set_title(f'Pretrained Peak Test Accuracy Minus \n Pretrained Peak Test Accuracy \n (Restricted to {n} Epochs)')
# ax1.set_xlabel('Difference in Accuracy')
# ax1.set_ylabel('Count')
# # fig.savefig('/project/rrg-lelliott/jsa378/misc_output/loss_epoch_histogram.png')
# fig.savefig('/Users/jesse/Downloads/meeting_temp/misc_output/difference_histogram_3_local.png')

# n = 50
# pretrained_acc_restricted = []
# for i in range(len(sample_perf_history['pretrained_cumulative_test_acc'])):
#     pretrained_acc_restricted.append(max(sample_perf_history['pretrained_cumulative_test_acc'][i][:n]))
# pretrained_acc_restricted = np.array(pretrained_acc_restricted)
# print()
# print('The average peak test accuracy for the pretrained AlexNet, if it had been restricted to', n, 'epochs of training, is', np.mean(pretrained_acc_restricted), '.')
# neg_pretrained_acc_restricted = (-1)*pretrained_acc_restricted
# difference4 = np.add(pretrained, neg_pretrained_acc_restricted)
# average4 = np.mean(difference4)
# print()
# print('The average gain from letting the pretrained AlexNet train for 200 epochs instead of', n, 'epochs is', average4, '.')

# fig, (ax1) = plt.subplots(1, 1, figsize=(6.4, 4.8), sharey=False, tight_layout=True)
# ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax1.hist(difference4, bins=10, color='tab:red')# color=color2)
# ax1.set_title(f'Pretrained Peak Test Accuracy Minus \n Pretrained Peak Test Accuracy \n (Restricted to {n} Epochs)')
# ax1.set_xlabel('Difference in Accuracy')
# ax1.set_ylabel('Count')
# # fig.savefig('/project/rrg-lelliott/jsa378/misc_output/loss_epoch_histogram.png')
# fig.savefig('/Users/jesse/Downloads/meeting_temp/misc_output/difference_histogram_4_local.png')