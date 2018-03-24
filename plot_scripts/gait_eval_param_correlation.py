import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

# Set the home directory
home_dir = os.path.expanduser('~')

# Set the directory for saving plots
plot_dir = os.path.join(home_dir, 'computing/repositories/MScThesis_SayantanAuddy_2017_NICOOscillatorWalking/plots')

# Directory for saving plot data files
plot_data_dir = os.path.join(home_dir, 'computing/repositories/MScThesis_SayantanAuddy_2017_NICOOscillatorWalking/plot_data')

# Set the directory where logs are stored
log_dir = os.path.join(home_dir, 'computing/repositories/MScThesis_SayantanAuddy_2017_NICOOscillatorWalking/logs/reinforcement_learning_logs')

global fignum
fignum=1

def create_corrmat_from_csv(datafile_fitness, datafile_x_dist,
                            datafile_y_dev, datafile_foot,
                            datafile_up, datafile_alpha,
                            datafile_beta, datafile_gamma,
                            plot_title=None, figsize=(10,10)):

    global fignum

    #plot_file = os.path.join(plot_dir,datafile.replace('.csv', '.svg'))
    colnames = ['OL1', 'OL2', 'OL3', 'AF1', 'AF2', 'AF3', 'PR1', 'PR2', 'PR3']

    df_fitness = pd.read_csv(os.path.join(plot_data_dir, datafile_fitness), delimiter=',', names=colnames, header=0)
    df_x_dist = pd.read_csv(os.path.join(plot_data_dir, datafile_x_dist), delimiter=',', names=colnames, header=0)
    df_y_dev = pd.read_csv(os.path.join(plot_data_dir, datafile_y_dev), delimiter=',', names=colnames, header=0)
    df_foot = pd.read_csv(os.path.join(plot_data_dir, datafile_foot), delimiter=',', names=colnames, header=0)
    df_up = pd.read_csv(os.path.join(plot_data_dir, datafile_up), delimiter=',', names=colnames, header=0)
    df_alpha = pd.read_csv(os.path.join(plot_data_dir, datafile_alpha), delimiter=',', names=colnames, header=0)
    df_beta = pd.read_csv(os.path.join(plot_data_dir, datafile_beta), delimiter=',', names=colnames, header=0)
    df_gamma = pd.read_csv(os.path.join(plot_data_dir, datafile_gamma), delimiter=',', names=colnames, header=0)

    # Concatenate columns 
    df_ol1 = pd.concat([df_fitness['OL1'],df_x_dist['OL1'],df_y_dev['OL1'],df_foot['OL1'],df_up['OL1'],df_alpha['OL1'],df_beta['OL1'],df_gamma['OL1'],], axis=1)
    df_ol2 = pd.concat([df_fitness['OL2'],df_x_dist['OL2'],df_y_dev['OL2'],df_foot['OL2'],df_up['OL2'],df_alpha['OL2'],df_beta['OL2'],df_gamma['OL2'],], axis=1)
    df_ol3 = pd.concat([df_fitness['OL3'],df_x_dist['OL3'],df_y_dev['OL3'],df_foot['OL3'],df_up['OL3'],df_alpha['OL3'],df_beta['OL3'],df_gamma['OL3'],], axis=1)

    df_af1 = pd.concat([df_fitness['AF1'], df_x_dist['AF1'], df_y_dev['AF1'], df_foot['AF1'], df_up['AF1'], df_alpha['AF1'],df_beta['AF1'], df_gamma['AF1'], ], axis=1)
    df_af2 = pd.concat([df_fitness['AF2'], df_x_dist['AF2'], df_y_dev['AF2'], df_foot['AF2'], df_up['AF2'], df_alpha['AF2'],df_beta['AF2'], df_gamma['AF2'], ], axis=1)
    df_af3 = pd.concat([df_fitness['AF3'], df_x_dist['AF3'], df_y_dev['AF3'], df_foot['AF3'], df_up['AF3'], df_alpha['AF3'],df_beta['AF3'], df_gamma['AF3'], ], axis=1)

    df_pr1 = pd.concat([df_fitness['PR1'], df_x_dist['PR1'], df_y_dev['PR1'], df_foot['PR1'], df_up['PR1'], df_alpha['PR1'],df_beta['PR1'], df_gamma['PR1'], ], axis=1)
    df_pr2 = pd.concat([df_fitness['PR2'], df_x_dist['PR2'], df_y_dev['PR2'], df_foot['PR2'], df_up['PR2'], df_alpha['PR2'],df_beta['PR2'], df_gamma['PR2'], ], axis=1)
    df_pr3 = pd.concat([df_fitness['PR3'], df_x_dist['PR3'], df_y_dev['PR3'], df_foot['PR3'], df_up['PR3'], df_alpha['PR3'],df_beta['PR3'], df_gamma['PR3'], ], axis=1)

    labels = [r'$f$', r'$distance_x$', r'$dev^{abs}_{y}$', r'$stride^{avg}$', r'$t_{up}$', r'$torso_\alpha^{var}$', r'$torso_\beta^{var}$', r'$torso_\gamma^{var}$']

    figsize=(17,5)
    cmap_name = 'RdYlBu' # viridis, RdYlBu

    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=True, figsize=figsize)
    # Row for open loop
    im = ax[0].matshow(df_ol1.corr(), vmin=-1, vmax=1, cmap=plt.get_cmap(cmap_name))
    ax[0].set_xticklabels([''] + labels, fontsize=20, rotation=45, ha='left')
    ax[0].set_yticklabels([''] + labels, fontsize=20)
    ax[0].autoscale(False)
    ax[1].matshow(df_ol2.corr(), vmin=-1, vmax=1, cmap=plt.get_cmap(cmap_name))
    ax[1].set_xticklabels([''] + labels, fontsize=20, rotation=45, ha='left')
    ax[1].autoscale(False)
    ax[1].set_yticklabels([''] + labels, fontsize=20)
    ax[2].matshow(df_ol3.corr(), vmin=-1, vmax=1, cmap=plt.get_cmap(cmap_name))
    ax[2].set_xticklabels([''] + labels, fontsize=20, rotation=45, ha='left')
    ax[2].autoscale(False)
    ax[2].set_yticklabels([''] + labels, fontsize=20)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.1, 0.02, 0.8])
    fig.colorbar(im, cax=cbar_ax)
    #for a in ax:
    #    for tick in a.get_xticklabels():
    #        tick.set_rotation(90)
    plt.axis('on', fontsize=22)
    plt.xticks(ha='left')
    plt.subplots_adjust(wspace=0.03, hspace=5)
    plt.savefig(os.path.join(plot_dir, 'corr_OL.svg'), bbox_inches='tight')  # Change the font size used in the plots


    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=True, figsize=figsize)
    # Row for angle feedback
    im=ax[0].matshow(df_af1.corr(), vmin=-1, vmax=1, cmap=plt.get_cmap(cmap_name))
    ax[0].set_xticklabels([''] + labels, fontsize=20, rotation=45, ha='left')
    ax[0].set_yticklabels([''] + labels, fontsize=20)
    ax[1].matshow(df_af2.corr(), vmin=-1, vmax=1, cmap=plt.get_cmap(cmap_name))
    ax[1].set_xticklabels([''] + labels, fontsize=20, rotation=45, ha='left')
    ax[1].set_yticklabels([''] + labels, fontsize=20)
    ax[2].matshow(df_af3.corr(), vmin=-1, vmax=1, cmap=plt.get_cmap(cmap_name))
    ax[2].set_xticklabels([''] + labels, fontsize=20, rotation=45, ha='left')
    ax[2].set_yticklabels([''] + labels, fontsize=20)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.1, 0.02, 0.8])
    fig.colorbar(im, cax=cbar_ax)
    #for a in ax:
    #    for tick in a.get_xticklabels():
    #        tick.set_rotation(90)
    #plt.axis('off')
    plt.axis('on', fontsize=22)
    plt.subplots_adjust(wspace=0.03, hspace=5)
    plt.savefig(os.path.join(plot_dir, 'corr_AF.svg'), bbox_inches='tight')  # Change the font size used in the plots

    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=True, figsize=figsize)
    # Row for phase reset
    im=ax[0].matshow(df_pr1.corr(), vmin=-1, vmax=1, cmap=plt.get_cmap(cmap_name))
    ax[0].set_xticklabels([''] + labels, fontsize=20, rotation=45, ha='left')
    ax[0].set_yticklabels([''] + labels, fontsize=20)
    ax[1].matshow(df_pr2.corr(), vmin=-1, vmax=1, cmap=plt.get_cmap(cmap_name))
    ax[1].set_xticklabels([''] + labels, fontsize=20, rotation=45, ha='left')
    ax[1].set_yticklabels([''] + labels, fontsize=20)
    ax[2].matshow(df_pr3.corr(), vmin=-1, vmax=1, cmap=plt.get_cmap(cmap_name))
    ax[2].set_xticklabels([''] + labels, fontsize=20, rotation=45, ha='left')
    ax[2].set_yticklabels([''] + labels, fontsize=20)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.1, 0.02, 0.8])
    fig.colorbar(im, cax=cbar_ax)
    #for a in ax:
    #    for tick in a.get_xticklabels():
    #        tick.set_rotation(90)
    plt.axis('on', fontsize=22)
    plt.xticks(ha='left')
    plt.subplots_adjust(wspace=0.03, hspace=5)
    plt.savefig(os.path.join(plot_dir, 'corr_PR.svg'), bbox_inches='tight')  # Change the font size used in the plots

create_corrmat_from_csv(datafile_fitness='wtmpc_gait_eval_fitness.csv', datafile_x_dist='wtmpc_gait_eval_x_dist.csv',
                            datafile_y_dev='wtmpc_gait_eval_abs_y_dev.csv', datafile_foot='wtmpc_gait_eval_avg_footstep.csv',
                            datafile_up='wtmpc_gait_eval_up_time.csv', datafile_alpha='wtmpc_gait_eval_var_alpha.csv',
                            datafile_beta='wtmpc_gait_eval_var_beta.csv', datafile_gamma='wtmpc_gait_eval_var_gamma.csv',
                            plot_title=None, figsize=(10,10))