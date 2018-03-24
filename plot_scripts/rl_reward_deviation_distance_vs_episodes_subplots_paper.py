import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib

# For avoiding type-3 fonts
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Set the home directoryfrom matplotlib import rcParams

home_dir = os.path.expanduser('~')

# Set the directory for saving plots
plot_dir = os.path.join(home_dir, 'computing/repositories/MScThesis_SayantanAuddy_2017_NICOOscillatorWalking/plots')

# Directory for saving plot data files
plot_data_dir = os.path.join(home_dir, 'computing/repositories/MScThesis_SayantanAuddy_2017_NICOOscillatorWalking/plot_data')

# Set the directory where logs are stored
log_dir = os.path.join(home_dir, 'computing/repositories/MScThesis_SayantanAuddy_2017_NICOOscillatorWalking/logs/reinforcement_learning_logs')


# For each training log file, 2 plots are required (reward vs test episode, distance and deviation vs test episode)
# Function to create a dataframe from a reinforcement learning training log file
def create_data_file(log_file_path):

    file_list = list()

    # First search for total reward at the end of test episodes
    # The line with thesearch string will contain the reward for the test episode
    # 2 lines above the search string will be the deviation and distance for the test episode
    # Equivalent grep command: grep -B2 '\[DDPG TEST\] TOTAL REWARD' log_20171029_203316.txt
    search_string = '[DDPG TEST] TOTAL REWARD'

    # Column names
    cols = ['episode', 'reward', 'deviation', 'distance', 'torso_gamma']

    file_list = list()

    lines = open(log_file_path, "r").read().splitlines()

    curr_minus_line = ''
    curr_line = ''
    j = 2

    for i, line in enumerate(lines):
        if search_string in line:
            curr_line = line
            # Search for the line containing 'Deviation' above this line
            for k in range(5):
                if 'Deviation' in lines[i - k]:
                    j=k
            curr_minus_line = lines[i - j]
            deviation = curr_minus_line.split(' ')[4]
            distance = curr_minus_line.split(' ')[6]
            torso_gamma = curr_minus_line.split(' ')[8]
            reward = curr_line.split(' ')[12]
            episode = curr_line.split(' ')[7].replace('-th', '')

            # Apend the current observations to the 1D list for first 500 episodes
            if int(episode) > 500 : pass
            file_list.append([episode, reward, deviation, distance, torso_gamma])

    # Crate a data frame by using the 1D list and reshaping it
    df = pd.DataFrame(np.array(file_list).reshape(99, 5), columns=cols)

    # Derive the name of the data file from the log file
    data_file_path = os.path.join(plot_data_dir, 'rl_train_data_' + os.path.basename(log_file_path))

    print 'Saving {}'.format(data_file_path)

    # Store the dataframe as a csv in the data folder
    df.to_csv(data_file_path, sep=',')

    return data_file_path


# Dict of RL training logs and plot titles
rl_log_plot_titles = {'log_20171102_191656.txt': 'Setup1',
                      'log_20171102_195120.txt': 'Setup2',
                      'log_20171104_010554.txt': 'Setup3',
                      'log_20171104_010409.txt': 'Setup4'
                      }

data_file_list = []
# Create a data file for each log file and append the file name to the list
# This list will be used for generating plots
for logfile in rl_log_plot_titles.keys():
    data_file_list.append(create_data_file(os.path.join(log_dir,logfile)))

# Change the font size used in the plots
font_size = 32
plt.rcParams.update({'font.size': font_size})

# Each column for one data file
fig, ax = plt.subplots(3, 4, figsize=(42, 10), sharex=True, sharey=False)

# Set the line width
linewidth = 3.5
# Whether to show grids or not
grid = True

filecount = 0
for data_file in data_file_list:

    i, episodes, rewards, deviations, distances, torso_gamma = np.loadtxt(data_file, delimiter=',', unpack=True, skiprows=1)

    # Set the x-axis limits
    ax[0][filecount].set_xlim([0, 1000])
    ax[1][filecount].set_xlim([0, 1000])
    ax[2][filecount].set_xlim([0, 1000])

    # Set the y-axis limits
    ax[0][filecount].set_ylim([-145,75])
    ax[1][filecount].set_ylim([-6,7.9])
    ax[2][filecount].set_ylim([-2.9, 2.9])

    # Set the plot titles
    ax[0][0].set_title('Setup S1', y=1.05, fontsize=32)
    ax[0][1].set_title('Setup S2', y=1.05, fontsize=32)
    ax[0][2].set_title('Setup S3', y=1.05, fontsize=32)
    ax[0][3].set_title('Setup S4', y=1.05, fontsize=32)


    # Set the axis labels (since x-axis is shared, only the last subplot has an xlabel)
    ax[2][filecount].set_xlabel('Episode', fontsize=font_size)
    plt.setp(ax[2][filecount].get_xticklabels(), rotation=30, fontsize=font_size, ha='right', rotation_mode='anchor')

    # Set y labels only for the first column
    if filecount == 0:
        ax[0][filecount].set_ylabel('Reward', fontsize=font_size, horizontalalignment='center')
        ax[0][filecount].get_yaxis().set_label_coords(-0.21,0.5)
        ax[0][filecount].tick_params(axis='y', pad=15)
        #ax[0][filecount].set_yticklabels([-150, -100, -50, 0, 50, visible=True, fontsize=30)
    else:
        ax[0][filecount].set_yticklabels([], visible=False)

    ax[0][filecount].yaxis.set_ticks_position('both')

    if filecount == 0:
        ax[1][filecount].set_ylabel('Deviation/\nDistance (m)', fontsize=font_size, horizontalalignment='center')
        ax[1][filecount].get_yaxis().set_label_coords(-0.15,0.5)
        ax[1][filecount].set_yticklabels(['',-4,'',0,'',4,''], visible=True)
        ax[1][filecount].tick_params(axis='y', pad=15)
    else:
        if filecount == 3:
            #ax[1][filecount].yaxis.tick_right()
            #ax[1][filecount].yaxis.set_ticks_position('both')
            ax[1][filecount].set_yticklabels([], visible=False)
        else:
            ax[1][filecount].set_yticklabels([], visible=False)
    ax[1][filecount].yaxis.set_ticks_position('both')

    if filecount == 0:
        ax[2][filecount].set_ylabel('Orientation \n(radians)', fontsize=font_size, horizontalalignment='center')
        ax[2][filecount].get_yaxis().set_label_coords(-0.15,0.5)
        ax[2][filecount].tick_params(axis='y', pad=15)
    else:
        if filecount == 3:
            #ax[2][filecount].yaxis.tick_right()
            #ax[2][filecount].yaxis.set_ticks_position('both')
            ax[2][filecount].set_yticklabels([], visible=False)
        else:
            ax[2][filecount].set_yticklabels([], visible=False)
    ax[2][filecount].yaxis.set_ticks_position('both')

    # First subplot contains max and avg fitness for run 1
    l0, = ax[0][filecount].plot(episodes, rewards, color='green', linewidth=linewidth, label='Reward')
    ax[0][filecount].axhline(y=0.0, color='black', linestyle='-', linewidth=0.5)
    if grid: ax[0][filecount].grid(color='#222222', linestyle='dotted', linewidth=1)

    # Second subplot contains max and avg fitness for run 2
    l11, = ax[1][filecount].plot(episodes, deviations, color='red', linewidth=linewidth, label='Deviation')
    l12, = ax[1][filecount].plot(episodes, distances, color='blue', linewidth=linewidth, label='Distance')
    ax[1][filecount].axhline(y=0.0, color='black', linestyle='-', linewidth=0.5)
    if grid: ax[1][filecount].grid(color='#222222', linestyle='dotted', linewidth=1)

    # Third subplot contains max and avg fitness for run 3
    l2, = ax[2][filecount].plot(episodes, torso_gamma, color='#734607', linewidth=linewidth, label='Orientation')
    ax[2][filecount].axhline(y=0.0, color='black', linestyle='-', linewidth=0.5)
    if grid: ax[2][filecount].grid(color='#222222', linestyle='dotted', linewidth=1)

    filecount += 1

#plt.setp([a.get_xticklabels() for a in ax[0, :]], visible=False)
#plt.setp([a.get_yticklabels() for a in x[:, 1:1:3]], visible=False)

plt.subplots_adjust(hspace=0.1)
plt.subplots_adjust(wspace=0.07)
plt.legend(handles=[l0, l11, l12, l2], bbox_to_anchor=(1.036, 3.9), loc='upper right', ncol=4, fontsize=33, borderpad=0.1)

plot_file_name = 'rl_train_paper.eps'
plt.savefig(os.path.join(plot_dir, plot_file_name), bbox_inches='tight')
