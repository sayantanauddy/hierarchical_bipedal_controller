import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os

# Set the home directory
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
font_size = 20
plt.rcParams.update({'font.size': font_size})

plotcount = 1
for data_file in data_file_list:

    plot_file_name_sub = ''
    for k,v in rl_log_plot_titles.items():
        if k in data_file:
            plot_file_name_sub = v

    i, episodes, rewards, deviations, distances, torso_gamma = np.loadtxt(data_file, delimiter=',', unpack=True, skiprows=1)

    fig, ax = plt.subplots(3, figsize=(18, 10), sharex=True, sharey=False)

    # Set the line width
    linewidth = 3.5

    # Whether to show grids or not
    grid = True

    # Set the x-axis limits
    ax[0].set_xlim([0, 1000])
    ax[1].set_xlim([0, 1000])
    ax[2].set_xlim([0, 1000])

    # Set the y-axis limits
    ax[0].set_ylim([-145,75])
    ax[1].set_ylim([-6,7.9])
    ax[2].set_ylim([-2.9, 2.9])

    # Set the axis labels (since x-axis is shared, only the last subplot has an xlabel)
    ax[2].set_xlabel('Episode', fontsize=22)

    ax[0].set_ylabel('Reward', fontsize=22, horizontalalignment='center')
    ax[0].yaxis.set_ticks_position('both')
    ax[1].set_ylabel('Deviation/\nDistance (m)', fontsize=22, horizontalalignment='center')
    ax[1].yaxis.set_ticks_position('both')
    ax[2].set_ylabel('Torso orientation \n(radians)', fontsize=22, horizontalalignment='center')
    ax[2].yaxis.set_ticks_position('both')

    # First subplot contains max and avg fitness for run 1
    l0, = ax[0].plot(episodes, rewards, color='green', linewidth=linewidth, label='Reward')
    ax[0].axhline(y=0.0, color='black', linestyle='-', linewidth=0.5)
    if grid: ax[0].grid(color='#222222', linestyle='dotted', linewidth=1)

    # Second subplot contains max and avg fitness for run 2
    l11, = ax[1].plot(episodes, deviations, color='red', linewidth=linewidth, label='Deviation')
    l12, = ax[1].plot(episodes, distances, color='blue', linewidth=linewidth, label='Distance')
    ax[1].axhline(y=0.0, color='black', linestyle='-', linewidth=0.5)
    if grid: ax[1].grid(color='#222222', linestyle='dotted', linewidth=1)

    # Third subplot contains max and avg fitness for run 3
    l2, = ax[2].plot(episodes, torso_gamma, color='#734607', linewidth=linewidth, label='Torso orientation')
    ax[2].axhline(y=0.0, color='black', linestyle='-', linewidth=0.5)
    if grid: ax[2].grid(color='#222222', linestyle='dotted', linewidth=1)

    plt.subplots_adjust(hspace=0.1)
    plt.legend(handles=[l0, l11, l12, l2], bbox_to_anchor=(1.01, 3.55), loc='upper right', ncol=4, fontsize=20)

    plot_file_name = 'rl_train_' + plot_file_name_sub + '.svg'
    plt.savefig(os.path.join(plot_dir, plot_file_name), bbox_inches='tight')
