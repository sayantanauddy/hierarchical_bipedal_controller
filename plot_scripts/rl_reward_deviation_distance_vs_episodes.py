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
rl_log_plot_titles = {'log_20171102_191656.txt': 'Setup 1',
                      'log_20171102_195120.txt': 'Setup 2',
                      'log_20171104_010554.txt': 'Setup 3',
                      'log_20171104_010409.txt': 'Setup 4'
                      }

data_file_list = []
# Create a data file for each log file and append the file name to the list
# This list will be used for generating plots
for logfile in rl_log_plot_titles.keys():
    data_file_list.append(create_data_file(os.path.join(log_dir,logfile)))

# Change the font size used in the plots
font_size = 12
plt.rcParams.update({'font.size': font_size})

plotcount = 1
for data_file in data_file_list:
    i, episodes, rewards, deviations, distances, torso_gamma = np.loadtxt(data_file, delimiter=',', unpack=True, skiprows=1)

    plt.figure(plotcount)
    plt.xlabel('Episode')
    plt.ylabel('Total reward for episode')
    plt.ylim(-150,60)
    reward_plot_title = 'Rewards vs episodes | ' + rl_log_plot_titles[os.path.basename(data_file).replace('rl_train_data_', '')]
    plt.title(reward_plot_title, fontsize=font_size)
    plt.plot(episodes, rewards, color='g', linewidth=2.0, label='Reward')
    plt.legend(prop={'size': font_size})
    plot_file_name1 = os.path.basename(data_file).replace('.txt', '') + '_reward.png'
    plt.axhline(y=0.0, color='black', linestyle='-', linewidth=0.5)
    plt.savefig(os.path.join(plot_dir, plot_file_name1))

    plt.figure(plotcount + 1)
    plt.xlabel('Episode')
    plt.ylabel('Deviation/Distance (m)')
    plt.ylim((-6,8))
    dev_plot_title = 'Distance/deviation vs episodes | ' + rl_log_plot_titles[os.path.basename(data_file).replace('rl_train_data_', '')]
    plt.title(dev_plot_title, fontsize=font_size)
    plt.plot(episodes, deviations, color='r', linewidth=2.0, label='Deviation')
    plt.plot(episodes, distances, color='b', linewidth=2.0, label='Distance')
    plot_file_name2 = os.path.basename(data_file).replace('.txt', '') + '_deviation.png'
    plt.legend(prop={'size': font_size})
    # Draw a horizontal line at 0 for reference
    plt.axhline(y=0.0, color='black', linestyle='-', linewidth=0.5)
    plt.savefig(os.path.join(plot_dir, plot_file_name2))

    plt.figure(plotcount + 2)
    plt.xlabel('Episode')
    plt.ylabel('Torso orientation (radians)')
    plt.ylim((-3, 3))
    dev_plot_title = 'Torso gamma angle vs episodes | ' + rl_log_plot_titles[os.path.basename(data_file).replace('rl_train_data_', '')]
    plt.title(dev_plot_title, fontsize=font_size)
    plt.plot(episodes, torso_gamma, color='brown', linewidth=2.0, label='Torso orientation')
    plot_file_name2 = os.path.basename(data_file).replace('.txt', '') + '_orientation.png'
    plt.legend(prop={'size': font_size})
    # Draw a horizontal line at 0 for reference
    plt.axhline(y=0.0, color='black', linestyle='-', linewidth=0.5)
    plt.savefig(os.path.join(plot_dir, plot_file_name2))

    plotcount += 3


