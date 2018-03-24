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

# Function to create a dataframe from a reinforcement learning test log file
def create_df(log_file_path):

    file_list = list()

    # First search for total reward at the end of test episodes
    # The line with thesearch string will contain the reward for the test episode
    # 2 lines above the search string will be the deviation and distance for the test episode
    # Equivalent grep command: grep -B2 '\[DDPG TEST\] TOTAL REWARD' log_20171029_203316.txt
    search_string = '[DDPG TEST] TOTAL REWARD'

    # Column names
    cols = ['episode', 'reward', 'deviation', 'distance', 'torso_orientation']

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
            torso_orientation = curr_minus_line.split(' ')[8]
            reward = curr_line.split(' ')[12]
            episode = curr_line.split(' ')[7].replace('-th', '')

            # Apend the current observations to the 1D list
            file_list.append([episode, reward, deviation, distance, torso_orientation])

    # Crate a data frame by using the 1D list and reshaping it
    df = pd.DataFrame(np.array(file_list).reshape(100, 5), columns=cols)

    # Derive the name of the data file from the log file
    data_file_path = os.path.join(plot_data_dir, 'rl_test_data_' + os.path.basename(log_file_path))

    print 'Saving {}'.format(data_file_path)

    return df

# Function to create a dataframe from a static deviation test log file
def create_df_static(log_file_path):
    file_list = list()

    # Lines with the search string will contain the distance, deviation and torso-gamma
    search_string_straight = 'STATIC TEST straight'

    # Column names
    cols = ['episode', 'deviation', 'distance', 'torso_orientation']

    file_list_straight = list()

    lines = open(log_file_path, "r").read().splitlines()

    for i, line in enumerate(lines):

        if search_string_straight in line:
            episode = line.split(' ')[6]
            distance = line.split(' ')[8]
            deviation = line.split(' ')[10]
            torso_orientation = line.split(' ')[12]
            file_list_straight.append([episode, deviation, distance, torso_orientation])

    # Crate a data frame by using the 1D list and reshaping it
    df_straight = pd.DataFrame(np.array(file_list_straight).reshape(100, 4), columns=cols)

    # Return the 3 dataframes
    return df_straight

global figcount
figcount = 1

def save_boxplot(datafile, plotfile, type, time, limits):
    font_size = 36
    plt.rcParams.update({'font.size': font_size})

    global figcount
    linewidth = 1
    # Create box plots using the data files saved above
    i, setup_control, setup1, setup2, setup3, setup4 = np.loadtxt(os.path.join(plot_data_dir, datafile), delimiter=',', unpack=True, skiprows=1)

    fig = plt.figure(figcount, figsize=(18,10))
    figcount += 1
    ax = fig.add_subplot(111)
    setup_list = [setup_control, setup1, setup2, setup3, setup4]
    setup_list.reverse()
    bp = ax.boxplot(setup_list,
                    showmeans=False, patch_artist=True, vert=False,
                    meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='yellow'),
                    flierprops=dict(marker='o', markerfacecolor='#888888', markersize=5, linestyle='none', markeredgecolor='black'))
    box_count = 0
    for box in bp['boxes']:
        # change outline color
        box.set(color='#000000', linewidth=linewidth)
        # change fill color
        if box_count == 4:
            box.set_facecolor('#FA8C8C')
        else:
            box.set_facecolor('#9D98FA')
        box_count += 1

    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#000000', linewidth=linewidth)

    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#000000', linewidth=linewidth)

    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#000000', linewidth=linewidth)

    for mean in bp['means']:
        mean.set(color='#000000', linewidth=linewidth)
        #mean.set_facecolor('#FF0000')

    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#E0636F', alpha=1.0, linewidth=linewidth)

    ax.set_xlim(limits)
    ax.set_xlabel('{0} after {1}s ({2})'.format(type, time, 'm' if not type=='Torso Orientation' else 'radians'), fontsize=font_size)
    #ax.set_ylabel('Different setups', fontsize=24)
    ylabel_list = [r'$Setup_{control}$', r'$Setup_1$', r'$Setup_2$', r'$Setup_3$', r'$Setup_4$']
    ylabel_list.reverse()
    ax.set_yticklabels(ylabel_list, fontsize=font_size)
    # Color the Y labels according to the box colors
    colors = ['b', 'b', 'b', 'b', 'r']
    for ytick, color in zip(ax.get_yticklabels(), colors):
        ytick.set_color(color)
    #ax.set_title('{0} for 100 test episodes of {1}s'.format(type, time))
    ax.axvline(x=0.0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(color='#222222', linestyle='dotted', linewidth=1)
    plt.savefig(os.path.join(plot_dir, plotfile), bbox_inches='tight')


# Here we have 4 test logs for the 4 settings combinations and 2 more logs for the static tests

# Setup: 1
# Reward function: 1.0*(-abs(deviation)) + 0.5*forward_x + 1.0*(-abs(torso_gamma))
# Lowest possible gain: 0.1
# Walk time: 40s
# Machine: kogspc16
test_log_setup1 = 'log_20171103_183052.txt'

# Setup: 2
# Reward function: 1.0*(-abs(deviation)) + 0.5*forward_x + 1.0*(-abs(torso_gamma))
# Lowest possible gain: 0.4
# Walk time: 40s
# Machine: kogspc37
test_log_setup2 = 'log_20171103_181730.txt'

# Setup: 3
# Reward function: 1.0*(-abs(deviation)) + 0.3*forward_x + 1.0*(-abs(torso_gamma))
# Lowest possible gain: 0.1
# Walk time: 40s
# Machine: kogspc16
test_log_setup3 = 'log_20171104_211720.txt'

# Setup: 4
# Reward function: 1.0*(-abs(deviation)) + 0.3*forward_x + 1.0*(-abs(torso_gamma))
# Lowest possible gain: 0.4
# Walk time: 40s
# Machine: kogspc37
test_log_setup4 = 'log_20171104_211750.txt'

# Setup: Static test (straight)
# Walk time: 40s
# Machine: kogspc37
test_log_static = 'log_20171103_230940.txt'

# Create dataframes from the test log files
df_test_log_setup1 = create_df(os.path.join(log_dir, test_log_setup1))
df_test_log_setup2 = create_df(os.path.join(log_dir, test_log_setup2))
df_test_log_setup3 = create_df(os.path.join(log_dir, test_log_setup3))
df_test_log_setup4 = create_df(os.path.join(log_dir, test_log_setup4))

# Create dataframe for the static test
df_test_log_static = create_df_static(os.path.join(log_dir, test_log_static))

# Now we need dataframes for the following parameters
# 1. Distance
# 2. Deviation
# 3. Torso orientation
# Each of these dataframes will have columns for the static test and each of the 4 setups

col_names = ['straight', 'Setup 1', 'Setup 2', 'Setup 3', 'Setup 4']

# dataframe for distance
df_boxplot_distance = pd.concat([df_test_log_static['distance'],
                                 df_test_log_setup1['distance'],
                                 df_test_log_setup2['distance'],
                                 df_test_log_setup3['distance'],
                                 df_test_log_setup4['distance'],
                                 ], axis=1)
df_boxplot_distance.columns = col_names

# dataframe for deviation
df_boxplot_deviation = pd.concat([df_test_log_static['deviation'],
                                 df_test_log_setup1['deviation'],
                                 df_test_log_setup2['deviation'],
                                 df_test_log_setup3['deviation'],
                                 df_test_log_setup4['deviation'],
                                 ], axis=1)
df_boxplot_deviation.columns = col_names

# dataframe for torso_orientation
df_boxplot_torso_orientation = pd.concat([df_test_log_static['torso_orientation'],
                                          df_test_log_setup1['torso_orientation'],
                                          df_test_log_setup2['torso_orientation'],
                                          df_test_log_setup3['torso_orientation'],
                                          df_test_log_setup4['torso_orientation'],
                                          ], axis=1)
df_boxplot_torso_orientation.columns = col_names

# Save the boxplot dataframes
boxplot_datafilenames = ['rl_test_data_boxplot_distance.csv',
                         'rl_test_data_boxplot_deviation.csv',
                         'rl_test_data_boxplot_torso_orientation.csv']

df_boxplot_distance.to_csv(os.path.join(plot_data_dir, boxplot_datafilenames[0]))
df_boxplot_deviation.to_csv(os.path.join(plot_data_dir, boxplot_datafilenames[1]))
df_boxplot_torso_orientation.to_csv(os.path.join(plot_data_dir, boxplot_datafilenames[2]))


# Create the plots
save_boxplot(boxplot_datafilenames[0], 'rl_test_data_boxplot_distance.eps', 'Distance', 40, (-1, 7))
save_boxplot(boxplot_datafilenames[1], 'rl_test_data_boxplot_deviation.eps', 'Deviation', 40, (-3, 5))
save_boxplot(boxplot_datafilenames[2], 'rl_test_data_boxplot_torso_orientation.eps', 'Torso Orientation', 40, (-1, 3))

# Print the median and standard deviation for each setup and parameters

for col in col_names:
    # col represents the different setups
    # Calculate the IRQs for the different parameters
    dist_q75, dist_q25 = np.percentile(df_boxplot_distance[col].apply(float), [75, 25])
    dist_iqr = dist_q75 - dist_q25

    dev_q75, dev_q25 = np.percentile(df_boxplot_deviation[col].apply(float), [75, 25])
    dev_iqr = dev_q75 - dev_q25

    gamma_q75, gamma_q25 = np.percentile(df_boxplot_torso_orientation[col].apply(float), [75, 25])
    gamma_iqr = gamma_q75 - gamma_q25

    # Print the medians and IQRs
    print '{}          & {}          & {}          & {}          & {}          & {}          & {}'.format(col,
                                                                                                          "{0:.3f}".format(df_boxplot_distance[col].apply(float).median()),
                                                                                                          "{0:.3f}".format(df_boxplot_deviation[col].apply(float).median()),
                                                                                                          "{0:.3f}".format(df_boxplot_torso_orientation[col].apply(float).median()),
                                                                                                          "{0:.3f}".format(dist_iqr),
                                                                                                          "{0:.3f}".format(dev_iqr),
                                                                                                          "{0:.3f}".format(gamma_iqr))

    print ''


