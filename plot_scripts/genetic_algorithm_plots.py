import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os

# Machine 1: ASUS
# Machine 2: wtmpc
ga_logs = [{'machine': '1', 'feedback': 'Open Loop', 'run': 1, 'log': 'log_20170830_172324.txt'},
           {'machine': '1', 'feedback': 'Open Loop', 'run': 2, 'log': 'log_20171027_225729.txt'},
           {'machine': '1', 'feedback': 'Open Loop', 'run': 3, 'log': 'log_20171030_033117.txt'},

           {'machine': '1', 'feedback': 'Angle feedback', 'run': 1, 'log': 'log_20170908_214752.txt'},
           {'machine': '1', 'feedback': 'Angle feedback', 'run': 2, 'log': 'log_20170929_131608.txt'},
           {'machine': '1', 'feedback': 'Angle feedback', 'run': 3, 'log': 'log_20171009_101803.txt'},

           {'machine': '1', 'feedback': 'Phase Reset', 'run': 1, 'log': 'log_20170921_110728.txt'},
           {'machine': '1', 'feedback': 'Phase Reset', 'run': 2, 'log': 'log_20171001_170505.txt'},
           {'machine': '1', 'feedback': 'Phase Reset', 'run': 3, 'log': 'log_20171012_012106.txt'},

           {'machine': '2', 'feedback': 'Open Loop', 'run': 1, 'log': 'log_20171002_113858.txt'},
           {'machine': '2', 'feedback': 'Open Loop', 'run': 2, 'log': 'log_20171004_094431.txt'},
           {'machine': '2', 'feedback': 'Open Loop', 'run': 3, 'log': 'log_20171006_074055.txt'},

           {'machine': '2', 'feedback': 'Angle feedback', 'run': 1, 'log': 'log_20171002_182720.txt'},
           {'machine': '2', 'feedback': 'Angle feedback', 'run': 2, 'log': 'log_20171004_173049.txt'},
           {'machine': '2', 'feedback': 'Angle feedback', 'run': 3, 'log': 'log_20171006_145448.txt'},

           {'machine': '2', 'feedback': 'Phase Reset', 'run': 1, 'log': 'log_20171002_120630.txt'},
           {'machine': '2', 'feedback': 'Phase Reset', 'run': 2, 'log': 'log_20171004_132604.txt'},
           {'machine': '2', 'feedback': 'Phase Reset', 'run': 3, 'log': 'log_20171006_140001.txt'}]

# Set the home directory
home_dir = os.path.expanduser('~')

# Set the directory for saving plots
plot_dir = os.path.join(home_dir, 'computing/repositories/MScThesis_SayantanAuddy_2017_NICOOscillatorWalking/plots')

# Directory for saving plot data files
plot_data_dir = os.path.join(home_dir, 'computing/repositories/MScThesis_SayantanAuddy_2017_NICOOscillatorWalking/plot_data')

# Set the directory where logs are stored
log_dir = os.path.join(home_dir, 'computing/repositories/MScThesis_SayantanAuddy_2017_NICOOscillatorWalking/logs/genetic_algorithm_logs')

# For each log file there will be 2 plots (2 data files)
# 1. Plot of fitness (max, min, avg, std) vs generations
# 2. Plot of best distance vs generations

# So for 18 log files there will be 36 plots and 36 data files


def generate_fitness_datafile(log_file_path):

    # Read the Min fitness for each generation
    min_fitness_search_string = 'Min'
    max_fitness_search_string = 'Max'
    avg_fitness_search_string = 'Avg'
    std_fitness_search_string = 'Std'

    min_list = list()
    max_list = list()
    avg_list = list()
    std_list = list()

    lines = open(log_file_path, "r").read().splitlines()

    for i, line in enumerate(lines):

        # Initialize the values
        min_fitness, max_fitness, avg_fitness, std_fitness = 0.0, 0.0, 0.0, 0.0

        if min_fitness_search_string in line:
            min_fitness = line.split(' ')[4]

        if max_fitness_search_string in line:
            max_fitness = line.split(' ')[4]

        if avg_fitness_search_string in line:
            avg_fitness = line.split(' ')[4]

        if std_fitness_search_string in line:
            std_fitness = line.split(' ')[4]

        # Append to the list if the values have been set
        if min_fitness > 0.0:
            min_list.append(min_fitness)

        if max_fitness > 0.0:
            max_list.append(max_fitness)

        if avg_fitness > 0.0:
            avg_list.append(avg_fitness)

        if std_fitness > 0.0:
            std_list.append(std_fitness)

        # Only process till the end of 30 generations
        if 'End of generation 30' in line:
            break

    # At this point file_list will have 30 values of min_fitness, max_fitness, avg_fitness, std_fitness
    # Create a list for generation numbers
    gen_list = range(1,31)

    # Crate data frames by using the 1D lists and reshaping them
    df_gen = pd.DataFrame(np.array(gen_list).reshape(30, 1), columns=['Generation'])
    df_min = pd.DataFrame(np.array(min_list).reshape(30, 1), columns=['Min_fitness'])
    df_max = pd.DataFrame(np.array(max_list).reshape(30, 1), columns=['Max_fitness'])
    df_avg = pd.DataFrame(np.array(avg_list).reshape(30, 1), columns=['Avg_fitness'])
    df_std = pd.DataFrame(np.array(std_list).reshape(30, 1), columns=['Std_fitness'])

    df_fitness = pd.concat([df_gen, df_min, df_max, df_avg, df_std], axis=1)

    # Derive the name of the data file from the log file
    fitness_data_file_path = os.path.join(plot_data_dir, 'ga_fitness_' + os.path.basename(log_file_path)).replace('.txt', '.csv')

    # Store the dataframe as a csv in the data folder
    df_fitness.to_csv(fitness_data_file_path, sep=',', index=False)

    # Now using the max fitness value in each generation, find the distance walked by that chromosome
    max_dist_list = list()
    curr_minus_line = ''
    curr_line = ''
    j = 2

    for max_fit in max_list:
        for i, line in enumerate(lines):
            if 'fitness: ' +str(max_fit) in line:
                curr_line = line
                for k in range(15):
                    if 'end_x' in lines[i - k]:
                        j=k
                curr_minus_line = lines[i - j]
                max_dist = curr_minus_line.split(' ')[3].split('=')[1].replace(',', '')
                max_dist_list.append(max_dist)

                # Only process till the end of generation 30
                if 'End of generation 30' in curr_line:
                    break

    # Create a dataframe for max distance
    df_max_dist = pd.DataFrame(np.array(max_dist_list).reshape(30, 1), columns=['Max_distance'])
    df_max_dist = pd.concat([df_gen, df_max_dist], axis=1)

    # Derive the name of the data file from the log file
    max_dist_data_file_path = os.path.join(plot_data_dir, 'ga_max_dist_' + os.path.basename(log_file_path)).replace('.txt', '.csv')

    # Store the dataframe as a csv in the data folder
    df_max_dist.to_csv(max_dist_data_file_path, sep=',', index=False)

    return fitness_data_file_path, max_dist_data_file_path


def generate_fitness_plot(datafile_path, machine, feedback, run):

    # Change the font size used in the plots
    font_size = 12
    plt.rcParams.update({'font.size': font_size})

    # Read the datafile
    gen, min_f, max_f, avg_f, std_f = np.loadtxt(datafile_path, delimiter=',', unpack=True, skiprows=1)

    plt.figure()
    plt.xlabel('Generations')
    plt.ylabel('Fitness Value')
    plt.xlim(1, 30)
    plt.ylim(-5, 25)
    reward_plot_title = 'Fitness vs generations | Machine: {0} | Type: {1} | Run: {2}'.format(machine, feedback, run)
    plt.title(reward_plot_title, fontsize=font_size)
    plt.plot(gen, min_f, color='red', linewidth=2.0, label='Minimum Fitness')
    plt.plot(gen, max_f, color='blue', linewidth=2.0, label='Maximum Fitness')
    plt.plot(gen, avg_f, color='green', linewidth=2.0, label='Average Fitness')
    plt.plot(gen, std_f, color='orange', linewidth=2.0, label='Std. Fitness')
    plt.legend(prop={'size': font_size})

    plot_file_path = os.path.basename(datafile_path).replace('.csv', '') + '.png'
    plt.axhline(y=0.0, color='black', linestyle='-', linewidth=0.5)
    plt.savefig(os.path.join(plot_dir, plot_file_path))
    plt.close()
    plt.gcf().clear()

    return plot_file_path


def generate_max_dist_plot(datafile_path, machine, feedback, run):
    # Change the font size used in the plots
    font_size = 12
    plt.rcParams.update({'font.size': font_size})

    # Read the datafile
    gen, max_dist = np.loadtxt(datafile_path, delimiter=',', unpack=True, skiprows=1)

    plt.figure()
    plt.xlabel('Generations')
    plt.ylabel('Max Distance (m)')
    plt.xlim(1, 30)
    plt.ylim(-1, 5)
    reward_plot_title = 'Max distance vs generations | Machine: {0} | Type: {1} | Run: {2}'.format(machine, feedback, run)
    plt.title(reward_plot_title, fontsize=font_size)
    plt.plot(gen, max_dist, color='blue', linewidth=2.0, label='Max Distance')
    plt.legend(prop={'size': font_size})
    plot_file_path = os.path.basename(datafile_path).replace('.csv', '') + '.png'
    plt.axhline(y=0.0, color='black', linestyle='-', linewidth=0.5)
    plt.savefig(os.path.join(plot_dir, plot_file_path))
    plt.gcf().clear()

    return plot_file_path


# Create a file to store the meaning of the different plots
plot_key_file_path = os.path.join(plot_dir, 'ga_plots_key.csv')
plot_key_file = open(plot_key_file_path, 'w')
plot_key_file.write('Machine, Type,Run,Fitnes_Plot,Fitness_Data,Distance_Plot,Distance_Data\n')

for ga_log in ga_logs:

    logfile_name = ga_log['log']
    machine = ga_log['machine']
    feedback = ga_log['feedback']
    run = ga_log['run']

    # Generate the data files
    fitness_data_file_path, max_dist_data_file_path = generate_fitness_datafile(os.path.join(log_dir, logfile_name))

    # Generate the fitness plot
    plot_fitness_file_path = generate_fitness_plot(fitness_data_file_path, machine, feedback, run)

    # Generate the max distance plot
    plot_max_dist_file_path = generate_max_dist_plot(max_dist_data_file_path, machine, feedback, run)

    plot_key_file.write('{0},{1},{2},{3},{4},{5},{6}\n'.format(machine,
                                                               feedback,
                                                               run,
                                                               os.path.basename(plot_fitness_file_path),
                                                               os.path.basename(fitness_data_file_path),
                                                               os.path.basename(plot_max_dist_file_path),
                                                               os.path.basename(max_dist_data_file_path)))

print 'Plot key file: {}'.format(plot_key_file_path)
