"""
This prgoram processes video data to extract and analyze features related to drowsiness detection.

The program consists of the following sections:

1. Import/Initialization Section
    - Import necessary python libraries.
    - Define folder paths.
    - Define global variables.

2. Data Processing Section
    - Function 'preprocess_features': Processes the dataframes with thresholding, EMA smoothing, removing irrelevant initial frames, and normalizing the dataset.
    - Function 'process_statistics': Processes the dataframe to call the functions to calculate video statistics and save the video boxplot image and return the updated participant list.

3. Data Statistics Section
    - Function 'get_video_statistics': Calculate each video statistics for plotting boxplots.
    - Function 'calculate_abs_diff': Calculate the absolute difference of the video mean and the initial frame average.

3. Visualization Section
    - Function 'save_video_boxplot': Create and save an image containing a plot of the video boxplots for each features.
    - Function 'save_participant_boxplot': Create and save an image containing a plot of the participant boxplots for each features of each videos.
    - Function 'write_participant_csv': Open and edit a csv file to save the computed video data statistics.
    
3. Main Section:
    - Function `main`: Iterates through each video data and calls necessary functions to calculate and visualize statistics for data analysis.

The program is designed to be run as a standalone module, and the main function `main` is called when the script is executed directly. 
It utilizes various parameters and configurations defined within the script.
"""

##############################################################################################
#                                                                                            #
#                               IMPORT/INITIALIZATION SECTION                                #
#                                                                                            #
##############################################################################################

import os
import time
import shutil
import textwrap
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

# Lists of folder paths
training_folder_paths = [
    r"D:\Processed Combined NTHU Training Dataset\glasses",
    r"D:\Processed Combined NTHU Training Dataset\noglasses",
    r"D:\Processed Combined NTHU Training Dataset\nightglasses",
    r"D:\Processed Combined NTHU Training Dataset\night_noglasses"
]

evaluation_folder_paths = [
    r"D:\Processed Combined NTHU Evaluation Dataset\glasses",
    r"D:\Processed Combined NTHU Evaluation Dataset\noglasses",
    r"D:\Processed Combined NTHU Evaluation Dataset\nightglasses",
    r"D:\Processed Combined NTHU Evaluation Dataset\night_noglasses"
]

test_folder_paths = [
    r"D:\Processed Combined NTHU Test Dataset\glasses",
    r"D:\Processed Combined NTHU Test Dataset\noglasses",
    r"D:\Processed Combined NTHU Test Dataset\nightglasses",
    r"D:\Processed Combined NTHU Test Dataset\night_noglasses"
]

# Combine all datasets into a single list for iteration
all_dataset_paths = [training_folder_paths, evaluation_folder_paths, test_folder_paths]

# Output directories
before_processing_path = r"D:\Video Dataset Data Analysis\Box Plots Before Processing"
after_processing_path = r"D:\Video Dataset Data Analysis\Box Plots After Processing"

# Define initial frame for initial average
initial_frame = 15

# Processing constants
alpha_ear = 0.15
alpha_mar = 0.15
alpha_yaw = 0.25
alpha_pitch = 0.3
alpha_roll = 0.4
ear_diff_threshold = 0.5
mar_diff_threshold = 0.5
yaw_diff_threshold = 30
pitch_diff_threshold = 30
roll_diff_threshold = 30

# Define the feature columns and statistics requirement for data analysis
features = ['EAR', 'MAR', 'Yaw', 'Pitch', 'Roll']
stat_names = ['Initial Avg.', 'Video Avg.', 'Percentile_100 (Max)', 'Percentile_75', 'Percentile_50', 'Percentile_25', 'Percentile_0 (Min)']

# Define the csv column width
column_width = 50

##############################################################################################
#                                                                                            #
#                                 DATA PROCESSING SECTION                                    #
#                                                                                            #
##############################################################################################

# Function to process the data
def preprocess_features(df, unstable_frames):
    """
    Preprocesses the feature data in the given DataFrame.

    Parameters:
        df (DataFrame): The input DataFrame containing feature data.

    Returns:
        DataFrame: A DataFrame with processed feature data including smoothing and outlier handling.
    """

    # Initialize holder variables for current data
    ear = 0.0
    mar = 0.0
    yaw = 0.0
    pitch = 0.0
    roll = 0.0

    # Initialize holder variables for past data
    prev_ear = np.nan
    prev_mar = np.nan
    prev_yaw = np.nan
    prev_pitch = np.nan
    prev_roll = np.nan

    # Initialize list to store processed data
    processed_data = []

    # Check if the first row has any '999.999' values in EAR, MAR, Yaw, Pitch, or Roll columns
    while df.iloc[0][['EAR', 'MAR', 'Yaw', 'Pitch', 'Roll']].astype(str).str.contains('999.999').any():
        # Remove first row
        df = df.iloc[1:, :]

    # Check if the first row contains -90 or 90 in Yaw, Pitch, or Roll columns
    while df.iloc[0][['Yaw', 'Pitch', 'Roll']].isin([-90, 90]).any():
        # Remove first row
        df = df.iloc[1:, :]

    # Skip the first `skip_rows` to allow for stabilization
    df = df.iloc[unstable_frames:, :].reset_index(drop=True)

    # Forward fill NaN values with the previous non-NaN value
    df = df.ffill()

    # Temporarily ignore warnings within this function
    with warnings.catch_warnings():
        # Set filter to ignore warnings
        warnings.filterwarnings("ignore")

        # Process each feature data with EMA and Outlier thresholds
        for _, row in df.iterrows():
            # Obtain the current feature data
            ear = row['EAR']
            mar = row['MAR']
            yaw = row['Yaw']
            pitch = row['Pitch']
            roll = row['Roll']

            # Check the difference of current and previous data
            ear_diff = abs(ear - prev_ear)
            mar_diff = abs(mar - prev_mar)
            yaw_diff = abs(yaw - prev_yaw)
            pitch_diff = abs(pitch - prev_pitch)
            roll_diff = abs(roll - prev_roll)

            # Outlier threshold for ear data
            if ear_diff >= ear_diff_threshold:
                # Use the previous value
                ear = prev_ear

            # Outlier threshold for mar data
            if mar_diff >= mar_diff_threshold:
                # Use the previous value
                mar = prev_mar

            # Outlier threshold for yaw data
            if yaw_diff >= yaw_diff_threshold:
                # Use the previous value
                yaw = prev_yaw

            # Outlier threshold for pitch data
            if pitch_diff >= pitch_diff_threshold:
                # Use the previous value
                pitch = prev_pitch

            # Outlier threshold for roll data
            if roll_diff >= roll_diff_threshold:
                # Use the previous value
                roll = prev_roll

            # Apply EMA smoothing for each feature
            # Check if previous ear data is nan (initial frame)
            if np.isnan(prev_ear):
                # When processing initial frame
                prev_ear = ear

            else:
                # When processing all other frame data
                ear = alpha_ear * ear + (1 - alpha_ear) * prev_ear
                prev_ear = ear

            # Check if previous mar data is nan (initial frame)
            if np.isnan(prev_mar):
                # When processing initial frame
                prev_mar = mar

            else:
                # When processing all other frame data
                mar = alpha_mar * mar + (1 - alpha_mar) * prev_mar
                prev_mar = mar

            # Check if previous yaw data is nan (initial frame)
            if np.isnan(prev_yaw):
                # When processing initial frame
                prev_yaw = yaw

            else:
                # When processing all other frame data
                yaw = alpha_yaw * yaw + (1 - alpha_yaw) * prev_yaw
                prev_yaw = yaw

            # Check if previous pitch data is nan (initial frame)
            if np.isnan(prev_pitch):
                # When processing initial frame
                prev_pitch = pitch

            else:
                # When processing all other frame data
                pitch = alpha_pitch * pitch + (1 - alpha_pitch) * prev_pitch
                prev_pitch = pitch

            # Check if previous roll data is nan (initial frame)
            if np.isnan(prev_roll):
                # When processing initial frame
                prev_roll = roll

            else:
                # When processing all other frame data
                roll = alpha_roll * roll + (1 - alpha_roll) * prev_roll
                prev_roll = roll

            # Cap the feature values
            ear = max(0, min(2.5, ear))
            mar = max(0, min(5, mar))
            yaw = max(-90, min(90, yaw))
            pitch = max(-90, min(90, pitch))
            roll = max(-90, min(90, roll))

            # Append processed values to the list       
            processed_data.append([row['Frame'], row['Drowsiness'], ear, mar, yaw, pitch, roll])

    # Convert the list of processed values to a DataFrame
    df_processed = pd.DataFrame(processed_data, columns=['Frame', 'Drowsiness', 'EAR', 'MAR', 'Yaw', 'Pitch', 'Roll'])

    return df_processed

# Helper function to obtain, append, and plot the video data statistics before processing
def process_statistics(df_processed, video_id, participant_id, initial_average_frames, participant_statistics, directory_path, after_title):
    """
    Process statistics before or after preprocessing step and update participant statistics.

    Parameters:
        df_processed (DataFrame): Processed DataFrame before or after preprocessing.
        video_id (str): Identifier for the video.
        participant_id (str): Identifier for the participant.
        initial_average_frames (int): Initial average frame count.
        features (list): List of features to extract statistics for.
        participant_statistics (dict): Participant statistics before or after preprocessing.
        directory_path (str): Directory path for saving processed data.
        after_title (bool): A flag indicating whether the statistics are calculated after preprocessing.

    Returns:
        None
    """

    # Get video data statistics
    video_stats = get_video_statistics(df_processed, initial_average_frames, after_title)
    video_stats['video_id'] = video_id

    # Dictionary to store statistics
    stats = {}

    # Extract relevant statistics for each feature
    for feature in features:
        # Iterate through each stat names
        for stat_name in stat_names:
            # Store obtained statistical data from video
            stats[f"{feature}_{stat_name}"] = video_stats[f"{feature}_{stat_name}"]

    # Plot the box plot for each feature of the video
    save_video_boxplot(df_processed, video_id, directory_path, stats, after_title)

    # Organize statistics by participant and feature
    if participant_id not in participant_statistics:
        # Create new dictionary for participant id
        participant_statistics[participant_id] = {}

    # Update participant statistics
    for feature in features:
        # Append participant statistic dictionary
        participant_statistics[participant_id].setdefault(feature, []).append({
            'Video Id': video_id,
            'Initial avg': video_stats[f"{feature}_Initial Avg."],
            'Video avg': video_stats[f"{feature}_Video Avg."],
            'Percentile_100 (Max)': video_stats[f"{feature}_Percentile_100 (Max)"],
            'Percentile_75': video_stats[f"{feature}_Percentile_75"],
            'Percentile_50': video_stats[f"{feature}_Percentile_50"],
            'Percentile_25': video_stats[f"{feature}_Percentile_25"],
            'Percentile_0 (Min)': video_stats[f"{feature}_Percentile_0 (Min)"],
            'Data': df_processed
        })

##############################################################################################
#                                                                                            #
#                                DATA STATISTICS SECTION                                     #
#                                                                                            #
##############################################################################################

# Function to calculate each video's statistics
def get_video_statistics(df_processed, initial_avg_frames, after_title):
    """
    Calculate statistics for each video based on the processed DataFrame.

    Parameters:
        df_processed (DataFrame): The processed DataFrame containing features for each frame.
        initial_avg_frames (int): The number of initial frames used to calculate the initial average.
        after_title (bool): A flag indicating whether the statistics are calculated after preprocessing.

    Returns:
        dict: A dictionary containing statistics for each feature, including initial average,
        video average, maximum, 75th percentile, median, 25th percentile, and minimum values.
    """

    # Initialize stat dictionary
    stats = {}

    # Iterate through each feature and calculate corresponding statistics
    for feature in features:
        # Iterate through stat names
        for stat_name in stat_names:
            # Initialize value 
            value = None

            # Obtain value based on stat name
            if stat_name == 'Initial Avg.':
                # Calculates the average of the initial frames specified by initial_avg_frames
                value = df_processed[feature].iloc[:initial_avg_frames].mean()

            elif stat_name == 'Video Avg.':
                # Calculates the average of the feature for the entire video
                value = df_processed[feature].mean()

            elif stat_name == 'Percentile_100 (Max)':
                # Finds the maximum value of the feature
                value = df_processed[feature].max()

            elif stat_name == 'Percentile_75':
                # Finds the 75th percentile value of the feature
                value = df_processed[feature].quantile(0.75)

            elif stat_name == 'Percentile_50':
                # Finds the median (50th percentile) value of the feature
                value = df_processed[feature].quantile(0.50)  # Median (50th percentile)

            elif stat_name == 'Percentile_25':
                # Finds the 25th percentile value of the feature
                value = df_processed[feature].quantile(0.25)

            elif stat_name == 'Percentile_0 (Min)':
                # Finds the minimum value of the feature
                value = df_processed[feature].min()

            # Store value to stats dictionary 
            stats[f"{feature}_{stat_name}"] = value
        
    # Print the header indicating whether statistics are calculated before or after processing
    if after_title:
        # Print the after header
        print("Video Statistics (After):".ljust(47), end="")

    else:
        # Print the before header
        print("Video Statistics (Before):".ljust(47), end="")

    # Print the names of the statistics
    for stat_name in stat_names:
        # Printed like header in a table
        print(stat_name.ljust(30), end="")

    # Just a spacer
    print()

    # Print the statistics values for each feature
    for feature in features:
        # Print the column 1 or the feature
        print(feature.ljust(47), end="")

        # Iterate through the stat names for succeeding columns
        for stat_name in stat_names:
            # Get the stored value in the stats dictionary
            value = stats.get(f"{feature}_{stat_name}", "N/A")

            # Check if its a float data
            if isinstance(value, float):
                # Print the data as float
                print(f"{value:.4f}".ljust(30), end="")

            else:
                # Print the data as string
                print(str(value).ljust(30), end="")
        print()

    return stats

# Function to calculate difference of mean and initial average
def calculate_abs_diff(df_csv, initial_average_frames, abs_diff_dict):
    """
    Calculate the absolute differences between the initial average and the total mean for each feature.

    Parameters:
        df_csv (DataFrame): The DataFrame containing the data.
        initial_average_frames (int): Number of initial frames to consider for the average.
        abs_diff_dict (dict): Dictionary to store absolute differences for each feature.

    Returns:
        None
    """

    # Calculate initial average for the first `initial_average_frame` frames
    initial_averages = df_csv[features].iloc[:initial_average_frames].mean()

    # Calculate the total mean for each feature
    total_means = df_csv[features].mean()

    # Calculate the absolute difference
    abs_diff = abs(initial_averages - total_means)

    # Append absolute differences to the dictionary
    for feature, diff in zip(features, abs_diff):
        # Append the absolute difference dictionary
        abs_diff_dict[feature].append(diff)
    
    # Print the results in a single line
    print(f"Abs(Video Average - Initial Average)", end=' | ')

    # Iterate through the features and abs diff dictionary
    for feature, diff in zip(features, abs_diff):
        # Print the feature and the corresponding computed absolute difference
        print(f"{feature}: {diff:.4f}", end=' | ')

    # Just a spacer    
    print()

##############################################################################################
#                                                                                            #
#                                   VISUALIZATION SECTION                                    #
#                                                                                            #
##############################################################################################

# Function to plot Boxplots for each features for each video
def save_video_boxplot(df_processed, video_id, output_dir, initial_avg_stats, after_title):
    """
    Generate and save a box plot for a specific video's features.

    Parameters:
        df_processed (DataFrame): The processed DataFrame containing features for each frame.
        video_id (str): The ID of the video.
        output_dir (str): The directory where the plot will be saved.
        initial_avg_stats (dict): A dictionary containing initial average statistics for each feature.
        after_title (bool): A flag indicating whether the statistics are plotted after preprocessing.

    Returns:
        None
    """

    # Create a figure with 1 row and 5 columns for the subplots
    fig, axs = plt.subplots(1, len(features), figsize=(20, 10))

    # Loop through each feature and create a subplot for each
    for col, feature in enumerate(features):
        # Create the subplot row 0 column col
        ax = axs[col]

        # Obtain the data statistics
        q0 = initial_avg_stats[f"{feature}_Percentile_0 (Min)"]
        q25 = initial_avg_stats[f"{feature}_Percentile_25"]
        q50 = initial_avg_stats[f"{feature}_Percentile_50"]
        q75 = initial_avg_stats[f"{feature}_Percentile_75"]
        q100 = initial_avg_stats[f"{feature}_Percentile_100 (Max)"]
        initial_avg = initial_avg_stats[f"{feature}_Initial Avg."]
        video_avg = initial_avg_stats[f"{feature}_Video Avg."]
        
        # Set limits based on feature
        if feature in ['EAR']:
            # Set EAR ylimit from 0 to 2.5
            ax.set_ylim(0, 2.5)

        elif feature in ['MAR']:
            # Set MAR ylimit from 0 to 5
            ax.set_ylim(0, 5)

        elif feature in ['Yaw', 'Pitch', 'Roll']:
            # Set Yaw, Pitch, and Roll ylimit from -90 to 90
            ax.set_ylim(-90, 90)

        # Set custom yticks based on the feature
        if feature == 'EAR':
            # Set EAR custom yticks
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5])

        elif feature == 'MAR':
            # Set MAR custom yticks
            ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])

        elif feature in ['Yaw', 'Pitch', 'Roll']:
            # Set Yaw, Pitch, and Roll custom yticks
            ax.set_yticks([-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90])

        # Plot the box plot with filled boxes
        ax.boxplot(df_processed[feature], positions=[0], widths=0.5, patch_artist=True, boxprops=dict(facecolor='lightgray'))

        # Plot the 100, 75, 50, 25, and 0 quantiles as horizontal lines
        ax.plot([-0.2, 0.2], [q100, q100], color='blue', linestyle='-', linewidth=2)
        ax.plot([-0.2, 0.2], [q75, q75], color='blue', linestyle='-', linewidth=2)
        ax.plot([-0.2, 0.2], [q25, q25], color='blue', linestyle='-', linewidth=2)
        ax.plot([-0.2, 0.2], [q0, q0], color='blue', linestyle='-', linewidth=2)

        # Plot median line with legend
        ax.plot([-0.25, 0.25], [q50, q50], color='red', linestyle='-', linewidth=2, label='Median')
        
        # Plot video average and initial average as lines
        ax.plot([-0.3, 0.3], [video_avg, video_avg], color='purple', linestyle='-', linewidth=2, label='Video Avg.')
        ax.plot([-0.3, 0.3], [initial_avg, initial_avg], color='green', linestyle='-', linewidth=2, label='Initial Avg.')

        # Set labels and title
        ax.set_xlabel('Feature Value')
        ax.set_ylabel('Values')
        ax.set_title(f'{feature}')

        # Hide spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Remove ticks for horizontal axis
        ax.xaxis.set_ticks([])

        # Set legend in the left bottom corner
        ax.legend(loc='lower left')
        
        # Display all the relevant statistics as text on the right side of the subplot
        stats_text = (
            f"Q100: {q100:.2f}\n"
            f"Q75: {q75:.2f}\n"
            f"Q50: {q50:.2f}\n"
            f"Q25: {q25:.2f}\n"
            f"Q0: {q0:.2f}\n"
            f"I Avg: {initial_avg:.2f}\n"
            f"V Avg: {video_avg:.2f}" 
        )
        ax.text(1.05, 0.5, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5))


    # Add a common x-axis label
    fig.text(0.5, 0.02, 'Feature Value', ha='center')

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85) 

    # Check if the title should be before or after processing
    if after_title is True:
        # Set title for after processing
        plt.suptitle(f"Boxplots for Each feature of Video After Processing: {video_id}", fontsize=24) 

    else:
        # Set title for before processing
        plt.suptitle(f"Boxplots for Each feature of Video Before Processing: {video_id}", fontsize=24)  

    # Save the plot figure as an image
    plt.savefig(os.path.join(output_dir, f"{video_id}_boxplot.png"))

    # Close the plot
    plt.close()

# Function to plot Boxplots for each features for each video for each participants
def save_participant_boxplots(participant_id, participant_data, output_dir, after_title):
    """
    Generate and save box plots for a participant's data.

    Parameters:
        participant_id (str): The ID of the participant.
        participant_data (dict): A dictionary containing data statistics for each feature.
                                 Keys are feature names, and values are lists of dictionaries,
                                 where each dictionary contains statistics for a video.
        output_dir (str): The directory where the plots will be saved.
        after_title (bool): A flag indicating whether the statistics are plotted after preprocessing.

    Returns:
        None
    """

    # Create a figure with layout based on the number of features and video of the participant
    # 5 features + 1 for the header
    num_rows = 6  

    # 4, 6, or 8 videos + 1 for the header
    num_cols = min(len(participant_data[next(iter(participant_data))]), 8) + 1  

    # fig and axs based on num rows and num cols
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 40))

    # Add the top-left header
    ax = axs[0, 0]

    # Set axis or borders to hide
    ax.axis('off')

    # Set text display
    ax.text(0.5, 0.5, "FEATURES", fontsize=32, ha='center', va='center')

    ## Add the column headers (video IDs)
    for col in range(1, num_cols):
        # Get the video id from the participant data
        video_id = participant_data[next(iter(participant_data))][col - 1]['Video Id']
        
        # Wrapped text the video id in the subplot area
        wrapped_text = textwrap.fill(video_id, width=10)  # Adjust width as needed
        
        # Create the subplot of row 0 column col
        ax = axs[0, col]

        # Set the axis off or borders as hidden
        ax.axis('off')

        # Set the text parameters
        ax.text(0.5, 0.5, wrapped_text, fontsize=24, ha='center', va='center')

    # Add the row headers (features)
    for row, feature in enumerate(participant_data.keys(), start=1):
        # Create the subplot of row row and column 0
        ax = axs[row, 0]

        # Set the axis off or borders as hidden
        ax.axis('off')

        # Set the text parameters
        ax.text(0.5, 0.5, feature, fontsize=32, ha='center', va='center')

    # Initialize feature limit dictionary
    feature_limits = {}

    # Iterate through the participant data to obtain limits of each features
    for feature, stats_list in participant_data.items():
        # Initialize all data list
        all_data = []

        # Iterate throught the stat lists of participant data
        for stat in stats_list:
            # Append all data list with the stat data of the current feature
            all_data.extend(stat['Data'][feature])
        
        # Calculate min and max values of each feature and store it
        feature_limits[feature] = (min(all_data), max(all_data))

    # Loop through each feature and video
    for row, (feature, stats_list) in tqdm(enumerate(participant_data.items(), start=1), total=len(participant_data), desc='Features', leave=False, position=1):
        # Get the min and max values for this feature
        y_min, y_max = feature_limits[feature]

        # set the y_ticks with min, max, and number of ticks
        y_ticks = np.linspace(y_min, y_max, num=7)

        # Add a small margin to the y-axis limits if they are too close or identical
        if y_min == y_max:
            # Adjust this value for the small margin
            margin = 1e-5

            # Apply margins to the min and max  
            y_min -= margin
            y_max += margin

        # Loop through the actual statistical data
        for col, stat in tqdm(enumerate(stats_list, start=1), total=len(stats_list), desc='Videos', leave=False, position=2):
            # Skip the first row and first column for actual plots by using start = 1
            # Create the subplot row and col
            ax = axs[row, col]

            # Extract the video data statistics
            initial_avg = stat['Initial avg']
            video_avg = stat['Video avg']
            q0 = stat['Percentile_0 (Min)']
            q25 = stat['Percentile_25']
            q50 = stat['Percentile_50']
            q75 = stat['Percentile_75']
            q100 = stat['Percentile_100 (Max)']
            data = stat['Data'][feature].tolist()

            # Set y-axis limits and ticks dynamically
            ax.set_ylim(y_min, y_max)
            ax.set_yticks(y_ticks)

            # Plot the box plot with filled boxes
            ax.boxplot([data], positions=[0], widths=0.5, patch_artist=True, boxprops=dict(facecolor='lightgray'))

            # Plot the 100, 75, 50, 25, and 0 quantiles as horizontal lines
            ax.plot([-0.2, 0.2], [q100, q100], color='blue', linestyle='-', linewidth=2)
            ax.plot([-0.2, 0.2], [q75, q75], color='blue', linestyle='-', linewidth=2)
            ax.plot([-0.2, 0.2], [q25, q25], color='blue', linestyle='-', linewidth=2)
            ax.plot([-0.2, 0.2], [q0, q0], color='blue', linestyle='-', linewidth=2)

            # Plot median line with legend
            ax.plot([-0.25, 0.25], [q50, q50], color='red', linestyle='-', linewidth=2, label='Median')

            # Plot video average and initial average as lines
            ax.plot([-0.3, 0.3], [video_avg, video_avg], color='purple', linestyle='-', linewidth=2, label='Video Avg.')
            ax.plot([-0.3, 0.3], [initial_avg, initial_avg], color='green', linestyle='-', linewidth=2, label='Initial Avg.')

            # Set labels and title
            ax.set_xlabel('')  # No x-label for individual subplots
            ax.set_ylabel('Values')

            # Configure axis and ticks for each subplot
            ax.tick_params(axis='both', which='both', labelsize=12) 
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            
    # Add legend outside the subplots
    handles, labels = ax.get_legend_handles_labels()

    # Set legend settings
    fig.legend(handles, labels, loc='lower center', ncols=20, fontsize=32)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)
    plt.subplots_adjust(top=0.95) 
    plt.subplots_adjust(right=0.975) 
    
    # Check if the title should be before or after processing
    if after_title is True:
        # Add title for after processing
        plt.suptitle(f"Boxplots for Each Feature of All Video of Participant {participant_id} After Processing", fontsize=32)

    else:
        # Add title for before processing
        plt.suptitle(f"Boxplots for Each feature of All Video of Participant {participant_id} Before Processing", fontsize=32)  

    # Save the plot figure as an image
    plt.savefig(os.path.join(output_dir, f"Participant {participant_id} Boxplots"))

    # Close the plot
    plt.close()

# Function to save participant data statistics to csv file
def write_participant_csv(participant_id, feature_stats, directory_path):
    """
    Write participant statistics to CSV files.

    Parameters:
        participant_id (str): Participant ID.
        feature_stats (dict): Dictionary containing feature statistics.
        directory_path (str): Path to the directory where CSV files will be saved.

    Returns:
        None
    """

    # Create individual csv for each participant
    participant_csv_path = os.path.join(directory_path, f"{participant_id}_statistics.csv")

    # Open the csv for editing
    with open(participant_csv_path, 'w', encoding='utf-8') as f:
        # Write the CSV header
        f.write("Feature,Video Id,Initial avg,Video avg,Percentile_100 (Max),Percentile_75,Percentile_50,Percentile_25,Percentile_0 (Min)\n")

        # Loop through each feature and its statistics
        for feature, stats_list in tqdm(feature_stats.items(), desc=f'Participant {participant_id}', ncols=150, leave=False, position=1):
            # Loop through each video's statistics
            for stats in stats_list:
                # Truncate values to specified column width and format numerical values
                feature_name = feature[:column_width]
                video_id = stats['Video Id'][:column_width]
                initial_avg = f"{stats['Initial avg']:.4f}".ljust(column_width)[:column_width]
                video_avg = f"{stats['Video avg']:.4f}".ljust(column_width)[:column_width]
                percentile_100 = f"{stats['Percentile_100 (Max)']:.4f}".ljust(column_width)[:column_width]
                percentile_75 = f"{stats['Percentile_75']:.4f}".ljust(column_width)[:column_width]
                percentile_50 = f"{stats['Percentile_50']:.4f}".ljust(column_width)[:column_width]
                percentile_25 = f"{stats['Percentile_25']:.4f}".ljust(column_width)[:column_width]
                percentile_0 = f"{stats['Percentile_0 (Min)']:.4f}".ljust(column_width)[:column_width]

                # Write the statistics to the CSV file
                f.write(f"{feature_name},{video_id},{initial_avg},{video_avg},{percentile_100},{percentile_75},{percentile_50},{percentile_25},{percentile_0}\n")

            # Add a blank row between features
            f.write("\n")

##############################################################################################
#                                                                                            #
#                                       MAIN SECTION                                         #
#                                                                                            #
##############################################################################################

# Define the main function
def main():
    """
    Main function to process datasets, calculate statistics, and save results. This function 
    orchestrates the preprocessing of data, including feature extraction, outlier handling, and 
    statistical analysis. It iterates through dataset folders, processes each video, calculates 
    participant statistics, and saves the results to CSV files and plots boxplots for visualization.

    Parameters:
        None

    Returns:
        None
    """

    # Record start time
    start_time = time.time()

    # Check if the path exists
    if os.path.exists(before_processing_path):
        # Delete current folder
        shutil.rmtree(before_processing_path)

    # Create the folder path
    os.makedirs(before_processing_path)

    # Check if the path exists
    if os.path.exists(after_processing_path):
        # Delete current folder
        shutil.rmtree(after_processing_path)

    # Create the folder path
    os.makedirs(after_processing_path)

    # Initialize a dictionary to store participant statistics by feature
    participant_statistics_before = {}
    participant_statistics_after = {}

    # Initialize a dictionary to store absolute differences for each feature
    abs_diff_dict_before = {feature: [] for feature in features}
    abs_diff_dict_after = {feature: [] for feature in features}

    # Run the functions for all datasets
    for dataset_path in all_dataset_paths:
        # Process each dataset folder path
        for folder_path in dataset_path:
            # Print the current folder path being processed
            print(f"\nProcessing {folder_path}")
            
            # Determine the initial average frame count based on the folder path
            if dataset_path == training_folder_paths:
                # Determine if the folder path contains the word night
                if 'night' in folder_path:                    
                    # Actual fps of the video
                    fps = 15

                    # Set the initial average frame as initial frame
                    initial_average_frame = initial_frame

                    # Set the number of unstable frames
                    unstable_frames = 15

                else:                    
                    # Actual fps of the video
                    fps = 30

                    # Set the initial average frame multiplied by 2 (based on fps ratio) as the initial frame
                    initial_average_frame = initial_frame * 2

                    # Set the number of unstable frames
                    unstable_frames = 30

            else:               
                # Actual fps of the video
                fps = 30

                # Set the initial average frame multiplied by 2 (based on fps ratio) as the initial frame
                initial_average_frame = initial_frame * 2

                # Set the number of unstable frames
                unstable_frames = 30

            # Iterate over each video and its corresponding annotation file
            csv_files = [file for file in os.listdir(folder_path) if file.endswith((".csv"))]

            # Process each video together with its respective annotation file
            for csv_file in csv_files:
                # Extract video name without extension
                video_id = os.path.splitext(csv_file)[0]

                # Print the current video id being processed as well as the fps andthe initial frame count used
                print(f"\nVideo: {video_id.ljust(60)}fps: {str(fps).ljust(20)}Initial Frames: {str(initial_average_frame).ljust(10)}")
                      
                # Extract participant ID from video ID
                participant_id = video_id.split('_')[0]

                # Extract csv data to a df
                df_csv = pd.read_csv(os.path.join(folder_path, csv_file))

                # Call the function for "before" preprocessing
                process_statistics(df_csv, 
                                   video_id, 
                                   participant_id, 
                                   initial_average_frame, 
                                   participant_statistics_before, 
                                   before_processing_path,
                                   after_title = False)

                calculate_abs_diff(df_csv, initial_average_frame, abs_diff_dict_before)

                # Apply preprocessing to the loaded data frame
                df_processed = preprocess_features(df_csv, unstable_frames)

                # Call the function for "after" preprocessing
                process_statistics(df_processed, 
                                   video_id, 
                                   participant_id, 
                                   initial_average_frame, 
                                   participant_statistics_after, 
                                   after_processing_path,
                                   after_title = True)
                
                calculate_abs_diff(df_processed, initial_average_frame, abs_diff_dict_after)
    
    # Calculate the total average for each feature
    total_avg_dict_before = {feature: np.mean(abs_diff_dict_before[feature]) for feature in features}
    total_avg_dict_after = {feature: np.mean(abs_diff_dict_after[feature]) for feature in features}

    # Print a header for total average data
    print("\nTotal Average Absolute Differences per Feature Before Processing:")

    # Iterate through the total average dictionary for before processing
    for feature, avg in total_avg_dict_before.items():
        # Print the total average absolute differences for each feature before processing
        print(f"{feature}: {avg:.4f}")

    # Print a header for total average data
    print("\nTotal Average Absolute Differences per Feature After Processing:")

    # Iterate through the total average dictionary for after processing
    for feature, avg in total_avg_dict_after.items():
        # Print the total average absolute differences for each feature after processing
        print(f"{feature}: {avg:.4f}")
        
    # Save the before statistics data to CSV files and plot the participant boxplots
    for participant_id, feature_stats in tqdm(participant_statistics_before.items(), desc='Creating CSV Files and Box Plotting for Before Processing Data ...', ncols=150, leave=False):
        # Plot the box plot for each feature of each video with the same participant
        save_participant_boxplots(participant_id, feature_stats, before_processing_path, after_title = False)

        # Call the write_participant_csv() function to write participant statistics to CSV files
        write_participant_csv(participant_id, feature_stats, before_processing_path)

    # Save the after statistics data to CSV files and plot the participant boxplots
    for participant_id, feature_stats in tqdm(participant_statistics_after.items(), desc='Creating CSV Files and Box Plotting for AFter Processing Data...', ncols=150, leave=False):
        # Plot the box plot for each feature of each video with the same participant
        save_participant_boxplots(participant_id, feature_stats, after_processing_path, after_title = True)

        # Call the write_participant_csv() function to write participant statistics to CSV files
        write_participant_csv(participant_id, feature_stats, after_processing_path)

    # Calculate processing time duration
    stop_time = time.time()
    duration_seconds = stop_time - start_time
    duration_readable = time.strftime("%H:%M:%S", time.gmtime(duration_seconds))
    print(f"\nTotal Processing Time: {duration_readable}")

# Check if the script is being run directly
if __name__ == "__main__":   
    # Call the main function
    main()

##############################################################################################
#                                                                                            #
#                                        END SECTION                                         #
#                                                                                            #
##############################################################################################