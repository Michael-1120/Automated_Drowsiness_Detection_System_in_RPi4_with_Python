"""
This program contains functions to load, process, and prepare training data, train LSTM models, calculate validation metrics, 
and save the trained models. It is designed for training LSTM models for video sequence classification based on specified data. 
The main function iteratively retrains the models on the training dataset. 

The program consists of the following sections:

1. Import/Initialization Section
    - Import necessary python libraries
    - Define folder paths
    - Define global variables

2. Data Loading Section
    - Function 'load_and_process_data': Loads the pre-processed csv with computed features data and call the process_features function.
    - Function 'preprocess_features': Processes the dataframes with thresholding, EMA smoothing, removing irrelevant initial frames, and normalizing the dataset.
    - Function 'debug_video_data': Display the processed and splitted dataframes details.

3. Model Section
    - Function 'lr_schedule': Schedules the learning rate that will be applied for each epoch iteration.
    - Class 'LRSchedule': Defines the class for scheduling learning rate at initialization and on epoch begin.
    - Function 'create_custom_lstm_model': Creates and compile a custom LSTM Model. 

4. Training Data Preparation Section:
    - Function `prepare_training_data`: Prepares training data for LSTM model training by creating sequences and labels from input DataFrame containing video sequences.
    - Function 'split_and_shuffle_video_ids': Ensures that the training and validation video ids for both category are not repeated.
    - Function `extract_data`: Extracts and combines sequence data and labels for a list of video IDs.

5. Model Metric Section:
    - Function `compute_metrics`: Compute evaluation metrics based on validation data.

6. Model Saving Section:
    - Function `save_and_print_model_shapes`: Saves trained models in the specified directory and prints their input and output shapes.
    - Function `print_model_shapes`: Prints the input and output shapes of a given Keras model.

7. Main Section:
    - Function `main`: Main function to process and calls the necessary function for the flow of training LSTM Drowsiness Detection Model. 

The program is designed to be run as a standalone module, and the main function `main` is called when the script is executed directly. 
It utilizes various parameters and configurations defined within the script to control model training, validation, and saving.
"""

##############################################################################################
#                                                                                            #
#                               IMPORT/INITIALIZATION SECTION                                #
#                                                                                            #
##############################################################################################

import os
import time
import shutil
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.regularizers import l1, l2
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense, Dropout

# List of folder paths
training_folder_paths = [
    r"D:\Processed Combined NTHU Training Dataset\glasses",
    r"D:\Processed Combined NTHU Training Dataset\noglasses",
    r"D:\Processed Combined NTHU Training Dataset\nightglasses",
    r"D:\Processed Combined NTHU Training Dataset\night_noglasses"
]

# Set random seed for reproducibility
np.random.seed(42)

# Define initial frame for initial average
initial_frame = 10 # From Data Analysis

# Define the unstable frames for processing
unstable_frames = 10 # For Mediapipe initial instability

# Define the feature and label columns
feature_columns = ['EAR', 'MAR', 'Yaw', 'Pitch', 'Roll']
selected_features = ['EAR', 'MAR', 'Yaw', 'Pitch', 'Roll']
label_column = 'Drowsiness'

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

# Define max sequence length and number of features
max_sequence_length = 100

# Define regularization strengths
l1_strength = 0.01
l2_strength = 0.01
dropout_ratio = 0.8

# Define batch size for model training
batch_size = 32

# Define a set for the used combination in training iteration
used_combinations = set()

# Set training iteration count
training_iterations = 100

# Define the initial learning rate
initial_lr = 0.0001

# Define the number of IDs for each split
# 100% of the category dataset = 71 video ids
train_count = 0.8       # 80% of the category dataset  = 56 video ids
                        # 20% left of the category dataset = 15 video ids

# Initialize patience for the number of epoch to wait for improvement
patience = 5

# Testing Weights
prediction_threshold = 0.5

# Define Final Model Path
final_model_path = r"D:\Trained Model"

# Define the checkpoint directory
checkpoint_dir = r"D:\Checkpoints"

# Remove the directory if it exists and recreate it
if os.path.exists(checkpoint_dir):
    shutil.rmtree(checkpoint_dir)  # Remove the directory and its contents
os.makedirs(checkpoint_dir)  # Recreate the directory

# Initialize ModelCheckpoint callbacks for saving the best model
checkpoint_sleepy = ModelCheckpoint(
    os.path.join(checkpoint_dir, 'best_sleepy_model.tf'),  # Path for the sleepy model checkpoint
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    mode='min'
)

checkpoint_nonsleepy = ModelCheckpoint(
    os.path.join(checkpoint_dir, 'best_nonsleepy_model.tf'),  # Path for the nonsleepy model checkpoint
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    mode='min'
)

##############################################################################################
#                                                                                            #
#                                   DATA LOADING SECTION                                     #
#                                                                                            #
##############################################################################################

# Function to load CSV data with specified filters
def load_and_process_data(main_folder_path):
    """
    Loads and processes CSV data from specified folders.

    Parameters:
        main_folder_path (list): A list of folder paths containing CSV files.

    Returns:
        pd.DataFrame: A DataFrame containing the processed data from CSV files.
    """
    
    # Initialize a list for storing dataframes of data
    data_frames = []

    # Use tqdm to create a progress bar for the outer loop
    for sub_folder in tqdm(main_folder_path, desc="Folders", unit="folder", ncols=150, leave=False):
        # Iterate over each video and its corresponding annotation file
        csv_files = [file for file in os.listdir(sub_folder) if file.endswith((".csv"))]

        # Iterate through each csv files
        for csv_file in tqdm(csv_files, desc="CSV Files", unit="files", ncols=150, leave=False):
            # Get the complete csv file path
            csv_file_path = os.path.join(sub_folder, csv_file)

            # Read the CSV and store the data into the df
            df = pd.read_csv(csv_file_path)

            # Check if the file was is nightglasses or nightnoglasses
            if 'night' in csv_file.lower():
                # 3:2 downsampling from 15 fps to 10 fps
                df = df.iloc[2::3, :].reset_index(drop=True) 
                
            else:
                # 3:1 downsampling from 30 fps to 10 fps
                df = df.iloc[::3, :].reset_index(drop=True)  

            # Process the df 
            df_processed = preprocess_features(df, unstable_frames, initial_frame)

            # Add the full csv file path to the video id in processed df
            df_processed['Video_ID'] = os.path.join(sub_folder, csv_file)

            # Append the data frames with the processed df
            data_frames.append(df_processed)

    # Check if data frame is empty
    if not data_frames:
        # Print debugging line
        print("No CSV files found in the specified folders.")

    return pd.concat(data_frames, ignore_index=True)

# Function to process the data
def preprocess_features(df, skipped_frames, initial_frames_count):
    """
    Preprocesses the feature data in the given DataFrame.

    Parameters:
        df (DataFrame): The input DataFrame containing feature data.
        skipped_frames (int): The number of initial frames to skip for stabilization.
        initial_frames_count (int): The number of initial frames used for feature averaging.

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
    df = df.iloc[skipped_frames:, :].reset_index(drop=True)

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
            processed_data.append([row['Drowsiness'], ear, mar, yaw, pitch, roll])

    # Convert the list of processed values to a DataFrame
    df_processed = pd.DataFrame(processed_data, columns=['Drowsiness', 'EAR', 'MAR', 'Yaw', 'Pitch', 'Roll'])

    # Average of first initial_frames rows
    df_feature_averages = df_processed[feature_columns][:initial_frames_count].mean()

    # Subtract the average of each column for n rows from all the rows in the respective column
    df_processed[feature_columns] -= df_feature_averages

    # Create padding data of zeroes
    padding_data = [[0.0] * len(feature_columns)] * (max_sequence_length - 1)

    # Create padding dataframe of the same column as the df
    df_padding = pd.DataFrame(padding_data, columns=feature_columns)

    # Add Drowsiness column to the padding df
    df_padding.insert(0, 'Drowsiness', 0)

    # Combine padding DataFrame and processed DataFrame
    df_combined = pd.concat([df_padding, df_processed], ignore_index=True)

    # Reset index
    df_combined.reset_index(drop=True, inplace=True)    

    return df_combined

# Function to show data details for debugging
def debug_video_data(all_df, sleepy_df, nonsleepy_df, folder_paths):
    """
    Debugs video data by printing counts of drowsy and non-drowsy frames for each video,
    total number of training videos, and number of rows and videos with NaN values.

    Parameters:
        all_df (pd.DataFrame): DataFrame containing all video data.
        sleepy_df (pd.DataFrame): DataFrame containing video data for sleepy videos.
        nonsleepy_df (pd.DataFrame): DataFrame containing video data for non-sleepy videos.
        folder_paths (list): List of folder paths containing video files.

    Returns:
        None
    """

    # Iterate through each unique video id
    for id in all_df['Video_ID'].unique():
        # Obtain the df of the respective id
        df_data = all_df[all_df['Video_ID'] == id]

        # Get the full csv file path
        full_file_path = os.path.join(folder_paths[0], df_data['Video_ID'].iloc[0] + ".csv")
        
        # Count of "1" in "Drowsiness" column
        drowsy_counts = int(df_data['Drowsiness'].sum())

        # Get the non-dowsy count by subtracting the total 
        non_drowsy_counts = len(df_data) - drowsy_counts
        
        # Print debugging line
        print(f"File: {full_file_path.ljust(145)} Frames: {len(df_data):<10} Drowsy Count: {drowsy_counts:<10} Non-Drowsy Count: {non_drowsy_counts:<10}")

    # Get the unique video ids in the dataset
    ids = all_df['Video_ID'].unique()

    # Get the unique video ids in the nonsleepy category
    nonsleepy_ids = nonsleepy_df['Video_ID'].unique()

    # Get the unique video ids in the sleepy category
    sleepy_ids = sleepy_df['Video_ID'].unique()

    # Get the total number of ids
    num_vid = len(ids)
    num_nonsleepy = len(nonsleepy_ids)
    num_sleepy = len(sleepy_ids)
    
    # Print the total video in the dataset 
    print(f"\nTotal Training Video Dataset: {num_vid } ")

    # Check for NaN values
    nan_rows = all_df[all_df.isna().any(axis=1)]

    # Get the number of rows with nan values
    num_nan_rows = len(nan_rows)

    # Get the number of video with nan values
    num_videos_with_nan = len(all_df[all_df.isna().any(axis=1)]['Video_ID'].unique())

    # Print debugging line
    print(f"Number of rows with NaN values: {num_nan_rows}")
    print(f"Number of video IDs with at least one NaN value: {num_videos_with_nan}")
    
    # Print the train-test split of video data for the whole training dataset
    print(f"\nVideo Datasets: Total = {num_vid}   NonSleepy = {num_nonsleepy}   Sleepy = {num_sleepy}")

    # Get the number of training video id for nonsleepy category
    num_train_nonsleepy_vid = int(0.8 * num_nonsleepy)

    # Get the number of training video id for sleepy category
    num_train_sleepy_vid = int(0.8 * num_sleepy)

    # Print the number of video data for the training 
    print(f"Training Video: Total = {num_train_nonsleepy_vid + num_train_sleepy_vid}   NonSleepy = {num_train_nonsleepy_vid}   Sleepy = {num_train_sleepy_vid}")

    # Get the number of validation video id for nonsleepy category
    num_val_nonsleepy_vid = num_nonsleepy -  num_train_nonsleepy_vid

    # Get the number of validation video id for sleepy category
    num_val_sleepy_vid = num_sleepy -  num_train_sleepy_vid
    
    # Print the number of video data for the validation
    print(f"Validation Video: Total = {num_val_nonsleepy_vid + num_val_sleepy_vid}   NonSleepy = {num_val_nonsleepy_vid}   Sleepy = {num_val_sleepy_vid}")

    # Debug variables to compute counts of Drowsiness for the whole dataset
    print("\nClass Imbalance for the Total Dataset:")
    print(all_df['Drowsiness'].value_counts())
    
    # Print the value counts of drowsiness for the nonsleepy df
    print("\nClass Imbalance for nonsleepyCombination video:")
    print(nonsleepy_df['Drowsiness'].value_counts())
    
    # Print the value counts of drowsiness for the sleepy df
    print("\nClass Imbalance for sleepyCombination video:")
    print(sleepy_df['Drowsiness'].value_counts())

    # Print shapes of DataFrames
    print("\nShape of df_processed: ", all_df.shape)
    print("Shape of df_nonsleepy: ", nonsleepy_df.shape)
    print("Shape of df_sleepy: ", sleepy_df.shape)

##############################################################################################
#                                                                                            #
#                                      MODEL SECTION                                         #
#                                                                                            #
##############################################################################################

# Function to define learning rate schedule
def lr_schedule(iteration_count, starting_learning_rate):
    """
    Defines a learning rate schedule based on the iteration count.

    Parameters:
        iteration_count (int): The current iteration count.
        initial_learning_rate (float): The initial learning rate.

    Returns:
        float: The adjusted learning rate.
    """

    # Define learning rate of the current epoch based on the iteration count
    if iteration_count < 55:
        # Calculate the learning rate based on iteration_count
        multiplier = int(iteration_count // 5)

        # Calculate the learning rate
        learning_rate = starting_learning_rate * (0.8 ** multiplier)
    
    elif iteration_count >= 55:
        # Use the learning rate equal to half of the initial learning rate
        learning_rate = starting_learning_rate / 10

    return learning_rate
    
# Custom LearningRateScheduler callback
class LRScheduler(Callback):
    """
    A custom Keras callback to adjust learning rate dynamically during training.

    Parameters:
        initial_learning_rate (float): The initial learning rate.
    
    Returns:
        None
    """

    # Initialize the custom callback with the initial learning rate
    def __init__(self, initial_learning_rate, current_iteration):
        """
        Initializes the LearningRateScheduler with the initial learning rate.

        Parameters:
            initial_learning_rate (float): The initial learning rate.
            current_iteration (int): The current iteration count.

        Returns:
            None
        """

        # Call the parent class constructor and initialize the initial learning rate
        super(LRScheduler, self).__init__()

        # Initialize the initial learning rate attribute
        self.initial_learning_rate = initial_learning_rate

        # Initialize the current iteration attribute
        self.current_iteration = current_iteration

    # This method is called at the beginning of each epoch
    def on_epoch_begin(self, epoch, logs=None):
        """
        Updates the learning rate at the beginning of each epoch.

        Parameters:
            epoch (int): The current epoch number.
            logs (dict): Dictionary of logs containing the training metrics.

        Returns:
            None
        """

        # Calculate the current learning rate using the iteration count
        lr = lr_schedule(self.current_iteration + 1, self.initial_learning_rate)

        # Update the model's learning rate
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)

        # Print the current learning rate
        print(f"Iteration {self.current_iteration + 1}: Learning rate is {lr:.8f}")
        
# Function to create a custom LSTM model layers
def create_custom_lstm_model(name, input_shape, initial_learning_rate):
    """
    Creates a custom LSTM model with specified parameters.

    Parameters:
        name (str): The name of the model.
        input_shape (tuple): The shape of the input data.
        initial_learning_rate (float): The initial learning rate.

    Returns:
        tf.keras.Sequential: The compiled LSTM model.
    """

    # Create a sequential model with the input name
    model = tf.keras.Sequential(name=name)
    
    # Add LSTM layers with regularization
    model.add(LSTM(units=32, input_shape=input_shape, return_sequences=True, kernel_regularizer=l1(l1_strength), recurrent_regularizer=l1(l1_strength)))
    model.add(Dropout(dropout_ratio))
    
    model.add(LSTM(units=64, return_sequences=True, kernel_regularizer=l1(l1_strength), recurrent_regularizer=l1(l1_strength)))
    model.add(Dropout(dropout_ratio))

    model.add(LSTM(units=128, return_sequences=True, kernel_regularizer=l1(l1_strength), recurrent_regularizer=l1(l1_strength)))
    model.add(Dropout(dropout_ratio))

    model.add(LSTM(units=64, return_sequences=True, kernel_regularizer=l1(l1_strength), recurrent_regularizer=l1(l1_strength)))
    model.add(Dropout(dropout_ratio))
    
    model.add(LSTM(units=32, return_sequences=False, kernel_regularizer=l1(l1_strength), recurrent_regularizer=l1(l1_strength)))
    model.add(Dropout(dropout_ratio))

    # Output layer
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_strength)))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=initial_learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    # Print model name before showing the summary
    model.summary()
    
    return model

##############################################################################################
#                                                                                            #
#                            TRAINING DATA PREPARATION SECTION                               #
#                                                                                            #
##############################################################################################

# Function to prepare the training data
def prepare_training_data(df):
    """
    Prepares the training data for LSTM model training.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the video sequences.

    Returns:
        dict: A dictionary containing the prepared training data for each video sequence.
    """

    # Initialize a dictionary to store all sequence data
    all_data_dict = {}

    # Loop through unique video IDs in the DataFrame
    for video_name in tqdm(df['Video_ID'].unique(), desc="Training Preparation", unit=" Video sequences and labels", ncols=150, leave=False):
        # Filter DataFrame for the current video ID
        video_df = df[df['Video_ID'] == video_name].copy()

        # Find the length of the current df being processed
        num_samples = len(video_df)

        # Compute the number of sequences based on the number of samples and the max sequence length
        num_sequences = (num_samples - max_sequence_length) + 1
        
        # Initialize arrays to store sequences and labels
        x_data = np.zeros((num_sequences, max_sequence_length, len(selected_features)))
        y_data = np.zeros((num_sequences, 1))
        
        # Iterate through each sequence in the video
        for i in range(num_sequences):
            # Set starting df index based on iteration
            start_index = i

            # Set last df index based on starting index and max sequence length
            end_index = start_index + max_sequence_length
            
            # Extract sequence data based on starting and ending indices for the current df
            sequence = video_df[selected_features].iloc[start_index:end_index, :].values

            # Extract the label for that sequence
            label = video_df[label_column].iloc[end_index - 1]

            # Initializes a zero-filled array of shape (max_sequence_length, len(selected_features)) to represent the padded sequence
            padded_sequence = np.zeros((max_sequence_length, len(selected_features)))

            # Assigns the sequence data to the padded sequence array, ensuring it fits within the desired length
            padded_sequence[:max_sequence_length, :] = sequence

            # Stores the padded sequence data in the x_data array at index i
            x_data[i] = padded_sequence

            # Assigns the label to the y_data array at index i
            y_data[i] = label

        # Store prepared data in the dictionary
        all_data_dict[video_name] = (x_data, y_data)
    
    return all_data_dict

# Function to split and shuffle video IDs
def split_and_shuffle_video_ids(df_sleepy, df_nonsleepy):
    """
    Splits and shuffles video IDs for training, validation, and exchange sets.
    
    Parameters:
        df_sleepy (DataFrame): DataFrame containing sleepy video IDs.
        df_nonsleepy (DataFrame): DataFrame containing nonsleepy video IDs.
    
    Returns:
        tuple: A tuple containing combined_train_sleepy, val_sleepy_ids, combined_train_nonsleepy, and val_nonsleepy_ids.
    """

    # Obtain the unique video IDs of the df_sleepy
    sleepy_ids = list(df_sleepy['Video_ID'].unique())

    # Obtain the unique video IDs of the df_nonsleepy
    nonsleepy_ids = list(df_nonsleepy['Video_ID'].unique())

    # Loop until the split video IDs have never been used
    while True:
        # Randomly shuffle the unique sleepy IDs
        np.random.shuffle(sleepy_ids)

        # Randomly shuffle the unique nonsleepy IDs
        np.random.shuffle(nonsleepy_ids)

        # Split the unique sleepy IDs into train and validation sets
        train_sleepy_ids, val_sleepy_ids = train_test_split(sleepy_ids, train_size=train_count, random_state=42)

        # Split the unique nonsleepy IDs into train and validation sets
        train_nonsleepy_ids, val_nonsleepy_ids = train_test_split(nonsleepy_ids, train_size=train_count, random_state=42)

        # Check if the combined train IDs are not in used combinations
        if tuple(train_sleepy_ids) not in used_combinations and tuple(train_nonsleepy_ids) not in used_combinations:
            # Add the combined training sleepy IDs to the used combinations
            used_combinations.add(tuple(train_sleepy_ids))

            # Add the combined training nonsleepy IDs to the used combinations
            used_combinations.add(tuple(train_nonsleepy_ids))

            return train_sleepy_ids, val_sleepy_ids, train_nonsleepy_ids, val_nonsleepy_ids
        
# Function to extract data based on video ids
def extract_data(video_ids, data_dict, desc):
    """
    Extracts and combines sequence data and labels for a list of video IDs.

    Parameters:
        video_ids (list): List of video IDs to extract data for.
        data_dict (dict): Dictionary containing data sequences and labels for each video ID.
        desc (str): Description for the progress bar.

    Returns:
        x_data_array (numpy array): Combined sequence data from all specified videos.
        y_data_array (numpy array): Combined labels from all specified videos.
    """

    # Initialize lists to store x and y data
    x_data_list, y_data_list = [], []
    
    # Iterate thought the video ids for data extraction
    for _, video_id in enumerate(tqdm(video_ids, desc=desc, unit="video", ncols=150, leave=False)):
        # Load the sequences and labels for the current video
        x_data, y_data = data_dict[video_id]

        # Append the sequences and labels to the lists
        x_data_list.extend(x_data)
        y_data_list.extend(y_data)

    # Convert to numpy arrays
    x_data_array = np.array(x_data_list)
    y_data_array = np.array(y_data_list)

    return x_data_array, y_data_array

##############################################################################################
#                                                                                            #
#                                  MODEL METRIC SECTION                                      #
#                                                                                            #
##############################################################################################

# Function to compute metric after validation for each iteration
def compute_metrics(y_val_sleepy, y_val_sleepy_pred, y_val_nonsleepy, y_val_nonsleepy_pred, history_sleepy, history_nonsleepy, current_iteration):
    """
    Compute evaluation metrics based on validation data.

    Parameters:
        y_val_sleepy (array): True labels for sleepy class in validation data.
        y_val_sleepy_pred (array): Predicted labels for sleepy class in validation data.
        y_val_nonsleepy (array): True labels for nonsleepy class in validation data.
        y_val_nonsleepy_pred (array): Predicted labels for nonsleepy class in validation data.
        history_sleepy (History): History object containing training history for the sleepy class.
        history_nonsleepy (History): History object containing training history for the nonsleepy class.
        current_iteration (int): Current iteration/epoch number.

    Returns:
        mean_val_loss (float): Mean validation loss computed from the histories of both classes.
        accuracy (float): Accuracy computed from the confusion matrix
    """

    # Compute confusion matrix of the sleepy validation predictions and label
    cm_sleepy = confusion_matrix(y_val_sleepy, (y_val_sleepy_pred > prediction_threshold).astype(int))

    # Compute confusion matrix of the nonsleepy validation predictions and label
    cm_nonsleepy = confusion_matrix(y_val_nonsleepy, (y_val_nonsleepy_pred > prediction_threshold).astype(int))

    # Ensure the sleepy confusion matrix is always 2x2
    if cm_sleepy.size < 4:
        # Pad the confusion matrix with zeros to make it 2x2 if it has less than 4 elements
        cm_sleepy_padded = np.pad(cm_sleepy, ((0, max(0, 2 - cm_sleepy.shape[0])), (0, max(0, 2 - cm_sleepy.shape[1]))), mode='constant')

    else:
        # Use the original confusion matrix if it is already 2x2
        cm_sleepy_padded = cm_sleepy

    # Ensure the nonsleepy confusion matrix is always 2x2
    if cm_nonsleepy.size < 4:
        # Pad the confusion matrix with zeros to make it 2x2 if it has less than 4 elements
        cm_nonsleepy_padded = np.pad(cm_nonsleepy, ((0, max(0, 2 - cm_nonsleepy.shape[0])), (0, max(0, 2 - cm_nonsleepy.shape[1]))), mode='constant')
        
    else:
        # Use the original confusion matrix if it is already 2x2
        cm_nonsleepy_padded = cm_nonsleepy

    # Unpack the padded sleepy confusion matrix into variables
    tn_s, fp_s, fn_s, tp_s = cm_sleepy_padded.flatten()

    # Unpack the padded nonsleepy confusion matrix into variables
    tn_ns, fp_ns, fn_ns, tp_ns = cm_nonsleepy_padded.flatten()

    # Sum up true positives from both metrics
    tp = tp_s + tp_ns

    # Sum up true negatives  from both metrics
    tn = tn_s + tn_ns

    # Sum up false positives  from both metrics
    fp = fp_s + fp_ns

    # Sum up false negatives  from both metrics
    fn = fn_s + fn_ns

    # Calculate accuracy metric
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Calculate val loss metric
    mean_val_loss = (np.mean(history_sleepy.history['val_loss']) + np.mean(history_nonsleepy.history['val_loss'])) / 2

    # Calculate precision metric
    precision = tp / (tp + fp) if tp + fp != 0 else 0.0

    # Calculate recall metric
    recall = tp / (tp + fn) if tp + fn != 0 else 0.0

    # Calculate f1-score metric
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0.0

    # Print metrics
    print("\nConfusion Matrix:")
    print(f"                            Predicted")
    print(f"   Actual        |   Positive  |   Negative  |")
    print(f"        Positive |    {tp:5}    |    {fn:5}    |")
    print(f"        Negative |    {fp:5}    |    {tn:5}    |")
    print(f"\nValidation Metrics - Epoch: {current_iteration + 1}, "
          f"Val Accuracy: {accuracy:.4f}, "
          f"Val Loss: {mean_val_loss:.4f}, "
          f"Precision: {precision:.4f}, "
          f"Recall: {recall:.4f}, "
          f"F1 Score: {f1:.4f}")

    return mean_val_loss, accuracy

##############################################################################################
#                                                                                            #
#                                  MODEL SAVING SECTION                                      #
#                                                                                            #
##############################################################################################

# Function to save the trained model and print the model shapes
def save_and_print_model_shapes(final_model_filepath, sleepy_model, nonsleepy_model):
    """
    Saves the provided models in the specified directory and prints their input and output shapes.

    Parameters:
        final_model_filepath (str): The path where the models will be saved.
        sleepy_model (tf.keras.Model): The trained model for sleepy data.
        nonsleepy_model (tf.keras.Model): The trained model for non-sleepy data.
    """

    if True:
        # Save the Sleepy model in Keras format
        sleepy_model.save(os.path.join(final_model_filepath, "Model_EAR_MAR_HP_Sleepy.keras"))

        # Save the Non-Sleepy model in Keras format
        nonsleepy_model.save(os.path.join(final_model_filepath, "Model_EAR_MAR_HP_NonSleepy.keras"))

        # Print the input and output shapes of the Sleepy model
        print_model_shapes("Sleepy_Model", sleepy_model)

        # Print the input and output shapes of the Non-Sleepy model
        print_model_shapes("NonSleepy_Model", nonsleepy_model)

# Function to print model input and output shapes
def print_model_shapes(model_name, model):
    """
    Prints the input and output shapes of the given model.

    Parameters:
        model_name (str): The name of the model to be printed.
        model (tf.keras.Model): The Keras model whose shapes will be printed.
    """

    # Print the model name
    print(f"\nModel: {model_name}")
    
    # Print the heading for input shapes
    print("Input Shapes:")
    
    # Iterate over the input nodes of the model and print their shapes
    for input_node in model.inputs:
        # Print the shape of each input node
        print(f"  - Shape: {input_node.shape}")
    
    # Print the heading for output shapes
    print("Output Shapes:")
    
    # Iterate over the output nodes of the model and print their shapes
    for output_node in model.outputs:
        # Print the shape of each output node
        print(f"  - Shape: {output_node.shape}")

##############################################################################################
#                                                                                            #
#                                       MAIN SECTION                                         #
#                                                                                            #
##############################################################################################

# Define the main function
def main():
    """
    Main function to process data loading, model training, evaluation, and saving.

    Parameters:
        None

    Returns:
        None
    """

    # Record start time
    start_time = time.time()

    # Load CSV data with specified filters
    print("\nLoading Training Dataset...")
    df_preprocessed = load_and_process_data(training_folder_paths)

    # Select specific features along with Drowsiness label and Video_ID
    df_processed = df_preprocessed[['Drowsiness', 'Video_ID'] + feature_columns]

    # Filter DataFrame for 'nonsleepyCombination'
    df_nonsleepy = df_processed[df_processed['Video_ID'].str.contains('_nonsleepyCombination_')]

    # Filter DataFrame for 'sleepyCombination'
    df_sleepy = df_processed[df_processed['Video_ID'].str.contains('_sleepyCombination_')]

    # Debug the loaded and procesed data
    print("\nDisplaying Dataset Details...")
    debug_video_data(df_processed, df_sleepy, df_nonsleepy, training_folder_paths)
    
    # Define the input shape based on the max sequence length and the number of features
    input_shape = (max_sequence_length, len(selected_features))

    # Create the models
    print("\nBuilding the Long-Short-Term Model (LSTM)...")
    Sleepy_Model = create_custom_lstm_model(name="Sleepy_Model", input_shape=input_shape, initial_learning_rate=initial_lr)
    NonSleepy_Model = create_custom_lstm_model(name="NonSleepy_Model", input_shape=input_shape, initial_learning_rate=initial_lr)

    # Prepare the training data into sequences
    print("\nPrepare Training Data into Sequences...")
    data_sequences = prepare_training_data(df_processed)

    # Train the model
    print("\nTraining the Model...")

    # Initialize variables for early stopping
    best_loss = float('inf')
    patience_counter = 0

    # Implement iterations for multiple epoch training
    for current_iteration in range(training_iterations):
        # Obtain the time the iteration started
        iteration_start = time.time()

        # Clear history for new iteration
        history_sleepy = None
        history_nonsleepy = None

        # Shuffling and splitting Video IDs
        train_sleepy, val_sleepy, train_nonsleepy, val_nonsleepy = split_and_shuffle_video_ids(df_sleepy, df_nonsleepy)

        # Extract and preprocess data for Sleepy videos for training
        x_train_sleepy, y_train_sleepy = extract_data(train_sleepy, data_sequences, "Sleepy Train Dataset")

        # Extract and preprocess data for Sleepy videos for validation
        x_val_sleepy, y_val_sleepy = extract_data(val_sleepy, data_sequences, "Sleepy Validation Dataset")

        # Extract and preprocess data for NonSleepy videos for training
        x_train_nonsleepy, y_train_nonsleepy = extract_data(train_nonsleepy, data_sequences, "NonSleepy Train Dataset")

        # Extract and preprocess data for NonSleepy videos for validation
        x_val_nonsleepy, y_val_nonsleepy = extract_data(val_nonsleepy, data_sequences, "NonSleepy Validation Dataset")

        # Set the LearningRateScheduler callback
        lr_scheduler = LRScheduler(initial_lr, current_iteration)

        # Train the Sleepy model on the extracted sleepy data
        print("Sleepy Model Training ", end='')
        history_sleepy = Sleepy_Model.fit(x_train_sleepy, y_train_sleepy,
                                        epochs=1,
                                        batch_size=batch_size,
                                        callbacks=[lr_scheduler, checkpoint_sleepy],
                                        verbose=1, 
                                        shuffle=False,
                                        validation_data=(x_val_sleepy, y_val_sleepy))
        
        # Train the NonSleepy model on the extracted nonsleepy data
        print("NonSleepy Model Training ", end='')
        history_nonsleepy = NonSleepy_Model.fit(x_train_nonsleepy, y_train_nonsleepy,  
                                                epochs=1,
                                                batch_size=batch_size,
                                                callbacks=[lr_scheduler, checkpoint_nonsleepy],
                                                verbose=1, 
                                                shuffle=False,
                                                validation_data=(x_val_nonsleepy, y_val_nonsleepy))

        # Predict with Sleepy model on the extracted sleepy validation set
        y_val_sleepy_pred = Sleepy_Model.predict(x_val_sleepy, verbose=0)

        # Predict with NonSleepy model on the extracted nonsleepy validation set
        y_val_nonsleepy_pred = NonSleepy_Model.predict(x_val_nonsleepy, verbose=0)

        # Compute and print metrics, and get mean validation loss
        val_loss, accuracy = compute_metrics(y_val_sleepy, y_val_sleepy_pred, y_val_nonsleepy, y_val_nonsleepy_pred, 
                                        history_sleepy, history_nonsleepy, current_iteration
        )
        
        # Check if current mean val loss is less than best loss for early stopping
        if val_loss < best_loss:
            # Set the current mean val loss as the best loss
            best_loss = val_loss

            # Obtain accuracy of the best epoch loss
            best_epoch_accuracy = accuracy

            # Reset patience counter
            patience_counter = 0

            # Set best epoch
            best_epoch = current_iteration + 1

        else:
            # Increment patience counter
            patience_counter += 1

        # Calculate processing time duration for the current iteration
        iteration_end = time.time()
        iteration_time = iteration_end - iteration_start
        iteration_time_readable = time.strftime("%H:%M:%S", time.gmtime(iteration_time))
        print(f"Epoch: {current_iteration + 1}/{training_iterations} Processing Time: {iteration_time_readable}    Best Epoch: {best_epoch}/{training_iterations}    Val. Loss: {best_loss:.4f}    Accuracy: {best_epoch_accuracy:.4f}\n")        

        # Check if patience counter is greater than or equal to the defined patience for early stopping
        if patience_counter >= patience:
            # Print debugging line and End the training
            print(f"Early stopping at iteration {current_iteration + 1} due to no improvement in validation loss for {patience} iterations.")
            break
    
    # Load the best model weights   
    Sleepy_Model.load_weights(os.path.join(checkpoint_dir, 'best_sleepy_model.tf'))
    NonSleepy_Model.load_weights(os.path.join(checkpoint_dir, 'best_nonsleepy_model.tf'))
    
    # Save the models and print their shapes
    save_and_print_model_shapes(final_model_path, Sleepy_Model, NonSleepy_Model)

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