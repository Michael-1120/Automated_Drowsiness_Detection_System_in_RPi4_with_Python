"""
This program contains functions to load, process, and prepare evaluation data, evaluate LSTM models and calculate evaluation metrics for each video, 
each, category, and overall dataset. The main function iteratively loops through all video data of the evaluation dataset. 

The program consists of the following sections:

1. Import/Initialization Section
    - Import necessary python libraries
    - Define folder paths
    - Define global variables

2. Data Loading Section
    - Function 'load_and_process_data': Loads the pre-processed csv with computed features data and call the process_features function.
    - Function 'preprocess_features': Processes the dataframes with thresholding, EMA smoothing, removing irrelevant initial frames, and normalizing the dataset.
    - Function 'debug_video_data': Display the processed and splitted dataframes details.

3. Evaluation Data Preparation Section:
    - Function `prepare_evaluation_data`: Prepares evaluation data for trained model evaluation creating sequences and labels from input DataFrame containing video sequences.

4. Model Metric Section:
    - Function 'calculate_and_record_metrics': Calculate evaluation metrics (accuracy, precision, recall, F1 score) from an array of true and predicted labels.
    - Function `calculate_combined_metrics`: Calculates combined evaluation metrics (accuracy, precision, recall, F1 score) from a list of metric dictionaries.

5. Main Section:
    - Function `main`: Main function to process and calls the necessary function for the flow of evaluating the trained LSTM Drowsiness Detection Model. 

The program is designed to be run as a standalone module, and the main function `main` is called when the script is executed directly. 
It utilizes various parameters and configurations defined within the script to control model evaluation.
"""

##############################################################################################
#                                                                                            #
#                               IMPORT/INITIALIZATION SECTION                                #
#                                                                                            #
##############################################################################################

import os
import time
import warnings

import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model

# Define lists of folder paths
evaluation_folder_paths = [
    r"D:\MJ Hard Drive Files\THESIS\Processed Combined NTHU Evaluation Dataset\glasses",
    r"D:\MJ Hard Drive Files\THESIS\Processed Combined NTHU Evaluation Dataset\noglasses",
    r"D:\MJ Hard Drive Files\THESIS\Processed Combined NTHU Evaluation Dataset\nightglasses",
    r"D:\MJ Hard Drive Files\THESIS\Processed Combined NTHU Evaluation Dataset\night_noglasses"
]

# Define trained model path
sleepy_model_path = r"D:\MJ Hard Drive Files\THESIS\Trained Model\Model_EAR_MAR_HP_Sleepy.keras"
nonsleepy_model_path = r"D:\MJ Hard Drive Files\THESIS\Trained Model\Model_EAR_MAR_HP_NonSleepy.keras"

# Define random seed for reproducibility
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

# Define prediction weights
prediction_threshold = 0.5
sleepy_prediction_weight = 0.6
nonsleepy_prediction_weight = 0.4

# Define list to store evaluation metics 
video_metrics = []

# Define lists for calculating metrics for each subfolder category
glasses_metrics = []
noglasses_metrics = []
nightglasses_metrics = []
night_noglasses_metrics = []

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

            # 3:1 downsampling from 30 fps to 10 fps
            df = df.iloc[::3, :].reset_index(drop=True)  

            # Process the df 
            df_processed = preprocess_features(df, unstable_frames)

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
def preprocess_features(df, skipped_frames):
    """
    Preprocesses the feature data in the given DataFrame.

    Parameters:
        df (DataFrame): The input DataFrame containing feature data.
        skipped_frames (int): The number of initial frames to skip for stabilization.

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
    df_feature_averages = df_processed[feature_columns][:initial_frame].mean()

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
def debug_video_data(all_df, folder_paths):
    """
    Debugs video data by printing counts of drowsy and non-drowsy frames for each video,
    total number of evaluation videos, and number of rows and videos with NaN values.

    Parameters:
        all_df (pd.DataFrame): DataFrame containing all video data.
        folder_paths (list): List of folder paths containing video files.

    Returns:
        None
    """

    # Iterate through each unique video id
    for vid_id in all_df['Video_ID'].unique():
        # Obtain the df of the respective id
        df_data = all_df[all_df['Video_ID'] == vid_id]

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

    # Get the total number of ids
    num_vid = len(ids)

    # Print the total video in the dataset 
    print(f"\nTotal Evaluation Video Dataset: {num_vid } ")

    # Check for NaN values
    nan_rows = all_df[all_df.isna().any(axis=1)]

    # Get the number of rows with nan values
    num_nan_rows = len(nan_rows)

    # Get the number of video with nan values
    num_videos_with_nan = len(all_df[all_df.isna().any(axis=1)]['Video_ID'].unique())

    # Print debugging line
    print(f"Number of rows with NaN values: {num_nan_rows}")
    print(f"Number of video IDs with at least one NaN value: {num_videos_with_nan}")

    # Debug variables to compute counts of Drowsiness for the whole dataset
    print("\nClass Imbalance for the Total Dataset:")
    print(all_df['Drowsiness'].value_counts())

    # Print shapes of DataFrames
    print("\nShape of df_processed: ", all_df.shape)

##############################################################################################
#                                                                                            #
#                           EVALUATION DATA PREPARATION SECTION                              #
#                                                                                            #
##############################################################################################

# Function to prepare the evaluation data
def prepare_evaluation_data(df):
    """
    Prepares the evaluation data for LSTM model evaluation.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the video sequences.

    Returns:
        dict: A dictionary containing the prepared evaluation data for each video sequence.
    """

    # Initialize a dictionary to store all sequence data
    all_data_dict = {}

    # Loop through unique video IDs in the DataFrame
    for video_name in tqdm(df['Video_ID'].unique(), desc="Evaluation Preparation", unit=" Video sequences and labels", ncols=150, leave=False):
        # Filter DataFrame for the current video ID
        video_df = df[df['Video_ID'] == video_name].copy()

        # Find the length of the current df being processed
        num_samples = len(video_df)

        # Compute the number of sequences based on the number of samples and the max sequence length
        num_sequences = (num_samples - max_sequence_length) + 1
        
        # Initialize arrays to store sequences and labels
        x_data = np.zeros((num_sequences, max_sequence_length, len(feature_columns)))
        y_data = np.zeros((num_sequences, 1))
        
        # Iterate through each sequence in the video
        for i in range(num_sequences):
            # Set starting df index based on iteration
            start_index = i

            # Set last df index based on starting index and max sequence length
            end_index = start_index + max_sequence_length
            
            # Extract sequence data based on starting and ending indices for the current df
            sequence = video_df[feature_columns].iloc[start_index:end_index, :].values

            # Extract the label for that sequence
            label = video_df[label_column].iloc[end_index - 1]

            # Initializes a zero-filled array of shape (max_sequence_length, len(feature_columns)) to represent the padded sequence
            padded_sequence = np.zeros((max_sequence_length, len(feature_columns)))

            # Assigns the sequence data to the padded sequence array, ensuring it fits within the desired length
            padded_sequence[:max_sequence_length, :] = sequence

            # Handle unused features
            for j, feature in enumerate(feature_columns):
                # Check if the feature is not in the selected features
                if feature not in selected_features:
                    # If feature is unused, set it to 0
                    padded_sequence[:, j] = 0

            # Stores the padded sequence data in the x_data array at index i
            x_data[i] = padded_sequence

            # Assigns the label to the y_data array at index i
            y_data[i] = label

        # Store prepared data in the dictionary
        all_data_dict[video_name] = (x_data, y_data)
    
    return all_data_dict

##############################################################################################
#                                                                                            #
#                                  MODEL METRIC SECTION                                      #
#                                                                                            #
##############################################################################################

# Function to compute metric after evaluation for each iteration
def calculate_and_record_metrics(combined_predictions, y_true, data_id):
    """
    Calculate evaluation metrics based on combined predictions and true labels.

    Parameters:
        combined_predictions (numpy.ndarray): Combined predictions from multiple models.
        y_true (numpy.ndarray): True labels.
        data_id (str): Identifier for the data being evaluated.

    Returns:
        metric (dict): Dictionary containing evaluation metrics.
    """

    # Force each prediction as a binary, either 0 or 1
    combined_predictions_binary = (combined_predictions > prediction_threshold).astype(int)
    
    # Obtain the confusion matrix
    cm = confusion_matrix(y_true, combined_predictions_binary)
    
    # Ensure the confusion matrix is always 2x2
    if cm.size < 4:
        # Pad the confusion matrix with zeros to make it 2x2 if it has less than 4 elements
        cm_padded = np.pad(cm, ((0, max(0, 2 - cm.shape[0])), (0, max(0, 2 - cm.shape[1]))), mode='constant')
        
    else:
        # Use the original confusion matrix if it is already 2x2
        cm_padded = cm
    
    # Flatten and unpack the padded confusion matrix into variables
    tn, fp, fn, tp = cm_padded.flatten()
    
    # Calculate accuracy metric
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Calculate precision metric
    precision = tp / (tp + fp) if tp + fp != 0 else 0.0

    # Calculate recall metric
    recall = tp / (tp + fn) if tp + fn != 0 else 0.0

    # Calculate f1-score metric
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0.0
    
    # Print metrics
    print(f"Metrics for Video ID: {data_id}")
    print("Confusion Matrix:")
    print("                            Predicted")
    print("   Actual        |   Positive  |   Negative  |")
    print(f"        Positive |    {tp:5}    |    {fn:5}    |")
    print(f"        Negative |    {fp:5}    |    {tn:5}    |")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}\n")

    # Create dictionary containing evaluation metrics
    metric = {
        'Video_ID': data_id,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metric

# Calculate combined metrics for each subfolder category
def calculate_combined_metrics(metrics_list):
    """
    Calculate combined metrics based on a list of metrics dictionaries.

    Parameters:
        metrics_list (list): List of metrics dictionaries.

    Returns:
        dict: Combined metrics dictionary.
    """

    # Check if the metric list is empty
    if not metrics_list:
        # Return 0 metrics if its empty
        return {
            'Accuracy': 0.0,
            'Precision': 0.0,
            'Recall': 0.0,
            'F1': 0.0
        }
    
    else:
        # Obtain the sum of all true positive in each metric in metric list
        tp_sum = sum([metrics['tp'] for metrics in metrics_list])

        # Obtain the sum of all true negative in each metric in metric list
        tn_sum = sum([metrics['tn'] for metrics in metrics_list])

        # Obtain the sum of all false positive in each metric in metric list
        fp_sum = sum([metrics['fp'] for metrics in metrics_list])

        # Obtain the sum of all false negative in each metric in metric list
        fn_sum = sum([metrics['fn'] for metrics in metrics_list])

        # Calculate the combined accuracy
        accuracy = (tp_sum + tn_sum) / (tp_sum + tn_sum + fp_sum + fn_sum)

        # Calculate the combined precision
        precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) != 0 else 0.0

        # Calculate the combined recall
        recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) != 0 else 0.0

        # Calculate the combined f1-score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0

        return {
            'Accuracy': '{:.4f}'.format(accuracy),
            'Precision': '{:.4f}'.format(precision),
            'Recall': '{:.4f}'.format(recall),
            'F1': '{:.4f}'.format(f1)
        }

##############################################################################################
#                                                                                            #
#                                       MAIN SECTION                                         #
#                                                                                            #
##############################################################################################

# Define the main function
def main():
    """
    Main function to evaluate the trained model.

    Parameters:
        None

    Returns:
        None
    """

    # Record start time
    start_time = time.time()

    # Load CSV data with specified filters
    df_preprocessed = load_and_process_data(evaluation_folder_paths)

    # Select specific features along with Drowsiness label and Video_ID
    df_processed = df_preprocessed[['Drowsiness', 'Video_ID'] + feature_columns]

    # Debug the loaded and procesed data
    print("\nDisplaying Dataset Details...")
    debug_video_data(df_processed, evaluation_folder_paths)

    # Load the trained model
    Sleepy_Model = load_model(sleepy_model_path)
    NonSleepy_Model = load_model(nonsleepy_model_path)

    # Prepare the evaluation data into sequences
    print("\nPrepare Evaluation Data into Sequences...")
    data_sequences = prepare_evaluation_data(df_processed)

    # Evaluate the model
    print("\nEvaluating the Model...")

    # Iterate over each video ID for evaluation
    for video_data in tqdm(df_processed['Video_ID'].unique(), desc="Evaluating", unit="Video sequences", ncols=50, leave=False):
        # Extract and preprocess data from current video id for evaluation
        x_data, y_true = data_sequences[video_data]

        # Print the shape of the x_data input
        print("\nInput data shape:", x_data.shape)

        # Predict using the Sleepy model
        sleepy_predictions = Sleepy_Model.predict(x_data, verbose=0)
        print("Sleepy Model Predictions Output Shape:", sleepy_predictions.shape)

        # Predict using the Non-Sleepy model
        nonsleepy_predictions = NonSleepy_Model.predict(x_data, verbose=0)
        print("Non-Sleepy Model Predictions Output Shape:", nonsleepy_predictions.shape)

        # Combine predictions with bias or weights
        combined_predictions = (sleepy_prediction_weight * sleepy_predictions) + (nonsleepy_prediction_weight * nonsleepy_predictions)

        # Calculate and print evaluation metric
        metrics = calculate_and_record_metrics(combined_predictions, y_true, video_data)

        # Append video metric
        video_metrics.append(metrics)

    # Iterate through all evaluation metrics
    for metrics in video_metrics:
        # Obtain the video id for the current metric
        data_id = metrics['Video_ID']

        # Extract subfolder name
        subfolder = data_id.split('\\')[-2]

        # Check if the subfolder is equal to any of the keywords  
        if subfolder == 'glasses':
            # Append the glasses metric list
            glasses_metrics.append(metrics)

        elif subfolder == 'noglasses':
            # Append the noglasses metric list
            noglasses_metrics.append(metrics)

        elif subfolder == 'nightglasses':
            # Append the nightglasses metric list
            nightglasses_metrics.append(metrics)

        elif subfolder == 'night_noglasses':
            # Append the night noglasses metric list
            night_noglasses_metrics.append(metrics)

    # Calculate combined metrics for each subfolder category
    glasses_combined_metrics = calculate_combined_metrics(glasses_metrics)
    noglasses_combined_metrics = calculate_combined_metrics(noglasses_metrics)
    nightglasses_combined_metrics = calculate_combined_metrics(nightglasses_metrics)
    night_noglasses_combined_metrics = calculate_combined_metrics(night_noglasses_metrics)

    # Compute metrics for additional categories
    day_category_metrics = calculate_combined_metrics(glasses_metrics + noglasses_metrics)
    night_category_metrics = calculate_combined_metrics(nightglasses_metrics + night_noglasses_metrics)
    glasses_category_metrics = calculate_combined_metrics(glasses_metrics + nightglasses_metrics)
    no_glasses_category_metrics = calculate_combined_metrics(noglasses_metrics + night_noglasses_metrics)

    # Calculate overall combined metrics for all evaluation video IDs
    overall_combined_metrics = calculate_combined_metrics(video_metrics)

    # Print combined metrics for each subfolder category
    print("\nCombined Metrics for each Subfolder Category:")
    print(f"Glasses:                                              {glasses_combined_metrics}")
    print(f"NoGlasses:                                            {noglasses_combined_metrics}")
    print(f"NightGlasses:                                         {nightglasses_combined_metrics}")
    print(f"Night_NoGlasses:                                      {night_noglasses_combined_metrics}")

    # Print additional category metrics
    print("\nAdditional Category Metrics:")
    print(f"Day (Glasses + No Glasses):                           {day_category_metrics}")
    print(f"Night (Night Glasses + Night No Glasses):             {night_category_metrics}")
    print(f"Glasses Category (Glasses + Night Glasses):           {glasses_category_metrics}")
    print(f"No Glasses Category (No Glasses + Night No Glasses):  {no_glasses_category_metrics}")

    # Print overall combined metrics
    print(f"\nOverall Combined Metrics for all Evaluation Video IDs:      {overall_combined_metrics}\n")

    # Calculate processing time duration
    stop_time = time.time()
    duration_seconds = stop_time - start_time
    duration_readable = time.strftime("%H:%M:%S", time.gmtime(duration_seconds))
    print(f"Total Processing Time: {duration_readable}")

# Check if the script is being run directly
if __name__ == "__main__":   
    # Call the main function
    main()

##############################################################################################
#                                                                                            #
#                                        END SECTION                                         #
#                                                                                            #
##############################################################################################