"""
This program contains functions to optimize the trained model and optimize a student model with distilled knowdlege.

The program consists of the following sections:

1. Import/Initialization Section
    - Import necessary python libraries
    - Define folder paths
    - Define global variables

2. Data Loading Section
    - Function 'load_and_process_data': Loads the pre-processed csv with computed features data and call the process_features function.
    - Function 'preprocess_features': Processes the dataframes with thresholding, EMA smoothing, removing irrelevant initial frames, and normalizing the dataset.

3. Optimization Section
    - Function `prune_models`: Prunes the models based on the provided pruning parameters.
    - Function `convert_and_save_tflite`: Converts and saves the pruned model to TFLite format with quantization.

4 STUDENT MODEL SECTION
    - Function `create_student_model`: Defines the student model architecture.
    - Function `prepare_training_data`: Prepares training data for LSTM student model training by creating sequences and labels from input DataFrame containing video sequences.
    - Function `extract_data`: Extracts and combines sequence data and labels for a list of video IDs.
    - Function 'softmax_with_temperature': Softens the logits (probabilities) from the teacher model using temperature scaling.
    - Function `distill_knowledge`: Distills knowledge from the teacher model to the student model.

5. Main Section:
    - Function `main`: Main function to optimize the trained model.

The program is designed to be run as a standalone module, and the main function `main` is called when the script is executed directly. 
It utilizes various parameters and configurations defined within the script to control model optimization.
"""

##############################################################################################
#                                                                                            #
#                               IMPORT/INITIALIZATION SECTION                                #
#                                                                                            #
##############################################################################################

import os
import sys
import time
import shutil
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from tqdm import tqdm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.layers import LSTM, Dropout, Dense

# List of folder paths
training_folder_paths = [
    r"D:\MJ Hard Drive Files\THESIS\Processed Combined NTHU Training Dataset\glasses",
    r"D:\MJ Hard Drive Files\THESIS\Processed Combined NTHU Training Dataset\noglasses",
    r"D:\MJ Hard Drive Files\THESIS\Processed Combined NTHU Training Dataset\nightglasses",
    r"D:\MJ Hard Drive Files\THESIS\Processed Combined NTHU Training Dataset\night_noglasses"
]

# Define trained model path
sleepy_model_path = r"D:\MJ Hard Drive Files\THESIS\Trained Model\Model_EAR_MAR_HP_Sleepy.keras"
nonsleepy_model_path = r"D:\MJ Hard Drive Files\THESIS\Trained Model\Model_EAR_MAR_HP_NonSleepy.keras"

# Define optimized model path
optimized_model_path = r"D:\MJ Hard Drive Files\THESIS\Optimized Model"

# Set full file path for each optimized model
tflite_student_model_path = os.path.join(optimized_model_path, "Quantized_Student_Model.tflite")

# Set random seed for reproducibility
np.random.seed(42)

# Define max sequence length and number of features
max_sequence_length = 100

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

# Defube the Pruning parameter
pruning_params = {'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=0.2, begin_step=0)}

# Define batch size for student model training
batch_size = 32

# Define the number of epoch for distilling knowledge
distill_epochs = 20

# Define regularization strengths
l1_strength = 0.0001
l2_strength = 0.0001
dropout_ratio = 0.2

# Define the initial learning rate
initial_lr = 0.0001

# Define a set for the used combination in training iteration
used_combinations = set()

# Define prediction weights
prediction_threshold = 0.5
sleepy_prediction_weight = 0.6
nonsleepy_prediction_weight = 0.4

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
        return None

    # Combine all processed DataFrames into a single DataFrame
    df_combined = pd.concat(data_frames, ignore_index=True)

    return df_combined

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

##############################################################################################
#                                                                                            #
#                                   OPTIMIZATION SECTION                                     #
#                                                                                            #
##############################################################################################

# Function to prune the models
def prune_models(input_model):
    """
    Prune the models based on the provided pruning parameters.

    Parameters:
        input_model: The model to be pruned.

    Returns:
        pruned_model: The pruned sleepy model.
    """

    # Prune model based on pruning parameter
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(input_model, **pruning_params)

    return pruned_model

# Function to convert and save the pruned models to TFLite format
def convert_and_save_tflite(pruned_model, tflite_model_path):
    """
    Convert and save the pruned model to TFLite format with quantization.

    Parameters:
        pruned_model: The pruned model to be converted.
        tflite_model_path (str): Path where the TFLite model will be saved.
    
    Returns:
        None
    """

    # Set the pruned model as converter
    converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)

    # Optimize the converter
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Defined the operations
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # TFLite default ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # Enable TensorFlow ops.
    ]

    # Disable experimental lowering of tensor list ops.
    converter._experimental_lower_tensor_list_ops = False  
    
    # Convert the model
    tflite_model = converter.convert()

    # Create the outpput file
    with open(tflite_model_path, "wb") as f:
        # Save the model
        f.write(tflite_model)

    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)

    # Allocate tensors for interpreter
    interpreter.allocate_tensors()

    # Get input details
    input_details = interpreter.get_input_details()
    
    # Get output details
    output_details = interpreter.get_output_details()

    # Print the model details
    print(f"TFLite Model Details for {tflite_model_path}:")

    # Print input details
    for detail in input_details:
        # Print all input details
        print("\nInput details:")
        print(f"Name: {detail['name']}")
        print(f"Shape: {detail['shape']}")
        print(f"Type: {detail['dtype']}")

    # Print output details
    for detail in output_details:
        # Print all output details
        print("\nOutput details:")
        print(f"Name: {detail['name']}")
        print(f"Shape: {detail['shape']}")
        print(f"Type: {detail['dtype']}")

##############################################################################################
#                                                                                            #
#                                  STUDENT MODEL SECTION                                     #
#                                                                                            #
##############################################################################################

# Function to create a custom LSTM model layers
def create_student_model(name):
    """
    Define the student model architecture.

    Parametes:
        name (str): The name of the model.

    Returns:
        model: A compiled Keras model.
    """
    
    # Obtain the input shape
    input_shape = (max_sequence_length, len(selected_features))  

    # Create a sequential model with the input name
    model = tf.keras.Sequential(name=name)
    
    # Add LSTM layers with regularization
    model.add(LSTM(units=32, input_shape=input_shape, return_sequences=True, kernel_regularizer=l1(l1_strength), recurrent_regularizer=l1(l1_strength)))
    model.add(Dropout(dropout_ratio))

    model.add(LSTM(units=64, return_sequences=True, kernel_regularizer=l1(l1_strength), recurrent_regularizer=l1(l1_strength)))
    model.add(Dropout(dropout_ratio))

    model.add(LSTM(units=32, return_sequences=False, kernel_regularizer=l1(l1_strength), recurrent_regularizer=l1(l1_strength)))
    model.add(Dropout(dropout_ratio))

    # Output layer
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_strength)))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=initial_lr), loss='binary_crossentropy', metrics=['accuracy'])

    # Print model name before showing the summary
    model.summary()
    
    return model

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

            # Initializes a zero-filled array of shape (max_sequence_length, len(feature_columns)) to represent the padded sequence
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
    """

    # Initialize lists to store x data
    x_data_list = []
    
    # Iterate thought the video ids for data extraction
    for _, video_id in enumerate(tqdm(video_ids, desc=desc, unit="video", ncols=150, leave=False)):
        # Load the sequences and labels for the current video
        x_data, _ = data_dict[video_id]

        # Append the sequences to the lists
        x_data_list.extend(x_data)

    # Convert to numpy array
    x_data_array = np.array(x_data_list)

    return x_data_array

# Soften the logits (probabilities) from the teacher model
def softmax_with_temperature(logits, temperature):
    """
    Soften the logits (probabilities) from the teacher model.

    Parametes:
        logits (np.ndarray): The logits from the teacher model.
        temperature (float): The scaling temperature.

    Returns:
        soft_logits (np.ndarray): Softened probabilities.
    """

    # Scale logits by temperature
    logits = logits / temperature

    # Compute softmax
    soft_logits = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)

    return soft_logits 

# Function for knowledge distillation
def distill_knowledge(sleepy_teacher_model, nonsleepy_teacher_model, student_model, data_sequences):
    """
    Distill knowledge from the teacher model to the student model.

    Args:
        sleepy_teacher_model: The teacher for sleepy model used for knowledge distillation.
        nonsleepy_teacher_model: The teacher for nonsleepy model used for knowledge distillation.
        student_model: The student model to be trained.
        data_sequences (numpy.array): Array of the data sequences for each video data

    Returns:
        trained_student_model: The trained student model.
    """

    # Iterate for the full range of the defined epochs
    for epoch in range(distill_epochs):
        # Adjust the temperature for the current epoch
        temperature = min(20.0, 1.0 + epoch)

        # Print debugging line for the current epoch
        print(f"\nEpoch {epoch + 1}/{distill_epochs}")
        
        # Shuffle video data sequences
        indices = np.arange(data_sequences.shape[0])
        np.random.shuffle(indices)
        sequences = data_sequences[indices]

        # Generate soft labels using the sleepy teacher model
        print("Sleepy Teacher Model Prediction...")
        pseudo__sleepy_labels = sleepy_teacher_model.predict(sequences, verbose=1)

        # Generate soft labels using the nonsleepy teacher model
        print("NonSleepy Teacher Model Prediction...")
        pseudo__nonsleepy_labels = nonsleepy_teacher_model.predict(sequences, verbose=1)

        # Combine predictions with bias or weights
        pseudo_labels = (sleepy_prediction_weight * pseudo__sleepy_labels) + (nonsleepy_prediction_weight * pseudo__nonsleepy_labels)

        # Softed the pseudo labels based on temperature
        softened_logits = softmax_with_temperature(pseudo_labels, temperature)

        # Train the student model using the soft labels
        print("Student Model Training...")
        student_model.fit(sequences, softened_logits, epochs=1, batch_size=batch_size, verbose=1)
        #sys.stdout.write("\033[F\033[K" * 7)
    
    return student_model

##############################################################################################
#                                                                                            #
#                                       MAIN SECTION                                         #
#                                                                                            #
##############################################################################################

# Main function to execute the script
def main():
    """
    Main function to optimize the trained model.

    Parameters:
        None

    Returns:
        None
    """
        
    # Record start time
    start_time = time.time()

    # Load and process data for mean and standard deviations 
    print("\nLoading Training Dataset...")
    df_preprocessed = load_and_process_data(training_folder_paths)

    # Select specific features along with Drowsiness label and Video_ID
    df_processed = df_preprocessed[['Drowsiness', 'Video_ID'] + selected_features]

    # Load the sleepy model and print the details
    sleepy_model = load_model(sleepy_model_path)
    print("\nSleepy Model Summary:")
    sleepy_model.summary()
    print("Sleepy Model Input Shape:", sleepy_model.input_shape)
    print("Sleepy Model Output Shape:", sleepy_model.output_shape)
    
    # Load the nonsleepy model and print the details
    nonsleepy_model = load_model(nonsleepy_model_path)
    print("\nNonSleepy Model Summary:")
    nonsleepy_model.summary()
    print("NonSleepy Model Input Shape:", nonsleepy_model.input_shape)
    print("NonSleepy Model Output Shape:", nonsleepy_model.output_shape)

    # Create the student model
    student_model = create_student_model("Final_Model")

    # Prepare the training data into sequences
    print("\nPrepare Training Data into Sequences...")
    data_sequences = prepare_training_data(df_processed)

    # Obtain the list of video id from the processed dataframe
    video_id_list = list(df_processed['Video_ID'].unique())

    # Extract and process data seuqences
    sequences = extract_data(video_id_list, data_sequences, "Re-Training Dataset")

    # Distill knowledge from the pruned model to the student model
    print("\nDistilling Knowledge to the Student Model...")
    student_model = distill_knowledge(sleepy_model, nonsleepy_model, student_model, sequences)
    print("\nStudent Model trained successfully.")

    print("\nPruning the Student Model...")
    pruned_student_model = prune_models(student_model)

    print("\nQuantizing the Student Model...")
    convert_and_save_tflite(pruned_student_model, tflite_student_model_path)
    print("\nTFLite student model saved successfully.")

    # Record stop time
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