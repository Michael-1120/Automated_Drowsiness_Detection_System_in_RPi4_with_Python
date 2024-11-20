"""
This program processes the NTHU Driver Drowsiness Dataset by organizing videos and text files into appropriate categories
and copying them to specified destination directories for training, evaluation, and testing purposes.

The program consists of the following sections:

1. Import/Initialization Section
    - Import necessary python libraries.
    - Define folder paths.
    - Define global variables.

2. Dataset Processing Section
    - Function 'training_dataset': Processes the training dataset by copying video and annotation files into appropriate categories.
    - Function 'evaluation_dataset': Processes the evaluation dataset by copying video and annotation files into appropriate categories.
    - Function 'testing_dataset': Processes the testing dataset by copying video and annotation files into appropriate categories.

3. Main Section:
    - Function `main`: Reads the source directories containing raw data, creates necessary directories in the destination paths, and ensures files are correctly categorized and renamed as needed.

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

from tqdm import tqdm

# Define the directories for training, evaluation, and testing datasets
training_source_directory = r'D:\NTHU Driver Drowsiness Dataset\Training_Evaluation_Dataset\Training Dataset'
training_destination_directory = r'D:\Processed Combined NTHU Training Dataset'

evaluation_source_directory = r'D:\NTHU Driver Drowsiness Dataset\Training_Evaluation_Dataset\Evaluation Dataset'
evaluation_destination_directory = r'D:\Processed Combined NTHU Evaluation Dataset'

testing_source_directory = r'D:\NTHU Driver Drowsiness Dataset\Testing_Dataset'
testing_destination_directory = r'D:\Processed Combined NTHU Test Dataset'

##############################################################################################
#                                                                                            #
#                                DATASET PROCESSING SECTION                                  #
#                                                                                            #
##############################################################################################

# Function to reorganize the training dataset
def training_dataset(source_dir, destination_dir):
    """
    Processes the training dataset by organizing videos and text files into 
    appropriate categories and copying them to the destination directory.

    Parameters:
        source_dir (str): Path to the source directory containing raw training data.
        destination_dir (str): Path to the destination directory to save processed data.
    Return:
        None
    """
    
    # Define the categories for training dataset
    categories = ['glasses', 'nightglasses', 'night_noglasses', 'noglasses']

    # Check the destination directory if it exists
    if os.path.exists(destination_dir):
        # Remove destination folder
        shutil.rmtree(destination_dir)
    
    # Create destination folder
    os.makedirs(destination_dir)
    
    # Iterate through each participant's directory
    for participant_dir in os.listdir(source_dir):
        # Obtain the participant path
        participant_path = os.path.join(source_dir, participant_dir)

        # Iterate through the categories 
        for category in categories:
            # Obtain category path from prticpant path
            category_path = os.path.join(participant_path, category)
            
            # Set destination path based on category
            destination_category_path = os.path.join(destination_dir, category)

            # Check the destination folder if it doesn't exist
            if not os.path.exists(destination_category_path):
                # Create the destination folder
                os.makedirs(destination_category_path)

            # Iterate through the file names in the category path
            for file_name in tqdm(os.listdir(category_path), desc=f"Processing {participant_dir} - {category}", ncols=150):
                # Exclude yawning and slowBlinkWithNodding videos from training
                if "yawning" in file_name.lower() or "slowblinkwithnodding" in file_name.lower():
                    continue

                # Obtain the participant number from the participant path
                participant_number = participant_dir.zfill(3)

                # Obtain the actual path of the video file
                source_file_path = os.path.join(category_path, file_name)

                # Check if the file is a video file and contains "Combination" in the name
                if file_name.lower().endswith(('.avi', '.mp4', '.mkv', '.mov')) and "combination" in file_name.lower():
                    # Add participant number and category name to the video file name
                    video_file_name = f"{participant_number}_{file_name.rsplit('.', 1)[0]}_{category}_mix.{file_name.rsplit('.', 1)[1]}"

                    # Set the destination path with revised file name
                    destination_file_path = os.path.join(destination_category_path, video_file_name)

                    # Copy the video file
                    shutil.copy(source_file_path, destination_file_path)

                # Check if the file is a text file and ends with "_drowsiness.txt"
                elif file_name.lower().endswith('_drowsiness.txt'):
                    # Remove the last word from the text file name
                    base_name = file_name.rsplit('_', 1)[0]

                    # Add category name at the end of the text file name as well as some texts
                    text_file_name = f"{base_name}_{category}_mixing_drowsiness.txt"
                    
                    # Set the destination path with revised file name
                    destination_file_path = os.path.join(destination_category_path, text_file_name)

                    # Copy the video file
                    shutil.copy(source_file_path, destination_file_path)

# Function to reorganize the evaluation dataset
def evaluation_dataset(source_dir, destination_dir):
    """
    Processes the evaluation dataset by organizing videos and text files into 
    appropriate categories and copying them to the destination directory.

    Parameters:
        source_dir (str): Path to the source directory containing raw evaluation data.
        destination_dir (str): Path to the destination directory to save processed data.
    Return:
        None
    """

    # Check the destination directory if it exists
    if os.path.exists(destination_dir):
        # Remove destination folder
        shutil.rmtree(destination_dir)
    
    # Create destination folder
    os.makedirs(destination_dir)

    # Iterate through each participant's directory
    for participant_dir in os.listdir(source_dir):
        # Obtain the participant path
        participant_path = os.path.join(source_dir, participant_dir)

        # Iterate through the files in the participant path
        for file_name in tqdm(os.listdir(participant_path), desc=f"Processing {participant_dir} - All Categories", ncols=150):
            # Check if the file is a video file or text file
            if file_name.lower().endswith(('.avi', '.mp4', '.mkv', '.mov')) or file_name.lower().endswith('ing_drowsiness.txt'):
                # Exclude sunglasses and yawning videos from evaluation
                if "sunglasses" in file_name.lower():
                    continue
                
                # obtain the actual source file path based on participant path an the file names
                source_file_path = os.path.join(participant_path, file_name)

                # Obtain the category of the file based on the file name
                category = file_name.split('_')[1]

                # Ensure night_noglasses folder name is used
                if category == "night":
                    # The category was obtained from 2nd word but for night no glasses its split with 2 words
                    category = "night_noglasses"

                # Set the destination category
                destination_category_path = os.path.join(destination_dir, category)

                # Check the destination path if it does not exist
                if not os.path.exists(destination_category_path):
                    # Create the destination folder
                    os.makedirs(destination_category_path)

                # Set the actual destination file path 
                destination_file_path = os.path.join(destination_category_path, file_name)

                # Copy the file
                shutil.copy(source_file_path, destination_file_path)

# Function to reorganize the testing dataset
def testing_dataset(source_dir, destination_dir):
    """
    Processes the testing dataset by organizing videos and text files into 
    appropriate categories and copying them to the destination directory.

    Parameters:
        source_dir (str): Path to the source directory containing raw testing data.
        destination_dir (str): Path to the destination directory to save processed data.
    Return:
        None
    """

    # Check the destination directory if it exists
    if os.path.exists(destination_dir):
        # Remove destination folder
        shutil.rmtree(destination_dir)
    
    # Create destination folder
    os.makedirs(destination_dir)

    # Define the categories for testing dataset
    categories = ['glasses', 'noglasses', 'nightglasses', 'night_noglasses']

    # Get list of file names in source directory
    file_names = os.listdir(source_dir)

    # Remove files containing "sunglasses" and the "test_label_txt" folder
    file_names_filtered = [file_name for file_name in file_names if "sunglasses" not in file_name.lower() and file_name.lower() != "test_label_txt"]

    # Iterate through filtered file names with progress bar
    for original_file_name in file_names_filtered:
        # Determine the destination category
        category = original_file_name.split('_')[1]

        # Rename category "nightnoglasses" to "night_noglasses"
        if category == "nightnoglasses":
            # Rename the category
            renamed_category = "night_noglasses"
            
            # Rename the file name
            renamed_file_name = original_file_name.replace("nightnoglasses", "night_noglasses")

        else:
            # Use the original category
            renamed_category = category

            # Use the original file name
            renamed_file_name = original_file_name

        # Check if the renamed category is in the list
        if renamed_category in categories:
            # Set the destination folder based on the category
            destination_folder = os.path.join(destination_dir, renamed_category)

            # Check path if it doesn't exist
            if not os.path.exists(destination_folder):
                # Create destination path
                os.makedirs(destination_folder)

            # Display progress bar for each file being processed
            with tqdm(desc=f"Processing {original_file_name}", total=1, ncols=150) as pbar:
                # Copy video file to destination folder
                source_file_path = os.path.join(source_dir, original_file_name)
                
                # Set destination file name based on destination category and renamed file name
                destination_file_path = os.path.join(destination_folder, renamed_file_name)

                # Copy the file
                shutil.copy(source_file_path, destination_file_path)

                # Obtain the annotation file name based on video file name
                original_annotation_file_name = original_file_name.replace(".mp4", "ing_drowsiness.txt")
                
                # Check if the annotation file name has the nightnoglasses word
                if "nightnoglasses" in original_annotation_file_name:
                    # Rename the annotation file name
                    renamed_annotation_file_name = original_annotation_file_name.replace("nightnoglasses", "night_noglasses")

                else:
                    # Use the original annotation file name
                    renamed_annotation_file_name = original_annotation_file_name

                # Obtain the actual annotaion file path
                source_annotation_file = os.path.join(source_dir, "test_label_txt", "wh", original_annotation_file_name)
                
                # Check if annotation file exists before copying
                if os.path.exists(source_annotation_file):
                    # Set the destination path for the annotation file
                    destination_annotation_file = os.path.join(destination_folder, renamed_annotation_file_name)

                    # Copy the annotation file
                    shutil.copy(source_annotation_file, destination_annotation_file)
                    
                else:
                    # Print a debugging statement
                    print(f"Annotation file {original_annotation_file_name} not found.")

                # Update progress bar
                pbar.update(1)

##############################################################################################
#                                                                                            #
#                                       MAIN SECTION                                         #
#                                                                                            #
##############################################################################################

# Define the main function
def main():
    """
    Main function to process and reorganize the training, evaluation, and testing datasets.

    Parameters:
        None

    Returns:
        None
    """

    # Start the timer to measure processing duration
    start_time = time.time()    

    # Start Organizing the Training Dataset
    print("Processing Training Dataset")
    training_dataset(training_source_directory, training_destination_directory)

    # Start Organizing the Evaluation Dataset
    print("\nProcessing Evaluation Dataset")
    evaluation_dataset(evaluation_source_directory, evaluation_destination_directory)

    # Start Organizing the Testing Dataset
    print("\nProcessing Testing Dataset")
    testing_dataset(testing_source_directory, testing_destination_directory)

    # Calculate and print the processing duration
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nProcessing completed in {duration:.2f} seconds.")

# Check if the script is being run directly
if __name__ == "__main__":   
    # Call the main function
    main()

##############################################################################################
#                                                                                            #
#                                        END SECTION                                         #
#                                                                                            #
##############################################################################################