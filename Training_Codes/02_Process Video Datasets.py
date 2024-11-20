"""
This prgoram processes video datasets for drowsiness detection using facial landmarks and annotations. It outputs a csv file
which contains the pre-processed computed features and a video file with text display information for each frame.

The program consists of the following sections:

1. Import/Initialization Section
    - Import necessary python libraries.
    - Define folder paths.
    - Define global variables.
    - Function 'process_dataset': Processes the dataset to clear previously files.

2. Feature Computation Section
    - Function 'distance': Calculates the distance between two points.
    - Function 'compute_eye_aspect_ratio': Computes the average EAR feature for both eyes.
    - Function 'compute_ear_for_eye': Computes the EAR feature of a single eye.
    - Function 'compute_mouth_aspect_ratio': Computes the MAR feature.  
    - Function 'compute_head_pose': Computes the head pose angles through PCA.

3. Visualization Section
    - Function 'image_visualization': Display text for the frame information and save the frame.

3. Main Section:
    - Function `main`: Iterates through each frame of each video of each datasets and calls necessary functions.

The program is designed to be run as a standalone module, and the main function `main` is called when the script is executed directly. 
It utilizes various parameters and configurations defined within the script.
"""

##############################################################################################
#                                                                                            #
#                               IMPORT/INITIALIZATION SECTION                                #
#                                                                                            #
##############################################################################################

import os
import cv2
import time
import concurrent.futures

import numpy as np
import pandas as pd
import mediapipe as mp

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

# Set the required video resolution
frame_width = 640 # Original = 640
frame_height = 480 # Original = 480

# Initialize MediaPipe objects
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Adjust the parameters for FaceDetection
face_detection = mp_face_detection.FaceDetection(
    min_detection_confidence=0.9
)

# Adjust the parameters for FaceMesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    min_detection_confidence=0.9, 
    min_tracking_confidence=0.9,
    max_num_faces=1     
)

# Relevant facial points for both eyes and mouth based on the updated mediapipe landmarks
eye_points = np.array([
    [33, 133], [160, 144], [159, 145], [158, 153],  # Right eye
    [263, 362], [387, 373], [386, 374], [385, 380]   # Left eye
])
mouth_points = np.array([
    [61, 291], [39, 181], [0, 17], [269, 405]        # Mouth
])

# Define the font for displaying text
FONT = cv2.FONT_HERSHEY_SIMPLEX # pylint: disable=no-member

# Function to delete processed files
def process_dataset(folder_paths):
    """
    Delete processed files (".mp4" and ".csv") from the specified folder paths.

    Parameters:
        folder_paths (list): A list of folder paths where processed files need to be deleted.

    Returns:
        None
    """

    # Iterate through the folder paths
    for folder_path in folder_paths:
        # Get a list of all files in the folder
        files = os.listdir(folder_path)

        # Filter out only the "Processed.mp4" files
        processed_files = [file for file in files if file.endswith("pre-processed.mp4")]

        # Iterate through each procesed file names
        for processed_file in processed_files:
            # Delete each "pre-rocessed.mp4" file
            os.remove(os.path.join(folder_path, processed_file))

        # Filter out only the ".csv" files
        csv_files = [file for file in files if file.endswith(".csv")]

        # Iterate through each ".csv" file
        for csv_file in csv_files:
            # Delete each ".csv" file
            os.remove(os.path.join(folder_path, csv_file))

##############################################################################################
#                                                                                            #
#                               FEATURE COMPUTATION SECTION                                  #
#                                                                                            #
##############################################################################################

# Function to calculate the distance between two points
def distance(p1, p2):
    """
    Calculate the Euclidean distance between two points.

    Parameters:
        p1: Tuple containing the coordinates of the first point (x1, y1).
        p2: Tuple containing the coordinates of the second point (x2, y2).

    Returns:
        float: The Euclidean distance between the two points.
    """

    return (((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5)

# Function to calculate Eye Aspect Ratio (EAR)
def compute_eye_aspect_ratio(eye_landmarks):
    """
    Compute the Eye Aspect Ratio (EAR) from facial landmarks.

    Parameters:
        eye_landmarks (array): Array of points defining the eye regions.

    Returns:
        float: The computed Eye Aspect Ratio (EAR).
    """

    # Extract landmarks for the right and left eyes
    right_eye_landmarks = eye_landmarks[:4]
    left_eye_landmarks = eye_landmarks[4:]
    
    # Compute EAR for each eye
    right_ear = compute_ear_for_eye(right_eye_landmarks)
    left_ear = compute_ear_for_eye(left_eye_landmarks)

    # Return the average EAR
    return (right_ear + left_ear) / 2

# Helper function to compute EAR for a single eye
def compute_ear_for_eye(eye_landmarks):
    """
    Compute the Eye Aspect Ratio (EAR) from facial landmarks.

    Parameters:
        eye_landmarks: Array of points defining the eye regions.

    Returns:
        float: The computed Eye Aspect Ratio (EAR).
    """

    # Compute distances between various landmarks of the eye
    eye_D = distance(eye_landmarks[0], eye_landmarks[1])
    eye_N1 = distance(eye_landmarks[1], eye_landmarks[2])
    eye_N2 = distance(eye_landmarks[2], eye_landmarks[3])
    eye_N3 = distance(eye_landmarks[3], eye_landmarks[0])

    # Compute EAR
    ear = (eye_N1 + eye_N2 + eye_N3) / (3 * eye_D)

    # Cap the EAR value between 0 and 2.5
    ear = max(0, min(2.5, ear))

    return ear

# Function to calculate Mouth Aspect Ratio (MAR)
def compute_mouth_aspect_ratio(mouth_landmarks):
    """
    Compute the Mouth Aspect Ratio (MAR) from facial landmarks.

    Parameters:
        mouth_landmarks (array): Array of points defining the mouth region.

    Returns:
        float: The computed Mouth Aspect Ratio (MAR).
    """

    # Compute distances between various landmarks of the mouth
    mouth_D = distance(mouth_landmarks[0], mouth_landmarks[1])
    mouth_N1 = distance(mouth_landmarks[1], mouth_landmarks[2])
    mouth_N2 = distance(mouth_landmarks[2], mouth_landmarks[3])
    mouth_N3 = distance(mouth_landmarks[3], mouth_landmarks[0])

    # Compute MAR
    mar = (mouth_N1 + mouth_N2 + mouth_N3) / (3 * mouth_D)

    # Cap the MAR value between 0 and 5
    mar = max(0, min(5, mar))

    return mar

# Function to estimate head pose (Yaw, Pitch, Roll)
def compute_head_pose(landmarks):
    """
    Calculates head pose (yaw, pitch, roll) from 3D facial landmarks.

    Parameters:
        landmarks: A NumPy array of shape (num_landmarks, 3) containing 3D coordinates of facial landmarks.

    Returns:
        A tuple of (yaw, pitch, roll) in degrees.
    """

    # Calculate the centroid of the landmarks
    centroid = np.mean(landmarks, axis=0)

    # Calculate deviation of each landmark from centroid
    deviations = landmarks - centroid

    # Covariance matrix of the deviations
    covariance_matrix = np.cov(deviations, rowvar=False)

    # Eigen decomposition of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select the eigenvector corresponding to the largest eigenvalue
    direction = eigenvectors[:, 0]

    # Compute yaw, pitch, and roll angles from the direction vector
    yaw = np.arctan2(direction[1], direction[0]) * 180 / np.pi
    pitch = np.arctan2(-direction[2], np.sqrt(direction[0]**2 + direction[1]**2)) * 180 / np.pi
    roll = np.arctan2(eigenvectors[0, 1], eigenvectors[1, 1]) * 180 / np.pi

    # Normalizer
    yaw = (-yaw - 83) * 4
    pitch = (-pitch -8.3) * 4
    roll = (roll - 89) * 5

    # Clamping the angles to the full range
    # Loop as long as yaw is outside the range -180 to 180
    while yaw < -180 or yaw > 180:  
        # Check yaw value
        if yaw < -180:  
            # Increase yaw by 360 to bring it within the range
            yaw += 360  

        elif yaw > 180:  
            # Decrease yaw by 360 to bring it within the range
            yaw -= 360  

    # Loop as long as pitch is outside the range -90 to 90
    while pitch < -90 or pitch > 90:  
        # Check pitch value
        if pitch < -90:  
            # Increase pitch by 180 to bring it within the range
            pitch += 180  

        elif pitch > 90:  
            # Decrease pitch by 180 to bring it within the range
            pitch -= 180  

    # Loop as long as roll is outside the range -180 to 180
    while roll < -180 or roll > 180:  
        # Check roll value
        if roll < -180:  
            # Increase roll by 360 to bring it within the range
            roll += 360  

        elif roll > 180:
            # Decrease roll by 360 to bring it within the range  
            roll -= 360  

    # Capping the angles to the realistic range
    # Cap yaw to -90 to 90
    if yaw > 90:  
        # Cap yaw at 90
        yaw = 90 
        
    elif yaw < -90:  
        # Cap yaw at -90
        yaw = -90 

    # Cap pitch to -90 to 90
    if pitch > 90:  
        # Cap pitch at 90
        pitch = 90 

    elif pitch < -90:  
        # Cap pitch at -90
        pitch = -90  

    # Cap roll to -90 to 90
    if roll > 90:  
        # Cap roll at 90
        roll = 90 

    elif roll < -90:  
        # Cap roll at -90
        roll = -90 

    return yaw, pitch, roll

##############################################################################################
#                                                                                            #
#                                   VISUALIZATION SECTION                                    #
#                                                                                            #
##############################################################################################

# Function to visualize the data when face is detected in the frame
def image_visualization(image, out, bbox, current_frame, testing_state, drowsiness_value, ear_value, mar_value, yaw_value, pitch_value, roll_value):
    """
    Visualize the data when a face is detected in the frame.

    Parameters:
        image (array): The frame image.
        out (cv2.VideoWriter): The video writer object for the processed video.
        bbox (tuple): Bounding box coordinates if a face is detected, otherwise None.
        current_frame (int): The current frame number.
        testing_state (bool): Flag indicating whether to predict.
        drowsiness_value (int): The drowsiness label value.
        ear_value (float): The Eye Aspect Ratio (EAR) value.
        mar_value (float): The Mouth Aspect Ratio (MAR) value.
        yaw_value (float): The yaw value of head pose.
        pitch_value (float): The pitch value of head pose.
        roll_value (float): The roll value of head pose.

    Returns:
        None
    """

    # Display the bounding box and defines the face detected display text
    if bbox is not None:
        # Draw the bounding box on the frame
        # pylint: disable=no-member  
        cv2.rectangle(image, bbox, (0, 255, 0), 2)
        
        # Set "Yes" in face detected display
        face_detected_display = "Yes"

    else:
        # Set "No" in face detected display
        face_detected_display = "No"

    # Check flag whether to predict is flase
    if testing_state is False:
        # Use inital nan values
        ear_display = np.nan
        mar_display = np.nan
        yaw_display = np.nan
        pitch_display = np.nan
        roll_display = np.nan

        # Use N/A for text displays
        text_yaw = "N/A"
        text_pitch = "N/A"
        text_roll = "N/A"

    # Check flag whether to predict is true
    if testing_state is True:
        # Use current value
        ear_display = ear_value
        mar_display = mar_value
        yaw_display = yaw_value
        pitch_display = pitch_value
        roll_display = roll_value

        # Display head pose information
        if yaw_value < -45:
            # When yaw is less than -45, set display to looking left
            text_yaw = "Looking Left"

        elif yaw_value > 45:
            # When yaw is greater than 45, set display to looking right
            text_yaw = "Looking Right"

        else:
            # Otherwise, set the display looking to the front
            text_yaw = "Front"

        if pitch_value < -45:
            # When pitch is less than -45, set display to looking down
            text_pitch = "Looking Down"

        elif pitch_value > 45:
            # When pitch is greater than 45, set display to looking up
            text_pitch = "Looking Up"

        else:
            # Otherwise, set the dispaly looking to the front
            text_pitch = "Front"

        if roll_value < -45:
            # When roll is less than -45, set display to tilted left
            text_roll = "Tilted Left"

        elif roll_value > 45:
            # When roll is greater than 45, set display to titled right
            text_roll = "Tilted Right"

        else:
            # Otherwise, set the display to leveled head
            text_roll = "Level"

    # pylint: disable=no-member    
    # Display additional information for face detected
    text = f"Frame: {current_frame + 1}"
    cv2.putText(image, text, (10, 30), FONT, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"Face Detected: {face_detected_display}", (10, 60), FONT, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"Drowsiness: {drowsiness_value}", (10, 90), FONT, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"EAR: {ear_display:.4f}", (10, 120), FONT, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"MAR: {mar_display:.4f}", (10, 150), FONT, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"Yaw: {yaw_display:.4f} ({text_yaw})", (10, 180), FONT, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"Pitch: {pitch_display:.4f} ({text_pitch})", (10, 210), FONT, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"Roll: {roll_display:.4f} ({text_roll})", (10, 240), FONT, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    out.write(image)

##############################################################################################
#                                                                                            #
#                                       MAIN SECTION                                         #
#                                                                                            #
##############################################################################################

# Define the main function
def main():
    """
    Main function to process video datasets for drowsiness detection.

    Parameters:
        None

    Returns:
        None
    """

    # Record start time
    start_time = time.time()

    # Run the functions for all datasets
    for dataset_path in all_dataset_paths:
        process_dataset(dataset_path)

        # Initialize dataset frame counters
        dataset_frames = 0
        dataset_frames_with_face = 0
        dataset_frames_without_face = 0

        # Process each dataset folder path
        for folder_path in dataset_path:
            print(f"\nProcessing {folder_path}")

            # Iterate over each file and obtain the corresponding video files
            video_files = [file for file in os.listdir(folder_path) if file.endswith(("mix.mp4", "mix.avi"))]

            # Count the total number of video in the folder path
            total_videos = len(video_files)

            # Initialize video counter
            video_count = 0

            # Process each video together with its respective annotation file
            for video_file in video_files:
                # Initialize video frame counters
                current_frames = 0
                current_frames_with_face = 0
                current_frames_without_face = 0
                
                # Extract video name without extension
                video_id = os.path.splitext(video_file)[0]
                
                # Open the video file
                # pylint: disable=no-member
                cap = cv2.VideoCapture(os.path.join(folder_path, video_file))

                # Set the frame resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width) 
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

                # Create a video writer object for processed video
                fps = cap.get(cv2.CAP_PROP_FPS)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                processed_video_id = f"{video_id}_pre-processed.mp4"
                processed_video_path = os.path.join(folder_path, processed_video_id)
                out = cv2.VideoWriter(processed_video_path, fourcc, fps, (frame_width, frame_height))

                # Construct the annotation file name
                annotation_file_id = f"{video_id}ing_drowsiness.txt"
                
                # Check if the annotation files exist
                if annotation_file_id not in os.listdir(folder_path):
                    # Print debugging line if the annotation file does not exist
                    print(f"Annotation file not found for {annotation_file_id}. Skipping...")
                    continue

                # Open the annotation file
                with open(os.path.join(folder_path, annotation_file_id), 'r', encoding='utf-8') as drowsiness_annotation:
                    # Store the characters from the annotation file
                    annotation_content = drowsiness_annotation.read().strip()
                    
                # Convert each character to an integer and create lists for the annotation columns
                drowsiness_values = [int(char) for char in annotation_content]
                
                # Variables to keep track of video
                video_count += 1
                
                # Create a list to store data for each frame
                data = []

                # Flag to track to start prediction
                testing_state = False
                face_detected = False

                # Initialize to store feature values
                frame_ear, frame_mar, frame_yaw, frame_pitch, frame_roll = 999.999, 999.999, 999.999, 999.999, 999.999

                # Loop through each frame in the video
                for index in tqdm(range(len(drowsiness_values)), 
                                desc=f"Processing Video {video_count}/{total_videos} ({video_file})", 
                                ncols=175,
                                unit="frame", 
                                leave=False):                                   
                    
                    # Read the frame
                    ret, frame = cap.read()              

                    # Check if the frame is read successfully
                    if not ret:
                        break

                    # Resize the frame to match the specified resolution
                    frame = cv2.resize(frame, (frame_width, frame_height))
                    
                    # Extract the label for the current frame index
                    label = drowsiness_values[index]

                    # Increment the total frame count
                    current_frames += 1
                    dataset_frames += 1

                    # Perform face detection using MediaPipe
                    results_detection = face_detection.process(frame)
                    
                    # Check if there is a face detected
                    if results_detection.detections:         
                        # Get the bounding box coordinate of the first detected face
                        bboxC = results_detection.detections[0].location_data.relative_bounding_box
                        
                        # Get the dimensions of the frame (height, width, and channels)
                        ih, iw, _ = frame.shape

                        # Calculate the bounding box coordinates in pixels
                        bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                        # Extract landmarks using MediaPipe
                        results_mesh = face_mesh.process(frame)

                        # Check if landmarks are detected
                        if results_mesh.multi_face_landmarks:
                            # Increment the frame with face count
                            current_frames_with_face += 1
                            dataset_frames_with_face += 1

                            # Set flag to 1
                            face_detected = True
                            testing_state = True
                            
                            # Assuming only one face is detected, get landmarks
                            face_landmarks = results_mesh.multi_face_landmarks[0]
                                                 
                            # Display face mesh and connections
                            mp_drawing.draw_landmarks(
                                image=frame,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing.DrawingSpec(color=(192, 128, 0), thickness=1)
                            )

                            # Convert landmarks to a NumPy array for easier processing
                            processed_landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark], dtype=np.float32)

                            # Extract relevant points from processed landmarks
                            eye_landmarks = processed_landmarks[eye_points[:, 0], :]
                            mouth_landmarks = processed_landmarks[mouth_points[:, 0], :]

                            # Create a thread pool executor
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                # Submit tasks to the executor
                                ear_future = executor.submit(compute_eye_aspect_ratio, eye_landmarks)
                                mar_future = executor.submit(compute_mouth_aspect_ratio, mouth_landmarks)
                                head_pose_future = executor.submit(compute_head_pose, processed_landmarks)

                                # Get the results from the futures
                                frame_ear = ear_future.result()
                                frame_mar = mar_future.result()
                                frame_yaw, frame_pitch, frame_roll = head_pose_future.result()

                        else:
                            # Increment the count of frames without detected landmarks
                            current_frames_without_face += 1
                            dataset_frames_without_face += 1

                            # Set flag to 0
                            face_detected = False
                            
                    else:
                        # Increment the count of frames without detected faces
                        current_frames_without_face += 1
                        dataset_frames_without_face += 1

                        # Set flag to 0
                        face_detected = False

                    # Append data for the current frame to the list
                    data.append([current_frames, label, frame_ear, frame_mar, frame_yaw, frame_pitch, frame_roll])

                    # Check whether the face is detected or not
                    if face_detected:
                        # Visualize with face bounding box
                        image_visualization(frame, out, bbox, current_frames, testing_state, label, frame_ear, frame_mar, frame_yaw, frame_pitch, frame_roll)
                        
                    else:
                        # Visualize without face bounding box
                        image_visualization(frame, out, None, current_frames, testing_state, label, frame_ear, frame_mar, frame_yaw, frame_pitch, frame_roll)


                # Get the total number of frames in the video
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 

                # Compute for the number of unprocessed frames
                unprocessed_frames = total_frames - current_frames

                # Release the VideoWriter object at the end of the video
                out.release()
                
                # Release the video capture object
                cap.release()

                # Create a DataFrame from the list of data
                df = pd.DataFrame(data, columns=["Frame", "Drowsiness", "EAR", "MAR", "Yaw", "Pitch", "Roll"]).astype(float)

                # Create the output CSV file name
                output_csv = f"{folder_path}\{video_id}.csv"

                # Save the DataFrame to a CSV file
                df.to_csv(os.path.join(folder_path, output_csv), index=False)

                # Print the results of for each video
                video_info = os.path.join(folder_path, video_id)
                print(f"Video: {video_info.ljust(140)} Frames: {str(current_frames).ljust(10)} Face Detected: {str(current_frames_with_face).ljust(10)} No Face Detected: {str(current_frames_without_face).ljust(10)} Unprocessed Frames: {str(unprocessed_frames).ljust(5)}")

        # Check the dataset path to set the display text
        if dataset_path == training_folder_paths:
            # Set the dataset display text
            dataset_display = "Training Dataset"
            
        elif dataset_path == evaluation_folder_paths:
            # Set the dataset display text
            dataset_display = "Evaluation Dataset"

        else:
            # Set the dataset display text
            dataset_display = "Testing Dataset"
        
        # Print the total frames, with face detected, and without face detected
        print(f"\nTotal Frames in {dataset_display}: {dataset_frames}")
        print(f"Total Frames in {dataset_display} (with face detected): {dataset_frames_with_face}")
        print(f"Total Frames in {dataset_display} (no face detected): {dataset_frames_without_face}")

    # Close MediaPipe instances
    face_detection.close()
    face_mesh.close()

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