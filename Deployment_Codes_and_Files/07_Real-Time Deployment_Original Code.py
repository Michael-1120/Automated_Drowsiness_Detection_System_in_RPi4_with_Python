"""
The program orchestrates the entire process of detecting and predicting drowsiness using facial landmarks in real-time drowsiness detection.

It performs the following tasks:
1. Initialization:
    - Initialize global variables.
    - Initialize trained model.
    - Initialize MediaPipe objects for face detection and face mesh processing with adjusted parameters.
    - Initialize Inputs and Output Pins.
    - Initialize audio paths.
    
2. Time Recording Section:
    - Function 'calculate_frame_fps': Calculate the FPS (Frames Per Second) for the current frame.

3. Feature Computation Section:
    - Function 'distance': Calculates the distance between two points.
    - Function 'compute_eye_aspect_ratio': Computes the average EAR feature for both eyes.
    - Function 'compute_ear_for_eye': Computes the EAR feature of a single eye.
    - Function 'compute_mouth_aspect_ratio': Computes the MAR feature.  
    - Function 'compute_head_pose': Computes the head pose angles through PCA.

4. Model Prediction Section:
    - Function 'Process features for sequencing and predict label with trained model.

5. Parallel Functions Section
    - Function 'check_button_press': Function to handle button press event to start and stop processing loop.
    - Function 'audio_output': Function to handle audio output based on events from a queue.
    - Function 'led_output': Controls LEDs based on signals received from a queue.

6. Other Functions Section
    - Function 'find_camera': Function to find and open a USB camera connected to the Raspberry Pi.
    - Function 'play_mp3': Plays an MP3 file using mpg123 command line utility.

7. Main Section:
    - Function `main`: Iterates through each frame of each video of each datasets and calls necessary functions.

The program is designed to be run as a standalone module, and the main function `main` is called when the script is executed directly. 
It utilizes various parameters and configurations defined within the script.
"""

##############################################################################################
#                                                                                            #
#                               IMPORT/INITIALIZATION SECTION                                #
#                                                                                            #
##############################################################################################

import cv2
import time
import queue
import threading
import concurrent.futures

import numpy as np
import pandas as pd
import soundfile as sf
import mediapipe as mp
import tensorflow as tf
import RPi.GPIO as GPIO
import sounddevice as sd

# Define the feature to be included in the testing
#selected_features = ['EAR']                                                             # For EAR Feature Only Test 
#selected_features = ['MAR']                                                             # For MAR Feature Only Test
#selected_features = ['Yaw', 'Pitch', 'Roll']                                            # For HP Feature Only Test
#selected_features = ['EAR', 'MAR']                                                      # For EAR and MAR Features Test
#selected_features = ['EAR', 'Yaw', 'Pitch', 'Roll']                                     # For EAR and HP Features Test
#selected_features = ['MAR', 'Yaw', 'Pitch', 'Roll']                                     # For MAR and HP Features Test
selected_features = ['EAR', 'MAR', 'Yaw', 'Pitch', 'Roll']                              # For All Feature Test

# Load the TFLite Sleepy Model
MODEL_PATH = r"/home/Quantized_Student_Model.tflite"
STUDENT_INTERPRETER = tf.lite.Interpreter(model_path=MODEL_PATH)
STUDENT_INTERPRETER.allocate_tensors()
STUDENT_INPUT_DETAILS = STUDENT_INTERPRETER.get_input_details()
STUDENT_OUTPUT_DETAILS = STUDENT_INTERPRETER.get_output_details()

# Global variable for monitoring
PREDICTION_COUNT = 0

# Set the desired FPS, width, and height
FPS = 10
FRAME_WIDTH = 640  # Original = 640
FRAME_HEIGHT = 480  # Original = 480

# Maximum allowed processing time per frame (in seconds)
MAX_PROCESSING_TIME = 0.095  

# Define initial frame for initial average
initial_frame = 10 # From Data Analysis

# Define the unstable frames for processing
UNSTABLE_FRAMES = 10 # For Mediapipe initial instability

# Initialize MediaPipe objects
MP_FACE_DETECTION = mp.solutions.face_detection
MP_DRAWING = mp.solutions.drawing_utils
MP_FACE_MESH = mp.solutions.face_mesh

# Adjust the parameters for FaceDetection
FACE_DETECTION = MP_FACE_DETECTION.FaceDetection(
    min_detection_confidence=0.9
)

# Adjust the parameters for FaceMesh
FACE_MESH = MP_FACE_MESH.FaceMesh(
    static_image_mode=False,
    min_detection_confidence=0.9, 
    min_tracking_confidence=0.9,
    max_num_faces=1     
)

# Relevant facial points for both eyes and mouth based on the updated mediapipe landmarks
EYE_POINTS = np.array([[33, 133], [160, 144], [159, 145], [158, 153],  # Right eye
                       [263, 362], [387, 373], [386, 374], [385, 380]]) # Left eye
MOUTH_POINTS = np.array([[61, 291], [39, 181], [0, 17], [269, 405]])    # Mouth

# Processing constants
ALPHA_EAR = 0.15
ALPHA_MAR = 0.15
ALPHA_YAW = 0.25
ALPHA_PITCH = 0.3
ALPHA_ROLL = 0.4
EAR_DIFF_THRESHOLD = 0.5
MAR_DIFF_THRESHOLD = 0.5
YAW_DIFF_THRESHOLD = 30
PITCH_DIFF_THRESHOLD = 30
ROLL_DIFF_THRESHOLD = 30

# Create initial data sequence
INITIAL_DATA_DF = pd.DataFrame(np.zeros((100, 5), dtype=np.float32), 
                               columns=['EAR', 'MAR', 'Yaw', 'Pitch', 'Roll'])

# Local Global variable for processing and predicting
PAST_VALUES = {'EAR': 0.0, 'MAR': 0.0, 'Yaw': 0.0, 'Pitch': 0.0, 'Roll': 0.0}
INITIAL_FRAMES_DATA = pd.DataFrame(columns=['EAR', 'MAR', 'Yaw', 'Pitch', 'Roll'])
INITIAL_FRAMES_AVERAGE = pd.Series({col: 0.0 for col in ['EAR', 'MAR', 'Yaw', 'Pitch', 'Roll']})
SEQUENCE = np.expand_dims(INITIAL_DATA_DF.values, axis=0)

# Define prediction weights
PREDICTION_THRESHOLD = 0.5

# Pin Definitions
# LED GROUND    # Pin 9
RED_PIN = 17    # GPIO 17 (Pin 11)
YELLOW_PIN = 27 # GPIO 27 (Pin 13)
GREEN_PIN = 22  # GPIO 22 (Pin 15)

BUTTON_PIN = 5    # GPIO 5 (Pin 29)
# BUTTON GROUND   # Pin 30

# Pin Setup
GPIO.setwarnings(False)  # Disable GPIO warnings
GPIO.setmode(GPIO.BCM)
GPIO.setup(GREEN_PIN, GPIO.OUT)
GPIO.setup(YELLOW_PIN, GPIO.OUT)
GPIO.setup(RED_PIN, GPIO.OUT)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Initialize all LEDs to OFF
GPIO.output(GREEN_PIN, GPIO.LOW)
GPIO.output(YELLOW_PIN, GPIO.LOW)
GPIO.output(RED_PIN, GPIO.LOW)

# Paths to MP3 files
SYSTEM_INITIALIZING_MP3 = r"/home/Audio Files/system_initializing.mp3"
NO_CAMERA_DETECTED_MP3 = r"/home/Audio Files/no_camera_detected.mp3"
STARTING_DETECTION_MP3 = r"/home/Audio Files/starting_drowsiness_detection.mp3"
FACE_DETECTED_MP3 = r"/home/Audio Files/face_detected.mp3"
FACE_NOT_DETECTED_MP3 = r"/home/Audio Files/face_not_detected.mp3"
DROWSINESS_DETECTED_MP3 = r"/home/Audio Files/drowsiness_detected.mp3"
SHUTTING_DOWN_MP3 = r"/home/Audio Files/shutting_down.mp3"

##############################################################################################
#                                                                                            #
#                                   TIME RECORDING SECTION                                   #
#                                                                                            #
##############################################################################################

# Function to calculate the frames per second (FPS) for the current frame
def calculate_frame_fps(frame_time):
    """
    Calculate the FPS (Frames Per Second) for the current frame.

    Parameters:
        frame_time: The start time of processing the current frame.

    Returns:
        The FPS (Frames Per Second) for the current frame.
    """

    # Obtain the current time
    frame_end = time.time()

    # Obtain the processing time for the current frame
    frame_processing_time = frame_end - frame_time

    # Check if the processing time is 0
    if frame_processing_time == 0:
        # Handle division by zero error
        fps = float('inf')  

    else:
        # Obtain the fps given a time
        fps = 1 / frame_processing_time
    
    return fps

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
    eye_d = distance(eye_landmarks[0], eye_landmarks[1])
    eye_n1 = distance(eye_landmarks[1], eye_landmarks[2])
    eye_n2 = distance(eye_landmarks[2], eye_landmarks[3])
    eye_n3 = distance(eye_landmarks[3], eye_landmarks[0])

    # Compute EAR
    ear = (eye_n1 + eye_n2 + eye_n3) / (3 * eye_d)

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
    mouth_d = distance(mouth_landmarks[0], mouth_landmarks[1])
    mouth_n1 = distance(mouth_landmarks[1], mouth_landmarks[2])
    mouth_n2 = distance(mouth_landmarks[2], mouth_landmarks[3])
    mouth_n3 = distance(mouth_landmarks[3], mouth_landmarks[0])

    # Compute MAR
    mar = (mouth_n1 + mouth_n2 + mouth_n3) / (3 * mouth_d)

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
#                                 MODEL PREDICTION SECTION                                   #
#                                                                                            #
##############################################################################################

# Function to process the data and predict with a consistent sequence 
def process_and_predict(ear, mar, yaw, pitch, roll):
    """
    Process features for sequencing and predict label with trained models.

    Parameters:
        ear (float): EAR (Eye Aspect Ratio) value.
        mar (float): MAR (Mouth Aspect Ratio) value.
        yaw (float): Yaw angle value.
        pitch (float): Pitch angle value.
        roll (float): Roll angle value.

    Returns:
        int: Predicted label (0 or 1).
    """

    # Disable pylint warning for statments for necessary program logic
    # pylint: disable=global-variable-not-assigned
    # pylint: disable=global-statement  
    global PREDICTION_COUNT # Integer

    global PAST_VALUES # Local Global Dictionary
    global INITIAL_FRAMES_DATA #Local Global DataFrame
    global INITIAL_FRAMES_AVERAGE # Local Global Pandas Series
    global SEQUENCE # Local Global Numpy Array

    # Increment predictions count
    PREDICTION_COUNT += 1
    
    # Load the computed features to their respective data holder
    processed_ear, processed_mar, processed_yaw, processed_pitch, processed_roll = ear, mar, yaw, pitch, roll

    # Process the values for succeeding predictions
    if PREDICTION_COUNT > 1:

        # Apply difference threshold to the new features using past values
        ear_diff = abs(ear - PAST_VALUES['EAR'])
        mar_diff = abs(mar - PAST_VALUES['MAR'])
        yaw_diff = abs(yaw - PAST_VALUES['Yaw'])
        pitch_diff = abs(pitch - PAST_VALUES['Pitch'])
        roll_diff = abs(roll - PAST_VALUES['Roll'])

        # Apply thresholding for EAR feature
        if ear_diff >= EAR_DIFF_THRESHOLD:
            # Set the current feature to use the past value
            processed_ear = PAST_VALUES['EAR']

        # Apply thresholding for MAR feature
        if mar_diff >= MAR_DIFF_THRESHOLD:
            # Set the current feature to use the past value
            processed_mar = PAST_VALUES['MAR']
        
        # Apply thresholding for Yaw feature
        if yaw_diff >= YAW_DIFF_THRESHOLD:
            # Set the current feature to use the past value
            processed_yaw = PAST_VALUES['Yaw']

        # Apply thresholding for Pitch feature
        if pitch_diff >= PITCH_DIFF_THRESHOLD:
            # Set the current feature to use the past value
            processed_pitch = PAST_VALUES['Pitch']

        # Apply thresholding for Roll feature
        if roll_diff >= ROLL_DIFF_THRESHOLD:
            # Set the current feature to use the past value
            processed_roll = PAST_VALUES['Roll']

        # Apply EMA smoothing
        processed_ear = ALPHA_EAR * processed_ear + (1 - ALPHA_EAR) * PAST_VALUES['EAR']
        processed_mar = ALPHA_MAR * processed_mar + (1 - ALPHA_MAR) * PAST_VALUES['MAR']
        processed_yaw = ALPHA_YAW * processed_yaw + (1 - ALPHA_YAW) * PAST_VALUES['Yaw']
        processed_pitch = ALPHA_PITCH * processed_pitch + (1 - ALPHA_PITCH) * PAST_VALUES['Pitch']
        processed_roll = ALPHA_ROLL * processed_roll + (1 - ALPHA_ROLL) * PAST_VALUES['Roll']

    # Checker to set features not in selected_features to 0
    new_row = {
        'EAR': processed_ear if 'EAR' in selected_features else 0,
        'MAR': processed_mar if 'MAR' in selected_features else 0,
        'Yaw': processed_yaw if 'Yaw' in selected_features else 0,
        'Pitch': processed_pitch if 'Pitch' in selected_features else 0,
        'Roll': processed_roll if 'Roll' in selected_features else 0
    }

    # Update past values for next frame
    PAST_VALUES['EAR'] = processed_ear
    PAST_VALUES['MAR'] = processed_mar
    PAST_VALUES['Yaw'] = processed_yaw
    PAST_VALUES['Pitch'] = processed_pitch
    PAST_VALUES['Roll'] = processed_roll

    # Store initial frames data and calculate average for normalization
    if PREDICTION_COUNT <= initial_frame:
        # Obtain the initial data for normalization
        if INITIAL_FRAMES_DATA.empty:
            # When the df is empty, use the first data
            INITIAL_FRAMES_DATA = pd.DataFrame([new_row])

        else:
            # For succeeding frames, concatenate the data
            INITIAL_FRAMES_DATA = pd.concat([INITIAL_FRAMES_DATA, pd.DataFrame([new_row])], ignore_index=True)

        # Calculate the initial data's average for each feature
        INITIAL_FRAMES_AVERAGE = INITIAL_FRAMES_DATA.mean(axis=0)
    
        # Normalize initial frames data using INITIAL_FRAMES_AVERAGE
        INITIAL_FRAMES_DATA_NORMALIZED = INITIAL_FRAMES_DATA - INITIAL_FRAMES_AVERAGE

        # Remove the last rows from SEQUENCE based on PREDICTION_COUNT
        SEQUENCE = SEQUENCE[:, :-PREDICTION_COUNT, :]

        # Reshape INITIAL_FRAMES_DATA_NORMALIZED to match SEQUENCE dimensions
        INITIAL_FRAMES_DATA_NORMALIZED = INITIAL_FRAMES_DATA_NORMALIZED.values.reshape(1, PREDICTION_COUNT, 5)

        # Concatenate normalized initial frames data to SEQUENCE along the time axis
        SEQUENCE = np.concatenate([SEQUENCE, INITIAL_FRAMES_DATA_NORMALIZED], axis=1)

    else:
        # Normalize new_row using INITIAL_FRAMES_AVERAGE
        new_row_normalized = pd.Series(new_row) - INITIAL_FRAMES_AVERAGE

        # Reshape new_row_normalized to match the sequence shape
        new_row_reshaped = pd.Series(new_row_normalized).values.reshape(1, 1, -1)

        # Remove the first row from the sequence
        SEQUENCE = SEQUENCE[:, 1:, :]

        # Concatenate new_row_reshaped to SEQUENCE along the first axis
        SEQUENCE = np.concatenate([SEQUENCE, new_row_reshaped], axis=1)

    # Convert sequence to FLOAT32
    SEQUENCE = SEQUENCE.astype(np.float32)

    # Predict using the Student model
    STUDENT_INTERPRETER.set_tensor(STUDENT_INPUT_DETAILS[0]['index'], SEQUENCE)
    STUDENT_INTERPRETER.invoke()
    model_prediction = STUDENT_INTERPRETER.get_tensor(STUDENT_OUTPUT_DETAILS[0]['index'])

    # Apply threshold if needed
    predicted_label = 1 if model_prediction > PREDICTION_THRESHOLD else 0

    return predicted_label

##############################################################################################
#                                                                                            #
#                                 PARALLEL FUNCTIONS SECTION                                 #
#                                                                                            #
##############################################################################################

# Function to handle button press to start and stop the loop
def check_button_press(start_event, stop_event):
    """
    Function to handle button press event to start and stop processing loop.

    Parameters:
        start_event (threading.Event): Event to signal start of processing loop.
        stop_event (threading.Event): Event to signal stop of processing loop.

    Returns:
        None
    """
    
    # Sub-function to call when button is pressed
    def on_button_press():
        # Check if the start event is not set
        if not start_event.is_set():
            # Set the start event
            start_event.set()
            print("\nStarted processing frames.")
        
        else:
            # Set the stop event
            stop_event.set()
            print("\nStopping processing frames.")
    
    # Wait indefinitely until stop_event is set
    while not stop_event.is_set():
        # Check for button press
        if GPIO.input(BUTTON_PIN) == GPIO.HIGH:  # Normally Closed button will be HIGH when pressed
            # Call the function to start or to end the program
            on_button_press()

            # Wait until the button is released before looping again
            while GPIO.input(BUTTON_PIN) == GPIO.HIGH:
                # Delay Timer
                time.sleep(0.1)
        
        # Delay Timer
        time.sleep(0.1)

    # Clean up: Reset GPIO configuration
    GPIO.cleanup()  

# Function for audio output
def audio_output(audio_queue, start_event, stop_event):
    """
    Function to handle audio output based on events from a queue.

    Parameters:
        audio_queue (Queue): Queue containing audio output events.
        start_event (Event): Event signaling the start of drowsiness detection.
        stop_event (Event): Event signaling to stop the function.

    Returns:
        None
    """

    # Play system initializing audio
    play_mp3(SYSTEM_INITIALIZING_MP3)  
    #print("\nSystem Initializing.")

    # Let Full Audio Phrase End
    time.sleep(2)

    # Wait for the start event to be set
    while not start_event.is_set():
        time.sleep(0.1)  # Add a small delay to avoid high CPU usage

    # Print/audio for start of real-time drowsiness detection
    play_mp3(STARTING_DETECTION_MP3)
    #print("\nStarting Real-Time Drowsiness Detection.")

    # Let Full Audio Phrase End
    time.sleep(3)

    # Initialize variable to track first face detection
    first_face_detected = True

    # Looping system while stop event is not set
    while not stop_event.is_set():
        try:
            # Get flags from the audio queue
            face_detect, state_change, predicted_label = audio_queue.get(timeout=1.0)
            
            # Print/audio for face detection
            if face_detect and (first_face_detected or state_change):
                # Code for audio alert
                play_mp3(FACE_DETECTED_MP3)
                #print("Face Detected")

                # Let Full Audio Phrase End
                time.sleep(1)

                # Update flag to indicate first face detection
                first_face_detected = False  

            elif not face_detect and (first_face_detected or state_change):
                # Code for audio alert
                play_mp3(FACE_NOT_DETECTED_MP3)
                #print("Face Not Detected")

                # Let Full Audio Phrase End 
                time.sleep(1.5)
                
                # Update flag to indicate first face detection
                first_face_detected = False  
            
            # Print/audio for predicted label (drowsiness detected)
            if face_detect and predicted_label == 1:
                # Code for audio warning
                play_mp3(DROWSINESS_DETECTED_MP3)
                #print("Drowsiness Detected")

                # Let Full Audio Phrase End
                time.sleep(2)

            # Clear items in the audio queue based on state_change
            while not audio_queue.empty():
                # Retrieve the first queue item
                item = audio_queue.get_nowait()

                # Check if the state change is true
                if item[1]:
                    # Loop through the queue for clearing
                    while not audio_queue.empty():
                        # Get the queue
                        audio_queue.get_nowait()

                    # Retain the current item with the first true state_change
                    audio_queue.put(item)  
                    
                    break

        except queue.Empty:
            # Continue for next processing
            continue

    # Play shutting down audio
    play_mp3(SHUTTING_DOWN_MP3)
    #print("Shutting Down...")

    # Let shutting down audio finish
    time.sleep(5) 
        
# Function for LED output (Green, Yellow, Red combined)
def led_output(led_queue, stop_event):
    """
    Controls LEDs based on signals received from a queue.

    Parameters:
        led_queue (queue.Queue): Queue containing signals (yellow_signal, red_signal).
        stop_event (threading.Event): Event to stop the function.

    Returns:
        None
    """

    # Turn on Green LED to indicate system operation
    GPIO.output(GREEN_PIN, GPIO.HIGH)  
    #print("System in Operation (Green LED on)")

    # Initialize flag
    face_detection = False

    # Looping system while program is not stop
    while not stop_event.is_set():
        try:
            # Retrieved the data from queue
            signal = led_queue.get(timeout=1.0)

            # Separate the touple data
            yellow_signal, red_signal = signal

            # Check the yellow signal data and the flag
            if yellow_signal and not face_detection:
                # Turn on Yellow LED
                GPIO.output(YELLOW_PIN, GPIO.HIGH)
                #print("Face Detected (Yellow LED on)")

                # Set flag
                face_detection = True

            elif not yellow_signal and face_detection:
                # Turn off Yellow LED
                GPIO.output(YELLOW_PIN, GPIO.LOW)
                #print("Face Not Detected (Yellow LED off)")

                # Clear flag
                face_detection = False
            
            # Check the red signal data
            if red_signal:
                # Turn on Red LED
                GPIO.output(RED_PIN, GPIO.HIGH)
                #print("Drowsiness Detected (Red LED on)")

            else:
                # Turn off Red LED
                GPIO.output(RED_PIN, GPIO.LOW)
                #print("No Drowsiness Detected (Red LED off)")

        except queue.Empty:
            # Skip current queue if empty
            continue
    
    # Turn off all LEDs when stopping
    GPIO.output(GREEN_PIN, GPIO.LOW)
    GPIO.output(YELLOW_PIN, GPIO.LOW)
    GPIO.output(RED_PIN, GPIO.LOW)
    GPIO.cleanup()
    
##############################################################################################
#                                                                                            #
#                                  OTHER FUNCTIONS SECTION                                   #
#                                                                                            #
##############################################################################################

# Function to find and open a USB camera
def find_camera():
    """
    Function to find and open a USB camera connected to the Raspberry Pi.

    Parameters:
        None

    Returns:
        cv2.VideoCapture object: The video capture object for the detected camera.
        None: If no camera is found.
    """

    # pylint: disable=no-member
    
    # Iterate through possible camera indices
    for i in range(4):  
        # Open the video capture
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)

        # Check if the camera is open
        if cap.isOpened():
            # Return the cap if the camera is found
            print(f"Camera found at index {i}")
            return cap
        
        # Code for audio warning
        play_mp3(NO_CAMERA_DETECTED_MP3)
        
        # Let Full Audio Phrase End
        time.sleep(3)

        # Release the cap
        cap.release()

    # If no camera found, return None
    print("No camera detected.")

    return None

# Function to play a WAV file using sounddevice
def play_mp3(filename):
    """
    Plays a mp3 file using sounddevice.

    Parameters:
        filename (str): The name or path of the mp3 file to be played.

    Returns:
        None
    """

    try:
        # Load the mp3 file
        data, fs = sf.read(filename, dtype='float32')

        # Play the audio
        sd.play(data, fs)

        # Wait until the file is done playing
        #sd.wait()  

    except Exception as e:
        # If an error occurs, print an error message
        print(f"Error playing {filename}: {e}")
        
    
##############################################################################################
#                                                                                            #
#                                       MAIN SECTION                                         #
#                                                                                            #
##############################################################################################

# Define the main function
def main():
    """
    Main function to process 

    Parameters:
        None

    Returns:
        None
    """

    # pylint: disable=no-member

    # Find the USB camera
    cap = find_camera()

    # Check if the camera was found
    if cap is None:
        print("No camera found. Exiting.")
        return

    # Set camera parameters
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Ensure camera settings are correctly applied
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(f"\nActual camera settings: FPS={actual_fps}, Width={actual_width}, Height={actual_height}\n")

    # Flag to track to start prediction
    model_testing = False

    # Initialize flag to store previous face detection state
    prev_face_detected = False

    # Mediapipe prediction counter
    mediapipe_prediction = 0

    # Initialize to store feature values
    frame_ear, frame_mar, frame_yaw, frame_pitch, frame_roll = 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Initialize initial prediction
    predicted_value = 0

    # Flags and Queues
    start_event = threading.Event()
    stop_event = threading.Event()

    led_queue = queue.Queue()
    audio_queue = queue.Queue()
    
    # Start the button press checking thread
    button_press_thread = threading.Thread(target=check_button_press, args=(start_event, stop_event))
    button_press_thread.start()

    # Start the audio output thread
    audio_thread = threading.Thread(target=audio_output, args=(audio_queue, start_event, stop_event))
    audio_thread.start()

    # Start the combined LED output thread
    led_thread = threading.Thread(target=led_output, args=(led_queue, stop_event))
    led_thread.start()

    # Wait for the button press to start the loop
    print("\nPress button to start real-time drowsiness detection.")
    start_event.wait()

    # Main loop for real-time drowsiness detection
    while not stop_event.is_set():
        # Record the start time of processing the frame
        frame_start_time = time.time()
        
        # Read the frame from the video capture object
        ret, frame = cap.read()
        
        # Check if the frame is read successfully
        if not ret:
            print("\nDebugging Line.")
            break
        
        # Resize the frame to match the specified resolution
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Initialize flag to track current face detection state
        face_detected = False

        # Initialize state monitoring flag
        face_state_change = False

        # Perform face detection using MediaPipe
        results_detection = FACE_DETECTION.process(frame)
        
        # Perform feature computation if face is detected
        if results_detection.detections: 
            # Extract landmarks using MediaPipe
            results_mesh = FACE_MESH.process(frame)
            
            # Check if landmarks are detected
            if results_mesh.multi_face_landmarks:
                # Set face detection monitoring flag
                face_detected = True

                # Assuming only one face is detected, get landmarks
                face_landmarks = results_mesh.multi_face_landmarks[0]

                # Convert landmarks to a NumPy array for easier processing
                processed_landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])

                # Extract relevant points from processed landmarks for EAR computation
                eye_landmarks = processed_landmarks[EYE_POINTS[:, 0], :]

                # Extract relevant points from processed landmarks for MAR computation
                mouth_landmarks = processed_landmarks[MOUTH_POINTS[:, 0], :]

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

                # Check if model_testing is still False
                if not model_testing:
                    # Check if yaw, pitch, and roll are within acceptable limits (excluding -90 and 90)
                    if -90 < frame_yaw < 90 and -90 < frame_pitch < 90 and -90 < frame_roll < 90:
                        # Set model_testing to True as the values are within the acceptable range
                        model_testing = True
                        
                    else:
                        # Initialize to store feature values
                        frame_ear, frame_mar, frame_yaw, frame_pitch, frame_roll = 0.0, 0.0, 0.0, 0.0, 0.0

                        # Skip the current frame processing
                        continue

        # Check if the model testing is true or that the program has started predicting labels
        if model_testing is True:
            # Increment mediapipe prediction counter
            mediapipe_prediction += 1

            # Skip the first 'unstable_frames' frames
            if mediapipe_prediction <= UNSTABLE_FRAMES:
                # Skip the current frame
                continue

            # Process and Predict with the computed data
            predicted_value = process_and_predict(frame_ear, frame_mar, frame_yaw, frame_pitch, frame_roll)

        # After processing, call calculate_frame_fps() to get the FPS for the frame
        frame_fps = calculate_frame_fps(frame_start_time)

        # Create the status message
        print(f"Image FPS: {frame_fps:5.2f} ms, "
              f"Prediction: {predicted_value}")
        
        # Check if face detection state changed
        if face_detected != prev_face_detected:
            # Set state monitoring flag
            face_state_change = True

        # Update previous face detection state
        prev_face_detected = face_detected

        # Calculate how much time has passed
        elapsed_time = time.time() - frame_start_time

        # Adjust sleep time if processing takes longer than expected
        if elapsed_time < MAX_PROCESSING_TIME:
            sleep_time = max(0, (1 / FPS) - elapsed_time)
            time.sleep(sleep_time)

        # Queue LED signals
        led_queue.put((face_detected, predicted_value == 1))

        # Queue audio and LED signals
        audio_queue.put((face_detected, face_state_change, predicted_value))

    # Release the video capture object
    cap.release()

    # Close MediaPipe instances
    FACE_DETECTION.close()
    FACE_MESH.close()

    # Wait for all parallel events to end
    button_press_thread.join()
    audio_thread.join()
    led_thread.join() 

# Check if the script is being run directly
if __name__ == "__main__":   
    # Call the main function
    main()

##############################################################################################
#                                                                                            #
#                                        END SECTION                                         #
#                                                                                            #
##############################################################################################