#  Multi-Feature LSTM Facial Recognition for Real-Time Automated Drowsiness Observation of Automobile Drivers with Raspberry Pi 4

## Table of Contents
- [Introduction](#introduction)
- [Objectives](#objectives)
- [Methodology](#methodology)
  - [Training](#training)
  - [Prototype Assembly](#prototype-assembly)
  - [Testing](#testing)
- [Code Flow](#code-flow)
- [Proponents](#proponents)
- [Published Paper/Journal Article Link](#published-paperjournal-article-link)

## Introduction

Driver fatigue is a significant cause of traffic accidents globally. This project focuses on developing an efficient and real-time drowsiness detection system using facial features like eye aspect ratio (EAR), mouth aspect ratio (MAR), and head pose estimation (Yaw, Pitch, Roll). The proposed system runs on the Raspberry Pi 4, leveraging its computational capability for embedded solutions.

## Objectives

The main goal of the study is to train, evaluate, and develop a drowsiness detection model using the NTHU-DDD Dataset, LSTM Deep Learning Algorithm, and the behavioral features EAR, MAR, and Head Pose Angles. The specific objectives of the study are as follows:
1)	To design and develop a hardware prototype system that captures live image frames from a web camera and configures the system to predict the driver’s drowsiness state as audio and light alerting signal outputs.
2)	To calibrate the camera placement for the optimum distance, height, and angular position from the face of each participants inside the car.
3)	To evaluate and categorize the developed system for all behavioral feature combinations: eye behaviors only; mouth behaviors only; head pose only; both eye and mouth behavior; both eye behavior and head pose; both mouth behavior and head pose; and all behavioral features varying conditions such as stationary and in-motion vehicle while testing

---

## Methodology

### Conceptual Framework
![Conceptual Framework](/Images/Conceptual_Framework.png)  
This framework illustrates the relationship between input, process, and output components of the drowsiness detection system. The input captures behavioral data from the driver's face, the process involves detection and classification algorithms, and the output triggers audio and light alerts when drowsiness is detected.


### General Flow of the Study
![General Flow of the Study](/Images/General_Flow_of_the_Study.png)   
This diagram outlines the overall workflow, starting from data collection and preprocessing to training, evaluation, and real-time deployment. Each step ensures that the system operates accurately in real-time conditions.

----

### Training Process

#### Dataset Overview
![NTHU-DDD](/Images/NTHU-DDD_Sample_Images.jpg)  
Sample images from the NTHU-DDD dataset used for training the model. This dataset contains diverse scenarios such as variations in lighting, head pose, and facial behaviors, providing robust training data for the detection system.  
For additional resources and ongoing updates of the dataset, contact the dataset provider, [National Tsing Hua University for the NTHU-DDD.](https://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/)

#### Preprocessing
![Pre-Processing Flowchart](/Images/Pre-Processing_Flowchart.png)  
The preprocessing flowchart details the steps for preparing the data. Key processes include feature computation (EAR, MAR, Yaw, Pitch, Roll), normalization, and handling class imbalance to ensure optimal model performance.

#### Training and Validation
![Training and Validation Flowchart](/Images/Training_and_Validation_Flowchart.png)  
This flowchart illustrates how the training and validation phases are conducted. It includes splitting the dataset, training LSTM models, and assessing the model’s accuracy on the validation set.

#### Evaluation
![Evaluation Flowchart](/Images/Evaluation_Flowchart.png)   
The evaluation flowchart shows the steps taken to test the model on unseen data. This step ensures the robustness and generalizability of the system across different scenarios.

----

### Prototype Assembly

#### Schematic Design
![Schematic Diagram Design of the Prototype](/Images/Prototype_Schematic_Diagram.png)   
The schematic diagram provides a detailed view of the hardware connections, including the Raspberry Pi, camera, and output components (audio and light signals).

#### Assembled Prototype
![Assembled Prototype](/Images/Assembled_Prototype.jpg)  
This image displays the fully assembled prototype, integrating all hardware components within a compact and functional design.

----

### Testing

#### Front-View Setup
![Front-View of the Setup](/Images/Front-View_Setup.png)  
The front-view setup highlights the camera placement and participant positioning, ensuring optimal feature capture for real-time testing.

#### Side-View Setup
![Side-View of the Setup](/Images/Side-View_Setup.png)  
The side-view image provides additional perspective on the setup, focusing on the ambient environment and hardware alignment.

#### Real-Time Deployment
![Real-Time Deployment Flowchart](/Images/Real-Time_Deployment_Flowchart.png)  
This flowchart explains the real-time operation of the system, from capturing live frames to processing and triggering alerts for drowsiness detection.

---

## Code Flow

### PC/Laptop Program Codes
1. **`01_Combine Video Datasets by Category.py`**
   - Organizes training, evaluation, and testing datasets by category.

2. **`02_Process Video Datasets.py`**
   - Processes videos to compute features (EAR, MAR, Yaw, Pitch, Roll) and generate CSV files for each video.

3. **`03_Video Dataset Statistics_Box Plots.py`**
   - Analyzes feature statistics and generates box plots for each participant’s dataset.

4. **`04_Train Model_EAR MAR Head Pose.py`**
   - Trains LSTM models with computed features and saves the trained models.

5. **`05_Evaluate Model.py`**
   - Evaluates trained models using metrics like accuracy, precision, recall, and F1-score.

6. **`06_Model Optimization.py`**
   - Optimizes models through knowledge distillation, pruning, and quantization for efficient deployment.


### Setting Up Raspberry Pi 4 Prototype

1. **Imaging the Raspberry Pi 4 SD Card**  
   - Download the appropriate Raspberry Pi OS (e.g., Lite or Desktop version) from the official Raspberry Pi website.  
   - Use tools like **Raspberry Pi Imager** or **balenaEtcher** to flash the OS image onto the SD card.  

2. **System Setup**  
   - Insert the SD card into the Raspberry Pi 4, connect it to a power supply, and boot up the system.  
   - Access the Raspberry Pi via **SSH** (use `hostname -I` to get the IP address) or connect it directly to a monitor and keyboard.  
   - Update the system and install essential tools:  
     ```bash
     sudo apt update && sudo apt full-upgrade -y  # Update and upgrade the system
     sudo apt install python3 python3-pip python3-venv -y  # Install Python and its tools
     sudo reboot  # Reboot to apply updates
     ```

3. **Deployment**  
   - Transfer the trained models and deployment script to the Raspberry Pi using SCP or a USB drive. Example SCP command:  
     ```bash
     scp -r /path/to/your/files rpi4@<RPi_IP>:/home/pi/
     ```  
   - Set up the Raspberry Pi to execute the deployment script on startup:  
     - Create a systemd service file:  
       ```bash
       sudo nano /etc/systemd/system/deployment.service
       ```  
     - Add the following content to the file:  
       ```
       [Unit]
       Description=Run deployment script on startup
       After=network.target

       [Service]
       ExecStart=/usr/bin/python3 /home/pi/deployment_script.py
       WorkingDirectory=/home/pi
       Restart=on-failure
       User=pi

       [Install]
       WantedBy=multi-user.target
       ```  
     - Enable and start the service:  
       ```bash
       sudo systemctl daemon-reload
       sudo systemctl enable deployment.service
       sudo reboot
       ```  

---

## Published Paper/Journal Article Link

For a comprehensive overview of the methodology, detailed testing results, and in-depth discussions, please refer to our [published journal article.](https://www.mdpi.com/2673-4591/92/1/52).

## Notes

For detailed setup, refer to individual scripts. Ensure dependencies are installed, and adjust paths as needed for your system configuration.

