# Controlling gym environment MountainCar-v0 Using Hand Gestures

Pre-constraints - Not using Existing Dataset or Pre-trained Models for hand/gesture/face

# Quick Start

## Introduction

As we impose ourselves with constraint such as not using any pre-trained model of face /hand for acheiving our task.
we will rely on core computer vision concepts such as background subtraction, motion detection and thresholding.
The main goal is to control the MountainCar-v0 using the gestures done infront of our webcam feed.

## MountainCar-v0 gym  environment

Open AI Gym environment is pretty straightforward to setup, Run `pip install gym'` 
As we have three states/actions to control MountainCar-v0 [left, right and stop], we will choose 3 gestures that map to a certain action.

Dataset![forward_correct](https://user-images.githubusercontent.com/67980173/190501977-4e5f7e18-b6ab-4cea-a3a7-ce56510f365d.png)

![back_correct](https://user-images.githubusercontent.com/67980173/190501903-26dec0bf-38d2-4646-8cb6-8dd8715b52e1.png)  ![result_wrong](https://user-images.githubusercontent.com/67980173/190502009-1a52d2fe-a114-4da9-b0b9-19756c146bac.png)



  
## Generating/Capturing Dataset
Before Moving to any training/ detection part, we will directly deep dive into how to generate a custom gesture dataset 
that can help us classify the different signs that we perform infront of our webcam feed. As we are generating dataset from the scratch  
we will follow the guidelines of ImageFolderDataset Style from Pytorch. This will make the whole training part easier.

Run `python Data_collection.py -- save_directory train/Move_Back` To capture and store deisred images, Change the directory when working with different label.

The whole process of capturing hand gesture is done by thresholding and subtracting background to obtain foreground object, Look into the codeflow for more documenation.


## Training a Gesture Classifier

We need to build a classifier that can be evaluated in real time and also keep up with the gym environment. With this constraint in mind we build a basic 
CNN model to learn features of different classes. As we are dealing with low-scale images with only one channel using any pre-trained network would be an overkill.
By choosing a data centric approach to solve the problem, we have already developed a dataset that can be easily classified by even using Support Vector Machines(SVM).
But we will stick to a baseline model constructed at `baseline_model.py` 

Train the model by running
`python train_loop.py --dataset --gesture_train_dataset --lr 0.01 `

Test the model performance on test data
`python inference.py --dataset --gesture_test_data`  

## Exploring MountainCar-v0 environment

Before Moving to control the mountaincar-v0, we can explore and play a random game to understand action and observation spaces in the environment by running `python Rough_interact_environment.py`


## Controlling MountainCar-v0 using Webcam Feed

As we have saved our trained classifier, we will use the classifier to detect gestures in real time and simultaneously pass the outputs to 
mountainCar-v0 environment to control the actions. Generally the mountainCar-v0 is underpowered to climb the hill, we need to create a momentum before accelearting to reach the flag and end the episode.

To achieve this run `python Controlling_mountain_car_real_time.py`  


The Training loss and Evaluation Metrics will be discussed in detail in the Report.

