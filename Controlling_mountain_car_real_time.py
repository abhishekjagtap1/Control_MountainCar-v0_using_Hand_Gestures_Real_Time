import cv2
import torchvision.transforms as transforms
import torch
from Models.baseline_model import CNNModel
import gym
from gym.utils import play
import numpy as np
import time





background = None
global model

def Extract_Background(image):
    """ 
    Source - https://www.programcreek.com/python/example/89364/cv2.accumulateWeighted

    This functions helps in calculating the running average of our image which can be used later on 
    to initialize  detection moving objects

    :param image: Background Image
    :return: Accumulated Weighted Image

    """""

    global background

    if background is None:
        background = image.copy().astype('float')
        return

    cv2.accumulateWeighted(image, background, 0.5)


def hand_segment(image, alpha=25):

    """
    Source - https://docs.opencv.org/3.4/d8/d38/tutorial_bgsegm_bg_subtraction.html

    This function lets us compute absolute difference between background and the current frame and then
    applies some threshold to obtain foreground object.

    :param image - The current frame
    :param alpha - Threshold value [Depends on indoor lighting conditions]



    :returns segmented image, threshold


    """
    global background

    difference = cv2.absdiff(background.astype('uint8'), image)
    threshold = cv2.threshold(difference, alpha, 255, cv2.THRESH_BINARY)[1]

    contours, hierachy = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print('Kindly Move Your Hand in Region of Interest(ROI)')
        #time.sleep(5)
    else:
        hand_segmentation = max(contours, key=cv2.contourArea)

        return threshold, hand_segmentation


""" 
As we implicitly trained on a pre-defined transformations
we will use the same tranformations during real time for succesive frames

"""

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(size=(200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])






device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# list containing all the Names of the class labels
labels = ['Move Back', 'Move Forward', 'Stop']




"""
Setup the model for real time inference
 
"""

model = CNNModel()
print("[!] Loading Gesture Trained Model")

# Load the saved model during training
model.load_state_dict(torch.load('Models/Final_Gesture_recognition_trained_model_cometml.pth')['model_state_dict'])
print('Model Loaded')
model.to(device)
model.eval()



# Set to default camera
camera = cv2.VideoCapture(0)


if not camera.isOpened():
    print('Camera is Disabled, Terminating.................!!')
    exit()




## Region of interest (ROI) co-ordinates
top, right, bottom, left = 10, 350, 230, 590


image_size = (700, 700)
num_frames = 0
imageNumber = 0
counter = 0
actiontaken = False
actioncount = 0
actiondelay = 1


""" 
Source - https://blog.paperspace.com/getting-started-with-openai-gym/

Setup the gym environment and render in a rgb array form, 

The Wrapper class in OpenAI Gym provides you with the functionality 
to modify various parts of an environment to suit our needs, so we will unwrap our environment

"""

env = gym.make('MountainCar-v0', render_mode='rgb_array')
env = env.unwrapped
# Understand the shape and range of observation and action space
obs_space = env.observation_space
action_space = env.action_space
# reset the environment and see the initial observation
state = env.reset()
#print(state[0])
#print(env.observation_space.low)


while True:

    """ 
        Source and references - The below code is slightly influenced from
        - https://github.com/cvzone/cvzone/blob/master/cvzone/HandTrackingModule.py
        - https://www.computervision.zone/lessons/code-and-files-23/
        - https://github.com/fvilmos/gesture_detector

        While the camera is running, capture the frames to preprocess.

    """

    sucess, frame = camera.read()

    if not sucess:
        print("Frame is not captured, terminating...")
        break

    #Resize and flip the frame so its not in a mirrored view

    frame = cv2.resize(frame, image_size, interpolation=cv2.INTER_LINEAR)
    frame = cv2.flip(frame, 1)

    #Clone/Copy the frame for further preprocessing
    clone = frame.copy()
    roi = frame[top:bottom, right:left]

    #Apply grayscale and Blur the image
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)


    if num_frames < 30:

        """ Let the camera calibrate for the first 30 frames """

        Extract_Background(gray)

        if num_frames == 1:
            print("Please wait! Let the camera calibrate...")

        elif num_frames == 29:
            print("Calibration successfull...")
            print('Move Your Hand to Region of Interest')

    else:
        # Segment our hand region
        # Experimemnted alpha in the current indoor conditions and set to alpha = 25
        hand = hand_segment(gray)


        if hand is not None and actiontaken is False:

            '''
            Run the Commented code only during debugging to understand the state of our env
            
            #state_adj = (state[0] - env.observation_space.low) * np.array([10, 100])
            #state_adj = np.round(state_adj, 0).astype(int)

            #print("The observation space: {}".format(obs_space))
            #print("The action space: {}".format(action_space))
            
            '''

            # we will perform the same steps when we collected data
            threshold, hand_segmentation = hand
            threshold = cv2.cvtColor(threshold, cv2.COLOR_GRAY2RGB)

            # Apply the transformation used during training on our segmented image
            threshold = transform(threshold)

            threshold = torch.unsqueeze(threshold, 0)

            #print(threshold.shape)
            # We will give the transformed images/frames to the model to predict the type/class of gesture
            outputs = model(threshold.to(device))
            output_label = torch.topk(outputs, 1)
            # Store the prediction probability
            pred_prob = torch.round(torch.max(outputs))

            # Do one hot encoding / map predictions to label
            pred_class = labels[int(output_label.indices)]



            #cv2.drawContours(clone, [hand_segmentation + (right, top)], -1, (0, 0, 255))
            # Write the Prediction along with its probability on the Webcam Feed
            text = pred_class, pred_prob.item()
            cv2.putText(clone, str(text), (20, 250), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

            '''
            The action required to move forward is int(2), Move Back is int(0) and to Stop is int(1) a
            ccording to the mountainCar-v0 helper scripts so we directly map our predictions 
            to the designated action 
            '''
            if pred_class =='Move Forward':
                action = 2

            if pred_class == 'Move Back':
                action = 0

            if pred_class == 'Stop':
                action = 1

            pred_action = action

            actiontaken = True
            # Let the environment take the predicted action
            env.step(pred_action)
            #Observe the state of the environmnet
            state2 = env.step(action)[0]
            reward = env.step(action)[1]
            done = env.step(action)[2]
            info = env.step(action)[3]

            """
            Run the Commented code only during debugging to understand the state of our env
            print(env.step(action))

            state2_adj = (state2 - env.observation_space.low) * np.array([10, 100])
            state2_adj = np.round(state2_adj, 0).astype(int)
            
            """
            time.sleep(0.001)

            if done:
                env.reset()



            cv2.imshow('Controlling Mountain Car-v0', env.render())



        if actiontaken:
            actioncount += 1
            if actioncount > actiondelay:
                actioncount = 0
                actiontaken = False



    cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

    # Increment the number of frames
    num_frames += 1

    # display the frame with segmented hand
    cv2.imshow("Webcam Feed", clone)






    # observe the keypress by the user
    keypress = cv2.waitKey(1)

    # if the user pressed "q", then stop looping
    if keypress == ord("q"):
        break


# free up memory
camera.release()
env.close()
cv2.destroyAllWindows()







