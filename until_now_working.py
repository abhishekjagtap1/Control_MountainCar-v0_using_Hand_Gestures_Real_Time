import cv2
import torchvision.transforms as transforms
import torch
from Models.baseline_model import CNNModel
import gym
from gym.utils import play
import numpy as np
from pynput.keyboard import Controller
import matplotlib.pyplot as plt



import time

import imutils

keyboard = Controller()

device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# list containing all the class labels
labels = [
    'Move Back', 'Move Forward', 'Stop'
    ]



transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(size=(200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
bg = None
global model




model = CNNModel().to(device)
checkpoint = torch.load('Models/Final_Gesture_recognition_trained_model_cometml.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def run_avg(image, accumWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, accumWeight)

def hand_segment(image, alpha=25):
    global background

    # We will find the absoloute difference between background and current frame

    difference = cv2.absdiff(bg.astype('uint8'), image)

    # Apply some threshold of value alpha to obtain foreground
    threshold = cv2.threshold(difference, alpha, 255, cv2.THRESH_BINARY)[1]

    contours, hierachy = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print('No Hands Found')
    else:
        hand_segmentation = max(contours, key=cv2.contourArea)

        return threshold, hand_segmentation
def count(thresholded, segmented):
    thresholded = transform(thresholded)
    thresholded = torch.unsqueeze(thresholded, 0)


    with torch.no_grad():
        outputs = model(thresholded.to(device))
        output_label = torch.topk(outputs, 1)
        pred_class = labels[int(output_label.indices)]
        print(pred_class)


    '''prob = loaded_model.predict(thresholded)
    if (max(prob[0]) > .99995):
        return loaded_model.predict_classes(thresholded)'''
    return pred_class


camera = cv2.VideoCapture(0)
accumWeight = 0.5

if not camera.isOpened():
    print('Camera is Disabled, Terminating.................!!')
    exit()

#Record Maximum 500 images per class
count = 0

## Region of interest (ROI) co-ordinates
top, right, bottom, left = 10, 350, 225, 590
image_size = (700, 700)
num_frames = 0
imageNumber = 0
counter = 0
buttonPressed = False
buttonCount = 0
buttonDelay = 1




while True:

    sucess, frame = camera.read()











    if not sucess:
        print("Frame is not captured, terminating...")
        break

    #Resize and flip the frame so its not in a mirrored view
    frame = imutils.resize(frame, width=700)
    frame = cv2.resize(frame, image_size, interpolation=cv2.INTER_LINEAR)
    frame = cv2.flip(frame, 1)

    #Clone/Copy the frame for further preprocessing
    clone = frame.copy()
    roi = frame[top:bottom, right:left]

    #Apply grayscale and Blur the image
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # To get the Background, Keep looking until a alpha threshold is reached
    if num_frames < 30:
        run_avg(gray, accumWeight)

        if num_frames == 1:
            print(">>> Please wait! calibrating...")
        elif num_frames == 29:
            print(">>> Calibration successfull...")
            print('Move Your Hand to Region of Intrest')



        #Extract_Background(gray, accumWeight)

    else:
        #Segment our hand region
        #Experimemnted alpha in the current indoor conditions and set to alpha = 25
        hand = hand_segment(gray)



        #threshold, hand_segmentation = hand_segment(gray)

        if hand is not None and buttonPressed is False:

            env = gym.make('MountainCar-v0', render_mode='rgb_array', )
            env = env.unwrapped###########################ee line en bek ahilla
            obs_space = env.observation_space
            action_space = env.action_space
            #print("The observation space: {}".format(obs_space))
            #print("The action space: {}".format(action_space))

            threshold, hand_segmentation = hand
            new_threshold = threshold.copy()

            threshold = cv2.cvtColor(threshold, cv2.COLOR_GRAY2RGB)
            '''plt.imshow(threshold)
            plt.show()'''

            threshold = transform(threshold)
            #threshold = threshold.to(device)

            '''threshold = cv2.resize(threshold,(200,200))
            threshold = cv2.cvtColor(threshold, cv2.COLOR_GRAY2RGB)
            cv2.imshow('shata', threshold)
            threshold = threshold / 255
            threshold = threshold.transpose(2, 1, 0)
            print(threshold.shape)
            threshold = torch.from_numpy(threshold)
            '''
            threshold = torch.unsqueeze(threshold, 0)
            #print(print(threshold.shape))
            outputs = model(threshold.to(device))
            output_label = torch.topk(outputs, 1)
            #print('output_label', output_label.indices)
            pred_class = labels[int(output_label.indices)]
            #print(pred_class)
            cv2.drawContours(clone, [hand_segmentation + (right, top)], -1, (0, 0, 255))
            '''gesture = count(thresholded=threshold, segmented=hand_segmentation)
            print(gesture)'''
            cv2.putText(clone, str(pred_class), (200, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 35), 2)




            if pred_class =='Move Forward':
                action = 2
                '''buttonPressed = True
                keyboard.press('d')
                keyboard.release('d')
                print('pressed and released D')'''

            if pred_class == 'Move Back':
                action= 0
                '''buttonPressed = True
                keyboard.press('a')
                keyboard.release('a')
                print('pressed and released A')'''


            if pred_class == 'Stop':
                action = 1
                '''buttonPressed = True
                keyboard.press('a')
                keyboard.release('a')
                print('pressed and released A')'''

            # reset the environment and see the initial observation
            obs = env.reset()
            print("The initial observation is {}".format(obs))

            # Sample a random action from the entire action space
            predicted_action = action
            print(predicted_action)
            #random_action = env.action_space.sample()
            #print(random_action)
            buttonPressed = True

            # # Take the action and get the new observation space
            env.step(action)
            print(env.step(action))

            '''time.sleep(0.1)'''

            env.close()

            cv2.imshow('Shata', env.render())


            '''env_screen = env.render()'''






            '''plt.imshow(env_screen)
            plt.show()'''

        if buttonPressed:
            buttonCount += 1
            if buttonCount> buttonDelay:
                buttonCount = 0
                buttonPressed = False











            #cv2.imshow("Thesholded", threshold)


    cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)



    num_frames += 1

    # display the frame with segmented hand

    cv2.imshow("Webcam Feed", clone)
    #print(pred_class)





    # observe the keypress by the user
    keypress = cv2.waitKey(1) & 0xFF

    # if the user pressed "q", then stop looping
    if keypress == ord("q"):
        break


camera.release()

cv2.destroyAllWindows()

# free up memory






