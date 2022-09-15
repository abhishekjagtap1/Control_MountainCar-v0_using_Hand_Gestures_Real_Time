## Run this Code to Generate/ Capture Hand Gestures

import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise
import time
import argparse




def get_argparser():

    """
    Parse arguments - Specify the directory to save images default = 'train/Move_Back'

    Save images in ImageFolder style from pytorch for easy training
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--save_directory", type=str, default='test/Stop_test',
            choices=['train/Move_Back', 'train/Move_Forward', 'train/Stop_Sign', 'test/Move_Back_test', 'test/Move_Forward_test', 'test/Stop_test'],
            help ='Specify the path to save images')

    return parser


background = None




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

def hand_segment(image, alpha=20):

    """
    Source - https://docs.opencv.org/3.4/d8/d38/tutorial_bgsegm_bg_subtraction.html

    This function lets us compute absolute difference between background and the current frame and then
    applies some threshold to obtain foreground object.

    :param image - The current frame
    :param alpha - Threshold value [Depends on indoor lighting conditions]



    :returns segmented image


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


opts = get_argparser().parse_args()

###########   -------- Change Directory When Capturing for Different Labels/Classes------    ###############


set_directory_to_save = f'hand_gestures_data/{opts.save_directory}'

# Set to default camera
camera = cv2.VideoCapture(0)
calibrated = False

if not camera.isOpened():
    """
    :If no break in the loop, then camera is turned on
    """
    print('Camera is Disabled, Terminating.................!!')
    exit()


#Record Maximum 500 images per class
count = 0

#Region of interest (ROI) co-ordinates
top, right, bottom, left = 10, 350, 210, 550
image_size = (700, 700)
num_frames = 0

counter=0


while count<500:

    """ 
    Source - The below code is slightly influenced from https://github.com/cvzone/cvzone/blob/master/cvzone/HandTrackingModule.py
    
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
    gray = cv2.GaussianBlur(gray, (7, 7), 0)


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


        if hand is not None:
            threshold, hand_segmentation = hand

            # Display the resulting thresholded image with hand
            cv2.imshow("Thesholded", threshold)

            key = cv2.waitKey(1)

            #Press 'c' to capture the desired image and store
            if key == ord("c"):
                counter += 1
                cv2.imwrite(f'{set_directory_to_save}/Image_{time.time()}.jpg', threshold)
                print('Image Saved')

    # Draw the segmented Hand
    cv2.rectangle(clone, (left, top), (right, bottom), (0, 0, 255), 2)

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
cv2.destroyAllWindows()





















