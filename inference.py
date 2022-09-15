from comet_ml import Experiment

import torch
import torchvision.transforms as transforms
from torch.utils import data
from torchvision.datasets import ImageFolder
from Models.baseline_model import CNNModel
from rich.progress import track
import argparse
import os
import numpy as np
import random

# Set up comet ml to keep track of the evaluation metrics
experiment = Experiment(
    api_key="OdiDlhbMMXN5k6GApQUdhp9LD",
    project_name="general_experiments",
    workspace="abhishekjagtap1",
)


def get_argparser():
    """ Parse required arguments for Inference, Default: Set to reproduce the results """

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default='gesture_test_data',
                        choices=['gesture_train_data', 'gesture_test_data'], help='Specify the train or test data')
    parser.add_argument('--random_seed', type=int, default=1, help='Random Seeding Value')
    parser.add_argument('--test_batch_size', type=int, default=64, help='Specify the  Val Batch Size')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID')

    return parser



def get_dataset(opts):
    """ Get Dataset for Augmentation """

    if opts.dataset == 'gesture_test_data':


        test_transform = transforms.Compose(
            [
                transforms.RandomCrop(size=(200, 200)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

            ]
        )

        gesture_test_dataset = ImageFolder(root='hand_gestures_data/test', transform=test_transform)


    return gesture_test_dataset


""" Initialize all the parameters using argparse """
opts = get_argparser().parse_args()
# Log all the hyper-parameters used when training
experiment.log_parameters(opts)

# Set the desired device to train our model
os.environ['CUDA_VISIBLE_DEVICE'] = opts.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: %s' % device)

# For efficient Reproducibility of Results, set a random seed
torch.manual_seed(opts.random_seed)
np.random.seed(opts.random_seed)
random.seed(opts.random_seed)

# Get the desired Dataset
gesture_test_dataset = get_dataset(opts)

test_loader = data.DataLoader(gesture_test_dataset, batch_size=opts.test_batch_size, shuffle=True)
print("Dataset: %s, Test_Set: %d," % (opts.dataset, len(test_loader)))



"""
Setup the model, 
See Models/baseline_model.py for the construction and details of the used model 
"""

model = CNNModel()
print("[!] Loading Gesture Trained Model")

# Load the saved model during training
model.load_state_dict(torch.load('Models/Final_Gesture_recognition_trained_model_cometml.pth')['model_state_dict'])
print('Model Loaded')
model.to(device)


with experiment.test():

    model.eval()
    test_running_correct = 0
    counter = 0
    test_accuracy = []
    with torch.no_grad():
        for i, data in track(enumerate(test_loader), total=len(test_loader)):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(image)
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            # Log the confusion metrics
            experiment.log_confusion_matrix(labels.cpu(), preds.cpu())
            test_running_correct += (preds == labels).sum().item()


        epoch_acc = 100. * (test_running_correct / len(test_loader.dataset))
        test_accuracy.append(epoch_acc)

    print(f"Test accuracy: ", test_accuracy)
    experiment.log_metric('Test Accuracy', test_accuracy)




