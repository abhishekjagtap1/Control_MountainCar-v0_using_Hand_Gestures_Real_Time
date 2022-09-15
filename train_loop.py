from comet_ml import Experiment
import argparse
import os, sys
import random
import numpy as np
import torch
import torch.nn as nn
import cv2
import torch.cuda
from torch.utils import data
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from Models.baseline_model import CNNModel
from rich.progress import track
import time

# Run experiments using cometml for keeping track of losses during training
experiment = Experiment(
    api_key="OdiDlhbMMXN5k6GApQUdhp9LD",
    project_name="general_experiments",
    workspace="abhishekjagtap1",
)




def get_argparser():

    """ Parse required arguments for efficient and flexibile Hyper-parameter tuning """

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default='gesture_train_data',
                        choices=['gesture_train_data', 'gesture_test_data'], help ='Specify the train or test data')
    parser.add_argument('--random_seed', type=int, default=1, help='Random Seeding Value')
    parser.add_argument('--batch_size', type=int, default=64, help='Specify the Batch Size' )
    parser.add_argument('--val_batch_size', type=int, default=64, help='Specify the  Val Batch Size')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID')
    parser.add_argument('--train_val_split_size', type=float, default=0.1, help='Train/Val Split Size')
    parser.add_argument('--lr', type=float, default=0.01, help='Specify the learning rate')

    parser.add_argument('--loss_type', type=str, default='cross_entropy', help='specify to use binary or categorical cross entropy')

    return parser







def get_dataset(opts):
    """ Get Dataset for Augmentation """

    if opts.dataset == 'gesture_train_data':

        """ 
            As our Dataset is minimal and only contains gestures in black and white, we will apply only 
            geometric augmentations so that our model learns features from various orientation 

        """

        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(size=(200, 200)),
                transforms.RandomRotation(degrees=30, fill=cv2.BORDER_REPLICATE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

            ]
        )

        gesture_dataset = ImageFolder(root='hand_gestures_data/train', transform=train_transform)


    return gesture_dataset


# training
def train(model, trainloader, optimizer, criterion):

    """
    This function lets us define our training loop to perform classification of our gesture dataset

    :param model: Basic CNN model built during modeling
    :param trainloader: Images to be trained on
    :param optimizer: Defining the required Optimizer for the current task of classification.
                      [Used to change the attributes such as wieghts and learning rate]
    :param criterion: Defining a loss function

    :returns: Training Loss and Accuracy

    """

    with experiment.train():
        model.train()
        print('Training')
        train_running_loss = 0.0
        train_running_correct = 0
        counter = 0
        for i, data in track(enumerate(train_loader), total=len(train_loader)):
            counter += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # forward pass
            outputs = model(image)
            # calculate the loss
            loss = criterion(outputs, labels)
            train_running_loss += loss.item()
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            train_running_correct += (preds == labels).sum().item()
            # backpropagation
            loss.backward()
            # update the optimizer parameters
            optimizer.step()

    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))


    return epoch_loss, epoch_acc


# validation
def validate(model, testloader, criterion):

    """
        This function validates the model performance on validation dataset

        :param model: Basic CNN model built during modeling
        :param testloader: Images to validate the performance of the so trained model
        :param criterion: Defining a loss function

        :returns: Validation Loss and Accuracy

        """

    with experiment.validate():
        model.eval()
        print('Validation')
        valid_running_loss = 0.0
        valid_running_correct = 0
        counter = 0
        with torch.no_grad():
            for i, data in track(enumerate(val_loader), total=len(val_loader)):
                counter += 1

                image, labels = data
                image = image.to(device)
                labels = labels.to(device)
                # forward pass
                outputs = model(image)
                # calculate the loss
                loss = criterion(outputs, labels)
                valid_running_loss += loss.item()
                # calculate the accuracy
                _, preds = torch.max(outputs.data, 1)
                # Log the confusion metrics
                experiment.log_confusion_matrix(labels.cpu(), preds.cpu())

                valid_running_correct += (preds == labels).sum().item()

    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))


    return epoch_loss, epoch_acc

def save_model(model, optimizer, criterion):
    """
           Function to save the trained model to disk.
           """
    torch.save({

        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, 'Models/Final_Gesture_recognition_trained_model_cometml.pth')



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
gesture_dataset = get_dataset(opts)
dataset_size = len(gesture_dataset)
print('Total Number of Images', dataset_size)

""" 
Split our dataset into training and validation subsets 
default split: opts.train_val_split_size

"""
val_split_index = int(np.floor(opts.train_val_split_size * dataset_size))
train_index = dataset_size - val_split_index

train_dataset, val_dataset = torch.utils.data.random_split(gesture_dataset, [train_index, val_split_index])

#Set up Datalaoders
train_loader = data.DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=0)
val_loader = data.DataLoader(val_dataset, batch_size=opts.val_batch_size, shuffle=True)

print("Dataset: %s, Train_Set: %d, Val_Set: %d" % (opts.dataset, len(train_loader), len(val_loader)))


"""
Setup the model, 
See Models/baseline_model.py for the construction and details of the used model 
"""
model = CNNModel().to(device)

# Set up the optimizer and the loss criterion
optimizer = torch.optim.SGD(model.parameters(), lr=opts.lr, momentum=0.9)
if opts.loss_type == 'cross_entropy':
    criterion = nn.CrossEntropyLoss()




def main():


    # lists to keep track of losses and accuracies
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    # set the number of Epochs to train
    opts.epochs = 30
    # start the training
    for epoch in range(opts.epochs):
        print(f"[INFO]: Epoch {epoch + 1} of {opts.epochs}")

        experiment.log_metric('Epoch', epoch)

        train_epoch_loss, train_epoch_acc = train(model, train_loader,
                                                  optimizer, criterion)
        valid_epoch_loss, valid_epoch_acc = validate(model, val_loader,
                                                     criterion)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        # Log all the computed Losses and Confusion Matrix

        experiment.log_metric('Training Loss', train_epoch_loss)
        experiment.log_metric('Training Accuracy', train_epoch_acc)
        experiment.log_metric('Validation Loss', valid_epoch_loss)
        experiment.log_metric('Validation Accuracy', valid_epoch_acc)

        print('-' * 50)
        time.sleep(5)

    # save the trained model weights
    save_model(model, optimizer, criterion)
    # save the loss and accuracy plots

    print('Completed Training and Saved the Model')



if __name__ == '__main__':
    main()








