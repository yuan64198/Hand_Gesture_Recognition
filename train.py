import torch
import argparse
import os
import numpy as np
from model import Gest_CNN
from checkpoints import save_checkpoint
from checkpoints import load_checkpoint
from sklearn.model_selection import train_test_split

MAX_LEN = 51
NUM_EPOCHS = 500
CURRENT_EPOCH = 0

def main():
    args = get_arguments()
    device = get_device()
    model = Gest_CNN()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    start_epoch = load_checkpoint(args.checkpoint, model, optimizer)
    if args.mode == "train":
        X_train, X_valid, y_train, y_valid = load_train_data()
        train(start_epoch, start_epoch+NUM_EPOCHS+1, model, criterion, optimizer, device, X_train, X_valid, y_train, y_valid)
    if args.mode == "test":
        X, y = load_test_data()
        test(X, y, model)
        
    

def train( start_epoch, end_epoch, model, criterion, optimizer, device, X_train, X_valid, y_train, y_valid):
    for epoch in range(start_epoch, end_epoch):
        optimizer.zero_grad() # Clears existing gradients from previous epoch
        X_train.to(device)
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward() # Does backpropagation and calculates gradients
        optimizer.step() # Updates the weights accordingly
        accuracy = test(X_valid, y_valid, model)
        print_loss_accuracy(epoch, loss.item(), accuracy,every = 10)
        save_checkpoint( { "epoch": epoch+1, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}, every = 25)

def test(X, y, model):
    output = model(X)
    
    y_pred = output.argmax(dim=1)
    
    correct = y_pred.eq(y.view_as(y_pred)).sum().item()
    accuracy = correct/y_pred.size(0)*100
    print("Testing Accuracy: {:2f}%".format(accuracy))
    return accuracy

def print_loss_accuracy( epoch, loss, accuracy, every):
    if epoch%every==0:
        print('Epoch: {}/{}.............'.format(epoch, epoch+NUM_EPOCHS), end=' ')
        print("Loss: {:.4f}".format(loss), end=' ')
        print("Accuracy: {:.2f}%".format(accuracy))

def load_train_data():
    X = []
    y = []
    for user_id in range(1, 9):
        for label in range(1, 21):
            for idx in range(1, 17):
                filename = "./train_set/U"+str(user_id).zfill(2)+"/"+str(label).zfill(2)+"/"+str(idx).zfill(2)+".txt"
                arr = np.loadtxt(filename)
                arr = arr[:, 3:]
                padding = np.zeros((MAX_LEN, 3))
                padding[:arr.shape[0], :arr.shape[1]] = arr
                padding = np.transpose(padding)
                X.append(padding)
                y.append(label-1)
    X = np.asarray(X)
    y = np.asarray(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=87)
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()

    return X_train, X_test, y_train, y_test

def load_test_data():
    X = []
    y = []
    for user_id in range(1, 9):
        for label in range(1, 21):
            for idx in range(17, 22):
                filename = "./test_set/U"+str(user_id).zfill(2)+"/"+str(label).zfill(2)+"/"+str(idx).zfill(2)+".txt"
                if os.path.exists(filename)==False:
                    continue
                arr = np.loadtxt(filename)
                arr = arr[:, 3:]
                padding = np.zeros((MAX_LEN, 3))
                padding[:arr.shape[0], :arr.shape[1]] = arr
                padding = np.transpose(padding)
                X.append(padding)
                y.append(label-1)
    X = np.asarray(X)
    y = np.asarray(y)
    
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()
    
    return X, y

def get_arguments():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--mode",
        default="train",
        type=str,
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
    )
    args = parser.parse_args()
    return args

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")
    return device

if __name__ == "__main__":
    main()