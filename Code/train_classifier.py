import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as tvt
from torchvision import datasets
from torchvision import models
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
from time import sleep


def fit(model, device, train_dataloader, val_dataloader, epochs=100, lr=1e-3):
    """
    Function to train the model and save the best and the last weights under a selected metric
    Args:
        model (torch model): NN model
        device (torch device): device (CPU or GPU)
        train_dataloader (torch dataloader): loads images for training
        val_dataloader (torch dataloader): loads images for validation
        epochs (int): number of epochs for training
        lr (float): initial learning rate

    Returns:
        a dictionary with the history training
    """

    # Loss functions
    criterion = torch.nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=1e-6, amsgrad=False)
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-5, amsgrad=False)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-6
    )

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Puts the model in a device (CPU or GPU)
    model.to(device)

    # Training history dictionary
    hist = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    # Initial validation loss reference
    # val_loss_ref = 1.

    # Initial validation accuracy reference
    val_acc_ref = 0

    # Training loop
    for epoch in range(1, epochs + 1):
        train_total_correct = 0
        train_total_samples = 0

        val_total_correct = 0
        val_total_samples = 0

        # Loads images and labels
        bar = tqdm(train_dataloader)

        # Initializes the loss and IoU values for training
        train_loss, train_acc = [], []

        # Sets the model to train mode
        model.train()

        # Training loop
        for images, labels in bar:
            # Puts the images and the labels in the device
            images, labels = images.to(device), labels.to(device)

            # Add a dim to the labels tensor
            labels = torch.unsqueeze(labels, -1)

            # Sets the gradients to None
            optimizer.zero_grad()

            # Model predictions
            predictions = model(images)

            # Computes the training loss
            loss = criterion(predictions, labels.float())

            # Performs the backpropagation
            loss.backward()

            # Performs the parameter update
            optimizer.step()

            # Converts predictions to binary values
            # predictions = (predictions > 0.5).type(torch.uint8)
            predictions = (torch.sigmoid(predictions) > 0.5).type(torch.uint8)

            # Updates the running total of correct predictions and samples
            train_total_correct += (predictions == labels).sum().item()
            train_total_samples += labels.size(0)

            # Calculates the accuracy for this epoch
            accuracy = train_total_correct / train_total_samples

            # Stores the training parameters for the current batch
            train_loss.append(loss.item())
            train_acc.append(accuracy)
            bar.set_description("loss=%.5f acc=%.5f" % (float(np.mean(train_loss)), float(np.mean(train_acc))))
            sleep(0.1)

        # Stores the training parameters for the current epoch
        hist['loss'].append(np.mean(train_loss))
        hist['accuracy'].append(np.mean(train_acc))

        # Loads images and labels for validation
        bar = tqdm(val_dataloader)

        # Initializes the loss and IoU values for validation
        val_loss, val_acc = [], []

        # Sets the model to evaluation mode
        model.eval()

        # Validation loop
        with torch.no_grad():
            for images, labels in bar:
                # Puts the images and the labels in the device
                images, labels = images.to(device), labels.to(device)

                # Add a final dim to the labels tensor
                labels = torch.unsqueeze(labels, -1)

                # Model predictions
                predictions = model(images)

                # Computes the validation loss
                loss = criterion(predictions, labels.float())

                # Converts predictions to binary
                # predictions = (predictions > 0.5).type(torch.uint8)
                predictions = (torch.sigmoid(predictions) > 0.5).type(torch.uint8)

                # Updates the running total of correct predictions and samples
                val_total_correct += (predictions == labels).sum().item()
                val_total_samples += labels.size(0)

                # Calculates the validation accuracy for this epoch
                accuracy = val_total_correct / val_total_samples

                # Stores the validation parameters for the current batch
                val_loss.append(loss.item())
                val_acc.append(accuracy)
                bar.set_description("val_loss=%.5f val_acc=%.5f" % (float(np.mean(val_loss)), float(np.mean(val_acc))))
                sleep(0.1)

        # Stores the validation parameters for the current epoch
        hist['val_loss'].append(np.mean(val_loss))
        hist['val_accuracy'].append(np.mean(val_acc))

        # Prints the results at ending the epoch
        print('Epoch %d/%d loss=%.5f acc=%.5f val_loss=%.5f val_acc=%.5f' %
              (epoch,
               epochs,
               float(np.mean(train_loss)),
               float(np.mean(train_acc)),
               float(np.mean(val_loss)),
               float(np.mean(val_acc))))

        # Saves the best model (using validation accuracy criteria)
        # if np.mean(val_loss) < val_loss_ref:
        if np.mean(val_acc) > val_acc_ref:
            print('Saving model...')
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, BEST_WEIGHT_FILE))
            # val_loss_ref = np.mean(val_loss)
            val_acc_ref = np.mean(val_acc)

        # Applies the learning rate scheduler policy
        scheduler.step(np.mean(val_loss))
        print()
        sleep(0.1)

    # Saves the last model
    print('Saving model...')
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, LAST_WEIGHT_FILE))
    return hist


def main():
    """
    Classes:
        0- Not Mature
        1- Mature
    """

    # Creates a directory for the current experiment (if the directory exists, the experiment is aborted)
    try:
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        else:
            sys.exit('Path to %s already exists. Please, assign another directory for the experiment results' %
                     RESULTS_DIR)
    except Exception as e:
        sys.exit('Exception: %s' % str(e))

    # Composes the image transformations for training/validation
    train_transform_img = tvt.Compose([
        # tvt.ToPILImage(),
        tvt.Resize((HEIGHT, WIDTH), tvt.InterpolationMode.BILINEAR),
        tvt.ColorJitter(brightness=0.2, contrast=0.2, saturation=0., hue=0.),
        tvt.Grayscale(num_output_channels=3),
        tvt.RandomHorizontalFlip(p=0.5),
        tvt.RandomVerticalFlip(p=0.5),
        tvt.ToTensor()])

    val_transform_img = tvt.Compose([
        # tvt.ToPILImage(),
        tvt.Resize((HEIGHT, WIDTH), tvt.InterpolationMode.BILINEAR),
        tvt.Grayscale(num_output_channels=3),
        tvt.ToTensor()])

    try:

        # Gets the train/validation datasets
        train_data = datasets.ImageFolder(os.path.join(DATA_DIR, TRAIN_DIR), transform=train_transform_img)
        val_data = datasets.ImageFolder(os.path.join(DATA_DIR, VAL_DIR), transform=val_transform_img)

        # Gets the dataloaders
        train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=BATCH_SIZE, pin_memory=True)

        # Sets the device (CPU or GPU)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # Loads a ResNet101 model with default weights (IMAGENET1K_V2)
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, N_OUTPUT_NEURONS)

        # Trains the model
        hist = fit(model, device, train_loader, val_loader, epochs=N_EPOCHS, lr=LR)
        df = pd.DataFrame(hist)
        df.to_csv(os.path.join(RESULTS_DIR, CSV_FILE), sep=',', index=False, encoding='latin1')
        title = 'Model: %s  Loss function: %s\n Input image format: %s' % \
                (MODEL_NAME, LOSS_FUNCTION_NAME, IMG_FORMAT)
        df.plot(grid=True, title=title, xlabel='Epoch', ylabel='Loss / Accuracy', figsize=(10, 8))
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, IMG_FILE))
    except Exception as e:
        sys.exit('Exception: %s' % str(e))


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    # Main parameters
    MODEL_NAME = 'ResNet101'
    LOSS_FUNCTION_NAME = 'SoftBCEWithLogits'
    IMG_FORMAT = 'RGB'
    DATA_DIR = '../data'
    WEIGHT_DIR = '../weights'
    BEST_WEIGHT_FILE = 'best.pt'
    LAST_WEIGHT_FILE = 'last.pt'
    RUN_DIR = 'runs'
    TRAIN_DIR = 'train'
    VAL_DIR = 'val'
    CSV_FILE = 'train.csv'
    IMG_FILE = 'train.jpg'
    WIDTH = 256  # image width
    HEIGHT = 256  # image height
    LR = 1e-4
    BATCH_SIZE = 10
    N_EPOCHS = 50
    N_OUTPUT_NEURONS = 1
    N_EXP = 1
    RESULTS_DIR = os.path.join(RUN_DIR, TRAIN_DIR, str(N_EXP))
    # SEED = 42
    # np.random.seed(SEED)
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed(SEED)

    # Main function call
    main()
