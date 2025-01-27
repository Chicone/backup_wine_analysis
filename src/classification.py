from pynndescent.optimal_transport import total_cost
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, PassiveAggressiveClassifier, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from concurrent.futures import ProcessPoolExecutor


import utils
from dimensionality_reduction import DimensionalityReducer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split, StratifiedKFold

from utils import find_first_and_last_position, normalize_dict, normalize_data
import numpy as np
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from torchvision import models
from torchvision import transforms
import torch.nn.functional as F

from tqdm import tqdm  # For the progress bar
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

from joblib import Parallel, delayed
import random
import matplotlib.pyplot as plt

class CustomDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class CNN1D(nn.Module):
    def __init__(self, input_length, num_classes, num_channels=1, nconv=2, multichannel=None):
        """
        A flexible 1D CNN where the number of convolutional layers can be set via `nconv`.

        Parameters:
        ----------
        input_length : int
            Length of the input sequence.
        num_classes : int
            Number of output classes.
        num_channels : int
            Number of input channels (e.g., m/z channels).
        nconv : int
            Number of convolutional layers.
        """
        super(CNN1D, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.multichannel=multichannel

        self.nconv = nconv

        # Base number of filters and kernel size
        base_filters = 64
        kernel_size = 5

        # Dynamically create convolutional and batch normalization layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        in_channels = num_channels  # Dynamically set the number of input channels
        for i in range(nconv):
            out_channels = base_filters * (2 ** i)  # Double the number of filters at each step
            self.conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2))
            self.bn_layers.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels  # Update for the next layer

        # Pooling and dropout
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.3)

        # Dynamically calculate the input size for the fully connected layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, num_channels, input_length)  # Shape: (batch_size, channels, length)
            for conv, bn in zip(self.conv_layers, self.bn_layers):
                dummy_input = self.pool(F.relu(bn(conv(dummy_input))))
            fc_input_size = dummy_input.numel()

        # Fully connected layers
        self.fc1 = nn.Linear(fc_input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.flatten = nn.Flatten()

        self._initialize_weights()

    def forward(self, x):
        # Apply convolutional layers with pooling, batch normalization, and ReLU
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = self.pool(F.relu(bn(conv(x))))

        # Flatten the output
        x = self.flatten(x)

        # Fully connected layers with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    def _initialize_weights(self):
        """
        Initialize weights for convolutional and fully connected layers.
        Uses Kaiming Normal Initialization for weights and constant initialization for biases.
        """
        for layer in self.modules():
            if isinstance(layer, nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)


    def fit(self, X_train, y_train, X_val, y_val, batch_size=32, num_epochs=10, learning_rate=0.001):
        """
        Train the model with validation checks.

        Parameters:
        ----------
        train_loader : DataLoader
            DataLoader for training data.
        val_loader : DataLoader
            DataLoader for validation data.
        num_epochs : int
            Number of epochs to train.
        learning_rate : float
            Learning rate for optimizer.
        """
        weight_decay = 0.001

        print(f'Batch size =   {batch_size}')
        print(f'Epochs =       {num_epochs}')
        print(f'Conv layers =  {self.nconv}')
        print(f'Learn rate =   {learning_rate}')
        print(f'Weight decay = {weight_decay}')


        # TODO add conditions for single or
        if not self.multichannel:
            train_dataset = TensorDataset(X_train.unsqueeze(1), y_train)
        else:
            train_dataset = TensorDataset(X_train.squeeze(), y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # plt.figure();plt.imshow(train_dataset[2][0].permute(1, 2, 0).numpy())

        if not self.multichannel:
            val_dataset =  TensorDataset(X_val.unsqueeze(1), y_val)
        else:
            val_dataset =  TensorDataset(X_val.squeeze(), y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

        for epoch in range(num_epochs):
            self.to(self.device)
            # Training phase
            self.train()
            running_loss = 0.0
            for tics_batch, labels_batch in train_loader:
                optimizer.zero_grad()
                outputs = self(tics_batch)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            scheduler.step()  # Adjust learning rate

            # Validation phase
            self.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for tics_batch, labels_batch in val_loader:
                    outputs = self(tics_batch)
                    loss = criterion(outputs, labels_batch)
                    val_loss += loss.item()

                    _, predicted = outputs.max(1)
                    total += labels_batch.size(0)
                    correct += (predicted == labels_batch).sum().item()

            val_accuracy = correct / total
            print(
                f"Epoch {epoch + 1}/{num_epochs}, "
                # f"Learning Rate = {scheduler.get_last_lr()}, "
                f"Train Loss: {running_loss / len(train_loader):.5f}, "
                f"Val Loss: {val_loss / len(val_loader):.5f}, "
                f"Val Accuracy: {val_accuracy:.5f}"
            )


    def predict(self, images):
        """
        Predict the class labels for the given input images.

        Parameters:
        ----------
        images : Tensor
            Input data, assumed to be of shape (batch_size, 3, 224, 224).

        Returns:
        -------
        y_pred : Tensor
            Predicted class labels.
        """
        self.eval()
        images = images.to(self.device)
        with torch.no_grad():
            outputs = self(images)
            _, y_pred = outputs.max(1)
        return y_pred.cpu()

    def score(self, X_test, y_test):
        """
        Compute the accuracy of the model on test data.

        Parameters:
        ----------
        X_test : torch.Tensor
            Test feature data of shape (num_samples, num_features).
        y_test : torch.Tensor
            True labels for the test data.

        Returns:
        -------
        float
            Accuracy of the model.
        """
        self.eval()  # Switch to evaluation mode
        correct = 0
        total = 0

        with torch.no_grad():  # Disable gradient computation
            # Forward pass
            outputs = self(X_test)
            _, predicted = outputs.max(1)  # Get class with maximum score
            total = y_test.size(0)
            correct = (predicted == y_test).sum().item()

        # Return accuracy
        return correct / total

# class CNNClassifier(nn.Module):
    # def __init__(self, input_channels, num_classes, input_height, input_width):
    #     super(CNNClassifier, self).__init__()
    #     self.device = "cpu"
    #     self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
    #     self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
    #     self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    #
    #     # Calculate the output size after convolution and pooling
    #     conv_output_height = input_height // 4  # Two pooling layers reduce height by a factor of 4
    #     conv_output_width = input_width // 4  # Two pooling layers reduce width by a factor of 4
    #     flattened_size = 32 * conv_output_height * conv_output_width
    #
    #     self.fc1 = nn.Linear(flattened_size, 128)
    #     self.fc2 = nn.Linear(128, num_classes)
    #     self.relu = nn.ReLU()
    #
    # def forward(self, x):
    #     x = self.pool(self.relu(self.conv1(x)))
    #     x = self.pool(self.relu(self.conv2(x)))
    #     x = x.view(x.size(0), -1)
    #     x = self.relu(self.fc1(x))
    #     x = self.fc2(x)
    #     return x

class PretrainedGoogleNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(PretrainedGoogleNet, self).__init__()
        # Load pretrained GoogleNet
        self.model = models.googlenet(pretrained=pretrained)

        # Freeze all layers except the fully connected (fc) layer
        for name, param in self.model.named_parameters():
            if "fc" not in name:  # Freeze everything except "fc"
                param.requires_grad = False

        # # Unfreeze specific layers (e.g., the last inception block and auxiliary layers)
        # for name, param in self.model.named_parameters():
        #     if "inception5" in name:  # Unfreeze certain layers
        #         param.requires_grad = True

        # Replace the final fully connected layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # Define device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # # Print which layers are frozen
        # for name, param in self.model.named_parameters():
        #     print(f"{name} is {'trainable' if param.requires_grad else 'frozen'}")

    def forward(self, x):
        return self.model(x)

    def fit(self, X_train, y_train, X_val, y_val, batch_size =32, num_epochs=10, learning_rate=0.001):
        """
        Train the model with validation checks.

        Parameters:
        ----------
        train_loader : DataLoader
            DataLoader for training data.
        val_loader : DataLoader
            DataLoader for validation data.
        num_epochs : int
            Number of epochs to train.
        learning_rate : float
            Learning rate for optimizer.
        """

        class ToThreeChannels:
            def __call__(self, img):
                if len(img.shape) == 2:  # Ensure input is (H, W)
                    img = img.unsqueeze(0)  # Add channel dimension -> (1, H, W)
                return img.repeat(3, 1, 1)  # Duplicate channel -> (3, H, W)

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ToThreeChannels(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        class TransformTensorDataset(torch.utils.data.Dataset):
            def __init__(self, tensors, transform=None):
                """
                Custom dataset to apply transforms on a TensorDataset.

                Parameters:
                ----------
                tensors : tuple
                    A tuple of tensors, e.g., (X_train, y_train).
                transform : callable, optional
                    Transform to apply to the first tensor (e.g., X_train).
                """
                self.tensors = tensors
                self.transform = transform

            def __len__(self):
                return len(self.tensors[0])

            def __getitem__(self, idx):
                x = self.tensors[0][idx]
                y = self.tensors[1][idx]

                if self.transform:
                    x = transforms.ToPILImage()(x)
                    x = self.transform(x)  # Apply transform to X only

                return x, y

        train_dataset = TransformTensorDataset((X_train, y_train), transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # plt.figure();plt.imshow(train_dataset[2][0].permute(1, 2, 0).numpy())

        val_dataset = TransformTensorDataset((X_val, y_val), transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.001)


        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_accuracy = correct / total
            print(
                f"Epoch {epoch+1}/{num_epochs}, "
                f"Train Loss: {running_loss / len(train_loader):.4f}, "
                f"Val Loss: {val_loss / len(val_loader):.4f}, "
                f"Val Accuracy: {val_accuracy:.4f}"
            )

    def predict(self, images):
        """
        Predict the class labels for the given input images.

        Parameters:
        ----------
        images : Tensor
            Input data, assumed to be of shape (batch_size, 3, 224, 224).

        Returns:
        -------
        y_pred : Tensor
            Predicted class labels.
        """
        self.model.eval()
        images = images.to(self.device)
        with torch.no_grad():
            outputs = self.model(images)
            _, y_pred = outputs.max(1)
        return y_pred.cpu()

    def score(self, data_loader):
        """
        Compute the accuracy of the model on test data.

        Parameters:
        ----------
        data_loader : DataLoader
            DataLoader for test data.

        Returns:
        -------
        float
            Accuracy of the model.
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool):
        super(InceptionBlock, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, red_3x3, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(red_3x3, out_3x3, kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, red_5x5, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(red_5x5, out_5x5, kernel_size=5, padding=2),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, kernel_size=1),
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)

# Define the GoogleNet-like Architecture
class GoogleNetLike(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(GoogleNetLike, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initial Convolution and Pooling Layers
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Inception Blocks
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)

        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)

        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Final Layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))

        x = self.inception3a(x)
        x = self.inception3b(x)

        x = self.pool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)

        x = self.pool4(x)

        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def fit(self, X_train, y_train, X_val, y_val, batch_size=32, num_epochs=20, learning_rate=0.001):
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        self.to(self.device)
        self.train()

        for epoch in range(num_epochs):
            total_loss = 0
            self.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validation
            self.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for val_X, val_y in val_loader:
                    val_X, val_y = val_X.to(self.device), val_y.to(self.device)
                    outputs = self(val_X)
                    _, predicted = torch.max(outputs, 1)
                    total += val_y.size(0)
                    correct += (predicted == val_y).sum().item()

            val_accuracy = correct / total
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}, Val Acc: {val_accuracy:.4f}")

        return self

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            outputs = self(X.to(self.device))
            _, y_pred = torch.max(outputs, 1)
        return y_pred.cpu().numpy()

    def score(self, X_test, y_test):
        self.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size=32):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection to match dimensions
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Add the residual connection
        out = self.relu(out)
        return out

class ImprovedGCMSCNN(nn.Module):
    def __init__(self, input_channels, num_classes, input_height, input_width):
        super(ImprovedGCMSCNN, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Convolutional and residual layers
        self.block1 = ResidualBlock(input_channels, 32)
        self.block2 = ResidualBlock(32, 64)
        self.block3 = ResidualBlock(64, 128)
        self.block4 = ResidualBlock(128, 256)
        self.block5 = ResidualBlock(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        x = self.pool(self.block3(x))
        x = self.pool(self.block4(x))
        x = self.pool(self.block5(x))
        x = self.global_avg_pool(x)  # Replace flattening with global avg pooling
        x = self.flatten(x)
        x = self.dropout(self.fc1(x))
        x = self.dropout(self.fc2(x))
        x = self.fc3(x)
        return x

    def fit(self, X_train, y_train, X_val, y_val, batch_size=32, num_epochs=20, learning_rate=0.001):
        """
        Train the model with validation checks.

        Parameters:
        ----------
        X_train : Tensor
            Training data of shape (num_train_samples, input_channels, input_height, input_width).
        y_train : Tensor
            Training labels of shape (num_train_samples,).
        X_val : Tensor
            Validation data of shape (num_val_samples, input_channels, input_height, input_width).
        y_val : Tensor
            Validation labels of shape (num_val_samples,).
        batch_size : int
            Batch size for training.
        num_epochs : int
            Number of epochs to train.
        learning_rate : float
            Learning rate for optimizer.

        Returns:
        -------
        self : SimpleGCMSCNN
            The trained model.
        """


        self.reset_parameters()

        # Create DataLoader for training and validation
        # X_train = preprocess_images(X_train)
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # X_val = preprocess_images(X_val)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Define optimizer and loss function
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        self.to(self.device)
        self.train()

        for epoch in range(num_epochs):
            total_loss = 0
            self.train()  # Set model to training mode
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validation step
            self.eval()  # Set model to evaluation mode
            correct = 0
            total = 0
            with torch.no_grad():
                for val_X, val_y in val_loader:
                    val_X, val_y = val_X.to(self.device), val_y.to(self.device)
                    outputs = self(val_X)
                    _, predicted = torch.max(outputs, 1)
                    total += val_y.size(0)
                    correct += (predicted == val_y).sum().item()

            val_accuracy = correct / total

            print(
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"Training Loss: {total_loss / len(train_loader):.4f}, "
                f"Validation Accuracy: {val_accuracy:.4f}"
            )

        return self


    def predict(self, X):
        """
        Predict the class labels for the given input data.

        Parameters:
        ----------
        X : numpy.ndarray
            Input data, assumed to be of shape (num_samples, input_channels, input_height, input_width).

        Returns:
        -------
        y_pred : numpy.ndarray
            Predicted class labels.
        """
        self.eval()
        # X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self(X)
            _, y_pred = torch.max(outputs, 1)

        return y_pred.cpu().numpy()

    def score(self, X_test, y_test):
        """
        Compute the accuracy of the model on test data.

        Parameters:
        -----------
        X_test : Tensor
            Test feature data.
        y_test : Tensor
            True labels for the test data.

        Returns:
        --------
        float
            Accuracy of the model.
        """
        self.eval()  # Switch to evaluation mode
        correct = 0
        total = 0

        with torch.no_grad():  # Disable gradient computation
            for inputs, labels in DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size=32):
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)  # Get predicted class
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total

    def reset_parameters(self):
        """
        Reset all model parameters to their initial states.
        """

        for layer in self.children():
            if isinstance(layer, nn.BatchNorm2d):
                layer.running_mean.zero_()
                layer.running_var.fill_(1)
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class SimpleGCMSCNN(nn.Module):
    def __init__(self, input_channels, num_classes, input_height, input_width):
        super(SimpleGCMSCNN, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        # Dynamically calculate the size of the feature map after convolutions and pooling
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_height, input_width)
            dummy_output = self.pool(self.conv2(self.pool(self.conv1(dummy_input))))
            dummy_output = self.pool(self.conv5(self.pool(self.conv4(self.pool(self.conv3(
                self.pool(self.conv2(self.pool(self.conv1(dummy_input))))
            ))))))
            flattened_size = dummy_output.numel()

        # Calculate the size of the feature map after convolution and pooling
        self.fc1 = nn.Linear(flattened_size, 128)  # Adjust size based on input image size
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.pool(self.relu(self.conv5(x)))
        x = self.pool(x)  # Additional pooling layer
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def fit(self, X_train, y_train, X_val, y_val, batch_size=32, num_epochs=20, learning_rate=0.001):
        """
        Train the model with validation checks.

        Parameters:
        ----------
        X_train : Tensor
            Training data of shape (num_train_samples, input_channels, input_height, input_width).
        y_train : Tensor
            Training labels of shape (num_train_samples,).
        X_val : Tensor
            Validation data of shape (num_val_samples, input_channels, input_height, input_width).
        y_val : Tensor
            Validation labels of shape (num_val_samples,).
        batch_size : int
            Batch size for training.
        num_epochs : int
            Number of epochs to train.
        learning_rate : float
            Learning rate for optimizer.

        Returns:
        -------
        self : SimpleGCMSCNN
            The trained model.
        """


        self.reset_parameters()

        # Create DataLoader for training and validation
        # X_train = preprocess_images(X_train)
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # X_val = preprocess_images(X_val)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Define optimizer and loss function
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        self.to(self.device)
        self.train()

        for epoch in range(num_epochs):
            total_loss = 0
            self.train()  # Set model to training mode
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validation step
            self.eval()  # Set model to evaluation mode
            correct = 0
            total = 0
            with torch.no_grad():
                for val_X, val_y in val_loader:
                    val_X, val_y = val_X.to(self.device), val_y.to(self.device)
                    outputs = self(val_X)
                    _, predicted = torch.max(outputs, 1)
                    total += val_y.size(0)
                    correct += (predicted == val_y).sum().item()

            val_accuracy = correct / total

            print(
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"Training Loss: {total_loss / len(train_loader):.4f}, "
                f"Validation Accuracy: {val_accuracy:.4f}"
            )

        return self


    def predict(self, X):
        """
        Predict the class labels for the given input data.

        Parameters:
        ----------
        X : numpy.ndarray
            Input data, assumed to be of shape (num_samples, input_channels, input_height, input_width).

        Returns:
        -------
        y_pred : numpy.ndarray
            Predicted class labels.
        """
        self.eval()
        # X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self(X)
            _, y_pred = torch.max(outputs, 1)

        return y_pred.cpu().numpy()

    def score(self, X_test, y_test):
        """
        Compute the accuracy of the model on test data.

        Parameters:
        -----------
        X_test : Tensor
            Test feature data.
        y_test : Tensor
            True labels for the test data.

        Returns:
        --------
        float
            Accuracy of the model.
        """
        self.eval()  # Switch to evaluation mode
        correct = 0
        total = 0

        with torch.no_grad():  # Disable gradient computation
            for inputs, labels in DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size=32):
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)  # Get predicted class
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total

    def reset_parameters(self):
        """
        Reset all model parameters to their initial states.
        """

        for layer in self.children():
            if isinstance(layer, nn.BatchNorm2d):
                layer.running_mean.zero_()
                layer.running_var.fill_(1)
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class MobileNetClassifier(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(MobileNetClassifier, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the pre-trained MobileNetV2 model
        self.mobilenet = models.mobilenet_v2(pretrained=True)

        # Freeze all the feature extractor layers
        for param in self.mobilenet.features.parameters():
            param.requires_grad = False

        # Modify the first convolutional layer to accept the desired number of input channels
        self.mobilenet.features[0][0] = nn.Conv2d(
            input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
        )

        # Replace the classifier to match the number of classes
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(self.mobilenet.last_channel, 128),  # Add a hidden layer for flexibility
            nn.ReLU(),
            nn.Linear(128, num_classes)  # Final layer for classification
        )

    def forward(self, x):
        return self.mobilenet(x)

    def fit(self, X_train, y_train, X_val, y_val, batch_size=32, num_epochs=20, learning_rate=0.001):
        """
        Train the MobileNet model with validation checks.

        Parameters:
        ----------
        X_train : Tensor
            Training data of shape (num_train_samples, input_channels, input_height, input_width).
        y_train : Tensor
            Training labels of shape (num_train_samples,).
        X_val : Tensor
            Validation data of shape (num_val_samples, input_channels, input_height, input_width).
        y_val : Tensor
            Validation labels of shape (num_val_samples,).
        batch_size : int
            Batch size for training.
        num_epochs : int
            Number of epochs to train.
        learning_rate : float
            Learning rate for optimizer.

        Returns:
        -------
        self : MobileNetClassifier
            The trained model.
        """
        # Reset model parameters in the classifier head (only trainable layers)
        for layer in self.mobilenet.classifier:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        # self.reset_parameters()

        # # Define preprocessing transforms
        # preprocess = transforms.Compose([
        #     transforms.ToPILImage(),  # Convert tensor to PIL image
        #     transforms.Resize((224, 224)),  # Resize to 224x224
        #     transforms.Grayscale(num_output_channels=3),  # Convert single-channel to 3 channels
        #     transforms.ToTensor(),  # Convert back to tensor
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize to ImageNet stats
        # ])
        #
        # # Apply the preprocessing to each image in the dataset
        # def preprocess_images(train_tensor):
        #     processed_images = []
        #     for img in train_tensor:  # Loop through each image in the dataset
        #         processed_img = preprocess(
        #             img.squeeze(0).cpu().numpy())  # Remove channel dimension and apply preprocess
        #         processed_images.append(processed_img)
        #     return torch.stack(processed_images)

        # Create DataLoader for training and validation
        # X_train = preprocess_images(X_train)
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # X_val = preprocess_images(X_val)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Define optimizer and loss function
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        self.to(self.device)
        self.train()

        for epoch in range(num_epochs):
            total_loss = 0
            self.train()  # Set model to training mode
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validation step
            self.eval()  # Set model to evaluation mode
            correct = 0
            total = 0
            with torch.no_grad():
                for val_X, val_y in val_loader:
                    val_X, val_y = val_X.to(self.device), val_y.to(self.device)
                    outputs = self(val_X)
                    _, predicted = torch.max(outputs, 1)
                    total += val_y.size(0)
                    correct += (predicted == val_y).sum().item()

            val_accuracy = correct / total

            print(
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"Training Loss: {total_loss / len(train_loader):.4f}, "
                f"Validation Accuracy: {val_accuracy:.4f}"
            )

        return self

    # def fit(self, X_tensor, y_tensor, batch_size=32, epochs=20, learning_rate=0.001):
    #     """
    #     Train the CNN model.
    #
    #     Parameters:
    #     ----------
    #     X : numpy.ndarray
    #         Training data, assumed to be of shape (num_samples, input_channels, input_height, input_width).
    #     y : numpy.ndarray
    #         Labels corresponding to the training data, shape (num_samples,).
    #     batch_size : int
    #         Batch size for training.
    #     epochs : int
    #         Number of epochs to train.
    #     learning_rate : float
    #         Learning rate for optimizer.
    #
    #     Returns:
    #     -------
    #     self : CNNClassifier
    #         The trained model.
    #     """
    #
    #     # Reset model parameters
    #     self.reset_parameters()
    #
    #     # Create DataLoader
    #     dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    #     data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #
    #     # Define optimizer and loss function
    #     optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    #     loss_fn = nn.CrossEntropyLoss()
    #
    #     self.to(self.device)
    #     self.train()
    #
    #     for epoch in range(epochs):
    #         total_loss = 0
    #         for batch_X, batch_y in data_loader:
    #             optimizer.zero_grad()
    #             outputs = self(batch_X)
    #             loss = loss_fn(outputs, batch_y)
    #             loss.backward()
    #             optimizer.step()
    #             total_loss += loss.item()
    #
    #         print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_loader):.4f}")
    #
    #     return self

    def predict(self, X):
        """
        Predict the class labels for the given input data.

        Parameters:
        ----------
        X : numpy.ndarray
            Input data, assumed to be of shape (num_samples, input_channels, input_height, input_width).

        Returns:
        -------
        y_pred : numpy.ndarray
            Predicted class labels.
        """
        self.eval()
        # X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self(X)
            _, y_pred = torch.max(outputs, 1)

        return y_pred.cpu().numpy()

    def score(self, X_test, y_test):
        """
        Compute the accuracy of the model on test data.

        Parameters:
        -----------
        X_test : Tensor
            Test feature data.
        y_test : Tensor
            True labels for the test data.

        Returns:
        --------
        float
            Accuracy of the model.
        """
        self.eval()  # Switch to evaluation mode
        correct = 0
        total = 0

        with torch.no_grad():  # Disable gradient computation
            for inputs, labels in DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size=32):
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)  # Get predicted class
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total

    def reset_parameters(self):
        """
        Reset all model parameters to their initial states.
        """

        for layer in self.children():
            if isinstance(layer, nn.BatchNorm2d):
                layer.running_mean.zero_()
                layer.running_var.fill_(1)
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class Classifier:
    """
    A classifier class that wraps around various machine learning algorithms
    provided by scikit-learn. This class allows for easy switching between different classifiers
    and provides methods for training and evaluating the models using cross-validation or separate datasets.

    Parameters
    ----------
    data : numpy.ndarray
        The input data to be used for training and evaluation.
    labels : numpy.ndarray
        The labels corresponding to the input data.
    classifier_type : str, optional
        The type of classifier to use. Default is 'LDA'.
        Supported values:
        - 'LDA': Linear Discriminant Analysis
        - 'LR': Logistic Regression
        - 'RFC': Random Forest Classifier
        - 'PAC': Passive Aggressive Classifier
        - 'PER': Perceptron
        - 'RGC': Ridge Classifier
        - 'SGD': Stochastic Gradient Descent Classifier
        - 'SVM': Support Vector Machine
        - 'KNN': K-Nearest Neighbors
        - 'DTC': Decision Tree Classifier
        - 'GNB': Gaussian Naive Bayes
        - 'GBC': Gradient Boosting Classifier
    """
    def __init__(self, data, labels, classifier_type='LDA', wine_kind='bordeaux', multichannel=True,
                 window_size=5000, stride=2500, nconv=3, alpha=1):
        self.data = data
        self.labels = labels
        self.multichannel = multichannel
        self.nconv = nconv
        self.window_size = window_size
        self.stride = stride
        self.wine_kind = wine_kind
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.alpha=alpha
        self.classifier = self._get_classifier(classifier_type, multichannel=self.multichannel)



    def _get_classifier(self, classifier_type, multichannel=None):
        """
        Return the classifier object based on the classifier type.

        Parameters
        ----------
        classifier_type : str
            The type of classifier to initialize. Supported types include 'LDA', 'LR', 'RFC',
            'PAC', 'PER', 'RGC', 'SGD', 'SVM', 'KNN', 'DTC', 'GNB', and 'GBC'.

        Returns
        -------
        sklearn.base.BaseEstimator
            An instance of the selected scikit-learn classifier.
        """
        print(f'Classifier: {classifier_type}')
        if classifier_type == 'LDA':
            return LinearDiscriminantAnalysis()
        elif classifier_type == 'LR':
            return LogisticRegression(C=1.0, random_state=0, n_jobs=-1, max_iter=10000)
        elif classifier_type == 'RFC':
            return RandomForestClassifier(n_estimators=100)
        elif classifier_type == 'PAC':
            return PassiveAggressiveClassifier()
        elif classifier_type == 'PER':
            return Perceptron()
        elif classifier_type == 'RGC':
            return RidgeClassifier(alpha=self.alpha)
        elif classifier_type == 'SGD':
            return SGDClassifier()
        elif classifier_type == 'SVM':
            return SVC(kernel='rbf', random_state=0)
        elif classifier_type == 'KNN':
            return KNeighborsClassifier(n_neighbors=3)
        elif classifier_type == 'DTC':
            return DecisionTreeClassifier()
        elif classifier_type == 'GNB':
            return GaussianNB()
        elif classifier_type == 'GBC':
            return GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.1)
        elif classifier_type == 'HGBC':
            return HistGradientBoostingClassifier(max_leaf_nodes=31, learning_rate=0.2, max_iter=50, max_bins=128)
        elif classifier_type == 'CNN':
            return self._initialize_cnn()
        elif classifier_type == 'CNN1D':
            return self._initialize_cnn1d()

    def _initialize_cnn1d(self):
        """
        Initialize the 1D CNN classifier. This method can be extended with hyperparameters as needed.
        """
        # input_length = self.data.shape[2]
        input_length = self.window_size
        if not self.multichannel:
            channels = 1
        else:
            channels = self.data.shape[1]
        num_classes = len(set(self.labels))  # Number of unique labels
        model = CNN1D(input_length, num_classes, num_channels=channels, nconv=self.nconv, multichannel=self.multichannel)
        return model

    def _initialize_cnn(self):
        """
        Initialize the CNN classifier. This method can be extended with hyperparameters as needed.
        """
        input_channels = 3  # Assuming grayscale images, modify for other cases
        num_classes = len(set(self.labels))  # Number of unique labels
        height, width = self.data.shape[-2:]
        # model = CNNClassifier(input_channels, num_classes, height, width)
        # model = MobileNetClassifier(input_channels, num_classes)
        # model = SimpleGCMSCNN(input_channels, num_classes, height, width)
        # model = ImprovedGCMSCNN(input_channels, num_classes, height, width)
        # model = GoogleNetLike(input_channels, num_classes)
        model = PretrainedGoogleNet(num_classes, pretrained=True)

        # model = CustomDataParallel(model)
        return model

    def train_and_evaluate(self, n_splits=50, vintage=False, random_seed=42, test_size=None, normalize=False,
                           scaler_type='standard', use_pca=False, vthresh=0.97, region=None):
        """
        Train and evaluate the classifier using cross-validation, and calculate the mean confusion matrix.
        Can also perform PCA-based classification when `pca=True`.

        Parameters
        ----------
        n_splits : int, optional
            The number of splits for cross-validation. Default is 50.
        vintage : bool, optional
            Whether to process labels for vintage data. Default is False.
        random_seed : int, optional
            The random seed for reproducibility. Default is 42.
        test_size : float, optional
            The proportion of the dataset to include in the test split. If None, only one sample is used for testing. Default is None.
        normalize : bool, optional
            Whether to normalize the data. Default is False.
        scaler_type : str, optional
            The type of scaler to use for normalization if `normalize` is True. Default is 'standard'.
        pca : bool, optional
            Whether to apply PCA to the data. Default is False.
        vthresh : float, optional
            The variance threshold to be explained by the PCA components. Default is 0.97.

        Returns
        -------
        dict
            A dictionary containing mean accuracy, precision, recall, F1-score, and the mean confusion matrix.
        """

        from sklearn.utils.class_weight import compute_sample_weight

        # Set the random seed for reproducibility
        np.random.seed(random_seed)

        # Initialize metrics accumulators
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        confusion_matrix_sum = None
        if region == 'winery':
            custom_order = ['D', 'E', 'Q', 'P', 'R', 'Z', 'C', 'W', 'Y', 'M', 'N', 'J', 'L', 'H', 'U', 'X']
        elif region  == 'origin':
            custom_order = ['Beaune', 'Alsace', 'Neuchatel', 'Genève', 'Valais', 'Californie', 'Oregon']
        else:
           custom_order = None
        # elif region  == 'bordeaux_chateaux':
        #     custom_order = ['D', 'E', 'F', 'G', 'A', 'B', 'C']


        n_components = 100


        if use_pca:
            # Estimate best number of components (on all data)
            reducer = DimensionalityReducer(self.data)
            _, _, n_components = reducer.cumulative_variance(self.labels, variance_threshold=vthresh, plot=False)
            pca = PCA(n_components=n_components, svd_solver='randomized')

            print(f'Applying PCA with {vthresh} variance threshold')
            print(f'PCA components= {n_components}')

        print('Split', end=' ', flush=True)
        # Perform cross-validation over the specified number of splits
        for i in range(n_splits):
            # Split the data into training and testing sets
            train_indices, test_indices, X_train, X_test, y_train, y_test = self.split_data(
                vintage=vintage, test_size=test_size
            )
            # sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

            # Normalize the data if normalization is enabled
            if normalize:
                X_train, scaler = normalize_data(X_train, scaler=scaler_type)  # Fit scaler on training data
                X_test = scaler.transform(X_test)  # Transform test data using the train scaler

            # Apply PCA if enabled
            if use_pca:
               # Perform PCA on the training data
                X_train = pca.fit_transform(X_train[:, ::10])
                X_test = pca.transform(X_test[:, ::10])

            # Train the classifier on the (optionally PCA-transformed) training data
            # self.classifier.fit(X_train, y_train, sample_weight=sample_weights)
            self.classifier.fit(X_train, y_train)

            # Print the current split number every 5 iterations to show progress
            print(i, end=' ', flush=True) if i % 5 == 0 else None

            # Make predictions on the test set
            y_pred = self.classifier.predict(X_test)

            # Calculate accuracy and other metrics
            accuracy_scores.append(self.classifier.score(X_test, y_test))
            precision_scores.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            recall_scores.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

            # Compute the confusion matrix for the current split
            if custom_order is not None:
                cm = confusion_matrix(y_test, y_pred, labels=custom_order)
            else:
                cm = confusion_matrix(y_test, y_pred)

            # Accumulate the confusion matrix
            if confusion_matrix_sum is None:
                confusion_matrix_sum = np.zeros_like(cm)  # Initialize the accumulator with zeros the same shape as `cm`

            confusion_matrix_sum += cm  # Add the confusion matrix from the current split

        # Print a new line after the loop completes
        print()

        # Convert lists of scores to numpy arrays for easier statistical calculations
        accuracy_scores = np.asarray(accuracy_scores)
        precision_scores = np.asarray(precision_scores)
        recall_scores = np.asarray(recall_scores)
        f1_scores = np.asarray(f1_scores)

        # Calculate the mean confusion matrix by dividing the accumulated matrix by the number of splits
        mean_confusion_matrix = confusion_matrix_sum / n_splits

        # Print summary of results
        print("\033[96m" + "Accuracy: %0.3f (+/- %0.3f)" % (
        accuracy_scores.mean(), accuracy_scores.std() * 2) + "\033[0m")
        print("\033[96m" + "Precision: %0.3f (+/- %0.3f)" % (
        precision_scores.mean(), precision_scores.std() * 2) + "\033[0m")
        print("\033[96m" + "Recall: %0.3f (+/- %0.3f)" % (recall_scores.mean(), recall_scores.std() * 2) + "\033[0m")
        print("\033[96m" + "F1 Score: %0.3f (+/- %0.3f)" % (f1_scores.mean(), f1_scores.std() * 2) + "\033[0m")

        # Optionally print the mean confusion matrix
        print("\033[96m" + "Mean Confusion Matrix (over all splits):" + "\033[0m")
        print(mean_confusion_matrix)

        # Return the mean scores and the mean confusion matrix
        return {
            'mean_accuracy': accuracy_scores.mean(),
            'mean_precision': precision_scores.mean(),
            'mean_recall': recall_scores.mean(),
            'mean_f1_score': f1_scores.mean(),
            'mean_confusion_matrix': mean_confusion_matrix
        }

    def train_and_evaluate_balanced(self, n_splits=50, vintage=False, random_seed=42, test_size=None, normalize=False,
                                    scaler_type='standard', use_pca=False, vthresh=0.97, region=None,
                                    batch_size=32, num_epochs=10, learning_rate=0.001
                                   ):
        """
        Train and evaluate the classifier using cross-validation, with accuracy metrics for imbalanced classes.

        Parameters
        ----------
        (same as original)

        Returns
        -------
        dict
            A dictionary containing mean accuracy, balanced accuracy, weighted accuracy, precision, recall, F1-score, and
            the mean confusion matrix.
        """
        # Initialize accumulators for metrics
        accuracy_scores = []
        balanced_accuracy_scores = []
        weighted_accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        confusion_matrix_sum = None

        if region == 'winery':
            custom_order = ['D', 'E', 'Q', 'P', 'R', 'Z', 'C', 'W', 'Y', 'M', 'N', 'J', 'L', 'H', 'U', 'X']
        elif region  == 'origin':
            custom_order = ['Beaune', 'Alsace', 'Neuchatel', 'Genève', 'Valais', 'Californie', 'Oregon']
        else:
           custom_order = None

        if use_pca:
            # Apply PCA if enabled, estimating number of components to capture specified variance
            reducer = DimensionalityReducer(self.data)
            _, _, n_components = reducer.cumulative_variance(self.labels, variance_threshold=vthresh, plot=False)
            # Find number of classes in training splits and correct n_components if larger
            check_classes = self.split_data(vintage=vintage, test_size=test_size)
            n_components = min(n_components, len(check_classes[0]))
            print(f'PCA = {n_components}')
            pca = PCA(n_components=n_components, svd_solver='randomized')

        print('Split', end=' ', flush=True)
        # Cross-validation loop
        for i in range(n_splits):
            # Split data into train and test sets
            # train_indices, test_indices, X_train, X_test, y_train, y_test = self.split_data(
            #     self.labels, self.data, vintage=vintage, test_size=test_size)
            X_train, X_test, y_train, y_test = train_test_split(
                self.data, self.labels, test_size=test_size, stratify=self.labels
            )

            # Normalize data if enabled
            if normalize:
                X_train, scaler = normalize_data(X_train, scaler=scaler_type)
                X_test = scaler.transform(X_test)

            # Apply PCA if enabled
            if use_pca:
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)

            # Train the classifier without sample_weight
            self.classifier.fit(X_train, y_train)

            # Print the current split number every 5 iterations to show progress
            print(i, end=' ', flush=True) if i % 5 == 0 else None

            # Predictions on test data
            y_pred = self.classifier.predict(X_test)

            # Calculate metrics
            accuracy_scores.append(self.classifier.score(X_test, y_test))
            balanced_accuracy_scores.append(balanced_accuracy_score(y_test, y_pred))

            # Compute weighted accuracy, precision, recall, and F1-score with sample weights
            sample_weights = compute_sample_weight(class_weight='balanced', y=y_test)
            weighted_accuracy = np.average(y_pred == y_test, weights=sample_weights)
            weighted_accuracy_scores.append(weighted_accuracy)
            precision_scores.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            recall_scores.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

            # Confusion matrix for the current split
            cm = confusion_matrix(y_test, y_pred, labels=custom_order if custom_order else None)
            confusion_matrix_sum = cm if confusion_matrix_sum is None else confusion_matrix_sum + cm

        # Print the current split number every 5 iterations to show progress
        print(i, end=' ', flush=True) if i % 5 == 0 else None

        # Calculate mean confusion matrix and print results
        mean_confusion_matrix = confusion_matrix_sum / n_splits
        print(f"Accuracy: {np.mean(accuracy_scores):.3f} (+/- {np.std(accuracy_scores) * 2:.3f})")
        print(
            f"Balanced Accuracy: {np.mean(balanced_accuracy_scores):.3f} (+/- {np.std(balanced_accuracy_scores) * 2:.3f})")
        print(
            f"Weighted Accuracy: {np.mean(weighted_accuracy_scores):.3f} (+/- {np.std(weighted_accuracy_scores) * 2:.3f})")
        print(f"Precision: {np.mean(precision_scores):.3f}")
        print(f"Recall: {np.mean(recall_scores):.3f}")
        print(f"F1 Score: {np.mean(f1_scores):.3f}")
        np.set_printoptions(linewidth=np.inf)
        print("Mean Confusion Matrix:", mean_confusion_matrix)

        # Return metrics
        return {
            'mean_accuracy': np.mean(accuracy_scores),
            'mean_balanced_accuracy': np.mean(balanced_accuracy_scores),
            'mean_weighted_accuracy': np.mean(weighted_accuracy_scores),
            'mean_precision': np.mean(precision_scores),
            'mean_recall': np.mean(recall_scores),
            'mean_f1_score': np.mean(f1_scores),
            'mean_confusion_matrix': mean_confusion_matrix
        }

    def train_and_evaluate_balanced_with_alpha(self, n_splits=50, vintage=False, random_seed=42, test_size=None,
                                               normalize=False,
                                               scaler_type='standard', use_pca=False, vthresh=0.97, region=None,
                                               batch_size=32, num_epochs=10, learning_rate=0.001,
                                               alpha_range=None, num_test=1, channel_weights=None):
        """
        Train and evaluate the classifier using cross-validation, with accuracy metrics for imbalanced classes.
        Learn the optimal alpha parameter for Ridge Classifier during training.

        Parameters
        ----------
        alpha_range : list or None
            Range of alpha values to test during cross-validation. If None, defaults to [0.1, 1.0, 10.0, 100.0].

        Returns
        -------
        dict
            A dictionary containing mean accuracy, balanced accuracy, weighted accuracy, precision, F1-score,
            mean confusion matrix, and the mean optimal alpha value.
        """


        if channel_weights is not None:
            self.data = self.data * channel_weights.detach().numpy()

        # Separate all the m/z profiles and give them the label of the sample
        self.labels = np.repeat(self.labels, self.data.shape[2])
        self.data = self.data.transpose(2, 0, 1).reshape(-1, self.data.shape[1])

        # Set default alpha range if not provided
        if alpha_range is None:
            alpha_range = [0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 1500.0, 2000.0, 4000.0]

        # Initialize accumulators for metrics and alpha
        accuracy_scores = []
        balanced_accuracy_scores = []
        precision_scores = []
        f1_scores = []
        best_alpha_values = []
        confusion_matrix_sum = None

        # Cross-validation loop
        for i in range(n_splits):
            # Split data into train and test sets
            train_indices, test_indices, X_train, X_test, y_train, y_test = self.split_data(self.labels, self.data,
                vintage=vintage,test_size=test_size, num_test=num_test)

            # Normalize data if enabled
            if normalize:
                X_train, scaler = normalize_data(X_train, scaler=scaler_type)
                X_test = scaler.transform(X_test)

            # Apply PCA if enabled
            if use_pca:
                reducer = DimensionalityReducer(self.data)
                _, _, n_components = reducer.cumulative_variance(self.labels, variance_threshold=vthresh, plot=False)
                n_components = min(n_components, len(set(y_train)))  # Adjust PCA components based on class count
                pca = PCA(n_components=n_components, svd_solver='randomized')
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)

            # Learn optimal alpha using RidgeClassifierCV
            ridge_classifier = RidgeClassifierCV(
                alphas=alpha_range, scoring='balanced_accuracy', store_cv_values=True
            )
            ridge_classifier.fit(X_train, y_train)
            best_alpha = ridge_classifier.alpha_
            best_alpha_values.append(best_alpha)

            # Predictions and metrics
            y_pred = ridge_classifier.predict(X_test)
            accuracy_scores.append(ridge_classifier.score(X_test, y_test))
            balanced_accuracy_scores.append(balanced_accuracy_score(y_test, y_pred))

            # Compute precision and F1 score
            precision_scores.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

            # Compute confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)))
            confusion_matrix_sum = cm if confusion_matrix_sum is None else confusion_matrix_sum + cm

            print(f"Split {i + 1}/{n_splits} completed. Best alpha: {best_alpha}")

        # Calculate mean metrics
        mean_confusion_matrix = confusion_matrix_sum / n_splits
        mean_alpha = np.mean(best_alpha_values)

        # Print summary
        print(f"Mean Alpha: {mean_alpha:.3f}")
        print(f"Accuracy: {np.mean(accuracy_scores):.3f} (+/- {np.std(accuracy_scores) * 2:.3f})")
        print(
            f"Balanced Accuracy: {np.mean(balanced_accuracy_scores):.3f} (+/- {np.std(balanced_accuracy_scores) * 2:.3f})")
        print(f"Precision: {np.mean(precision_scores):.3f}")
        print(f"F1 Score: {np.mean(f1_scores):.3f}")
        print("Mean Confusion Matrix:\n", mean_confusion_matrix)

        # Return metrics
        return {
            'mean_accuracy': np.mean(accuracy_scores),
            'mean_balanced_accuracy': np.mean(balanced_accuracy_scores),
            'mean_precision': np.mean(precision_scores),
            'mean_f1_score': np.mean(f1_scores),
            'mean_confusion_matrix': mean_confusion_matrix,
            'mean_alpha': mean_alpha
        }


    def train_and_evaluate_balanced_with_passed_weights(self, n_splits=50, vintage=False, random_seed=42, test_size=None,
                                               normalize=False,
                                               scaler_type='standard', use_pca=False, vthresh=0.97, region=None,
                                               batch_size=32, num_epochs=10, learning_rate=0.001,
                                               alpha_range=None, num_test=1, channel_weights=None):
        """
        Train and evaluate the classifier using cross-validation, with accuracy metrics for imbalanced classes.
        Learn the optimal alpha parameter for Ridge Classifier during training.

        Parameters
        ----------
        alpha_range : list or None
            Range of alpha values to test during cross-validation. If None, defaults to [0.1, 1.0, 10.0, 100.0].

        Returns
        -------
        dict
            A dictionary containing mean accuracy, balanced accuracy, weighted accuracy, precision, F1-score,
            mean confusion matrix, and the mean optimal alpha value.
        """

        if channel_weights is not None:
            self.data = self.data * channel_weights

        # Separate all the m/z profiles and give them the label of the sample
        self.labels = np.repeat(self.labels, self.data.shape[2])
        self.data = self.data.transpose(2, 0, 1).reshape(-1, self.data.shape[1])

        # Set default alpha range if not provided
        if alpha_range is None:
            alpha_range = [0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 1500.0, 2000.0, 4000.0]

        # Initialize accumulators for metrics and alpha
        accuracy_scores = []
        balanced_accuracy_scores = []
        precision_scores = []
        f1_scores = []
        best_alpha_values = []
        confusion_matrix_sum = None

        # Cross-validation loop
        for i in range(n_splits):
            # Split data into train and test sets
            train_indices, test_indices, X_train, X_test, y_train, y_test = self.split_data(self.labels, self.data,
                vintage=vintage,test_size=test_size, num_test=num_test)

            # Normalize data if enabled
            if normalize:
                X_train, scaler = normalize_data(X_train, scaler=scaler_type)
                X_test = scaler.transform(X_test)

            # Apply PCA if enabled
            if use_pca:
                reducer = DimensionalityReducer(self.data)
                _, _, n_components = reducer.cumulative_variance(self.labels, variance_threshold=vthresh, plot=False)
                n_components = min(n_components, len(set(y_train)))  # Adjust PCA components based on class count
                pca = PCA(n_components=n_components, svd_solver='randomized')
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)

            # Learn optimal alpha using RidgeClassifierCV
            ridge_classifier = RidgeClassifier()
            ridge_classifier.fit(X_train, y_train)

            # Predictions and metrics
            y_pred = ridge_classifier.predict(X_test)
            accuracy_scores.append(ridge_classifier.score(X_test, y_test))
            balanced_accuracy_scores.append(balanced_accuracy_score(y_test, y_pred))

            # Compute precision and F1 score
            precision_scores.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

            # Compute confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)))
            confusion_matrix_sum = cm if confusion_matrix_sum is None else confusion_matrix_sum + cm

            print(f"Split {i + 1}/{n_splits} completed.")

        # Calculate mean metrics
        mean_confusion_matrix = confusion_matrix_sum / n_splits

        # Print summary
        print(f"Accuracy: {np.mean(accuracy_scores):.3f} (+/- {np.std(accuracy_scores) * 2:.3f})")
        print(
            f"Balanced Accuracy: {np.mean(balanced_accuracy_scores):.3f} (+/- {np.std(balanced_accuracy_scores) * 2:.3f})")
        print(f"Precision: {np.mean(precision_scores):.3f}")
        print(f"F1 Score: {np.mean(f1_scores):.3f}")
        print("Mean Confusion Matrix:\n", mean_confusion_matrix)

        # Return metrics
        return {
            'mean_accuracy': np.mean(accuracy_scores),
            'mean_balanced_accuracy': np.mean(balanced_accuracy_scores),
            'mean_precision': np.mean(precision_scores),
            'mean_f1_score': np.mean(f1_scores),
            'mean_confusion_matrix': mean_confusion_matrix,
        }


    def train_and_evaluate_balanced_with_best_alpha(
            self, n_splits=50, vintage=False, test_size=None, normalize=False, scaler_type='standard', use_pca=False,
            vthresh=0.97, best_alpha=1, region=None):
        """
        Train and evaluate the classifier using cross-validation, with accuracy metrics for imbalanced classes.
        Learn the optimal alpha parameter for Ridge Classifier during training.

        Parameters
        ----------


        Returns
        -------
        dict
            A dictionary containing mean accuracy, balanced accuracy, weighted accuracy, precision, F1-score,
            mean confusion matrix, and the mean optimal alpha value.
        """

        if region == 'winery':
            custom_order = ['D', 'E', 'Q', 'P', 'R', 'Z', 'C', 'W', 'Y', 'M', 'N', 'J', 'L', 'H', 'U', 'X']
        elif region  == 'origin':
            custom_order = ['Beaune', 'Alsace', 'Neuchatel', 'Genève', 'Valais', 'Californie', 'Oregon']
        else:
           custom_order = None

        # Separate all the m/z profiles and give them the label of the sample
        self.labels = np.repeat(self.labels, self.data.shape[2])
        # self.data = self.data.transpose(2, 0, 1).reshape(-1, self.data.shape[1])
        self.data = self.data.transpose(0, 2, 1).reshape(-1, self.data.shape[1])

        # Initialize accumulators for metrics and alpha
        accuracy_scores = []
        balanced_accuracy_scores = []
        precision_scores = []
        f1_scores = []
        confusion_matrix_sum = None

        ridge_classifier = RidgeClassifier(alpha=best_alpha)

        # Cross-validation loop
        for i in range(n_splits):
            # Split data into train and test sets
            # train_indices, test_indices, X_train, X_test, y_train, y_test = self.split_data(self.labels, self.data,
            #     vintage=vintage,test_size=test_size, num_test=num_test)

            X_train, X_test, y_train, y_test = train_test_split(
                self.data, self.labels, test_size=0.1, stratify=self.labels
            )

            # Normalize data if enabled
            if normalize:
                X_train, scaler = normalize_data(X_train, scaler=scaler_type)
                X_test = scaler.transform(X_test)

            # Apply PCA if enabled
            if use_pca:
                reducer = DimensionalityReducer(self.data)
                _, _, n_components = reducer.cumulative_variance(self.labels, variance_threshold=vthresh, plot=False)
                n_components = min(n_components, len(set(y_train)))  # Adjust PCA components based on class count
                pca = PCA(n_components=n_components, svd_solver='randomized')
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)

            ridge_classifier.fit(X_train, y_train)

            # Predictions and metrics
            y_pred = ridge_classifier.predict(X_test)
            accuracy_scores.append(ridge_classifier.score(X_test, y_test))
            balanced_accuracy_scores.append(balanced_accuracy_score(y_test, y_pred))

            # Compute precision and F1 score
            precision_scores.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

            # Compute confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)))
            # cm = confusion_matrix(y_test, y_pred, labels=custom_order)
            confusion_matrix_sum = cm if confusion_matrix_sum is None else confusion_matrix_sum + cm

            print(f"Split {i + 1}/{n_splits} completed.")

        # Calculate mean metrics
        mean_confusion_matrix = confusion_matrix_sum / n_splits

        # Print summary
        print(f"Accuracy: {np.mean(accuracy_scores):.3f} (+/- {np.std(accuracy_scores) * 2:.3f})")
        print(
            f"Balanced Accuracy: {np.mean(balanced_accuracy_scores):.3f} (+/- {np.std(balanced_accuracy_scores) * 2:.3f})")
        print(f"Precision: {np.mean(precision_scores):.3f}")
        print(f"F1 Score: {np.mean(f1_scores):.3f}")
        print("Mean Confusion Matrix:\n", mean_confusion_matrix)

        # Return metrics
        return {
            'mean_accuracy': np.mean(accuracy_scores),
            'mean_balanced_accuracy': np.mean(balanced_accuracy_scores),
            'mean_precision': np.mean(precision_scores),
            'mean_f1_score': np.mean(f1_scores),
            'mean_confusion_matrix': mean_confusion_matrix,
        }


    def train_and_evaluate_balanced_with_best_alpha2(
            self, n_splits=50, test_size=0.2, normalize=False, scaler_type='standard',
            use_pca=False, vthresh=0.97, best_alpha=1, region=None
    ):
        """
        Train and evaluate the classifier using cross-validation, with stratified train-test splits at the sample level.

        Parameters
        ----------
        n_splits : int
            Number of cross-validation splits.
        test_size : float
            Proportion of the data to include in the test split.
        normalize : bool
            Whether to normalize the data.
        scaler_type : str
            Type of scaler to use for normalization.
        use_pca : bool
            Whether to apply PCA for dimensionality reduction.
        vthresh : float
            Variance threshold for PCA.
        best_alpha : float
            Regularization parameter for RidgeClassifier.
        region : str
            Custom label ordering for visualization.

        Returns
        -------
        dict
            A dictionary containing evaluation metrics and the confusion matrix.
        """

        # Custom label ordering for specific regions (if needed)
        custom_order = {
            'winery': ['D', 'E', 'Q', 'P', 'R', 'Z', 'C', 'W', 'Y', 'M', 'N', 'J', 'L', 'H', 'U', 'X'],
            'origin': ['Beaune', 'Alsace', 'Neuchatel', 'Genève', 'Valais', 'Californie', 'Oregon']
        }.get(region, None)

        # Labels and data at the sample level
        sample_labels = self.labels
        sample_data = self.data  # Shape: (num_samples, features, channels)

        # Initialize accumulators for metrics
        accuracy_scores = []
        balanced_accuracy_scores = []
        precision_scores = []
        f1_scores = []
        confusion_matrix_sum = None

        ridge_classifier = RidgeClassifier(alpha=best_alpha)

        # Cross-validation loop
        for i in range(n_splits):
            # Perform stratified split at the sample level
            train_idx, test_idx = train_test_split(
                np.arange(len(sample_labels)),
                test_size=test_size,
                stratify=sample_labels,
                random_state=i
            )

            # Prepare train and test data while keeping channels grouped by sample
            X_train = sample_data[train_idx].transpose(0, 2, 1).reshape(-1, sample_data.shape[1])  # Flatten channels
            X_test = sample_data[test_idx].transpose(0, 2, 1).reshape(-1, sample_data.shape[1])
            y_train = np.repeat(sample_labels[train_idx], sample_data.shape[2])  # Repeat labels for all channels
            y_test = np.repeat(sample_labels[test_idx], sample_data.shape[2])

            # Normalize data if required
            if normalize:
                X_train, scaler = normalize_data(X_train, scaler=scaler_type)
                X_test = scaler.transform(X_test)

            # Apply PCA if required
            if use_pca:
                reducer = DimensionalityReducer(self.data)
                _, _, n_components = reducer.cumulative_variance(self.labels, variance_threshold=vthresh, plot=False)
                n_components = min(n_components, len(set(y_train)))  # Adjust PCA components based on class count
                pca = PCA(n_components=n_components, svd_solver='randomized')
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)

            # Train and predict
            ridge_classifier.fit(X_train, y_train)
            y_pred = ridge_classifier.predict(X_test)

            # Compute metrics
            accuracy_scores.append(ridge_classifier.score(X_test, y_test))
            balanced_accuracy_scores.append(balanced_accuracy_score(y_test, y_pred))
            precision_scores.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

            # Compute confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y_train)))
            confusion_matrix_sum = cm if confusion_matrix_sum is None else confusion_matrix_sum + cm

            print(f"Split {i + 1}/{n_splits} completed.")

        # Calculate mean metrics
        mean_confusion_matrix = confusion_matrix_sum / n_splits

        # Print summary
        print(f"Accuracy: {np.mean(accuracy_scores):.3f} (+/- {np.std(accuracy_scores) * 2:.3f})")
        print(
            f"Balanced Accuracy: {np.mean(balanced_accuracy_scores):.3f} (+/- {np.std(balanced_accuracy_scores) * 2:.3f})")
        print(f"Precision: {np.mean(precision_scores):.3f}")
        print(f"F1 Score: {np.mean(f1_scores):.3f}")
        print("Mean Confusion Matrix:\n", mean_confusion_matrix)

        # Return metrics
        return {
            'mean_accuracy': np.mean(accuracy_scores),
            'mean_balanced_accuracy': np.mean(balanced_accuracy_scores),
            'mean_precision': np.mean(precision_scores),
            'mean_f1_score': np.mean(f1_scores),
            'mean_confusion_matrix': mean_confusion_matrix,
        }


    def train_and_evaluate_balanced_with_minimize(
            self, n_splits=50, vintage=False, random_seed=42, test_size=None,
            normalize=False, scaler_type='standard', use_pca=False, vthresh=0.97, region=None,
            alpha_range=None, num_test=1):
        """
        Train and evaluate the classifier using cross-validation, optimizing channel weights with numerical optimization.
        Learn the optimal alpha parameter for Ridge Classifier during training.

        Parameters
        ----------
        alpha_range : list or None
            Range of alpha values to test during cross-validation. If None, defaults to [0.1, 1.0, 10.0, 100.0].

        Returns
        -------
        dict
            A dictionary containing mean accuracy, balanced accuracy, weighted accuracy, precision, F1-score,
            mean confusion matrix, final weights, and the mean optimal alpha value.
        """

        # Default alpha range
        if alpha_range is None:
            alpha_range = [0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 1500.0, 2000.0, 4000.0]
            alpha_range = [10.0]

        # Separate m/z profiles and repeat labels for each channel
        n_channels = self.data.shape[2]

        # Metrics accumulators
        accuracy_scores = []
        balanced_accuracy_scores = []
        precision_scores = []
        f1_scores = []
        best_alpha_values = []
        confusion_matrix_sum = None

        def objective(weights):
            """
            Objective function to optimize weights. It computes the negative balanced accuracy
            of the Ridge ClassifierCV with the given weights applied to the m/z channels.
            """

            weighted_data = self.data * weights.reshape(1, 1, -1)
            reshaped_data = weighted_data.transpose(2, 0, 1).reshape(-1, self.data.shape[1])
            labels_expanded = np.repeat(self.labels, n_channels)
            accuracy_scores_split = []

            # Train RidgeClassifierCV
            ridge_classifier = RidgeClassifierCV(alphas=alpha_range, scoring='balanced_accuracy', store_cv_values=True)

            for i in range(n_splits):
                train_indices, test_indices, X_train, X_test, y_train, y_test = self.split_data(
                    labels_expanded, reshaped_data,  vintage=vintage, test_size=test_size, num_test=num_test
                )

                ridge_classifier.fit(X_train, y_train)
                best_alpha_values.append(ridge_classifier.alpha_)

                # Accuracy for this split
                y_pred = ridge_classifier.predict(X_test)
                balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
                accuracy_scores_split.append(balanced_accuracy)

            # Return the negative mean balanced accuracy as the loss
            # print(np.mean(accuracy_scores_split))
            return -np.mean(accuracy_scores_split)

        # Initialize weights for m/z channels
        initial_weights = np.ones(n_channels)

        # Optimize weights
        result = minimize(objective, initial_weights, method='L-BFGS-B', bounds=[(0, None)] * n_channels,
                          options={'maxiter': 10, 'disp': True, 'iprint': 1}
                          )
        optimized_weights = result.x

        # Apply optimized weights to data
        print(f'Optimzed weights: {optimized_weights}')
        weighted_data = self.data * optimized_weights
        reshaped_data = weighted_data.transpose(2, 0, 1).reshape(-1, self.data.shape[1])
        labels_expanded = np.repeat(self.labels, n_channels)

        # Final evaluation
        for i in range(n_splits):
            train_indices, test_indices, X_train, X_test, y_train, y_test = self.split_data(
                labels_expanded, reshaped_data, vintage=vintage, test_size=test_size, num_test=num_test
            )

            if normalize:
                X_train, scaler = normalize_data(X_train, scaler=scaler_type)
                X_test = scaler.transform(X_test)

            if use_pca:
                reducer = DimensionalityReducer(self.data)
                _, _, n_components = reducer.cumulative_variance(self.labels, variance_threshold=vthresh, plot=False)
                n_components = min(n_components, len(set(y_train)))
                pca = PCA(n_components=n_components, svd_solver='randomized')
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)

            ridge_classifier = RidgeClassifierCV(alphas=alpha_range, scoring='balanced_accuracy', store_cv_values=True)
            ridge_classifier.fit(X_train, y_train)

            y_pred = ridge_classifier.predict(X_test)
            accuracy_scores.append(ridge_classifier.score(X_test, y_test))
            balanced_accuracy_scores.append(balanced_accuracy_score(y_test, y_pred))
            precision_scores.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

            cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)))
            confusion_matrix_sum = cm if confusion_matrix_sum is None else confusion_matrix_sum + cm

        # Final metrics
        mean_confusion_matrix = confusion_matrix_sum / n_splits
        mean_alpha = np.mean(best_alpha_values)

        print(f"Optimized Weights: {optimized_weights}")
        print(f"Mean Alpha: {mean_alpha:.3f}")
        print(f"Mean Accuracy: {np.mean(accuracy_scores):.3f}")
        print(f"Mean Balanced Accuracy: {np.mean(balanced_accuracy_scores):.3f}")

        return {
            'mean_accuracy': np.mean(accuracy_scores),
            'mean_balanced_accuracy': np.mean(balanced_accuracy_scores),
            'mean_precision': np.mean(precision_scores),
            'mean_f1_score': np.mean(f1_scores),
            'mean_confusion_matrix': mean_confusion_matrix,
            'final_weights': optimized_weights,
            'mean_alpha': mean_alpha
        }


    def train_and_evaluate_balanced_with_weights_and_alpha0(
            self, n_splits=50, vintage=False, random_seed=42, test_size=None,
            normalize=False, scaler_type='standard', use_pca=False, vthresh=0.97, region=None,
            batch_size=32, num_epochs=10, learning_rate=0.001, alpha_range=None, num_test=1):
        """
        Train and evaluate the classifier using cross-validation, with accuracy metrics for imbalanced classes.
        Learn the weights for each m/z channel and the optimal alpha parameter for Ridge Classifier.

        Returns
        -------
        dict
            Metrics including mean accuracy, balanced accuracy, precision, F1-score,
            mean confusion matrix, final weights, and mean alpha.
        """

        def ridge_loss(predictions, targets, weights, alpha):
            """
            Compute Ridge Loss: Mean Squared Error + Regularization on weights.

            Parameters
            ----------
            predictions : torch.Tensor
                Predicted values from the model.
            targets : torch.Tensor
                Ground truth labels.
            weights : torch.Tensor
                Learnable weights for channels.
            alpha : torch.Tensor
                Ridge regularization parameter.

            Returns
            -------
            torch.Tensor
                Computed Ridge Loss.
            """

            if predictions.dim() > 1:
                # Example: Taking the score for the correct class if multi-class
                predictions = predictions.gather(1, targets_train_tensor.long().unsqueeze(1)).squeeze()

            # Mean Squared Error Loss
            # mse_loss = torch.nn.functional.mse_loss(predictions, targets)
            ce_loss = torch.nn.functional.cross_entropy(predictions, targets)

            # Ridge Regularization (L2 Norm on Weights)
            # l2_reg = alpha * torch.sum(weights ** 2)
            l2_reg = optimizer.param_groups[0]['weight_decay'] * torch.sum(weights ** 2)

            # Combine MSE with L2 regularization
            return ce_loss + l2_reg

        # Set default alpha range if not provided
        if alpha_range is None:
            alpha_range = [0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 1500.0, 2000.0, 4000.0]

        # Number of channels in the original data
        n_channels = self.data.shape[2]

        # Initialize weights for each m/z channel
        channel_weights = torch.ones(n_channels, requires_grad=True, dtype=torch.float32)
        optimizer = torch.optim.Adam([channel_weights], lr=learning_rate, weight_decay=0.01)

        # Initialize accumulators for metrics and alpha
        accuracy_scores_epoch = []
        balanced_accuracy_scores_epoch = []
        precision_scores_epoch = []
        f1_scores_epoch = []
        confusion_matrix_epoch_sum = None
        best_alpha_values_epoch = []

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")

            # Apply channel weights to the data
            weighted_data = self.data * channel_weights.detach().numpy()
            # weighted_data = self.data * channel_weights.unsqueeze(0).unsqueeze(2)

            # Reshape the data so channels become independent samples
            labels_expanded = np.repeat(self.labels, n_channels)
            data_expanded = weighted_data.transpose(2, 0, 1).reshape(-1, self.data.shape[1])

            # Initialize accumulators for this epoch
            accuracy_scores = []
            balanced_accuracy_scores = []
            precision_scores = []
            f1_scores = []
            best_alpha_values = []
            confusion_matrix_sum = None

            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(labels_expanded)

            ridge_classifier = RidgeClassifierCV(alphas=alpha_range, scoring='balanced_accuracy', store_cv_values=True)

            for i in range(n_splits):
                # Split data into train and test sets
                train_indices, test_indices, X_train, X_test, y_train, y_test = self.split_data(
                    encoded_labels, data_expanded, vintage=vintage, test_size=test_size, num_test=num_test
                )

                # Normalize data if enabled
                if normalize:
                    X_train, scaler = normalize_data(X_train, scaler=scaler_type)
                    X_test = scaler.transform(X_test)

                # Apply PCA if enabled
                if use_pca:
                    reducer = DimensionalityReducer(self.data)
                    _, _, n_components = reducer.cumulative_variance(self.labels, variance_threshold=vthresh,
                                                                     plot=False)
                    n_components = min(n_components, len(set(y_train)))  # Adjust PCA components based on class count
                    pca = PCA(n_components=n_components, svd_solver='randomized')
                    X_train = pca.fit_transform(X_train)
                    X_test = pca.transform(X_test)

                # Learn optimal alpha using RidgeClassifierCV
                ridge_classifier.fit(X_train, y_train)
                best_alpha = ridge_classifier.alpha_
                best_alpha_values.append(ridge_classifier.alpha_)

                # Predictions and metrics
                y_pred = ridge_classifier.predict(X_test)
                accuracy = ridge_classifier.score(X_test, y_test)
                accuracy_scores.append(accuracy)
                balanced_accuracy_scores.append(balanced_accuracy_score(y_test, y_pred))
                precision_scores.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
                f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

                # Compute confusion matrix
                cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)))
                confusion_matrix_sum = cm if confusion_matrix_sum is None else confusion_matrix_sum + cm

                print(f"Split {i + 1}/{n_splits}: Best Alpha = {best_alpha:.3f}, Accuracy = {accuracy:.4f}")

            # Compute metrics for the epoch
            mean_accuracy = np.mean(accuracy_scores)
            mean_balanced_accuracy = np.mean(balanced_accuracy_scores)
            mean_precision = np.mean(precision_scores)
            mean_f1_score = np.mean(f1_scores)
            mean_alpha = np.mean(best_alpha_values)

            # Append metrics for this epoch
            accuracy_scores_epoch.append(mean_accuracy)
            balanced_accuracy_scores_epoch.append(mean_balanced_accuracy)
            precision_scores_epoch.append(mean_precision)
            f1_scores_epoch.append(mean_f1_score)

            if confusion_matrix_epoch_sum is None:
                confusion_matrix_epoch_sum = confusion_matrix_sum
            else:
                confusion_matrix_epoch_sum += confusion_matrix_sum

            best_alpha_values_epoch.append(mean_alpha)

            # Train on the entire training data with the best alpha
            predictions_train = torch.tensor(ridge_classifier.decision_function(data_expanded), dtype=torch.float32)
            targets_train_tensor = torch.tensor(encoded_labels, dtype=torch.long)

            # Validate shapes
            assert predictions_train.shape[0] == targets_train_tensor.shape[0], \
                f"Shape mismatch: predictions {predictions_train.shape}, targets {targets_train_tensor.shape}"

            # Compute loss for the epoch
            # ridge_loss_value = ridge_loss(predictions_train, targets_train_tensor, channel_weights, mean_alpha)
            # predictions_train = torch.nn.functional.softmax(predictions_train, dim=1)  # Ensure normalized probabilities
            ce_loss = torch.nn.functional.cross_entropy(predictions_train, targets_train_tensor)

            # Add regularization
            # l2_reg = optimizer.param_groups[0]['weight_decay'] * torch.sum(channel_weights ** 2)
            ridge_loss_value = ce_loss # + l2_reg

            # Update weights
            optimizer.zero_grad()
            ridge_loss_value.backward()
            print("Gradients:", channel_weights.grad)
            optimizer.step()
            print(channel_weights)

            print(f"Epoch {epoch + 1} Summary: Accuracy = {mean_accuracy:.4f}, Learned Alpha = {mean_alpha:.3f}")

        # Final metrics
        mean_confusion_matrix = confusion_matrix_epoch_sum / n_splits
        final_weights = channel_weights.detach().numpy()
        mean_alpha_final = np.mean(best_alpha_values_epoch)

        print(f"Final Weights: {final_weights}")
        print(f"Final Learned Alpha: {mean_alpha_final:.3f}")

        return {
            'mean_accuracy': np.mean(accuracy_scores_epoch),
            'mean_balanced_accuracy': np.mean(balanced_accuracy_scores_epoch),
            'mean_precision': np.mean(precision_scores_epoch),
            'mean_f1_score': np.mean(f1_scores_epoch),
            'mean_confusion_matrix': mean_confusion_matrix,
            'final_weights': final_weights,
            'mean_alpha': mean_alpha_final
        }


    def train_and_evaluate_separate_datasets(self, X_train, y_train, X_test, y_test, n_splits=50, vintage=False,
                                             random_seed=42, normalize=True, scaler_type='standard'):
        """
        Train the classifier on the provided training dataset and evaluate its performance on the testing dataset
        using cross-validation.
        Parameters
        ----------
        X_train : numpy.ndarray
            Training data.
        y_train : numpy.ndarray
            Training labels.
        X_test : numpy.ndarray
            Testing data.
        y_test : numpy.ndarray
            Testing labels.
        n_splits : int, optional
            The number of splits for cross-validation. Default is 50.
        vintage : bool, optional
            Whether to process labels for vintage data. Default is False.
        random_seed : int, optional
            The random seed for reproducibility. Default is 42.
        normalize : bool, optional
            Whether to normalize the data. Default is True.
        scaler_type : str, optional
            The type of scaler to use for normalization if `normalize` is True. Default is 'standard'.

        Returns
        -------
        float
            The mean accuracy score from cross-validation.

        Notes
        -----
        This function trains the classifier on the training data and evaluates it on the testing data.
        """

        # Set the random seed for reproducibility, ensuring that the data splits and other random processes are consistent
        np.random.seed(random_seed)

        # Normalize the training data if the normalize flag is set to True
        if normalize:
            X_train, scaler_train = normalize_data(X_train, scaler=scaler_type)

        # Train the classifier using the training data
        self.classifier.fit(X_train, y_train)

        # Initialize a new Classifier instance for the testing data
        test_cls = Classifier(X_test, y_test)

        # Initialize a list to store the accuracy scores from each split
        scores = []

        # Print 'Split' to indicate the start of cross-validation, keeping the output on the same line
        print('Split', end=' ', flush=True)

        # Perform cross-validation for the specified number of splits
        for i in range(n_splits):
            # Split the testing data into "in" and "out" samples for cross-validation
            in_indices, out_indices, X_in, X_out, y_in, y_out = test_cls.split_data(vintage=vintage, test_size=None)

            # Normalize the samples if normalization is enabled
            if normalize:
                X_in, scaler_test = normalize_data(X_in, scaler=scaler_type)
                # Use scaler fitted on X_in to transform X_out to ensure consistent scaling and prevent data leakage.
                X_out = scaler_test.transform(X_out)
            # X_out = scaler_train.transform(X_out)

            # Evaluate the classifier on the "out" sample and append the score to the list
            scores.append(self.classifier.score(X_out, y_out))
            print(self.classifier.score(X_out, y_out))

            # Print the current split number every 5 iterations to show progress
            print(i, end=' ', flush=True) if i % 5 == 0 else None

        # Convert the list of scores to a numpy array for easier statistical calculations
        scores = np.asarray(scores)

        # Print a new line after the loop completes
        print()

        # Print the mean accuracy and the standard deviation across the cross-validation splits
        print("\033[96m" + "Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2) + "\033[0m")

        # Return the mean accuracy score as the final result
        return scores.mean()

    def _process_labels(self, vintage=False):
        """
        Process the labels to extract relevant parts based on whether the data is vintage or not.

        Parameters
        ----------
        vintage : bool
            If True, the function processes labels to extract a substring starting from the first digit
            found in the label (assuming vintage data formatting). If False, it processes labels to
            extract a single character or digit before the first digit found.

        Returns
        -------
        numpy.ndarray
            An array of processed labels.
        """
        if self.wine_kind == 'pinot_noir':
            processed_labels = self.labels
        elif self.wine_kind == 'bordeaux':
            processed_labels = []

            # Iterate over each label in the labels list
            for label in self.labels:
                # Search for the first digit in the label
                match = re.search(r'\d+', label)

                if vintage:
                    # If processing vintage data, extract the substring starting from the first digit
                    processed_labels.append(label[match.start():])
                else:
                    # If not vintage, extract the character before the first digit
                    if label[match.start() - 1] == '_':
                        # If the character before the digit is an underscore, take the character before the underscore
                        lb = label[match.start() - 2]
                    else:
                        # Otherwise, take the character directly before the first digit
                        lb = label[match.start() - 1]
                    processed_labels.append(lb)

        # Return the processed labels as a numpy array
        return np.array(processed_labels)

    def split_data(self, labels, data, vintage=False, test_size=None, num_test=1):
        """
        Split the data into training and testing sets based on labels.

        Parameters
        ----------
        vintage : bool, optional
            Whether to process labels for vintage data. Default is False.
        test_size : float, optional
            The proportion of the dataset to include in the test split. If None, only one sample
            per unique label is used for testing. Default is None.

        Returns
        -------
        tuple
            A tuple containing the following elements:

            - train_indices : numpy.ndarray
                Indices of the training data samples.

            - test_indices : numpy.ndarray
                Indices of the testing data samples.

            - X_train : numpy.ndarray
                The training data.

            - X_test : numpy.ndarray
                The testing data.

            - y_train : numpy.ndarray
                The labels for the training data.

            - y_test : numpy.ndarray
                The labels for the testing data.

        Notes
        -----
        This function splits the dataset into training and testing sets by first processing the labels.
        The splitting is done in such a way that either one sample per unique label is reserved for testing
        (if test_size is None) or a specified proportion of samples per label is reserved for testing.
        The samples are randomly shuffled before splitting to ensure randomness in the selection.
        """

        # if self.wine_kind == 'bordeaux':
        #     # Process the labels according to whether they are vintage or not
        #     processed_labels = self._process_labels(vintage)
        # else:
        processed_labels = labels

        # Initialize lists to store indices for training and testing samples
        test_indices = []
        train_indices = []

        # Iterate over each unique label to perform stratified splitting
        for label in np.unique(processed_labels):
            # Find indices of all samples corresponding to the current label
            label_indices = np.where(np.array(processed_labels) == label)[0]

            # Shuffle these indices to ensure randomness in splitting
            np.random.shuffle(label_indices)

            if test_size is None:
                # If test_size is not specified, select one sample per label for testing
                test_indices.extend(label_indices[:num_test])  # Take the first shuffled index for testing
                train_indices.extend(label_indices[num_test:])  # The rest is for training
            else:
                # If test_size is specified, calculate the split point based on the test_size proportion
                split_point = int(len(label_indices) * test_size)
                test_indices.extend(label_indices[:split_point])  # The first part goes into testing
                train_indices.extend(label_indices[split_point:])  # The remaining is for training

        test_indices = np.array(test_indices)
        train_indices = np.array(train_indices)

        # Split the data and labels into training and testing sets based on the calculated indices
        X_train, X_test = data[train_indices], data[test_indices]
        y_train, y_test = np.array(processed_labels)[train_indices], np.array(processed_labels)[test_indices]

        # Return the indices, data, and labels for both training and testing sets
        return train_indices, test_indices, X_train, X_test, y_train, y_test


    def train_and_evaluate_all_mz_per_sample(self, n_splits=50, vintage=False, random_seed=42, test_size=None,
                                             normalize=False, scaler_type='standard', use_pca=False, vthresh=0.97,
                                             region=None, best_alpha=1.0):
        """
        Train and evaluate the classifier using cross-validation, with accuracy metrics for imbalanced classes.

        Parameters
        ----------
        alpha : float
            Regularization strength for RidgeClassifier.
        n_splits : int
            Number of cross-validation splits.
        vintage : bool
            Whether to use vintage data for splitting.
        random_seed : int
            Seed for reproducibility.
        test_size : float or None
            Proportion of the data to include in the test split.
        normalize : bool
            Whether to normalize the data.
        scaler_type : str
            Type of scaler to use for normalization ('standard' or 'minmax').
        use_pca : bool
            Whether to apply PCA for dimensionality reduction.
        vthresh : float
            Variance threshold for PCA.
        region : str or None
            Region-specific configurations for label ordering.

        Returns
        -------
        dict
            A dictionary containing mean accuracy, balanced accuracy, precision, F1-score,
            mean confusion matrix, and the fixed alpha value.
        """
        # Initialize accumulators for metrics
        accuracy_scores = []
        balanced_accuracy_scores = []
        precision_scores = []
        f1_scores = []
        confusion_matrix_sum = None

        # Use a dynamic seed if none is provided
        if random_seed is None:
            random_seed = np.random.randint(0, 1e6)
        rng = np.random.default_rng(random_seed)

        # Predefine splits for consistency
        predefined_splits = []
        for _ in range(n_splits):
            train_idx, temp_idx = train_test_split(
                np.arange(len(self.labels)), test_size=test_size + test_size, stratify=self.labels, random_state=rng.integers(0, 1e6)
            )
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=0.5, stratify=self.labels[temp_idx], random_state=rng.integers(0, 1e6)
            )
            predefined_splits.append((train_idx, val_idx, test_idx))

        # Cross-validation loop
        for i, (train_idx, val_idx, test_idx) in enumerate(predefined_splits):
            X_train = self.data[train_idx]
            X_val = self.data[val_idx]
            X_test = self.data[test_idx]
            y_train, y_val, y_test = self.labels[train_idx], self.labels[val_idx], self.labels[test_idx]

            # Concatenate the m/z profiles for each sample
            X_train = X_train.transpose(0, 2, 1).reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
            X_test = X_test.transpose(0, 2, 1).reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

            # Normalize data if enabled
            if normalize:
                X_train, scaler = normalize_data(X_train, scaler=scaler_type)
                X_test = scaler.transform(X_test)

            # Apply PCA if enabled
            if use_pca:
                reducer = DimensionalityReducer(self.data)
                _, _, n_components = reducer.cumulative_variance(self.labels, variance_threshold=vthresh, plot=False)
                n_components = min(n_components, len(set(y_train)))  # Adjust PCA components based on class count
                pca = PCA(n_components=n_components, svd_solver='randomized')
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)

            # Train RidgeClassifier with the given alpha
            ridge_classifier = RidgeClassifier(alpha=best_alpha)
            ridge_classifier.fit(X_train, y_train)

            # Predictions and metrics
            y_pred = ridge_classifier.predict(X_test)
            accuracy_scores.append(ridge_classifier.score(X_test, y_test))
            balanced_accuracy_scores.append(balanced_accuracy_score(y_test, y_pred))

            # Compute precision and F1 score
            precision_scores.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

            # Compute confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)))
            confusion_matrix_sum = cm if confusion_matrix_sum is None else confusion_matrix_sum + cm

            # print(f"Split {i + 1}/{n_splits} completed.")

        # Calculate mean metrics
        mean_confusion_matrix = confusion_matrix_sum / n_splits

        # Print summary
        print(f"Alpha: {best_alpha:.3f}")
        print(f"Accuracy: {np.mean(accuracy_scores):.3f} (+/- {np.std(accuracy_scores) * 2:.3f})")
        print(f"Balanced Accuracy: {np.mean(balanced_accuracy_scores):.3f} (+/- {np.std(balanced_accuracy_scores) * 2:.3f})")
        print(f"Precision: {np.mean(precision_scores):.3f}")
        print(f"F1 Score: {np.mean(f1_scores):.3f}")
        print("Mean Confusion Matrix:\n", mean_confusion_matrix)

        # Return metrics
        return {
            'mean_accuracy': np.mean(accuracy_scores),
            'mean_balanced_accuracy': np.mean(balanced_accuracy_scores),
            'mean_precision': np.mean(precision_scores),
            'mean_f1_score': np.mean(f1_scores),
            'mean_confusion_matrix': mean_confusion_matrix,
            'alpha': best_alpha
        }


class BayesianParamOptimizer:
    def __init__(self, data, labels, n_channels=None, n_splits=5):
        """
        Initialize the Bayesian Weight Optimizer.

        Parameters
        ----------
        data : np.ndarray
            The input data with shape (samples, features, channels).
        labels : np.ndarray
            The labels for each sample.
        n_channels : int
            Number of m/z channels.
        n_splits : int
            Number of splits for cross-validation.
        """
        self.data = data
        self.labels = labels
        self.n_channels = n_channels
        self.n_splits = n_splits

        # Determine the base number with many multiples

        if self.n_channels is not None:
            self.base_channels = self._determine_base_channels()

    def _determine_base_channels(self):
        """
        Determine a base number close to n_channels that has many multiples.

        Returns
        -------
        int
            A base number with many divisors.
        """
        candidates = [i for i in range(self.n_channels, 1, -1) if 180 % i == 0]
        return candidates[0] if candidates else self.n_channels

    def evaluate_n_and_alpha(self, n, alpha, ch_treat, num_splits=5):
        """
        Evaluate the balanced accuracy for a given number of grouped channels and alpha.

        Parameters
        ----------
        n : int
            Number of channels grouped together.
        alpha : float
            Regularization strength for RidgeClassifier.
        ch_treat : string
            How to use the channels, independently or concatenated

        Returns
        -------
        float
            Negative mean balanced accuracy (to minimize).
        """
        num_groups = self.n_channels // n
        aggregated_data = np.zeros((self.data.shape[0], self.data.shape[1], num_groups))

        # Aggregate channels into groups
        for i in range(num_groups):
            start_idx = i * n
            end_idx = min((i + 1) * n, self.n_channels)
            aggregated_data[:, :, i] = np.mean(self.data[:, :, start_idx:end_idx], axis=-1)

        # # reshaped_data = aggregated_data.transpose(2, 0, 1).reshape(-1, self.data.shape[1])
        # reshaped_data = aggregated_data.transpose(0, 2, 1).reshape(-1, self.data.shape[1])
        # labels_expanded = np.repeat(self.labels, num_groups)

        # Evaluate performance using multiple train-test splits
        balanced_accuracies = []

        model = RidgeClassifier(alpha=alpha)
        for split in range(num_splits):
            # X_train, X_test, y_train, y_test = train_test_split(
            #     reshaped_data, labels_expanded, test_size=0.4, stratify=labels_expanded
            # )
            if ch_treat == 'individual':
                # Stratified split at the sample level
                train_idx, test_idx = train_test_split(
                    np.arange(len(self.labels)),
                    test_size=0.4,
                    stratify=self.labels,
                    random_state=split
                )

                # Prepare train and test data while keeping channels grouped
                X_train = aggregated_data[train_idx].transpose(0, 2, 1).reshape(-1, self.data.shape[1])
                X_test = aggregated_data[test_idx].transpose(0, 2, 1).reshape(-1, self.data.shape[1])
                y_train = np.repeat(np.array(self.labels)[train_idx], num_groups)
                y_test = np.repeat(np.array(self.labels)[test_idx], num_groups)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
                balanced_accuracies.append(balanced_accuracy)

            elif ch_treat == 'concatenated':
                # Concatenate grouped channels for each sample
                concatenated_data = aggregated_data.transpose(0, 2, 1).reshape(self.data.shape[0], -1)

                # Stratified split at the sample level
                train_idx, test_idx = train_test_split(
                    np.arange(len(self.labels)),
                    test_size=0.4,
                    stratify=self.labels,
                    random_state=split
                )

                # Prepare train and test data with concatenated channels
                X_train, X_test = concatenated_data[train_idx], concatenated_data[test_idx]
                y_train, y_test = np.array(self.labels)[train_idx], np.array(self.labels)[test_idx]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
                balanced_accuracies.append(balanced_accuracy)

            else:
                raise ValueError("Invalid value for ch_treat. Use 'individual' or 'concatenated'.")

        return -np.mean(balanced_accuracies)  # Negative score for minimization



    def evaluate_alpha(self, alpha, num_splits=5):
        """
        Evaluate the balanced accuracy for a given alpha.

        Parameters
        ----------
        alpha : float
            Regularization strength for RidgeClassifier.

        Returns
        -------
        float
            Negative mean balanced accuracy (to minimize).
        """
        balanced_accuracies = []

        model = RidgeClassifier(alpha=alpha)
        for _ in range(num_splits):
            X_train, X_test, y_train, y_test = train_test_split(
                self.data, self.labels, test_size=0.4, stratify=self.labels, random_state=_
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
            balanced_accuracies.append(balanced_accuracy)

        return -np.mean(balanced_accuracies)  # Negative score for minimization


    def optimize_gcms(self, n_calls=50, random_state=42, num_splits=5, ch_treat='concatenated'):
        """
        Optimize the number of channels grouped together and alpha using Bayesian Optimization.

        Parameters
        ----------
        n_calls : int
            Number of evaluations of the objective function.
        random_state : int
            Random seed for reproducibility.

        Returns
        -------
        dict
            Results of the optimization process.
        """
        # Generate multiples of base_channels that are valid for grouping
        n_values = [i for i in range(1, self.base_channels + 1) if self.base_channels % i == 0]
        # n_values = [180]

        # Define the search space
        space = [
            Categorical(n_values, name="n"),  # Multiples of the base number
            Real(0.1, 20000.0, name="alpha", prior="log-uniform")  # Alpha range for RidgeClassifier
        ]

        @use_named_args(space)
        def objective(**params):
            n = params["n"]
            alpha = params["alpha"]
            return self.evaluate_n_and_alpha(n, alpha, ch_treat, num_splits=num_splits)

        # Initialize progress bar
        with tqdm(total=n_calls, desc="Optimizing Channels and Alpha") as pbar:
            def progress_callback(res):
                """Update the progress bar after each iteration."""
                pbar.update(1)

            # Perform Bayesian Optimization with a callback for progress
            result = gp_minimize(objective, space, n_calls=n_calls, random_state=random_state,
                                 callback=[progress_callback])

        return result

    def optimize_tic(self, n_calls=50, random_state=42, num_splits=5):
        """
        Optimize the alpha parameter using Bayesian Optimization.

        Parameters
        ----------
        n_calls : int
            Number of evaluations of the objective function.
        random_state : int
            Random seed for reproducibility.

        Returns
        -------
        dict
            Results of the optimization process.
        """
        # Define the search space
        space = [Real(0.1, 20000.0, name="alpha", prior="log-uniform")]

        @use_named_args(space)
        def objective(**params):
            alpha = params["alpha"]
            return self.evaluate_alpha(alpha, num_splits=num_splits)

        # Initialize progress bar
        with tqdm(total=n_calls, desc="Optimizing Alpha") as pbar:
            def progress_callback(res):
                """Update the progress bar after each iteration."""
                pbar.update(1)

            # Perform Bayesian Optimization with a callback for progress
            result = gp_minimize(objective, space, n_calls=n_calls, random_state=random_state,
                                 callback=[progress_callback])

        return result



# class BayesianParamOptimizer:
#     def __init__(self, data, labels, n_channels, n_splits=5):
#         """
#         Initialize the Bayesian Weight Optimizer.
#
#         Parameters
#         ----------
#         data : np.ndarray
#             The input data with shape (samples, features, channels).
#         labels : np.ndarray
#             The labels for each sample.
#         n_channels : int
#             Number of m/z channels.
#         n_splits : int
#             Number of splits for cross-validation.
#         """
#         self.data = data
#         self.labels = labels
#         self.n_channels = n_channels
#         self.n_splits = n_splits
#
#     def evaluate_n(self, n):
#         """
#         Evaluate the balanced accuracy for a given number of aggregated channels.
#
#         Parameters
#         ----------
#         n : int
#             Number of aggregated channels.
#
#         Returns
#         -------
#         float
#             Negative mean balanced accuracy (to minimize).
#         """
#         group_size = self.n_channels // n
#         aggregated_data = np.zeros((self.data.shape[0], self.data.shape[1], n))
#
#         for i in range(n):
#             start_idx = i * group_size
#             end_idx = (i + 1) * group_size if i != n - 1 else self.n_channels
#             aggregated_data[:, :, i] = np.mean(self.data[:, :, start_idx:end_idx], axis=-1)
#
#         reshaped_data = aggregated_data.transpose(2, 0, 1).reshape(-1, self.data.shape[1])
#         labels_expanded = np.repeat(self.labels, n)
#
#         accuracy_scores = []
#         ridge_classifier = RidgeClassifierCV(alphas=self.alpha_range, scoring='balanced_accuracy')
#
#         for _ in range(self.n_splits):
#             X_train, X_test, y_train, y_test = train_test_split(
#                 reshaped_data, labels_expanded, random_state=None
#             )
#             ridge_classifier.fit(X_train, y_train)
#             y_pred = ridge_classifier.predict(X_test)
#             balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
#             accuracy_scores.append(balanced_accuracy)
#
#         return -np.mean(accuracy_scores)
#
#     def evaluate_n_and_alpha(self, n, alpha, num_splits=20):
#         """
#         Evaluate the balanced accuracy for a given number of aggregated channels and alpha.
#
#         Parameters
#         ----------
#         n : int
#             Number of aggregated channels.
#         alpha : float
#             Regularization strength for RidgeClassifier.
#
#         Returns
#         -------
#         float
#             Negative mean balanced accuracy (to minimize).
#         """
#         group_size = self.n_channels // n
#         remainder = self.n_channels % n  # Channels left after even grouping
#         aggregated_data = np.zeros((self.data.shape[0], self.data.shape[1], n))
#
#         for i in range(n):
#             start_idx = i * group_size
#             if i < remainder:
#                 # Distribute one extra channel to the first 'remainder' groups
#                 end_idx = start_idx + group_size + 1
#             else:
#                 # Remaining groups take only 'group_size' channels
#                 end_idx = start_idx + group_size
#
#             aggregated_data[:, :, i] = np.sum(self.data[:, :, start_idx:end_idx], axis=-1)
#
#         reshaped_data = aggregated_data.transpose(2, 0, 1).reshape(-1, self.data.shape[1])
#         labels_expanded = np.repeat(self.labels, n)
#
#         balanced_accuracies = []
#
#         for _ in range(num_splits):
#             # Stratified train-test split
#             X_train, X_test, y_train, y_test = train_test_split(
#                 reshaped_data,
#                 labels_expanded,
#                 test_size=0.2,
#                 stratify=labels_expanded,
#                 random_state=None
#             )
#
#             # Train RidgeClassifier with the given alpha
#             ridge_classifier = RidgeClassifier(alpha=alpha)
#             ridge_classifier.fit(X_train, y_train)
#             y_pred = ridge_classifier.predict(X_test)
#
#             # Compute the balanced accuracy for this split
#             balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
#             balanced_accuracies.append(balanced_accuracy)
#
#         # Return the negative mean balanced accuracy (to minimize)
#         return -np.mean(balanced_accuracies)
#
#     def optimize(self, n_calls=50, random_state=42, num_splits=5):
#         """
#         Optimize the number of channels and alpha using Bayesian Optimization.
#
#         Parameters
#         ----------
#         n_calls : int
#             Number of evaluations of the objective function.
#         random_state : int
#             Random seed for reproducibility.
#
#         Returns
#         -------
#         dict
#             Results of the optimization process.
#         """
#         space = [
#             Integer(1, self.n_channels, name="n"),
#             Real(0.1, 20000.0, name="alpha")
#         ]
#
#         @use_named_args(space)
#         def objective(**params):
#             n = params["n"]
#             alpha = params["alpha"]
#             return self.evaluate_n_and_alpha(n, alpha, num_splits=num_splits)
#
#         # Initialize progress bar
#         with tqdm(total=n_calls, desc="Optimizing Channels and Alpha") as pbar:
#             def progress_callback(res):
#                 """Update the progress bar after each iteration."""
#                 pbar.update(1)
#
#             # Perform Bayesian Optimization with a callback for progress
#             result = gp_minimize(objective, space, n_calls=n_calls, random_state=random_state,
#                                  callback=[progress_callback])
#
#         return result
#
#
#     # def evaluate_weights(self, weights):
#     #     """
#     #     Evaluate the balanced accuracy for given weights.
#     #
#     #     Parameters
#     #     ----------
#     #     weights : list
#     #         Weights for each channel.
#     #
#     #     Returns
#     #     -------
#     #     float
#     #         Negative mean balanced accuracy (to minimize).
#     #     """
#     #     weights = np.array(weights).reshape(1, 1, -1)
#     #     weighted_data = self.data * weights
#     #     reshaped_data = weighted_data.transpose(2, 0, 1).reshape(-1, self.data.shape[1])
#     #     labels_expanded = np.repeat(self.labels, self.n_channels)
#     #
#     #     accuracy_scores = []
#     #     ridge_classifier = RidgeClassifierCV(alphas=self.alpha_range, scoring='balanced_accuracy')
#     #
#     #     for _ in range(self.n_splits):
#     #         X_train, X_test, y_train, y_test = train_test_split(
#     #             reshaped_data, labels_expanded, random_state=None
#     #         )
#     #         ridge_classifier.fit(X_train, y_train)
#     #         y_pred = ridge_classifier.predict(X_test)
#     #         balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
#     #         accuracy_scores.append(balanced_accuracy)
#     #
#     #     # Return negative accuracy (to minimize)
#     #     return -np.mean(accuracy_scores)
#
#     # def optimize(self, n_calls=50, random_state=42):
#     #     """
#     #     Optimize weights using Bayesian Optimization.
#     #
#     #     Parameters
#     #     ----------
#     #     n_calls : int
#     #         Number of evaluations of the objective function.
#     #     random_state : int
#     #         Random seed for reproducibility.
#     #
#     #     Returns
#     #     -------
#     #     dict
#     #         Results of the optimization process.
#     #     """
#     #     # Define the search space for weights (0 to 2 for each channel)
#     #     space = [Real(0, 2, name=f"w{i}") for i in range(self.n_channels)]
#     #
#     #     # Decorate the objective function with the search space
#     #     @use_named_args(space)
#     #     def objective(**weights):
#     #         weight_array = np.array([weights[f"w{i}"] for i in range(self.n_channels)])
#     #         return self.evaluate_weights(weight_array)
#     #
#     #     # Perform Bayesian Optimization
#     #     result = gp_minimize(objective, space, n_calls=n_calls, random_state=random_state)
#     #
#     #     return result



class CoordinateDescentOptimizer:
    def __init__(self, data, labels, alpha_range=None, n_splits=5, max_iter=100, tol=1e-4):
        """
        Initialize the optimizer.

        Parameters
        ----------
        data : np.ndarray
            The input data of shape (samples, features, channels).
        labels : np.ndarray
            The corresponding labels.
        alpha_range : list
            The range of alphas to test with RidgeClassifierCV.
        n_splits : int
            Number of cross-validation splits.
        max_iter : int
            Maximum number of iterations for coordinate descent.
        tol : float
            Convergence tolerance.
        """
        self.data = data
        self.labels = labels
        self.alpha_range = alpha_range if alpha_range is not None else [0.1, 1.0, 10.0, 100.0]
        self.n_splits = n_splits
        self.max_iter = max_iter
        self.tol = tol
        self.weights = np.ones(data.shape[2])  # Initialize weights to 1 (one per channel)

    def evaluate_weight(self, channel_idx):
        """
        Evaluate the balanced accuracy for a specific channel's weight.

        Parameters
        ----------
        channel_idx : int
            The index of the channel to optimize.

        Returns
        -------
        float
            Negative mean balanced accuracy (to minimize).
        """
        # Create a copy of weights, setting all other weights to 1
        channel_weights = np.ones(self.weights.shape)
        channel_weights[channel_idx] = self.weights[channel_idx]

        # Apply weights to the data for the specified channel
        weighted_data = self.data * channel_weights.reshape(1, 1, -1)
        reshaped_data = weighted_data.transpose(2, 0, 1).reshape(-1, self.data.shape[1])
        labels_expanded = np.repeat(self.labels, self.data.shape[2])

        # Ridge Classifier with cross-validation
        ridge_classifier = RidgeClassifierCV(alphas=self.alpha_range, scoring='balanced_accuracy')
        accuracy_scores = []

        for _ in range(self.n_splits):
            X_train, X_test, y_train, y_test = train_test_split(
                reshaped_data, labels_expanded, random_state=None, stratify=labels_expanded
            )

            ridge_classifier.fit(X_train, y_train)
            y_pred = ridge_classifier.predict(X_test)
            balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
            accuracy_scores.append(balanced_accuracy)

        # Return negative accuracy to minimize
        return -np.mean(accuracy_scores)


    def optimize(self):
        """
        Optimize weights using coordinate descent.

        Returns
        -------
        np.ndarray
            The optimized weights.
        """
        for iteration in range(self.max_iter):
            previous_weights = self.weights.copy()

            for channel_idx in range(len(self.weights)):
                # Optimize weight for this channel
                best_weight = self.find_best_weight(channel_idx)
                self.weights[channel_idx] = best_weight

            # Check convergence
            weight_change = np.linalg.norm(self.weights - previous_weights)
            # Calculate accuracy with current weights
            weighted_data = self.data * self.weights.reshape(1, 1, -1)
            reshaped_data = weighted_data.transpose(2, 0, 1).reshape(-1, self.data.shape[1])
            labels_expanded = np.repeat(self.labels, self.data.shape[2])
            accuracy_scores = []

            for _ in range(self.n_splits):
                X_train, X_test, y_train, y_test = train_test_split(reshaped_data, labels_expanded, random_state=None)
                ridge_classifier = RidgeClassifier()
                ridge_classifier.fit(X_train, y_train)
                y_pred = ridge_classifier.predict(X_test)
                accuracy_scores.append(balanced_accuracy_score(y_test, y_pred))

            # Calculate accuracy with current weights
            weighted_data = self.data * self.weights.reshape(1, 1, -1)
            reshaped_data = weighted_data.transpose(2, 0, 1).reshape(-1, self.data.shape[1])
            labels_expanded = np.repeat(self.labels, self.data.shape[2])
            accuracy_scores = []

            for _ in range(self.n_splits):
                X_train, X_test, y_train, y_test = train_test_split(reshaped_data, labels_expanded, random_state=None)
                ridge_classifier = RidgeClassifier()
                ridge_classifier.fit(X_train, y_train)
                y_pred = ridge_classifier.predict(X_test)
                accuracy_scores.append(balanced_accuracy_score(y_test, y_pred))

            mean_accuracy = np.mean(accuracy_scores)

            print(f"Iteration {iteration + 1}, Weight Change: {weight_change:.6f}, Weights: {self.weights}, "
                  f"Mean Accuracy: {mean_accuracy:.4f}")

            if weight_change < self.tol:
                print("Converged!")
                break

        return self.weights

    def find_best_weight(self, channel_idx):
        """
        Perform a line search to find the best weight for a given channel.

        Parameters
        ----------
        channel_idx : int
            The index of the channel to optimize.

        Returns
        -------
        float
            The best weight for the given channel.
        """

        best_weight = self.weights[channel_idx]
        best_score = self.evaluate_weight(channel_idx)

        # Search over a range of weights (e.g., [-5, 5])
        for candidate_weight in np.linspace(0, 1, 20):
            self.weights[channel_idx] = candidate_weight
            score = self.evaluate_weight(channel_idx)

            if score < best_score:  # Minimize negative balanced accuracy
                best_weight = candidate_weight
                best_score = score

        return best_weight



def process_labels(labels, vintage):
    """
    Process a list of labels to extract relevant parts based on whether the data is vintage or not.

    Parameters
    ----------
    labels : list of str
        A list of label strings to be processed.
    vintage : bool
        If True, the function processes labels to extract a substring starting from the first digit
        found in each label (assuming vintage data formatting). If False, it processes labels to
        extract a single character or digit before the first digit found.

    Returns
    -------
    numpy.ndarray
        An array of processed labels.

    Notes
    -----
    This function is similar to the `_process_labels` method within the `Classifier` class, but
    it operates on an external list of labels rather than an instance attribute.
    """
    processed_labels = []

    # Iterate over each label in the provided list of labels
    for label in labels:
        # Search for the first digit in the label
        match = re.search(r'\d+', label)

        if vintage:
            # If processing vintage data, extract the substring starting from the first digit
            processed_labels.append(label[match.start():])
        else:
            # If not vintage, extract the character before the first digit
            if label[match.start() - 1] == '_':
                # If the character before the digit is an underscore, take the character before the underscore
                lb = label[match.start() - 2]
            else:
                # Otherwise, take the character directly before the first digit
                lb = label[match.start() - 1]
            processed_labels.append(lb)

    # Return the processed labels as a numpy array
    return np.array(processed_labels)


def assign_country_to_pinot_noir(original_keys):
    """
        Map wine sample keys to their corresponding country .

        This function takes a list of wine sample keys, where the first letter of each key represents
        the Chateau and returns a list of corresponding countries (Switzerland, US, or France).

        Parameters
        ----------
        original_keys : list of str
            A list of strings where each string is a wine sample key. The first letter of each key
            corresponds to a Chateau(e.g., 'C14', 'M08').

        Returns
        -------
        origine_keys : list of str
            A list of strings where each string is the corresponding country ('Switzerland',
            'US', 'France') of the wine sample based on the first letter of the key.

        Examples
        --------
        >>> original_keys = ['C14', 'M08', 'U08', 'D10', 'X13']
        >>> assign_country_to_pinot_noir(original_keys)
        ['France', 'Switzerland', 'US', 'France', 'US']

        Notes
        -----
        The first letter of the key is used to determine the country:
            - 'M', 'N', 'J', 'L', 'H' => Switzerland
            - 'U', 'X' => US
            - 'D', 'E', 'Q', 'P', 'R', 'Z', 'C', 'K', 'W', 'Y' => France
        """
    # Dictionary to map letters to their origins
    letter_to_country = {
        # Switzerland
        'M': 'Switzerland',
        'N': 'Switzerland',
        'J': 'Switzerland',
        'L': 'Switzerland',
        'H': 'Switzerland',

        # US
        'U': 'US',
        'X': 'US',

        # France
        'D': 'France',
        'E': 'France',
        'Q': 'France',
        'P': 'France',
        'R': 'France',
        'Z': 'France',
        'C': 'France',
        'K': 'France',
        'W': 'France',
        'Y': 'France'
    }

    # Create a new list by mapping the first letter of each key to its "Origine"
    country_keys = [letter_to_country[key[0]] for key in original_keys]

    return country_keys


def assign_origin_to_pinot_noir(original_keys):
    """
    Map wine sample keys to their corresponding region of origin (Origine).

    This function takes a list of wine sample keys, where the first letter of each key represents
    a region of origin, and returns a list of corresponding regions ("Origine") for each key.

    Parameters
    ----------
    original_keys : list of str
        A list of strings where each string is a wine sample key. The first letter of each key
        corresponds to a specific region of origin (e.g., 'C14', 'M08').

    Returns
    -------
    origine_keys : list of str
        A list of strings where each string is the corresponding region of origin based on the
        first letter of the key.

    Examples
    --------
    >>> original_keys = ['C14', 'M08', 'U08', 'D10', 'X13']
    >>> assign_origin_to_pinot_noir(original_keys)
    ['Alsace', 'Neuchatel', 'Californie', 'Beaune', 'Oregon']

    Notes
    -----
    The first letter of the key is used to determine the specific region of origin:
        - 'M', 'N' => Neuchatel (Switzerland)
        - 'J', 'L' => Genève (Switzerland)
        - 'H' => Valais (Switzerland)
        - 'U' => Californie (US)
        - 'X' => Oregon (US)
        - 'D', 'E', 'Q', 'P', 'R', 'Z' => Beaune (France)
        - 'C', 'K', 'W', 'Y' => Alsace (France)
    """
    # Dictionary to map letters to their specific regions (Origine)
    letter_to_origine = {
        # Switzerland
        'M': 'Neuchatel',
        'N': 'Neuchatel',
        'J': 'Genève',
        'L': 'Genève',
        'H': 'Valais',

        # US
        'U': 'Californie',
        'X': 'Oregon',

        # France
        'D': 'Beaune',
        'E': 'Beaune',
        'Q': 'Beaune',
        'P': 'Beaune',
        'R': 'Beaune',
        'Z': 'Beaune',
        'C': 'Alsace',
        'K': 'Alsace',
        'W': 'Alsace',
        'Y': 'Alsace'
    }

    # Create a new list by mapping the first letter of each key to its specific "Origine"
    origin_keys = [letter_to_origine[key[0]] for key in original_keys]

    return origin_keys


def assign_continent_to_pinot_noir(original_keys):
    """
    Map wine sample keys to their corresponding continent.

    This function takes a list of wine sample keys, where the first letter of each key represents
    a region of origin, and returns a list of corresponding continents for each key.

    Parameters
    ----------
    original_keys : list of str
        A list of strings where each string is a wine sample key. The first letter of each key
        corresponds to a specific region of origin.

    Returns
    -------
    continent_keys : list of str
        A list of strings where each string is the corresponding continent based on the
        first letter of the key.

    Examples
    --------
    >>> original_keys = ['C14', 'M08', 'U08', 'D10', 'X13']
    >>> assign_continent_to_pinot_noir(original_keys)
    ['Europe', 'Europe', 'North America', 'Europe', 'North America']

    Notes
    -----
    The first letter of the key is used to determine the continent:
        - 'M', 'N', 'J', 'L', 'H' => Europe (Switzerland)
        - 'U', 'X' => North America (US)
        - 'D', 'E', 'Q', 'P', 'R', 'Z', 'C', 'K', 'W', 'Y' => Europe (France)
    """
    # Dictionary to map letters to their continents
    letter_to_continent = {
        # Switzerland (Europe)
        'M': 'Europe',
        'N': 'Europe',
        'J': 'Europe',
        'L': 'Europe',
        'H': 'Europe',

        # US (North America)
        'U': 'North America',
        'X': 'North America',

        # France (Europe)
        'D': 'Europe',
        'E': 'Europe',
        'Q': 'Europe',
        'P': 'Europe',
        'R': 'Europe',
        'Z': 'Europe',
        'C': 'Europe',
        'K': 'Europe',
        'W': 'Europe',
        'Y': 'Europe'
    }

    # Create a new list by mapping the first letter of each key to its continent
    continent_keys = [letter_to_continent[key[0]] for key in original_keys]

    return continent_keys


def assign_north_south_to_beaune(original_keys):
    """
    Map wine sample keys to either 'North Beaune (NB)' or 'South Beaune (SB)'.

    This function takes a list of wine sample keys, where the first letter of each key represents
    a region of origin, and returns a list of corresponding regions ('North Beaune' or 'South Beaune') for each key.

    Parameters
    ----------
    original_keys : list of str
        A list of strings where each string is a wine sample key. The first letter of each key
        corresponds to a specific region of origin.

    Returns
    -------
    beaune_region_keys : list of str
        A list of strings where each string is either 'North Beaune' or 'South Beaune' based on the
        first letter of the key.

    """
    if len(original_keys) != 61:
        raise ValueError(f"Incorrect wines passed. Input should be Beaume wines only")

    # Dictionary to map letters to North or South Beaune
    letter_to_beaune_region = {
        # North Beaune (NB) or Côte de Nuits
        'Q': 'NB',
        'R': 'NB',
        'Z': 'NB',

        # South Beaune (SB) or Côte de Beaune
        'D': 'SB',
        'E': 'SB',
        'P': 'SB',
    }

    # Create a new list by mapping the first letter of each key to North or South Beaune
    beaune_region_keys = [letter_to_beaune_region[key[0]] for key in original_keys]

    return beaune_region_keys


def assign_winery_to_pinot_noir(labels):
    """
    Assign the first letter of each label, which corresponds to the winery (Chateau)

    Parameters
    ----------
    labels : list of str
        A list of label strings.

    Returns
    -------
    first_letters : list of str
        A list of the first letters of each label.
    """
    # Create a list of the first letters of each label
    first_letters = [label[0] for label in labels]

    return first_letters


def assign_year_to_pinot_noir(labels):
    """
    Assign the last two letters  of each label, which corresponds to the year.

    Parameters
    ----------
    labels : list of str
        A list of label strings.

    Returns
    -------
    year : list of str
        A list of the years from each label.
    """
    # Create a list of the first letters of each label
    first_letters = [label[-2:] for label in labels]

    return first_letters


# def train_and_evaluate_all_mz_per_sample(self, n_splits=50, vintage=False, random_seed=42, test_size=None,
#                                          normalize=False, scaler_type='standard', use_pca=False, vthresh=0.97,
#                                          region=None, best_alpha=1.0):
#     """
#     Train and evaluate the classifier using cross-validation, with accuracy metrics for imbalanced classes.
#
#     Parameters
#     ----------
#     alpha : float
#         Regularization strength for RidgeClassifier.
#     n_splits : int
#         Number of cross-validation splits.
#     vintage : bool
#         Whether to use vintage data for splitting.
#     random_seed : int
#         Seed for reproducibility.
#     test_size : float or None
#         Proportion of the data to include in the test split.
#     normalize : bool
#         Whether to normalize the data.
#     scaler_type : str
#         Type of scaler to use for normalization ('standard' or 'minmax').
#     use_pca : bool
#         Whether to apply PCA for dimensionality reduction.
#     vthresh : float
#         Variance threshold for PCA.
#     region : str or None
#         Region-specific configurations for label ordering.
#
#     Returns
#     -------
#     dict
#         A dictionary containing mean accuracy, balanced accuracy, precision, F1-score,
#         mean confusion matrix, and the fixed alpha value.
#     """
#     # Initialize accumulators for metrics
#     accuracy_scores = []
#     balanced_accuracy_scores = []
#     precision_scores = []
#     f1_scores = []
#     confusion_matrix_sum = None
#
#     # Cross-validation loop
#     for i in range(n_splits):
#         # Split data into train and test sets
#         # train_indices, test_indices, X_train, X_test, y_train, y_test = self.split_data(
#         #     self.labels, self.data, vintage=vintage, test_size=test_size, num_test=1)
#
#         # X_train, X_test, y_train, y_test = train_test_split(
#         #     self.data, self.labels, test_size=test_size, stratify=self.labels
#         # )
#         X_train, X_val, X_test, y_train, y_val, y_test = utils.split_train_val_test(
#             self.data, self.labels, test_size=test_size, validation_size=test_size)
#
#
#         # Concatenate the m/z profiles for each sample
#         X_train = X_train.transpose(0, 2, 1).reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
#         X_test = X_test.transpose(0, 2, 1).reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
#
#         # Normalize data if enabled
#         if normalize:
#             X_train, scaler = normalize_data(X_train, scaler=scaler_type)
#             X_test = scaler.transform(X_test)
#
#         # Apply PCA if enabled
#         if use_pca:
#             reducer = DimensionalityReducer(self.data)
#             _, _, n_components = reducer.cumulative_variance(self.labels, variance_threshold=vthresh, plot=False)
#             n_components = min(n_components, len(set(y_train)))  # Adjust PCA components based on class count
#             pca = PCA(n_components=n_components, svd_solver='randomized')
#             X_train = pca.fit_transform(X_train)
#             X_test = pca.transform(X_test)
#
#         # Train RidgeClassifier with the given alpha
#         ridge_classifier = RidgeClassifier(alpha=best_alpha)
#         ridge_classifier.fit(X_train, y_train)
#         # utils.plot_aggregated_weights(ridge_classifier.coef_[0], bin_size=10)
#         # utils.compute_channel_correlation_single_sample(self.data, 0, figsize=(8, 6), cmap="coolwarm")
#
#         # Predictions and metrics
#         y_pred = ridge_classifier.predict(X_test)
#         accuracy_scores.append(ridge_classifier.score(X_test, y_test))
#         balanced_accuracy_scores.append(balanced_accuracy_score(y_test, y_pred))
#
#         # Compute precision and F1 score
#         precision_scores.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
#         f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
#
#         # Compute confusion matrix
#         cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)))
#         confusion_matrix_sum = cm if confusion_matrix_sum is None else confusion_matrix_sum + cm
#
#         print(f"Split {i + 1}/{n_splits} completed.")
#
#     # Calculate mean metrics
#     mean_confusion_matrix = confusion_matrix_sum / n_splits
#
#     # Print summary
#     print(f"Alpha: {best_alpha:.3f}")
#     print(f"Accuracy: {np.mean(accuracy_scores):.3f} (+/- {np.std(accuracy_scores) * 2:.3f})")
#     print(
#         f"Balanced Accuracy: {np.mean(balanced_accuracy_scores):.3f} (+/- {np.std(balanced_accuracy_scores) * 2:.3f})")
#     print(f"Precision: {np.mean(precision_scores):.3f}")
#     print(f"F1 Score: {np.mean(f1_scores):.3f}")
#     print("Mean Confusion Matrix:\n", mean_confusion_matrix)
#
#     # Return metrics
#     return {
#         'mean_accuracy': np.mean(accuracy_scores),
#         'mean_balanced_accuracy': np.mean(balanced_accuracy_scores),
#         'mean_precision': np.mean(precision_scores),
#         'mean_f1_score': np.mean(f1_scores),
#         'mean_confusion_matrix': mean_confusion_matrix,
#         'alpha': best_alpha
#     }






# def greedy_channel_selection(
#         data, labels, alpha=1.0, test_size=0.2, max_channels=None, num_splits=5, tolerated_no_improvement_steps=2,
#         normalize=True, scaler_type='standard', random_seed=None, parallel=True, n_jobs=-1):
#     """
#     Greedy Forward Selection with multiple train-validation-test splits to find the best m/z channels for classification.
#     Allows toggling between parallel and non-parallel execution. Always adds the best channel but stops after a configurable
#     number of consecutive non-improving steps.
#
#     Parameters
#     ----------
#     data : np.ndarray
#         Input 3D array with shape (samples, features, channels).
#     labels : np.ndarray
#         Array of labels for the samples.
#     alpha : float
#         Regularization strength for RidgeClassifier.
#     test_size : float
#         Proportion of data to be used for testing.
#     max_channels : int or None
#         Maximum number of channels to include. If None, selects all channels.
#     num_splits : int
#         Number of train-validation-test splits to average the performance.
#     tolerated_no_improvement_steps : int
#         Number of consecutive steps without accuracy improvement before stopping.
#     random_seed : int
#         Random seed for reproducibility.
#     parallel : bool
#         If True, runs the channel evaluation in parallel. If False, runs sequentially.
#     n_jobs : int
#         Number of CPU cores to use in parallel mode. Use -1 to utilize all available cores.
#
#     Returns
#     -------
#     list
#         Ordered list of selected channels.
#     list
#         List of accuracies at each step on the test data.
#     """
#     num_channels = data.shape[2]
#     max_channels = max_channels or num_channels
#
#     selected_channels = []  # List to store selected channel indices
#     remaining_channels = list(range(num_channels))  # Channels not yet selected
#     accuracies_on_test = []  # Track performance on test data after each step
#
#     print("Starting Greedy Channel Selection with Averaged Splits...")
#
#     model = RidgeClassifier(alpha=alpha)
#     best_accuracy_overall = 0.0  # Track the best accuracy so far
#     non_improving_steps = 0  # Counter for consecutive non-improving steps
#
#     # Use a dynamic seed if none is provided
#     if random_seed is None:
#         random_seed = np.random.randint(0, 1e6)
#
#     rng = np.random.default_rng(random_seed)
#
#     # Predefine splits for consistency
#     predefined_splits = []
#     for _ in range(num_splits):
#         train_idx, temp_idx = train_test_split(
#             np.arange(len(labels)), test_size=test_size + test_size, stratify=labels, random_state=rng.integers(0, 1e6)
#         )
#         val_idx, test_idx = train_test_split(
#             temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=rng.integers(0, 1e6)
#         )
#         predefined_splits.append((train_idx, val_idx, test_idx))
#
#     def evaluate_channel(ch):
#         candidate_channels = selected_channels + [ch]
#
#         split_accuracies_on_val = []
#         split_accuracies_on_test = []
#
#         for train_idx, val_idx, test_idx in predefined_splits:
#             # Extract data subsets using the indices
#             X_train = data[train_idx][:, :, candidate_channels].reshape(len(train_idx), -1)
#             X_val = data[val_idx][:, :, candidate_channels].reshape(len(val_idx), -1)
#             X_test = data[test_idx][:, :, candidate_channels].reshape(len(test_idx), -1)
#             y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]
#
#             # Normalize data if enabled
#             if normalize:
#                 X_train, scaler = normalize_data(X_train, scaler=scaler_type)
#                 X_test = scaler.transform(X_test)
#
#             # Train model on the current set of selected channels
#             model.fit(X_train, y_train)
#
#             # Evaluate performance on the validation data
#             y_pred_val = model.predict(X_val)
#             accuracy_val = balanced_accuracy_score(y_val, y_pred_val)
#             split_accuracies_on_val.append(accuracy_val)
#
#             # Evaluate performance on the test data
#             y_pred_test = model.predict(X_test)
#             accuracy_test = balanced_accuracy_score(y_test, y_pred_test)
#             split_accuracies_on_test.append(accuracy_test)
#
#         # Average accuracy across splits
#         avg_accuracy_on_val = np.mean(split_accuracies_on_val)
#         avg_accuracy_on_test = np.mean(split_accuracies_on_test)
#
#         return ch, avg_accuracy_on_val, avg_accuracy_on_test
#
#     for step in range(min(max_channels, num_channels)):
#         if parallel:
#             # Parallelized evaluation
#             results = Parallel(n_jobs=n_jobs)(
#                 delayed(evaluate_channel)(ch) for ch in remaining_channels
#             )
#         else:
#             # Sequential evaluation
#             results = [evaluate_channel(ch) for ch in remaining_channels]
#
#         # Find the best channel based on validation accuracy
#         best_channel, best_step_accuracy_val, best_step_accuracy_test = max(results, key=lambda x: x[1])
#
#         # Add the best channel (even if it does not improve test accuracy)
#         if best_channel is not None:
#             selected_channels.append(best_channel)
#             remaining_channels.remove(best_channel)
#             accuracies_on_test.append(best_step_accuracy_test)
#
#             print(f"Step {step + 1}: Added Channel {best_channel} - Validation Accuracy: {best_step_accuracy_val:.4f}, Test Accuracy: {best_step_accuracy_test:.4f}")
#
#             # Check if accuracy improved on the validation data
#             if best_step_accuracy_test > best_accuracy_overall:
#                 best_accuracy_overall = best_step_accuracy_test
#                 non_improving_steps = 0  # Reset counter
#             else:
#                 non_improving_steps += 1
#
#         # Stop if the tolerated number of non-improving steps is reached
#         if non_improving_steps >= tolerated_no_improvement_steps:
#             print(f"No improvement in the last {tolerated_no_improvement_steps} steps. Stopping selection.")
#             break
#
#     print("Greedy Channel Selection Completed.")
#
#     # Plot the accuracy on the test data after each channel is added
#     plt.plot(range(1, len(accuracies_on_test) + 1), accuracies_on_test, marker='o')
#     plt.xlabel('Number of Channels')
#     plt.ylabel('Test Accuracy')
#     plt.title('Test Accuracy During Channel Selection')
#     plt.grid(True)
#     plt.show()
#
#     return selected_channels, accuracies_on_test


def greedy_channel_selection(
        data, labels, alpha=1.0, test_size=0.2, max_channels=None, num_splits=5, tolerated_no_improvement_steps=2,
        normalize=True, scaler_type='standard', random_seed=None, parallel=True, n_jobs=-1, corr_threshold=0.85):
    """
    Greedy Forward Selection with correlation-based filtering to remove highly correlated channels.
    """
    num_channels = data.shape[2]
    max_channels = max_channels or num_channels

    selected_channels = []
    remaining_channels = list(range(num_channels))
    accuracies_on_test = []

    print("Starting Greedy Channel Selection with Correlation Filtering...")

    model = RidgeClassifier(alpha=alpha)
    best_accuracy_overall = 0.0
    non_improving_steps = 0

    if random_seed is None:
        random_seed = np.random.randint(0, 1e6)

    rng = np.random.default_rng(random_seed)

    predefined_splits = []
    for _ in range(num_splits):
        train_idx, temp_idx = train_test_split(
            np.arange(len(labels)), test_size=test_size + test_size, stratify=labels, random_state=rng.integers(0, 1e6)
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=rng.integers(0, 1e6)
        )
        predefined_splits.append((train_idx, val_idx, test_idx))

    def evaluate_channel(ch):
        candidate_channels = selected_channels + [ch]
        split_accuracies_on_val = []
        split_accuracies_on_test = []

        for train_idx, val_idx, test_idx in predefined_splits:
            X_train = data[train_idx][:, :, candidate_channels].reshape(len(train_idx), -1)
            X_val = data[val_idx][:, :, candidate_channels].reshape(len(val_idx), -1)
            X_test = data[test_idx][:, :, candidate_channels].reshape(len(test_idx), -1)
            y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]

            if normalize:
                X_train, scaler = normalize_data(X_train, scaler=scaler_type)
                X_val = scaler.transform(X_val)
                X_test = scaler.transform(X_test)

            model.fit(X_train, y_train)
            y_pred_val = model.predict(X_val)
            accuracy_val = balanced_accuracy_score(y_val, y_pred_val)
            split_accuracies_on_val.append(accuracy_val)

            y_pred_test = model.predict(X_test)
            accuracy_test = balanced_accuracy_score(y_test, y_pred_test)
            split_accuracies_on_test.append(accuracy_test)

        avg_accuracy_on_val = np.mean(split_accuracies_on_val)
        avg_accuracy_on_test = np.mean(split_accuracies_on_test)
        return ch, avg_accuracy_on_val, avg_accuracy_on_test

    # def remove_correlated_channels(new_channel):
    #     new_channel_data = data[:, :, new_channel].flatten()
    #     to_remove = []
    #     for ch in remaining_channels:
    #         if ch == new_channel:
    #             continue
    #         ch_data = data[:, :, ch].flatten()
    #         corr = np.corrcoef(new_channel_data, ch_data)[0, 1]
    #         if abs(corr) > corr_threshold:
    #             to_remove.append(ch)
    #     return to_remove

    def remove_correlated_channels(new_channel):
        """
        Optimized function to remove highly correlated channels using NumPy.
        Computes the correlation matrix once for all channels.
        """
        # Reshape data for correlation computation
        reshaped_data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])
        correlation_matrix = np.corrcoef(reshaped_data, rowvar=False)

        # Find highly correlated channels to the newly added one
        correlated_indices = np.where(np.abs(correlation_matrix[new_channel]) > corr_threshold)[0]

        # Remove the selected channel itself from the correlated list
        correlated_indices = correlated_indices[correlated_indices != new_channel]

        return list(correlated_indices)


    for step in range(min(max_channels, num_channels)):
        if not remaining_channels:
            print("No channels left to evaluate.")
            break
        if parallel:
            results = Parallel(n_jobs=n_jobs)(delayed(evaluate_channel)(ch) for ch in remaining_channels)
        else:
            results = [evaluate_channel(ch) for ch in remaining_channels]

        if results:
            best_channel, best_step_accuracy_val, best_step_accuracy_test = max(results, key=lambda x: x[1])
        else:
            print("No valid channels found for evaluation. Stopping early.")
            break

        best_channel, best_step_accuracy_val, best_step_accuracy_test = max(results, key=lambda x: x[1])

        if best_channel is not None:
            selected_channels.append(best_channel)
            remaining_channels.remove(best_channel)
            accuracies_on_test.append(best_step_accuracy_test)

            print(
                f"Step {step + 1}: Added Channel {best_channel} - Validation Accuracy: {best_step_accuracy_val:.4f}, Test Accuracy: {best_step_accuracy_test:.4f}")

            if best_step_accuracy_test > best_accuracy_overall:
                best_accuracy_overall = best_step_accuracy_test
                non_improving_steps = 0
            else:
                non_improving_steps += 1

            # if corr_threshold >= 1:
            #     correlated_channels = []
            # else:
            #     correlated_channels = remove_correlated_channels(best_channel)
            # for ch in correlated_channels:
            #     remaining_channels.remove(ch)
            #     print(f"Removed correlated channel {ch} (correlation > {corr_threshold})")

            if corr_threshold < 1:
                correlated_channels = remove_correlated_channels(best_channel)
                for ch in correlated_channels:
                    if ch in remaining_channels:
                        remaining_channels.remove(ch)
                        print(f"Removed correlated channel {ch} (correlation > {corr_threshold})")

        if non_improving_steps >= tolerated_no_improvement_steps:
            print(f"No improvement in the last {tolerated_no_improvement_steps} steps. Stopping selection.")
            break

    print("Greedy Channel Selection Completed.")

    # plt.plot(range(1, len(accuracies_on_test) + 1), accuracies_on_test, marker='o')
    # plt.xlabel('Number of Channels')
    # plt.ylabel('Test Accuracy')
    # plt.title('Test Accuracy During Channel Selection')
    # plt.grid(True)
    # plt.show()

    return selected_channels, accuracies_on_test


def classify_all_channels(data, labels, alpha=1.0, test_size=0.2, num_splits=5, use_pca=False, n_components=None):
    """
    Perform classification on all concatenated m/z channels with optional PCA for dimensionality reduction.

    Parameters
    ----------
    data : np.ndarray
        Input 3D array with shape (samples, features, channels).
    labels : np.ndarray
        Array of labels for the samples.
    alpha : float
        Regularization strength for RidgeClassifier.
    test_size : float
        Proportion of data to use for testing.
    num_splits : int
        Number of train-test splits for averaging accuracy.
    use_pca : bool
        Whether to apply PCA for dimensionality reduction.
    n_components : int or None
        Number of PCA components to retain. If None, PCA will not be applied.

    Returns
    -------
    dict
        A dictionary containing mean accuracy, balanced accuracy, and explained variance (if PCA is used).
    """
    # Flatten channels into a single feature vector per sample
    flattened_data = data.transpose(0, 2, 1).reshape(data.shape[0], -1)

    # Initialize metrics
    accuracies = []
    balanced_accuracies = []
    explained_variance_ratios = [] if use_pca else None

    print("Starting classification with all concatenated channels...")
    model = RidgeClassifier(alpha=alpha)

    # Cross-validation loop
    for split in range(num_splits):
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            flattened_data, labels, test_size=test_size, stratify=labels, random_state=split
        )

        # Apply PCA if enabled
        if use_pca:
            pca = PCA(n_components=n_components)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

            # Track explained variance
            explained_variance_ratios.append(np.sum(pca.explained_variance_ratio_))

        # Train and evaluate the Ridge Classifier
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Compute metrics
        accuracies.append(model.score(X_test, y_test))
        balanced_accuracies.append(balanced_accuracy_score(y_test, y_pred))

        print(f"Split {split + 1}/{num_splits} completed.")

    # Calculate mean metrics
    mean_accuracy = np.mean(accuracies)
    mean_balanced_accuracy = np.mean(balanced_accuracies)
    mean_explained_variance = np.mean(explained_variance_ratios) if use_pca else None

    # Print summary
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Mean Balanced Accuracy: {mean_balanced_accuracy:.4f}")
    if use_pca:
        print(f"Mean Explained Variance (PCA): {mean_explained_variance:.4f}")

    # Return metrics
    results = {
        "mean_accuracy": mean_accuracy,
        "mean_balanced_accuracy": mean_balanced_accuracy,
    }
    if use_pca:
        results["mean_explained_variance"] = mean_explained_variance

    return results


def remove_highly_correlated_channels(data, correlation_threshold=0.9):
    """
    Removes highly correlated channels from the data based on the correlation of their full feature profiles.

    Parameters
    ----------
    data : np.ndarray
        Input 3D array with shape (samples, features, channels).
    correlation_threshold : float
        Threshold for considering two channels as highly correlated.

    Returns
    -------
    np.ndarray
        Reduced data with highly correlated channels removed.
    list
        Indices of retained channels.
    """
    # Reshape data to merge samples and features into one dimension for correlation
    reshaped_data = data.transpose(2, 0, 1).reshape(data.shape[2], -1)  # Shape: (channels, samples * features)

    # Compute the correlation matrix across channels
    correlation_matrix = np.corrcoef(reshaped_data)

    # Keep track of selected channels
    num_channels = correlation_matrix.shape[0]
    selected_channels = []

    # Track channels to exclude
    excluded_channels = set()

    for i in range(num_channels):
        if i not in excluded_channels:
            selected_channels.append(i)
            # Mark all highly correlated channels for exclusion
            for j in range(i + 1, num_channels):
                if abs(correlation_matrix[i, j]) > correlation_threshold:
                    excluded_channels.add(j)

    # Select only the retained channels
    reduced_data = data[:, :, selected_channels]

    return reduced_data, selected_channels