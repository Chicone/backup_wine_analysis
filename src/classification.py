from pynndescent.optimal_transport import total_cost
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, PassiveAggressiveClassifier, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier

import utils
from dimensionality_reduction import DimensionalityReducer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight


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
    def __init__(self, data, labels, classifier_type='LDA', wine_kind='bordeaux', cnn_dim=1, multichannel=True,
                 window_size=5000, stride=2500, nconv=3):
        self.data = data
        self.labels = labels
        self.multichannel = multichannel
        self.nconv = nconv
        self.window_size = window_size
        self.stride = stride
        self.classifier = self._get_classifier(classifier_type, multichannel=self.multichannel)
        self.wine_kind = wine_kind
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cnn_dim = cnn_dim


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
            return RidgeClassifier()
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
                                    batch_size=32, num_epochs=10, learning_rate=0.001,
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
            train_indices, test_indices, X_train, X_test, y_train, y_test = self.split_data(vintage=vintage,
                                                                                            test_size=test_size)

            # Normalize data if enabled
            if normalize:
                X_train, scaler = normalize_data(X_train, scaler=scaler_type)
                X_test = scaler.transform(X_test)

            # Apply PCA if enabled
            if use_pca:
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)

            if self.cnn_dim:
                # Transform input data for CNN
                if self.cnn_dim == 2:

                    def preprocess_images_fast(train_tensor):
                        """
                        Preprocess images in a tensor by resizing, normalizing, and optionally converting to 3 channels.

                        Parameters:
                        -----------
                        train_tensor : torch.Tensor
                            Tensor of shape (num_samples, channels, height, width).
                        target_size : tuple
                            Target size (height, width) for resizing.

                        Returns:
                        --------
                        torch.Tensor
                            Preprocessed tensor of shape (num_samples, channels, target_height, target_width).
                        """
                        # Resize using interpolation (fast and supports batch processing)
                        # train_tensor = F.interpolate(train_tensor, size=target_size, mode='bilinear', align_corners=False)

                        # Normalize the tensor (e.g., mean=[0.5], std=[0.5])
                        mean = train_tensor.mean(dim=(0, 2, 3), keepdim=True).to(train_tensor.device)  # Mean across batch, height, and width
                        std = train_tensor.std(dim=(0, 2, 3), keepdim=True).to(train_tensor.device)
                        std = std + 1e-6
                        train_tensor = (train_tensor - mean) / std

                        return train_tensor

                    # Convert labels to integers
                    label_to_index = {label: idx for idx, label in enumerate(set(y_train))}
                    integer_labels = [label_to_index[label] for label in y_train]
                    integer_labels = np.repeat(integer_labels, X_train.shape[1])
                    y_train = torch.tensor(integer_labels, dtype=torch.long).to(self.device)

                    # Flatten the first two dimensions
                    X_train = X_train.reshape(-1, X_train.shape[2], X_train.shape[3])
                    X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device).unsqueeze(1)
                    # X_train = preprocess_images_fast(X_train)

                    integer_labels = [label_to_index[label] for label in y_test]
                    integer_labels = np.repeat(integer_labels, X_test.shape[1])
                    y_test = torch.tensor(integer_labels, dtype=torch.long).to(self.device)

                    X_test = X_test.reshape(-1, X_test.shape[2], X_test.shape[3])
                    X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device).unsqueeze(1)
                    # X_test = preprocess_images_fast(X_test)

                    self.classifier.fit(X_train, y_train, X_test, y_test,
                                        batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate)
                elif self.cnn_dim == 1:

                    self.classifier = self._initialize_cnn1d() # Reinitialize the classifier

                    # Convert labels to integers
                    label_to_index = {label: idx for idx, label in enumerate(set(y_train))}
                    integer_labels = [label_to_index[label] for label in y_train]
                    if not self.multichannel: # if one channel at a time, repeat the labels to the number of m/z channels
                        integer_labels = np.repeat(integer_labels, X_train.shape[1])
                    y_train = torch.tensor(integer_labels, dtype=torch.long).to(self.device)

                    # Reshape
                    if not self.multichannel:
                        X_train = torch.tensor(X_train.reshape(len(y_train), X_train.shape[2]), dtype=torch.float32).to(self.device)
                    else:
                        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
                        X_train, num_overlaps = utils.split_tensor_into_overlapping_windows(X_train, self.window_size, self.stride)
                        y_train = torch.repeat_interleave(y_train, repeats=num_overlaps)
                        print(f'Input samples = {len(y_train)}')

                    integer_labels = [label_to_index[label] for label in y_test]
                    if not self.multichannel:
                        integer_labels = np.repeat(integer_labels, X_test.shape[1])
                    y_test = torch.tensor(integer_labels, dtype=torch.long).to(self.device)

                    if not self.multichannel:
                        X_test = torch.tensor(X_test.reshape(len(y_test), X_test.shape[2]), dtype=torch.float32).to(self.device)
                    else:
                        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
                        X_test, num_overlaps = utils.split_tensor_into_overlapping_windows(X_test, self.window_size, self.stride)
                        y_test = torch.repeat_interleave(y_test, repeats=num_overlaps)

                    self.classifier.fit(X_train, y_train, X_test, y_test,
                                        batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate)

            else:
                # Train the classifier without sample_weight
                self.classifier.fit(X_train, y_train)

            # Print the current split number every 5 iterations to show progress
            print(i, end=' ', flush=True) if i % 5 == 0 else None

            # Predictions on test data
            y_pred = self.classifier.predict(X_test)



            # Calculate metrics
            accuracy_scores.append(self.classifier.score(X_test, y_test))
            # print(f'Mean accuracy = {np.mean(accuracy_scores)}')
            if self.cnn_dim is not None:
                y_test = y_test.cpu().numpy()
                y_pred = y_pred.cpu().numpy()

            balanced_accuracy_scores.append(balanced_accuracy_score(y_test, y_pred))

            # Compute weighted accuracy, precision, recall, and F1-score with sample weights
            sample_weights = compute_sample_weight(class_weight='balanced', y=y_test)
            weighted_accuracy = np.average(y_pred == y_test, weights=sample_weights)
            weighted_accuracy_scores.append(weighted_accuracy)
            precision_scores.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            recall_scores.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

            if self.cnn_dim:
                index_to_label = {index: label for label, index in label_to_index.items()}
                # Convert back to labels
                y_test = [index_to_label[i] for i in y_test]
                y_pred = [index_to_label[i] for i in y_pred]

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

    def split_data(self, vintage=False, test_size=None):
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
        processed_labels = self.labels

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
                test_indices.extend(label_indices[:1])  # Take the first shuffled index for testing
                train_indices.extend(label_indices[1:])  # The rest is for training
            else:
                # If test_size is specified, calculate the split point based on the test_size proportion
                split_point = int(len(label_indices) * test_size)
                test_indices.extend(label_indices[:split_point])  # The first part goes into testing
                train_indices.extend(label_indices[split_point:])  # The remaining is for training

        test_indices = np.array(test_indices)
        train_indices = np.array(train_indices)

        # Split the data and labels into training and testing sets based on the calculated indices
        X_train, X_test = self.data[train_indices], self.data[test_indices]
        y_train, y_test = np.array(processed_labels)[train_indices], np.array(processed_labels)[test_indices]

        # Return the indices, data, and labels for both training and testing sets
        return train_indices, test_indices, X_train, X_test, y_train, y_test


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


