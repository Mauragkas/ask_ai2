import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
import os

class DeepAlzheimerNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, activation_fn='relu'):
        super(DeepAlzheimerNet, self).__init__()

        # Create list to hold all layers
        layers = []

        # Input layer to first hidden layer
        prev_size = input_size

        # Add hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))

            # Add activation function
            if activation_fn == 'relu':
                layers.append(nn.ReLU())
            elif activation_fn == 'tanh':
                layers.append(nn.Tanh())
            elif activation_fn == 'silu':
                layers.append(nn.SiLU())

            prev_size = hidden_size

        # Add output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())

        # Create sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class AlzheimerNet(nn.Module):
    def __init__(self, input_size, hidden_size, activation_fn='relu'):
        super(AlzheimerNet, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)  # Binary classification

        # Set activation function
        if activation_fn == 'relu':
            self.activation = nn.ReLU()
        elif activation_fn == 'tanh':
            self.activation = nn.Tanh()
        elif activation_fn == 'silu':
            self.activation = nn.SiLU()

        self.sigmoid = nn.Sigmoid()  # Output activation

    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=100, early_stop_patience=10):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        # Validation phase
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item()

        train_losses.append(epoch_train_loss/len(train_loader))
        val_losses.append(epoch_val_loss/len(val_loader))

        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            if __debug__:
                print(f'Early stopping at epoch {epoch}')
            break

    return train_losses, val_losses

def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            # Move tensors to CPU before converting to numpy
            predictions.extend((outputs > 0.5).float().cpu().numpy())
            actuals.extend(targets.cpu().numpy())

    predictions_array = np.array(predictions)
    actuals_array = np.array(actuals)

    return {
        'ce_loss': nn.BCELoss()(torch.tensor(predictions_array), torch.tensor(actuals_array)).item(),
        'mse': nn.MSELoss()(torch.tensor(predictions_array), torch.tensor(actuals_array)).item(),
        'accuracy': accuracy_score(actuals, predictions)
    }

def save_model_weights(model, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = os.path.join(folder, filename)
    torch.save(model.state_dict(), path)

def load_model_weights(model, weights_path):
    model.load_state_dict(torch.load(weights_path))
    return model
