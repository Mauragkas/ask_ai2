import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

def plot_training_curves(train_losses, val_losses, title):
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.figure(figsize=(12, 7))
    plt.plot(train_losses, linewidth=2, label='Training Loss')
    plt.plot(val_losses, linewidth=2, label='Validation Loss')
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    if not os.path.exists('./a2_res'):
        os.makedirs('./a2_res')
    plt.savefig(f'./a2_res/{title}.png', dpi=200, bbox_inches='tight')
    plt.close()
