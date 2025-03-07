import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from model import AlzheimerNet, train_model, evaluate_model, plot_training_curves

def load_fold_data(fold_num):
    # Load the preprocessed data
    X_train = pd.read_csv(f'data/X_train_fold_{fold_num}.csv')
    X_val = pd.read_csv(f'data/X_val_fold_{fold_num}.csv')
    y_train = pd.read_csv(f'data/y_train_fold_{fold_num}.csv')
    y_val = pd.read_csv(f'data/y_val_fold_{fold_num}.csv')

    return X_train, X_val, y_train, y_val

def prepare_dataloaders(X_train, X_val, y_train, y_val, batch_size=128, device='cuda'):
    # Convert to PyTorch tensors and move to GPU
    X_train_tensor = torch.FloatTensor(X_train.values).to(device)
    X_val_tensor = torch.FloatTensor(X_val.values).to(device)
    y_train_tensor = torch.FloatTensor(y_train.values).to(device)
    y_val_tensor = torch.FloatTensor(y_val.values).to(device)

    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader

def run_experiments():
    # Set device to ROCm GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    input_size = None  # Will be set from data
    hidden_sizes = []  # Will be calculated based on input_size
    activation_functions = ['relu', 'tanh', 'silu']
    results = []

    # Run experiments for each fold
    for fold in range(5):
        X_train, X_val, y_train, y_val = load_fold_data(fold)

        if input_size is None:
            input_size = X_train.shape[1]
            hidden_sizes = [input_size//2, 2*input_size//3, input_size, 2*input_size]

        train_loader, val_loader = prepare_dataloaders(X_train, X_val, y_train, y_val, device=str(device))

        # Test different architectures
        for hidden_size in hidden_sizes:
            for activation in activation_functions:
                model = AlzheimerNet(input_size, hidden_size, activation).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                criterion = torch.nn.BCELoss()

                train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer)
                metrics = evaluate_model(model, val_loader)

                results.append({
                    'fold': fold,
                    'hidden_size': hidden_size,
                    'activation': activation,
                    **metrics
                })

                plot_training_curves(train_losses, val_losses,
                                  f'Training Curves (h={hidden_size}, act={activation})')

    # Aggregate and display results
    results_df = pd.DataFrame(results)
    print("\nAverage results across folds:")
    print(results_df.groupby(['hidden_size', 'activation']).mean().round(4))

if __name__ == "__main__":
    # run_experiments()
    try:
        run_experiments()
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"An error occurred during training: {e}")
