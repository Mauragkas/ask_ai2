#!/usr/bin/env python3
import torch
import os
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from model import AlzheimerNet, evaluate_model, save_model_weights
from plotting import plot_training_curves, plot_momentum_results
import numpy as np
import gc
from dotenv import load_dotenv
load_dotenv()

def prepare_dataloaders(X_train, X_val, y_train, y_val, batch_size=256, device='cuda', seed=420):
    # Convert to PyTorch tensors and move to GPU
    X_train_tensor = torch.FloatTensor(X_train.values).to(device)
    X_val_tensor = torch.FloatTensor(X_val.values).to(device)
    y_train_tensor = torch.FloatTensor(y_train.values).to(device)
    y_val_tensor = torch.FloatTensor(y_val.values).to(device)

    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    # Create dataloaders with seed
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader

def train_with_momentum(model, train_loader, val_loader, criterion, learning_rate, momentum, epochs=100, early_stop_patience=10):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
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

def run_momentum_experiments():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set seed from environment variable
    seed = int(os.getenv('SEED', 420))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Get batch size from environment
    batch_size = int(os.getenv('BATCH_SIZE', 256))

    # Momentum and learning rate configurations
    configs = [
        {'momentum': 0.2, 'lr': 0.001},
        {'momentum': 0.2, 'lr': 0.05},
        {'momentum': 0.2, 'lr': 0.1},
        {'momentum': 0.6, 'lr': 0.001},
        {'momentum': 0.6, 'lr': 0.05},
        {'momentum': 0.6, 'lr': 0.1}
    ]

    results = []

    # Run experiments for each fold
    for fold in range(5):
        # Load the preprocessed data
        X_train = pd.read_csv(f'data/X_train_fold_{fold}.csv')
        X_val = pd.read_csv(f'data/X_val_fold_{fold}.csv')
        y_train = pd.read_csv(f'data/y_train_fold_{fold}.csv')
        y_val = pd.read_csv(f'data/y_val_fold_{fold}.csv')

        input_size = X_train.shape[1]
        hidden_size = input_size*2  # Using the best architecture from A2

        # Prepare dataloaders with seed
        train_loader, val_loader = prepare_dataloaders(
            X_train, X_val, y_train, y_val,
            device=str(device),
            seed=seed,
            batch_size=batch_size
        )

        # Test different configurations
        for config in configs:
            model = AlzheimerNet(input_size, hidden_size, 'relu').to(device)  # Using ReLU as activation
            criterion = torch.nn.BCELoss()

            train_losses, val_losses = train_with_momentum(
                model,
                train_loader,
                val_loader,
                criterion,
                config['lr'],
                config['momentum']
            )

            # Save model weights
            weights_folder = 'a3_res/model_weights'
            weights_filename = f'model_fold{fold}_m{config["momentum"]}_lr{config["lr"]}.pth'
            save_model_weights(model, weights_folder, weights_filename)

            metrics = evaluate_model(model, val_loader)

            results.append({
                'fold': fold,
                'momentum': config['momentum'],
                'learning_rate': config['lr'],
                **metrics
            })

            plot_training_curves(
                train_losses,
                val_losses,
                title=f'Training_Curves_m{config["momentum"]}_lr{config["lr"]}',
                fold=fold,
                folder='a3_res'
            )

    # Aggregate and display results
    results_df = pd.DataFrame(results)
    print("\nAverage results across folds:")
    grouped_results = results_df.groupby(['momentum', 'learning_rate']).mean().round(4)
    print(grouped_results)

    # Plot the results
    plot_momentum_results(results_df)

    # Save results to txt file
    if not os.path.exists('a3_res'):
        os.makedirs('a3_res')

    with open('a3_res/momentum_results.txt', 'w') as f:
        f.write("Average Results Across Folds:\n\n")
        f.write(grouped_results.to_string())

    # Save per-fold results
    with open('a3_res/momentum_results_per_fold.txt', 'w') as f:
        f.write("Momentum Results - Per Fold Results\n")
        f.write("--------------------------------------------\n\n")
        for fold in range(5):
            f.write(f"\nResults for Fold {fold}:\n")
            fold_results = results_df[results_df['fold'] == fold].groupby(['momentum', 'learning_rate']).mean().round(4)
            f.write(fold_results.to_string())
            f.write("\n")

if __name__ == "__main__":
    try:
        run_momentum_experiments()
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
