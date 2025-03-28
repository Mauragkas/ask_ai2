#!/usr/bin/env python3
import torch
import os
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from model import AlzheimerNet, train_model, evaluate_model, save_model_weights
from plotting import plot_training_curves, plot_results
import threading
import gc
from dotenv import load_dotenv
load_dotenv()

def load_fold_data(fold_num):
    # Load the preprocessed data
    X_train = pd.read_csv(f'data/X_train_fold_{fold_num}.csv')
    X_val = pd.read_csv(f'data/X_val_fold_{fold_num}.csv')
    y_train = pd.read_csv(f'data/y_train_fold_{fold_num}.csv')
    y_val = pd.read_csv(f'data/y_val_fold_{fold_num}.csv')

    return X_train, X_val, y_train, y_val

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

def run_experiments():
    # Set device to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set seed from environment variable
    seed = int(os.getenv('SEED', 420))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Set batch size from environment variable
    batch_size = int(os.getenv('BATCH_SIZE', 256))

    input_size = None  # Will be set from data
    hidden_sizes = []  # Will be calculated based on input_size
    activation_functions = ['relu', 'silu', 'tanh']
    results = []

    # Run experiments for each fold
    for fold in range(5):
        X_train, X_val, y_train, y_val = load_fold_data(fold)

        if input_size is None:
            input_size = X_train.shape[1]
            hidden_sizes = [input_size//2, 2*input_size//3, input_size, input_size*2]

        train_loader, val_loader = prepare_dataloaders(X_train, X_val, y_train, y_val, batch_size=batch_size, device=str(device), seed=seed)

        # Test different architectures
        for hidden_size in hidden_sizes:
            for activation in activation_functions:
                model = AlzheimerNet(input_size, hidden_size, activation).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                criterion = torch.nn.BCELoss()

                train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer)
                metrics = evaluate_model(model, val_loader)

                # Save model weights
                weights_folder = 'a2_res/model_weights'
                weights_filename = f'model_fold{fold}_h{hidden_size}_act{activation}.pth'
                save_model_weights(model, weights_folder, weights_filename)

                results.append({
                    'fold': fold,
                    'hidden_size': hidden_size,
                    'activation': activation,
                    **metrics
                })

                # plot_training_curves(train_losses, val_losses, title=f'Training Curves (h={hidden_size:02d}, act={activation})')

    # Aggregate and display results
    results_df = pd.DataFrame(results)
    print("\nAverage results across folds:")
    grouped_results = results_df.groupby(['hidden_size', 'activation']).mean().round(4)
    print(grouped_results)

    # Plot the results
    plot_results(results_df)

    # Create a2_res directory if it doesn't exist
    if not os.path.exists('a2_res'):
        os.makedirs('a2_res')

    # Save results to txt file
    with open('a2_res/experiment_results.txt', 'w') as f:
        f.write("Average Results Across Folds:\n\n")
        f.write(grouped_results.to_string())

    # Save per-fold results
    with open('a2_res/experiment_results_per_fold.txt', 'w') as f:
        f.write("Single Hidden Layer Results - Per Fold Results\n")
        f.write("--------------------------------------------\n\n")
        for fold in range(5):
            f.write(f"\nResults for Fold {fold}:\n")
            fold_results = results_df[results_df['fold'] == fold].groupby(['hidden_size', 'activation']).mean().round(4)
            f.write(fold_results.to_string())
            f.write("\n")

if __name__ == "__main__":
    try:
        run_experiments()
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
