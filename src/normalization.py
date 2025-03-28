#!/usr/bin/env python3
import torch
import os
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from model import AlzheimerNet, evaluate_model, save_model_weights
from plotting import plot_training_curves, plot_regularization_results
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gc
from dotenv import load_dotenv
load_dotenv()

def l1_regularization(model):
    # Calculate L1 regularization term for the model
    l1_reg = torch.tensor(0., requires_grad=True)
    for param in model.parameters():
        l1_reg = l1_reg + torch.norm(param, 1)
    return l1_reg

def train_with_regularization(model, train_loader, val_loader, criterion, learning_rate, momentum,
                            reg_type='l2', reg_factor=0.001, epochs=100, early_stop_patience=10):
    # Train the model with L1 or L2 regularization
    if reg_type == 'l2':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                  momentum=momentum, weight_decay=reg_factor)
    else:  # L1 regularization
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

            # Add L1 regularization if selected
            if reg_type == 'l1':
                l1_term = reg_factor * l1_regularization(model)
                loss += l1_term

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

def run_regularization_experiments():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set seed from environment variable
    seed = int(os.getenv('SEED', 420))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)  # Also set numpy seed

    # Get batch size from environment variable
    batch_size = int(os.getenv('BATCH_SIZE', 256))

    # Regularization factors to test
    reg_factors = [0.0001, 0.001, 0.01]
    reg_types = ['l1', 'l2']

    # Using best parameters from A3
    momentum = 0.6
    learning_rate = 0.05

    results = []

    # Run experiments for each fold
    for fold in range(5):
        # Load the preprocessed data
        X_train = pd.read_csv(f'data/X_train_fold_{fold}.csv')
        X_val = pd.read_csv(f'data/X_val_fold_{fold}.csv')
        y_train = pd.read_csv(f'data/y_train_fold_{fold}.csv')
        y_val = pd.read_csv(f'data/y_val_fold_{fold}.csv')

        input_size = X_train.shape[1]
        hidden_size = input_size*2

        # Use the new prepare_dataloaders function
        train_loader, val_loader = prepare_dataloaders(
            X_train, X_val, y_train, y_val,
            batch_size=batch_size,
            device=str(device),
            seed=seed
        )

        # Test different regularization types and factors
        for reg_type in reg_types:
            for reg_factor in reg_factors:
                model = AlzheimerNet(input_size, hidden_size, 'relu').to(device)
                criterion = torch.nn.BCELoss()

                train_losses, val_losses = train_with_regularization(
                    model,
                    train_loader,
                    val_loader,
                    criterion,
                    learning_rate,
                    momentum,
                    reg_type,
                    reg_factor
                )

                # Save model weights
                weights_folder = 'a4_res/model_weights'
                weights_filename = f'model_fold{fold}_{reg_type}_factor{reg_factor}.pth'
                save_model_weights(model, weights_folder, weights_filename)

                metrics = evaluate_model(model, val_loader)

                results.append({
                    'fold': fold,
                    'reg_type': reg_type,
                    'reg_factor': reg_factor,
                    **metrics
                })

                # plot_training_curves(
                #     train_losses,
                #     val_losses,
                #     title=f'Training Curves ({reg_type.upper()}={reg_factor})',
                #     folder='a4_res'
                # )

    # Aggregate and display results
    results_df = pd.DataFrame(results)
    print("\nAverage results across folds:")
    grouped_results = results_df.groupby(['reg_type', 'reg_factor']).mean().round(4)
    print(grouped_results)

    # Plot the results
    plot_regularization_results(results_df)

    # Save results to txt file
    if not os.path.exists('a4_res'):
        os.makedirs('a4_res')

    with open('a4_res/regularization_results.txt', 'w') as f:
        f.write("Average Results Across Folds:\n\n")
        f.write("L1 vs L2 Regularization Comparison\n")
        f.write("---------------------------------\n\n")
        f.write(grouped_results.to_string())

    # Save per-fold results
    with open('a4_res/regularization_results_per_fold.txt', 'w') as f:
        f.write("L1 vs L2 Regularization Results - Per Fold Results\n")
        f.write("--------------------------------------------\n\n")
        for fold in range(5):
            f.write(f"\nResults for Fold {fold}:\n")
            fold_results = results_df[results_df['fold'] == fold].groupby(['reg_type', 'reg_factor']).mean().round(4)
            f.write(fold_results.to_string())
            f.write("\n")

if __name__ == "__main__":
    try:
        run_regularization_experiments()
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
