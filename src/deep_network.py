#!/usr/bin/env python3
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from model import DeepAlzheimerNet, evaluate_model, save_model_weights
from plotting import plot_training_curves, plot_architecture_comparison
import gc
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def train_deep_model(model, train_loader, val_loader, criterion, learning_rate, momentum,
                    reg_factor=0.001, epochs=100, early_stop_patience=10):
    # Initialize optimizer with momentum (no weight decay since we'll add L1 manually)
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

            # Add L1 regularization
            l1_loss = 0
            for param in model.parameters():
                l1_loss += param.abs().sum()
            loss += reg_factor * l1_loss

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

def run_deep_experiments():
    # Set seed from environment variable
    seed = int(os.getenv('SEED', 420))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Get batch size from environment
    batch_size = int(os.getenv('BATCH_SIZE', 256))

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Define architectures to test
    architectures = [
        [128, 64, 32],        # Pyramid
        [64, 32],             # Decreasing
        [32, 32, 32],         # Same size
        [32, 64, 128],        # Increasing
        [64, 32, 64],         # Hourglass
    ]

    # L1 regularization factor
    reg_factor = 0.001

    # Training parameters
    learning_rate = 0.1
    momentum = 0.6

    results = []

    # Run experiments for each fold
    for fold in range(5):
        # Load the preprocessed data
        X_train = pd.read_csv(f'data/X_train_fold_{fold}.csv')
        X_val = pd.read_csv(f'data/X_val_fold_{fold}.csv')
        y_train = pd.read_csv(f'data/y_train_fold_{fold}.csv')
        y_val = pd.read_csv(f'data/y_val_fold_{fold}.csv')

        input_size = X_train.shape[1]

        # Prepare dataloaders
        X_train_tensor = torch.FloatTensor(X_train.values).to(device)
        X_val_tensor = torch.FloatTensor(X_val.values).to(device)
        y_train_tensor = torch.FloatTensor(y_train.values).to(device)
        y_val_tensor = torch.FloatTensor(y_val.values).to(device)

        # Create generator with seed for DataLoader
        generator = torch.Generator()
        generator.manual_seed(seed)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                generator=generator)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Test different architectures
        for hidden_sizes in architectures:
            model = DeepAlzheimerNet(input_size, hidden_sizes, 'relu').to(device)
            criterion = torch.nn.BCELoss()

            train_losses, val_losses = train_deep_model(
                model,
                train_loader,
                val_loader,
                criterion,
                learning_rate,
                momentum,
                reg_factor,
                epochs=200
            )

            # Save model weights
            weights_folder = 'a5_res/model_weights'
            weights_filename = f'model_fold{fold}_arch{"_".join(map(str, hidden_sizes))}.pth'
            save_model_weights(model, weights_folder, weights_filename)

            metrics = evaluate_model(model, val_loader)

            results.append({
                'fold': fold,
                'architecture': str(hidden_sizes),
                'reg_factor': reg_factor,
                **metrics
            })

            # plot_training_curves(
            #     train_losses,
            #     val_losses,
            #     title=f'Deep Network Training (arch={hidden_sizes}, L1={reg_factor})',
            #     folder='a5_res'
            # )

    # Aggregate and display results
    results_df = pd.DataFrame(results)
    print("\nAverage results across folds:")
    grouped_results = results_df.groupby(['architecture']).mean().round(4)
    print(grouped_results)

    # Plot architecture comparisons
    plot_architecture_comparison(results_df)

    # Save results to txt file
    if not os.path.exists('a5_res'):
        os.makedirs('a5_res')

    # Save average results
    with open('a5_res/deep_network_results_average.txt', 'w') as f:
        f.write("Deep Neural Network Results with L1 Regularization - Average Across Folds\n")
        f.write("--------------------------------------------\n\n")
        f.write("Average Results Across Folds:\n")
        f.write(grouped_results.to_string())

    # Save per-fold results
    with open('a5_res/deep_network_results_per_fold.txt', 'w') as f:
        f.write("Deep Neural Network Results with L1 Regularization - Per Fold Results\n")
        f.write("--------------------------------------------\n\n")
        for fold in range(5):
            f.write(f"\nResults for Fold {fold}:\n")
            fold_results = results_df[results_df['fold'] == fold].groupby(['architecture']).mean().round(4)
            f.write(fold_results.to_string())
            f.write("\n")

if __name__ == "__main__":
    try:
        run_deep_experiments()
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
