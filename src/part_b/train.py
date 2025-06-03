#!/usr/bin/env python3
import torch
import os
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from part_a.model import AlzheimerNet, train_model, evaluate_model, save_model_weights
from part_a.plotting import plot_training_curves
import gc
from dotenv import load_dotenv

load_dotenv()

def plot_feature_importance(model, feature_names, save_path):
    """Plot feature importance based on model weights"""
    # Get weights from the first layer
    weights = model.hidden.weight.data.cpu().numpy()

    # Calculate the absolute importance for each input feature
    importance = np.mean(np.abs(weights), axis=0)

    # Create a DataFrame for easier plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance Based on Model Weights', fontsize=14)
    plt.xlabel('Average Weight Magnitude', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_comparison(results_df, save_path):
    """Plot comparison of metrics across folds"""
    metrics = ['accuracy', 'ce_loss', 'mse']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, metric in enumerate(metrics):
        sns.barplot(x='fold', y=metric, data=results_df, ax=axes[i])
        axes[i].set_title(f'{metric.upper()} by Fold')
        axes[i].set_xlabel('Fold')
        axes[i].set_ylabel(metric.upper())

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def train_selective_feature_model():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set seed
    seed = int(os.getenv('SEED', 420))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Set batch size
    batch_size = int(os.getenv('BATCH_SIZE', 256))

    # Define the features from the best configuration
    # These are the selected features from the GA optimization
    selected_features_patterns = [
        'MMSE',
        'FunctionalAssessment',
        'MemoryComplaints',
        'BehavioralProblems',
        'ADL'
    ]

    results = []

    # Create output directories
    os.makedirs('b_res/model_weights', exist_ok=True)
    os.makedirs('b_res/plots', exist_ok=True)

    # For each fold
    for fold in range(5):
        # Load preprocessed data (like in Part A)
        X_train = pd.read_csv(f'data/X_train_fold_{fold}.csv')
        X_val = pd.read_csv(f'data/X_val_fold_{fold}.csv')
        y_train = pd.read_csv(f'data/y_train_fold_{fold}.csv')
        y_val = pd.read_csv(f'data/y_val_fold_{fold}.csv')

        # Filter columns to keep only those containing our target features
        # This handles transformed feature names (e.g., one-hot encoded columns)
        selected_cols = []
        for pattern in selected_features_patterns:
            matched_cols = [col for col in X_train.columns if pattern in col]
            selected_cols.extend(matched_cols)

        # Keep only the selected columns
        X_train_selected = X_train[selected_cols]
        X_val_selected = X_val[selected_cols]

        # Convert to PyTorch tensors and continue as before
        X_train_tensor = torch.FloatTensor(X_train_selected.values).to(device)
        X_val_tensor = torch.FloatTensor(X_val_selected.values).to(device)
        y_train_tensor = torch.FloatTensor(y_train.values).to(device)
        y_val_tensor = torch.FloatTensor(y_val.values).to(device)

        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        generator = torch.Generator()
        generator.manual_seed(seed)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Get input size
        input_size = X_train_selected.shape[1]

        hidden_size = input_size * 2

        print(f"Fold {fold} - Input size: {input_size}, Hidden size: {hidden_size}")

        # Create model with ReLU activation
        model = AlzheimerNet(input_size, hidden_size, 'relu').to(device)

        # Set optimizer and criterion
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCELoss()

        # Train the model
        train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer)

        # Evaluate the model
        metrics = evaluate_model(model, val_loader)

        print(f"\nFold {fold} Evaluation:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        # Save model weights in b_res folder
        weights_folder = 'b_res/model_weights'
        weights_filename = f'selective_model_fold{fold}.pth'
        save_model_weights(model, weights_folder, weights_filename)

        # Plot training curves
        plot_training_curves(
            train_losses,
            val_losses,
            title=f'Training Curves - Fold {fold}',
            folder='b_res/plots'
        )

        # For feature importance, use the filtered column names
        feature_names = selected_cols

        # Plot feature importance
        plot_feature_importance(
            model,
            feature_names,
            save_path=f'b_res/plots/feature_importance_fold{fold}.png'
        )

        results.append({
            'fold': fold,
            **metrics
        })

    # Calculate and save average results
    results_df = pd.DataFrame(results)
    avg_results = results_df.mean().round(4)

    print("\nAverage Results Across Folds:")
    print(avg_results)

    # Plot metrics comparison across folds
    plot_metrics_comparison(
        results_df,
        save_path='b_res/plots/metrics_comparison.png'
    )

    # Create output directory
    os.makedirs('b_res', exist_ok=True)

    # Save model details and results
    with open('b_res/model_info.txt', 'w') as f:
        f.write("# Selective Feature Neural Network\n\n")
        f.write("## Model Configuration\n")
        f.write(f"- Hidden layer size: {hidden_size}\n")
        f.write("- Activation function: ReLU\n\n")

        f.write("## Selected Features\n")
        for pattern in selected_features_patterns:
            f.write(f"- {pattern}\n")

        f.write("\n## Average Results Across Folds\n")
        f.write(avg_results.to_string())

        f.write("\n\n## Per-Fold Results\n")
        f.write(results_df.to_string())

    return results_df, selected_cols, selected_features_patterns

if __name__ == "__main__":
    try:
        results, selected_cols, features = train_selective_feature_model()
        print("\nTraining completed successfully!")
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
