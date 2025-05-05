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
from part_a.pre_proc import identify_feature_types, create_preprocessing_pipeline
from part_a.plotting import plot_training_curves
from sklearn.model_selection import StratifiedKFold
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

    # Selected features
    selected_features = [
        'EducationLevel',
        'Depression',
        'MMSE',
        'FunctionalAssessment',
        'MemoryComplaints',
        'BehavioralProblems',
        'ADL',
        'Confusion'
    ]

    # Load the original dataset
    df = pd.read_csv('./data/alzheimers_disease_data.csv')

    # Select only the specified features + target variable
    features_df = df[selected_features + ['Diagnosis', 'PatientID', 'DoctorInCharge']]

    # Identify feature types for the selected features
    cat_features, num_features, bin_features = identify_feature_types(features_df)

    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(cat_features, num_features, bin_features)

    # Prepare data - drop ID columns
    features_cleaned = features_df.drop(['PatientID', 'DoctorInCharge'], axis=1)
    X = features_cleaned.drop('Diagnosis', axis=1)
    y = features_cleaned['Diagnosis']

    # Results storage
    results = []

    # Create output directories
    os.makedirs('b_res/model_weights', exist_ok=True)
    os.makedirs('b_res/plots', exist_ok=True)

    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # Get training and validation sets
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        # Fit and transform training data
        X_train_transformed = preprocessor.fit_transform(X_train_fold)
        X_val_transformed = preprocessor.transform(X_val_fold)

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_transformed).to(device)
        X_val_tensor = torch.FloatTensor(X_val_transformed).to(device)
        y_train_tensor = torch.FloatTensor(y_train_fold.values.reshape(-1, 1)).to(device)
        y_val_tensor = torch.FloatTensor(y_val_fold.values.reshape(-1, 1)).to(device)

        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        generator = torch.Generator()
        generator.manual_seed(seed)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Get input size
        input_size = X_train_transformed.shape[1]

        # Set hidden layer size to 16 as requested
        hidden_size = 16

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

        # Get feature names after preprocessing
        feature_names = preprocessor.get_feature_names_out()

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
        for feature in selected_features:
            f.write(f"- {feature}\n")

        f.write("\n## Average Results Across Folds\n")
        f.write(avg_results.to_string())

        f.write("\n\n## Per-Fold Results\n")
        f.write(results_df.to_string())

    return results_df, preprocessor, selected_features

if __name__ == "__main__":
    try:
        results, preprocessor, features = train_selective_feature_model()
        print("\nTraining completed successfully!")
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
