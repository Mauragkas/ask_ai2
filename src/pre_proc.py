import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Load the dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# 2. Identify feature types
def identify_feature_types(df):
    # Remove PatientID and DoctorInCharge
    df_cleaned = df.drop(['PatientID', 'DoctorInCharge'], axis=1)

    # Separate features and target
    X = df_cleaned.drop('Diagnosis', axis=1)

    # First get all non-float/non-int columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

    # Check numerical columns for binary values
    binary_features = []
    true_numerical = []
    for col in numerical_features:
        if X[col].nunique() == 2:
            binary_features.append(col)
        else:
            true_numerical.append(col)

    return categorical_features.tolist(), true_numerical, binary_features

# 3. Create preprocessing pipeline
def create_preprocessing_pipeline(categorical_features, numerical_features, binary_features):
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    binary_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            ('bin', binary_transformer, binary_features)
        ])

    return preprocessor

# 4. Prepare data for modeling
def prepare_data(df):
    # Remove PatientID and DoctorInCharge
    df_cleaned = df.drop(['PatientID', 'DoctorInCharge'], axis=1)

    # Separate features and target
    X = df_cleaned.drop('Diagnosis', axis=1)
    y = df_cleaned['Diagnosis']

    return X, y

# 5. Implement cross-validation
def implement_cross_validation(X, y, preprocessor):
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Store fold indices
    fold_indices = []

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # Get training and validation sets
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        # Fit and transform training data
        X_train_transformed = pd.DataFrame(
            preprocessor.fit_transform(X_train_fold),
            columns=preprocessor.get_feature_names_out()
        )

        # Transform validation data
        X_val_transformed = pd.DataFrame(
            preprocessor.transform(X_val_fold),
            columns=preprocessor.get_feature_names_out()
        )

        # Save to CSV files
        X_train_transformed.to_csv(f'data/X_train_fold_{fold}.csv', index=False)
        X_val_transformed.to_csv(f'data/X_val_fold_{fold}.csv', index=False)
        pd.DataFrame(y_train_fold).to_csv(f'data/y_train_fold_{fold}.csv', index=False)
        pd.DataFrame(y_val_fold).to_csv(f'data/y_val_fold_{fold}.csv', index=False)

        fold_indices.append({
            'X_train': X_train_transformed,
            'X_val': X_val_transformed,
            'y_train': y_train_fold,
            'y_val': y_val_fold
        })

    return fold_indices

# Main execution
def main():
    # Load data
    df = load_data('alzheimers_disease_data.csv')

    # Identify feature types
    categorical_features, numerical_features, binary_features = identify_feature_types(df)

    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(categorical_features, numerical_features, binary_features)

    # Prepare data
    X, y = prepare_data(df)

    # Implement cross-validation
    fold_indices = implement_cross_validation(X, y, preprocessor)

    # Print some information about the processed data
    print("Data Overview:")
    print(f"Number of samples: {len(X)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of folds: {len(fold_indices)}")
    print("\nFeature types:")
    print(f"Categorical features: {categorical_features}")
    print(f"Numerical features: {numerical_features}")
    print(f"Binary features: {binary_features}")
    print("\nSample of preprocessed features (first fold):")
    print(fold_indices[0]['X_train'].head())
    print("\nClass distribution:")
    print(y.value_counts())

    return fold_indices

if __name__ == "__main__":
    fold_indices = main()
