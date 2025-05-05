import torch
import pandas as pd
import numpy as np
from tqdm import tqdm  # Add this import at the top of the file
from model import AlzheimerNet, DeepAlzheimerNet
from pre_proc import identify_feature_types, create_preprocessing_pipeline
from sklearn.compose import ColumnTransformer

def prepare_single_input(data_dict, preprocessor):
    # Convert dictionary to DataFrame
    df = pd.DataFrame([data_dict])

    # Drop ID and Doctor columns if they exist
    if 'PatientID' in df.columns:
        df = df.drop('PatientID', axis=1)
    if 'DoctorInCharge' in df.columns:
        df = df.drop('DoctorInCharge', axis=1)

    # Transform the data using the preprocessor
    transformed_data = preprocessor.transform(df)

    # Convert to tensor
    device = torch.device('cpu')
    input_tensor = torch.FloatTensor(transformed_data).to(device)

    return input_tensor

def load_and_evaluate_model(model_path, input_tensor, model_type='simple', architecture=None):
    # Get input size from tensor
    input_size = input_tensor.shape[1]
    device = torch.device('cpu')

    # Initialize model
    if model_type == 'simple':
        model = AlzheimerNet(input_size=input_size, hidden_size=input_size*2, activation_fn='relu').to(device)
    else:  # deep
        model = DeepAlzheimerNet(input_size=input_size, hidden_sizes=architecture, activation_fn='relu').to(device)

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Make prediction
    with torch.no_grad():
        probability = model(input_tensor)
        prediction = (probability > 0.5).int().item()  # Convert to binary prediction

    return {
        'probability': probability.item(),
        'prediction': prediction
    }

def load_and_evaluate_model_ensemble(model_paths, input_tensor, model_type='simple', architecture=None):
    input_size = input_tensor.shape[1]
    probabilities = []
    device = torch.device('cpu')

    for path in model_paths:
        # Initialize model
        if model_type == 'simple':
            model = AlzheimerNet(input_size=input_size, hidden_size=input_size*2, activation_fn='relu').to(device)
        else:  # deep
            model = DeepAlzheimerNet(input_size=input_size, hidden_sizes=architecture, activation_fn='relu').to(device)

        # Load weights
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()

        # Make prediction
        with torch.no_grad():
            probability = model(input_tensor)
            probabilities.append(probability.item())

    # Average probabilities across folds
    avg_probability = sum(probabilities) / len(probabilities)
    avg_prediction = int(avg_probability > 0.5)

    return {
        'probability': avg_probability,
        'prediction': avg_prediction,
        'fold_probabilities': probabilities
    }

def create_preprocessor(sample_data_path):
    # Load sample data to fit preprocessor
    sample_df = pd.read_csv(sample_data_path)

    # Identify feature types
    categorical_features, numerical_features, binary_features = identify_feature_types(sample_df)

    # Create preprocessor
    preprocessor = create_preprocessing_pipeline(
        categorical_features,
        numerical_features,
        binary_features
    )

    # Fit preprocessor (excluding ID and Doctor columns)
    df_cleaned = sample_df.drop(['PatientID', 'DoctorInCharge', 'Diagnosis'], axis=1)
    preprocessor.fit(df_cleaned)

    return preprocessor

def main():
    # Path to sample data
    sample_data_path = './data/alzheimers_disease_data.csv'
    n = 10

    # Load full dataset
    df = pd.read_csv(sample_data_path)
    sample_rows = df.sample(n).sort_values(by='PatientID')

    try:
        # Create and fit preprocessor
        preprocessor = create_preprocessor(sample_data_path)

        # Model paths for all 5 folds
        # model_paths = {
        #     'simple': [f'a2_res/model_weights/model_fold{i}_h64_actrelu.pth' for i in range(0, 5)],
        #     'deep': [f'a5_res/model_weights/model_fold{i}_arch32_64_128.pth' for i in range(0, 5)]
        # }
        model_paths = {
            'simple': ['a2_res/model_weights/model_fold0_h64_actrelu.pth'],
            'deep': ['a5_res/model_weights/model_fold0_arch32_64_128.pth']
        }

        print(f"Evaluating {n} random patients with ensemble predictions:")
        print("PatientID | Actual | Simple Pred (Prob) | Deep Pred (Prob)")
        print("-" * 65)

        mistakes = {
            'simple': [],
            'deep': []
        }

        # Add progress bar for patient evaluation
        for _, row in tqdm(sample_rows.iterrows(),
                         total=len(sample_rows),
                         desc="Patient evaluation",
                         leave=True,
                         ncols=100):
            # Prepare input
            input_data = row.to_dict()
            input_tensor = prepare_single_input(input_data, preprocessor)

            # Get ensemble predictions
            simple_result = load_and_evaluate_model_ensemble(
                model_paths['simple'],
                input_tensor,
                model_type='simple'
            )

            deep_result = load_and_evaluate_model_ensemble(
                model_paths['deep'],
                input_tensor,
                model_type='deep',
                architecture=[32, 64, 128]
            )

            print(f"{row['PatientID']:9d} | {row['Diagnosis']:6d} | {simple_result['prediction']:d} ({simple_result['probability']:.3f}) | {deep_result['prediction']:d} ({deep_result['probability']:.3f})")

            # Optionally print individual fold probabilities
            print(f"Simple fold probabilities: {[f'{p:.3f}' for p in simple_result['fold_probabilities']]}")
            print(f"Deep fold probabilities: {[f'{p:.3f}' for p in deep_result['fold_probabilities']]}")
            print("-" * 65)

            # Track mistakes
            if simple_result['prediction'] != row['Diagnosis']:
                mistakes['simple'].append({
                    'PatientID': row['PatientID'],
                    'Actual': row['Diagnosis'],
                    'Predicted': simple_result['prediction'],
                    'Probability': simple_result['probability']
                })

            if deep_result['prediction'] != row['Diagnosis']:
                mistakes['deep'].append({
                    'PatientID': row['PatientID'],
                    'Actual': row['Diagnosis'],
                    'Predicted': deep_result['prediction'],
                    'Probability': deep_result['probability']
                })

        # Print mistake summary
        print("\nMistakes Summary:")
        print("\nSimple Model Mistakes:")
        if mistakes['simple']:
            for mistake in mistakes['simple']:
                print(f"PatientID: {mistake['PatientID']} - Actual: {mistake['Actual']}, Predicted: {mistake['Predicted']} (Prob: {mistake['Probability']:.3f})")
        else:
            print("No mistakes made by simple model")

        print("\nDeep Model Mistakes:")
        if mistakes['deep']:
            for mistake in mistakes['deep']:
                print(f"PatientID: {mistake['PatientID']} - Actual: {mistake['Actual']}, Predicted: {mistake['Predicted']} (Prob: {mistake['Probability']:.3f})")
        else:
            print("No mistakes made by deep model")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Evaluation interrupted by user")
