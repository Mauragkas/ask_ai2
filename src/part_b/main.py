#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import random
import torch
import gc
from multiprocessing import Pool
from dotenv import load_dotenv
from ga_runner import run_single_experiment
from ga_visualization import plot_experiment_comparisons
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import json

load_dotenv()

def run_all_experiments():
    """Run all requested GA experiments"""
    # Set seed from environment variable
    seed = int(os.getenv('SEED', 420))
    np.random.seed(seed)
    random.seed(seed)

    print("Loading dataset...")
    # Load the original dataset
    df = pd.read_csv('./data/alzheimers_disease_data.csv')

    # Drop ID and doctor columns
    df = df.drop(['PatientID', 'DoctorInCharge'], axis=1)

    # Split features and target
    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']

    print(f"Dataset loaded with {X.shape[1]} features and {len(y)} samples")

    # Create results directory
    result_dir = "b_res"
    os.makedirs(result_dir, exist_ok=True)

    # Define parameter values to test
    population_sizes = [
        20,
        # 200
    ]

    # Define valid combinations of crossover_prob and mutation_prob
    valid_combinations = [
        # (0.6, 0.0),
        (0.6, 0.01),
        # (0.6, 0.10),
        # (0.9, 0.01),
        # (0.1, 0.01)
    ]

    # Create configs with nested loops
    configs = []
    for pop_size in population_sizes:
        for cross_prob, mut_prob in valid_combinations:
            configs.append({
                'population_size': pop_size,
                'crossover_prob': cross_prob,
                'mutation_prob': mut_prob
            })

    # Prepare arguments for each experiment with GPU assignment
    experiment_args = []
    for i, config in enumerate(configs):
        # Alternate between GPU 0 and 1
        gpu_id = i % 2
        experiment_args.append((X, y, config, result_dir, gpu_id))

    # Run experiments in parallel
    with Pool() as pool:
        all_results = pool.map(run_single_experiment, experiment_args)

    # Filter out None results from failed experiments
    all_results = [result for result in all_results if result is not None]

    if not all_results:
        print("All experiments failed. No results to analyze.")
        return

    # Create comparison table
    comparison_data = []
    for result in all_results:
        params = result['parameters']
        res = result['results']
        comparison_data.append({
            'Population Size': params['population_size'],
            'Crossover Prob': params['crossover_prob'],
            'Mutation Prob': params['mutation_prob'],
            'Avg Fitness': res['avg_best_fitness'],
            'Avg Selected Features': res['avg_features_selected']
        })

    comparison_df = pd.DataFrame(comparison_data)

    # Save comparison table
    comparison_df.to_csv(f"{result_dir}/experiment_comparison.csv", index=False)

    # Create comparison plots
    plot_experiment_comparisons(comparison_df, result_dir)

    # Create scatter plot without requiring convex hull
    plt.figure(figsize=(10, 7))

    # Split data by Population Size
    data_pop_20 = comparison_df[comparison_df['Population Size'] == 20]
    data_pop_200 = comparison_df[comparison_df['Population Size'] == 200]

    # Plot data points
    plt.scatter(data_pop_20['Avg Fitness'], data_pop_20['Avg Selected Features'],
                c='blue', marker='o', s=100, label='Population Size = 20', zorder=5)
    plt.scatter(data_pop_200['Avg Fitness'], data_pop_200['Avg Selected Features'],
                c='red', marker='^', s=100, label='Population Size = 200', zorder=5)

    # Add labels with Crossover Prob and Mutation Prob for each point
    for i, row in data_pop_20.iterrows():
        label = f"C:{row['Crossover Prob']}, M:{row['Mutation Prob']}"
        plt.annotate(label,
                     (row['Avg Fitness'], row['Avg Selected Features']),
                     xytext=(5, 5),
                     textcoords='offset points',
                     fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    for i, row in data_pop_200.iterrows():
        label = f"C:{row['Crossover Prob']}, M:{row['Mutation Prob']}"
        plt.annotate(label,
                     (row['Avg Fitness'], row['Avg Selected Features']),
                     xytext=(5, -5),
                     textcoords='offset points',
                     fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    # Try to create convex hulls if possible
    try:
        # Create points for the convex hulls
        points_pop_20 = data_pop_20[['Avg Fitness', 'Avg Selected Features']].values
        points_pop_200 = data_pop_200[['Avg Fitness', 'Avg Selected Features']].values

        # Add small random jitter to prevent collinearity
        if len(points_pop_20) >= 3:  # Need at least 3 points for a hull
            jitter = np.random.normal(0, 0.0001, points_pop_20.shape)
            jittered_points_20 = points_pop_20 + jitter
            hull_pop_20 = ConvexHull(jittered_points_20)

            # Plot the convex hull
            for simplex in hull_pop_20.simplices:
                plt.plot(jittered_points_20[simplex, 0], jittered_points_20[simplex, 1],
                        'b-', alpha=0.7, linewidth=2)
            plt.fill(jittered_points_20[hull_pop_20.vertices, 0],
                    jittered_points_20[hull_pop_20.vertices, 1],
                    'blue', alpha=0.1)

        if len(points_pop_200) >= 3:  # Need at least 3 points for a hull
            jitter = np.random.normal(0, 0.0001, points_pop_200.shape)
            jittered_points_200 = points_pop_200 + jitter
            hull_pop_200 = ConvexHull(jittered_points_200)

            # Plot the convex hull
            for simplex in hull_pop_200.simplices:
                plt.plot(jittered_points_200[simplex, 0], jittered_points_200[simplex, 1],
                        'r-', alpha=0.7, linewidth=2)
            plt.fill(jittered_points_200[hull_pop_200.vertices, 0],
                    jittered_points_200[hull_pop_200.vertices, 1],
                    'red', alpha=0.1)

    except Exception as e:
        print(f"Could not create convex hulls: {e}")
        print("Plotting points without hull visualization")

    # Add labels and title
    plt.title('Parameter Comparison in Fitness-Features Space', fontsize=14)
    plt.xlabel('Average Fitness', fontsize=12)
    plt.ylabel('Average Selected Features', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{result_dir}/parameter_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Create summary report
    with open(f"{result_dir}/experiment_summary_all.txt", 'w') as f:
        f.write("GA Feature Selection - All Experiments Summary\n")
        f.write("==========================================\n\n")
        f.write("Parameter Configurations and Results:\n\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")

        # Find best configuration
        best_idx = comparison_df['Avg Fitness'].idxmax()
        best_config = comparison_df.iloc[best_idx]

        f.write("Best Configuration:\n")
        f.write(f"- Population Size: {best_config['Population Size']}\n")
        f.write(f"- Crossover Probability: {best_config['Crossover Prob']}\n")
        f.write(f"- Mutation Probability: {best_config['Mutation Prob']}\n")
        f.write(f"- Average Fitness: {best_config['Avg Fitness']:.4f}\n")
        f.write(f"- Average Selected Features: {best_config['Avg Selected Features']:.2f}\n")

    # Find the best configuration
    best_idx = comparison_df['Avg Fitness'].idxmax()
    best_config = comparison_df.iloc[best_idx]

    # Format directory name appropriately (handling potential floating point values)
    pop_size = int(best_config['Population Size']) if best_config['Population Size'].is_integer() else best_config['Population Size']
    cross_prob = best_config['Crossover Prob']
    mut_prob = best_config['Mutation Prob']

    # Add a section to display selected features in the summary report
    with open(f"{result_dir}/selected_features_summary.txt", 'w') as f:
        f.write("Selected Features from Best Configuration\n")
        f.write("=======================================\n\n")
        f.write(f"Best Configuration:\n")
        f.write(f"- Population Size: {pop_size}\n")
        f.write(f"- Crossover Probability: {cross_prob}\n")
        f.write(f"- Mutation Probability: {mut_prob}\n\n")

        # Load the selected features from individual trials
        from genetic_algorithm import GeneticFeatureSelector

        # Run GA to get selected features directly
        try:
            # Re-run best configuration once to get features
            ga = GeneticFeatureSelector(
                X=X,
                y=y,
                population_size=int(best_config['Population Size']),
                crossover_prob=best_config['Crossover Prob'],
                mutation_prob=best_config['Mutation Prob'],
                random_state=int(os.getenv('SEED', 420))
            )
            ga.fit()
            selected_features = ga.get_selected_features()

            f.write("Selected Features:\n")
            for i, feature in enumerate(selected_features):
                f.write(f"{i+1}. {feature}\n")

            print(f"Selected features saved to {result_dir}/selected_features_summary.txt")

        except Exception as e:
            f.write(f"Error getting selected features: {str(e)}\n")
            f.write("To get selected features, try running the GA separately with the best configuration.\n")

if __name__ == "__main__":
    try:
        run_all_experiments()
        print("\nAll experiments completed successfully!")
    except KeyboardInterrupt:
        print("\nExperiments interrupted by user.")
    except Exception as e:
        print(f"An error occurred during experiments: {str(e)}")
    finally:
        # Clean up resources
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
