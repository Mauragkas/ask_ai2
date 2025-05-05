#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool
from dotenv import load_dotenv
from genetic_algorithm import GeneticFeatureSelector
import gc
import json

load_dotenv()

out_dir = "b_res"

def run_method_experiment(X, y, selection_method, crossover_method, n_trials=5, result_dir=out_dir+"method_comparison"):
    """Run an experiment with a specific selection and crossover method"""
    # Fixed configuration parameters
    population_size = 200
    crossover_prob = 0.9
    mutation_prob = 0.01
    generations = 300

    config_name = f"sel_{selection_method}_cross_{crossover_method}"
    experiment_dir = f"{result_dir}/{config_name}"
    os.makedirs(experiment_dir, exist_ok=True)

    print(f"\nRunning experiment: Selection={selection_method}, Crossover={crossover_method}")

    all_best_fitness = []
    all_selected_features_count = []
    all_selected_features = []
    all_evolution_curves = []

    for trial in range(n_trials):
        print(f"  Trial {trial+1}/{n_trials}")

        # Set seed for reproducibility but different for each trial
        seed = int(os.getenv('SEED', 420)) + trial

        # Create genetic selector with specified methods
        ga_selector = GeneticFeatureSelector(
            X=X,
            y=y,
            cv=5,
            population_size=population_size,
            generations=generations,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            n_elite=2,
            random_state=seed,
            selection_method=selection_method,
            crossover_method=crossover_method
        )

        try:
            # Run the genetic algorithm
            ga_selector.fit()

            # Store results
            selected_features = ga_selector.get_selected_features()
            all_selected_features.append(selected_features)
            all_selected_features_count.append(len(selected_features))
            all_best_fitness.append(ga_selector.best_fitness)
            all_evolution_curves.append({
                'best': ga_selector.best_fitness_per_gen,
                'avg': ga_selector.avg_fitness_per_gen
            })

            # Save individual trial evolution curve
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(ga_selector.best_fitness_per_gen)+1),
                     ga_selector.best_fitness_per_gen, 'b-', label='Best Fitness')
            plt.plot(range(1, len(ga_selector.avg_fitness_per_gen)+1),
                     ga_selector.avg_fitness_per_gen, 'r--', label='Average Fitness')
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.title(f'{selection_method} selection, {crossover_method} crossover - Trial {trial+1}')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{experiment_dir}/fitness_evolution_trial{trial+1}.png", dpi=300)
            plt.close()

        except Exception as e:
            print(f"Error in trial {trial+1}: {str(e)}")

    # Calculate average performance
    avg_best_fitness = np.mean(all_best_fitness)
    avg_features_selected = np.mean(all_selected_features_count)

    # Find common length for evolution curves
    min_len = min(len(curve['best']) for curve in all_evolution_curves)

    # Average evolution curves
    avg_best_curve = np.mean([curve['best'][:min_len] for curve in all_evolution_curves], axis=0)
    avg_avg_curve = np.mean([curve['avg'][:min_len] for curve in all_evolution_curves], axis=0)

    # Plot average evolution curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, min_len+1), avg_best_curve, 'b-', label='Best Fitness')
    plt.plot(range(1, min_len+1), avg_avg_curve, 'r--', label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title(f'Average Fitness Evolution\n{selection_method} selection, {crossover_method} crossover')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{experiment_dir}/avg_fitness_evolution.png", dpi=300)
    plt.close()

    # Save experiment summary
    results = {
        'parameters': {
            'selection_method': selection_method,
            'crossover_method': crossover_method,
            'population_size': population_size,
            'crossover_prob': crossover_prob,
            'mutation_prob': mutation_prob,
            'generations': generations,
            'n_trials': n_trials
        },
        'results': {
            'avg_best_fitness': float(avg_best_fitness),
            'avg_features_selected': float(avg_features_selected),
            'all_best_fitness': [float(f) for f in all_best_fitness],
            'all_selected_features_count': all_selected_features_count
        }
    }

    with open(f"{experiment_dir}/experiment_summary.json", 'w') as f:
        json.dump(results, f, indent=2)

    with open(f"{experiment_dir}/experiment_summary.txt", 'w') as f:
        f.write(f"GA Methods Experiment Summary\n")
        f.write(f"===========================\n\n")
        f.write(f"Methods:\n")
        f.write(f"- Selection Method: {selection_method}\n")
        f.write(f"- Crossover Method: {crossover_method}\n\n")
        f.write(f"Parameters:\n")
        f.write(f"- Population Size: {population_size}\n")
        f.write(f"- Crossover Probability: {crossover_prob}\n")
        f.write(f"- Mutation Probability: {mutation_prob}\n")
        f.write(f"- Maximum Generations: {generations}\n")
        f.write(f"- Number of Trials: {n_trials}\n\n")
        f.write(f"Results (averaged over {n_trials} trials):\n")
        f.write(f"- Average Best Fitness: {avg_best_fitness:.4f}\n")
        f.write(f"- Average Number of Selected Features: {avg_features_selected:.2f}\n\n")
        f.write(f"Individual Trial Results:\n")
        for i, (fitness, n_features) in enumerate(zip(all_best_fitness, all_selected_features_count)):
            f.write(f"Trial {i+1}: Fitness={fitness:.4f}, Selected Features={n_features}\n")

    return results

def plot_method_comparisons(all_results, result_dir):
    """Create comparison plots for different methods"""
    # Prepare data for plotting
    comparison_data = []
    for result in all_results:
        params = result['parameters']
        res = result['results']
        comparison_data.append({
            'Selection Method': params['selection_method'],
            'Crossover Method': params['crossover_method'],
            'Avg Fitness': res['avg_best_fitness'],
            'Avg Selected Features': res['avg_features_selected']
        })

    comparison_df = pd.DataFrame(comparison_data)

    # Save comparison table
    comparison_df.to_csv(f"{result_dir}/method_comparison.csv", index=False)

    # Plot 1: Bar chart of fitness by method combination
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(x='Crossover Method', y='Avg Fitness', hue='Selection Method', data=comparison_df)
    plt.title('Average Fitness by Selection and Crossover Methods')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{result_dir}/method_fitness_comparison.png", dpi=300)
    plt.close()

    # Plot 2: Bar chart of selected features by method combination
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(x='Crossover Method', y='Avg Selected Features', hue='Selection Method', data=comparison_df)
    plt.title('Average Selected Features by Selection and Crossover Methods')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{result_dir}/method_features_comparison.png", dpi=300)
    plt.close()

    # Plot 3: Scatter plot - fitness vs features
    plt.figure(figsize=(10, 8))
    for selection in comparison_df['Selection Method'].unique():
        subset = comparison_df[comparison_df['Selection Method'] == selection]
        plt.scatter(subset['Avg Fitness'], subset['Avg Selected Features'],
                    label=selection, s=100, alpha=0.7)

        # Add labels for crossover methods
        for _, row in subset.iterrows():
            plt.annotate(row['Crossover Method'],
                         (row['Avg Fitness'], row['Avg Selected Features']),
                         xytext=(5, 5), textcoords='offset points')

    plt.xlabel('Average Fitness')
    plt.ylabel('Average Selected Features')
    plt.title('Fitness vs. Selected Features by Method')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{result_dir}/method_scatter_comparison.png", dpi=300)
    plt.close()

    # Create summary report
    with open(f"{result_dir}/method_comparison_summary.txt", 'w') as f:
        f.write("GA Method Comparison Summary\n")
        f.write("===========================\n\n")
        f.write("Method Configurations and Results:\n\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")

        # Find best method combination
        best_idx = comparison_df['Avg Fitness'].idxmax()
        best_methods = comparison_df.iloc[best_idx]

        f.write("Best Method Combination:\n")
        f.write(f"- Selection Method: {best_methods['Selection Method']}\n")
        f.write(f"- Crossover Method: {best_methods['Crossover Method']}\n")
        f.write(f"- Average Fitness: {best_methods['Avg Fitness']:.4f}\n")
        f.write(f"- Average Selected Features: {best_methods['Avg Selected Features']:.2f}\n")

def run_experiment_worker(args):
    """Worker function for parallel processing"""
    X, y, selection_method, crossover_method, n_trials, result_dir = args
    try:
        return run_method_experiment(
            X=X,
            y=y,
            selection_method=selection_method,
            crossover_method=crossover_method,
            n_trials=n_trials,
            result_dir=result_dir
        )
    except Exception as e:
        print(f"Error with {selection_method} selection and {crossover_method} crossover: {str(e)}")
        return None

def run_all_method_experiments():
    """Run experiments for all combinations of selection and crossover methods"""
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
    result_dir = out_dir+"/method_comparison"
    os.makedirs(result_dir, exist_ok=True)

    # Define method combinations to test
    selection_methods = ['tournament', 'roulette']
    crossover_methods = ['single_point', 'two_point', 'uniform']

    # Prepare arguments for parallel processing
    experiment_args = []
    for selection_method in selection_methods:
        for crossover_method in crossover_methods:
            experiment_args.append((
                X, y, selection_method, crossover_method, 3, result_dir
            ))

    print(f"Running {len(experiment_args)} experiments in parallel...")

    # Run experiments in parallel
    with Pool() as pool:
        all_results = pool.map(run_experiment_worker, experiment_args)

    # Filter out None results from failed experiments
    all_results = [result for result in all_results if result is not None]

    # Create comparison plots
    if all_results:
        plot_method_comparisons(all_results, result_dir)
    else:
        print("All experiments failed. No results to compare.")

if __name__ == "__main__":
    try:
        run_all_method_experiments()
        print("\nAll method experiments completed successfully!")
    except KeyboardInterrupt:
        print("\nExperiments interrupted by user.")
    except Exception as e:
        print(f"An error occurred during experiments: {str(e)}")
    finally:
        # Clean up resources
        gc.collect()
