#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import random
import json
from genetic_algorithm import GeneticFeatureSelector
from ga_visualization import plot_fitness_evolution, plot_feature_importance
import gc
import torch
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

def run_ga_experiment(X, y, population_size, crossover_prob, mutation_prob, generations=300, n_trials=10, result_dir="b_res"):
    """Run GA experiment with specified parameters multiple times and average results"""

    config_name = f"pop{population_size}_cross{crossover_prob}_mut{mutation_prob}"
    experiment_dir = f"{result_dir}/{config_name}"
    os.makedirs(experiment_dir, exist_ok=True)

    if __debug__:
        print(f"\n{'='*80}")
        print(f"Running experiment with: Population={population_size}, Crossover={crossover_prob}, Mutation={mutation_prob}")
        print(f"{'='*80}\n")

    all_best_fitness = []
    all_selected_features_count = []
    all_selected_features = []
    all_evolution_curves = []
    all_generations_needed = []  # Track generations needed for each trial

    for trial in range(n_trials):
        if __debug__:
            print(f"Trial {trial+1}/{n_trials}")

        # Set seed for reproducibility but different for each trial
        seed = int(os.getenv('SEED', 420)) + trial

        # Create the genetic feature selector with experiment parameters
        ga_selector = GeneticFeatureSelector(
            X=X,
            y=y,
            cv=5,
            population_size=population_size,
            generations=generations,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            n_elite=2,
            random_state=seed
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
            all_generations_needed.append(ga_selector.best_gen_found)  # Store when best solution was found

            # Save individual trial results
            plot_fitness_evolution(ga_selector, save_path=f"{experiment_dir}/fitness_evolution_trial{trial+1}.png")

            # Print trial results only in debug mode
            if __debug__:
                print(f"Trial {trial+1} complete - Best fitness: {ga_selector.best_fitness:.4f}, Selected {len(selected_features)} features, Found at generation: {ga_selector.best_gen_found}")

        except Exception as e:
            print(f"Error in trial {trial+1}: {str(e)}")

    # Calculate average performance
    avg_best_fitness = np.mean(all_best_fitness)
    avg_features_selected = np.mean(all_selected_features_count)
    avg_generations_needed = np.mean(all_generations_needed)  # Calculate average generations needed

    # Find minimum length of evolution curves
    min_len = min(len(curve['best']) for curve in all_evolution_curves)
    # Truncate all curves to the minimum length before averaging
    avg_best_curve = np.mean([curve['best'][:min_len] for curve in all_evolution_curves], axis=0)
    avg_avg_curve = np.mean([curve['avg'][:min_len] for curve in all_evolution_curves], axis=0)

    # Plot average evolution curve
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(avg_best_curve)+1), avg_best_curve, 'b-', label='Best Fitness')
    plt.plot(range(1, len(avg_avg_curve)+1), avg_avg_curve, 'r--', label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title(f'Average Fitness Evolution - {config_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{experiment_dir}/avg_fitness_evolution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save experiment summary
    results = {
        'parameters': {
            'population_size': population_size,
            'crossover_prob': crossover_prob,
            'mutation_prob': mutation_prob,
            'generations': generations,
            'n_trials': n_trials
        },
        'results': {
            'avg_best_fitness': float(avg_best_fitness),
            'avg_features_selected': float(avg_features_selected),
            'avg_generations_needed': float(avg_generations_needed),  # Add this to results
            'all_best_fitness': [float(f) for f in all_best_fitness],
            'all_selected_features_count': all_selected_features_count,
            'selected_features': all_selected_features[-1]  # Save the last trial's selected features
        }
    }

    with open(f"{experiment_dir}/experiment_summary.json", 'w') as f:
        json.dump(results, f, indent=2)

    with open(f"{experiment_dir}/experiment_summary.txt", 'w') as f:
        f.write(f"GA Feature Selection Experiment Summary\n")
        f.write(f"====================================\n\n")
        f.write(f"Parameters:\n")
        f.write(f"- Population Size: {population_size}\n")
        f.write(f"- Crossover Probability: {crossover_prob}\n")
        f.write(f"- Mutation Probability: {mutation_prob}\n")
        f.write(f"- Maximum Generations: {generations}\n")
        f.write(f"- Number of Trials: {n_trials}\n\n")
        f.write(f"Results (averaged over {n_trials} trials):\n")
        f.write(f"- Average Best Fitness: {avg_best_fitness:.4f}\n")
        f.write(f"- Average Number of Selected Features: {avg_features_selected:.2f}\n")
        f.write(f"- Average Generations Needed: {avg_generations_needed:.2f}\n\n")  # Add this line
        f.write(f"Individual Trial Results:\n")
        for i, (fitness, n_features, gens) in enumerate(zip(all_best_fitness, all_selected_features_count, all_generations_needed)):
            f.write(f"Trial {i+1}: Fitness={fitness:.4f}, Selected Features={n_features}, Generations Needed={gens}\n")

    return results

def run_single_experiment(config_and_data_with_gpu):
    X, y, config, result_dir, gpu_id = config_and_data_with_gpu

    try:
        # Set the GPU for this process
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"Running experiment on GPU {gpu_id}: {config}")

        return run_ga_experiment(
            X=X,
            y=y,
            population_size=config['population_size'],
            crossover_prob=config['crossover_prob'],
            mutation_prob=config['mutation_prob'],
            generations=300,
            n_trials=3,
            result_dir=result_dir
        )
    except Exception as e:
        print(f"Error during experiment with config {config} on GPU {gpu_id}: {str(e)}")
        return None
