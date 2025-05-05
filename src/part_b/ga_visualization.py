#!/usr/bin/env python3
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_fitness_evolution(selector, save_path=None):
    """Plot the evolution of fitness across generations"""
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(selector.best_fitness_per_gen)+1), selector.best_fitness_per_gen, 'b-', label='Best Fitness')
    plt.plot(range(1, len(selector.avg_fitness_per_gen)+1), selector.avg_fitness_per_gen, 'r--', label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Combined accuracy and feature penalty)')
    plt.title(f'Fitness Evolution - {selector.config_name}')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(selector, save_path=None):
    """Plot feature importance"""
    if selector.feature_names is not None:
        feature_df = pd.DataFrame({
            'Feature': selector.feature_names,
            'Importance': selector.feature_importance,
            'Selected': selector.best_chromosome == 1
        })
    else:
        feature_df = pd.DataFrame({
            'Feature': [f"Feature {i+1}" for i in range(selector.n_features)],
            'Importance': selector.feature_importance,
            'Selected': selector.best_chromosome == 1
        })

    # Sort by importance and filter to show only top features
    feature_df = feature_df.sort_values('Importance', ascending=False).head(20)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_df,
               palette=[sns.color_palette()[0] if selected else 'gray' for selected in feature_df['Selected']])
    plt.title('Feature Importance (Top 20)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_experiment_comparisons(comparison_df, result_dir="b_res"):
    """Create comparison plots for different GA configurations"""
    plt.figure(figsize=(14, 10))

    # Plot 1: Population Size vs Fitness, colored by mutation prob, with markers by crossover
    plt.subplot(2, 1, 1)
    for mut_prob in comparison_df['Mutation Prob'].unique():
        for cross_prob in comparison_df['Crossover Prob'].unique():
            subset = comparison_df[(comparison_df['Mutation Prob'] == mut_prob) &
                                  (comparison_df['Crossover Prob'] == cross_prob)]
            if not subset.empty:
                marker = 'o' if cross_prob < 0.5 else ('^' if cross_prob < 0.8 else 's')
                plt.scatter(subset['Population Size'], subset['Avg Fitness'],
                          label=f'Mut={mut_prob}, Cross={cross_prob}',
                          marker=marker, s=100, alpha=0.7)

    plt.xlabel('Population Size')
    plt.ylabel('Average Fitness')
    plt.title('Effect of GA Parameters on Model Fitness')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot 2: Population Size vs Feature Count, colored by mutation prob, with markers by crossover
    plt.subplot(2, 1, 2)
    for mut_prob in comparison_df['Mutation Prob'].unique():
        for cross_prob in comparison_df['Crossover Prob'].unique():
            subset = comparison_df[(comparison_df['Mutation Prob'] == mut_prob) &
                                  (comparison_df['Crossover Prob'] == cross_prob)]
            if not subset.empty:
                marker = 'o' if cross_prob < 0.5 else ('^' if cross_prob < 0.8 else 's')
                plt.scatter(subset['Population Size'], subset['Avg Selected Features'],
                          label=f'Mut={mut_prob}, Cross={cross_prob}',
                          marker=marker, s=100, alpha=0.7)

    plt.xlabel('Population Size')
    plt.ylabel('Average Selected Features')
    plt.title('Effect of GA Parameters on Feature Selection')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(f"{result_dir}/parameter_comparison_combined.png", dpi=300, bbox_inches='tight')
    plt.close()
