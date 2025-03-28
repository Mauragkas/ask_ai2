import os
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(train_losses, val_losses, title, folder='a2_res'):
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.figure(figsize=(12, 7))
    plt.plot(train_losses, linewidth=2, label='Training Loss')
    plt.plot(val_losses, linewidth=2, label='Validation Loss')
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.ylim(bottom=0)
    plt.legend(fontsize=10)
    plt.tight_layout()
    if not os.path.exists(f'./{folder}'):
        os.makedirs(f'./{folder}')
    plt.savefig(f'./{folder}/{title}.png', dpi=200, bbox_inches='tight')
    plt.close()

def plot_results(results_df, folder='a2_res'):
    metrics = ['ce_loss', 'accuracy']
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Calculate mean values for each combination of hidden_size and activation
    mean_results = results_df.groupby(['hidden_size', 'activation'])[metrics].mean()

    colors = {'relu': sns.color_palette("deep")[0],
              'tanh': sns.color_palette("deep")[1],
              'silu': sns.color_palette("deep")[2]}

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Create lines for each activation function
        for activation in ['relu', 'tanh', 'silu']:
            data = mean_results.xs(activation, level='activation')[metric]
            color = colors[activation]

            # Plot scatter points
            ax.plot(data.index, data.values, marker='o', label=activation, linewidth=2, markersize=8, color=color)

            # Add trend line
            z = np.polyfit(data.index, data.values, 1)
            p = np.poly1d(z)
            ax.plot(data.index, p(data.index), '--', alpha=0.5,
                   label=f'{activation} trend', color=color)

        ax.set_xlabel('Hidden Layer Size', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f'{metric.upper()} vs Hidden Layer Size', fontsize=14)
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig(f'./{folder}/performance_metrics.png', dpi=200, bbox_inches='tight')
    plt.close()

def plot_momentum_results(results_df, folder='a3_res'):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    metric = 'accuracy'

    # Mean heatmap
    pivot_mean = results_df.pivot_table(
        values=metric,
        index='momentum',
        columns='learning_rate',
        aggfunc='mean'
    )

    # Standard deviation heatmap
    pivot_std = results_df.pivot_table(
        values=metric,
        index='momentum',
        columns='learning_rate',
        aggfunc='std'
    )

    # Plot mean
    sns.heatmap(pivot_mean, annot=True, fmt='.3f', cmap='YlGn', ax=axes[0])
    axes[0].set_title(f'{metric.upper()} Mean', fontsize=12)

    # Plot standard deviation
    sns.heatmap(pivot_std, annot=True, fmt='.3f', cmap='Reds', ax=axes[1])
    axes[1].set_title(f'{metric.upper()} Std Dev', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'./{folder}/momentum_results_heatmap.png', dpi=200, bbox_inches='tight')
    plt.close()

def plot_regularization_results(results_df, folder='a4_res'):
    # Plot comparison of L1 and L2 regularization results
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    metrics = ['accuracy', 'ce_loss']

    # Set a fun color palette
    colors = ['#FFA500', '#4169E1']  # Orange and Royal Blue
    sns.set_palette(colors)

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Create grouped bar plot with custom colors
        sns.barplot(data=results_df, x='reg_factor', y=metric, hue='reg_type', ax=ax)

        ax.set_title(f'{metric.upper()} vs Regularization Factor', fontsize=12)
        ax.set_xlabel('Regularization Factor', fontsize=10)
        ax.set_ylabel(metric.upper(), fontsize=10)
        # Fix for ticklabels warning
        ticks = ax.get_xticks()
        ax.set_xticks(ticks)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    plt.suptitle('Comparison of L1 and L2 Regularization',
                fontsize=14,
                y=1.02)
    plt.tight_layout()

    if not os.path.exists(f'./{folder}'):
        os.makedirs(f'./{folder}')
    plt.savefig(f'./{folder}/regularization_comparison.png',
                dpi=200,
                bbox_inches='tight')
    plt.close()

def plot_architecture_comparison(results_df, folder='a5_res'):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    metrics = ['accuracy', 'ce_loss']

    # Set vibrant seaborn color palette
    sns.set_palette("husl", 4)

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Calculate mean and standard error for each architecture
        stats = results_df.groupby('architecture')[metric].agg(['mean', 'std']).reset_index()

        # Create bar plot
        sns.barplot(data=stats, x='architecture', y='mean', ax=ax)

        # Add error bars
        ax.errorbar(
            x=range(len(stats)),
            y=stats['mean'],
            yerr=stats['std'],
            fmt='none',
            color='black',
            capsize=5
        )

        ax.set_title(f'{metric.upper()} by Architecture', fontsize=12)
        ax.set_xlabel('Architecture', fontsize=10)
        ax.set_ylabel(metric.upper(), fontsize=10)

        # Fix ticklabels warning by setting ticks explicitly
        ticks = range(len(stats))
        ax.set_xticks(ticks)
        ax.set_xticklabels(stats['architecture'], rotation=45, ha='right')

    plt.suptitle('Performance Comparison of Different Architectures',
                fontsize=14,
                y=1.02)
    plt.tight_layout()

    if not os.path.exists(f'./{folder}'):
        os.makedirs(f'./{folder}')
    plt.savefig(f'./{folder}/architecture_comparison.png',
                dpi=200,
                bbox_inches='tight')
    plt.close()
