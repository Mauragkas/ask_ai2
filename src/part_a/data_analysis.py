#!/usr/bin/env python3
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def analyze_data(df, output_dir='a1_res'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Drop ID and doctor columns
    df = df.drop(['PatientID', 'DoctorInCharge'], axis=1)

    # Set dark theme
    plt.style.use('dark_background')
    sns.set_theme(style="darkgrid", rc={'axes.facecolor': '#2F2F2F',
                                       'figure.facecolor': '#1C1C1C',
                                       'grid.color': '#404040',
                                       'text.color': 'white',
                                       'axes.labelcolor': 'white',
                                       'xtick.color': 'white',
                                       'ytick.color': 'white'})
    sns.set_palette("Set2")

    # Create pairplot with all features
    plt.figure(figsize=(20, 20))
    g = sns.pairplot(df, hue='Diagnosis', palette={0: '#808080', 1: '#00ff00'},
                     diag_kind='kde', plot_kws={'alpha': 0.6})
    g.fig.patch.set_facecolor('#1C1C1C')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pairplot.png', bbox_inches='tight', dpi=100, facecolor='#1C1C1C')
    plt.close()

    # 7. Save Summary Statistics
    with open(f'{output_dir}/summary_statistics.txt', 'w') as f:
        f.write("Dataset Summary Statistics\n")
        f.write("=======================\n\n")
        f.write(f"Total Samples: {len(df)}\n")
        f.write(f"Total Features: {len(df.columns)-1}\n\n")

        f.write("Class Distribution:\n")
        f.write(df['Diagnosis'].value_counts().to_string())
        f.write("\n\n")

        f.write("Missing Values:\n")
        f.write(df.isnull().sum().to_string())

if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv('./data/alzheimers_disease_data.csv')

    # Run analysis
    analyze_data(df)
    print("Analysis complete! Check the data_analysis directory for results.")
