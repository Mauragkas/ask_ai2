#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
from dotenv import load_dotenv
import json

load_dotenv()

class GeneticFeatureSelector:
    def __init__(self, X, y, estimator=None, cv=5, population_size=50, generations=300,
                 crossover_prob=0.8, mutation_prob=0.1, n_elite=2, random_state=None,
                 selection_method='tournament', crossover_method='single_point'):
        """Initialize the Genetic Algorithm for feature selection."""
        self.X = X
        self.y = y
        self.n_features = X.shape[1]
        self.feature_names = X.columns if isinstance(X, pd.DataFrame) else None

        # Set default estimator if not provided
        if estimator is not None:
            self.estimator = estimator
        else:
            # Create a pipeline with scaling and increased max_iter
            self.estimator = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(max_iter=10000, solver='liblinear'))
            ])

        self.cv = cv
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.n_elite = n_elite

        # Add parameters for selection and crossover methods
        self.selection_method = selection_method  # 'tournament' or 'roulette'
        self.crossover_method = crossover_method  # 'single_point', 'two_point', or 'uniform'

        # Set random state
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

        # Results tracking
        self.best_fitness_per_gen = []
        self.avg_fitness_per_gen = []
        self.best_chromosome = None
        self.best_fitness = -np.inf
        self.feature_importance = np.zeros(self.n_features)

        # Termination criteria
        self.early_stopping_generations = 10  # Stop if no improvement for this many generations
        self.min_improvement_threshold = 0.01

        # For saving results
        self.config_name = f"pop{population_size}_cross{crossover_prob}_mut{mutation_prob}_sel{selection_method}_xover{crossover_method}"

    def _initialize_population(self):
        """Initialize random population of chromosomes (feature masks)"""
        population = []
        for _ in range(self.population_size):
            # Each chromosome is a binary vector where 1=feature selected, 0=feature not selected
            chromosome = np.random.randint(0, 2, self.n_features)
            # Ensure at least one feature is selected
            if np.sum(chromosome) == 0:
                chromosome[np.random.randint(0, self.n_features)] = 1
            population.append(chromosome)
        return population

    def _evaluate_fitness(self, chromosome):
            """Evaluate the fitness of a chromosome using cross-validation"""
            # If no features selected, return very low fitness
            if np.sum(chromosome) == 0:
                return -np.inf

            # Calculate feature penalty (fewer features is better)
            n_selected_features = np.sum(chromosome)
            feature_penalty = n_selected_features / self.n_features  # Normalize to [0, 1]

            # Select features based on chromosome
            X_selected = self.X.iloc[:, chromosome == 1] if isinstance(self.X, pd.DataFrame) else self.X[:, chromosome == 1]

            try:
                # Evaluate using cross-validation
                cv_scores = cross_val_score(self.estimator, X_selected, self.y,
                                          cv=self.cv, scoring='accuracy', error_score=-1)

                # Calculate accuracy score
                accuracy = np.mean(cv_scores)

                # Combine accuracy and feature penalty (weight accuracy more)
                # We want high accuracy (close to 1) and low feature count (low penalty)
                fitness = 0.95 * accuracy - 0.05 * feature_penalty

                return fitness
                # return np.mean(cv_scores)
            except Exception as e:
                print(f"Error in fitness evaluation: {str(e)}")
                return -np.inf

    def _tournament_selection(self, population, fitnesses, tournament_size=3):
        """Select an individual using tournament selection"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        return population[tournament_indices[np.argmax(tournament_fitnesses)]].copy()

    def _roulette_selection(self, population, fitnesses):
        """Select an individual using roulette wheel selection"""
        # Adjust fitness values to ensure all are positive
        min_fitness = min(fitnesses)
        adjusted_fitnesses = [f - min_fitness + 0.1 if min_fitness < 0 else f + 0.1 for f in fitnesses]

        # Calculate selection probabilities
        total_fitness = sum(adjusted_fitnesses)
        selection_probs = [f/total_fitness for f in adjusted_fitnesses]

        # Select based on probabilities
        selected_idx = np.random.choice(len(population), p=selection_probs)
        return population[selected_idx].copy()

    def _crossover(self, parent1, parent2):
        """Perform crossover between two parents using the specified method"""
        if random.random() < self.crossover_prob:
            if self.crossover_method == 'single_point':
                return self._single_point_crossover(parent1, parent2)
            elif self.crossover_method == 'two_point':
                return self._two_point_crossover(parent1, parent2)
            elif self.crossover_method == 'uniform':
                return self._uniform_crossover(parent1, parent2)
            else:
                return self._single_point_crossover(parent1, parent2)  # Default
        else:
            return parent1.copy(), parent2.copy()

    def _single_point_crossover(self, parent1, parent2):
        """Single point crossover"""
        crossover_point = random.randint(1, self.n_features - 1)
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])

        # Ensure at least one feature is selected
        if np.sum(child1) == 0:
            child1[np.random.randint(0, self.n_features)] = 1
        if np.sum(child2) == 0:
            child2[np.random.randint(0, self.n_features)] = 1

        return child1, child2

    def _two_point_crossover(self, parent1, parent2):
        """Two point crossover"""
        # Ensure point1 < point2
        point1, point2 = sorted(random.sample(range(1, self.n_features), 2))

        child1 = np.concatenate([parent1[:point1], parent2[point1:point2], parent1[point2:]])
        child2 = np.concatenate([parent2[:point1], parent1[point1:point2], parent2[point2:]])

        # Ensure at least one feature is selected
        if np.sum(child1) == 0:
            child1[np.random.randint(0, self.n_features)] = 1
        if np.sum(child2) == 0:
            child2[np.random.randint(0, self.n_features)] = 1

        return child1, child2

    def _uniform_crossover(self, parent1, parent2):
        """Uniform crossover"""
        # Create a random mask
        mask = np.random.randint(0, 2, self.n_features)

        # Apply mask to create children
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)

        # Ensure at least one feature is selected
        if np.sum(child1) == 0:
            child1[np.random.randint(0, self.n_features)] = 1
        if np.sum(child2) == 0:
            child2[np.random.randint(0, self.n_features)] = 1

        return child1, child2

    def _mutate(self, chromosome):
        """Apply mutation to a chromosome"""
        for i in range(self.n_features):
            if random.random() < self.mutation_prob:
                chromosome[i] = 1 - chromosome[i]  # Flip bit

        # Ensure at least one feature is selected
        if np.sum(chromosome) == 0:
            chromosome[np.random.randint(0, self.n_features)] = 1

        return chromosome

    def fit(self):
        """Run the genetic algorithm to find the best feature subset"""
        # Initialize population
        population = self._initialize_population()

        # Track generations without improvement
        generations_no_improvement = 0
        last_best_fitness = -np.inf

        for generation in range(self.generations):
            # Evaluate fitness for each individual
            fitnesses = [self._evaluate_fitness(chromosome) for chromosome in population]

            # Track best solution overall
            max_fitness_idx = np.argmax(fitnesses)
            if fitnesses[max_fitness_idx] > self.best_fitness:
                self.best_fitness = fitnesses[max_fitness_idx]
                self.best_chromosome = population[max_fitness_idx].copy()

            # Track statistics
            self.best_fitness_per_gen.append(np.max(fitnesses))
            self.avg_fitness_per_gen.append(np.mean(fitnesses))

            # Print progress only in debug mode
            if __debug__:
                n_selected = np.sum(self.best_chromosome) if self.best_chromosome is not None else 0
                print(f"Generation {generation+1}/{self.generations}, Best fitness: {self.best_fitness_per_gen[-1]:.4f}, "
                      f"Avg fitness: {self.avg_fitness_per_gen[-1]:.4f}, Features selected: {n_selected}/{self.n_features}")

            # Check early stopping conditions
            improvement = 0
            if last_best_fitness > -np.inf:
                if self.best_fitness_per_gen[-1] > last_best_fitness:
                    improvement = (self.best_fitness_per_gen[-1] - last_best_fitness) / abs(last_best_fitness)

            if improvement < self.min_improvement_threshold:
                generations_no_improvement += 1
            else:
                generations_no_improvement = 0

            last_best_fitness = self.best_fitness_per_gen[-1]

            # Check if we should stop early
            if generations_no_improvement >= self.early_stopping_generations:
                if __debug__:
                    print(f"Early stopping at generation {generation+1} - no significant improvement for {self.early_stopping_generations} generations")
                break

            # Check if we're at the last generation
            if generation == self.generations - 1:
                break

            # Create new population
            new_population = []

            # Elitism: keep the best individuals
            sorted_indices = np.argsort(fitnesses)[::-1]
            sorted_population = [population[i].copy() for i in sorted_indices]
            new_population.extend([sorted_population[i].copy() for i in range(self.n_elite)])

            # Create the rest of the population
            while len(new_population) < self.population_size:
                # Use the selected selection method
                if hasattr(self, 'selection_method') and self.selection_method == 'tournament':
                    parent1 = self._tournament_selection(population, fitnesses)
                    parent2 = self._tournament_selection(population, fitnesses)
                elif hasattr(self, 'selection_method') and self.selection_method == 'roulette':
                    parent1 = self._roulette_selection(population, fitnesses)
                    parent2 = self._roulette_selection(population, fitnesses)
                else:
                    # Default to tournament
                    parent1 = self._tournament_selection(population, fitnesses)
                    parent2 = self._tournament_selection(population, fitnesses)

                child1, child2 = self._crossover(parent1, parent2)

                child1 = self._mutate(child1)
                child2 = self._mutate(child2)

                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            # Update population
            population = new_population

            # Update feature importance
            for chromosome, fitness in zip(population, fitnesses):
                if fitness > 0:  # Only consider positive fitness
                    self.feature_importance += chromosome * fitness

        # Normalize feature importance
        total_importance = np.sum(self.feature_importance)
        if total_importance > 0:
            self.feature_importance = self.feature_importance / total_importance

        return self

    def transform(self, X):
        """Transform X by selecting the best features"""
        if self.best_chromosome is None:
            raise ValueError("You must call fit() before transform()")

        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.best_chromosome == 1]
        else:
            return X[:, self.best_chromosome == 1]

    def fit_transform(self, X=None, y=None):
        """Fit and transform in one step"""
        if X is None:
            X = self.X
        if y is None:
            y = self.y

        return self.fit().transform(X)

    def get_selected_features(self):
        """Return the names of the selected features"""
        if self.best_chromosome is None:
            raise ValueError("You must call fit() before get_selected_features()")

        if self.feature_names is not None:
            return self.feature_names[self.best_chromosome == 1].tolist()
        else:
            return np.where(self.best_chromosome == 1)[0].tolist()
