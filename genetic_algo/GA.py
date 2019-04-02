#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:14:31 2019

@author: cheshirecat12
"""
from copy import deepcopy

import numpy as np
from numpy.random import choice, randint


class GeneticAlgorithm():
    """Algorithm genetic."""

    def __init__(self, genes, population_size=10):
        self.genes = genes
        self.population_size = population_size

    @property
    def genes(self):
        return self.__genes

    @genes.setter
    def genes(self, val):
        if isinstance(val, dict):
            self.__genes = [v for v in val.values()]
        elif isinstance(val, list):
            self.__genes = val
        else:
            Exception("Wrong type of genes.")

    def init_population(self):
        self.population = np.array([np.array([choice(gene)
                                              for gene in self.genes])
                                    for _ in range(self.population_size)])

        return self.population

    def _crossover(self, fitness):
        fitness = regularization(fitness)

        res = deepcopy(self.population)

        for i in range(0, len(self.population), 2):
            # I hate myself... ^^'
            parents = deepcopy(self.population)

            idx = [choice([True, False])
                   for _ in range(len(self.genes))]

            idx_parent1, idx_parent2 = choice(self.population.shape[0],
                                              size=2,
                                              p=fitness)

            tmp = parents[idx_parent1][idx]
            parents[idx_parent1][idx] = parents[idx_parent2][idx]
            parents[idx_parent2][idx] = tmp

            res[i] = parents[idx_parent1][:]
            res[i+1] = parents[idx_parent2][:]

        self.population = res

    def _mutation(self):
        for i, _ in enumerate(self.population):
            idx_gene = randint(len(self.genes))
            self.population[i][idx_gene] = choice(self.genes[idx_gene])

    def next_population(self, fitness):
        fitness = regularization(fitness)

        parents_idx = choice(self.population.shape[0],
                             size=self.population_size,
                             p=fitness)

        self.population = self.population[parents_idx, :]

        self._crossover(fitness[parents_idx])
        self._mutation()

        return self.population

def regularization(fitness):
    """Regularize the fitness vector to sum to 1."""
    if all(val == 0 for val in fitness):
        return np.array([1/i for i in range(fitness.shape[0])])
    return fitness/sum(fitness)
