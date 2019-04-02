#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 17:07:27 2019

@author: cheshirecat12
"""
from itertools import product
from string import ascii_lowercase as letters
from GA import GeneticAlgorithm

import numpy as np


def accuracy(word1, word2):
    return sum(1 for i, _ in enumerate(word1) if word1[i] == word2[i])

def func1():
    pass

def func2():
    pass

def main():
    parameters = {"lr": [10**-i*j for i, j in product(range(3, 6), [1, 5])],
                  "dropout": np.arange(0.0, 0.55, .05),
                  "weight_decay": [10**-i*j
                                   for i, j in product(range(4, 6), [1, 5])],
                  "optimizer": [func1, func2],
                  "nb_layers": list(range(2, 21)),
                  "nb_nodes": [2**i for i in range(1, 11)],
                  "batch_size": [2**i for i in range(4, 10)]}
    # 601920
    word = "qwertzu"

#    parameters = [list(letters) for _ in range(len(word))]

    genetic_algo = GeneticAlgorithm(parameters, 40)
    population = genetic_algo.init_population()
    print(population)
    epochs = 100
    final_counter = 0

    for _ in range(epochs):
        counter = 0
        run = True
        while run:
            accuracies = np.zeros((genetic_algo.population_size, ))

            for i, chromosome in enumerate(population):
                accuracies[i] = accuracy(word, chromosome)

                if accuracies[i] == len(word):
                    res = "".join(chromosome)
                    print(f'{res} in {counter} generations')
                    final_counter += counter
                    run = False

            counter += 1
            population = genetic_algo.next_population(accuracies)

    print(f'Mean generation: {final_counter/epochs}')



if __name__ == "__main__":
    main()
