# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import pandas as pd
import torch

from comb_opt.optimizers.optimizer_base import OptimizerBase
from comb_opt.search_space import SearchSpace


class GeneticAlgorithm(OptimizerBase):

    @property
    def name(self) -> str:
        return 'Genetic Algorithm'

    def __init__(self,
                 search_space: SearchSpace,
                 pop_size: int = 40,
                 num_parents: int = 20,
                 num_elite: int = 10,
                 store_observations: bool = True,
                 allow_repeating_suggestions: bool = False,
                 dtype: torch.dtype = torch.float32,
                 ):

        assert search_space.num_nominal + search_space.num_ordinal == search_space.num_dims, \
            'Genetic Algorithm currently supports only nominal and ordinal variables'

        super(GeneticAlgorithm, self).__init__(search_space, dtype)

        self.pop_size = pop_size
        self.num_parents = num_parents
        self.num_elite = num_elite
        self.store_observations = store_observations
        self.allow_repeating_suggestions = allow_repeating_suggestions

        # Ensure that the number of elite samples is even
        if self.num_elite % 2 != 0:
            self.num_elite += 1

        assert self.num_parents >= self.num_elite, \
            "\n The number of parents must be greater than the number of elite samples"

        # Storage for the population
        self.x_pop = torch.zeros((0, self.search_space.num_dims), dtype=self.dtype)
        self.y_pop = torch.zeros((0, 1), dtype=self.dtype)

        # Initialising variables that will store elite samples
        self.x_elite = None
        self.y_elite = None

        self.x_queue = search_space.sample(self.pop_size)

        self.map_to_canonical = self.search_space.nominal_dims + self.search_space.ordinal_dims
        self.map_to_original = [self.map_to_canonical.index(i) for i in range(len(self.map_to_canonical))]

        self.lb = self.search_space.nominal_lb + self.search_space.ordinal_lb
        self.ub = self.search_space.nominal_ub + self.search_space.ordinal_ub

    def initialize(self, x: pd.DataFrame, y: np.ndarray):
        assert len(x) < self.pop_size, 'Initialise currently does not support len(x) > population_size'
        assert y.ndim == 2
        assert y.shape[1] == 1
        assert x.shape[0] == y.shape[0]
        assert x.shape[1] == self.search_space.num_dims

        x = self.search_space.transform(x)

        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=self.dtype)

        # Add data to all previously observed data
        if self.store_observations or (not self.allow_repeating_suggestions):
            self.data_buffer.append(x.clone(), y.clone())

        # Add data to current trust region data
        self.x_pop = torch.cat((self.x_pop, x.clone()), axis=0)
        self.y_pop = torch.cat((self.y_pop, y.clone()), axis=0)

        # update best fx
        best_idx = y.flatten().argmin()
        best_y = y[best_idx, 0].item()

        if self.best_y is None or best_y < self.best_y:
            self.best_y = best_y
            self._best_x = x[best_idx: best_idx + 1]

    def set_x_init(self, x: pd.DataFrame):
        self.x_queue = x

    def restart(self):
        self._restart()

        self.x_pop = torch.zeros((0, self.search_space.num_dims), dtype=self.dtype)
        self.y_pop = torch.zeros((0, 1), dtype=self.dtype)

        self.x_queue = self.search_space.sample(self.pop_size)

    def suggest(self, n_suggestions: int = 1) -> pd.DataFrame:
        assert n_suggestions <= self.pop_size

        idx = 0
        n_remaining = n_suggestions
        x_next = pd.DataFrame(index=range(n_suggestions), columns=self.search_space.df_col_names, dtype=float)

        # _x_init contains points from algorithm or trust region initialisation
        if n_remaining and len(self.x_queue):
            n = min(n_remaining, len(self.x_queue))
            x_next.iloc[idx: idx + n] = self.x_queue.iloc[idx: idx + n]
            self.x_queue = self.x_queue.drop([i for i in range(idx, idx + n)]).reset_index(drop=True)

            idx += n
            n_remaining -= n

        while n_remaining:
            self._generate_new_population()

            n = min(n_remaining, len(self.x_queue))
            x_next.iloc[idx: idx + n] = self.x_queue.iloc[idx: idx + n]
            self.x_queue = self.x_queue.drop([i for i in range(idx, idx + n)]).reset_index(drop=True)

            idx += n
            n_remaining -= n

        return x_next

    def observe(self, x: pd.DataFrame, y: np.ndarray):

        x = self.search_space.transform(x)

        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=self.dtype)

        assert len(x) == len(y)

        # Add data to all previously observed data
        if self.store_observations or (not self.allow_repeating_suggestions):
            self.data_buffer.append(x, y)

        # Add data to current population
        self.x_pop = torch.cat((self.x_pop, x.clone()), axis=0)
        self.y_pop = torch.cat((self.y_pop, y.clone()), axis=0)

        # update best fx
        if self.best_y is None:
            idx = y.flatten().argmin()
            self.best_y = y[idx, 0].item()
            self._best_x = x[idx: idx + 1]

        else:
            idx = y.flatten().argmin()
            y_ = y[idx, 0].item()

            if y_ < self.best_y:
                self.best_y = y_
                self._best_x = x[idx: idx + 1]

    def _generate_new_population(self):

        # Sort the current population
        indices = self.y_pop.flatten().argsort()
        x_sorted = self.x_pop[indices]
        y_sorted = self.y_pop[indices].flatten()

        # Normalise the objective function
        min_y = y_sorted[0]
        if min_y < 0:
            norm_y = y_sorted + abs(min_y)

        else:
            norm_y = y_sorted.clone()

        max_y = norm_y.max()
        norm_y = max_y - norm_y + 1

        # Calculate probability
        sum_norm_y = norm_y.sum()
        prob = norm_y / sum_norm_y
        cum_prob = prob.cumsum(dim=0)

        if (self.x_elite is None) and (self.y_elite is None):
            self.x_elite = x_sorted[:self.num_elite].clone()
            self.y_elite = y_sorted[:self.num_elite].clone().view(-1, 1)

        else:
            x_elite = torch.cat((self.x_elite.clone(), x_sorted[:self.num_elite].clone()))
            y_elite = torch.cat((self.y_elite.clone(), y_sorted[:self.num_elite].clone().view(-1, 1)))
            indices = np.argsort(y_elite.flatten())
            self.x_elite = x_elite[indices[:self.num_elite]]
            self.y_elite = y_elite[indices[:self.num_elite]]

        # Select parents
        parents = torch.full((self.num_parents, self.search_space.num_dims), fill_value=torch.nan, dtype=self.dtype)

        # First, append the best performing samples to the list of parents
        parents[:self.num_elite] = self.x_elite

        # Then append random samples to the list of parents. The probability of a sample being picked is
        # proportional to the fitness of a sample
        for k in range(self.num_elite, self.num_parents):
            index = np.searchsorted(cum_prob, np.random.random())
            parents[k] = x_sorted[index].clone()

        # New population
        pop = torch.full((self.pop_size, self.search_space.num_dims), fill_value=torch.nan, dtype=self.dtype)

        # Second, perform crossover with the previously determined subset of all the parents
        # for k in range(self.num_elite, self.population_size, 2):
        for k in range(0, self.pop_size, 2):
            r1 = np.random.randint(0, self.num_parents)
            r2 = np.random.randint(0, self.num_parents)
            pvar1 = parents[r1].clone()
            pvar2 = parents[r2].clone()

            # Constraint satisfaction with rejection sampling
            # constraints_satisfied = False
            # while not constraints_satisfied:
            ch1, ch2 = self._crossover(pvar1, pvar2)
            ch1, ch2 = ch1.unsqueeze(0), ch2.unsqueeze(0)

            # Mutate child 1
            done = False
            counter = 0
            if not self.allow_repeating_suggestions:
                x_observed = self.data_buffer.x
            while not done:
                _ch1 = self._mutate(ch1)
                # Check if sample is already present in pop
                if torch.logical_not((_ch1 == pop).all(axis=1)).all():
                    # Check if the sample was observed before
                    if not self.allow_repeating_suggestions:
                        if torch.logical_not((_ch1 == x_observed).all(axis=1)).all():
                            done = True
                    else:
                        if torch.logical_not((_ch1 == self.x_elite).all(axis=1)).all():
                            done = True
                    counter += 1

                    # If its not possible to generate a sample that has not been observed before, perform the crossover again
                    if not done and counter == 100:
                        r1 = np.random.randint(0, self.num_parents)
                        r2 = np.random.randint(0, self.num_parents)
                        pvar1 = parents[r1].clone()
                        pvar2 = parents[r2].clone()
                        ch1, ch2 = self._crossover(pvar1, pvar2)
                        ch1, ch2 = ch1.unsqueeze(0), ch2.unsqueeze(0)
                        counter = 0

            # Mutate child 2
            done = False
            counter = 0
            while not done:
                _ch2 = self._mutate(ch2)
                # Check if sample is already present in X_queue or in X_elites
                # Check if the sample was observed before
                if not self.allow_repeating_suggestions:
                    if torch.logical_not((_ch2 == x_observed).all(axis=1)).all():
                        done = True
                else:
                    if torch.logical_not((_ch2 == self.x_elite).all(axis=1)).all():
                        done = True
                counter += 1

                # If its not possible to generate a sample that has not been observed before, perform the crossover again
                if not done and counter == 100:
                    r1 = np.random.randint(0, self.num_parents)
                    r2 = np.random.randint(0, self.num_parents)
                    pvar1 = parents[r1].clone()
                    pvar2 = parents[r2].clone()
                    _, ch2 = self._crossover(pvar1, pvar2)
                    ch2 = ch2.unsqueeze(0)
                    counter = 0

            # constraints_satisfied = check_constraint_satisfaction_batch(np.array([ch1, ch2])).all()

            pop[k] = _ch1.clone()
            pop[k + 1] = _ch2.clone()

        self.x_queue = self.search_space.inverse_transform(pop)

        self.x_pop = torch.zeros((0, self.search_space.num_dims), dtype=self.dtype)
        self.y_pop = torch.zeros((0, 1), dtype=self.dtype)

        return

    def _crossover(self, x1: torch.Tensor, x2: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        assert self.search_space.num_ordinal + self.search_space.num_nominal == self.search_space.num_dims, \
            'Current crossover can\'t handle permutations'

        x1_ = x1.clone()
        x2_ = x2.clone()

        # starts from 1 and end at num_dims - 1 to always perform a crossover
        idx = np.random.randint(low=1, high=self.search_space.num_dims - 1)

        x1_[:idx] = x2[:idx]
        x2_[:idx] = x1[:idx]

        return x1_, x2_

    def _mutate(self, x: torch.Tensor) -> torch.Tensor:
        assert self.search_space.num_ordinal + self.search_space.num_nominal == self.search_space.num_dims, \
            'Current mutate can\'t handle permutations'

        x_ = x.clone()[:, self.map_to_canonical]
        for i in range(len(x)):
            idx = np.random.randint(low=0, high=self.search_space.num_dims)
            categories = np.array([j for j in range(int(self.lb[idx]), int(self.ub[idx])) if j != x[i, idx]])
            x_[i, idx] = np.random.choice(categories)

        x_ = x_[:, self.map_to_original]

        return x_
