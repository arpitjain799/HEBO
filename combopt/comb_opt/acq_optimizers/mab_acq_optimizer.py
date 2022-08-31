# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import copy
import warnings
from typing import Optional

import numpy as np
import torch
from torch.quasirandom import SobolEngine

from comb_opt.acq_funcs import AcqBase
from comb_opt.acq_optimizers import AcqOptimizerBase
from comb_opt.models import ModelBase
from comb_opt.search_space import SearchSpace
from comb_opt.trust_region import TrManagerBase
from comb_opt.utils.data_buffer import DataBuffer
from comb_opt.utils.dependant_rounding import DepRound
from comb_opt.utils.model_utils import add_hallucinations_and_retrain_model


class MabAcqOptimizer(AcqOptimizerBase):

    def __init__(self,
                 search_space: SearchSpace,
                 acq_func: AcqBase,
                 batch_size: int = 1,
                 max_n_iter: int = 200,
                 n_cand: int = 5000,
                 n_restarts: int = 5,
                 cont_optimizer: str = 'adam',
                 cont_lr: float = 1e-3,
                 cont_n_iter: int = 100,
                 dtype: torch.dtype = torch.float32,
                 ):

        assert search_space.num_dims == search_space.num_cont + search_space.num_nominal, \
            'The Multi-armed bandit acquisition optimizer only supports continuous and nominal variables.'

        assert n_cand >= n_restarts, \
            'The number of random candidates must be > number of points selected for gradient based optimisation'

        super(MabAcqOptimizer, self).__init__(search_space, dtype)

        self.acq_func = acq_func
        self.n_cats = [int(ub + 1) for ub in search_space.nominal_ub]
        self.n_cand = n_cand
        self.n_restarts = n_restarts
        self.cont_optimizer = cont_optimizer
        self.cont_lr = cont_lr
        self.cont_n_iter = cont_n_iter
        self.batch_size = batch_size

        # Algorithm initialisation
        if search_space.num_cont > 0:
            seed = np.random.randint(int(1e6))
            self.sobol_engine = SobolEngine(search_space.num_cont, scramble=True, seed=seed)

        best_ube = 2 * max_n_iter / 3  # Upper bound estimate

        self.gamma = []
        for n_cats in self.n_cats:
            if n_cats > batch_size:
                self.gamma.append(np.sqrt(n_cats * np.log(n_cats / batch_size) / (
                        (np.e - 1) * batch_size * best_ube)))
            else:
                self.gamma.append(np.sqrt(n_cats * np.log(n_cats) / ((np.e - 1) * best_ube)))

        self.weights = [np.ones(C) for C in self.n_cats]
        self.prob_dist = None

        self.inverse_mapping = [(self.search_space.cont_dims + self.search_space.nominal_dims).index(i) for i in
                                range(self.search_space.num_dims)]

    def update_mab_prob_dist(self):

        prob_dist = []

        for j in range(len(self.n_cats)):
            weights = self.weights[j]
            gamma = self.gamma[j]
            norm = float(sum(weights))
            prob_dist.append(list((1.0 - gamma) * (w / norm) + (gamma / len(weights)) for w in weights))

        self.prob_dist = prob_dist

    def optimize(self,
                 x: torch.Tensor,
                 n_suggestions: int,
                 x_observed: torch.Tensor,
                 model: ModelBase,
                 acq_func: AcqBase,
                 acq_evaluate_kwargs: dict,
                 tr_manager: Optional[TrManagerBase],
                 **kwargs
                 ) -> torch.Tensor:

        assert (self.n_restarts == 0 and self.n_cand >= n_suggestions) or (self.n_restarts >= n_suggestions)

        if tr_manager is not None:
            raise RuntimeError("MAB does not support TR for now")  # TODO: handle TR

        if self.batch_size != n_suggestions:
            warnings.warn('batch_size used for initialising the algorithm is not equal to n_suggestions received by' + \
                          ' the acquisition optimizer. If the batch size is known in advance, consider initialising' + \
                          ' the acquisition optimizer with the correct batch size for better performance.')

        x_next = torch.zeros((0, self.search_space.num_dims), dtype=self.dtype)

        if n_suggestions > 1:
            # create a local copy of the model
            model = copy.deepcopy(model)
        else:
            model = model

        # Update the probability distribution of each Multi-armed bandit
        self.update_mab_prob_dist()

        if self.search_space.num_nominal > 0 and self.search_space.num_cont > 0:

            x_nominal = self.sample_nominal(n_suggestions)

            x_nominal_unique, x_nominal_counts = torch.unique(x_nominal, return_counts=True, dim=0)

            for idx, curr_x_nominal in enumerate(x_nominal_unique):

                if len(x_next):
                    # Add the last point to the model and retrain it
                    add_hallucinations_and_retrain_model(model, x_next[-x_nominal_counts[idx - 1].item()])

                x_cont_ = self.optimize_x_cont(curr_x_nominal, x_nominal_counts[idx], model, acq_evaluate_kwargs)
                x_nominal_ = curr_x_nominal * torch.ones((x_nominal_counts[idx], curr_x_nominal.shape[0]))

                x_next = torch.cat((x_next, self.reconstruct_x(x_cont_, x_nominal_)))

        elif self.search_space.num_cont > 0:
            x_next = torch.cat((x_next, self.optimize_x_cont(torch.tensor([]), n_suggestions, acq_func)))

        elif self.search_space.num_nominal > 0:
            x_next = torch.cat((x_next, self.sample_nominal(n_suggestions)))
        return x_next

    def optimize_x_cont(self, x_nominal: torch.Tensor,
                        n_suggestions: int,
                        model: ModelBase,
                        acq_evaluate_kwargs: dict,
                        ):

        # Make a copy of the acquisition function if necessary to avoid changing original model parameters
        if n_suggestions > 1:
            model = copy.deepcopy(model)

        x_cont = torch.zeros((0, self.search_space.num_cont), dtype=self.dtype)

        for i in range(n_suggestions):

            if len(x_cont) > 0:
                add_hallucinations_and_retrain_model(model, self.reconstruct_x(x_cont[-1], x_nominal))

            # Sample x_cont
            x_cont_cand = self.sobol_engine.draw(self.n_cand)  # Note that this assumes x in [0, 1]

            x_cand = self.reconstruct_x(x_cont_cand, x_nominal * torch.ones((self.n_cand, x_nominal.shape[0])))

            # Evaluate all random samples
            with torch.no_grad():
                acq = self.acq_func(x_cand, model, **acq_evaluate_kwargs)

            if self.n_restarts > 0:

                x_cont_best = None
                best_acq = None

                x_local_cand = x_cand[acq.argsort()[:self.n_restarts]]

                for x_ in x_local_cand:

                    x_cont_, x_nominal_ = x_[self.search_space.cont_dims], x_[self.search_space.nominal_dims]
                    x_cont_.requires_grad_(True)

                    if self.cont_optimizer == 'adam':
                        optimizer = torch.optim.Adam([{"params": x_cont_}], lr=self.cont_lr)
                    elif self.cont_optimizer == 'sgd':
                        optimizer = torch.optim.SGD([{"params": x_cont_}], lr=self.cont_lr)
                    else:
                        raise NotImplementedError(f'optimiser {self.num_optimiser} is not implemented.')

                    for _ in range(self.cont_n_iter):
                        optimizer.zero_grad()
                        x_cand = self.reconstruct_x(x_cont_, x_nominal_)
                        acq_x = self.acq_func(x_cand, model, **acq_evaluate_kwargs)

                        try:
                            acq_x.backward()
                            optimizer.step()
                        except RuntimeError:
                            print('Exception occurred during backpropagation. NaN encountered?')
                            pass
                        with torch.no_grad():
                            x_cont_.data = torch.clip(x_cont_, min=0, max=1)

                    x_cont_.requires_grad_(False)

                    if best_acq is None or acq_x < best_acq:
                        best_acq = acq_x.item()
                        x_cont_best = x_cont_

            else:
                x_cont_best = x_cont_cand[acq.argsort()[0]]

            x_cont = torch.cat((x_cont, x_cont_best.unsqueeze(0)))

        return x_cont

    def sample_nominal(self, n_suggestions):

        x_nominal = np.zeros((n_suggestions, self.search_space.num_nominal))

        for j, num_cat in enumerate(self.n_cats):
            # draw a batch here
            if 1 < n_suggestions < num_cat:
                ht = DepRound(self.prob_dist[j], k=n_suggestions)
            else:
                ht = np.random.choice(num_cat, n_suggestions, p=self.prob_dist[j])

            # ht_batch_list size: len(self.C_list) x B
            x_nominal[:, j] = ht[:]

        return torch.tensor(x_nominal, dtype=self.dtype)

    def reconstruct_x(self, x_cont: torch.Tensor, x_nominal: torch.Tensor) -> torch.Tensor:
        if x_cont.ndim == x_nominal.ndim == 1:
            return torch.cat((x_cont, x_nominal))[self.inverse_mapping]
        else:
            return torch.cat((x_cont, x_nominal), dim=1)[:, self.inverse_mapping]

    def post_observe_method(self, x: torch.Tensor, y: torch.Tensor, data_buffer: DataBuffer, n_init: int, **kwargs):
        """
        Function used to update the weights of each of the multi-armed bandit agents.

        :param x:
        :param y:
        :param data_buffer:
        :param n_init:
        :param kwargs:
        :return:
        """
        if len(data_buffer) <= n_init:
            return

        x_observed, y_observed = data_buffer.x, data_buffer.y

        # Compute the MAB rewards for each of the suggested categories
        mab_rewards = torch.zeros((len(x), self.search_space.num_nominal), dtype=self.dtype)

        # Iterate over the batch
        for batch_idx in range(len(x)):
            x_nominal_next = x[batch_idx, self.search_space.nominal_dims]

            # Iterate over all categorical variables
            for dim_dix in range(self.search_space.num_nominal):
                indices = x_observed[:, self.search_space.nominal_dims][:, dim_dix] == x_nominal_next[dim_dix]

                # In MAB, we aim to maximise the reward, hence, take negative of bb values
                rewards = - y_observed[indices]

                if len(rewards) == 0:
                    reward = torch.tensor(0., dtype=self.dtype)
                else:
                    # Map rewards to range[-0.5, 0.5]
                    reward = 2 * (rewards.max() - (- y_observed).min()) / (
                            (- y_observed).max() - (-y_observed).min()) - 1.
                    # reward = rewards.max()

                mab_rewards[batch_idx, dim_dix] = reward

        # Update the probability distribution
        x_nominal = x[:, self.search_space.nominal_dims]

        for dim_dix in range(self.search_space.num_nominal):
            weights = self.weights[dim_dix]
            num_cats = self.n_cats[dim_dix]
            gamma = self.gamma[dim_dix]
            prob_dist = self.prob_dist[dim_dix]

            x_nominal = x_nominal.to(torch.long)
            reward = mab_rewards[:, dim_dix]
            nominal_vars = x_nominal[:, dim_dix]  # 1xB
            for ii, ht in enumerate(nominal_vars):
                Gt_ht_b = reward[ii]
                estimated_reward = 1.0 * Gt_ht_b / prob_dist[ht]
                # if ht not in self.S0:
                weights[ht] = (weights[ht] * np.exp(len(mab_rewards) * estimated_reward * gamma / num_cats)).clip(
                    min=1e-6, max=1e6)

            self.weights[dim_dix] = weights
