# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import copy
import warnings
from typing import Optional, List

import gpytorch
import numpy as np
import torch
from gpytorch.constraints import Interval, GreaterThan
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.kernels import Kernel, MultitaskKernel
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.models import ExactGP
from gpytorch.priors import Prior, LogNormalPrior
from gpytorch.utils.errors import NotPSDError

from comb_opt.models.model_base import ModelBase
from comb_opt.search_space import SearchSpace
from comb_opt.utils.training_utils import subsample_training_data, remove_repeating_samples


class ExactGPModel(ModelBase, torch.nn.Module):
    supports_cuda = True
    support_grad = True
    support_multi_output = True

    def __init__(self,
                 search_space: SearchSpace,
                 num_out: int, kernel: Kernel,
                 noise_prior: Optional[Prior] = None,
                 noise_constr: Optional[Interval] = None,
                 noise_lb: float = 1e-5,
                 pred_likelihood: bool = True,
                 lr: float = 3e-3,
                 num_epochs: int = 100,
                 optimizer: str = 'adam',
                 max_cholesky_size: int = 2000,
                 max_training_dataset_size: int = 1000,
                 max_batch_size: int = 1000,
                 verbose: bool = False,
                 print_every: int = 10,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device('cpu'),
                 ):

        super(ExactGPModel, self).__init__(search_space, num_out, dtype, device)

        self.kernel = copy.deepcopy(kernel)
        self.lr = lr
        self.num_epochs = num_epochs
        self.optimiser = optimizer
        self.max_cholesky_size = max_cholesky_size
        self.max_training_dataset_size = max_training_dataset_size
        self.max_batch_size = max_batch_size
        self.verbose = verbose
        self.print_every = print_every

        if noise_prior is None:
            noise_prior = LogNormalPrior(-4.63, 0.5)
        else:
            assert isinstance(noise_prior, Prior)

        if noise_constr is None:
            assert noise_lb is not None
            self.noise_lb = noise_lb
            noise_constr = GreaterThan(noise_lb)
        else:
            assert isinstance(noise_constr, Interval)

        # Model settings
        self.pred_likelihood = pred_likelihood

        if self.num_out == 1:
            self.likelihood = GaussianLikelihood(noise_constraint=noise_constr, noise_prior=noise_prior)
        else:
            self.likelihood = MultitaskGaussianLikelihood(num_tasks=self.num_out, noise_constraint=noise_constr,
                                                          noise_prior=noise_prior)

        self.y_mean = torch.tensor(0, dtype=self.dtype, device=self.device)
        self.y_std = torch.tensor(1.0, dtype=self.dtype, device=self.device)

        self.gp = None

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> List[float]:

        assert x.ndim == 2
        assert y.ndim == 2
        assert x.shape[0] == y.shape[0]
        assert y.shape[1] == self.num_out

        # Remove repeating data points
        x, y = remove_repeating_samples(x, y)

        # Determine if the dataset is not too large
        if len(y) > self.max_training_dataset_size:
            x, y = subsample_training_data(x, y, self.max_training_dataset_size)

        self.x = x.to(dtype=self.dtype, device=self.device)
        self.y = y.to(dtype=self.dtype, device=self.device)

        # Normalise target values
        if len(y) > 1:
            self.y_mean = self.y.mean(axis=0)
            self.y_std = self.y.std(axis=0)
            self.y_std[self.y_std < 1e-6] = 1.
            self.y = (self.y - self.y_mean) / self.y_std

        # Add a small amount of noise to prevent training instabilities
        self.y += 1e-6 * torch.randn_like(self.y)

        if self.gp is None:
            self.gp = GPyTorchGPModel(self.x, self.y, self.kernel, self.likelihood).to(self.x)
            self.likelihood = self.likelihood.to(self.x)
        else:
            self.gp.set_train_data(self.x, self.y.flatten(), strict=False)
            self.gp.to(self.x)
            self.likelihood.to(self.x)

        # Attempt to make a local copy of the class to possibly later recover from a ValueError Exception
        self_copy = None
        try:
            self_copy = copy.deepcopy(self)
        except:
            pass

        self.gp.train()
        self.likelihood.train()

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp)

        if self.optimiser == 'adam':
            opt = torch.optim.Adam([{'params': mll.parameters()}], lr=self.lr)
        else:
            raise NotImplementedError(f'Optimiser {self.optimiser} was not implemented.')
        losses = []
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):

            for epoch in range(self.num_epochs):
                def closure(append_loss=True):
                    opt.zero_grad()
                    dist = self.gp(self.x)
                    loss = -1 * mll(dist, self.y.squeeze())
                    loss.backward()
                    if append_loss:
                        losses.append(loss.item())
                    return loss

                try:
                    opt.step(closure)
                except NotPSDError:
                    warnings.warn('\n\nMatrix is singular during GP training. Resetting GP parameters to ' + \
                                  'what they were before gp training and moving to the next BO iteration. Possible ' + \
                                  ' solutions: \n\n - Consider changing to double precision \n - Decrease the number of ' + \
                                  'GP training epochs per BO iteration or the GP learning rate to avoid overfitting.\n')

                    self.__class__ = copy.deepcopy(self_copy.__class__)
                    self.__dict__ = copy.deepcopy(self_copy.__dict__)

                    break
                except ValueError as e:
                    kernel_any_nan = False
                    mean_any_nan = False
                    likelihood_any_nan = False
                    for _, param in self.gp.mean.named_parameters():
                        if torch.isnan(param).any():
                            mean_any_nan = True
                            break
                    for _, param in self.gp.kernel.named_parameters():
                        if torch.isnan(param).any():
                            kernel_any_nan = True
                            break
                    for _, param in self.likelihood.named_parameters():
                        if torch.isnan(param).any():
                            likelihood_any_nan = True
                            break

                    if (mean_any_nan or kernel_any_nan or likelihood_any_nan) and (self_copy is not None):
                        warnings.warn(f'\n\nSome parameters (mean: {mean_any_nan} | kernel: {kernel_any_nan} | '
                                      f'likelihood: {likelihood_any_nan}) became NaN. Resetting GP parameters to ' + \
                                      'what they were before gp training and moving to the next BO iteration.\n\n')
                        self.__class__ = copy.deepcopy(self_copy.__class__)
                        self.__dict__ = copy.deepcopy(self_copy.__dict__)
                        break
                    else:
                        raise e

                if self.verbose and ((epoch + 1) % self.print_every == 0 or epoch == 0):
                    print('After %d epochs, loss = %g' % (epoch + 1, closure(append_loss=False).item()), flush=True)

        self.gp.eval()
        self.likelihood.eval()
        return losses

    def predict(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        num_points = len(x)

        if num_points < self.max_batch_size:
            # Evaluate all points at once
            with gpytorch.settings.fast_pred_var(), gpytorch.settings.debug(False):
                x = x.to(device=self.device, dtype=self.dtype)
                pred = self.gp(x)
                if self.pred_likelihood:
                    pred = self.likelihood(pred)
                mu_ = pred.mean.reshape(-1, self.num_out)
                var_ = pred.variance.reshape(-1, self.num_out)
        else:
            # Evaluate all points in batches
            mu_ = torch.zeros((len(x), self.num_out), device=self.device, dtype=self.dtype)
            var_ = torch.zeros((len(x), self.num_out), device=self.device, dtype=self.dtype)
            for i in range(int(np.ceil(num_points / self.max_batch_size))):
                x_ = x[i * self.max_batch_size: (i + 1) * self.max_batch_size].to(self.device, self.dtype)
                pred = self.gp(x_)
                if self.pred_likelihood:
                    pred = self.likelihood(pred)
                mu_temp = pred.mean.reshape(-1, self.num_out)
                var_temp = pred.variance.reshape(-1, self.num_out)

                mu_[i * self.max_batch_size: (i + 1) * self.max_batch_size] = mu_temp
                var_[i * self.max_batch_size: (i + 1) * self.max_batch_size] = var_temp

        mu = (self.y_std.to(mu_) * mu_ + self.y_mean.to(mu_))
        var = (var_ * self.y_std.to(mu_) ** 2)
        return mu, var.clamp(min=torch.finfo(var.dtype).eps)

    def sample_y(self, x: torch.FloatTensor, n_samples=1) -> torch.FloatTensor:
        """
        Should return (n_samples, Xc.shape[0], self.num_out)
        """
        x = x.to(dtype=self.dtype, device=self.device)
        with gpytorch.settings.debug(False):
            pred = self.gp(x)
            if self.pred_likelihood:
                pred = self.likelihood(pred)
            sample = pred.rsample(torch.Size((n_samples,))).view(n_samples, x.shape[0], 1)
            sample = self.y_std * sample + self.y_mean
            return sample

    @property
    def noise(self) -> torch.Tensor:
        if self.num_out == 1:
            return (self.gp.likelihood.noise * self.y_std ** 2).view(self.num_out).detach()
        else:
            return (self.gp.likelihood.task_noises * self.y_std ** 2).view(self.num_out).detach()

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        return super().to(device=device, dtype=dtype)


class GPyTorchGPModel(ExactGP):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, kernel: Kernel, likelihood: GaussianLikelihood):
        super(GPyTorchGPModel, self).__init__(x, y.squeeze(), likelihood)
        self.multi_task = y.shape[1] > 1
        self.mean = ConstantMean() if not self.multi_task else MultitaskMean(ConstantMean(), num_tasks=y.shape[1])
        self.kernel = kernel if not self.multi_task else MultitaskKernel(kernel, num_tasks=y.shape[1])

    def forward(self, x: torch.FloatTensor) -> MultivariateNormal:
        mean = self.mean(x)
        cov = self.kernel(x)
        return MultivariateNormal(mean, cov) if not self.multi_task else MultitaskMultivariateNormal(mean, cov)
