# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from abc import ABC
from abc import abstractmethod
from typing import Optional, List

import torch

from comb_opt.search_space import SearchSpace


class ModelBase(ABC):
    supports_cuda = False
    support_ts = False
    support_grad = False
    support_multi_output = False
    support_warm_start = False
    ensemble = False

    def __init__(self, search_space: SearchSpace, num_out: int, dtype: torch.dtype, device: torch.device, **kwargs):
        """
        Base class for probabilistic regression models
        """
        super(ModelBase, self).__init__()

        self.x = None
        self.y = None

        self.num_out = num_out
        self.search_space = search_space
        self.cont_dims = search_space.cont_dims
        self.disc_dims = search_space.disc_dims
        self.ordinal_dims = search_space.ordinal_dims
        self.nominal_dims = search_space.nominal_dims
        self.perm_dims = search_space.perm_dims
        self.dtype = dtype
        self.device = device
        self.kwargs = kwargs

        # Basic checks
        assert self.num_out > 0
        assert (len(self.cont_dims) >= 0)
        assert (len(self.disc_dims) >= 0)
        assert (len(self.ordinal_dims) >= 0)
        assert (len(self.nominal_dims) >= 0)
        assert (len(self.perm_dims) >= 0)
        assert (len(self.cont_dims) + len(self.disc_dims) + len(self.ordinal_dims) + len(self.nominal_dims) + len(
            self.perm_dims) > 0)

    @abstractmethod
    def fit(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Optional[List[float]]:
        """
        Function used to fit the parameters of the model

        :param x:
        :param y:
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def predict(self, x: torch.Tensor, **kwargs) -> (torch.Tensor, torch.Tensor):
        """
        Function used to return the mean and variance of the (possibly approximated) Gaussian predictive distribution
        for the input x. Output shape of each tensor is (N, num_out) where N is the number of input points

        :param x:
        :param kwargs:
        :return:
        """
        pass

    @property
    def noise(self) -> torch.Tensor:
        """
        Return estimated noise variance, for example, GP can view noise level as a hyperparameter and optimize it via
         MLE, another strategy could be using the MSE of training data as noise estimation Should return a float tensor
         of shape self.num_out
        """
        return torch.zeros(self.num_out, dtype=self.dtype)

    def sample_y(self, x: torch.Tensor, n_samples: int, **kwargs) -> torch.Tensor:
        py, ps2 = self.predict(x)
        ps = ps2.sqrt()
        samp = torch.zeros(n_samples, py.shape[0], self.num_out)
        for i in range(n_samples):
            samp[i] = py + ps * torch.randn(py.shape).to(py)
        return samp

    @abstractmethod
    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """
        Function used to move model to target device and dtype. Note that this should also change self.dtype and
        self.device

        :param device:
        :param dtype:
        :return:
        """
        pass

    def pre_fit_method(self, x: torch.Tensor, y: torch.Tensor, **kwargs):
        """
        Function called at the before fitting the model in the suggest method. Can be used to update the internal state
        of the model based on the the data that will be used to fit the model. Use cases may include training a VAE
        for latent space BO, or re-initialising the model before fitting it to the data.

        :param x:
        :param y:
        :param kwargs:
        :return:
        """
        pass


class EnsembleModelBase(ModelBase):
    """
    Ensemble of models. This class is commonly used when sampling from the model's parameter's posterior.
    """

    ensemble = True

    def __init__(self,
                 search_space: SearchSpace,
                 num_out: int,
                 num_models: int,
                 dtype: torch.dtype,
                 device: torch.device,
                 **kwargs):
        self.num_models = num_models
        self.models = []

        super(EnsembleModelBase, self).__init__(search_space, num_out, dtype, device)

    @abstractmethod
    def fit(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Optional[List[float]]:
        """
        Function used to fit num_models models

        :param x:
        :param y:
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def predict(self, x: torch.Tensor, **kwargs) -> (torch.Tensor, torch.Tensor):
        """
        Function used to return the mean and variance of the (possibly approximated) Gaussian predictive distribution
        for the input x. Output shape (N, num_out, num_models) where N is the number of input points.

        If the model uses a device, this method should automatically move x to the target device.

        :param x:
        :param kwargs:
        :return:
        """
        pass

    @property
    def noise(self) -> torch.Tensor:
        """
        Return estimated noise variance, for example, GP can view noise level
        as a hyperparameter and optimize it via MLE, another strategy could be
        using the MSE of training data as noise estimation
        Should return a float tensor of shape self.num_out
        """
        noise = 0
        for model in self.models:
            noise += model.noise

        return noise / len(self.models)

    def sample_y(self, x: torch.Tensor, n_samples: int, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """
        Function used to move model to target device and dtype. Note that this should also change self.dtype and
        self.device

        :param device:
        :param dtype:
        :return:
        """
        for model in self.models:
            model.to(device, dtype)

    def pre_fit_method(self, x: torch.Tensor, y: torch.Tensor, **kwargs):
        """
        Function called at the before fitting the model in the suggest method. Can be used to update the internal state
        of the model based on the the data that will be used to fit the model. Use cases may include training a VAE
        for latent space BO, or re-initialising the model before fitting it to the data.

        :param x:
        :param y:
        :param kwargs:
        :return:
        """
        for model in self.models:
            model.pre_fit_method(x, y, **kwargs)
