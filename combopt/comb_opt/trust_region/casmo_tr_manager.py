# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from typing import Union, Optional

import pandas as pd
import torch

from comb_opt.acq_funcs import AcqBase
from comb_opt.models import ModelBase
from comb_opt.search_space import SearchSpace
from comb_opt.trust_region import TrManagerBase
from comb_opt.trust_region.tr_utils import sample_numeric_and_nominal_within_tr
from comb_opt.utils.data_buffer import DataBuffer
from comb_opt.utils.discrete_vars_utils import get_discrete_choices
from comb_opt.utils.distance_metrics import hamming_distance
from comb_opt.utils.model_utils import move_model_to_device


class CasmopolitanTrManager(TrManagerBase):

    def __init__(self,
                 search_space: SearchSpace,
                 model: ModelBase,
                 acq_func: AcqBase,
                 n_init: int,
                 min_num_radius: Union[int, float],
                 max_num_radius: Union[int, float],
                 init_num_radius: Union[int, float],
                 min_nominal_radius: Union[int, float],
                 max_nominal_radius: Union[int, float],
                 init_nominal_radius: Union[int, float],
                 radius_multiplier: float = 1.5,
                 succ_tol: int = 20,
                 fail_tol: int = 2,
                 restart_n_cand: int = 1000,
                 max_n_perturb_num: int = 20,
                 verbose=False,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device('cpu')
                 ):
        super(CasmopolitanTrManager, self).__init__(search_space, dtype)

        assert self.search_space.num_cont + self.search_space.num_disc + self.search_space.num_nominal \
               == self.search_space.num_dims, \
            'The Casmopolitan Trust region manager only supports continuous, ' \
            'discrete and nominal variables'

        self.is_numeric = search_space.num_numeric > 0
        self.is_mixed = self.is_numeric and search_space.num_nominal > 0
        self.numeric_dims = self.search_space.cont_dims + self.search_space.disc_dims
        self.discrete_choices = get_discrete_choices(search_space)

        # Register radii for useful variable types
        if search_space.num_numeric > 0:
            self.register_radius('numeric', min_num_radius, max_num_radius, init_num_radius)
        #  if there is only one dim for a variable type: do not use TR for it
        if search_space.num_nominal > 1:
            self.register_radius('nominal', min_nominal_radius, max_nominal_radius, init_nominal_radius)

        self.verbose = verbose
        self.model = model
        self.acq_func = acq_func
        self.n_init = n_init
        self.restart_n_cand = restart_n_cand
        self.max_n_perturb_num = max_n_perturb_num

        self.succ_tol = succ_tol
        self.fail_tol = fail_tol
        self.radius_multiplier = radius_multiplier
        self.device = device

        self.succ_count = 0
        self.fail_count = 0
        self.guided_restart_buffer = DataBuffer(self.search_space, 1, self.data_buffer.dtype)
        assert self.is_numeric or self.search_space.num_nominal > 0

    def adjust_counts(self, y: torch.Tensor):
        if y.min() < self.data_buffer.y.min():  # Originally we had np.min(fX_next) <= tr_min - 1e-3 * abs(tr_min)
            self.succ_count += 1
            self.fail_count = 0
        else:
            self.succ_count = 0
            self.fail_count += 1

    def adjust_tr_radii(self, y: torch.Tensor, **kwargs):
        self.adjust_counts(y=y)

        if self.succ_count == self.succ_tol:  # Expand trust region
            self.succ_count = 0
            if self.is_numeric:
                self.radii['numeric'] = min(self.radii['numeric'] * self.radius_multiplier, self.max_radii['numeric'])
            if self.search_space.num_nominal > 1:
                self.radii['nominal'] = int(
                    min(self.radii['nominal'] * self.radius_multiplier, self.max_radii['nominal']))
            if self.verbose:
                print(f"Expanding trust region...")

        elif self.fail_count == self.fail_tol:  # Shrink trust region
            self.fail_count = 0
            if self.is_numeric:
                self.radii['numeric'] = self.radii['numeric'] / self.radius_multiplier
            if self.search_space.num_nominal > 1:
                self.radii['nominal'] = int(self.radii['nominal'] / self.radius_multiplier)
            if self.verbose:
                print(f"Shrinking trust region...")

    def suggest_new_tr(self, n_init: int, x_init: pd.DataFrame, observed_data_buffer: DataBuffer,
                       best_y: Optional[Union[float, torch.Tensor]] = None, **kwargs) -> pd.DataFrame:

        trigger_reset = False
        for variable_type in self.variable_types:
            if self.radii[variable_type] < self.min_radii[variable_type]:
                trigger_reset = True
                break

        if not trigger_reset:
            return x_init

        if self.verbose:
            print("Algorithm is stuck in a local optimum. Triggering a guided restart.")

        x_init = pd.DataFrame(index=range(n_init), columns=self.search_space.df_col_names, dtype=float)

        if len(self.data_buffer) > 0:

            tr_x, tr_y = self.data_buffer.x, self.data_buffer.y

            # store best observed point within current trust region
            best_idx, best_y = self.data_buffer.y_argmin, self.data_buffer.y_min
            self.guided_restart_buffer.append(tr_x[best_idx: best_idx + 1], tr_y[best_idx: best_idx + 1])

            # Determine the device to run on
            move_model_to_device(self.model, self.guided_restart_buffer, self.device)

            # Fit the model
            self.model.fit(self.guided_restart_buffer.x, self.guided_restart_buffer.y)

            # Sample random points and evaluate the acquisition at these points
            x_cand = self.search_space.transform(self.search_space.sample(self.restart_n_cand))
            with torch.no_grad():
                acq = self.acq_func(x_cand, self.model, best_y=best_y)

            # The new trust region centre is the point with the lowest acquisition value
            best_idx = acq.argmin()

            tr_centre = x_cand[best_idx]
            x_init.iloc[0: 1] = self.search_space.inverse_transform(tr_centre.unsqueeze(0))

        else:
            x_init.iloc[0: 1] = self.search_space.sample(1)
            tr_centre = self.search_space.transform(x_init.iloc[0:1]).squeeze()

        self.restart_tr()

        # Sample remaining points in the trust region of the new centre
        if self.n_init - 1 > 0:
            # Sample the remaining points
            x_in_tr = sample_numeric_and_nominal_within_tr(x_centre=tr_centre,
                                                           search_space=self.search_space,
                                                           tr_manager=self,
                                                           n_points=self.n_init - 1,
                                                           is_numeric=self.is_numeric,
                                                           is_mixed=self.is_mixed,
                                                           numeric_dims=self.numeric_dims,
                                                           discrete_choices=self.discrete_choices,
                                                           max_n_perturb_num=self.max_n_perturb_num,
                                                           model=self.model,
                                                           return_numeric_bounds=False)

            # Store them
            x_init.iloc[1: self.n_init] = self.search_space.inverse_transform(x_in_tr)

        # update data_buffer with previously observed points that are in the same trust region
        x_observed, y_observed = observed_data_buffer.x, observed_data_buffer.y
        for i in range(len(observed_data_buffer)):
            x = x_observed[i:i + 1]

            # Check the numeric and hamming distance
            if ((tr_centre[self.numeric_dims] - x[0, self.numeric_dims]).abs() < self.radii['numeric']).all() \
                    and hamming_distance(tr_centre[self.search_space.nominal_dims].unsqueeze(0),
                                         x[:, self.search_space.nominal_dims],
                                         False).squeeze() <= self.get_nominal_radius():
                self.data_buffer.append(x, y_observed[i:i + 1])

        return x_init

    def restart(self):
        self.restart_tr()
        self.guided_restart_buffer.restart()

    def restart_tr(self):
        super(CasmopolitanTrManager, self).restart_tr()
        self.succ_count = 0
        self.fail_count = 0

    def __getstate__(self):
        d = dict(self.__dict__)
        to_remove = ["model", "search_space"]  # fields to remove when pickling this object
        for attr in to_remove:
            if attr in d:
                del d[attr]
        return d
