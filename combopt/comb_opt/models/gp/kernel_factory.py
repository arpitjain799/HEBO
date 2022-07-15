# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from typing import Optional

import numpy as np
import torch
from gpytorch.constraints import Interval
from gpytorch.kernels import Kernel, MaternKernel, RBFKernel, ScaleKernel

from comb_opt.models.gp.kernels import DiffusionKernel, MixtureKernel, Overlap, TransformedOverlap, \
    SubStringKernel, ConditionalTransformedOverlapKernel
from comb_opt.search_space import SearchSpace
from comb_opt.tasks.eda_seq_opt.utils.utils_eda_search_space import get_active_dims
from comb_opt.tasks.eda_seq_opt.utils.utils_operators import get_operator_space
from comb_opt.tasks.eda_seq_opt.utils.utils_operators_hyp import get_operator_hyperparms_space


def kernel_factory(kernel_name: str, active_dims: Optional[list] = None, use_ard: bool = True,
                   lengthscale_constraint: Optional[Interval] = None, outputscale_constraint: Optional[Interval] = None,
                   **kwargs) -> Optional[Kernel]:
    if active_dims is not None:
        if len(active_dims) == 0:
            return None

    ard_num_dims = len(active_dims) if use_ard else None

    # Kernels for numeric variables
    if kernel_name is None:
        kernel = None

    elif kernel_name == 'diffusion':
        assert 'fourier_freq_list' in kwargs
        assert 'fourier_basis_list' in kwargs
        kernel = DiffusionKernel(active_dims=active_dims, ard_num_dims=ard_num_dims,
                                 fourier_freq_list=kwargs.get('fourier_freq_list'),
                                 fourier_basis_list=kwargs.get('fourier_basis_list'))

    elif kernel_name == 'rbf':
        kernel = RBFKernel(active_dims=active_dims, ard_num_dims=ard_num_dims,
                           lengthscale_constraint=lengthscale_constraint)
    elif kernel_name == 'mat52':
        kernel = MaternKernel(active_dims=active_dims, ard_num_dims=ard_num_dims,
                              lengthscale_constraint=lengthscale_constraint)
    elif kernel_name == 'overlap':
        kernel = Overlap(active_dims=active_dims, ard_num_dims=ard_num_dims,
                         lengthscale_constraint=lengthscale_constraint)
    elif kernel_name == 'transformed_overlap':
        kernel = TransformedOverlap(active_dims=active_dims, ard_num_dims=ard_num_dims,
                                    lengthscale_constraint=lengthscale_constraint)

    elif kernel_name == 'ssk':
        assert 'device' in kwargs
        assert 'search_space' in kwargs

        # Firstly check that the ssk kernel is applied to the nominal dimensions
        assert active_dims == kwargs.get(
            'search_space').nominal_dims, 'The SSK kernel can only be applied to nominal variables'

        # Secondly check that all of the ordinal dims share the same alphabet size
        alphabet_per_var = [kwargs.get('search_space').params[param_name].categories for param_name in
                            kwargs.get('search_space').nominal_names]
        assert all(alphabet == alphabet_per_var[0] for alphabet in
                   alphabet_per_var), 'The alphabet must be the same for each of the nominal variables'

        kernel = SubStringKernel(seq_length=len(active_dims),
                                 alphabet_size=len(alphabet_per_var[0]),
                                 gap_decay=0.5,
                                 match_decay=0.8,
                                 max_subsequence_length=len(active_dims) if len(active_dims) < 3 else 3,
                                 normalize=False,
                                 device=kwargs.get('device'),
                                 active_dims=active_dims)

    else:
        raise NotImplementedError(f'{kernel_name} was not implemented')

    if kernel is not None:
        kernel = ScaleKernel(kernel, outputscale_constraint=outputscale_constraint)

    return kernel


def mixture_kernel_factory(search_space: SearchSpace, is_mixed: bool, is_numeric: bool, is_nominal: bool,
                           numeric_kernel_name: Optional[str] = None,
                           numeric_kernel_use_ard: Optional[bool] = True,
                           numeric_lengthscale_constraint: Optional[Interval] = None,
                           nominal_kernel_name: Optional[str] = None,
                           nominal_kernel_use_ard: Optional[bool] = True,
                           nominal_lengthscale_constraint: Optional[Interval] = None,
                           ) -> Kernel:
    assert is_mixed or (is_numeric or is_nominal)

    if is_mixed:

        assert numeric_kernel_name is not None
        assert nominal_kernel_name is not None
        assert numeric_kernel_name in ['mat52', 'rbf']
        assert nominal_kernel_name in ['overlap', 'transformed_overlap']

        kernel = ScaleKernel(MixtureKernel(
            search_space=search_space,
            numeric_kernel_name=numeric_kernel_name,
            numeric_use_ard=numeric_kernel_use_ard,
            numeric_lengthscale_constraint=numeric_lengthscale_constraint,
            categorical_kernel_name=nominal_kernel_name,
            categorical_use_ard=nominal_kernel_use_ard,
            categorical_lengthscale_constraint=nominal_lengthscale_constraint))
    else:
        if is_numeric:

            assert numeric_kernel_name is not None
            assert numeric_kernel_name in ['mat52', 'rbf']
            active_dims = search_space.cont_dims + search_space.disc_dims

            if numeric_kernel_name == 'mat52':
                kernel = ScaleKernel(MaternKernel(active_dims=active_dims,
                                                  nu=2.5,
                                                  ard_num_dims=len(active_dims) if numeric_kernel_use_ard else None,
                                                  lengthscale_constraint=numeric_lengthscale_constraint))

            elif numeric_kernel_name == 'rbf':
                kernel = ScaleKernel(RBFKernel(active_dims=active_dims,
                                               ard_num_dims=len(active_dims) if numeric_kernel_use_ard else None,
                                               lengthscale_constraint=numeric_lengthscale_constraint))

            else:
                raise ValueError(numeric_kernel_name)

        elif is_nominal:

            assert nominal_kernel_name is not None
            assert nominal_kernel_name in ['overlap', 'transformed_overlap']

            if nominal_kernel_name == 'overlap':
                kernel = ScaleKernel(Overlap(
                    active_dims=search_space.nominal_dims,
                    lengthscale_constraint=nominal_lengthscale_constraint,
                    ard_num_dims=len(search_space.nominal_dims) if nominal_kernel_use_ard else None))

            elif nominal_kernel_name == 'transformed_overlap':
                kernel = ScaleKernel(TransformedOverlap(
                    active_dims=search_space.nominal_dims,
                    lengthscale_constraint=None,
                    ard_num_dims=len(search_space.nominal_dims) if nominal_kernel_use_ard else None))

            else:
                raise ValueError(nominal_kernel_name)

        else:
            raise ValueError("Not numeric nor nominal")

    return kernel


def get_conditional_sequence_kernel(cond_kernel_type: str, seq_len: int, operator_space_id: str,
                                    operator_hyperparams_space_id: str,
                                    seq_kern_name: str = 'transformed_overlap',
                                    nominal_kern_name: str = 'transformed_overlap', numeric_kern_name: str = 'mat52',
                                    device: torch.device = torch.device('cuda:0')):
    assert nominal_kern_name in ['overlap', 'transformed_overlap']
    assert numeric_kern_name in ['mat52', 'rbf']

    # Get active dims
    seq_indices, param_indices, param_dims, param_active_dims = get_active_dims(
        seq_len=seq_len,
        operator_space_id=operator_space_id,
        operator_hyperparams_space_id=operator_hyperparams_space_id
    )

    param_kernels = []
    operator_hyperparams_space = get_operator_hyperparms_space(
        operator_hyperparams_space_id=operator_hyperparams_space_id)

    operator_space = get_operator_space(operator_space_id=operator_space_id)

    map_cat_to_kernel_ind: np.ndarray = np.arange(len(operator_space))

    # TODO: be cleaner -> don't recompute these each time and use search space
    for operator_ind, operator in enumerate(operator_space.all_operators):
        op_id = operator.op_id
        if op_id in operator_hyperparams_space.all_hyps and len(operator_hyperparams_space.all_hyps[op_id]) > 0:
            op_params = operator_hyperparams_space.all_hyps[op_id]
            if np.all([op_param['type'] not in ['bool', 'cat'] for op_param in op_params]):  # continuous / int dims
                if numeric_kern_name == 'mat52':
                    param_kernels.append(MaternKernel(nu=2.5, active_dims=param_active_dims[op_id],
                                                      ard_num_dims=len(param_active_dims[op_id])))
                elif numeric_kern_name == 'rbf':
                    param_kernels.append(
                        RBFKernel(active_dims=param_active_dims[op_id],
                                  ard_num_dims=len(param_active_dims[op_id])))
                else:
                    raise ValueError(numeric_kern_name)
            elif np.all([op_param['type'] in ['bool', 'cat'] for op_param in op_params]):  # categorical:
                if nominal_kern_name == 'overlap':
                    param_kernels.append(Overlap(active_dims=param_active_dims[op_id]))
                elif nominal_kern_name == 'transformed_overlap':
                    param_kernels.append(TransformedOverlap(active_dims=param_active_dims[op_id]))
                else:
                    raise ValueError(nominal_kern_name)
            else:  # mixed
                num_dims = []
                cat_dims = []
                for ind_param, op_param in enumerate(op_params):
                    if op_param['type'] in ['bool', 'cat']:
                        cat_dims.append(ind_param)
                    else:
                        num_dims.append(ind_param)
                assert len(num_dims) > 0 and len(cat_dims) > 0, (num_dims, cat_dims)
                param_kernels.append(
                    MixtureKernel(search_space=None,
                                  num_dims=num_dims, nominal_dims=cat_dims, active_dims=param_active_dims[op_id],
                                  categorical_kernel_name=nominal_kern_name, numeric_kernel_name=numeric_kern_name))

        else:
            map_cat_to_kernel_ind[operator_ind] = -1  # no hyp for this operator
            if cond_kernel_type == 'cond-prod-kernel':
                raise ValueError(operator, operator_hyperparams_space_id)

    if cond_kernel_type == 'cond-transf-cat-kernel':
        # Define the conditional transformed-categorical kernel
        kernel = ConditionalTransformedOverlapKernel(
            *param_kernels,
            seq_indices=seq_indices,
            param_indices=param_indices,
            n_categories=len(operator_space.all_operators),
            map_cat_to_kernel_ind=map_cat_to_kernel_ind,
            ard_num_dims=seq_len
        )
    else:
        raise ValueError(cond_kernel_type)

    return ScaleKernel(kernel)
