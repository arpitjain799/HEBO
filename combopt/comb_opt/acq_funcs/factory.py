# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from comb_opt.acq_funcs.ei import EI
from comb_opt.acq_funcs.lcb import LCB
from comb_opt.acq_funcs.thompson_sampling import ThompsonSampling


def acq_factory(acq_func_name: str, **kwargs):
    if acq_func_name == 'lcb':
        beta = kwargs.get('beta', 1.96)
        acq_func = LCB(beta)

    elif acq_func_name == 'ei':
        acq_func = EI(augmented_ei=False)

    elif acq_func_name == 'aei':
        acq_func = EI(augmented_ei=True)

    elif acq_func_name == 'thompson':
        acq_func = ThompsonSampling()

    else:
        raise NotImplementedError(f'Acquisition function {acq_func_name} is not implemented.')

    return acq_func
