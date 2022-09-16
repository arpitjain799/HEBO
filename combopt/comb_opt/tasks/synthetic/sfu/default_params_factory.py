# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
from typing import Optional

import numpy as np


def default_sfu_params_factory(task_name: str, num_dims: int, task_name_suffix: Optional[str] = None):
    assert isinstance(task_name, str)
    assert isinstance(num_dims, int)

    if task_name == 'ackley':
        params = {'num_dims': num_dims,
                  'lb': -32.768,
                  'ub': 32.768,
                  'a': 20,
                  'b': 0.2,
                  'c': 2 * np.pi
                  }

    elif task_name == 'griewank':
        params = {'num_dims': num_dims,
                  'lb': -600,
                  'ub': 600
                  }

    elif task_name == 'langermann':
        params = {'num_dims': num_dims,
                  'lb': 0,
                  'ub': 10
                  }
        if num_dims == 2:
            params['m'] = 5
            params['c'] = np.array([[1., 2., 5., 2., 3.]])
            params['a'] = np.array([[3., 5.], [5., 2.], [2., 1.], [1., 4], [7., 9]])

    elif task_name == 'levy':
        params = {'num_dims': num_dims,
                  'lb': -10,
                  'ub': 10
                  }

    elif task_name == 'rastrigin':
        params = {'num_dims': num_dims,
                  'lb': -5.12,
                  'ub': 5.12
                  }

    elif task_name == 'schwefel':
        params = {'num_dims': num_dims,
                  'lb': -500,
                  'ub': 500
                  }

    elif task_name == 'perm0':
        params = {'num_dims': num_dims,
                  'lb': -num_dims,
                  'ub': num_dims,
                  'beta': 10
                  }

    elif task_name == 'rot_hyp':
        params = {'num_dims': num_dims,
                  'lb': -65.536,
                  'ub': 65.536,
                  }

    elif task_name == 'sphere':
        params = {'num_dims': num_dims,
                  'lb': -5.12,
                  'ub': 5.12,
                  }

    elif task_name == 'modified_sphere':
        params = {'num_dims': num_dims,
                  'lb': 0,
                  'ub': 1,
                  }

    elif task_name == 'sum_pow':
        params = {'num_dims': num_dims,
                  'lb': -1,
                  'ub': 1,
                  }

    elif task_name == 'sum_squares':
        params = {'num_dims': num_dims,
                  'lb': -10,
                  'ub': 10,
                  }

    elif task_name == 'trid':
        params = {'num_dims': num_dims,
                  'lb': -num_dims ** 2,
                  'ub': num_dims ** 2,
                  }

    elif task_name == 'power_sum':
        params = {'num_dims': num_dims,
                  'lb': 0.,
                  'ub': num_dims
                  }
        if num_dims == 4:
            params['b'] = np.array([[8., 18., 44., 114.]])

    elif task_name == 'zakharov':
        params = {'num_dims': num_dims,
                  'lb': -5,
                  'ub': 10
                  }

    elif task_name == 'dixon_price':
        params = {'num_dims': num_dims,
                  'lb': -10,
                  'ub': 10
                  }

    elif task_name == 'rosenbrock':
        params = {'num_dims': num_dims,
                  'lb': -5,
                  'ub': 10
                  }

    elif task_name == 'michalewicz':
        params = {'num_dims': num_dims,
                  'lb': 0,
                  'ub': np.pi
                  }

    elif task_name == 'perm':
        params = {'num_dims': num_dims,
                  'lb': -num_dims,
                  'ub': num_dims,
                  'beta': 10
                  }
    elif task_name == 'powell':
        params = {'num_dims': num_dims,
                  'lb': -4,
                  'ub': 5,
                  }
    elif task_name == 'styblinski_tang':
        params = {'num_dims': num_dims,
                  'lb': -5,
                  'ub': 5,
                  }

    else:
        raise NotImplemented(f'Task {task_name} is not implemented')
    params["task_name_suffix"] = task_name_suffix
    return params
