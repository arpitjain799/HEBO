# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from typing import Union, List, Optional, Dict, Any

import numpy as np
import torch

from comb_opt.search_space import SearchSpace
from comb_opt.tasks import TaskBase, RandomTSP, PestControl, default_sfu_params_factory, SFU_FUNCTIONS, CDRH3Design, \
    MigSeqOpt
from comb_opt.tasks.bayesmark.bayesmark_task import BayesmarkTask
from comb_opt.tasks.rna_inverse_fold.rna_inverse_fold_task import RNAInverseFoldTask
from comb_opt.tasks.rna_inverse_fold.utils import get_target_from_id, RNA_BASES

SFU_SYNTHETIC_FUNC_NAMES = list(SFU_FUNCTIONS.keys())


def _sfu_search_space_params_factory(variable_type: Union[str, List[str]], num_dims: Union[int, List[int]],
                                     lb: Union[int, float], ub: Union[int, float],
                                     num_categories: Optional[Union[int, List[int]]] = None,
                                     **kwargs) \
        -> List[Union[Dict[str, Union[Union[str, float, int], Any]], Dict[str, Union[Union[str, object], Any]], Dict[
            str, Union[str, float, int]], Dict[str, Union[str, object]]]]:
    # Basic checks to ensure all arguments are correct
    assert isinstance(variable_type, str) or isinstance(variable_type, list)
    assert isinstance(num_dims, int) or isinstance(num_dims, list)

    if isinstance(variable_type, list):
        assert isinstance(num_dims, list)
    else:
        assert isinstance(num_dims, int)

    if isinstance(variable_type, list):
        for var_type in variable_type:
            assert isinstance(var_type, str)
            assert var_type in ['num', 'int', 'nominal', 'ordinal']

    if isinstance(variable_type, list) and ('nominal' in variable_type or 'ordinal' in variable_type):
        assert isinstance(num_categories, list)
        for num_cats, var_type in zip(num_categories, variable_type):
            if var_type in ['nominal', 'ordinal']:
                assert isinstance(num_cats, int) and num_cats > 0

    elif isinstance(variable_type, str) and (variable_type == 'nominal' or variable_type == 'ordinal'):
        assert isinstance(num_categories, int) and num_categories > 0

    if isinstance(num_dims, list):
        for num in num_dims:
            assert isinstance(num, int)

    assert isinstance(lb, int) or isinstance(lb, float)
    assert isinstance(ub, int) or isinstance(ub, float)
    assert lb < ub

    params = []
    counter = 1

    if isinstance(variable_type, list):
        for i, var_type in enumerate(variable_type):
            if var_type in ['num', 'int']:
                for _ in range(num_dims[i]):
                    params.append({'name': f'var_{counter}', 'type': var_type, 'lb': lb, 'ub': ub})
                    counter += 1
            elif var_type in ['nominal', 'ordinal']:
                categories = np.linspace(lb, ub, num_categories[i]).tolist()
                for _ in range(num_dims[i]):
                    params.append({'name': f'var_{counter}', 'type': var_type, 'categories': categories})
                    counter += 1

    else:
        if variable_type in ['num', 'int']:
            for _ in range(num_dims):
                params.append({'name': f'var_{counter}', 'type': variable_type, 'lb': lb, 'ub': ub})
                counter += 1
        elif variable_type in ['nominal', 'ordinal']:
            categories = np.linspace(lb, ub, num_categories).tolist()
            for _ in range(num_dims):
                params.append({'name': f'var_{counter}', 'type': variable_type, 'categories': categories})
                counter += 1

    return params


def search_space_factory(task_name: str, dtype: torch.dtype, **kwargs) -> SearchSpace:
    if task_name in SFU_SYNTHETIC_FUNC_NAMES:
        assert 'variable_type' in kwargs
        assert 'num_dims' in kwargs
        assert 'lb' in kwargs
        assert 'ub' in kwargs

        params = _sfu_search_space_params_factory(**kwargs)

    elif task_name == 'tsp':
        assert 'num_dims' in kwargs
        from comb_opt.tasks.utils.tsp_utils import tsp_repair_func

        params = [
            {'name': 'route', 'type': 'permutation', 'length': kwargs.get('num_dims'), 'repair_func': tsp_repair_func}]

    elif task_name == 'pest':
        categories = ['do nothing', 'pesticide 1', 'pesticide 2', 'pesticide 3', 'pesticide 4']
        params = []
        for i in range(1, 26):
            params.append({'name': f'stage_{i}', 'type': 'nominal', 'categories': categories})

    elif task_name == 'antibody_design':
        if 'cdrh3_length' in kwargs:
            assert isinstance(kwargs.get('cdrh3_length'), int)
            assert kwargs.get('cdrh3_length') > 0

        amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V',
                       'W', 'Y']
        params = [{'name': f'Amino acid {i + 1}', 'type': 'nominal', 'categories': amino_acids} for i in
                  range(kwargs.get('cdrh3_length', 11))]

    elif task_name == "rna_inverse_fold":
        binary_mode = kwargs.get("binary_mode", 0)

        assert "target" in kwargs, "Need to provide a target structure (e.g. `(((...))).(..)..(....)...`)"
        target = kwargs.get("target")

        if binary_mode:
            params = [{'name': f'x{i + 1}', 'type': 'bool'} for i in
                      range(2 * len(target))]
        else:
            params = [{'name': f'Base {i + 1}', 'type': 'nominal', 'categories': RNA_BASES} for i in
                      range(len(target))]

    elif task_name == 'mig_optimization':
        assert 'seq_len' in kwargs
        seq_len = kwargs.get('seq_len')

        operation_names = ['balance', 'cut rewrite', 'cut rewrite -z', 'refactor', 'refactor -z', 'resubstitute',
                           'functional_reduction']
        params = []
        for i in range(1, seq_len + 1):
            params.append({'name': f'op_{i}', 'type': 'nominal', 'categories': operation_names})

    else:
        raise NotImplementedError(f'{task_name} is not an implemented task.')

    return SearchSpace(params, dtype)


def task_factory(task_name: str, dtype: torch.dtype = torch.float32, **kwargs) -> (TaskBase, SearchSpace):
    """
    The task name specifies the task that should be returned.

    :param task_name:
    :param dtype:
    :param kwargs:
    :return:
    """

    if task_name in SFU_SYNTHETIC_FUNC_NAMES:
        assert 'variable_type' in kwargs
        assert 'num_dims' in kwargs

        if isinstance(kwargs.get('num_dims'), int):
            num_dims = kwargs.get('num_dims')
        elif isinstance(kwargs.get('num_dims'), list):
            num_dims = sum(kwargs.get('num_dims'))
        else:
            raise Exception('Expect num_dims to be either an integer or a list of integers')

        task_params = default_sfu_params_factory(task_name, num_dims)

        if 'lb' in kwargs:
            task_params['lb'] = kwargs.get('lb')
        if 'ub' in kwargs:
            task_params['ub'] = kwargs.get('ub')

        search_space = search_space_factory(task_name,
                                            dtype,
                                            variable_type=kwargs.get('variable_type'),
                                            num_dims=kwargs.get('num_dims'),
                                            lb=task_params.get('lb'),
                                            ub=task_params.get('ub'),
                                            num_categories=kwargs.get('num_categories', None))

        task = SFU_FUNCTIONS[task_name](**task_params)

    elif task_name == 'random_tsp':
        assert 'num_dims' in kwargs and isinstance(kwargs.get('num_dims'), int)

        search_space = search_space_factory('tsp', dtype, num_dims=kwargs.get('num_dims'))
        task = RandomTSP(kwargs.get('num_dims'), kwargs.get('min_dist', 1), kwargs.get('max_dist', 1000),
                         kwargs.get('seed', 0))

    elif task_name == 'pest':
        task = PestControl()
        search_space = search_space_factory(task_name, dtype)

    elif task_name == 'antibody_design':
        if 'antigen' not in kwargs:
            print('Target antigen not specified. Using antigen 1ADQ_A.')
        task = CDRH3Design(antigen=kwargs.get('antigen', '1ADQ_A'), cdrh3_length=kwargs.get('cdrh3_length', 11),
                           num_cpus=kwargs.get('num_cpus', 1), first_cpu=kwargs.get('first_cpu', 0))
        search_space = search_space_factory('antibody_design', dtype, cdrh3_length=kwargs.get('cdrh3_length', 11))

    elif task_name == "rna_inverse_fold":
        target = kwargs.get("target", 23)
        if isinstance(target, int):
            target = get_target_from_id(target)
        binary_mode = kwargs.get("binary_mode", False)
        task = RNAInverseFoldTask(target=target, binary_mode=binary_mode)
        search_space = search_space_factory('rna_inverse_fold', dtype, target=target, binary_mode=binary_mode)

    elif task_name == "bayesmark":
        model_name = kwargs.get("model_name", "lasso")
        metric = kwargs.get("metric", "mse")
        database_id = kwargs.get("database_id", "boston")
        seed = kwargs.get("seed", 0)
        task = BayesmarkTask(model_name=model_name, metric=metric, database_id=database_id, seed=seed)
        search_space = SearchSpace(params=task.params, dtype=dtype)

    elif "aig_optimization" in task_name:
        from comb_opt.tasks.eda_seq_opt.eda_seq_opt_task import EDASeqOptimization
        from comb_opt.search_space.search_space_eda import SearchSpaceEDA

        designs_group_id = kwargs.get("designs_group_id", "adder")
        operator_space_id = kwargs.get("operator_space_id", "basic")
        seq_operators_pattern_id = kwargs.get("seq_operators_pattern_id", "basic")
        evaluator = kwargs.get("evaluator", "abc")
        return_best_intermediate = kwargs.get("return_best_intermediate", "abc")
        lut_inputs = kwargs.get("lut_inputs", "6")
        ref_abc_seq = kwargs.get("ref_abc_seq", "resyn2")
        objective = kwargs.get("objective", "lut")
        n_parallel = kwargs.get("n_parallel", None)
        if task_name == "aig_optimization":
            operator_hyperparams_space_id = None
        elif task_name == "aig_optimization_hyp":
            operator_hyperparams_space_id = kwargs.get("operator_hyperparams_space_id", "boils_hyp_op_space")
        else:
            raise ValueError(task_name)
        task = EDASeqOptimization(designs_group_id=designs_group_id, operator_space_id=operator_space_id,
                                  seq_operators_pattern_id=seq_operators_pattern_id,
                                  operator_hyperparams_space_id=operator_hyperparams_space_id,
                                  evaluator=evaluator, lut_inputs=lut_inputs, ref_abc_seq=ref_abc_seq,
                                  objective=objective, n_parallel=n_parallel,
                                  return_best_intermediate=return_best_intermediate
                                  )
        search_space = SearchSpaceEDA(task.optim_space.search_space_params, dtype=dtype,
                                      seq_operators_pattern_id=seq_operators_pattern_id,
                                      op_ind_per_type_dic=task.optim_space.op_ind_per_type_dic)

    elif task_name == 'mig_optimization':
        seq_len = kwargs.get('seq_len', 10)
        ntk_name = kwargs.get('ntk_name', 'div')
        objective = kwargs.get('objective', 'size')

        task = MigSeqOpt(ntk_name=ntk_name, objective=objective, seq_len=seq_len)
        search_space = search_space_factory('mig_optimization', dtype, seq_len=seq_len)

    else:
        raise NotImplementedError(f'Task {task_name} is not implemented.')

    return task, search_space
