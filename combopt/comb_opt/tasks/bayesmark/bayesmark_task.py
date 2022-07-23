# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
import os
from typing import Any, List, Dict

import numpy as np
import pandas as pd
from bayesmark.constants import MODEL_NAMES, DATA_LOADER_NAMES, METRICS
from joblib import delayed, Parallel

from comb_opt.tasks import TaskBase
from comb_opt.tasks.bayesmark.utils.utils_wrapper import SklearnModelWrapper


class BayesmarkTask(TaskBase):

    @property
    def name(self) -> str:
        return f'Bayesmark task | Mod-{self.model_name} | DB {self.database_id} | Metr-{self.metric}'

    def __init__(self, model_name: str, database_id: str, metric: str, seed: int = 0, **kwargs):
        """
        Args:
            model_name: name of the sklearn ML model
            database_id: database ID of this benchmark experiment
            metric: {acc,mae,mse,nll} scoring metric to use
            seed: for reproducibility
        """
        super().__init__(**kwargs)
        assert model_name in MODEL_NAMES
        assert database_id in DATA_LOADER_NAMES
        assert metric in METRICS
        self.model_name = model_name
        self.database_id = database_id
        self.metric = metric
        self.seed = seed
        self.test_func = SklearnModelWrapper(model=self.model_name, dataset=self.database_id, metric=self.metric,
                                             shuffle_seed=self.seed)
        self.params = self.build_params()

    def evaluate(self, x: pd.DataFrame) -> np.ndarray:
        """ Evaluate the provided set of hyperparameters """
        x_as_dict = x.to_dict()
        n_parallel = len(os.sched_getaffinity(0))
        evaluations = Parallel(n_jobs=n_parallel, backend="multiprocessing")(
            delayed(self.test_func.evaluate)({k: d[ind] for k, d in x_as_dict.items()}) for ind in x.index
        )
        evaluations = [evaluation[0] for evaluation in evaluations]
        return np.array(evaluations).reshape(-1, 1)

    def build_params(self) -> List[Dict[str, Any]]:
        """ Get the set of hyperparameters to tune """
        params = []
        for k, param_conf in self.test_func.api_config.items():
            element = {"name": k}
            param_type = param_conf['type']
            param_space = param_conf.get('space', None)
            param_range = param_conf.get("range", None)
            # param_values = param_conf.get("values", None)  # for categorical

            if param_type == "real":
                if param_space == "linear":
                    element["type"] = "num"
                elif param_space == "log":
                    element["type"] = "pow"
                elif param_space == "logit":
                    element["type"] = "sigmoid"
                else:
                    raise ValueError(param_space)
                element["lb"] = param_range[0]
                element["ub"] = param_range[1]
            elif param_type == "int":
                if param_space == "linear":
                    element["type"] = "int"
                elif param_space == "log":
                    element["type"] = "pow_int"
                else:
                    raise ValueError(param_space)
                element["lb"] = param_range[0]
                element["ub"] = param_range[1]
            elif param_type == "bool":
                element["type"] = "bool"
            else:
                raise ValueError(param_type)

            params.append(element)
        return params
