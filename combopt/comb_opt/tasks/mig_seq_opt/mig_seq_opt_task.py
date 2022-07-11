# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from comb_opt.tasks.task_base import TaskBase


class MigSeqOpt(TaskBase):

    def __init__(self, ntk_name: str = 'div', objective: str = 'size', seq_len: int = 3):
        super(MigSeqOpt, self).__init__()

        self.arithmetic_dataset = ['adder', 'bar', 'div', 'hyp', 'log2', 'max', 'multiplier', 'sin', 'sqrt', 'square']
        self.random_control_dataset = ['arbiter', 'cavlc', 'ctrl', 'dec', 'i2c', 'int2float', 'mem_ctrl', 'priority',
                                       'router', 'voter']

        assert ntk_name in self.arithmetic_dataset or ntk_name in self.random_control_dataset
        assert objective in ['size', 'depth'], \
            'Choose objective=\'size\' for LUT count optimization and objective=\'depth\' for LUT delay optimization.'
        self.seq_len = seq_len
        self.objective = objective

        if ntk_name in self.arithmetic_dataset:
            self.path_to_network = os.path.join(Path(__file__).parent.parent.resolve(), 'data', 'epfl_benchmark',
                                                'arithmetic', ntk_name + '.aig')

        elif ntk_name in self.random_control_dataset:
            self.path_to_network = os.path.join(Path(__file__).parent.parent.resolve(), 'data', 'epfl_benchmark',
                                                'random_control', ntk_name + '.aig')

        self.path_to_executable = os.path.join(Path(__file__).parent.resolve(), 'mig_task_executable')
        self.temp_save_dir = os.path.join(Path(__file__).parent.resolve())

        self.idx_to_op = {0: "balance",
                          1: "cut rewrite",
                          2: "cut rewrite -z",
                          3: "refactor",
                          4: "refactor -z",
                          5: "resubstitute",
                          6: 'functional_reduction'}

        self.op_to_idx = {}
        for key in self.idx_to_op:
            self.op_to_idx[self.idx_to_op[key]] = key

    @property
    def name(self) -> str:
        return 'MIG Sequence Optimisation'

    def evaluate(self, x: pd.DataFrame) -> np.ndarray:
        n = len(x)
        y = np.zeros((n, 1), dtype=np.float32)
        for i in range(n):
            y[i] = self._black_box(x.iloc[i])
        return y

    def _black_box(self, x):

        # Change dataframe to numpy array of indices
        x = np.array([self.op_to_idx[op] for op in x.array])

        path_to_sequence = os.path.join(self.temp_save_dir, f"sequence_{os.getpid()}.txt")

        # Save sequence to a text file
        with open(path_to_sequence, "w") as file:
            file.write(' '.join(str(idx) for idx in x))

        try:
            result = subprocess.run(
                [self.path_to_executable, self.path_to_network, path_to_sequence, str(self.seq_len)], shell=False,
                capture_output=True, text=True)
        except PermissionError:
            print(f"\n\nRun \'chmod u+x {self.path_to_executable}\' in terminal\n\n")

        result = result.stdout.split(", ")[1:-1]
        init_size, init_depth, final_size, final_depth = int(result[0]), int(result[1]), int(result[2]), int(
            result[3])

        if self.objective == 'size':
            fx = final_size
        elif self.objective == 'depth':
            fx = final_depth

        os.remove(path_to_sequence)
        return fx
