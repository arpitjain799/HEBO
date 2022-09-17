# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import glob
import os
from typing import List

import pandas as pd
import torch

from comb_opt import RESULTS_DIR
from comb_opt.optimizers import OptimizerBase
from comb_opt.tasks import TaskBase
from comb_opt.utils.general_utils import create_save_dir, current_time_formatter, set_random_seed
from comb_opt.utils.results_logger import ResultsLogger
from comb_opt.utils.stopwatch import Stopwatch


def replace_spaces(string: str) -> str:
    return string.replace(' ', '_')


def run_experiment(task: TaskBase,
                   optimizers: List[OptimizerBase],
                   random_seeds: List[int],
                   max_num_iter: int,
                   save_results_every: int = 100,
                   very_verbose=False,
                   ):
    # Basic assertion checks
    assert isinstance(task, TaskBase)
    assert isinstance(optimizers, list)
    assert isinstance(random_seeds, list)
    assert isinstance(max_num_iter, int) and max_num_iter > 0
    assert isinstance(save_results_every, int) and save_results_every > 0
    for seed in random_seeds:
        assert isinstance(seed, int)
    for optimizer in optimizers:
        assert isinstance(optimizer, OptimizerBase)

    # Create the save directory
    # exp_save_dir = os.path.join(RESULTS_DIR, replace_spaces(task.name))
    exp_save_dir = os.path.join(RESULTS_DIR, task.name)
    create_save_dir(exp_save_dir)

    stopwatch = Stopwatch()
    results_logger = ResultsLogger()

    print(f'{current_time_formatter()} - Starting experiment for {task.name} Task')

    # Obtain maximum optimizer name length for formatting
    max_name_len = 0
    for optimizer in optimizers:
        max_name_len = max(max_name_len, len(optimizer.name))

    for optimizer in optimizers:

        # optim_save_dir = os.path.join(exp_save_dir, replace_spaces(optimizer.name))
        optim_save_dir = os.path.join(exp_save_dir, optimizer.name)
        create_save_dir(optim_save_dir)

        for i, seed in enumerate(random_seeds):
            print(
                f'{current_time_formatter()} - Optimizer : {optimizer.name:>{max_name_len}} - Seed {seed} {i + 1:2d}/{len(random_seeds):2d}')

            set_random_seed(seed)
            task.restart()
            optimizer.restart()
            stopwatch.reset()
            results_logger.restart()

            # Main loop
            for iter_num in range(1, max_num_iter + 1):

                torch.cuda.empty_cache()  # Clear cached memory

                # Suggest a point
                stopwatch.start()
                x_next = optimizer.suggest(1)
                stopwatch.stop()

                # Compute the Black-box value
                y_next = task(x_next)

                # Observe the point
                stopwatch.start()
                optimizer.observe(x_next, y_next)
                stopwatch.stop()

                results_logger.append(eval_num=task.num_func_evals,
                                      y=y_next[0, 0],
                                      y_star=optimizer.best_y,
                                      elapsed_time=stopwatch.get_total_time())

                if very_verbose:
                    print(
                        f'{current_time_formatter()} - Iteration {iter_num:3d}/{max_num_iter:3d} - y {y_next[0, 0]:.3f} - y* {optimizer.best_y:.3f}')

                if iter_num % save_results_every == 0:
                    results_logger.save(os.path.join(optim_save_dir, f'seed_{seed}_results.csv'))

            results_logger.save(os.path.join(optim_save_dir, f'seed_{seed}_results.csv'))

    print(f'{current_time_formatter()} - Experiment finished.')


def filter_results(results: pd.DataFrame, method_names: List[str]):
    results = results[results['Optimizer'].isin(method_names)]
    results.Optimizer = results.Optimizer.astype('category').cat.set_categories(method_names)
    return results.sort_values(['Task', 'Optimizer'])


LR_SPARSE_HS_VARIANTS = ['BOCS',
                         'LR (Sparse HS) - GA acq optim',
                         'LR (Sparse HS) - LS acq optim',
                         'LR (Sparse HS) - TR-based LS acq optim'
                         ]

GP_DIFFUSION_VARIANTS = ['COMBO',
                         'GP (Diffusion) - GA acq optim',
                         'GP (Diffusion) - SA acq optim',
                         'GP (Diffusion) - TR-based LS acq optim'
                         ]

GP_TO_VARIANTS = ['Casmopolitan',
                  'GP (TO) - GA acq optim',
                  'GP (TO) - SA acq optim',
                  'GP (TO) - LS acq optim',
                  ]

GP_SSK_Variants = ['BOSS',
                   'BOiLS',
                   'GP (SSK) - SA acq optim',
                   'GP (SSK) - LS acq optim',
                   ]


def load_single_task_results(task_name: str) -> pd.DataFrame:
    columns = ['Task', 'Optimizer', 'Model', 'Seed', 'Eval Num', 'f(x)', 'f(x*)', 'Elapsed Time']
    results = pd.DataFrame(columns=columns)

    # task_name = replace_spaces(task_name)
    save_dir = os.path.join(RESULTS_DIR, task_name)
    optimizers = [dir_path.split('/')[-1] for dir_path in glob.glob(os.path.join(save_dir, '*'))]

    for optimizer in optimizers:
        sub_folder_dir = os.path.join(save_dir, optimizer)
        seeds = [file_path.split('/')[-1].split('_')[1] for file_path in glob.glob(os.path.join(sub_folder_dir, '*'))]

        for seed in seeds:
            df = pd.read_csv(os.path.join(sub_folder_dir, f'seed_{seed}_results.csv'))
            df['Optimizer'] = len(df['Eval Num']) * [optimizer]
            df['Task'] = len(df['Eval Num']) * [task_name]
            df['Seed'] = len(df['Eval Num']) * [seed]
            if optimizer in LR_SPARSE_HS_VARIANTS:
                df['Model'] = len(df['Eval Num']) * ['LR (Sparse HS)']
            elif optimizer in GP_DIFFUSION_VARIANTS:
                df['Model'] = len(df['Eval Num']) * ['GP (Diffusion)']
            elif optimizer in GP_TO_VARIANTS:
                df['Model'] = len(df['Eval Num']) * ['GP (TO)']
            elif optimizer in GP_SSK_Variants:
                df['Model'] = len(df['Eval Num']) * ['GP (SSK)']
            else:
                df['Model'] = len(df['Eval Num']) * ['Unknown']
            df = df[columns]
            results = pd.concat([results, df], ignore_index=True, sort=False)

    return results


def load_results(task_names: List[str]) -> pd.DataFrame:
    columns = ['Task', 'Optimizer', 'Model', 'Seed', 'Eval Num', 'f(x)', 'f(x*)', 'Elapsed Time']
    results = pd.DataFrame(columns=columns)

    for task_name in task_names:
        results = pd.concat([results, load_single_task_results(task_name)], ignore_index=True, sort=False)

    results.Task = results.Task.astype('category').cat.set_categories(task_names)

    return results.sort_values(['Task', 'Optimizer', 'Seed', 'Eval Num'])
