import argparse
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent)
sys.path[0] = ROOT_PROJECT

import torch

from comb_opt.utils.general_utils import save_w_pickle, load_w_pickle
from comb_opt.tasks.eda_seq_opt.utils.utils import get_eda_available_obf_funcs, EDAExpPathManager
from comb_opt.factory import task_factory

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True,
                                     description='Performs logic synthesis optimization using HyperBOiLS')

    parser.add_argument("--n_parallel", type=int, default=-1, help="number of threads to compute the stats")
    parser.add_argument("--objective", type=str, choices=tuple(get_eda_available_obf_funcs()),
                        help="which objective should be optimized")
    parser.add_argument("--designs_group_id", type=str, required=True, help="Circuit name")
    parser.add_argument("--lut_inputs", type=int, required=True, help="number of LUT inputs (2 < num < 33)")
    parser.add_argument("--ref_abc_seq", type=str, help="ID of reference sequence used to "
                                                        "measure improvements on area and delay")
    parser.add_argument("--evaluator", type=str, required=True, choices=['yosys', 'abc', 'abcpy'],
                        help="whether to use yosys-abc (`yosys`) or compiled abc repo (`abc`)")
    parser.add_argument("--operator_space_id", type=str, required=True,
                        help="id of the pattern of the sequence of operators to optimise")
    parser.add_argument("--seq_operators_pattern_id", type=str, required=True,
                        help="id of operator space ")
    parser.add_argument("--operator_hyperparams_space_id", type=str,
                        help="id of hyperparams space defining available hyperparams"
                             " for each abc operations")
    parser.add_argument("--return_best_intermediate", type=int, required=True, choices=[0, 1],
                        help="whether to optimise a sequence based on final stats or based on best intermediate stats")

    parser.add_argument("--seed", type=int, nargs='+', help="seed for reproducibility")

    parser.add_argument("--n_init", type=int, help="Number of initial random acquisitions")
    parser.add_argument("--acq_batch_size", type=int, help="Number of points acquired at each acquisition step")
    parser.add_argument("--max_n_evals", type=int, help="Max number of sequence evaluations")
    parser.add_argument("--cond_kernel_type", type=str, choices=["cond-transf-cat-kernel"],
                        required=True,
                        help="Type of conditional kernel to use.")
    parser.add_argument("--device", type=int, help="cuda id (cpu if None or negative)")

    parser.add_argument("-f", "--force", action="store_true", help="cuda id (cpu if None or negative)")

    args_ = parser.parse_args()

    if args_.n_parallel <= 0:
        args_.n_parallel = len(os.sched_getaffinity(0))

    if args_.device >= 0:
        device = torch.device(f'cuda:{args_.device}')
    else:
        device = torch.device(f'cpu')

    dtype_ = torch.float64

    n_init = args_.n_init
    operator_space_id = args_.operator_space_id
    operator_hyperparams_space_id = args_.operator_hyperparams_space_id
    seq_operators_pattern_id = args_.seq_operators_pattern_id
    designs_group_id = args_.designs_group_id
    objective = args_.objective
    lut_inputs = args_.lut_inputs
    ref_abc_seq = args_.ref_abc_seq
    evaluator = args_.evaluator
    return_best_intermediate = args_.return_best_intermediate

    acq_optim_n_iter = 150

    N_TRIALS_ = 10
    for seed_ in args_.seed:

        for trial_ in range(N_TRIALS_):
            try:
                np.random.seed(seed_ + trial_)
                torch.random.manual_seed(seed_ + trial_)

                eda_task, search_space = task_factory(
                    task_name='aig_optimization_hyp',
                    dtype=dtype_,
                    designs_group_id=args_.designs_group_id,
                    operator_space_id=operator_space_id,
                    seq_operators_pattern_id=seq_operators_pattern_id,
                    operator_hyperparams_space_id=operator_hyperparams_space_id,
                    n_parallel=args_.n_parallel,
                    objective=objective,
                    evaluator=evaluator,
                    lut_inputs=lut_inputs,
                    ref_abc_seq=ref_abc_seq,
                    return_best_intermediate=return_best_intermediate
                )

                optimiser: HyperBOiLS = HyperBOiLS(
                    search_space=search_space,
                    operator_space_id=operator_space_id,
                    operator_hyperparams_space_id=operator_hyperparams_space_id,
                    seq_operators_pattern_id=seq_operators_pattern_id,
                    n_init=n_init,
                    sequence_kernel_name='transformed_overlap',
                    nominal_param_kernel_name='transformed_overlap',
                    numeric_param_kernel_name='mat52',
                    device=device,
                    cond_kernel_type=args_.cond_kernel_type,
                    tr_succ_tol=2,
                    tr_fail_tol=20,
                    model_max_training_dataset_size=500,
                    acq_optim_n_iter=acq_optim_n_iter,
                    tr_min_num_radius=0.5 ** 5,
                    tr_max_num_radius=1,
                    tr_min_nominal_radius=1,
                    tr_max_nominal_radius=max(1, len(search_space.nominal_dims) - eda_task.optim_space.seq_len),
                    tr_min_sequence_radius=1,
                    tr_max_sequence_radius=eda_task.optim_space.seq_len,
                    ls_acq_name='ei',
                    restart_acq_name='lcb',
                    seed=seed_
                )

                result_path_manager = EDAExpPathManager(task_root_path=eda_task.exp_path(),
                                                        opt_id=optimiser.opt_id(), seed=seed_)
                full_ckpt_path = result_path_manager.eda_seq_opt_result_full_ckpt_path()
                optimiser_ckpt_path = result_path_manager.eda_seq_opt_optimiser_path()
                is_running_path = result_path_manager.eda_seq_opt_is_running_path()
                os.makedirs(result_path_manager.eda_seq_opt_result_path_root(), exist_ok=True)

                if not args_.force:
                    if os.path.exists(is_running_path):
                        print(f"Exp is already running: {os.path.dirname(is_running_path)}")
                        assert not os.path.exists(is_running_path), f"Exp is already running: {os.path.dirname(is_running_path)}"

                args_.force = 0

                if os.path.exists(optimiser_ckpt_path):
                    eda_task.log(f"Load optimiser... ", end="")
                    device = optimiser.device
                    optimiser = load_w_pickle(optimiser_ckpt_path)
                    optimiser.fill_field_after_pkl_load(search_space=search_space, device=device)
                    if len(optimiser.x_init) > 0:
                        optimiser.x_init = search_space.sample(len(optimiser.x_init))
                    eda_task.log(f"{len(optimiser.data_buffer)} already evaluated points")

                if os.path.exists(full_ckpt_path):
                    eda_task.load_ckpt_data(load_w_pickle(full_ckpt_path))

                    if len(eda_task.ckpt_data.samples_X) > 0:
                        eda_task.log(f"Restart from {len(eda_task.ckpt_data.samples_X)} already evaluated points")
                        assert os.path.exists(optimiser_ckpt_path)
                        # already_evaluated_x = pd.DataFrame(eda_task.ckpt_data.samples_X, columns=search_space.df_col_names)
                        # sample_ = search_space.sample(1)
                        # for c_ in already_evaluated_x.columns:
                        #     already_evaluated_x[c_] = already_evaluated_x[c_].astype(sample_[c_].dtype)
                        # if not os.path.exists(optimiser_ckpt_path):
                        #   optimiser.initialise(already_evaluated_x, eda_task.ckpt_data.ys)

                assert len(optimiser.data_buffer) == eda_task.num_func_evals, (
                    len(optimiser.data_buffer), eda_task.num_func_evals)

                try:
                    save_w_pickle(time.time(), is_running_path)
                    while eda_task._n_bb_evals < args_.max_n_evals:
                        x_next = optimiser.suggest(args_.acq_batch_size)
                        y_next = eda_task(x_next)
                        optimiser.observe(x_next, y_next)

                        os.makedirs(os.path.dirname(full_ckpt_path), exist_ok=True)
                        save_w_pickle(eda_task.get_ckpt_data(), full_ckpt_path)
                        os.makedirs(os.path.dirname(optimiser_ckpt_path), exist_ok=True)
                        save_w_pickle(optimiser, optimiser_ckpt_path)
                        save_w_pickle(time.time(), is_running_path)

                except Exception as e:
                    os.remove(is_running_path)
                    logs = traceback.format_exc()
                    exc = e
                    f = open(os.path.join(os.path.dirname(optimiser_ckpt_path), 'out_logs.txt'), "a")
                    f.write("-------" * 10 + '\n')
                    f.write(datetime.now().strftime("%y-%h-%d | %Hh%M") + '\n')
                    f.write(logs)
                    f.write("-------" * 10 + '\n')
                    f.close()
                    print(logs)
                    raise
                os.remove(is_running_path)

            except Exception as e:
                if trial_ == (N_TRIALS_ - 1):
                    raise e