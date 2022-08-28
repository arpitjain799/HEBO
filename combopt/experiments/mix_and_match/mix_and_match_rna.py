import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(os.path.realpath(__file__)).parent.parent.parent))

from comb_opt.factory import task_factory
from comb_opt.utils.experiment_utils import run_experiment

from comb_opt.optimizers.mix_and_match.gp_to_kernel_ga_acq_optim import GpToGaAcqOptim
from comb_opt.optimizers.mix_and_match.gp_to_kernel_sa_acq_optim import GpToSaAcqOptim
from comb_opt.optimizers.mix_and_match.gp_to_kernel_exhaustive_ls_acq_optim import GpToExhaustiveLsAcqOptim
from comb_opt.optimizers.mix_and_match.gp_diffusion_kernel_ga_acq_optim import GpDiffusionGaAcqOptim
from comb_opt.optimizers.mix_and_match.gp_diffusion_kernel_sa_acq_optim import GpDiffusionSaAcqOptim
from comb_opt.optimizers.mix_and_match.gp_diffusion_kernel_tr_stochastic_ls_acq_optim import GpDiffusionTrLsAcqOptim
from comb_opt.optimizers.mix_and_match.gp_ssk_kernel_exhaustive_ls_acq_optim import GpSskExhaustiveLsAcqOptim
from comb_opt.optimizers.mix_and_match.gp_ssk_kernel_sa_acq_optim import GpSskSaAcqOptim
from comb_opt.optimizers.mix_and_match.lr_sparse_hs_ga_acq_optim import LrSparseHsGaAcqOptim
from comb_opt.optimizers.mix_and_match.lr_sparse_hs_tr_stochastic_ls_acq_optim import LrSparseHsTrLsAcqOptim
from comb_opt.optimizers.mix_and_match.lr_sparse_hs_exhaustive_ls_acq_optim import LrSparseHsExhaustiveLsAcqOptim

if __name__ == '__main__':
    task_name = 'rna_inverse_fold'
    task_kwargs = {'target': 65}
    bo_n_init = 20
    bo_device = torch.device('cuda:0')
    max_num_iter = 200
    dtype = torch.float32
    random_seeds = [42, 43, 44, 45, 46]

    task, search_space = task_factory(task_name, dtype, **task_kwargs)

    gp_to_ga = GpToGaAcqOptim(search_space, bo_n_init, dtype=dtype, device=bo_device)
    gp_to_sa = GpToSaAcqOptim(search_space, bo_n_init, dtype=dtype, device=bo_device)
    gp_to_exhaustive_ls = GpToExhaustiveLsAcqOptim(search_space, bo_n_init, dtype=dtype, device=bo_device)

    gp_diffusion_ga = GpDiffusionGaAcqOptim(search_space, bo_n_init, dtype=dtype, device=bo_device)
    gp_diffusion_sa = GpDiffusionSaAcqOptim(search_space, bo_n_init, dtype=dtype, device=bo_device)
    gp_diffusion_tr_ls = GpDiffusionTrLsAcqOptim(search_space, bo_n_init, dtype=dtype, device=bo_device)

    gp_ssk_sa = GpSskSaAcqOptim(search_space, bo_n_init, model_max_batch_size=50, dtype=dtype, device=bo_device)
    gp_ssk_exhaustive_ls = GpSskExhaustiveLsAcqOptim(search_space, bo_n_init, model_max_batch_size=50, dtype=dtype,
                                                     device=bo_device)

    lr_sparse_hs_ga = LrSparseHsGaAcqOptim(search_space, bo_n_init, dtype=dtype, device=bo_device)
    lr_sparse_hs_tr_ls = LrSparseHsTrLsAcqOptim(search_space, bo_n_init, dtype=dtype, device=bo_device)
    lr_sparse_hs_exhaustive_ls = LrSparseHsExhaustiveLsAcqOptim(search_space, bo_n_init, dtype=dtype, device=bo_device)

    optimizers = [gp_ssk_sa, gp_ssk_exhaustive_ls, gp_diffusion_ga, gp_diffusion_sa, gp_diffusion_tr_ls, gp_to_ga,
                  gp_to_sa, gp_to_exhaustive_ls, lr_sparse_hs_ga, lr_sparse_hs_tr_ls, lr_sparse_hs_exhaustive_ls]

    run_experiment(task=task, optimizers=optimizers, random_seeds=random_seeds, max_num_iter=max_num_iter,
                   very_verbose=False)
