#!/bin/bash

TASKS=( ackley aig_optim_basic antibody_design mig_optim pest_control rna_inverse_fold )

# ------------------------------------------------------------
# BEGIN .35
task_name=ackley

PYTHON_SCRIPT=./experiments/comb_task_exps.py

models=(gp_to gp_o gp_diffusion gp_ssk lr_sparse_hs)
acq_opts=(ga ls is sa)

#for use_tr in 0 1; do
for use_tr in 1; do
  if (( use_tr == 0 )); then use_tr=''; else use_tr='--use_tr'; fi
  taskset_end=29
  for i in $(seq 0 $((${#models[@]} - 1))); do
    model=${models[i]}
    device_id=$(( i % 4))
    device_id=-1

    taskset_start=$(( taskset_end + 1 ))
    taskset_end=$(( taskset_start + 5 ))

    optimizers_ids=""
    for j in $(seq 0 $((${#acq_opts[@]} - 1))); do
      acq_opt=${acq_opts[j]}
      optimizers_ids="${optimizers_ids} ${model}_${acq_opt}"
    done

    cmd="taskset -c ${taskset_start}-${taskset_end} python ${PYTHON_SCRIPT} --task_name $task_name --optimizers_ids ${optimizers_ids} $use_tr --device_id ${device_id}"
    echo $cmd
    $cmd &
  done
  wait
done

optimizers_ids="gp_o_ls"
device_id=-1

#for use_tr in 0 1; do
#  if (( use_tr == 0 )); then use_tr=''; else use_tr='--use_tr'; fi
#  taskset -c 60-64 python ./experiments/comb_task_exps.py --task_name ackley --optimizers_ids ${optimizers_ids} --device_id ${device_id}
#done

taskset -c 35-39   python ./experiments/comb_task_exps.py --task_name "levy-80" --optimizers_ids gp_ssk_ga gp_ssk_is --use_tr --device_id -1 --seeds 42 43 44 45 46
taskset -c 30-34   python ./experiments/comb_task_exps.py --task_name "levy-80" --optimizers_ids gp_diffusion_ga gp_diffusion_is gp_to_sa gp_to_is --use_tr --device_id 0 --seeds 42 43 44 45 46
taskset -c 25-29   python ./experiments/comb_task_exps.py --task_name "levy-40" --optimizers_ids gp_ssk_ga gp_ssk_is --use_tr --device_id -1 --seeds 42 43 44 45 46
taskset -c 20-24   python ./experiments/comb_task_exps.py --task_name "levy-40" --optimizers_ids gp_diffusion_ga gp_diffusion_is gp_to_sa gp_to_is --use_tr --device_id 1 --seeds 42 43 44 45 46
taskset -c 15-19   python ./experiments/comb_task_exps.py --task_name "levy-20" --optimizers_ids gp_ssk_ga gp_ssk_is --use_tr --device_id 2 --seeds 42 43 44 45 46
taskset -c 10-14   python ./experiments/comb_task_exps.py --task_name "levy-20" --optimizers_ids gp_diffusion_ga gp_diffusion_is gp_to_sa gp_to_is --use_tr --device_id 2 --seeds 42 43 44 45 46
taskset -c 5-9     python ./experiments/comb_task_exps.py --task_name "levy-10" --optimizers_ids gp_ssk_ga gp_ssk_is --use_tr --device_id 3 --seeds 42 43 44 45 46
taskset -c 0-4     python ./experiments/comb_task_exps.py --task_name "levy-10" --optimizers_ids gp_diffusion_ga gp_diffusion_is gp_to_sa gp_to_is --use_tr --device_id 3 --seeds 42 43 44 45 46

#scp -r ~/Projects/combopt_release/results/MIG\ Sequence\ Optimisation\ -\ sqrt\ -\ both antoine@10.227.91.34:/home/antoine/Projects/combopt_lib_release/results/
# END .35
# ------------------------------------------------------------


# ------------------------------------------------------------
# BEGIN .34



taskset -c 0-4     python ./experiments/comb_task_exps.py --task_name "levy-80" --optimizers_ids gp_ssk_ga gp_ssk_is --device_id -1 --seeds 42 43 44 45 46
taskset -c 5-9     python ./experiments/comb_task_exps.py --task_name "levy-80" --optimizers_ids gp_diffusion_ga gp_diffusion_is gp_to_sa gp_to_is --device_id 0 --seeds 42 43 44 45 46
taskset -c 10-14   python ./experiments/comb_task_exps.py --task_name "levy-40" --optimizers_ids gp_ssk_ga gp_ssk_is --device_id -1 --seeds 42 43 44 45 46
taskset -c 15-19   python ./experiments/comb_task_exps.py --task_name "levy-40" --optimizers_ids gp_diffusion_ga gp_diffusion_is gp_to_sa gp_to_is --device_id 1 --seeds 42 43 44 45 46
taskset -c 20-24   python ./experiments/comb_task_exps.py --task_name "levy-20" --optimizers_ids gp_ssk_ga gp_ssk_is --device_id 1 --seeds 42 43 44 45 46
taskset -c 25-29   python ./experiments/comb_task_exps.py --task_name "levy-20" --optimizers_ids gp_diffusion_ga gp_diffusion_is gp_to_sa gp_to_is --device_id 2 --seeds 42 43 44 45 46
taskset -c 30-34   python ./experiments/comb_task_exps.py --task_name "levy-10" --optimizers_ids gp_ssk_ga gp_ssk_is --device_id 0 --seeds 42 43 44 45 46
taskset -c 35-39   python ./experiments/comb_task_exps.py --task_name "levy-10" --optimizers_ids gp_diffusion_ga gp_diffusion_is gp_to_sa gp_to_is --device_id 3 --seeds 42 43 44 45 46

# END .34
# ------------------------------------------------------------


# ------------------------------------------------------------
# BEGIN .23
task_name=mig_optimization

PYTHON_SCRIPT=./experiments/comb_task_exps.py

models=(gp_to gp_o gp_diffusion gp_ssk lr_sparse_hs)
acq_opts=(ga ls is sa)

taskset_end=-1
for i in $(seq 0 $((${#models[@]} - 1))); do
  for use_tr in 0 1; do
  if (( use_tr == 0 )); then use_tr=''; else use_tr='--use_tr'; fi
    model=${models[i]}
    device_id=$(( i % 4))
    device_id=-1

    taskset_start=$(( taskset_end + 1 ))
    taskset_end=$(( taskset_start + 5 ))

    optimizers_ids=""
    for j in $(seq 0 $((${#acq_opts[@]} - 1))); do
      acq_opt=${acq_opts[j]}
      optimizers_ids="${optimizers_ids} ${model}_${acq_opt}"
    done

    cmd="taskset -c ${taskset_start}-${taskset_end} python ${PYTHON_SCRIPT} --task_name $task_name --optimizers_ids ${optimizers_ids} $use_tr --device_id ${device_id}"
    echo $cmd
    $cmd &
  done
done
wait

#taskset -c 70-74 python ./experiments/comb_task_exps.py --task_name mig_optimization --optimizers_ids lr_sparse_hs_ga lr_sparse_hs_ls lr_sparse_hs_is lr_sparse_hs_sa --device_id -1

#scp -r ~/antoineg/Projects/combopt_release/results/MIG\ Sequence\ Optimisation\ -\ sqrt\ -\ both antoine@10.227.91.34:/home/antoine/Projects/combopt_lib_release/results/
# END .23
# ------------------------------------------------------------
